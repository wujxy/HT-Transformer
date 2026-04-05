"""
Hybrid Token Transformer for JUNO Ordered Dual-Endpoint Reconstruction (V2).

Architecture:
  WP hits (hit-level) + CD patches (HEALPix) + Global tokens + Query tokens
  → Hybrid Encoder with structured attention
  → 2 Query outputs → Endpoint Heads → ordered unit vectors

V2 Features:
  - WPProjector: dual-branch [ux,uy,uz]⊕[q,t] feature extraction
  - DeepSphereEncoder: local spherical aggregation for CD
  - CDCompression: HEALPix hierarchical pooling
  - SignedTimeBucketBias: WP time-aware self-attention

Attention rules:
  WP↔WP: dense with signed time bias
  WP↔CD: dense cross-attention (bidirectional)
  CD↔CD: DeepSphere local aggregation (NOT full attention)
  Global↔All: dense
  Query↔All: dense cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from models.components.token_projectors import (
    WPProjector, CDProjector, TokenTypeEmbedding, FourierPositionEncoding,
    TOKEN_WP, TOKEN_CD, TOKEN_GLOBAL, TOKEN_QUERY,
)
from models.components.deepsphere import DeepSphereEncoder, CDCompression, build_healpix_knn_adjacency, RMSNorm
from models.components.wp_time_bias import SignedTimeBucketBias


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional mask and time bias."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                time_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Sq, _ = query.shape
        _, Sk, _ = key.shape

        q = self.wq(query).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(key).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(value).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)

        if time_bias is None:
            # SDPA path: FlashAttention / memory-efficient backend
            attn_mask = None
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                attn_mask = torch.zeros_like(mask, dtype=q.dtype)
                attn_mask.masked_fill_(mask, float('-inf'))

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            # Manual path: with additive time bias
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = scores + time_bias

            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                scores = scores.masked_fill(mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            attn = attn.nan_to_num(0.0)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, Sq, self.d_model)
        return self.wo(out)


class FeedForward(nn.Module):
    """Position-wise FFN with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridFusionLayer(nn.Module):
    """
    Hybrid Fusion Layer for V2.

    Features:
    - WP self-attention with optional time bias
    - WP↔CD cross-attention (bidirectional)
    - Global↔All attention
    - Query↔All attention

    Note: CD self-attention is handled by DeepSphereEncoder before this layer.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_global: int, num_queries: int,
                 use_wp_time_bias: bool = False,
                 dropout: float = 0.1, norm_type: str = 'layernorm'):
        super().__init__()
        self.d_model = d_model
        self.use_wp_time_bias = use_wp_time_bias

        # Attention modules
        self.wp_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.wp_cd_cross = MultiHeadAttention(d_model, num_heads, dropout)
        self.cd_wp_cross = MultiHeadAttention(d_model, num_heads, dropout)
        self.global_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.query_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # FFN
        self.wp_ffn = FeedForward(d_model, d_ff, dropout)
        self.cd_ffn = FeedForward(d_model, d_ff, dropout)
        self.global_ffn = FeedForward(d_model, d_ff, dropout)
        self.query_ffn = FeedForward(d_model, d_ff, dropout)

        # Normalization (Pre-LN style)
        NormClass = RMSNorm if norm_type == 'rmsnorm' else nn.LayerNorm
        self.wp_attn_norm = NormClass(d_model)
        self.wp_cross_norm = NormClass(d_model)
        self.cd_cross_norm = NormClass(d_model)
        self.global_norm = NormClass(d_model)
        self.query_norm = NormClass(d_model)
        self.wp_ffn_norm = NormClass(d_model)
        self.cd_ffn_norm = NormClass(d_model)
        self.global_ffn_norm = NormClass(d_model)
        self.query_ffn_norm = NormClass(d_model)

    def forward(self, wp_emb: torch.Tensor, cd_emb: torch.Tensor,
                global_emb: torch.Tensor, query_emb: torch.Tensor,
                wp_mask: torch.Tensor, cd_mask: torch.Tensor,
                wp_time_bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        B = wp_emb.shape[0]
        N_wp = wp_emb.shape[1]
        N_cd = cd_emb.shape[1]
        M = global_emb.shape[1]
        Q = query_emb.shape[1]

        # 1. WP self-attention (with optional time bias)
        wp_normed = self.wp_attn_norm(wp_emb)
        wp_attn_mask = wp_mask.unsqueeze(1) | wp_mask.unsqueeze(2)
        wp_out = wp_emb + self.wp_self_attn(wp_normed, wp_normed, wp_normed,
                                            mask=wp_attn_mask, time_bias=wp_time_bias)

        # 2. WP↔CD cross-attention (bidirectional)
        wp_cross_in = self.wp_cross_norm(wp_out)
        cd_cross_in = self.cd_cross_norm(cd_emb)

        wp_cross_mask = cd_mask.unsqueeze(1).expand(-1, N_wp, -1)
        wp_out = wp_out + self.wp_cd_cross(wp_cross_in, cd_cross_in, cd_cross_in,
                                            mask=wp_cross_mask)

        cd_cross_mask = wp_mask.unsqueeze(1).expand(-1, N_cd, -1)
        cd_out = cd_emb + self.cd_wp_cross(cd_cross_in, wp_cross_in, wp_cross_in,
                                            mask=cd_cross_mask)

        # 3. Global↔All
        all_tokens = torch.cat([wp_out, cd_out], dim=1)
        all_mask = torch.cat([wp_mask, cd_mask], dim=1)

        global_normed = self.global_norm(global_emb)
        global_attn_mask = all_mask.unsqueeze(1).expand(-1, M, -1)
        global_out = global_emb + self.global_attn(global_normed, all_tokens, all_tokens,
                                                    mask=global_attn_mask)

        # 4. Query↔All
        all_with_global = torch.cat([wp_out, cd_out, global_out], dim=1)
        all_global_mask = torch.cat([wp_mask, cd_mask,
                                      torch.zeros(B, M, dtype=torch.bool, device=wp_mask.device)], dim=1)

        query_normed = self.query_norm(query_emb)
        query_attn_mask = all_global_mask.unsqueeze(1).expand(-1, Q, -1)
        query_out = query_emb + self.query_attn(query_normed, all_with_global, all_with_global,
                                                 mask=query_attn_mask)

        # 5. FFN
        wp_out = wp_out + self.wp_ffn(self.wp_ffn_norm(wp_out))
        cd_out = cd_out + self.cd_ffn(self.cd_ffn_norm(cd_out))
        global_out = global_out + self.global_ffn(self.global_ffn_norm(global_out))
        query_out = query_out + self.query_ffn(self.query_ffn_norm(query_out))

        return wp_out, cd_out, global_out, query_out


def _masked_mean(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over dim=1 ignoring padded positions."""
    valid = (~mask).float().unsqueeze(-1)
    return (emb * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)


class EndpointHead(nn.Module):
    """Predicts a single unit vector endpoint from a query token."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        v = self.mlp(query)
        return F.normalize(v, dim=-1)


class HTTransformer(nn.Module):
    """Hybrid Token Transformer for ordered dual-endpoint reconstruction (V2)."""

    def __init__(self, cfg: dict):
        super().__init__()
        data_cfg = cfg['data']
        model_cfg = cfg['model']

        self.d_model = model_cfg['d_model']
        self.num_layers = model_cfg['num_layers']
        self.num_global = model_cfg['num_global_tokens']
        self.num_queries = model_cfg['num_queries']
        self.num_time_bins = data_cfg['num_time_bins']
        self.norm_type = model_cfg.get('norm_type', 'layernorm')

        # WP time bias module
        wp_time_bias_type = model_cfg.get('wp_time_bias', 'signed_bucket')
        if wp_time_bias_type == 'signed_bucket':
            self.wp_time_bias = SignedTimeBucketBias(
                num_buckets=model_cfg.get('wp_num_time_buckets', 64),
                num_heads=model_cfg['num_heads'],
                heads_shared=model_cfg.get('wp_time_bias_heads_shared', False),
                max_time=1.0,
            )
        else:
            self.wp_time_bias = None

        # WP time bias: only apply to first k layers (default: all layers)
        # This allows later layers to use SDPA fast path
        self.wp_time_bias_layers = model_cfg.get('wp_time_bias_layers', self.num_layers)

        # Token projectors
        self.wp_projector = WPProjector(
            d_model=self.d_model,
            d_geo=model_cfg.get('wp_geo_hidden', 32),
            d_qt=model_cfg.get('wp_qt_hidden', 32),
            hidden=model_cfg.get('wp_projector_hidden', 64),
            dropout=model_cfg.get('dropout', 0.1)
        )
        self.cd_projector = CDProjector(
            num_time_bins=self.num_time_bins,
            d_model=self.d_model,
        )

        # Type embedding
        self.type_embedding = TokenTypeEmbedding(self.d_model)

        # Position encoding
        self.abs_pe = FourierPositionEncoding(
            d_model=self.d_model,
            num_frequencies=model_cfg.get('fourier_freq', 32),
        )

        # Learnable tokens
        self.global_tokens = nn.Parameter(
            torch.randn(1, self.num_global, self.d_model) * 0.02)
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_queries, self.d_model) * 0.02)

        # DeepSphere encoder and CD compression
        self.cd_encoder = DeepSphereEncoder(
            d_model=self.d_model,
            num_layers=model_cfg.get('cd_deepsphere_layers', 4),
            hidden_dim=model_cfg.get('cd_deepsphere_hidden', 256),
            k_neighbors=model_cfg.get('cd_knn_k', 16),
            dropout=model_cfg.get('dropout', 0.1),
            norm_type=self.norm_type,
        )
        self.cd_compression = CDCompression(
            d_model=self.d_model,
            target_tokens=model_cfg.get('cd_fusion_tokens', 128),
            method=model_cfg.get('cd_compression', 'healpix_pool'),
            nside_in=data_cfg.get('nside', 8),
            nside_out=model_cfg.get('cd_compression_nside', 4),
        )

        # Precompute HEALPix kNN adjacency
        nside = data_cfg.get('nside', 8)
        k = model_cfg.get('cd_knn_k', 16)
        try:
            self.register_buffer('cd_knn_adj',
                build_healpix_knn_adjacency(nside, k))
        except:
            self.cd_knn_adj = None

        # Fusion encoder layers
        self.encoder_layers = nn.ModuleList([
            HybridFusionLayer(
                d_model=self.d_model,
                num_heads=model_cfg['num_heads'],
                d_ff=model_cfg['d_ff'],
                num_global=self.num_global,
                num_queries=self.num_queries,
                use_wp_time_bias=(self.wp_time_bias is not None),
                dropout=model_cfg.get('dropout', 0.1),
                norm_type=self.norm_type,
            )
            for _ in range(self.num_layers)
        ])

        # Output heads
        self.head1 = EndpointHead(self.d_model, dropout=model_cfg.get('dropout', 0.1))
        self.head2 = EndpointHead(self.d_model, dropout=model_cfg.get('dropout', 0.1))

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'norm' not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = batch['wp_tokens'].shape[0]

        # Project tokens
        wp_emb = self.wp_projector(batch['wp_tokens'])
        cd_emb = self.cd_projector(
            batch['cd_unit_vecs'], batch['cd_stats'], batch['cd_time_bins']
        )

        # Add type embeddings
        wp_type = torch.full((B, wp_emb.shape[1]), TOKEN_WP, device=wp_emb.device, dtype=torch.long)
        cd_type = torch.full((B, cd_emb.shape[1]), TOKEN_CD, device=cd_emb.device, dtype=torch.long)
        global_type = torch.full((B, self.num_global), TOKEN_GLOBAL, device=wp_emb.device, dtype=torch.long)
        query_type = torch.full((B, self.num_queries), TOKEN_QUERY, device=wp_emb.device, dtype=torch.long)

        wp_emb = wp_emb + self.type_embedding(wp_type)
        cd_emb = cd_emb + self.type_embedding(cd_type)

        # Add absolute position encoding
        wp_pe = self.abs_pe(batch['wp_unit_vecs'])
        cd_pe = self.abs_pe(batch['cd_unit_vecs'])
        wp_emb = wp_emb + wp_pe
        cd_emb = cd_emb + cd_pe

        # Expand learnable tokens
        global_emb = self.global_tokens.expand(B, -1, -1)
        query_emb = self.query_tokens.expand(B, -1, -1)
        global_emb = global_emb + self.type_embedding(global_type)
        query_emb = query_emb + self.type_embedding(query_type)

        # DeepSphere encoding + Compression
        cd_pixel_ids = batch.get('cd_pixel_ids', None)
        cd_emb = self.cd_encoder(
            cd_emb,
            pixel_ids=cd_pixel_ids,
            full_knn_adj=self.cd_knn_adj,
            cd_mask=batch.get('cd_mask', None),
        )
        cd_emb, cd_pixel_ids_fused = self.cd_compression(
            cd_emb, pixel_ids=cd_pixel_ids, mask=batch['cd_mask'])
        cd_mask = (cd_emb.abs().sum(dim=-1) == 0) if cd_pixel_ids_fused is None else \
                  (cd_pixel_ids_fused == -1)

        # Compute WP time bias if enabled (once, reused across layers)
        wp_time_bias = None
        if self.wp_time_bias is not None:
            wp_times_input = batch.get('wp_times', None)
            if wp_times_input is not None:
                wp_time_bias = self.wp_time_bias(wp_times_input)

        # Encoder layers with time bias only applied to first k layers
        for layer_idx, layer in enumerate(self.encoder_layers):
            # Only apply time bias to first wp_time_bias_layers layers
            layer_wp_time_bias = wp_time_bias if layer_idx < self.wp_time_bias_layers else None

            wp_emb, cd_emb, global_emb, query_emb = layer(
                wp_emb, cd_emb, global_emb, query_emb,
                wp_mask=batch['wp_mask'],
                cd_mask=cd_mask,
                wp_time_bias=layer_wp_time_bias,
            )

        # Output heads
        global_context = global_emb.mean(dim=1)
        global_context = global_context + _masked_mean(wp_emb, batch['wp_mask'])
        global_context = global_context + _masked_mean(cd_emb, cd_mask)

        q1 = query_emb[:, 0, :] + global_context
        q2 = query_emb[:, 1, :] + global_context

        pred_u1 = self.head1(q1)
        pred_u2 = self.head2(q2)

        return {
            'pred_u1': pred_u1,
            'pred_u2': pred_u2,
        }
