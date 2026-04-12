"""
Hybrid Token Transformer for JUNO Ordered Dual-Endpoint Reconstruction (V3).

Architecture:
  WP hits (hit-level) + CD sparse PMT tokens + Global tokens + Query tokens
  → WP backbone (self-attention) + CD sparse encoder (self-attention)
  → Bidirectional CD↔WP cross-attention fusion
  → Late fusion readout → 2 Query outputs → Endpoint Heads → ordered unit vectors

V3 Architecture (CD Sparse PMT Token + Cross-Attention Fusion):
  - CD: per-PMT aggregation + TopK charge selection → sparse PMT tokens
  - CD: self-attention encoder (1 layer)
  - WP: independent self-attention backbone (2 layers)
  - Bidirectional fusion: CrossAttentionFusion (CD↔WP cross-attention) ×N
  - Late fusion: Global/Query read from [WP, CD]
  - Information flows: WP↔CD (bidirectional), WP/CD → Query

WP features:
  - WPProjector: dual-branch [ux,uy,uz]⊕[q,t] feature extraction
  - WPTimeEncoding: token-level time encoding (SDPA-compatible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from models.components.token_projectors import (
    WPProjector, CDHitProjector, CDTimeEmbedding,
    TokenTypeEmbedding, FourierPositionEncoding,
    TOKEN_WP, TOKEN_CD, TOKEN_GLOBAL, TOKEN_QUERY,
)
from models.components.wp_time_encoding import WPTimeEncoding
from models.components.norms import RMSNorm


def _get_norm(norm_type: str, d_model: int):
    """Get normalization layer by type."""
    if norm_type == 'rmsnorm':
        return RMSNorm(d_model)
    return nn.LayerNorm(d_model)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention using SDPA (FlashAttention) fast path.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, D)
            key:   (B, N_k, D)
            value: (B, N_k, D)
            mask:  (B, N_q, N_k) bool, True = ignore (padding)
        Returns:
            (B, N_q, D)
        """
        B = query.shape[0]

        q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA with attention mask
        attn_mask = None
        if mask is not None:
            # Expand for heads: (B, N_q, N_k) -> (B, 1, N_q, N_k)
            attn_mask = mask.unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Standard FFN with GELU."""

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


class WPSelfAttentionLayer(nn.Module):
    """WP backbone layer: self-attention + FFN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, norm_type: str = 'layernorm'):
        super().__init__()
        NormClass = _get_norm(norm_type, d_model) if callable(_get_norm(norm_type, d_model)) else nn.LayerNorm
        self.wp_attn_norm = _get_norm(norm_type, d_model)
        self.wp_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.wp_ffn_norm = _get_norm(norm_type, d_model)
        self.wp_ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, wp_emb: torch.Tensor, wp_attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wp_emb: (B, N_wp, D)
            wp_attn_mask: (B, N_wp, N_wp) bool, True = ignore
        Returns:
            (B, N_wp, D)
        """
        wp_normed = self.wp_attn_norm(wp_emb)
        wp_out = wp_emb + self.wp_self_attn(wp_normed, wp_normed, wp_normed, mask=wp_attn_mask)
        wp_out = wp_out + self.wp_ffn(self.wp_ffn_norm(wp_out))
        return wp_out


class CDSparseEncoderLayer(nn.Module):
    """Self-attention layer for CD sparse PMT tokens (v3).

    Pre-LN + self-attention + residual + FFN + residual.
    SDPA-compatible, uses cd_mask for padding.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, norm_type: str = 'layernorm'):
        super().__init__()
        self.self_attn_norm = _get_norm(norm_type, d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn_norm = _get_norm(norm_type, d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, cd_emb: torch.Tensor, cd_attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cd_emb: (B, K_cd, D)
            cd_attn_mask: (B, K_cd, K_cd) bool, True = ignore
        Returns:
            (B, K_cd, D)
        """
        normed = self.self_attn_norm(cd_emb)
        cd_out = cd_emb + self.self_attn(normed, normed, normed, mask=cd_attn_mask)
        cd_out = cd_out + self.ffn(self.ffn_norm(cd_out))
        return cd_out


class CrossAttentionFusionLayer(nn.Module):
    """Bidirectional CD↔WP cross-attention fusion layer.

    In one layer:
      CD→WP: WP queries attend to CD keys/values (WP gets CD context)
      WP→CD: CD queries attend to WP keys/values (CD gets WP context)
    Plus per-branch FFN.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, norm_type: str = 'layernorm'):
        super().__init__()
        # WP→CD cross-attention (CD queries attend to WP)
        self.cd_cross_norm = _get_norm(norm_type, d_model)
        self.wp_cross_norm = _get_norm(norm_type, d_model)
        self.cd_cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # CD→WP cross-attention (WP queries attend to CD)
        self.wp_cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Per-branch FFN
        self.wp_ffn_norm = _get_norm(norm_type, d_model)
        self.wp_ffn = FeedForward(d_model, d_ff, dropout)
        self.cd_ffn_norm = _get_norm(norm_type, d_model)
        self.cd_ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, wp_emb: torch.Tensor, cd_emb: torch.Tensor,
                wp_mask: torch.Tensor, cd_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wp_emb: (B, N_wp, D)
            cd_emb: (B, K_cd, D)
            wp_mask: (B, N_wp) bool, True = padding
            cd_mask: (B, K_cd) bool, True = padding
        Returns:
            (wp_emb, cd_emb): updated embeddings
        """
        wp_normed = self.wp_cross_norm(wp_emb)
        cd_normed = self.cd_cross_norm(cd_emb)

        # CD→WP: WP queries attend to CD. mask: (B, N_wp, K_cd)
        wp_cd_mask = cd_mask.unsqueeze(1).expand(-1, wp_emb.shape[1], -1)
        wp_emb = wp_emb + self.wp_cross_attn(wp_normed, cd_normed, cd_normed, mask=wp_cd_mask)

        # WP→CD: CD queries attend to WP. mask: (B, K_cd, N_wp)
        cd_wp_mask = wp_mask.unsqueeze(1).expand(-1, cd_emb.shape[1], -1)
        cd_emb = cd_emb + self.cd_cross_attn(cd_normed, wp_normed, wp_normed, mask=cd_wp_mask)

        # Per-branch FFN
        wp_emb = wp_emb + self.wp_ffn(self.wp_ffn_norm(wp_emb))
        cd_emb = cd_emb + self.cd_ffn(self.cd_ffn_norm(cd_emb))

        return wp_emb, cd_emb


class CrossModalReadout(nn.Module):
    """Final readout: Global and Query tokens attend to [WP, CD]."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_global: int, num_queries: int,
                 dropout: float = 0.1, norm_type: str = 'layernorm'):
        super().__init__()
        self.num_global = num_global
        self.num_queries = num_queries

        # Global attends to [WP, CD]
        self.global_norm = _get_norm(norm_type, d_model)
        self.global_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.global_ffn_norm = _get_norm(norm_type, d_model)
        self.global_ffn = FeedForward(d_model, d_ff, dropout)

        # Query attends to [WP, CD, Global]
        self.query_norm = _get_norm(norm_type, d_model)
        self.query_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.query_ffn_norm = _get_norm(norm_type, d_model)
        self.query_ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, wp_emb: torch.Tensor, cd_emb: torch.Tensor,
                global_emb: torch.Tensor, query_emb: torch.Tensor,
                mask_pack: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wp_emb: (B, N_wp, D) WP embeddings (read-only)
            cd_emb: (B, N_cd, D) CD latent tokens Z_cd (read-only)
            global_emb: (B, M, D) global tokens
            query_emb: (B, Q, D) query tokens
            mask_pack: dict with 'global' and 'query' masks

        Returns:
            (global_out, query_out): updated global and query embeddings
        """
        # Global attends to [WP, CD]
        all_tokens = torch.cat([wp_emb, cd_emb], dim=1)
        global_normed = self.global_norm(global_emb)
        global_out = global_emb + self.global_attn(global_normed, all_tokens, all_tokens,
                                                    mask=mask_pack['global'])

        # Query attends to [WP, CD, Global]
        all_with_global = torch.cat([wp_emb, cd_emb, global_out], dim=1)
        query_normed = self.query_norm(query_emb)
        query_out = query_emb + self.query_attn(query_normed, all_with_global, all_with_global,
                                                 mask=mask_pack['query'])

        # FFN
        global_out = global_out + self.global_ffn(self.global_ffn_norm(global_out))
        query_out = query_out + self.query_ffn(self.query_ffn_norm(query_out))

        return global_out, query_out


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
    """Hybrid Token Transformer for ordered dual-endpoint reconstruction (V3).

    V3 Architecture:
      WP: WPProjector → WPTimeEncoding → WPSelfAttentionLayer ×N
      CD: CDHitProjector → CDTimeEmbedding → CDSparseEncoderLayer ×1
      Fusion: CrossAttentionFusion (CD↔WP bidirectional) ×N
      Readout: CrossModalReadout (Global/Query read [WP, CD])
      Output: dual endpoint heads
    """

    def __init__(self, cfg: dict):
        super().__init__()
        data_cfg = cfg['data']
        model_cfg = cfg['model']

        self.d_model = model_cfg['d_model']
        self.num_global = model_cfg['num_global_tokens']
        self.num_queries = model_cfg['num_queries']
        self.norm_type = model_cfg.get('norm_type', 'layernorm')

        # WP time encoding module
        if model_cfg.get('wp_time_encoding', True):
            self.wp_time_encoding = WPTimeEncoding(
                d_model=self.d_model,
                hidden=model_cfg.get('wp_time_hidden', 32),
                fourier_dim=model_cfg.get('wp_time_fourier_dim', 16),
            )
        else:
            self.wp_time_encoding = None

        # WP projector
        self.wp_projector = WPProjector(
            d_model=self.d_model,
            d_geo=model_cfg.get('wp_geo_hidden', 32),
            d_qt=model_cfg.get('wp_qt_hidden', 32),
            hidden=model_cfg.get('wp_projector_hidden', 64),
            dropout=model_cfg.get('dropout', 0.1),
        )

        # CD Hit Projector (v3)
        self.cd_hit_projector = CDHitProjector(
            d_model=self.d_model,
            d_geo=model_cfg.get('wp_geo_hidden', 32),
            d_pt=64,
            hidden=64,
            dropout=model_cfg.get('dropout', 0.1),
        )

        # CD Time Embedding (v3)
        if model_cfg.get('cd_time_embedding', True):
            self.cd_time_embedding = CDTimeEmbedding(
                d_model=self.d_model,
                hidden=model_cfg.get('cd_time_hidden', 32),
                fourier_dim=model_cfg.get('cd_time_fourier_dim', 16),
            )
        else:
            self.cd_time_embedding = None

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

        # WP self-attention encoder
        wp_layers = model_cfg.get('wp_self_layers', 2)
        self.wp_layers = nn.ModuleList([
            WPSelfAttentionLayer(
                d_model=self.d_model,
                num_heads=model_cfg['num_heads'],
                d_ff=model_cfg['d_ff'],
                dropout=model_cfg.get('dropout', 0.1),
                norm_type=self.norm_type,
            )
            for _ in range(wp_layers)
        ])

        # CD sparse self-attention encoder (v3)
        cd_layers = model_cfg.get('cd_self_layers', 1)
        self.cd_sparse_encoder = nn.ModuleList([
            CDSparseEncoderLayer(
                d_model=self.d_model,
                num_heads=model_cfg['num_heads'],
                d_ff=model_cfg['d_ff'],
                dropout=model_cfg.get('dropout', 0.1),
                norm_type=self.norm_type,
            )
            for _ in range(cd_layers)
        ])

        # Bidirectional CD↔WP cross-attention fusion
        fusion_layers = model_cfg.get('fusion_layers', 2)
        self.fusion_layers = nn.ModuleList([
            CrossAttentionFusionLayer(
                d_model=self.d_model,
                num_heads=model_cfg['num_heads'],
                d_ff=model_cfg['d_ff'],
                dropout=model_cfg.get('dropout', 0.1),
                norm_type=self.norm_type,
            )
            for _ in range(fusion_layers)
        ])

        # Cross-modal readout
        self.readout = CrossModalReadout(
            d_model=self.d_model,
            num_heads=model_cfg['num_heads'],
            d_ff=model_cfg['d_ff'],
            num_global=self.num_global,
            num_queries=self.num_queries,
            dropout=model_cfg.get('dropout', 0.1),
            norm_type=self.norm_type,
        )

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

        # ── 1. WP projection + time encoding ──
        wp_emb = self.wp_projector(batch['wp_tokens'])
        if self.wp_time_encoding is not None:
            wp_times = batch.get('wp_times', None)
            if wp_times is not None:
                wp_emb = wp_emb + self.wp_time_encoding(wp_times)

        # ── 2. CD projection + time embedding ──
        cd_tokens = batch['cd_tokens']                        # (B, K_cd, 10)
        cd_emb = self.cd_hit_projector(cd_tokens)             # (B, K_cd, d_model)
        if self.cd_time_embedding is not None:
            cd_time_feat = cd_tokens[..., 6:10]               # [t_first, t_mean, t_late, t_span]
            cd_emb = cd_emb + self.cd_time_embedding(cd_time_feat)

        # ── 3. Type + position embeddings ──
        wp_type = torch.full((B, wp_emb.shape[1]), TOKEN_WP, device=wp_emb.device, dtype=torch.long)
        cd_type = torch.full((B, cd_emb.shape[1]), TOKEN_CD, device=cd_emb.device, dtype=torch.long)
        global_type = torch.full((B, self.num_global), TOKEN_GLOBAL, device=wp_emb.device, dtype=torch.long)
        query_type = torch.full((B, self.num_queries), TOKEN_QUERY, device=wp_emb.device, dtype=torch.long)

        wp_emb = wp_emb + self.type_embedding(wp_type)
        cd_emb = cd_emb + self.type_embedding(cd_type)

        wp_pe = self.abs_pe(batch['wp_unit_vecs'])
        cd_pe = self.abs_pe(cd_tokens[..., :3])                # PMT direction vectors
        wp_emb = wp_emb + wp_pe
        cd_emb = cd_emb + cd_pe

        # Expand learnable tokens
        global_emb = self.global_tokens.expand(B, -1, -1)
        query_emb = self.query_tokens.expand(B, -1, -1)
        global_emb = global_emb + self.type_embedding(global_type)
        query_emb = query_emb + self.type_embedding(query_type)

        # ── 4. WP self-attention encoder ──
        wp_mask = batch['wp_mask']
        wp_attn_mask = wp_mask.unsqueeze(1) | wp_mask.unsqueeze(2)

        for wp_layer in self.wp_layers:
            wp_emb = wp_layer(wp_emb, wp_attn_mask)

        # ── 5. CD sparse self-attention encoder ──
        cd_mask = batch['cd_mask']                             # (B, K_cd)
        cd_attn_mask = cd_mask.unsqueeze(1) | cd_mask.unsqueeze(2)

        for cd_layer in self.cd_sparse_encoder:
            cd_emb = cd_layer(cd_emb, cd_attn_mask)

        # ── 6. Bidirectional CD↔WP cross-attention fusion ──
        for fusion_layer in self.fusion_layers:
            wp_emb, cd_emb = fusion_layer(wp_emb, cd_emb, wp_mask, cd_mask)

        # ── 7. Late fusion: CrossModalReadout ──
        M = self.num_global
        Q = self.num_queries

        # CD mask: real padding mask from cd_tokens
        all_mask = torch.cat([wp_mask, cd_mask], dim=1)
        global_attn_mask = all_mask.unsqueeze(1).expand(-1, M, -1)

        all_global_mask = torch.cat([wp_mask, cd_mask,
                                      torch.zeros(B, M, dtype=torch.bool, device=wp_mask.device)], dim=1)
        query_attn_mask = all_global_mask.unsqueeze(1).expand(-1, Q, -1)

        readout_mask_pack = {
            'global': global_attn_mask,
            'query': query_attn_mask,
        }

        global_emb, query_emb = self.readout(wp_emb, cd_emb, global_emb, query_emb, readout_mask_pack)

        # ── 7. Output heads (with global context pooling) ──
        global_context = global_emb.mean(dim=1)
        global_context = global_context + _masked_mean(wp_emb, batch['wp_mask'])
        global_context = global_context + _masked_mean(cd_emb, batch['cd_mask'])

        q1 = query_emb[:, 0, :] + global_context
        q2 = query_emb[:, 1, :] + global_context

        pred_u1 = self.head1(q1)
        pred_u2 = self.head2(q2)

        return {
            'pred_u1': pred_u1,
            'pred_u2': pred_u2,
        }
