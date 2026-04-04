"""
Hybrid Token Transformer for JUNO Ordered Dual-Endpoint Reconstruction.

Architecture:
  WP hits (hit-level) + CD patches (HEALPix) + Global tokens + Query tokens
  → Hybrid Encoder with structured attention
  → 2 Query outputs → Endpoint Heads → ordered unit vectors

Attention rules:
  WP↔WP: dense
  WP↔CD: dense cross-attention (bidirectional)
  CD↔CD: kNN local self-attention (mask-based)
  Global↔All: dense
  Query↔All: dense cross-attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from TokenProjector import (
    WPProjector, CDProjector, TokenTypeEmbedding, FourierPositionEncoding,
    TOKEN_WP, TOKEN_CD, TOKEN_GLOBAL, TOKEN_QUERY,
)
from PositionEncoding import BucketRPE


# =============================================================================
# Core building blocks
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional mask and RPE bias."""

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
                rpe_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (B, S_q, D)
            key:   (B, S_k, D)
            value: (B, S_k, D)
            mask:  (B, S_q, S_k) bool, True = IGNORE (padding mask)
            rpe_bias: (B, H, S_q, S_k) additive bias

        Returns:
            (B, S_q, D)
        """
        B, Sq, _ = query.shape
        _, Sk, _ = key.shape

        q = self.wq(query).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(key).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(value).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)

        if rpe_bias is None:
            # --- SDPA path: FlashAttention / memory-efficient backend ---
            attn_mask = None
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)  # (B, 1, Sq, Sk)
                attn_mask = torch.zeros_like(mask, dtype=q.dtype)
                attn_mask.masked_fill_(mask, float('-inf'))

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            # --- Manual path: CD self-attention with RPE bias ---
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, Sq, Sk)
            scores = scores + rpe_bias

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


class HybridEncoderLayer(nn.Module):
    """
    One encoder layer with structured hybrid attention.

    Steps:
    1. WP self-attention (dense)
    2. CD self-attention (kNN masked)
    3. WP↔CD cross-attention (bidirectional, dense)
    4. Global↔All dense attention
    5. Query↔All dense cross-attention
    6. FFN for each group
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 cd_knn_k: int, num_global: int, num_queries: int,
                 use_rpe: bool = True, num_rpe_angle_buckets: int = 64,
                 num_rpe_time_buckets: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.use_rpe = use_rpe

        # Attention modules
        self.wp_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cd_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.wp_cd_cross = MultiHeadAttention(d_model, num_heads, dropout)  # WP queries CD
        self.cd_wp_cross = MultiHeadAttention(d_model, num_heads, dropout)  # CD queries WP
        self.global_attn = MultiHeadAttention(d_model, num_heads, dropout)  # Global queries All
        self.query_attn = MultiHeadAttention(d_model, num_heads, dropout)   # Query queries All

        # FFN
        self.wp_ffn = FeedForward(d_model, d_ff, dropout)
        self.cd_ffn = FeedForward(d_model, d_ff, dropout)
        self.global_ffn = FeedForward(d_model, d_ff, dropout)
        self.query_ffn = FeedForward(d_model, d_ff, dropout)

        # LayerNorms (Pre-LN style)
        self.wp_attn_norm = nn.LayerNorm(d_model)
        self.cd_attn_norm = nn.LayerNorm(d_model)
        self.wp_cross_norm = nn.LayerNorm(d_model)
        self.cd_cross_norm = nn.LayerNorm(d_model)
        self.global_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.wp_ffn_norm = nn.LayerNorm(d_model)
        self.cd_ffn_norm = nn.LayerNorm(d_model)
        self.global_ffn_norm = nn.LayerNorm(d_model)
        self.query_ffn_norm = nn.LayerNorm(d_model)

        # RPE (shared across this layer's CD attention)
        if use_rpe:
            self.rpe = BucketRPE(num_heads, num_rpe_angle_buckets, num_rpe_time_buckets)

    def _build_cd_knn_mask(self, cd_knn_adj: torch.Tensor,
                           cd_mask: torch.Tensor,
                           N_cd: int) -> torch.Tensor:
        """
        Build attention mask for CD kNN local attention.

        Args:
            cd_knn_adj: (N_cd,) each row contains allowed neighbor indices,
                        padded to cd_knn_k. Shape may vary per batch if patches differ.
            cd_mask: (B, N_cd) True = padding (no hit in this patch)
            N_cd: number of CD patches in this batch

        Returns:
            (B, N_cd, N_cd) bool mask: True = block attention
        """
        B = cd_mask.shape[0]
        device = cd_mask.device
        k = cd_knn_adj.shape[1] if cd_knn_adj.dim() == 2 else cd_knn_adj.shape[0]

        # Start with all blocked
        attn_mask = torch.ones(B, N_cd, N_cd, dtype=torch.bool, device=device)

        # Allow kNN connections
        # cd_knn_adj: (N_cd_max, k) or (N_cd, k)
        if cd_knn_adj.dim() == 2:
            src = torch.arange(N_cd, device=device).unsqueeze(1).expand(-1, cd_knn_adj.shape[1])
            valid = cd_knn_adj[:N_cd] < N_cd
            attn_mask[:, src[valid], cd_knn_adj[:N_cd][valid]] = False
        else:
            # 1D: per-patch neighbors
            for i in range(min(N_cd, len(cd_knn_adj))):
                neighbors = cd_knn_adj[i]
                valid_nbrs = neighbors[neighbors < N_cd]
                attn_mask[:, i, valid_nbrs] = False

        # Also block padded positions
        padding_mask = cd_mask.unsqueeze(1) | cd_mask.unsqueeze(2)  # (B, N_cd, N_cd)
        attn_mask = attn_mask | padding_mask

        return attn_mask

    def forward(self, wp_emb: torch.Tensor, cd_emb: torch.Tensor,
                global_emb: torch.Tensor, query_emb: torch.Tensor,
                wp_mask: torch.Tensor, cd_mask: torch.Tensor,
                cd_knn_adj: Optional[torch.Tensor] = None,
                wp_unit_vecs: Optional[torch.Tensor] = None,
                cd_unit_vecs: Optional[torch.Tensor] = None,
                wp_times: Optional[torch.Tensor] = None,
                cd_times: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            wp_emb: (B, N_wp, D) WP token embeddings
            cd_emb: (B, N_cd, D) CD patch embeddings
            global_emb: (B, M, D) global token embeddings
            query_emb: (B, Q, D) query token embeddings
            wp_mask: (B, N_wp) bool True=padding
            cd_mask: (B, N_cd) bool True=padding
            cd_knn_adj: (N_max_cd, k) or (N_cd, k) kNN adjacency
            wp/cd_unit_vecs: for RPE computation
            wp/cd_times: for RPE computation

        Returns:
            (wp_out, cd_out, global_out, query_out)
        """
        B = wp_emb.shape[0]
        N_wp = wp_emb.shape[1]
        N_cd = cd_emb.shape[1]
        M = global_emb.shape[1]
        Q = query_emb.shape[1]

        # --- 1. WP self-attention (dense) ---
        wp_normed = self.wp_attn_norm(wp_emb)
        wp_attn_mask = wp_mask.unsqueeze(1) | wp_mask.unsqueeze(2)  # (B, N_wp, N_wp)
        wp_out = wp_emb + self.wp_self_attn(wp_normed, wp_normed, wp_normed, mask=wp_attn_mask)

        # --- 2. CD self-attention (kNN masked) ---
        cd_normed = self.cd_attn_norm(cd_emb)
        if cd_knn_adj is not None:
            cd_attn_mask = self._build_cd_knn_mask(cd_knn_adj, cd_mask, N_cd)
        else:
            cd_attn_mask = cd_mask.unsqueeze(1) | cd_mask.unsqueeze(2)

        # Optional RPE for CD attention
        cd_rpe = None
        if self.use_rpe and cd_unit_vecs is not None:
            cd_rpe = self._compute_cd_rpe(cd_unit_vecs, cd_times)

        cd_out = cd_emb + self.cd_self_attn(cd_normed, cd_normed, cd_normed,
                                             mask=cd_attn_mask, rpe_bias=cd_rpe)

        # --- 3. WP↔CD cross-attention (dense, bidirectional) ---
        wp_cross_in = self.wp_cross_norm(wp_out)
        cd_cross_in = self.cd_cross_norm(cd_out)

        # WP queries CD
        wp_cross_mask = cd_mask.unsqueeze(1).expand(-1, N_wp, -1)  # (B, N_wp, N_cd)
        wp_out = wp_out + self.wp_cd_cross(wp_cross_in, cd_cross_in, cd_cross_in,
                                            mask=wp_cross_mask)

        # CD queries WP
        cd_cross_mask = wp_mask.unsqueeze(1).expand(-1, N_cd, -1)  # (B, N_cd, N_wp)
        cd_out = cd_out + self.cd_wp_cross(cd_cross_in, wp_cross_in, wp_cross_in,
                                            mask=cd_cross_mask)

        # --- 4. Global↔All dense attention ---
        all_tokens = torch.cat([wp_out, cd_out], dim=1)  # (B, N_wp+N_cd, D)
        all_mask = torch.cat([wp_mask, cd_mask], dim=1)   # (B, N_wp+N_cd)

        global_normed = self.global_norm(global_emb)
        global_attn_mask = all_mask.unsqueeze(1).expand(-1, M, -1)  # (B, M, N_wp+N_cd)
        global_out = global_emb + self.global_attn(global_normed, all_tokens, all_tokens,
                                                    mask=global_attn_mask)

        # --- 5. Query↔All dense cross-attention ---
        all_with_global = torch.cat([wp_out, cd_out, global_out], dim=1)
        all_global_mask = torch.cat([wp_mask, cd_mask,
                                      torch.zeros(B, M, dtype=torch.bool, device=wp_mask.device)], dim=1)

        query_normed = self.query_norm(query_emb)
        query_attn_mask = all_global_mask.unsqueeze(1).expand(-1, Q, -1)
        query_out = query_emb + self.query_attn(query_normed, all_with_global, all_with_global,
                                                 mask=query_attn_mask)

        # --- 6. FFN (Pre-LN) ---
        wp_out = wp_out + self.wp_ffn(self.wp_ffn_norm(wp_out))
        cd_out = cd_out + self.cd_ffn(self.cd_ffn_norm(cd_out))
        global_out = global_out + self.global_ffn(self.global_ffn_norm(global_out))
        query_out = query_out + self.query_ffn(self.query_ffn_norm(query_out))

        return wp_out, cd_out, global_out, query_out

    def _compute_cd_rpe(self, cd_unit_vecs: torch.Tensor,
                        cd_times: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Compute RPE bias for CD attention."""
        B, N_cd, _ = cd_unit_vecs.shape
        # Angular distances: (B, N_cd, N_cd)
        dots = torch.bmm(cd_unit_vecs, cd_unit_vecs.transpose(1, 2))
        angles = torch.acos(dots.clamp(-1.0 + 1e-6, 1.0 - 1e-6))

        if cd_times is not None:
            # Time differences: (B, N_cd, N_cd)
            dt = (cd_times.unsqueeze(2) - cd_times.unsqueeze(1)).abs()
        else:
            dt = torch.zeros_like(angles)

        # Expand for heads: (B, H, N_cd, N_cd)
        angles = angles.unsqueeze(1).expand(-1, self.rpe.num_heads, -1, -1)
        dt = dt.unsqueeze(1).expand(-1, self.rpe.num_heads, -1, -1)

        return self.rpe(angles, dt)


class EndpointHead(nn.Module):
    """
    Predicts a single unit vector endpoint from a query token.

    query → Linear → GELU → Dropout → Linear → 3D vector → Normalize
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, D)

        Returns:
            (B, 3) unit vector
        """
        v = self.mlp(query)
        return F.normalize(v, dim=-1)


# =============================================================================
# Full model
# =============================================================================

class HTTransformer(nn.Module):
    """
    Hybrid Token Transformer for ordered dual-endpoint reconstruction.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        data_cfg = cfg['data']
        model_cfg = cfg['model']

        self.d_model = model_cfg['d_model']
        self.num_layers = model_cfg['num_layers']
        self.num_global = model_cfg['num_global_tokens']
        self.num_queries = model_cfg['num_queries']
        self.num_time_bins = data_cfg['num_time_bins']

        # Token projectors
        self.wp_projector = WPProjector(input_dim=5, d_model=self.d_model)
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

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            HybridEncoderLayer(
                d_model=self.d_model,
                num_heads=model_cfg['num_heads'],
                d_ff=model_cfg['d_ff'],
                cd_knn_k=model_cfg['cd_knn_k'],
                num_global=self.num_global,
                num_queries=self.num_queries,
                use_rpe=(model_cfg['rel_posenc'] == 'bucket'),
                num_rpe_angle_buckets=model_cfg.get('num_rpe_angle_buckets', 64),
                num_rpe_time_buckets=model_cfg.get('num_rpe_time_buckets', 64),
                dropout=model_cfg.get('dropout', 0.1),
            )
            for _ in range(self.num_layers)
        ])

        # Output heads (one per query)
        self.head1 = EndpointHead(self.d_model, dropout=model_cfg.get('dropout', 0.1))
        self.head2 = EndpointHead(self.d_model, dropout=model_cfg.get('dropout', 0.1))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'norm' not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch dict with keys:
                wp_tokens: (B, N_wp, 5) raw WP features [ux,uy,uz,q,t]
                wp_mask: (B, N_wp) bool True=padding
                cd_unit_vecs: (B, N_cd, 3) CD patch center unit vectors
                cd_stats: (B, N_cd, 4) CD patch stats [sumQ,count,t_min,t_mean]
                cd_time_bins: (B, N_cd, B_bins) CD patch time-bin sequences
                cd_mask: (B, N_cd) bool True=padding
                cd_knn_adj: (N_max_cd, k) kNN adjacency indices
                wp_unit_vecs: (B, N_wp, 3) for RPE
                cd_times_mean: (B, N_cd) mean time per CD patch for RPE
                wp_times: (B, N_wp) time per WP hit for RPE

        Returns:
            dict with pred_u1, pred_u2 (B, 3) unit vectors
        """
        B = batch['wp_tokens'].shape[0]

        # --- Project tokens ---
        wp_emb = self.wp_projector(batch['wp_tokens'])       # (B, N_wp, D)
        cd_emb = self.cd_projector(
            batch['cd_unit_vecs'], batch['cd_stats'], batch['cd_time_bins']
        )  # (B, N_cd, D)

        # --- Add type embeddings ---
        wp_type = torch.full((B, wp_emb.shape[1]), TOKEN_WP, device=wp_emb.device, dtype=torch.long)
        cd_type = torch.full((B, cd_emb.shape[1]), TOKEN_CD, device=cd_emb.device, dtype=torch.long)
        global_type = torch.full((B, self.num_global), TOKEN_GLOBAL, device=wp_emb.device, dtype=torch.long)
        query_type = torch.full((B, self.num_queries), TOKEN_QUERY, device=wp_emb.device, dtype=torch.long)

        wp_emb = wp_emb + self.type_embedding(wp_type)
        cd_emb = cd_emb + self.type_embedding(cd_type)

        # --- Add absolute position encoding ---
        wp_pe = self.abs_pe(batch['wp_unit_vecs'])
        cd_pe = self.abs_pe(batch['cd_unit_vecs'])
        wp_emb = wp_emb + wp_pe
        cd_emb = cd_emb + cd_pe

        # --- Expand learnable tokens ---
        global_emb = self.global_tokens.expand(B, -1, -1)
        query_emb = self.query_tokens.expand(B, -1, -1)
        global_emb = global_emb + self.type_embedding(global_type)
        query_emb = query_emb + self.type_embedding(query_type)

        # --- Encoder layers ---
        cd_knn_adj = batch.get('cd_knn_adj', None)
        cd_times = batch.get('cd_times_mean', None)
        wp_times = batch.get('wp_times', None)

        for layer in self.encoder_layers:
            wp_emb, cd_emb, global_emb, query_emb = layer(
                wp_emb, cd_emb, global_emb, query_emb,
                wp_mask=batch['wp_mask'],
                cd_mask=batch['cd_mask'],
                cd_knn_adj=cd_knn_adj,
                wp_unit_vecs=batch['wp_unit_vecs'],
                cd_unit_vecs=batch['cd_unit_vecs'],
                wp_times=wp_times,
                cd_times=cd_times,
            )

        # --- Output heads ---
        # query_emb: (B, num_queries=2, D)
        q1 = query_emb[:, 0, :]  # (B, D)
        q2 = query_emb[:, 1, :]  # (B, D)

        pred_u1 = self.head1(q1)  # (B, 3) unit vector
        pred_u2 = self.head2(q2)  # (B, 3) unit vector

        return {
            'pred_u1': pred_u1,
            'pred_u2': pred_u2,
        }
