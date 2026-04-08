"""
DeepSphere module for CD branch local spherical modeling.

Provides:
- DeepSphereBlock: Local spherical feature aggregation with graph-like structure
- DeepSphereEncoder: Stack of DeepSphereBlocks for CD patch encoding
- CDCompression: Compress CD tokens before fusion stage

Replaces the original CD self-attention (which had O(N^2) complexity)
with local neighborhood aggregation (O(N*k) complexity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More stable for deep architectures than LayerNorm in some cases.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim)

        Returns:
            (..., dim) RMS-normalized tensor
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight


def build_local_neighbor_graph(
    pixel_ids: torch.Tensor,
    full_knn_adj: torch.Tensor,
    cd_mask: Optional[torch.Tensor] = None,
    padding_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DEPRECATED in Stage B: This function is no longer needed.

    Stage B uses fixed HEALPix grid representation where:
    - token index == global HEALPix pixel id
    - all events share the same graph topology
    - no per-batch local graph construction is required

    Kept for legacy compatibility only. Use fixed knn_adj directly instead.
    """
    # Implementation left for reference, but not used in Stage B main path
    """
    Build local neighbor graph once per batch.

    Vectorized implementation using scatter/gather for GPU acceleration.

    Args:
        pixel_ids: (B, N) global HEALPix pixel ids for active/padded CD tokens
        full_knn_adj: (npix, k) global neighbor table
        cd_mask: (B, N) bool, True = padding
        padding_value: value for invalid local neighbor

    Returns:
        local_neighbor_indices: (B, N, k) local indices
        valid_neighbor_mask: (B, N, k) bool, True = valid
    """
    B, N = pixel_ids.shape
    k = full_knn_adj.shape[1]
    device = pixel_ids.device
    max_pix_id = full_knn_adj.shape[0]

    # Get global neighbors for each active pixel
    global_neighbors = full_knn_adj[pixel_ids]  # (B, N, k)

    # Build global -> local mapping table using scatter
    # mapping[b, global_id] = local_idx (or padding_value if not present)
    mapping = torch.full((B, max_pix_id), padding_value, dtype=torch.long, device=device)

    if cd_mask is not None:
        # Only map valid (non-padding) tokens
        valid_indices = (~cd_mask).nonzero(as_tuple=True)  # (batch_indices, seq_indices)
        valid_pixel_ids = pixel_ids[valid_indices]
        valid_local_indices = valid_indices[1]  # seq_indices are the local indices
        batch_indices = valid_indices[0]
        mapping[batch_indices, valid_pixel_ids] = valid_local_indices
    else:
        local_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
        mapping.scatter_(1, pixel_ids, local_indices)

    # Lookup local indices for all global neighbors at once
    local_neighbor_indices = mapping.gather(1, global_neighbors.view(B, -1)).view(B, N, k)
    valid_neighbor_mask = local_neighbor_indices != padding_value

    return local_neighbor_indices, valid_neighbor_mask


class DeepSphereBlock(nn.Module):
    """
    DeepSphere block for local spherical feature aggregation on fixed HEALPix grid.

    Stage B optimization: Runs on fixed graph topology where node index = HEALPix pixel id.
    All events share the same graph structure, eliminating runtime local graph construction.

    Structure: Pre-LN -> LocalConv -> GELU -> Residual

    Input: (B, npix, D) where npix is fixed HEALPix grid size (e.g., 768 for nside=8)
    Output: (B, npix, D)
    """

    def __init__(self, d_model: int, hidden_dim: int, k_neighbors: int = 16,
                 dropout: float = 0.1, norm_type: str = 'rmsnorm'):
        super().__init__()
        self.d_model = d_model
        self.k_neighbors = k_neighbors
        self.norm_type = norm_type

        # Normalization
        if norm_type == 'rmsnorm':
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        # Local feature transformation: similar to a graph convolution
        # We use a simple MLP that operates on aggregated neighbor features
        self.neighbor_proj = nn.Linear(d_model, hidden_dim)
        self.self_proj = nn.Linear(d_model, hidden_dim)

        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def _aggregate_neighbors(
        self,
        x: torch.Tensor,
        neighbor_indices: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate features from kNN neighbors with valid mask support.

        Vectorized implementation using torch.gather (no Python loops).

        Args:
            x: (B, N, D) input features
            neighbor_indices: (B, N, k) or (N, k) local neighbor indices
            valid_mask: (B, N, k) bool mask, True = valid neighbor

        Returns:
            (B, N, D) aggregated neighbor features
        """
        B, N, D = x.shape
        k = neighbor_indices.shape[-1]

        # Handle fixed global neighbor indices (N, k) -> (B, N, k)
        if neighbor_indices.dim() == 2:
            neighbor_indices = neighbor_indices.unsqueeze(0).expand(B, -1, -1)

        # Vectorized gather: clamp indices to avoid out-of-bounds
        safe_indices = neighbor_indices.clamp(min=0, max=N-1)  # (B, N, k)

        # Expand x for gathering: (B, N, D) -> (B, N, k, D)
        x_expand = x.unsqueeze(2).expand(B, N, k, D)
        gather_idx = safe_indices.unsqueeze(-1).expand(B, N, k, D)

        # Gather neighbor features: (B, N, k, D)
        neighbor_features = torch.gather(x_expand, dim=1, index=gather_idx)

        # Apply valid mask for aggregation
        if valid_mask is not None:
            # Masked mean pooling: only average valid neighbors
            valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, N, 1)
            masked_features = neighbor_features * valid_mask.unsqueeze(-1).to(x.dtype)  # (B, N, k, D)
            return masked_features.sum(dim=2) / valid_count.to(x.dtype)  # (B, N, D)
        else:
            # Simple mean pooling over all k neighbors
            return neighbor_features.mean(dim=2)  # (B, N, D)

    def forward(
        self,
        x: torch.Tensor,
        knn_adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, npix, D) input CD patch embeddings on fixed HEALPix grid
            knn_adj: (npix, k) global kNN adjacency table (fixed for all batches)
            mask: (B, npix) bool mask, True = padding (no data at this pixel)

        Returns:
            (B, npix, D) output embeddings
        """
        # Pre-LN
        x_norm = self.norm(x)

        if knn_adj is not None:
            # Build valid neighbor mask from node mask if provided
            valid_mask = None
            if mask is not None:
                # (B, N) -> (B, N, 1) -> (B, N, k) indicating which neighbors are valid
                valid_mask = ~mask.unsqueeze(-1).expand(-1, -1, self.k_neighbors)

            # Local aggregation with fixed kNN adjacency (no per-batch remapping needed)
            neighbor_agg = self._aggregate_neighbors(x_norm, knn_adj, valid_mask)
        else:
            # Fallback: global mean pooling (for compatibility)
            neighbor_agg = x_norm.mean(dim=1, keepdim=True).expand_as(x_norm)

        # Transform
        neighbor_feat = self.neighbor_proj(neighbor_agg)
        self_feat = self.self_proj(x_norm)
        combined = neighbor_feat + self_feat

        # MLP + Residual
        out = x + self.mlp(combined)
        return out


class DeepSphereEncoder(nn.Module):
    """
    DeepSphere encoder: stack of DeepSphereBlocks on fixed HEALPix grid.

    Stage B optimization: Works on fixed graph topology without runtime local graph construction.
    Input and output have fixed shape (B, npix, D) where npix corresponds to HEALPix grid size.

    Input: (B, npix, D) dense HEALPix grid embeddings
    Output: (B, npix, D)
    """

    def __init__(self, d_model: int, num_layers: int = 4, hidden_dim: int = 256,
                 k_neighbors: int = 16, dropout: float = 0.1, norm_type: str = 'rmsnorm'):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.blocks = nn.ModuleList([
            DeepSphereBlock(d_model, hidden_dim, k_neighbors, dropout, norm_type)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        knn_adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, npix, D) input embeddings on fixed HEALPix grid
            knn_adj: (npix, k) global kNN adjacency table (fixed for all batches)
            mask: (B, npix) bool mask, True = padding (no data at this pixel)

        Returns:
            (B, npix, D) encoded embeddings
        """
        # Stage B: No runtime local graph construction needed
        # The graph is fixed: node index = HEALPix pixel id
        # knn_adj is directly used by all blocks without per-batch remapping

        for block in self.blocks:
            x = block(x, knn_adj=knn_adj, mask=mask)
        return x


class CDCompression(nn.Module):
    """
    Compress CD patch tokens before fusion stage.

    Reduces token count from high-resolution HEALPix patches (e.g., 768)
    to a manageable number for cross-attention (e.g., 64-192).

    Methods supported:
    - 'healpix_pool': HEALPix hierarchical pooling (nside reduction)
    - 'attention_pool': Learnable attention-based pooling
    - 'mean_pool': Simple mean pooling to fixed number of tokens
    """

    def __init__(self, d_model: int, target_tokens: int = 128,
                 method: str = 'healpix_pool', nside_in: int = 8,
                 nside_out: int = 4):
        super().__init__()
        self.d_model = d_model
        self.target_tokens = target_tokens
        self.method = method
        self.nside_in = nside_in
        self.nside_out = nside_out

        if method == 'healpix_pool':
            # HEALPix hierarchical pooling
            # nside=8 -> 768 pixels, nside=4 -> 192 pixels
            self._init_healpix_pooling()
        elif method == 'attention_pool':
            # Learnable query-based pooling
            self.pool_queries = nn.Parameter(torch.randn(1, target_tokens, d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        elif method == 'mean_pool':
            # Simple learnable pooling
            self.pool_proj = nn.Linear(d_model, d_model)
        else:
            raise ValueError(f"Unknown compression method: {method}")

    def _init_healpix_pooling(self):
        """Initialize HEALPix hierarchical pooling mappings."""
        try:
            import healpy as hp

            npix_in = hp.nside2npix(self.nside_in)
            npix_out = hp.nside2npix(self.nside_out)

            # Create mapping from high-res to low-res pixels
            # Each nside_in pixel maps to one nside_out pixel
            self.register_buffer('npix_in', torch.tensor(npix_in))
            self.register_buffer('npix_out', torch.tensor(npix_out))

            # Precompute mapping indices
            # For each low-res pixel, which high-res pixels map to it?
            high_to_low = []
            for i in range(npix_in):
                # Get direction vector
                vec = hp.pix2vec(self.nside_in, i, nest=False)
                # Map to low-res pixel
                theta = np.arccos(np.clip(vec[2], -1.0, 1.0))
                phi = np.arctan2(vec[1], vec[0]) % (2 * np.pi)
                low_pix = hp.ang2pix(self.nside_out, theta, phi, nest=False)
                high_to_low.append(low_pix)

            high_to_low = torch.tensor(high_to_low, dtype=torch.long)
            self.register_buffer('high_to_low', high_to_low)  # (npix_in,)

            # Create reverse mapping: for each low-res pixel, list of high-res indices
            self.low_to_high = []
            for j in range(npix_out):
                high_indices = torch.where(high_to_low == j)[0]
                self.low_to_high.append(high_indices)

        except ImportError:
            # Fallback if healpy not available
            self.high_to_low = None

    def _healpix_pool(
        self,
        x: torch.Tensor,
        pixel_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool high-resolution HEALPix patches to low-resolution.

        Args:
            x: (B, N_cd, D) patch embeddings
            pixel_ids: (B, N_cd) pixel IDs (in high-res nside)
            mask: (B, N_cd) bool mask, True = padding token to ignore

        Returns:
            pooled: (B, N_out, D) pooled embeddings
            pixel_ids_out: (B, N_out) low-res pixel IDs, padding = -1
        """
        B, N, D = x.shape

        if self.high_to_low is None:
            # Fallback: mean pooling
            return x.mean(dim=1, keepdim=True).expand(B, self.target_tokens, D), \
                   torch.full((B, self.target_tokens), -1, dtype=torch.long, device=x.device)

        # Map high-res pixel IDs to low-res
        pixel_ids_flat = pixel_ids.view(-1).clamp(0, len(self.high_to_low) - 1)
        low_ids_flat = self.high_to_low[pixel_ids_flat].view(B, N)

        # Gather all features that map to same low-res pixel
        # This is done via scatter_add
        npix_out = int(self.npix_out.item())

        pooled_list = []
        pixel_ids_out_list = []

        for b in range(B):
            # Output buffer
            out = torch.zeros(npix_out, D, device=x.device, dtype=x.dtype)
            counts = torch.zeros(npix_out, device=x.device, dtype=x.dtype)

            # Get valid tokens (non-padding)
            if mask is not None:
                valid_tokens = ~mask[b]
                x_b = x[b][valid_tokens]  # (N_valid, D)
                low_ids_b = low_ids_flat[b][valid_tokens]  # (N_valid,)
            else:
                x_b = x[b]  # (N, D)
                low_ids_b = low_ids_flat[b]  # (N,)

            # Only process valid (non-padding) entries
            valid = low_ids_b < npix_out
            low_ids_valid = low_ids_b[valid]
            x_valid = x_b[valid]

            if len(low_ids_valid) > 0:
                out.index_add_(0, low_ids_valid, x_valid)
                counts.index_add_(0, low_ids_valid, torch.ones(len(low_ids_valid), device=x.device, dtype=x.dtype))

            # Average
            active_mask = counts > 0
            out[active_mask] = out[active_mask] / counts[active_mask].unsqueeze(-1)

            # Get active pixels
            active = torch.where(active_mask)[0]
            if len(active) > 0:
                pooled_list.append(out[active])
                pixel_ids_out_list.append(active)
            else:
                # Fallback: keep first token
                pooled_list.append(out[:1])
                pixel_ids_out_list.append(torch.zeros(1, dtype=torch.long, device=x.device))

        # Pad to same length for batching
        max_len = max(p.shape[0] for p in pooled_list)
        pooled_padded = torch.zeros(B, max_len, D, device=x.device, dtype=x.dtype)
        # Use -1 for padding pixel ids (distinguishes from valid pixel 0)
        pixel_ids_padded = torch.full((B, max_len), -1, dtype=torch.long, device=x.device)

        for b, (p, pid) in enumerate(zip(pooled_list, pixel_ids_out_list)):
            n = p.shape[0]
            pooled_padded[b, :n] = p
            pixel_ids_padded[b, :n] = pid

        return pooled_padded, pixel_ids_padded

    def _healpix_pool_dense(
        self,
        x: torch.Tensor,
        pixel_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pool high-resolution HEALPix patches to fixed dense low-resolution grid.

        This is the optimized version that returns fixed-shape tensors suitable
        for SDPA-compatible attention. Unlike _healpix_pool, this does not
        compact active tokens - it returns a dense (B, npix_out, D) tensor.

        Args:
            x: (B, N_cd, D) patch embeddings
            pixel_ids: (B, N_cd) pixel IDs (in high-res nside)
            mask: (B, N_cd) bool mask, True = padding token to ignore

        Returns:
            pooled: (B, npix_out, D) pooled embeddings (dense fixed grid)
            pixel_ids_out: (B, npix_out) low-res pixel IDs, padding = -1
            cd_mask_out: (B, npix_out) bool mask, True = padding (no hits)
        """
        B, N, D = x.shape
        npix_out = int(self.npix_out.item())

        if self.high_to_low is None:
            # Fallback: mean pooling to target_tokens
            out = x.mean(dim=1, keepdim=True).expand(B, self.target_tokens, D)
            pixel_ids_out = torch.arange(self.target_tokens, device=x.device).view(1, -1).expand(B, -1)
            cd_mask = torch.zeros(B, self.target_tokens, dtype=torch.bool, device=x.device)
            return out, pixel_ids_out, cd_mask

        # Map high-res pixel IDs to low-res
        pixel_ids_flat = pixel_ids.view(-1).clamp(0, len(self.high_to_low) - 1)
        low_ids = self.high_to_low[pixel_ids_flat].view(B, N)

        # Create batch offsets for flattened indexing
        # Each batch item's low-res pixels are at offset: batch_idx * npix_out
        batch_offsets = (torch.arange(B, device=x.device) * npix_out).view(B, 1)
        bins = low_ids + batch_offsets  # (B, N)

        # Valid tokens mask
        if mask is None:
            valid = torch.ones(B, N, dtype=torch.bool, device=x.device)
        else:
            valid = ~mask

        bins_valid = bins[valid]  # (N_valid,)
        x_valid = x[valid]        # (N_valid, D)

        # Flattened output buffers: (B * npix_out, D)
        out = torch.zeros(B * npix_out, D, device=x.device, dtype=x.dtype)
        counts = torch.zeros(B * npix_out, device=x.device, dtype=x.dtype)

        # Scatter add: accumulate features for each low-res pixel
        out.index_add_(0, bins_valid, x_valid)
        counts.index_add_(0, bins_valid, torch.ones_like(bins_valid, dtype=x.dtype))

        # Average by count (avoid div by zero)
        nonzero = counts > 0
        safe_counts = counts.clamp(min=1.0)
        out = out / safe_counts.unsqueeze(-1)

        # Reshape to (B, npix_out, D)
        out = out.view(B, npix_out, D)
        counts = counts.view(B, npix_out)

        # Create mask: True where no hits were pooled (count == 0)
        cd_mask_out = ~nonzero.view(B, npix_out)

        # Pixel IDs for the fixed grid
        pixel_ids_out = torch.arange(npix_out, device=x.device).view(1, npix_out).expand(B, -1)
        # Masked positions get -1
        pixel_ids_out = pixel_ids_out.masked_fill(cd_mask_out, -1)

        return out, pixel_ids_out, cd_mask_out

    def _healpix_pool_dense_stageb(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage B: Dense fixed-grid pooling without pixel_ids.

        Args:
            x: (B, npix_in, D) patch embeddings on fixed high-res HEALPix grid
            mask: (B, npix_in) bool mask, True = inactive/no hits

        Returns:
            pooled: (B, npix_out, D) pooled embeddings on fixed low-res grid
            out_mask: (B, npix_out) bool mask, True = inactive
        """
        B, N, D = x.shape
        npix_out = int(self.npix_out.item())

        if self.high_to_low is None:
            # Fallback: mean pooling to target_tokens
            out = x.mean(dim=1, keepdim=True).expand(B, self.target_tokens, D)
            out_mask = torch.zeros(B, self.target_tokens, dtype=torch.bool, device=x.device)
            return out, out_mask

        # Stage B: fixed grid, high_to_low maps from high-res pixel index to low-res
        # high_to_low is (npix_in,), we use it directly
        high_to_low = self.high_to_low.view(1, N).expand(B, -1)  # (B, N)
        batch_offsets = (torch.arange(B, device=x.device) * npix_out).view(B, 1)
        bins = high_to_low + batch_offsets

        # Valid tokens mask
        if mask is None:
            valid = torch.ones(B, N, dtype=torch.bool, device=x.device)
        else:
            valid = ~mask

        bins_valid = bins[valid]
        x_valid = x[valid]

        # Flattened output buffers
        out = torch.zeros(B * npix_out, D, device=x.device, dtype=x.dtype)
        counts = torch.zeros(B * npix_out, device=x.device, dtype=x.dtype)

        # Scatter add
        out.index_add_(0, bins_valid, x_valid)
        counts.index_add_(0, bins_valid, torch.ones_like(bins_valid, dtype=x.dtype))

        # Average
        nonzero = counts > 0
        out = out / counts.clamp(min=1.0).unsqueeze(-1)

        # Reshape
        out = out.view(B, npix_out, D)
        out_mask = ~nonzero.view(B, npix_out)

        return out, out_mask

    def _attention_pool(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Learnable attention-based pooling."""
        B = x.shape[0]
        queries = self.pool_queries.expand(B, -1, -1)  # (B, target_tokens, D)

        # Cross-attention: queries attend to x
        pooled, _ = self.pool_attn(queries, x, x, key_padding_mask=mask)
        return pooled

    def _mean_pool(self, x: torch.Tensor, target_n: int) -> torch.Tensor:
        """Simple mean pooling to fixed number of tokens."""
        B, N, D = x.shape
        if N <= target_n:
            # Pad to target_n
            padding = torch.zeros(B, target_n - N, D, device=x.device, dtype=x.dtype)
            return torch.cat([x, padding], dim=1)

        # Split into groups and pool
        group_size = N // target_n
        x_trimmed = x[:, :group_size * target_n, :]
        x_reshaped = x_trimmed.view(B, target_n, group_size, D)
        return self.pool_proj(x_reshaped.mean(dim=2))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage B: Fixed dense grid pooling.

        Args:
            x: (B, npix_in, D) CD patch embeddings on fixed HEALPix grid
            mask: (B, npix_in) bool mask, True = inactive/no hits

        Returns:
            pooled: (B, npix_out, D) compressed embeddings on fixed low-res grid
            out_mask: (B, npix_out) bool mask, True = inactive
        """
        if self.method == 'healpix_pool':
            # Fast path: skip compression when nside_in == nside_out
            if self.nside_in == self.nside_out:
                out_mask = mask.clone() if mask is not None else torch.zeros(
                    x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
                return x, out_mask
            # Stage B: Dense fixed-grid pooling (no pixel_ids needed)
            pooled, out_mask = self._healpix_pool_dense_stageb(x, mask=mask)
            return pooled, out_mask
        elif self.method == 'attention_pool':
            pooled = self._attention_pool(x, mask)
            # All positions valid for attention_pool
            out_mask = torch.zeros(pooled.shape[0], pooled.shape[1],
                                   dtype=torch.bool, device=pooled.device)
            return pooled, out_mask
        elif self.method == 'mean_pool':
            pooled = self._mean_pool(x, self.target_tokens)
            out_mask = torch.zeros(pooled.shape[0], pooled.shape[1],
                                   dtype=torch.bool, device=pooled.device)
            return pooled, out_mask
        else:
            raise ValueError(f"Unknown compression method: {self.method}")


def build_healpix_knn_adjacency(nside: int, k: int = 16) -> torch.Tensor:
    """
    Build kNN adjacency table for HEALPix pixels based on angular distance.

    Args:
        nside: HEALPix nside parameter
        k: number of nearest neighbors

    Returns:
        (npix, k) tensor of neighbor indices
    """
    try:
        import healpy as hp
    except ImportError:
        raise ImportError("healpy is required for kNN adjacency")

    npix = hp.nside2npix(nside)

    # Get pixel center vectors
    vecs = np.array(hp.pix2vec(nside, np.arange(npix), nest=False)).T  # (npix, 3)

    # Compute pairwise angular distances
    adj = np.zeros((npix, k), dtype=np.int64)

    for i in range(npix):
        # Dot products with all pixels
        dots = vecs @ vecs[i]
        dots = np.clip(dots, -1.0, 1.0)
        ang_dist = np.arccos(dots)

        # Get k+1 nearest (excluding self)
        nearest = np.argpartition(ang_dist, k + 1)[:k + 1]
        nearest = nearest[nearest != i][:k]

        if len(nearest) < k:
            nearest = np.pad(nearest, (0, k - len(nearest)), constant_values=i)

        adj[i] = nearest[:k]

    return torch.from_numpy(adj)
