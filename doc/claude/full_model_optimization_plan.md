# Full Model Optimization Plan for HT-Transformer-DS

## Purpose

This document is a **Claude Code executable implementation plan** for the next-stage performance optimization of the current `HT-Transformer-DS` model.

It targets the **remaining true runtime pain points** after the recent DeepSphere and training-speed fixes, with the goal of **fully resolving the current model-side bottlenecks**, not just partially reducing them.

This plan is intentionally detailed and professional. It includes:

- optimization objectives
- design principles
- staged implementation strategy
- per-file refactor tasks
- code-level reference implementations
- migration notes
- validation criteria
- explicit non-goals / forbidden shortcuts

---

# 1. Executive Summary

The current model has already fixed the worst previous bottlenecks:

- CD local graph construction is no longer rebuilt inside every DeepSphere block
- neighbor aggregation has been vectorized
- WP time bias is only applied to the first `k` layers
- CD compression now uses `-1` as padding sentinel

These were important and effective.

However, **two major pain points still remain**:

## 1.1 WP branch still constructs explicit pairwise time bias tensors

The current `SignedTimeBucketBias` still materializes a full:

- `(B, N_wp, N_wp)` time-difference tensor
- then `(B, H, N_wp, N_wp)` bias tensor

As long as this path exists, the corresponding WP self-attention layers cannot use the fully optimized SDPA fast path.

## 1.2 CD branch is still based on variable-length active HEALPix patches

Even after local-graph reuse optimization, the current CD representation still requires:

- runtime active patch indexing
- variable-length token handling
- dynamic masking
- compression logic that depends on active token compaction

This continues to add runtime complexity and makes the entire pipeline less compile-friendly and less kernel-friendly.

---

# 2. Final Optimization Target

The final target architecture is:

## WP branch
- Remove explicit pairwise time-bias matrices from the training main path
- Replace pairwise time bias with **token-level time encoding**
- Make **all WP self-attention layers SDPA-compatible**

## CD branch
- Replace variable-length active patch representation with **fixed dense HEALPix grid representation**
- Remove runtime local graph construction entirely
- Run DeepSphere directly on a fixed graph
- Make compression a fixed-grid pooling operation

This is the cleanest path to:

- fully eliminate remaining model-side runtime pain points
- maximize SDPA usage
- make the model much more friendly to `torch.compile`
- simplify masking and graph logic
- make performance more predictable and stable

---

# 3. Design Principles

Claude Code must follow these principles:

## 3.1 Do not change the task definition
Keep unchanged:
- ordered dual-endpoint regression
- `pred_u1`, `pred_u2`
- existing endpoint loss interface
- current output semantics

## 3.2 Preserve the overall hybrid architecture
Keep unchanged at high level:
- WP branch
- CD branch
- global tokens
- query tokens
- hybrid fusion structure
- endpoint heads

This is an optimization refactor, not a brand-new model.

## 3.3 Performance-first implementation
Any new implementation must prefer:
- fixed-shape tensor operations
- batch-wise vectorization
- avoiding Python loops in forward paths
- avoiding runtime dictionary construction
- avoiding explicit large pairwise tensors where possible

## 3.4 Prefer compile-friendly code
New code should be as compatible as possible with:
- `torch.compile`
- SDPA
- fused kernels
- static or semi-static tensor shapes

---

# 4. Staged Implementation Strategy

Implement in **two stages**.

## Stage A — Immediate model-path optimization

Goal:
- remove explicit WP pairwise time bias
- remove dynamic active-token compaction inside CD compression
- preserve current data pipeline as much as possible

This stage should already deliver a significant performance improvement with moderate code changes.

## Stage B — Full CD representation refactor

Goal:
- replace current variable-length active-patch CD representation with fixed dense HEALPix grid
- remove runtime local graph construction completely
- make DeepSphere and compression operate on fixed graph tensors

This is the true endgame for solving the remaining CD pain points.

---

# 5. File-by-File Refactor Plan

## 5.1 `models/components/wp_time_bias.py`

### Current problem
Current implementation creates explicit pairwise time-difference matrices and bias tensors.

This is expensive and prevents full SDPA usage.

### Required action
This file should no longer be used in the main training path for WP self-attention.

Two acceptable options:

### Option A (recommended)
Keep the file for legacy / ablation only, but remove it from the default forward path.

### Option B
Deprecate the file and clearly mark it as optional / research-only.

### Required edits
- Add a module-level docstring note:
  - this module is deprecated for main training
  - token-level time encoding is now preferred
- Add a warning in comments that explicit `(B,H,N,N)` bias is expensive
- Keep class definitions for compatibility / ablation

### Do not
- delete it immediately if older configs or ablation scripts may still import it
- keep it in the main path by default

## 5.2 Create new file: `models/components/wp_time_encoding.py`

This is a **new required file**.

### Purpose
Provide token-level time encoding for WP tokens to replace pairwise time bias.

### Required interface
```python
class WPTimeEncoding(nn.Module):
    def __init__(self, d_model: int, hidden: int = 32, fourier_dim: int = 16):
        ...
    def forward(self, wp_times: torch.Tensor) -> torch.Tensor:
        # wp_times: (B, N_wp)
        # return: (B, N_wp, d_model)
```

### Recommended implementation
Use a lightweight Fourier-time encoding + small MLP.

### Reference code
```python
import torch
import torch.nn as nn


class WPTimeEncoding(nn.Module):
    """
    Token-level time encoding for WP hits.

    Replaces explicit pairwise time bias by encoding normalized hit time
    directly into token embeddings.

    Input:
        wp_times: (B, N_wp), normalized to [0, 1]

    Output:
        time_emb: (B, N_wp, d_model)
    """

    def __init__(self, d_model: int, hidden: int = 32, fourier_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.fourier_dim = fourier_dim

        freqs = torch.randn(1, 1, fourier_dim) * 4.0
        self.register_buffer("freqs", freqs)

        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * fourier_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, wp_times: torch.Tensor) -> torch.Tensor:
        t = wp_times.unsqueeze(-1)                     # (B, N, 1)
        proj = t * self.freqs                         # (B, N, F)
        feat = torch.cat([t, torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.mlp(feat)
```

### Validation
- shape correctness
- no pairwise tensor construction
- works on GPU and CPU
- output dtype matches model path expectations

## 5.3 `models/ht_transformer.py`

This file requires the most important refactor in Stage A.

### Current problems
1. WP time bias is still explicitly computed and passed to attention
2. `HybridFusionLayer` still accepts `wp_time_bias`
3. `MultiHeadAttention` still has a slow manual path when `time_bias is not None`
4. CD masks and concatenation masks are recomputed inside every layer
5. CD branch still relies on variable-length compressed tokens

### Required Stage A changes

### A. Remove pairwise time bias from the main path

#### Required edits
1. Remove import of `SignedTimeBucketBias` from default path
2. Import new `WPTimeEncoding`
3. In `HTTransformer.__init__()`:
   - create `self.wp_time_encoding`
   - remove `self.wp_time_bias`
   - remove `self.wp_time_bias_layers`

#### Reference code
```python
from models.components.wp_time_encoding import WPTimeEncoding

self.wp_time_encoding = WPTimeEncoding(
    d_model=self.d_model,
    hidden=model_cfg.get("wp_time_hidden", 32),
    fourier_dim=model_cfg.get("wp_time_fourier_dim", 16),
)
```

#### Forward-path change
```python
wp_emb = self.wp_projector(batch["wp_tokens"])
wp_emb = wp_emb + self.wp_time_encoding(batch["wp_times"])
```

Then remove:
- pairwise `wp_time_bias` computation
- `layer_wp_time_bias`
- all passing of `wp_time_bias` into encoder layers

### B. Simplify `MultiHeadAttention`

Once pairwise bias is gone from the main path, simplify attention.

#### Required edits
- remove `time_bias` argument from the default attention path
- keep only SDPA-based path in the main implementation
- if legacy support is necessary, make a separate legacy attention class instead of one overloaded class

#### Recommended new interface
```python
def forward(self, query, key, value, mask=None):
    ...
```

#### Reference code
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, Sq, _ = query.shape
        _, Sk, _ = key.shape

        q = self.wq(query).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(key).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(value).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_mask = torch.zeros_like(mask, dtype=q.dtype)
            attn_mask.masked_fill_(mask, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, Sq, self.d_model)
        return self.wo(out)
```

### C. Precompute masks outside the layer loop

#### Why
Current mask construction is repeated inside each layer.

#### Required edits
Build mask packs once in `HTTransformer.forward()` before the layer loop, and pass them into `HybridFusionLayer`.

#### Example
```python
mask_pack = {
    "wp_mask": batch["wp_mask"],
    "cd_mask": cd_mask,
}
```

If possible, also precompute:
- `wp_attn_mask`
- `wp_cross_mask`
- `cd_cross_mask`

### D. Stage A change for CD compression: use fixed `npix_out` output instead of active-token compaction

Instead of:
- collecting active pooled tokens
- creating variable-length lists
- padding afterward

return dense low-resolution HEALPix outputs with fixed shape `(B, npix_out, D)`.

Then `cd_mask` is just derived from counts.

### E. Optional but strongly recommended: introduce fused self-attention for self-attn modules

This is not mandatory for correctness, but recommended.

At minimum, consider replacing self-attention paths with fused QKV projection.

## 5.4 `models/components/deepsphere.py`

This file has two separate tasks:

- Stage A: improve current compression implementation
- Stage B: completely remove runtime local-graph construction by switching to dense HEALPix grids

### Stage A task: replace variable-length active pooling with fixed dense low-resolution pooling

#### Current problem
`CDCompression._healpix_pool()` still:
- loops over batch items
- collects active pooled pixels
- pads variable-length outputs

This remains a runtime pain point.

#### Required Stage A refactor
Make `_healpix_pool()` return:
- `pooled: (B, npix_out, D)`
- `pixel_ids_out: (B, npix_out)` fixed grid ids
- `cd_mask_out: (B, npix_out)` derived from counts

Do not compact active tokens.

#### Reference code
```python
def _healpix_pool_dense(self, x, pixel_ids, mask=None):
    B, N, D = x.shape
    npix_out = int(self.npix_out.item())

    if mask is None:
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)

    pixel_ids_flat = pixel_ids.view(-1).clamp(0, len(self.high_to_low) - 1)
    low_ids = self.high_to_low[pixel_ids_flat].view(B, N)

    batch_offsets = (torch.arange(B, device=x.device) * npix_out).view(B, 1)
    bins = low_ids + batch_offsets

    valid = ~mask
    bins_valid = bins[valid]
    x_valid = x[valid]

    out = torch.zeros(B * npix_out, D, device=x.device, dtype=x.dtype)
    counts = torch.zeros(B * npix_out, device=x.device, dtype=x.dtype)

    out.index_add_(0, bins_valid, x_valid)
    counts.index_add_(0, bins_valid, torch.ones_like(bins_valid, dtype=x.dtype))

    nonzero = counts > 0
    safe_counts = counts.clamp(min=1.0)
    out = out / safe_counts.unsqueeze(-1)

    out = out.view(B, npix_out, D)
    counts = counts.view(B, npix_out)
    out_mask = ~nonzero.view(B, npix_out)

    pixel_ids_out = torch.arange(npix_out, device=x.device).view(1, npix_out).expand(B, -1)
    pixel_ids_out = pixel_ids_out.masked_fill(out_mask, -1)
    return out, pixel_ids_out, out_mask
```

Then adapt `forward()` accordingly.

### Stage B task: remove runtime local graph construction completely

#### Current problem
Even in the optimized version, there is still a runtime local graph builder based on `pixel_ids`.

This is fundamentally unnecessary if CD is represented as a fixed HEALPix dense tensor.

#### Required final design
DeepSphere should run on:
- fixed `npix_in`
- fixed `full_knn_adj`
- fixed-shape tensors

No per-batch `build_local_neighbor_graph`.

#### Final expected encoder interface
```python
def forward(self, x, full_knn_adj, cd_mask=None):
    ...
```

No `pixel_ids` are required in the final dense-grid version.

#### Final neighbor aggregation assumption
The graph is fixed and global:
- node index = HEALPix pixel id
- all events share the same graph topology

This is the key architectural simplification.

#### Stage B reference direction

If input `x` is `(B, npix, D)` and `full_knn_adj` is `(npix, k)`, then:
- local neighbor indices are identical for all events
- they can be pre-expanded once
- no dictionary or per-event remapping is needed

Example pattern:
```python
neighbor_indices = full_knn_adj.unsqueeze(0).expand(B, -1, -1)
```

Then use vectorized gather.

## 5.5 Data pipeline changes required for Stage B

This will affect data/preprocess code, even though the current request is model-focused.

Claude Code must understand that **full pain-point resolution is impossible without changing the CD representation upstream**.

### Required data-side change
Instead of storing only active CD patches, preprocessing must produce dense HEALPix tensors:

- `cd_unit_vecs_dense: (npix, 3)` fixed per event or globally referenced
- `cd_stats_dense: (npix, 4)`
- `cd_time_bins_dense: (npix, B_bins)`
- `cd_mask_dense: (npix,)`

### Benefits
- no runtime local graph construction
- no variable-length CD tokens
- much better compile friendliness
- easier batching
- easier masking
- simpler compression

## 5.6 `models/components/token_projectors.py`

This file does not need major architectural change for performance pain-point removal.

### Required action
Only minor updates:

1. Ensure `CDProjector` cleanly supports dense fixed-grid CD inputs
2. Keep interfaces consistent
3. Optionally add comments clarifying that dense-grid inputs are now expected in Stage B

### Optional improvement
If profiling later shows projector cost is non-negligible, you may:
- reduce Conv1d channels from 16 to 8
- or replace Conv1d with lighter time-bin MLP

But this is **not a first-priority change**.

## 5.7 `models/components/position_encoding.py`

No urgent action required.

### Recommended action
- keep as-is
- if this file contains legacy modules not used by the new main path, mark them clearly as legacy or ablation-only
- do not spend effort here before fixing the major bottlenecks

## 5.8 `models/losses/endpoint_loss.py`

No performance bottleneck here.

### Required action
- leave unchanged
- only ensure interfaces remain consistent after model refactor

# 6. Required Config Migration

After Stage A / Stage B changes, the config should be updated.

## Remove / deprecate
```yaml
model:
  wp_time_bias: "signed_bucket"
  wp_num_time_buckets: 64
  wp_time_bias_heads_shared: true
  wp_time_bias_layers: 1
```

## Add
```yaml
model:
  wp_time_encoding: true
  wp_time_hidden: 32
  wp_time_fourier_dim: 16
```

For Stage B, add explicit note that CD uses dense HEALPix grid representation.

Example:
```yaml
data:
  cd_representation: "dense_healpix"
```

# 7. Validation Plan

Claude Code must not stop after code compiles.
It must validate performance and correctness.

## 7.1 Functional validation
- training forward pass works
- backward pass works
- loss decreases normally
- output shapes unchanged:
  - `pred_u1: (B, 3)`
  - `pred_u2: (B, 3)`

## 7.2 Performance validation
Compare before vs after on identical hardware and config:
- average train step time
- epoch time
- GPU utilization
- memory usage

### Stage A expected result
- noticeable speedup from removing explicit pairwise WP time bias
- noticeable speedup from fixed-output CD compression

### Stage B expected result
- larger speedup from removing runtime local graph construction entirely
- much more stable step time
- better compile friendliness

## 7.3 Profiling validation
Use PyTorch profiler or timing logs to compare:
- `wp_time_bias` path before/after
- `DeepSphereEncoder` runtime before/after
- `CDCompression` runtime before/after

# 8. Final Acceptance Criteria

The optimization task is considered complete only if:

## Stage A complete
- WP main path no longer builds `(B,H,N,N)` time bias
- all WP self-attn layers use SDPA-compatible attention
- CD compression no longer performs active-token compaction
- training speed improves measurably

## Stage B complete
- CD is represented as dense fixed HEALPix tensors
- no runtime local graph construction remains
- DeepSphere runs on fixed global graph
- compression runs on fixed-grid pooling
- step time is significantly more stable and lower than current version

# 9. Explicit Non-Goals / Forbidden Shortcuts

Claude Code must not do any of the following:

1. Do not change endpoint semantics
2. Do not remove CD branch or WP branch
3. Do not replace the whole architecture with an unrelated model
4. Do not keep pairwise time bias in the main path and claim the pain point is solved
5. Do not keep variable-length active CD representation and claim CD pain point is fully solved
6. Do not rely on Python dictionaries or Python loops in model forward paths if a tensorized alternative exists
7. Do not optimize only trainer/runtime while leaving the core model bottlenecks untouched

# 10. Recommended Execution Order for Claude Code

1. Add `WPTimeEncoding`
2. Remove pairwise WP time bias from the main path
3. Simplify `MultiHeadAttention` to SDPA-first path
4. Change CD compression to fixed-output low-res grid
5. Validate speedup
6. Refactor preprocessing/data pipeline to emit dense HEALPix grids
7. Remove runtime local graph construction
8. Validate speedup again
9. Update configs and comments
10. Document all migration changes

# 11. Short Final Instruction

Claude Code should treat this as a **performance-critical refactor**, not a cosmetic cleanup.

The correct mindset is:

- reduce runtime dynamic behavior
- reduce explicit pairwise tensor construction
- maximize SDPA compatibility
- move graph structure from runtime to fixed topology where possible

That is the path to fully solving the current pain points.
