# suggestion.md

> Purpose: a lightweight reference note for Claude Code.
> 
> Tone: suggestions only, not mandatory instructions.
> 
> Scope: current `HT-Transformer-DS` data pipeline and training hot path.

## Overall impression

The recent redesign is a meaningful improvement in structure:

- preprocessing now writes split-level artifacts directly (`train.h5`, `val.h5`, `test.h5`)
- redundant fields have been removed from disk (`cd_mask`, `cd_unit_vecs`, `cd_times_mean`)
- shared CD geometry has been moved to `cd_unit_vecs.npy`
- loader responsibilities are much cleaner than before

So the current issue does **not** look like “the redesign failed”.
A more likely interpretation is:

- the old complexity/problematic cache logic was reduced successfully
- but the **largest remaining bottlenecks are now elsewhere in the training hot path**
- therefore throughput can stay close to the previous value even though the data design is cleaner

---

## Likely remaining bottlenecks

### 1. CPU-side dtype conversion in `__getitem__`

A likely cost is that data is stored as `float16` on disk, but is converted to `float32` very early during loading.

Why this may matter:

- disk footprint is reduced, but CPU memory traffic goes back up immediately
- every sample pays this conversion cost
- if training already uses mixed precision / bf16, early CPU-side upcasting may not bring much benefit

Possible direction:

- keep `cd_stats`, `cd_time_bins`, `wp_tokens`, `labels` in `float16` on CPU longer
- delay precision conversion until GPU side / autocast side if numerically acceptable

---

### 2. Heavy batch assembly in `collate_fn`

The current collate path still appears to do substantial per-batch work:

- allocate large tensors for padded WP inputs
- allocate dense CD tensors per batch
- copy sample-by-sample into batch storage
- expand/clone shared CD geometry to batch shape

Why this may matter:

- even if disk IO improves, CPU-side batch construction can still dominate
- this kind of work scales with batch size and can hide the benefit of a simpler preprocess design

Possible direction:

- make collate thinner where possible
- avoid cloning batch copies of constant CD geometry unless truly required
- check whether some fields can remain shared or be materialized inside the model instead of in collate

---

### 3. Recomputing constant CD geometry / position encoding every step

The CD geometry is globally fixed.
If the model still recomputes fixed geometry-dependent tensors every step, this can become a steady training tax.

Possible examples to review:

- repeated expansion of `cd_unit_vecs` from shared metadata into `[B, npix, 3]`
- repeated absolute position encoding on the same CD geometry every step

Possible direction:

- consider precomputing fixed CD geometry encodings once
- store them as model buffers if appropriate
- only broadcast views at runtime when needed, instead of cloning full batch copies

---

### 4. Global random access over a single `train.h5`

The current design is much cleaner than before, but training may still be reading one large `train.h5` with globally random event access.

Why this may matter:

- HDF5 can still lose locality under fully random training order
- simplification from “one total file” to “one split file” is helpful, but it is not yet the same as locality-friendly shard/block reading

Possible direction:

- consider block-wise or shard-wise sampling
- preserve SGD randomness at a coarse level, but keep local sequentiality inside blocks
- this may help more than further micro-optimizing HDF5 cache settings

---

### 5. Host-to-device transfer and per-step Python overhead

Even when not the main bottleneck, these can still matter after larger issues have been reduced.

Possible areas to review:

- `.to(device)` calls without `non_blocking=True`
- very frequent progress bar / postfix updates
- repeated Python dictionary transformations per step

Possible direction:

- use non-blocking transfer if the memory pipeline supports it
- reduce logging/progress update frequency
- profile whether Python-side step bookkeeping is non-negligible

---

## Suggested priority order

This is a **suggested** order only.

### Priority A: check the training hot path before redesigning preprocess again

The preprocess redesign already looks substantially cleaner.
It may be more useful now to profile:

- `__getitem__`
- `collate_fn`
- host-to-device transfer
- model forward fixed-cost components

rather than immediately redesign preprocessing yet again.

---

### Priority B: try removing early CPU `.float()` conversions

A relatively low-risk experiment:

- keep loaded tensors in `float16` on CPU
- compare throughput and stability
- only keep exceptions in `float32` if a specific tensor truly needs it

This is one of the most promising “small change, potentially visible gain” experiments.

---

### Priority C: move constant CD geometry work out of the per-step path

This looks like a strong candidate for wasted repeated work.

If feasible:

- precompute fixed CD positional encodings once
- keep them as buffers
- avoid cloning full `[B, npix, ...]` geometry tensors in collate

This may improve throughput even if dataload itself is no longer the dominant issue.

---

### Priority D: consider block-wise or shard-wise train sampling

If throughput is still limited after simplifying CPU hot-path work, the next structural optimization to consider may be:

- split `train.h5` into multiple shards, or
- keep one file but sample in block-wise order

This is not necessarily required immediately, but it is a plausible next step if random HDF5 access remains expensive.

---

## What may *not* be worth over-focusing on right now

Based on the current state, it may be less useful to spend too much time on:

- adding more complexity back into preprocess caching
- building a more elaborate manifest/index layer again
- repeatedly tuning HDF5 cache parameters without first thinning the runtime hot path

The recent simplification appears directionally correct.
The next gains may come more from **runtime cost removal** than from **more preprocessing machinery**.

---

## Practical profiling suggestions

To decide what matters most, it may help to measure one training step as separate segments:

1. dataloader wait time
2. collate time
3. host-to-device copy time
4. forward time
5. backward + optimizer time

If possible, compare a few short runs under:

- current code
- CPU-side float16 retained longer
- CD geometry/precomputed PE optimization
- reduced progress-bar update frequency
- block-wise sampling

That should make it much clearer whether the current 1.5 it/s is still data-limited, or already mostly compute / host-assembly limited.

---

## Short summary for Claude Code

A reasonable interpretation of the current codebase is:

- the new preprocess/load redesign is **structurally better**
- lack of speedup does **not** necessarily mean the redesign was wrong
- the main remaining bottlenecks may now be in:
  - CPU dtype conversion
  - heavy collate work
  - repeated fixed-geometry computation
  - random access pattern over `train.h5`
  - host-side per-step overhead

So the next optimization pass could focus more on **training hot-path simplification** than on another large preprocess redesign.
