# Reduction Loop Peeling for TorchInductor Triton Codegen

**Issue:** https://github.com/pytorch/pytorch/issues/148402
**Status:** Design
**Author:** Liqiang Lu

## 1. Problem

For non-persistent reductions on `[xnumel, rnumel]` tensors, Inductor generates a single reduction loop where every iteration carries a mask:

```python
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    tmp0 = tl.load(in_ptr0 + (r0_1 + 50304*x0), r0_mask, other=0.0).to(tl.float32)
    _tmp3_max = tl.where(r0_mask, _tmp3_max_next, _tmp3_max)
    _tmp3_sum = tl.where(r0_mask, _tmp3_sum_next, _tmp3_sum)
```

Two performance problems:
1. **No vectorization:** The non-constant `r0_mask` on `tl.load` prevents Triton from emitting vectorized loads (`ld.global.v4`).
2. **Redundant `tl.where`:** For all iterations except the last (tail) block, `r0_mask` is all-True, so the `tl.where` is wasted work.

**Measured impact:** 1.5x slowdown on log_softmax kernel (3.01ms -> 2.08ms with fix). See the [gist benchmark](https://gist.github.com/shunting314/2fb1f5381b62b363d1046a2e05741e7b).

## 2. Solution: Loop Peeling

Split the reduction loop into two:

```python
r0_numel_aligned = (r0_numel // R0_BLOCK) * R0_BLOCK

# Main loop: all R0_BLOCK elements valid, no mask needed
for r0_offset in tl.range(0, r0_numel_aligned, R0_BLOCK):
    r0_index = r0_offset + r0_base
    # No r0_mask
    tmp0 = tl.load(in_ptr0 + (r0_1 + 50304*x0), None).to(tl.float32)  # vectorizable!
    _tmp3_max = _tmp3_max_next   # no tl.where needed
    _tmp3_sum = _tmp3_sum_next

# Tail loop: partial block, needs mask
for r0_offset in tl.range(r0_numel_aligned, r0_numel, R0_BLOCK):
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    tmp0 = tl.load(in_ptr0 + (r0_1 + 50304*x0), r0_mask, other=0.0).to(tl.float32)
    _tmp3_max = tl.where(r0_mask, _tmp3_max_next, _tmp3_max)
    _tmp3_sum = tl.where(r0_mask, _tmp3_sum_next, _tmp3_sum)
```

Both loops update the **same accumulator variables** -- state carries across naturally.

## 3. Interaction with xmask_unswitch

When spatial masks (`xmask`) are also present (2D reductions with dynamic shapes), the main loop body still has `xmask`:

```python
# Main loop after peeling: r0_mask removed, xmask remains
tmp0 = tl.load(ptr, xmask, other=0.0)
_acc = tl.where(xmask, _acc + tmp0, _acc)
```

Now `xmask` is the **sole mask** -- exactly the condition where `xmask_unswitch` applies! So the two optimizations compose:

```python
# Main loop with peeling + xmask_unswitch:
for r0_offset in tl.range(0, r0_numel_aligned, R0_BLOCK):
    r0_index = r0_offset + r0_base
    if xoffset + XBLOCK <= xnumel:
        tmp0 = tl.load(ptr, None)          # fully vectorizable!
        _acc = _acc + tmp0                  # no tl.where at all
    else:
        tmp0 = tl.load(ptr, xmask, other=0.0)
        _acc = tl.where(xmask, _acc + tmp0, _acc)
```

This recovers full vectorization for reduction kernels with dynamic shapes, which neither optimization achieves alone.

## 4. Implementation Design

### 4.1 Files to Modify

| File | Change |
|------|--------|
| `torch/_inductor/config.py` | Add `triton.loop_peeling` config flag |
| `torch/_inductor/codegen/triton.py` | Core: eligibility check, `_force_constant_rmask` context manager, `_has_constant_mask` 2-line addition, `_codegen_peeled_reduction_loop` emission |
| `torch/_inductor/codegen/simd.py` | `_codegen_node_schedule_with_peeling` for dual buffer generation |
| `test/inductor/test_codegen_triton.py` | Tests |

### 4.2 Config Flag

```python
# torch/_inductor/config.py, in class triton:

# Enable loop peeling for non-persistent reductions. Splits the reduction
# loop into a main (unmasked, vectorizable) loop and a tail (masked) loop.
loop_peeling: bool = False
```

Disabled by default until broader validation is complete. Enable with `config.triton.loop_peeling = True` or the `@inductor_config.patch("triton.loop_peeling", True)` decorator in tests.

### 4.3 Eligibility

New method `_should_peel_reduction_loop(self, loop_trees)` on `TritonKernel`:

```python
def _should_peel_reduction_loop(self, loop_trees):
    return (
        config.triton.loop_peeling
        and len(loop_trees) == 1                          # single reduction dim
        and not self.cooperative_reduction                 # not cooperative
        and not self._has_constant_mask(loop_trees[0])    # mask is not already elided
        and not self.pointer_advancements.get(loop_trees[0].symt)  # no block_ptr
    )
```

Conditions explained:
- **Single reduction dim:** Multi-dim reductions (`len(loop_trees) > 1`) have nested loops with interacting masks. Deferred to follow-up.
- **Not cooperative:** Cooperative reduction uses `rsplit_start`/`rsplit_end` bounds, complicating alignment computation. Deferred.
- **Mask not constant:** If `_has_constant_mask` is already True, there's no `r0_mask` to remove -- peeling is unnecessary.
- **No block_ptr:** Block pointer reductions use `tl.advance()` and `boundary_check` instead of `r0_mask`. Peeling those requires modifying `boundary_check` lists. Deferred.

### 4.4 Buffer Generation Strategy

The core challenge: `self.loads`, `self.compute`, `self.stores`, `self.indexing_code` are `IndentedBuffer` objects populated once (during node codegen in `codegen_node_schedule_with_kernel`) before `codegen_body()` is called. They contain baked-in mask expressions like `r0_mask & xmask` and `tl.where(r0_mask & xmask, ...)`. We need an unmasked copy for the main loop.

**Two approaches considered:**

| Approach | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| **A. String replacement** | `IndentedBuffer.map()` post-processes buffers to remove `r0_mask` from baked-in strings | Local to `codegen_body()`, no changes to buffer-filling flow | Fragile: must match all mask patterns; regex rules are coupled to codegen output format |
| **B. Dual generation** | Run node codegen twice — once with `r0_mask` elided via `filter_masks`, once normally — capturing each into separate buffers | Uses the codegen's native mask elision; no string matching at all | Touches `codegen_node_schedule_with_kernel`; must manage CSE state for two passes |

**Chosen: Approach B (dual generation).** The codegen already has a clean mechanism for mask elision: `filter_masks()` discards mask names where `_has_constant_mask()` returns True, causing `indexing()` to emit `None` masks and `reduction()` to emit bare accumulator updates (no `tl.where`). We add a flag to force this for the reduction tree, then run the buffer-filling codegen twice.

### 4.5 Forcing Mask Elision

New context manager on `TritonKernel`:

```python
@contextlib.contextmanager
def _force_constant_rmask(self):
    """Temporarily force filter_masks to discard the reduction mask.

    When active, _has_constant_mask returns True for reduction trees,
    causing indexing() to emit mask=None and reduction() to skip tl.where
    for the reduction mask. Spatial masks (xmask) are unaffected.
    """
    self._force_rmask_constant = True
    try:
        yield
    finally:
        self._force_rmask_constant = False
```

Modify `_has_constant_mask` to check the flag:

```python
def _has_constant_mask(self, tree: IterationRangesRoot) -> bool:
    # Loop peeling: treat reduction mask as constant for main (unmasked) loop
    if getattr(self, '_force_rmask_constant', False) and tree.is_reduction:
        return True

    # ... existing logic unchanged ...
```

This is the **only string-level change** in the codegen: a 2-line early return. All downstream mask elision (in `filter_masks`, `indexing`, `reduction`) follows automatically through existing code paths.

### 4.6 Dual Buffer Capture

New method on `SIMDScheduling` that wraps the existing node codegen loop to produce two sets of buffers:

```python
def _codegen_node_schedule_with_peeling(self, node_schedule, kernel):
    """Run node codegen twice: once without r0_mask (main loop), once with (tail loop)."""

    cse = kernel.cse

    # --- Save all mutable state before the unmasked pass ---
    # The unmasked pass runs node.codegen() which writes to body (accumulator
    # init), post_loop_combine (tl.sum), post_loop_store (tl.store), and
    # mutates CSE caches, _load_counts, outside_loop_vars.  All must be
    # restored so the masked pass produces identical variable names.
    saved_body = kernel.body
    saved_plc = kernel.post_loop_combine
    saved_pls = kernel.post_loop_store
    saved_lc = kernel._load_counts.copy()
    saved_olv = kernel.outside_loop_vars.copy()
    saved_cc = dict(cse._cache)
    saved_rc = dict(cse.reduction_cache)
    saved_sc = dict(cse.store_cache)
    saved_vm = dict(cse.varname_map)
    saved_is = cse.invalidated_stores.copy()

    # --- First pass: unmasked buffers (main loop body) ---
    # Redirect body/post_loop to throwaway buffers so accumulator init
    # and final reduction from the unmasked pass are discarded.
    kernel.body = IndentedBuffer()
    kernel.post_loop_combine = IndentedBuffer()
    kernel.post_loop_store = IndentedBuffer()

    unmasked_bufs = {
        'indexing_code': IndentedBuffer(),
        'loads': IndentedBuffer(),
        'compute': IndentedBuffer(),
        'stores': IndentedBuffer(),
    }
    with kernel._force_constant_rmask():
        with kernel.swap_buffers(unmasked_bufs['loads'],
                                 unmasked_bufs['compute'],
                                 unmasked_bufs['stores']):
            # swap_buffers only redirects loads/compute/stores.
            # indexing_code must be swapped manually.
            # TODO: extend swap_buffers to accept indexing_code.
            old_indexing = kernel.indexing_code
            kernel.indexing_code = unmasked_bufs['indexing_code']
            try:
                # Only process nodes inside the reduction loop.
                # Skip DisableReduction/EnableReduction and any nodes
                # between them (non-reduction nodes like stores).
                inside = True
                for node in node_schedule:
                    if node is DisableReduction:
                        inside = False
                    elif node is EnableReduction:
                        inside = True
                    elif inside:
                        indexing_dtype_strength_reduction(node._body)
                        index_vars = kernel.split_and_set_ranges(node.get_ranges())
                        node.codegen(index_vars)
            finally:
                kernel.indexing_code = old_indexing

    # --- Restore all state ---
    # Reset CSE counter to 0 so the masked pass produces the same
    # variable names (tmp0, tmp1, _tmp2, ...) as the unmasked pass.
    # Both passes must share accumulator names for correctness.
    kernel.body = saved_body
    kernel.post_loop_combine = saved_plc
    kernel.post_loop_store = saved_pls
    kernel._load_counts = saved_lc
    kernel.outside_loop_vars = saved_olv
    cse._cache = saved_cc
    cse.reduction_cache = saved_rc
    cse.store_cache = saved_sc
    cse.varname_map = saved_vm
    cse.invalidated_stores = saved_is
    cse.iter_buffer_ids = itertools.count(0)
    for tree in kernel.range_trees:
        tree.cache_clear()

    # --- Second pass: masked buffers (tail loop body) ---
    # Normal codegen — fills kernel.loads/compute/stores (tail loop body),
    # kernel.body (accumulator init), kernel.post_loop_combine (tl.sum),
    # and kernel.post_loop_store (tl.store).
    stack = contextlib.ExitStack()
    for node in node_schedule:
        if node is DisableReduction:
            stack.enter_context(kernel.disable_reduction())
        elif node is EnableReduction:
            stack.close()
        else:
            indexing_dtype_strength_reduction(node._body)
            index_vars = kernel.split_and_set_ranges(node.get_ranges())
            node.codegen(index_vars)

    # Store unmasked buffers on kernel for codegen_body to use
    kernel._peeled_unmasked_bufs = unmasked_bufs
```

**CSE counter reset:** Both passes must produce the same variable names (`tmp0`, `_tmp2`, etc.) so the unmasked loop body references the same accumulators as the masked pass. We reset `iter_buffer_ids` to `count(0)` before the masked pass. This works because `finalize_indexing` (which runs before both passes) does not allocate `tmpN` variables — the first `tmpN` comes from `node.codegen()`.

**Why save/restore body, post_loop_combine, post_loop_store?** The unmasked pass's `node.codegen()` calls `reduction()`, which writes accumulator init to `self.body`, final reduction to `self.post_loop_combine`, and output store to `self.post_loop_store`. These side effects must be discarded — only the masked (second) pass should produce the real accumulator lifecycle. We redirect them to throwaway buffers during the unmasked pass, then restore the originals.

**Why skip non-reduction nodes in the first pass?** Fused kernels like `sum(x) + 1.0` have a `node_schedule` containing `[reduction_node, DisableReduction, pointwise_node, EnableReduction]`. The pointwise node has `ranges=((N,), ())` — no reduction dimension. Processing it with `inside_reduction=True` would fail in `split_and_set_ranges`. We only need the reduction loop body for the unmasked pass, so we skip nodes outside `DisableReduction`/`EnableReduction` boundaries.

**Why `swap_buffers` for indexing_code too?** `swap_buffers` only redirects `loads`, `compute`, `stores`. `indexing_code` needs manual save/restore (as shown above). An alternative is to add `indexing_code` support to `swap_buffers`, but manual handling keeps the change minimal.

**How `swap_buffers` fills the unmasked buffers:**

`node.codegen()` replays the LoopBody FX graph. Each op dispatches to `TritonKernel` methods that write into `self.loads`, `self.compute`, `self.stores`. `swap_buffers` redirects these to our empty `unmasked_bufs`, so the same codegen machinery fills different targets:

```
1. Create four empty IndentedBuffers (unmasked_bufs)

2. swap_buffers redirects:
     self.loads   -> unmasked_bufs['loads']
     self.compute -> unmasked_bufs['compute']
     self.stores  -> unmasked_bufs['stores']
   Also: self.cse = cse.scoped_copy()  (isolated cache)

3. _force_constant_rmask makes filter_masks() drop r0_mask

4. node.codegen() runs, dispatching through V.ops -> TritonKernel:

     ops.load(...)  ->  TritonKernel.load()
        -> self.indexing() sees no r0_mask (forced constant)
        -> emits "tl.load(ptr, xmask, ...)"  (no r0_mask!)
        -> cse.generate(self.loads, expr)
                        ^^^^^^^^^^
                        this IS unmasked_bufs['loads'] (swapped)
        -> writes: "tmp0 = tl.load(ptr, xmask, ...)" into it

     ops.reduction(...)  ->  TritonKernel.reduction()
        -> filter_masks returns masks={xmask}  (r0_mask dropped!)
        -> cond = "xmask"  (not "r0_mask & xmask")
        -> self.compute.writeline("_tmp2 = tl.where(xmask, tmp3, _tmp2)")
           ^^^^^^^^^^^^
           this IS unmasked_bufs['compute'] (swapped)

5. swap_buffers exits:
     self.loads/compute/stores restored to kernel's originals
     self.cse restored to parent CSE

6. unmasked_bufs now contain the mask-elided loop body code
```

Without `swap_buffers`, `node.codegen()` would write into the kernel's real buffers, corrupting them. The scoped CSE is also critical — without it the unmasked pass would populate the CSE cache, and the subsequent masked pass would hit stale entries and skip code generation.

### 4.7 Two-Loop Emission

New method `_codegen_peeled_reduction_loop(self, loop_trees)` on `TritonKernel`:

```python
def _codegen_peeled_reduction_loop(self, loop_trees):
    tree = loop_trees[0]
    prefix = tree.prefix                    # "r0"
    unmasked = self._peeled_unmasked_bufs   # set by _codegen_node_schedule_with_peeling

    numel_var = f"{prefix}numel"            # "r0numel"
    block_var = f"{prefix.upper()}BLOCK"    # "R0BLOCK"
    aligned_var = f"{prefix}numel_aligned"

    # num_stages for HIP
    if torch.version.hip and get_triton_version() > (3, 2):
        num_stages = ", num_stages = 2"
    else:
        num_stages = ""

    # Emit aligned numel computation
    self.body.writeline(f"{aligned_var} = ({numel_var} // {block_var}) * {block_var}")

    # --- Main loop (unmasked) ---
    self.body.writeline(
        f"for {prefix}offset in tl.range(0, {aligned_var}, {block_var}{num_stages}):"
    )
    with self.body.indent():
        # Header: r0_index only, NO r0_mask line
        self.body.writeline(f"{tree.name} = {prefix}offset + {prefix}base")
        self.codegen_reduction_indices(self.body)
        self.body.splice(unmasked['indexing_code'])
        self.body.splice(unmasked['loads'])
        self.body.splice(unmasked['compute'])
        self.body.splice(unmasked['stores'])

    self.cse.invalidate(self.outside_loop_vars)
    tree.cache_clear()

    # --- Tail loop (original masked) ---
    self.body.writeline(
        f"for {prefix}offset in tl.range({aligned_var}, {numel_var}, {block_var}{num_stages}):"
    )
    with self.body.indent():
        self.iteration_ranges_codegen_header(tree, self.body)  # emits r0_index + r0_mask
        self.codegen_reduction_indices(self.body)
        self.body.splice(self.indexing_code)
        self.body.splice(self.loads)
        self.body.splice(self.compute)
        self.body.splice(self.stores)

    self.cse.invalidate(self.outside_loop_vars)
    tree.cache_clear()
```

**Why no string replacement is needed:**
- `unmasked['loads']` was generated with `_force_rmask_constant=True`, so `filter_masks()` discarded `r0_mask` → `indexing()` produced `mask=None` → `tl.load(..., None)` already in the buffer.
- `unmasked['compute']` was generated with `r0_mask` discarded from the `masks` set in `reduction()` → `cond` was empty or `xmask`-only → `where_cond` returned `tval` directly (no `tl.where(r0_mask, ...)`).
- `self.loads` / `self.compute` (masked) are the normal codegen output, used as-is for the tail loop.

### 4.8 Fallback: Minimal String Replacement (not implemented)

The dual-generation approach (Approach B) was successfully implemented. The CSE state save/restore + counter reset technique avoids the CSE side effects mentioned below. This fallback section is retained for reference only.

If the dual-generation approach had proved too complex due to CSE side effects (e.g., `reduction_cache` cross-contamination, `_load_counts` doubling, `outside_loop_vars` divergence), the fallback would use `IndentedBuffer.map()` with string replacement:

```python
def _peel_rmask_from_line(line, rmask_name):
    """Transform a single buffer line to remove rmask_name from mask expressions."""
    if isinstance(line, DeferredLineBase):
        new_str = _peel_rmask_from_str(line.line, rmask_name)
        return line._new_line(new_str) if new_str != line.line else line
    elif isinstance(line, str):
        return _peel_rmask_from_str(line, rmask_name)
    return line  # LineContext, etc.
```

`_peel_rmask_from_str(s, rmask_name)` applies these transformations:

| # | Pattern | Before | After |
|---|---------|--------|-------|
| 1 | Compound mask (rmask first) | `r0_mask & xmask` | `xmask` |
| 2 | Compound mask (rmask last) | `xmask & r0_mask` | `xmask` |
| 3 | `tl.where` with sole rmask | `tl.where(r0_mask, X, acc)` | `X` |
| 4 | Standalone mask arg | `, r0_mask,` | `, None,` |
| 5 | Dangling `other=` | `, None, other=0.0)` | `, None)` |

### 4.9 Integration Points

**Two integration points** are needed:

**1. Buffer generation** — in `SIMDScheduling.codegen_node_schedule_with_kernel()` (simd.py:2031):

```python
def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
    with kernel:
        # ... first pass (indexing decisions) unchanged ...
        kernel.finalize_indexing(all_indexing.keys())

        # Check peeling eligibility before the codegen pass
        loop_trees = [t for t in kernel.range_trees if t.is_loop]
        if hasattr(kernel, '_should_peel_reduction_loop') and kernel._should_peel_reduction_loop(loop_trees):
            self._codegen_node_schedule_with_peeling(node_schedule, kernel)
        else:
            # ... existing second pass (normal codegen) unchanged ...
```

**2. Loop emission** — in `TritonKernel.codegen_body()` (triton.py:5201):

```python
elif self.inside_reduction and len(loop_trees) > 0:
    if hasattr(self, '_peeled_unmasked_bufs'):
        self._codegen_peeled_reduction_loop(loop_trees)
    else:
        # Original single-loop path (unchanged)
        ...
```

The presence of `_peeled_unmasked_bufs` (set by the dual-generation pass) is the signal to emit two loops instead of one. This avoids threading a flag through the call chain.

## 5. Correctness by Reduction Type

All reduction types use the same `tl.where(cond, next_val, acc)` pattern for masked accumulator updates. With dual generation, the unmasked pass uses the same `reduction()` code path — it simply sees an empty `cond` (because `filter_masks` discarded `r0_mask`) and calls `where_cond(tval, fval)` which returns `tval` directly. No `tl.where` is emitted. This handles all reduction types uniformly:

| Reduction Type | Accumulator Pattern | After Peeling (sole mask) |
|----------------|--------------------|----|
| sum | `_acc = tl.where(r0_mask, _acc + val, _acc)` | `_acc = _acc + val` |
| max/min | `_acc = tl.where(r0_mask, max(_acc, val), _acc)` | `_acc = max(_acc, val)` |
| argmax/argmin | Two accumulators, same pattern | Same |
| welford | Three accumulators (`mean`, `m2`, `weight`) | Same |
| online_softmax | Two accumulators (`max`, `sum`) | Same |
| prod | `_acc = tl.where(r0_mask, _acc * val, _acc)` | `_acc = _acc * val` |

## 6. Edge Cases

| Case | Behavior |
|------|----------|
| `r0_numel` statically multiple of `R0_BLOCK` | `_has_constant_mask` = True -> peeling skipped (no mask to remove) |
| `r0_numel` runtime multiple of `R0_BLOCK` | `aligned = r0_numel`, tail loop runs 0 iterations |
| `r0_numel < R0_BLOCK` | `aligned = 0`, main loop runs 0 iterations, tail runs 1 |
| Dynamic `r0_numel` (symbolic) | Works: `aligned` computed at runtime |
| Cooperative reduction | Skipped (follow-up) |
| Multi-dim reduction | Skipped (follow-up) |
| Block pointer reductions | Skipped (follow-up) |

## 7. Test Plan

Add `TestLoopPeeling(InductorTestCase)` in `test/inductor/test_codegen_triton.py`:

1. **`test_reduction_generates_peeled_loop`** -- compile `torch.sum(x, dim=-1)` with non-aligned rnumel, verify generated code contains `r0_numel_aligned` and two `tl.range` loops, first loop lacks `r0_mask`.

2. **`test_reduction_peeling_correctness_{sum,max,var}`** -- numerical correctness for sum, max, var (welford) with `rnumel=1027` (non-aligned).

3. **`test_static_aligned_no_peeling`** -- `rnumel` known multiple of `R0_BLOCK` -> no peeling (no `r0_numel_aligned` in output).

4. **`test_disabled_no_peeling`** -- `config.triton.loop_peeling = False` -> no peeling regardless of shape.

5. **`test_dynamic_shapes_peeling`** -- compile with `dynamic=True`, verify peeling triggers and produces correct results at three runtime sizes: non-aligned (1027), runtime-aligned (1024, tail runs 0 iterations), and small (3, main runs 0 iterations).

## 8. Future Work

1. **Block pointer reductions:** Modify `boundary_check` tuple; may need additional handling in `_force_constant_rmask` or a separate code path.
2. **Cooperative reductions:** Compute `aligned` relative to `rsplit_start`/`rsplit_end`.
3. **Multi-dim reductions:** Peel innermost loop only.
4. **Compose with xmask_unswitch:** After peeling removes `r0_mask`, `xmask` becomes sole mask in main loop, enabling xmask_unswitch to remove it too for full vectorization on dynamic-shape 2D reductions.
5. **Stride alignment for full vectorization:** Even with masks removed, dynamic strides (`ks0`) prevent pointer alignment proof. Dual-kernel dispatch (Option B) or `tl.assume` needed for the last mile.
