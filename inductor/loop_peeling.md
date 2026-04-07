# Reduction Loop Peeling for TorchInductor Triton Codegen

**Issue:** https://github.com/pytorch/pytorch/issues/148402


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

## 3. Implementation Design

### 3.1 Files to Modify

| File | Change |
|------|--------|
| `torch/_inductor/config.py` | Add `triton.loop_peeling` config flag |
| `torch/_inductor/codegen/common.py` | Base `Kernel.make_isolated_codegen_copy()` method for shallow-copying kernels with fresh mutable state |
| `torch/_inductor/codegen/simd.py` | `SIMDKernel.make_isolated_codegen_copy()` override (adds `body`/`indexing_code` buffers), `_codegen_node_schedule_with_peeling` for dual buffer generation |
| `torch/_inductor/codegen/triton.py` | `TritonKernel.make_isolated_codegen_copy()` override (adds block-ptr state, prologue, helper functions), eligibility check, `_force_constant_rmask` context manager, `_has_constant_mask` 2-line addition, `_codegen_peeled_reduction_loop` emission |
| `test/inductor/test_codegen_triton.py` | Tests |

### 3.2 Config Flag

```python
# torch/_inductor/config.py, in class triton:

# Enable loop peeling for non-persistent reductions. Splits the reduction
# loop into a main (unmasked, vectorizable) loop and a tail (masked) loop.
loop_peeling: bool = False
```

Disabled by default until broader validation is complete. Enable with `config.triton.loop_peeling = True` or the `@inductor_config.patch("triton.loop_peeling", True)` decorator in tests.

### 3.3 Eligibility

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

### 3.4 Buffer Generation Strategy

The core challenge: `self.loads`, `self.compute`, `self.stores`, `self.indexing_code` are `IndentedBuffer` objects populated once (during node codegen in `codegen_node_schedule_with_kernel`) before `codegen_body()` is called. They contain baked-in mask expressions like `r0_mask & xmask` and `tl.where(r0_mask & xmask, ...)`. We need an unmasked copy for the main loop.

**Two approaches considered:**

| Approach | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| **A. String replacement** | `IndentedBuffer.map()` post-processes buffers to remove `r0_mask` from baked-in strings | Local to `codegen_body()`, no changes to buffer-filling flow | Fragile: must match all mask patterns; regex rules are coupled to codegen output format |
| **B. Dual generation** | Run node codegen twice — once with `r0_mask` elided via `filter_masks`, once normally — capturing each into separate buffers | Uses the codegen's native mask elision; no string matching at all | Touches `codegen_node_schedule_with_kernel`; must manage CSE state for two passes |

**Chosen: Approach B (dual generation).** The codegen already has a clean mechanism for mask elision: `filter_masks()` discards mask names where `_has_constant_mask()` returns True, causing `indexing()` to emit `None` masks and `reduction()` to emit bare accumulator updates (no `tl.where`). We add a flag to force this for the reduction tree, then run the buffer-filling codegen twice.

### 3.5 Forcing Mask Elision

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

### 3.6 Kernel Cloning: `make_isolated_codegen_copy()`

Dual buffer generation (Section 3.7) needs to run the node codegen twice — once without `r0_mask`, once with. The unmasked pass must be fully isolated so its side effects (counter increments, buffer writes, CSE cache pollution) don't leak into the real kernel. Rather than manually saving/restoring every mutable field, we use a **three-level `make_isolated_codegen_copy()` method chain** that creates a shallow copy with fresh mutable state:

**`Kernel.make_isolated_codegen_copy()`** (common.py) — base method:
```python
def make_isolated_codegen_copy(self) -> Self:
    clone = copy.copy(self)
    clone.exit_stack = contextlib.ExitStack()
    clone.loads = IndentedBuffer()
    clone.compute = IndentedBuffer()
    clone.stores = IndentedBuffer()
    clone.cse = self.cse.clone()
    # Start counter from the same position so the clone generates
    # the same variable names without advancing the real counter.
    _current = next(self.cse.iter_buffer_ids)
    self.cse.iter_buffer_ids = itertools.count(_current)
    clone.cse.iter_buffer_ids = itertools.count(_current)
    clone.must_keep_buffers = OrderedSet()
    clone.store_buffer_names = OrderedSet()
    return clone
```

**`SIMDKernel.make_isolated_codegen_copy()`** (simd.py) — adds SIMD-specific buffers:
```python
def make_isolated_codegen_copy(self):
    clone = super().make_isolated_codegen_copy()
    clone.body = IndentedBuffer()
    clone.indexing_code = IndentedBuffer()
    return clone
```

**`TritonKernel.make_isolated_codegen_copy()`** (triton.py) — adds Triton-specific state:
```python
def make_isolated_codegen_copy(self):
    clone = super().make_isolated_codegen_copy()
    clone.prologue = IndentedBuffer()
    clone.prologue_cache = {}
    clone.post_loop_combine = IndentedBuffer()
    clone.post_loop_store = IndentedBuffer()
    clone.outside_loop_vars = OrderedSet()
    clone.autotune_hints = OrderedSet()
    clone.helper_functions = HelperFunctions()
    clone._load_counts = collections.Counter()
    clone.stores_with_contiguous_rdim = []
    return clone
```

| State type | Isolation mechanism |
|-----------|-------------------|
| Scalar counters (`num_load`, `num_store`, `num_reduction`, `atomic_add_found`) | Python rebind-on-assignment: `clone.num_load += 1` creates a new int on the clone, original unchanged |
| `IndentedBuffer` fields (`body`, `loads`, `compute`, `prologue`, etc.) | Replaced with fresh instances across the three override levels |
| Mutable sets (`autotune_hints`, `store_buffer_names`, `must_keep_buffers`, `outside_loop_vars`) | Replaced with fresh `OrderedSet()` |
| CSE caches | `kernel.cse.clone()` creates a new CSE; counter starts from the same position via `itertools.count(_current)` |
| `exit_stack` | Replaced with fresh `ExitStack()` so `with clone:` doesn't corrupt the original kernel's handler stack |
| Block-pointer state (`block_ptr_id`, `block_ptr_to_buffer`, `pointer_advancements`) | Not reset — peeling is gated by `not self.pointer_advancements.get(...)`, so these are empty when the clone is created |
| Read-only shared state (`range_trees`, `args`, `features`, `numels`) | Shared by reference — safe because codegen only reads these |

**CSE counter sharing:** Both `self.cse.iter_buffer_ids` and `clone.cse.iter_buffer_ids` are set to `itertools.count(_current)` where `_current` is peeked from the original counter. This ensures both passes produce variable names starting from the same position. The original counter is also replaced (since `next()` consumed one value from it).

### 3.7 Dual Buffer Capture

New method on `SIMDScheduling` that runs the node codegen twice. The unmasked pass runs on a clone from `make_isolated_codegen_copy()`. When the schedule has multiple reduction segments (e.g. max, pointwise, sum), each segment gets its own set of unmasked buffers.

```python
def _codegen_node_schedule_with_peeling(self, node_schedule, kernel):
    """Run node codegen twice to produce unmasked (main) and masked (tail) buffers."""

    # First pass (unmasked) on an isolated copy
    clone = kernel.make_isolated_codegen_copy()

    # One entry per reduction segment, consumed in order by codegen_body.
    unmasked_bufs_list: list[dict[str, IndentedBuffer]] = []

    with clone._force_constant_rmask():
        with clone:  # sets V.ops and V.kernel to clone
            inside = True
            for node in node_schedule:
                if node is DisableReduction:
                    # End of a reduction segment — snapshot its buffers.
                    unmasked_bufs_list.append({
                        'indexing_code': clone.indexing_code,
                        'loads': clone.loads,
                        'compute': clone.compute,
                        'stores': clone.stores,
                    })
                    # Reset for the next segment, mirroring the CSE
                    # invalidation that codegen_body does between loops.
                    clone.indexing_code = IndentedBuffer()
                    clone.loads = IndentedBuffer()
                    clone.compute = IndentedBuffer()
                    clone.stores = IndentedBuffer()
                    clone.cse.invalidate(clone.outside_loop_vars)
                    for tree in clone.range_trees:
                        tree.cache_clear()
                    inside = False
                elif node is EnableReduction:
                    inside = True
                elif inside:
                    indexing_dtype_strength_reduction(node._body)
                    index_vars = clone.split_and_set_ranges(node.get_ranges())
                    node.codegen(index_vars)

    # Capture trailing reduction segment (no final DisableReduction).
    unmasked_bufs_list.append({
        'indexing_code': clone.indexing_code,
        'loads': clone.loads,
        'compute': clone.compute,
        'stores': clone.stores,
    })

    # Clear range tree caches populated by the clone pass so the
    # real kernel's second pass recomputes them.
    for tree in kernel.range_trees:
        tree.cache_clear()

    # Set unmasked buffers BEFORE the masked pass.  The masked pass may
    # hit DisableReduction which calls kernel.disable_reduction() ->
    # kernel.codegen_body(), flushing the reduction loop.  codegen_body
    # pops from this list to get the matching segment's unmasked buffers.
    kernel._peeled_unmasked_bufs_list = unmasked_bufs_list

    # --- Second pass (masked) on the real kernel: normal codegen ---
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
```

**Why `make_isolated_codegen_copy()` instead of save/restore?** `node.codegen()` dispatches through `CSEProxy` which increments `kernel.num_load`, `kernel.num_store`, `kernel.num_reduction`, and kernel methods may set `atomic_add_found` or add to `autotune_hints`, `store_buffer_names`, `must_keep_buffers`. Manually saving/restoring every mutable field is fragile — it's easy to miss one and get double-counted metadata. The three-level copy chain isolates all of these automatically.

**Multi-segment handling:** A fused kernel like `max(x) + sum(x)` has a `node_schedule` like `[max_reduction, DisableReduction, pointwise, EnableReduction, sum_reduction]`. Each `DisableReduction` boundary snapshots the current clone buffers into `unmasked_bufs_list` and resets the clone's buffers for the next segment. The trailing segment (after the last `EnableReduction`) is captured after the loop. During the masked pass, `kernel.disable_reduction()` triggers `codegen_body()`, which pops from `_peeled_unmasked_bufs_list` to get the matching segment's unmasked buffers.

**Why `with clone:`?** `node.codegen()` replays the LoopBody FX graph through `V.ops` / `V.kernel`. These are set by `Kernel.__enter__` (called by `with clone:`). Without it, `V.ops` would be a `MockHandler` and codegen would produce no output.

**Why skip non-reduction nodes in the first pass?** Fused kernels like `sum(x) + 1.0` have a `node_schedule` containing `[reduction_node, DisableReduction, pointwise_node, EnableReduction]`. The pointwise node has `ranges=((N,), ())` — no reduction dimension. Processing it with `inside_reduction=True` would fail in `split_and_set_ranges`. We only need the reduction loop body for the unmasked pass, so we skip nodes outside `DisableReduction`/`EnableReduction` boundaries.

**How the clone fills the unmasked buffers:**

`node.codegen()` replays the LoopBody FX graph. Each op dispatches to `TritonKernel` methods that write into `self.loads`, `self.compute`, `self.stores`. Since `V.kernel` points to the clone, all writes go to the clone's fresh buffers:

```
1. clone has fresh empty IndentedBuffers for all code sections

2. _force_constant_rmask makes filter_masks() drop r0_mask

3. with clone: sets V.ops = CSEProxy(clone, TritonOverrides)
               sets V.kernel = clone

4. node.codegen() runs, dispatching through V.ops -> clone (as TritonKernel):

     ops.load(...)  ->  clone.load()
        -> clone.indexing() sees no r0_mask (forced constant)
        -> emits "tl.load(ptr, xmask, ...)"  (no r0_mask!)
        -> clone.cse.generate(clone.loads, expr) -> writes to clone.loads

     ops.reduction(...)  ->  clone.reduction()
        -> filter_masks returns masks={xmask}  (r0_mask dropped!)
        -> cond = "xmask"  (not "r0_mask & xmask")
        -> clone.compute.writeline("_tmp2 = tl.where(xmask, tmp3, _tmp2)")

5. At DisableReduction: snapshot clone's buffers, reset for next segment

6. with clone: exits, V.ops/V.kernel restored to previous values

7. unmasked_bufs_list contains one entry per reduction segment
   clone is discarded; only the list is kept
```

### 3.8 Two-Loop Emission

New method `_codegen_peeled_reduction_loop(self, loop_trees)` on `TritonKernel`:

```python
def _codegen_peeled_reduction_loop(self, loop_trees):
    tree = loop_trees[0]
    prefix = tree.prefix                    # "r0"
    unmasked = self._peeled_unmasked_bufs_list.pop(0)  # consume next segment

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

### 3.9 Fallback: Minimal String Replacement (not implemented)

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

### 3.10 Integration Points

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

**2. Loop emission** — in `TritonKernel.codegen_body()` (triton.py:5262):

```python
elif self.inside_reduction and len(loop_trees) > 0:
    if (
        hasattr(self, '_peeled_unmasked_bufs_list')
        and self._peeled_unmasked_bufs_list
        and not self.pointer_advancements.get(loop_trees[0].symt)
    ):
        self._codegen_peeled_reduction_loop(loop_trees)
    else:
        # Original single-loop path (unchanged)
        ...
```

The presence of `_peeled_unmasked_bufs_list` (set by the dual-generation pass) is the signal to emit two loops instead of one. The additional guard on `pointer_advancements` ensures block-pointer reductions fall back to the original single-loop path, since peeled loops don't emit `tl.advance()`. This avoids threading a flag through the call chain.

## 4. Correctness by Reduction Type

All reduction types use the same `tl.where(cond, next_val, acc)` pattern for masked accumulator updates. With dual generation, the unmasked pass uses the same `reduction()` code path — it simply sees an empty `cond` (because `filter_masks` discarded `r0_mask`) and calls `where_cond(tval, fval)` which returns `tval` directly. No `tl.where` is emitted. This handles all reduction types uniformly:

| Reduction Type | Accumulator Pattern | After Peeling (sole mask) |
|----------------|--------------------|----|
| sum | `_acc = tl.where(r0_mask, _acc + val, _acc)` | `_acc = _acc + val` |
| max/min | `_acc = tl.where(r0_mask, max(_acc, val), _acc)` | `_acc = max(_acc, val)` |
| argmax/argmin | Two accumulators, same pattern | Same |
| welford | Three accumulators (`mean`, `m2`, `weight`) | Same |
| online_softmax | Two accumulators (`max`, `sum`) | Same |
| prod | `_acc = tl.where(r0_mask, _acc * val, _acc)` | `_acc = _acc * val` |

## 5. Edge Cases

| Case | Behavior |
|------|----------|
| `r0_numel` statically multiple of `R0_BLOCK` | `_has_constant_mask` = True -> peeling skipped (no mask to remove) |
| `r0_numel` runtime multiple of `R0_BLOCK` | `aligned = r0_numel`, tail loop runs 0 iterations |
| `r0_numel < R0_BLOCK` | `aligned = 0`, main loop runs 0 iterations, tail runs 1 |
| Dynamic `r0_numel` (symbolic) | Works: `aligned` computed at runtime |
| Cooperative reduction | Skipped (follow-up) |
| Multi-dim reduction | Skipped (follow-up) |
| Block pointer reductions | Skipped (follow-up) |

## 6. Test Plan

Add `TestLoopPeeling(InductorTestCase)` in `test/inductor/test_codegen_triton.py`. All tests use a shared `_check_peeling` helper that compiles the function, asserts `r0_numel_aligned` appears in the generated code (proving peeling triggered), and checks numerical correctness against eager execution.

1. **`test_inner_reduction`** -- `torch.sum(x, dim=-1)` with shape `(32, 1027)` (non-aligned rnumel). Verifies basic peeling for inner-dim reduction.

2. **`test_outer_reduction`** -- `torch.sum(x, dim=0)` with shape `(1027, 32)`. Verifies peeling works for outer-dim reductions.

3. **`test_two_inner_reductions`** -- `max(x, dim=-1, keepdim=True)` followed by `sum(x - a, dim=-1)` with shape `(32, 1027)`. Tests multi-segment reduction in a single kernel: the schedule has two reduction segments separated by `DisableReduction`/`EnableReduction`, each getting its own entry in `_peeled_unmasked_bufs_list`.

4. **`test_one_inner_one_outer_reduction`** -- `sum(x, dim=-1)` and `sum(x, dim=0)` on the same input with shape `(32, 1027)`. Tests peeling when inner and outer reductions coexist.

5. **`test_combo_kernel_two_reductions`** -- Two separate `sum(dim=-1)` reductions on different inputs `(32, 1027)` and `(64, 1027)` with `combo_kernels=True`. Tests peeling interacts correctly with combo kernel fusion.

## 7. Future Work

1. **Block pointer reductions:** Modify `boundary_check` tuple; may need additional handling in `_force_constant_rmask` or a separate code path.
2. **Cooperative reductions:** Compute `aligned` relative to `rsplit_start`/`rsplit_end`.
3. **Multi-dim reductions:** Peel innermost loop only.
4. **Compose with xmask_unswitch for full mask removal:** After peeling removes `r0_mask`, the main loop body still has `xmask` (spatial mask) when shapes are dynamic:

    ```python
    # Main loop after peeling: r0_mask removed, xmask remains
    tmp0 = tl.load(ptr, xmask, other=0.0)
    _acc = tl.where(xmask, _acc + tmp0, _acc)
    ```

    Now `xmask` is the **sole mask** -- exactly the condition where `xmask_unswitch` applies. The two optimizations compose:

    ```python
    # Main loop with peeling + xmask_unswitch:
    for r0_offset in tl.range(0, r0_numel_aligned, R0_BLOCK):
        r0_index = r0_offset + r0_base
        if xoffset + XBLOCK <= xnumel:
            tmp0 = tl.load(ptr, None)          # mask=None -> vectorizable!
            _acc = _acc + tmp0                  # no tl.where at all
        else:
            tmp0 = tl.load(ptr, xmask, other=0.0)
            _acc = tl.where(xmask, _acc + tmp0, _acc)
    ```

    This recovers `mask=None` for the main loop body, which is a prerequisite for Triton to emit vectorized loads (`ld.global.v4`). Neither optimization achieves this alone for dynamic-shape 2D reductions.

5. **Stride divisibility hint for vectorization:** Even with `mask=None`, Triton still needs to prove pointer alignment to emit vectorized loads. For dynamic shapes, the stride (e.g., `ks0` in `in_ptr0 + (r0_1 + ks0*x0)`) is not known at compile time. Without a `tl.assume(ks0 % 16 == 0)` or a divisibility hint in the kernel metadata (`'tt.divisibility': 16`), Triton falls back to scalar loads. Full vectorization requires both:
    - **mask=None** (from peeling + xmask_unswitch) — tells Triton no lane is masked out
    - **Pointer alignment proof** (from stride divisibility hints) — tells Triton the address is aligned for wide loads

    Options for the stride hint:
    - **`tl.assume`**: Emit `tl.assume(ks0 % 16 == 0)` at kernel entry. Requires runtime guard that the hint holds.
    - **Dual-kernel dispatch**: Generate two kernel variants (aligned / unaligned) and dispatch based on runtime stride check.
    - **Signature metadata**: Pass `ks0` with `'tt.divisibility': 16` in the Triton kernel metadata when the inductor can prove divisibility from symbolic constraints.
