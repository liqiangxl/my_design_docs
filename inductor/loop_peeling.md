# Reduction Loop Peeling for TorchInductor Triton Codegen

**Issue:** https://github.com/pytorch/pytorch/issues/148402

## Summary

Addressed the reviewer's concern about 20% codegen overhead by replacing the two-pass clone approach with a single-pass masked→unmasked line map (same pattern as xmask_unswitch in PR #178318). During the normal masked codegen pass, `load()`, `store()`, and `reduction()` register `masked_line → unmasked_line` entries in `_unmasked_line_map` by building the unmasked variant from the same structured `mask_vars` data. At loop emission, `_splice_unmasked()` uses `IndentedBuffer.map()` to substitute mapped lines for the main loop; the tail loop splices the originals via the existing `_emit_reduction_loops()` — no code duplication.

**Result:** codegen overhead eliminated (was +17%, now 0%). Runtime: **1.42x** on bf16 softmax (32768, 50257) static, **1.22x** dynamic. 12 tests covering sum, argmax/argmin, softmax, log_softmax, welford (var/std), multi-segment, combo kernels, and dynamic shapes.

Changes: `config.py` (+4), `simd.py` (+9 −3), `triton.py` (+330 net), `test_codegen_triton.py` (+100). No changes to `common.py`.


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

**Measured impact:** 1.38x slowdown on bf16 log_softmax kernel (1.588ms → 1.151ms with fix).

## 2. Solution: Loop Peeling

Split the reduction loop into two:

```python
r0_numel_aligned = (r0_numel // R0_BLOCK) * R0_BLOCK

# Main loop: all R0_BLOCK elements valid, no mask needed
for r0_offset in tl.range(0, r0_numel_aligned, R0_BLOCK):
    r0_index = r0_offset + r0_base
    tmp0 = tl.load(in_ptr0 + ..., xmask, ...)   # no r0_mask
    _tmp3_max = _tmp3_max_next                    # no tl.where
    _tmp3_sum = _tmp3_sum_next

# Tail loop: partial block, needs mask
for r0_offset in tl.range(r0_numel_aligned, r0_numel, R0_BLOCK):
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    tmp0 = tl.load(in_ptr0 + ..., r0_mask & xmask, ...)
    _tmp3_max = tl.where(r0_mask & xmask, _tmp3_max_next, _tmp3_max)
    _tmp3_sum = tl.where(r0_mask & xmask, _tmp3_sum_next, _tmp3_sum)
```

Both loops update the **same accumulator variables** — state carries across naturally.

## 3. Implementation

### 3.1 Config Flag

```python
# torch/_inductor/config.py
class triton:
    loop_peeling: bool = False  # disabled by default
```

### 3.2 Eligibility

`_should_peel_reduction_loop(self, loop_trees)` on `TritonKernel`:

```python
config.triton.loop_peeling
and len(loop_trees) == 1              # single reduction dim only
and not self.cooperative_reduction     # not cooperative
and not self._has_constant_mask(...)   # mask is not already elided
and not self.pointer_advancements.get(...)  # no block_ptr
```

### 3.3 Single-Pass Map Approach

The core challenge: `self.loads`, `self.compute`, `self.stores` are `IndentedBuffer` objects containing baked-in mask expressions like `r0_mask & xmask`. We need an unmasked variant for the main loop.

**Approach:** During the single (masked) codegen pass, `load()`, `store()`, and `reduction()` build both the masked line and the unmasked line from the same structured data (`mask_vars` set), then register `masked_line → unmasked_line` in `_unmasked_line_map`. At loop emission time, `_splice_unmasked()` replaces mapped lines with their unmasked variants for the main loop, while `splice()` emits the originals for the tail loop.

**Why not two-pass clone approach?** An earlier design ran codegen twice (clone kernel for unmasked, real kernel for masked). Profiling showed this added ~2ms / ~17% overhead to `GraphLowering.codegen`, entirely from redundant `node.codegen()` calls. The single-pass map approach has zero overhead.

### 3.4 Key Data Structures

```python
# TritonKernel.__init__:
self._unmasked_line_map: dict[str, str] | None = None
# None = peeling inactive (zero overhead). {} = peeling active.
```

Set to `{}` by `codegen_node_schedule_with_kernel` in `simd.py` before the codegen pass, when `_should_peel_reduction_loop` returns True.

### 3.5 Registration Points

During codegen, four sites register masked→unmasked mappings:

**1. `load()` — unmasked load:**
```python
# Masked:  tl.load(ptr, r0_mask & xmask, other=0.0)
# Unmasked: tl.load(ptr, xmask, other=0.0)
# (when r0_mask is sole mask: tl.load(ptr, None) — also drops `other`)
```

Built from `indexing.mask_vars` via `_get_peeled_mask()` which filters out reduction masks using `prefix_is_reduction()`.

**2. `store()` — unmasked store:**
```python
# Masked:  tl.store(ptr, val, r0_mask & xmask)
# Unmasked: tl.store(ptr, val, xmask)
```

**3. `reduction()` — unmasked accumulator updates:**
```python
# Masked:  _acc = tl.where(r0_mask & xmask, _acc_next, _acc)
# Unmasked: _acc = tl.where(xmask, _acc_next, _acc)
# (when r0_mask is sole mask: _acc = _acc_next)
```

Handles sum, max/min, argmax/argmin (2 accumulators), online_softmax (2 accumulators), and welford (3 accumulators).

**4. `load()` broadcast+where path — unmasked where after broadcast:**
```python
# Masked:  tl.where(r0_mask & xmask, result_var, 0.0)
# Unmasked: tl.where(xmask, result_var, 0.0)
```

### 3.6 Loop Emission

In `codegen_body()`:

```python
elif self.inside_reduction and len(loop_trees) > 0:
    peeling = (
        self._unmasked_line_map is not None
        and not self.pointer_advancements.get(loop_trees[0].symt)
    )
    if peeling:
        self._emit_unmasked_reduction_loop(loop_trees)  # [0, aligned)
    self._emit_reduction_loops(loop_trees, peeling=peeling)  # [aligned, numel) or [0, numel)
```

`_emit_reduction_loops` is the refactored general-purpose reduction loop emitter (extracted from the original inline code). When `peeling=True`, it emits `[aligned, numel)`; otherwise `[0, numel)`. Both peeling and non-peeling paths use it — no code duplication.

`_emit_unmasked_reduction_loop` emits the `[0, aligned)` loop using `_splice_unmasked()`:

```python
def _splice_unmasked(self, buf):
    def _replace(line):
        raw = line.line if isinstance(line, DeferredLineBase) else line
        if isinstance(raw, str):
            stripped = raw.lstrip()
            if stripped in peel_map:
                new_raw = raw[:len(raw) - len(stripped)] + peel_map[stripped]
                return line._new_line(new_raw) if isinstance(line, DeferredLineBase) else new_raw
        return line
    self.body.splice(buf.map(_replace))
```

Uses `IndentedBuffer.map()` for line transformation and `splice()` for indent handling — no manual dedent logic.

## 4. Edge Cases

| Case | Behavior |
|------|----------|
| `r0_numel` statically multiple of `R0_BLOCK` | `_has_constant_mask` = True → peeling skipped |
| `r0_numel` runtime multiple of `R0_BLOCK` | `aligned = r0_numel`, tail loop runs 0 iterations |
| `r0_numel < R0_BLOCK` | `aligned = 0`, main loop runs 0 iterations, tail runs 1 |
| Dynamic `r0_numel` (symbolic) | Works: `aligned` computed at runtime |
| Cooperative / multi-dim / block-ptr | Skipped (follow-up) |

## 5. Test Coverage: Masked vs Unmasked Code by Reduction Type

All tests use shape `(32, 1027)` (rnumel not a power of 2). The table shows lines that differ between the main (unmasked) and tail (masked) loops.

| Reduction Type | Main Loop (unmasked) | Tail Loop (masked) |
|---|---|---|
| **sum** | `tl.load(..., xmask, ...)` | `tl.load(..., r0_mask & xmask, ...)` |
| | `_tmp2 = tl.where(xmask, tmp3, _tmp2)` | `_tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)` |
| **argmax** | `tl.load(..., xmask, ...)` | `tl.load(..., r0_mask & xmask, ...)` |
| | `_tmp2 = tl.where(xmask, _tmp2_next, _tmp2)` | `_tmp2 = tl.where(r0_mask & xmask, _tmp2_next, _tmp2)` |
| | `_tmp2_index = tl.where(xmask, ...)` | `_tmp2_index = tl.where(r0_mask & xmask, ...)` |
| **argmin** | same as argmax (with `minimum_with_index`) | same as argmax |
| **softmax** (pass 1: online_softmax) | `tl.load(..., xmask, ...)` | `tl.load(..., r0_mask & xmask, ...)` |
| | `_tmp2_max = tl.where(xmask, ...)` | `_tmp2_max = tl.where(r0_mask & xmask, ...)` |
| | `_tmp2_sum = tl.where(xmask, ...)` | `_tmp2_sum = tl.where(r0_mask & xmask, ...)` |
| **softmax** (pass 2: store) | `tl.store(..., tmp7, xmask)` | `tl.store(..., tmp7, r0_mask & xmask)` |
| **log_softmax** | same as softmax (two passes) | same as softmax |
| **var/std** (welford) | `tl.load(..., xmask, ...)` | `tl.load(..., r0_mask & xmask, ...)` |
| | `tmp2_mean = tl.where(xmask, ...)` | `tmp2_mean = tl.where(r0_mask & xmask, ...)` |
| | `tmp2_m2 = tl.where(xmask, ...)` | `tmp2_m2 = tl.where(r0_mask & xmask, ...)` |
| | `tmp2_weight = tl.where(xmask, ...)` | `tmp2_weight = tl.where(r0_mask & xmask, ...)` |
| **max + sum** (two segments) | Each segment peeled independently | Each segment peeled independently |

**Pattern:** The only difference is `r0_mask &` prepended to existing masks in `tl.load`, `tl.where`, and `tl.store`.

## 6. Benchmarks

GB200 (SM 10.0, 152 SMs, HBM3e 7928 GB/s peak).

### Raw Triton kernel: bf16 log_softmax (32768, 50257)

| Config | Time | Bandwidth | Speedup |
|--------|------|-----------|---------|
| 1 loop (masked) | 1.588 ms | 2074 GB/s | — |
| 2 loops (peeled) | 1.151 ms | 2862 GB/s | **1.38x** |

### torch.compile static shapes: bf16 softmax (32768, 50257)

| Config | Time | Speedup |
|--------|------|---------|
| peeling=off | 3.874 ms | — |
| peeling=on | 2.720 ms | **1.42x** |

### torch.compile dynamic shapes: bf16 softmax (32768, 50257)

| Config | Time | Speedup |
|--------|------|---------|
| peeling=off | 3.715 ms | — |
| peeling=on | 3.053 ms | **1.22x** |

**Why dynamic shapes are slower (1.22x vs 1.42x):** With dynamic shapes, the stride `ks0` is a runtime value, so the load address is `r0_1 + ks0*x0`. This has two effects:

1. **No vectorization in either case.** Triton emits scalar `ld.global.b16` loads (not vectorized `ld.global.v4.b16`) because it can't prove pointer alignment from a symbolic stride. Both peeling=on and peeling=off use scalar loads, so vectorization is not the source of speedup.

2. **Extra runtime IMAD.** Each load address requires a runtime multiply `ks0*x0`, adding an IMAD instruction per load that static shapes avoid (the stride is folded into the pointer at compile time).

**Where the 1.22x speedup comes from (without vectorization):**

- **Eliminated `tl.where` overhead:** The main loop does `_acc = _acc_next` instead of `_acc = tl.where(r0_mask & xmask, _acc_next, _acc)` — saves a select instruction per accumulator per iteration.
- **Eliminated `r0_mask` computation:** The main loop skips `r0_mask = r0_index < r0_numel` and the `r0_mask & xmask` AND — saves a compare + AND per iteration.
- **Better loop optimization:** `tl.range(0, aligned, BLOCK)` with uniform iterations (no mask divergence) allows Triton's compiler to optimize more aggressively (e.g., software pipelining).

With rnumel=50257 and typical RBLOCK=2048, there are ~24 full iterations and 1 tail — savings apply to all 24.

**To recover vectorization on dynamic shapes**, two additional optimizations are needed (see Future Work):
1. Remove `xmask` via xmask_unswitch → achieves `mask=None`
2. Prove stride divisibility via `tt.divisibility` hint → achieves alignment proof

### Compilation time overhead

| Metric | peeling=off | peeling=on | Delta |
|--------|------------|-----------|-------|
| `codegen_node_schedule` | 16.86 ms | 16.56 ms | **-0.3 ms (-1.8%)** |

Zero compilation overhead with the single-pass map approach.

## 7. Test Plan

12 tests in `TestLoopPeeling` class (`test/inductor/test_codegen_triton.py`). Each verifies `r0_numel_aligned` appears in generated code and checks numerical correctness.

1. `test_inner_reduction` — `torch.sum(x, dim=-1)`
2. `test_outer_reduction` — `torch.sum(x, dim=0)`
3. `test_two_inner_reductions` — `max` + `sum` (multi-segment)
4. `test_one_inner_one_outer_reduction` — `sum(dim=-1)` + `sum(dim=0)`
5. `test_combo_kernel_two_reductions` — two `sum` with `combo_kernels=True`
6. `test_argmax` — argmax accumulator pair
7. `test_argmin` — argmin accumulator pair
8. `test_softmax` — online_softmax accumulators + store
9. `test_log_softmax` — online_softmax + sub/log/store
10. `test_var` — welford (3 accumulators)
11. `test_std` — welford
12. `test_dynamic_shapes` — `softmax` with `dynamic=True`, two different input shapes

## 8. Future Work

1. **Block pointer reductions:** Use `boundary_check` instead of `r0_mask`.
2. **Cooperative reductions:** Compute `aligned` relative to `rsplit_start`/`rsplit_end`.
3. **Multi-dim reductions:** Peel innermost loop only.
4. **Compose with xmask_unswitch:** After peeling removes `r0_mask`, `xmask` becomes the sole mask — exactly the condition for xmask_unswitch. The two compose to recover `mask=None` for dynamic-shape 2D reductions. Both use `_unmasked_line_map` for masked→unmasked line mappings.
5. **Stride divisibility hints:** Even with `mask=None`, Triton needs alignment proof for vectorized loads. Requires `tl.assume` or dual-kernel dispatch.
