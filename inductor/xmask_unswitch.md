# xmask Unswitch for Inductor Triton Codegen

**PR**: https://github.com/pytorch/pytorch/pull/178318

## Problem

Dynamic shapes in `torch.compile(dynamic=True)` prevent Triton from vectorizing loads/stores because `xnumel` is symbolic, so `xmask = xindex < xnumel` is always emitted. Triton cannot prove alignment → no vectorized `ld.global.v4` / `st.global.v4`.

## Solution

Generate an if/else split separating full blocks (unmasked, vectorizable) from the tail block (masked):

```python
if xoffset + XBLOCK <= xnumel:          # full block
    tmp0 = tl.load(in_ptr0 + x0, None)  # unmasked → vectorized
    tl.store(out_ptr0 + x0, tmp1, None)
else:                                     # tail block
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp1, xmask)
```

## Implementation

### Two-Phase Data Flow

```
Phase 1: load()/store() — register masked→unmasked pairs
  For each op where mask_vars == {"xmask"} and xnumel is dynamic:
    unmasked_line = line.replace(mask_str, "None", 1)
    _xmask_unswitch_map[line] = unmasked_line

Phase 2: _codegen_body() — emit if/else from map
  mode 0: plain (no unswitch)
  mode 1: per-load/store if/else, compute shared (default)
  mode 2: single if/else wrapping all loads+compute+stores
```

### Why Two-Phase (Not Inline at load/store)

The if/else cannot be emitted at `load()` time because the final control flow depends on the mode:
- **Mode 2** wraps all loads+compute+stores in a single if/else — needs all lines first.
- **Mode 1** needs compute *between* load and store branches — not yet visited at load() time.

The `_xmask_unswitch_map` bridges the two phases: lightweight registration at load/store time, consumed once all buffers are complete.

### Codegen: `writeline` + `indent()`

Uses `IndentedBuffer.writeline()` and `indent()` to emit if/else blocks directly, matching existing patterns (e.g., reduction loop codegen uses `writeline("for ...")` + `indent()`). No custom `IfThenElse` class needed.

### Unmasked Line Derivation

The unmasked variant is derived from the final `line` after all suffixes (`.to(tl.float32)`, `.to(tl.int1)`) are applied:

```python
if _register_xmask_load:
    unmasked_line = line.replace(f", {indexing.mask_str}", ", None", 1)
```

This avoids duplicating suffix logic for both masked and unmasked paths.

## Modes and Benchmarks

Config: `config.triton.xmask_unswitch` (int, default 1) / `TORCHINDUCTOR_XMASK_UNSWITCH`

| Mode | Memory-bound (x*y) | Compute-bound (tanh(sin+cos)) |
|------|-------------------|-------------------------------|
| disabled (0) | 0.92 ms | 1.41 ms |
| ldst-only (1) | 0.50 ms | 1.37 ms |
| whole-body (2) | 0.50 ms | 1.24 ms |

(GB200, fp16, 8192x73728)

**ldst-only is the default** because:
- Matches whole-body on the dominant memory-bound case (~1.8x speedup)
- Smaller Triton IR (no duplicated compute) — avoids code bloat when Triton's JIT auto-detects xnumel as div16
- Users can opt into mode 2 via `TORCHINDUCTOR_XMASK_UNSWITCH=2`

## Combo Kernel Interaction

Unswitch composes naturally with combo kernels — no special coupling needed.

Each combo sub-kernel is a regular `TritonKernel` that independently builds its own `_xmask_unswitch_map`. After `codegen_body()`, `ComboKernel.uniquify_block_sizes()` renames `xnumel` → `xnumel_0`, `xnumel_1`, etc., so the unswitch predicate becomes `xoffset + XBLOCK <= xnumel_0` per sub-kernel.

```python
pid = tl.program_id(0)
if pid < num_xblocks_0:                    # sub-kernel 0
    if xoffset + XBLOCK <= xnumel_0:      # unswitch: full tile
        tmp0 = tl.load(ptr, None)
    else:
        tmp0 = tl.load(ptr, xmask)
elif pid < num_xblocks_1:                  # sub-kernel 1
    if xoffset + XBLOCK <= xnumel_1:      # own xnumel
        ...
```

## Files Modified

| File | Changes |
|------|---------|
| `torch/_inductor/codegen/triton.py` | `_should_use_xmask_unswitch()`, `_codegen_body()`, `_codegen_body_unswitch_ldst_only()`, `_codegen_body_unswitch_whole()`, `_emit_unswitched_lines()`. Modified `load()` and `store()` to register unmasked variants. |
| `torch/_inductor/config.py` | `xmask_unswitch: int` (0/1/2, default 1) |
| `test/inductor/test_codegen_triton.py` | Parametrized tests for modes 1/2: dynamic shape, static shape (no unswitch), 2D transpose (no unswitch), combo kernel. |

## Future: IR-Level Unswitch

The current implementation is a codegen-level string transform. A reviewer suggested this should eventually be a lower-level IR transform. Three approaches analyzed:

### nvFuser Reference

nvFuser has a first-class IR-level unswitch: `ParallelType::Unswitch` on IterDomain axes → `UnrollPass` creates `kir::IfThenElse` with full-tile (no predicates) and partial-tile (inline predicates) body clones. This works because nvFuser has **imperative loop IR** — explicit `kir::ForLoop`, `kir::IfThenElse` nodes.

### Why Inductor Is Different

Inductor's IR is **declarative**: `ir.Loops` = `ranges` + `inner_fn` callable. No loop nests, no mask fields. `ops.load(name, index)` has no mask argument — masks are derived from `numel % BLOCK` at codegen time. BLOCK sizes are unknown until autotuning.

### Approaches

**A. Current (codegen string transform)** — works today, pragmatic. Fragile if codegen format changes. Triton-specific.

**B. IR-annotation + codegen realization** — add `SIMDKernelFeatures.unswitch_info()` as scheduling-layer analysis, replace string map with structured `MaskElidable` line type in `SIMDKernel`. Backend-agnostic, visible to tiling/autotuning. The right next step when Inductor's restructuring lands.

**C. LoopBody-level transform** — clone FX graph into full-tile/partial-tile variants. Closest to nvFuser but blocked by the fact that masks aren't in the IR. Requires `ops.unmasked_load` or similar — a much larger architectural change.

**Recommendation**: A now, B when the restructuring lands, C as long-term direction (requires masks as first-class IR concepts).
