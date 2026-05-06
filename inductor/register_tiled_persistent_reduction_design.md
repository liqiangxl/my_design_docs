# Register-Tiled Persistent Reductions

## Summary

**What**: For fused reduction+epilogue kernels (RMSNorm, softmax-like patterns),
retain shared bf16/fp16 inputs in registers across the tile loop so the epilogue
reuses them without reloading from global memory. Cuts GMEM traffic by up to 2×.

**Results** (GB200, RMSNorm bf16): 1.05–1.29× speedup for M ≥ 8192.
Best gain at [8192, 2048]: 1.29× (66% → 85% SOL). Peak 92% SOL at large shapes.

**Eligibility**: bf16/fp16 inputs, inner reduction, 1–2 shared reads,
rnumel ∈ [2048, 16384], power-of-2 tile size. Disabled by default
(`config.triton.register_tiled_persistent_reductions`).

**Branch**: `register-tiled-persistent-reduction`

---

## What It Does

When a fused reduction+epilogue kernel re-reads the same input in both
phases (e.g. RMSNorm reads `x` for the variance reduction and again for
the normalization epilogue), this optimization splits the reduction
dimension into tiles and keeps each tile's input in registers so the
epilogue can reuse it without a second global memory load.

## Comparison: Reload from GMEM vs Register-Tiled

Both approaches use two loops.  The critical difference is where the
epilogue gets its input data:

```
─── Two-pass reload (baseline) ────────────────────────────────────────
Loop 1 (reduction):
  for chunk in chunks:
    load x[chunk] from GMEM → accumulate → discard x[chunk]
  finalize reduction result

Loop 2 (epilogue):
  for chunk in chunks:
    RELOAD x[chunk] from GMEM → apply epilogue(x[chunk], reduction_result) → store
  ✗ 2× memory traffic — every element loaded twice

─── Register-tiled persistent ─────────────────────────────────────────
Loop 1 (reduction):
  for tile in tiles:
    load x[tile] from GMEM → accumulate → RETAIN x[tile] in registers
  finalize reduction result

Loop 2 (epilogue):
  for tile in tiles:
    retrieve x[tile] FROM REGISTERS → apply epilogue(x[tile], reduction_result) → store
  ✓ 1× memory traffic — no reload, data stays in registers across loops
```

The register-tiled approach trades register pressure for memory bandwidth.
It works when:
- `rnumel / num_tiles` fits in registers without spilling
- The kernel is memory-bound (not compute-bound)
- The shared input is read in both phases with identical indexing

When the reduction dimension is too large for a single flat persistent
block (which causes spills and kills occupancy), register-tiling splits
it into manageable chunks while still avoiding the second GMEM pass.

## Why Not Just Flat Persistent?

Flat persistent loads the entire reduction dimension into one block.
For large N, this creates prohibitive register pressure:

```
Shape [32768, 16384] bf16 RMSNorm:
  Two-pass reload:   322 us  83% SOL  (2× GMEM traffic)
  Flat persistent:   549 us  49% SOL  (register spills kill occupancy)
  Register-tiled:    302 us  89% SOL  (1× GMEM, registers fit)
```

## Concrete Example: `x / x.sum(dim=-1, keepdim=True)` with `x: bf16[512, 8192]`

This section traces the full path from Python through codegen for a
simple row-wise normalize using register-tiled persistent reduction with
`num_tiles=2` (autotuner winner), `R0_BLOCK=4096`.

### Why register-tiling kicks in

```
rnumel = 8192
threshold (min_numel) = 2048       → 8192 ≥ 2048 ✓
max_tiles = 4
tile selection: pick largest nt in [4..1] where 8192 % nt == 0
  8192 % 4 = 0 → num_tiles = 4 ✓
  Autotuner generates configs for both:
    {R0_BLOCK=2048, NUM_TILES=4}
    {R0_BLOCK=4096, NUM_TILES=2}   ← wins on GB200 (better occupancy)
```

Without register-tiling, this shape would use a **two-pass reload** reduction
(rnumel=8192 exceeds the ~4096 persistent threshold).  Register-tiling forces
`persistent_reduction=True` anyway, keeping the entire kernel in one
persistent launch with data retained in registers.

### The problem this solves (visually)

```
Shape: x[512, 8192] bf16
Operation: x / x.sum(dim=-1, keepdim=True)

─── Two-pass reload (no persistent) ──────────────────────────────────
Loop 1 (reduction): iterate r in [0, 8192), accumulate sum
  Each iteration: load x[row, r] from GMEM, add to acc
  ✗ No reuse — x must be re-read in epilogue

Loop 2 (epilogue): iterate r in [0, 8192), divide
  Each iteration: RELOAD x[row, r] from GMEM, divide by sum, store
  ✗ 2× memory traffic — 8192 elements loaded twice per row

Total memory loads: 2 × 512 × 8192 = 8,388,608 elements

─── Register-tiled persistent (2 tiles of 4096) ──────────────────────
Loop 1 (reduction): _tile in [0, 2)
  _tile=0: load x[row, 0:4096], stash in _retained_0, accumulate
  _tile=1: load x[row, 4096:8192], stash in _retained_1, accumulate
  → sum complete, x is retained in registers

Loop 2 (epilogue): _tile in [0, 2)
  _tile=0: retrieve _retained_0 (no GMEM load!), divide, store
  _tile=1: retrieve _retained_1 (no GMEM load!), divide, store

Total memory loads: 1 × 512 × 8192 = 4,194,304 elements (2× savings)
```

### Data Flow Walkthrough

#### 1. Kernel Construction (`SIMDKernel.__init__` + `TritonKernel.__init__`)

```
SIMDKernel.__init__()
  → persistent_reduction = False  (rnumel=8192 > default persistent threshold)
  → register-tiled eligibility check:
      ✓ not cooperative_reduction
      ✓ features.is_reduction()
      ✓ features.get_reduction_hint() == ReductionHint.INNER
      → get_reg_cached_persistent_reduction_config()
        ✓ feature flag enabled
        ✓ not HIP
        ✓ exactly 1 reduction + 1 pointwise in node_schedule
        ✓ shared_read_names = {arg0_1} (both nodes read x)
        ✓ rnumel = 8192 (static integer)
        ✓ 8192 ≥ min_numel(2048)
        ✓ 8192 % 4 == 0 → num_tiles = 4
        → returns PersistentReductionTileConfig(
              num_tiles=4, rnumel=8192,
              shared_read_names=("arg0_1",))
      → overrides persistent_reduction = True
      → sets num_persistent_tiles = 4
      → persistent_shared_read_names = ("arg0_1",)

TritonKernel.__init__()
  → _persistent_tile_phase = "reduction"  (num_persistent_tiles > 1)
  → _retained_loads_current = {}
  → _retained_load_var_names = {}
```

#### 2. Node Schedule

The scheduler builds:
```
node_schedule = [op0, DisableReduction, EnableReduction, op1]
```

The `DisableReduction`/`EnableReduction` pair is adjacent — it marks the
boundary between reduction and epilogue phases, flushing the reduction code.
The epilogue node (op1) codegens AFTER `EnableReduction` restores
`inside_reduction=True`.

#### 3. Code Generation (Second Pass)

**Step A — op0 (sum reduction) codegens:**
```
inside_reduction = True, phase = "reduction"

V.ops.load("arg0_1", index):
  → dtype is bf16, phase is "reduction", name in shared_read_names
  → _retain_native = True
  → Emits into staging buffers:
      loads: tmp0 = tl.load(in_ptr0 + (...), xmask)  # bf16, raw
             tmp1 = tmp0.to(tl.float32)               # upcast for accumulation
  → Stashes: _retained_loads_current["arg0_1"] = tmp0  (the bf16 CSE var)

V.ops.reduction("sum", ...):
  → Emits accumulator logic into staging buffers
```

**Step B — DisableReduction → `codegen_body()` (FLUSH #1):**
```
inside_reduction = True, phase = "reduction", staging has content ✓
→ Enters register-tiled branch (num_persistent_tiles > 1)

Emits into kernel.body:
  - Pre-declare _retained_arg0_1_{0..3} variables
  - for _tile in tl.static_range(NUM_TILES):
      [spliced reduction code]
      [retained-load stash: if _tile == i: _retained_arg0_1_i = tmp0]
  - [post_loop_combine: tmp3 = tl.sum(_tmp3, 1)[:, None]]

Sets: _persistent_tile_phase = "epilogue"
Clears: staging buffers, _retained_loads_current
```

**Step C — EnableReduction → restores `inside_reduction=True`**

**Step D — op1 (division epilogue) codegens:**
```
inside_reduction = True, phase = "epilogue"

V.ops.load("arg0_1", index):
  → phase is "epilogue", name in persistent_shared_read_names
  → Does NOT emit tl.load!
  → Returns CSE var wrapping: _forwarded_arg0_1
  → (Value comes from retained-load retrieval in the tile loop)

V.ops.truediv(...) + V.ops.store(...):
  → Emits epilogue compute + store into staging buffers
```

**Step E — `codegen_kernel()` → `codegen_body()` (final flush):**
```
inside_reduction = True, phase = "epilogue", staging has content ✓
→ Enters register-tiled branch

Emits into kernel.body:
  for _tile in tl.static_range(NUM_TILES):
      [retained-load retrieval: if _tile == i: _forwarded = _retained_i]
      [spliced epilogue code]
```

#### 4. Summary of `codegen_body()` calls

| # | Trigger | `inside_reduction` | `phase` | Has content | Action |
|---|---------|-------------------|---------|-------------|--------|
| 1 | First pass DisableReduction | True | reduction | No | Early return |
| 2 | First pass EnableReduction | False | reduction | No | Early return |
| 3 | Second pass DisableReduction | **True** | **reduction** | **Yes** | Emits reduction tile loop |
| 4 | Second pass EnableReduction | False | epilogue | No | Early return |
| 5 | codegen_kernel() final | **True** | **epilogue** | **Yes** | Emits epilogue tile loop |

Only calls #3 and #5 produce output.  Both use `_persistent_tile_phase`
to determine whether to emit stash (reduction) or retrieval (epilogue)
code around the spliced staging buffers.

#### 5. Key Hooks in V.ops

**`load(name, index)`** — three behaviors depending on phase:
- Normal: emit `tl.load(...)` as usual
- Reduction + shared read + bf16: emit raw bf16 load as separate CSE var,
  stash ref in `_retained_loads_current[name]`, then emit `.to(tl.float32)` upcast
- Epilogue + shared read: return CSE var wrapping `_forwarded_{name}` (no tl.load)

**`reduction()`** — for register-tiled, takes the non-persistent accumulator
path (init + combine + finalize) instead of the persistent mask-and-reduce
path.  The `tl.static_range` loop handles tile repetition.

### Generated Kernel (actual output for `[512, 8192]`)

```python
@triton.jit
def triton_per_fused_div_sum_0(in_ptr0, out_ptr1, xnumel, r0_numel,
                                XBLOCK: tl.constexpr,
                                R0_BLOCK: tl.constexpr,
                                NUM_TILES: tl.constexpr):
    xnumel = 512
    r0_numel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex

    # ─── Accumulator + retained-load declarations ───────────────
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tl.static_assert(NUM_TILES <= 4, 'NUM_TILES must be <= 4')
    _retained_arg0_1_0 = tl.full([XBLOCK, R0_BLOCK], 0.0, tl.bfloat16)
    _retained_arg0_1_1 = tl.full([XBLOCK, R0_BLOCK], 0.0, tl.bfloat16)
    _retained_arg0_1_2 = tl.full([XBLOCK, R0_BLOCK], 0.0, tl.bfloat16)
    _retained_arg0_1_3 = tl.full([XBLOCK, R0_BLOCK], 0.0, tl.bfloat16)

    # ─── Loop 1: Reduction (load from GMEM, retain in registers) ──
    for _tile in tl.static_range(NUM_TILES):
        r0_offset = _tile * R0_BLOCK
        r0_index = r0_offset + tl.arange(0, R0_BLOCK)[None, :]
        r0_1 = r0_index

        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), xmask, other=0.0)  # bf16
        tmp1 = tmp0.to(tl.float32)                                      # upcast
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(xmask, tmp4, _tmp3)

        # Stash bf16 input for epilogue reuse (stays in registers)
        if _tile == 0: _retained_arg0_1_0 = tmp0
        if _tile == 1: _retained_arg0_1_1 = tmp0
        if _tile == 2: _retained_arg0_1_2 = tmp0
        if _tile == 3: _retained_arg0_1_3 = tmp0

    # ─── Cross-tile finalization ───────────────────────────────
    tmp3 = tl.sum(_tmp3, 1)[:, None]    # row sum: [XBLOCK, 1]

    # ─── Loop 2: Epilogue (retrieve from registers, no GMEM reload) ─
    for _tile in tl.static_range(NUM_TILES):
        r0_offset = _tile * R0_BLOCK
        r0_index = r0_offset + tl.arange(0, R0_BLOCK)[None, :]
        r0_1 = r0_index

        # Retrieve stashed bf16 input (no global memory load!)
        if _tile == 0: _forwarded_arg0_1 = _retained_arg0_1_0
        if _tile == 1: _forwarded_arg0_1 = _retained_arg0_1_1
        if _tile == 2: _forwarded_arg0_1 = _retained_arg0_1_2
        if _tile == 3: _forwarded_arg0_1 = _retained_arg0_1_3

        tmp5 = _forwarded_arg0_1
        tmp6 = (tmp5 / tmp3).to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp6, xmask)
```

`tl.static_range` fully unrolls at compile time.  When `NUM_TILES=2`,
the compiler eliminates `_tile == 2` and `_tile == 3` branches and
their associated `_retained_2/3` variables (dead code elimination).

### Autotuning

`R0_BLOCK` and `NUM_TILES` are constexpr function parameters.  The
`persistent_reduction` heuristic generates configs with all valid pairs:

```
rnumel=8192, max_tiles=4, min_tiles=1:
  {R0_BLOCK: 2048, NUM_TILES: 4, num_warps: 4}
  {R0_BLOCK: 2048, NUM_TILES: 4, num_warps: 8}
  {R0_BLOCK: 4096, NUM_TILES: 2, num_warps: 4}
  {R0_BLOCK: 4096, NUM_TILES: 2, num_warps: 8}
  ...
```

`num_warps` is varied because tiled kernels have smaller `R0_BLOCK`,
so fewer warps may be optimal (4 warps at R0_BLOCK=4096 is 17% faster
than 16 warps on GB200).

## Eligibility Gates

1. `config.triton.register_tiled_persistent_reductions` enabled (default off)
2. CUDA only (not HIP)
3. Inner reduction (`ReductionHint.INNER`)
4. Not cooperative reduction
5. Simple fused schedule: exactly one reduction node + one pointwise (epilogue) node
6. At least one shared source read between reduction and epilogue
7. Static integer rnumel
8. `rnumel >= min_numel` (default 2048)
9. `rnumel` evenly divisible by at least one tile count in `[1, max_tiles]`

## Key Design Decisions

**Retained loads in native dtype.**  For bf16 inputs, the stashed values
are bf16 (2 bytes per element vs 4 for fp32), halving retained-load
register usage.  The upcast to fp32 happens separately in both phases.

**`NUM_TILES` is autotunable.**  Retained-load variables are pre-declared
for `max_tiles`.  Unused tiles are dead-code-eliminated by the Triton
compiler since `tl.static_range` unrolls at compile time.

**`tl.static_range` loops, not unrolled codegen.**  Each node is codegen'd
once.  The loop handles tile repetition.  PTX is identical to hand-unrolled
(confirmed via triton.compile comparison).

## Files Changed

| File | Role |
|------|------|
| `config.py` | Feature gate + max_tiles, min_tiles, min_numel configs |
| `simd_kernel_features.py` | `PersistentReductionTileConfig`; schedule analysis, shared-read detection, tile geometry |
| `simd.py` | Register-tiled eligibility check in `SIMDKernel.__init__`; `num_persistent_tiles` fields |
| `triton.py` | `codegen_body()` tile loops; `load()` epilogue forwarding + bf16 stash; `to_dtype` redundant cast fix |
| `triton_heuristics.py` | Generate `R0_BLOCK`/`NUM_TILES`/`num_warps` autotuning configs |
| `test_register_tiled_persistent_reduction.py` | 16 tests: fp32/bf16 correctness, codegen assertions, fallbacks |

## Performance

GB200, RMSNorm (`x.pow(2).mean().rsqrt() * w`), bfloat16, max_tiles=8, peak BW=7928 GB/s.
Generated by `bench_register_tiled.py`.

### Speedup (register-tiled / baseline)

| M \ N |  2048 |  4096 |  8192 | 16384 |
|------:|------:|------:|------:|------:|
|  2048 |  0.81 |  0.80 |  1.16 |  1.25 |
|  4096 |  1.09 |  0.96 |  0.93 |  1.25 |
|  8192 |  1.29 |  1.16 |  1.08 |  1.16 |
| 16384 |  1.15 |  1.10 |  1.07 |  1.12 |
| 32768 |  1.13 |  1.07 |  1.05 |  1.06 |

### SOL% (baseline → register-tiled)

| M \ N |     2048 |     4096 |     8192 |    16384 |
|------:|---------:|---------:|---------:|---------:|
|  2048 | 50→41%   | 62→50%   | 67→78%   | 58→73%   |
|  4096 | 62→67%   | 67→64%   | 76→71%   | 67→84%   |
|  8192 | 66→85%   | 74→86%   | 82→88%   | 77→89%   |
| 16384 | 73→84%   | 81→89%   | 85→91%   | 81→91%   |
| 32768 | 78→89%   | 86→92%   | 88→92%   | 85→90%   |

### Kernel Latency (us): baseline → register-tiled

| M \ N |        2048 |        4096 |        8192 |       16384 |
|------:|------------:|------------:|------------:|------------:|
|  2048 |   4.2→5.2   |   6.8→8.5   |  12.6→10.9  |  29.3→23.4  |
|  4096 |   6.9→6.3   |  12.7→13.2  |  22.4→24.0  |  50.7→40.4  |
|  8192 |  12.8→9.9   |  22.8→19.7  |  41.6→38.4  |  87.7→75.8  |
| 16384 |  23.2→20.2  |  41.8→38.0  |  79.6→74.3  | 166.4→148.6 |
| 32768 |  43.3→38.2  |  78.6→73.8  | 154.3→147.3 | 320.5→301.3 |

Key observations:
- Best gains at large N with large M: up to 1.29x at [8192, 2048] and 1.25x
  at [2048/4096, 16384]
- **Regression at M=2048 with N=2048-4096**: register-tiling hurts when
  M is too small — few rows means low occupancy with the tile overhead
- **Regression at M=4096 with N=4096-8192**: autotuner picks suboptimal
  configs in this range
- For M ≥ 8192, register-tiled is consistently faster across all N values
  (1.05–1.29x)
- Peak SOL reaches 92% at large M (vs 88% baseline)
- The feature should gate on M as well as N to avoid small-batch regressions

### Note on `run_reg_tiled_benchmark.py` measurement

The earlier `run_reg_tiled_benchmark.py` reported ~98% SOL for hand-written
kernels, which was inflated.  The original `benchmark_with_profiler` ran all
trials inside a single `with profile()` block without an explicit
`torch.cuda.synchronize()` between the L2 flush and the kernel launch.
This meant some iterations hit warm L2 cache despite the flush attempt,
understating kernel latency.

After fixing to per-trial profiling with explicit sync between L2 flush and
kernel (matching `bench_register_tiled.py`'s methodology), the hand-written
kernel measures **91% SOL** — not 98%.  All numbers in this document use the
corrected per-trial methodology.

## Future Work

### 1. Retain computed intermediates instead of raw inputs

**Problem**: For `h = x + residual; rmsnorm(h, w)`, the current implementation
retains raw inputs (`x`, `r`) in bf16 registers and **recomputes**
`h = x.to(f32) + r.to(f32)` in the epilogue. This is suboptimal when:
- The intermediate is expensive to recompute (`tanh`, `gelu`, complex expressions)
- There are many inputs (e.g., `h = a + b + c + d` — retaining 4 bf16 inputs uses
  8B per element vs 4B for one f32 intermediate)

Currently guarded to <= 2 shared inputs to avoid register pressure until this is
implemented.

**How the three approaches compare:**

```
─── Two-pass reload (baseline) ────────────────────────────────────────
Loop 1: load x,r from GMEM → compute h → reduce
Loop 2: RELOAD x,r from GMEM → RECOMPUTE h → epilogue
Cost: 2× GMEM for x,r + 2× compute h

─── Register-tiled (current) ──────────────────────────────────────────
Loop 1: load x,r from GMEM → retain x,r in regs → compute h → reduce
Loop 2: retrieve x,r from regs → RECOMPUTE h → epilogue
Cost: 1× GMEM for x,r + 2× compute h (saves GMEM, still recomputes)

─── Register-tiled with intermediate retention (future) ───────────────
Loop 1: load x,r from GMEM → compute h → retain h in regs → reduce
Loop 2: retrieve h from regs → epilogue
Cost: 1× GMEM for x,r + 1× compute h (saves both GMEM and recompute)
```

**Why it's hard**: The IR does not create a shared intermediate buffer for `h`.
Both the reduction and epilogue nodes independently compute `h` from raw inputs.
Requires a **scheduler-level common subexpression hoisting** pass to:
1. Detect both nodes compute the same expression from the same inputs
2. Create a `ComputedBuffer` for that expression
3. Rewrite both nodes to load from that buffer
4. Register-tiled codegen then retains the buffer per-tile

### 2. Mixed-precision inline PTX (SM80+)

**Motivation**: PTX provides instructions that take bf16/f16 inputs and produce
f32 outputs directly, eliminating explicit conversion:

| Instruction | Operation | Syntax | Constraints |
|---|---|---|---|
| `add.rn.f32.bf16` | a + b → f32 | `add.rn.f32.bf16 %f0, %h1, %h2;` | `"=f,h,h"` |
| `sub.rn.f32.bf16` | a - b → f32 | `sub.rn.f32.bf16 %f0, %h1, %h2;` | `"=f,h,h"` |
| `fma.rn.f32.bf16` | a * b + c → f32 | `fma.rn.f32.bf16 %f0, %h1, %h2, %f3;` | `"=f,h,h,f"` |

Available since SM80 (Ampere). For RMSNorm variance (`x.pow(2)`),
`fma.rn.f32.bf16` computes bf16×bf16→f32 in one instruction vs two.

**Implementation** (on branch `register-tiled-persistent-reduction-ptx`):
- Pattern match: `aten.pow.Tensor_Scalar(x_bf16, 2)` → `fma_bf16_to_f32(x, x)`
- Requires `codegen_upcast_to_fp32=False` so loads stay in native bf16
- Lowering emits `tl.inline_asm_elementwise` with `"=f,h,h"` constraints

**Benchmark (GB200, RMSNorm bf16)**: Up to 1.13x at small M (2048-4096) where
kernel is less memory-bound. Neutral at large M.

**Open issues**:
- `codegen_upcast_to_fp32=False` affects ALL ops, causing numerical differences
  for precision-sensitive ops (`rsqrt`, `libdevice`). Need selective per-load
  upcast control.
- Pattern matching timing: `_misc_patterns_init` must be called from
  `post_grad.lazy_init` to fire on the inference path.
- Tests need relaxed tolerances or explicit f32 casts for precision-sensitive ops.
