# ISSUE-009: Prototype output-cell ownership with Cartesian bins

**Created:** 2026-10-21
**Priority:** P1
**Status:** In progress
**Labels:** wayfinder:prototype
**Started:** 2026-10-22
**Completed:** _not completed_

**Related commits:**
- ee88a2fc — v11as block-shared staging + measured dispatch
- 92014e4b — v11at preprocessing rewrite (packed gather + GPU binstart)
- 674ad159 — v11au three-point coefficient packing

**Dependencies:**
- ISSUE-002
- ISSUE-004
- ISSUE-006
- ISSUE-007
- ISSUE-008

## Description

Build the smallest standalone CUDA 3D type-1 FP32 prototype that assigns each fine-grid
cell to exactly one CUDA thread, with one CUDA threadblock owning the tile, and accumulates
contributions from spatially binned points, eliminating the global atomic merge. Method 3 already avoids shared atomics (it uses
a non-atomic shared RMW plus one barrier per point); this experiment isolates what output
ownership buys over method 3's barrier-per-point local accumulation and per-tile global
merge. Use the existing Cartesian bin ordering (or a simple dense bin table) so the result
is not confounded with Morton ordering.

The prototype should:

- live under `devel/morton_spread/` with a README, build command, and benchmark script; it
  must not change the library build;
- reproduce the fine-grid spread output of the current cuFINUFFT path, using
  `gpu_spreadinterponly=1` as the reference and the correctness contract from
  [Pin down the spread correctness contract for an output-owned kernel](../done/ISSUE-004-spread-correctness-contract.md);
- use FP32, `ns=6`, and the geometry chosen in
  [Choose the bin size, tile size, and sorted-point lookup](../done/ISSUE-008-bin-tile-and-sorted-point-lookup.md);
- validate on the mandatory workloads from
  [Settle the prototype success protocol and scope](../done/ISSUE-006-prototype-success-protocol.md)
  and report relative L2 error, spread-stage time, effective throughput, occupancy,
  registers per thread, shared memory per CUDA threadblock, and where time goes;
- benchmark against `gpu_method=3` and record whether the atomic-free output ownership is
  already enough to move the needle;
- end with a short recommendation for Experiment B: what to keep, what to change, and
  whether Morton ordering is still worth testing.

Link the prototype artifact, commands, results, and recommendation from `## Answer`.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
- 2026-10-22 — Claimed by the wayfinder session to build and measure the Experiment A prototype.
- 2026-10-22 — Prototype built under `devel/morton_spread/`, validated against method 3, measured on the
  mandatory workloads; results in `docs/issues/research/ISSUE-009-exp-a-results.md`; resolved with a
  no-benefit finding and a recommendation not to run Experiment B as scoped.
- 2026-10-22 18:09 — Reopened at the user's direction: the working session was closed by accident,
  not the issue. The issue is still active while Experiment A is pushed further; the prior no-go
  findings remain in the Answer sections as history. Status assessment and fresh measurements
  in progress.
- 2026-10-22 18:12 — Fresh warm re-measurement after reopening, best configuration per workload:
  uniform `1e6` (`qb=1`, radix) spread 16.77 vs method-3 15.96 ms (1.05x slower), total 22.99 vs
  16.77 (1.37x slower); clustered `1e6` (`qb=4`, counting, split 8) spread 25.76 vs 15.89
  (1.62x slower), total 27.48 vs 16.80 (1.64x slower); sparse `1e5` `128^3` (`qb=4`, radix)
  spread 2.06 vs 2.82 ms (1.37x faster), total 2.46 vs 2.95 (1.18x faster). `relL2`
  5.5e-7 / 6.9e-7 / 6.8e-8. Two of three distributions are competitive on the spread stage
  (uniform at parity, sparse faster); clustered remains the outlier, and the uniform total is
  blocked by 6.2 ms preprocessing vs method 3's 0.8 ms `setpts`. Existing v10 profiling in
  `profile-artifacts/expA-output-owned/v10/` still matches these timings; no re-profile needed yet.
- 2026-10-22 18:47 — Tried the `scan_point` optimizations incrementally; artifacts and binaries are
  under `profile-artifacts/expA-output-owned/opt-2026-10-22/`. Benchmark medians (4 process runs,
  best config per workload): baseline v10 uniform 16.77 / clustered 25.88 / sparse 2.076 ms.
  v11a (delete the dead `d >= nfd` correction): 15.91 / 24.88 / 1.989 ms (-5.1% / -3.8% / -4.2%),
  ncu instructions uniform 917.3M (from 965.5M). v11b (v11a + non-wrapping min/max
  `dim_support`): 16.14 / 24.49 / 1.932 ms; support-placement instructions fell 262M -> 139M
  (-47%), but the per-point branch added scan-body/control-flow instructions and uniform time
  regressed 1.4% vs v11a, so it helps clustered/sparse only. v11c (v11b + min/max slow path):
  16.33 / 24.23 / 1.938 ms, uniform worse. v11d (v11a + warp relevance filter): 17.43 / 28.06 /
  2.039 ms, instructions 972.6M, rejected. v11e (fast path enabled only when `nsplit > 1`):
  17.24 / 24.31 / 2.172 ms and 990.7M uniform ncu instructions, rejected; the branch/code
  duplication costs even when disabled.
  Kept v11a as the working source (`devel/morton_spread/exp_a_cartesian.cu`), the only clean win.
  The 47% support-instruction cut is real but needs warp/tile-level specialization rather than a
  per-point branch to show up in uniform time.
- 2026-10-22 22:22 — Tried the five-task `stage_coeff` parallelization (v11n): five
  `(point, dimension)` tasks per warp iteration, 30 lanes, 9 iterations instead of 14 for
  BATCH=14. Registers fell 96 -> 94, but the task-mapping arithmetic added instructions:
  uniform 749.2M -> 760.9M, clustered 1111.4M -> 1127.4M. Interleaved A/B versus v11m:
  uniform 14.515 vs 14.505 ms (flat), clustered 21.359 vs 21.119 (1.1% worse), sparse 1.760
  vs 1.815 (3.1% better). Rejected; reverted to v11m. The coefficient pass is not purely
  lane-utilization-bound, and the lower register count does not change occupancy because
  shared memory still caps at 5 blocks. The next meaningful kernel lever is occupancy
  (BATCH=12 + launch_bounds(128,6), if registers can reach <=85) or preprocessing; the
  shared-memory LDS dependency remains the scan cost, but the easy prefetch and coefficient
  parallelism experiments are now exhausted.
- 2026-10-22 23:17 — Option A prototype (v11p) rejected. It kept raw-point staging but removed
  coefficient staging: lanes 0..17 compute each point's weights in registers, and the scan
  lanes fetch them with 8 `__shfl_sync` coefficient shuffles plus 3 support shuffles. It is
  correct (`relL2` unchanged) but much slower: uniform 19.4 / clustered 28.3 / sparse 2.45 ms,
  107 registers and 4 blocks/SM. ncu instructions rose to 968.4M uniform (v11m 749.2M) and
  1440.3M clustered (v11m 1111.4M). The shuffles and fused weight computation cost more than
  the shared loads they replaced, and the batch-level fast path/double-buffered
  coefficient/scan overlap was lost. Reverted to v11m. The register/shuffle path is closed on
  sm_75; remaining options are preprocessing or the scatter/shared-accumulation redesign.
- 2026-10-22 22:30 — Occupancy experiment (v11o) failed. `BATCH=12` + `__launch_bounds__(128, 6)`
  reached 80 registers and 6 blocks/SM (10,752 B/block), but ptxas spilled 56 B stores / 60 B
  loads per thread. Active warps rose 4.20 -> 4.74, eligible warps stayed ~1.42, no-eligible
  rose to 35.9%. Benchmark: uniform 14.95, clustered 22.05, sparse 1.93 ms (v11m: 14.45 /
  21.14 / 1.81). Rejected; reverted to v11m. Both remaining cheap kernel levers (five-task
  stage_coeff and 6-block occupancy) are now measured and rejected. The shared-memory LDS
  dependency remains, but the next moves are structural (block-level staging / larger warp
  region) or preprocessing, not more micro-tuning.
- 2026-10-22 22:07 — Session progress summary. Starting point v10: uniform 16.77 /
  clustered 25.88 / sparse 2.076 ms, gate failing. Current best **v11m**: uniform **14.45** /
  clustered **21.14** / sparse **1.81** ms. Uniform spread is now 10% faster than method 3;
  clustered is still 1.33x slower; uniform total is 1.23x because preprocessing is 6.2 ms
  vs method 3's 0.8 ms `setpts`. `relL2` 5.49e-7 / ~7.1e-7 / 6.76e-8; 96 registers, 5
  blocks/SM.

  Kept changes:
  1. v11a — delete the dead second wrap correction.
  2. v11f — one `__all_sync` per staged batch, then `scan_point<true>` (non-wrapping min/max)
     or `scan_point<false>` (general periodic), so the fast path is outside the point loop.
  3. v11i — store support starts as ints; remove the `LDS -> F2I` chain.
  4. v11m — padded coefficient layout (`STAGE=28`, `BATCH=14`): support ints 0–2, strength
     3–4, x 5–10 + zero 11, y 12–17 + zero 18, z 19–24 + zeros 25–27. The padding removes
     the `nx/ny/nz` coefficient selects; BATCH=14 keeps 5 blocks/SM (12,544 B/block).

  Rejected after measurement: v11b per-point fast path (uniform regression), v11c min/max
  slow path, v11d per-point relevance filter, v11e `nsplit` hybrid, v11g `STAGE=26`
  (occupancy 5 -> 4), v11h `STAGE=25` (misaligned `float2`), v11j/v11k support-start
  prefetch (102 regs/4 blocks; launch-bounds version still slower than v11i in A/B).

  v11m ncu: uniform 14.54 ms / 749.2M instructions (v10: 16.59 ms / 965.5M), clustered
  20.64 ms / 1111.4M (v10: 25.79 / 1514.0M). Current shares: `scan_point` 48.5%,
  `stage_coeff` 30.0%, `stage_raw` 5.5%; support placement within scan is ~11.9%.

  Next low-hanging fruit:
  - **`stage_coeff` parallelization**: it is still 30% of the kernel and runs one point per
    warp iteration with 18/32 lanes active. Processing five `(point, dimension)` tasks per
    iteration (30 lanes, `ci = lane % 6`) should cut the coefficient-pass warp iterations by
    ~1.5x for BATCH=14. This is the cleanest remaining kernel win.
  - **Preprocessing** is now the uniform total blocker: 6.2 ms vs method 3's 0.8 ms. At the
    current 14.45 ms spread, the 5% total guard requires preprocessing <= ~3.1 ms, so radix
    sort/fusion or a cheaper counting sort is needed for the gate.
  - **Clustered** remains 1.33x slower; improving scan/stage helps, but the split/reduction
    and clustered per-point work are still open.
  - **Occupancy** is the remaining scan-side lever: BATCH=12 + `__launch_bounds__(128, 6)`
    would fit 6 blocks only if registers drop to <=85; explicit s-prefetch is exhausted.
  The shared-memory LDS dependency is still present in `scan_point`, but it is no longer the
  easiest lever: the coefficient selects are gone, and the explicit prefetch experiments
  showed the compiler already schedules the two scans. Occupancy or structural reread
  reduction are the next moves there.

  Artifacts: `profile-artifacts/expA-output-owned/opt-2026-10-22/` (`v11m/` current,
  `ab_v11i_v11m.txt`); binaries `build/gpu/exp_a_cartesian_v11{a..m}`; the profile tool
  ranges and v11m flamegraphs/calltrees are updated.
- 2026-10-22 22:02 — Zero-padded coefficient layout (v11m) is a large win. The row layout is now
  3 support ints + 2 strength floats + 23 padded coefficient slots = STAGE=28; the padding lets
  `scan_point` load `kx1`, `ky1`, and `kz1..kz3` unconditionally, removing the `nx/ny/nz`
  range selects. To keep 5 blocks/SM, BATCH drops to 14 (12,544 B/block). BATCH=14 alone
  (v11l) regressed clustered to 23.14 ms, but the padded layout more than recovered it:
  uniform 14.45 / clustered 21.14 / sparse 1.81 ms (vs v11i 15.53 / 22.61 / 1.90 and v10
  16.77 / 25.88 / 2.08). ncu uniform 14.54 ms / 749.2M instructions, clustered 20.64 ms /
  1111.4M. `relL2` unchanged (5.49e-7 / ~7.1e-7 / 6.76e-8). 96 registers, 5 blocks/SM.
  v11m is the working source; the profile tool ranges and v11m flamegraphs/calltrees are
  updated.
- 2026-10-22 21:05 — Tried the shared-load software pipeline. v11j preloaded both points'
  support starts before scanning either; it used 102 registers and dropped to 4 blocks/SM,
  regressing to 16.55 / 23.90 / 2.078 ms. v11k added `__launch_bounds__(128, 5)`, recovering
  96 registers and 5 blocks. Interleaved A/B still showed v11k slower than v11i: uniform
  15.418 vs 15.357 ms, clustered 22.784 vs 22.628 ms. The compiler already overlaps the
  two-point scans; explicit prefetch only adds instructions. Reverted to v11i. The remaining
  shared-memory lever is occupancy (BATCH=14 + launch_bounds to reach 6 blocks/SM), not
  instruction-level prefetch.
- 2026-10-22 20:50 — STAGE tests and the s-as-int dependency change. STAGE=26 removes the
  staging-store bank conflict but drops occupancy 5 -> 4 blocks/SM and regresses to 17.36 /
  24.93 / 2.171 ms. STAGE=25 keeps 5 blocks but faults on the misaligned `float2` loads
  (`cudaEventSynchronize: misaligned address`); not usable without scalar loads. Reverted to
  STAGE=24. v11i (STAGE=24 + support starts stored as int, batch check reads ints, scan_point
  drops the `LDS -> F2I`) is the new best: 15.53 / 22.61 / 1.902 ms (vs v11f 15.65 / 22.99 /
  1.948, and vs v10 16.77 / 25.88 / 2.076). ncu uniform 15.39 ms / 812.8M instructions,
  clustered 22.46 ms / 1223.0M. `relL2` unchanged. The F2I is gone but the shared-load latency
  moves to the next consumer (`dim_support`/coefficient selects), so the dependency chain,
  not the conversion, is the remaining target. v11i is the working source.
- 2026-10-22 20:32 — Per-batch template specialization (v11f) succeeded and is the new working
  source. One `__all_sync` check per staged batch; the batch takes `scan_point<true>` only when
  the warp region and every staged support are non-wrapping. Benchmark medians: uniform 15.65
  (vs v11a 15.91, -1.6%; vs v10 16.77, -6.7%), clustered 22.99 (vs 24.88, -7.6%; vs 25.88,
  -11.2%), sparse 1.948 (vs 1.989, -2.1%; vs 2.076, -6.2%). ncu uniform 15.60 ms / 828.0M
  instructions (vs v11a 917.3M), clustered 22.56 / 1248.4M (vs 1442.0M). Support-placement
  instructions 262.4M -> 153.3M uniform and 392.0M -> 181.5M clustered; `scan_point` body stayed
  ~488M / ~806M, so the fast path finally pays. `relL2` unchanged (5.49e-7 / 6.8e-7 / 6.76e-8).
  Registers 96 (from 95), still 5 blocks/SM. Artifacts: `opt-2026-10-22/v11f/`.
- 2026-10-22 18:35 — Re-ran Nsight Compute on a fresh `-lineinfo` build. Counters match v10
  (uniform duration 16.59 ms, 0.966e9 instructions; clustered 25.79 ms, 1.514e9). Fixed
  `ncu_samples_to_chrome.py` to apply the static source call graph instead of treating every
  inlined function as a top-level sibling: `dim_support` now nests under `scan_point`,
  `fold_rescale`/`cudaFMA` under `stage_coeff`, `__ldg` under `stage_raw`, and the batch loop
  contains its callees. New artifacts:
  `profile-artifacts/expA-output-owned/reopen-2026-10-22/`. Corrected shares: `scan_point`
  58.6% (own 37.3%, `dim_support` 21.4%), `stage_coeff` 26.4% (own 19.1%, `fold_rescale` 7.3%),
  `stage_raw` 4.0%; batch-loop own 9.7%. Line hotspots: `dim_support` wrap arithmetic 9.6%,
  `j0`/`n` selects 6.5%; `stage_coeff` Horner 6.6%, fold 7.3%, support-start ceil 3.4%;
  `stage_raw` line 241 store 2.8% with 72% excessive shared wavefronts (STAGE=24 stride gives
  bank conflicts). This is a source-derived hierarchy, not a captured device call stack: ncu
  PC sampling is flat and the functions are inlined.


## Session log — 2026-10-22 (comprehensive)

This is the full optimization record for Experiment A. The working source is **v11m**;
the other variants are preserved under
`profile-artifacts/expA-output-owned/opt-2026-10-22/`.

### Starting point and gate

- v10 baseline: uniform `1e6` 16.77 ms, clustered `1e6` 25.88 ms, sparse `1e5` 2.076 ms
  spread; ncu 965.5M / 1514.0M instructions; 95 registers, 5 blocks/SM.
- Method 3: spread/total 15.95/16.78 ms uniform, 15.90/16.79 clustered, 2.82/2.95 sparse.
- Gate: >=20% faster spread on both mandatory `1e6` cases: <=12.76 ms uniform, <=12.72 ms
  clustered; total preprocessing+spread within 5%: <=17.62 ms uniform.
- Correctness: only pinned method-3 spread-only output is the oracle; `relL2` must stay in the
  FP32 reordering noise floor.

### Current best (v11m)

- Uniform 14.45 ms, clustered 21.14 ms, sparse 1.81 ms spread (4-run medians).
- Interleaved A/B v11i -> v11m: uniform 15.30 -> 14.53 ms (5.0%), clustered 22.64 -> 21.17
  (6.5%).
- ncu: uniform 14.54 ms / 749.2M instructions; clustered 20.64 ms / 1111.4M.
- 96 registers, 5 blocks/SM, 12,544 B shared/block.
- `relL2`: 5.49e-7 uniform, ~7.1e-7 clustered, 6.76e-8 sparse.
- Uniform spread is 10% faster than method 3; clustered is 1.33x slower; uniform total is
  1.23x slower because preprocessing is 6.2 ms vs method 3's 0.8 ms `setpts`.
- v11m layout: `STAGE=28`, `BATCH=14`: support ints 0-2, strength 3-4, x 5-10 + zero 11,
  y 12-17 + zero 18, z 19-24 + zeros 25-27. The padded slots are initialized once per warp
  buffer. The scan loads `kx1`, `ky1`, `kz1..kz3` unconditionally.

### Experiment table

Times are spread-stage medians (ms) at the best config. ncu instructions are uniform /
clustered (millions of warp instructions). "A/B" marks interleaved comparisons.

| Variant | Change | Uniform | Clustered | Sparse | Regs/blocks | ncu inst | Verdict |
|---|---|---:|---:|---:|---|---|---|
| v10 | baseline | 16.77 | 25.88 | 2.076 | 95/5 | 965.5/1514.0 | baseline |
| v11a | delete dead second wrap correction | 15.91 | 24.88 | 1.989 | 95/5 | 917.3/1442.0 | kept |
| v11b | per-point non-wrapping fast path | 16.14 | 24.49 | 1.932 | 95/5 | 877.4/1327.8 | rejected as a whole; uniform worse |
| v11c | v11b + min/max slow path | 16.33 | 24.23 | 1.938 | 95/5 | 881.3/1327.8 | rejected |
| v11d | v11a + per-point relevance filter | 17.43 | 28.06 | 2.039 | 96/5 | 972.6/1575.8 | rejected |
| v11e | fast path only when `nsplit > 1` | 17.24 | 24.31 | 2.172 | 95/5 | 990.7/1327.8 | rejected |
| v11f | per-batch `__all_sync` + `scan_point<true/false>` | 15.65 | 22.99 | 1.948 | 96/5 | 828.0/1248.4 | kept |
| v11g | `STAGE=26` | 17.36 | 24.93 | 2.171 | 96/4 | — | rejected; occupancy loss |
| v11h | `STAGE=25` | invalid (misaligned `float2`) | — | — | — | — | rejected |
| v11i | int support starts | 15.53 | 22.61 | 1.902 | 96/5 | 812.8/1223.0 | kept |
| v11j | prefetch both points' support starts | 16.55 | 23.90 | 2.078 | 102/4 | — | rejected; occupancy |
| v11k | v11j + `__launch_bounds__(128,5)` | 15.42 A/B | 22.78 A/B | — | 96/5 | 816.0/1231.2 | rejected; v11i faster in A/B |
| v11l | `BATCH=14` control (`STAGE=24`) | 15.46 | 23.14 | 1.921 | 96/5 | — | rejected; clustered regression |
| **v11m** | padded coefficients, `STAGE=28`, `BATCH=14` | **14.45** | **21.14** | **1.81** | **96/5** | **749.2/1111.4** | **kept/current** |
| v11n | five-task `stage_coeff` pass | 14.515 A/B | 21.359 A/B | 1.760 A/B | 94/5 | 760.9/1127.4 | rejected; flat/worse clustered |
| v11o | `BATCH=12` + `launch_bounds(128,6)` | 14.95 | 22.05 | 1.93 | 80/6 + spills | 756.0/1120.5 | rejected; spills |
| v11p | Option A: register weights + warp shuffles | 19.4 | 28.3 | 2.45 | 107/4 | 968.4/1440.3 | rejected |

### What each experiment showed

- **v11a** — `d -= nfd & -(d >= nfd)` is dead after the first wrap correction for valid
  `s,g0`. Removing it cut ~48M warp instructions and improved all three workloads. Kept.
- **v11b** — the non-wrapping min/max `dim_support` cut support-placement instructions
  262.4M -> 138.5M on uniform, but the per-point branch inflated the scan body/control flow
  (485.3M -> 562.4M), so uniform regressed 1.4% vs v11a. It helped clustered/sparse.
  Rejected as a whole; the fast path itself was kept for v11f.
- **v11c** — min/max slow path added instructions on uniform; rejected.
- **v11d** — per-point warp relevance filter added compares/branch/register pressure; it
  reduced support work but increased total instructions (972.6M) and regressed all cases.
- **v11e** — enabling the fast path only when `nsplit > 1` did not help; the branch/code
  duplication remained even when disabled.
- **v11f** — the key structural fix: one `__all_sync` per staged batch, then a single branch
  to `scan_point<true>` (min/max support) or `scan_point<false>` (general periodic). This
  kept the 47% support-instruction cut without inflating the per-point loop. Kept.
- **v11g/v11h** — `STAGE=26` removed the staging-store bank conflict but dropped occupancy
  5 -> 4 blocks and regressed. `STAGE=25` was conflict-free and kept 5 blocks, but the
  existing `float2` loads faulted with a misaligned address.
- **v11i** — storing support starts as ints removed the `LDS -> F2I -> dim_support` chain.
  Kept.
- **v11j/v11k** — preloading both points' support starts used 102 registers and dropped to
  4 blocks; `__launch_bounds__(128,5)` recovered 96/5 but the interleaved A/B was still slower
  than v11i. The compiler already overlaps the two-point scans.
- **v11l** — the `BATCH=14` control with `STAGE=24` regressed clustered to 23.14 ms; this was
  the control for v11m.
- **v11m** — padding the coefficient rows removed the `nx/ny/nz` range selects and shortened
  the dependency chain. It required `STAGE=28`; `BATCH=14` keeps 5 blocks. This more than
  recovered the batch-size loss. It is the current best.
- **v11n** — five `(point,dimension)` tasks per coefficient-pass iteration reduced registers
  to 94 and helped sparse, but the task-mapping arithmetic added instructions; uniform was
  flat and clustered 1.1% worse. The coefficient pass is not purely lane-utilization-bound.
- **v11o** — six-block occupancy at 80 registers with `BATCH=12` caused 56 B spill stores /
  60 B spill loads. Active warps rose 4.20 -> 4.74, eligible warps stayed ~1.42, and
  no-eligible rose to 35.9%; all workloads regressed. 96 registers / 5 blocks is the better
  balance.
- **v11p** — Option A: keep raw staging, remove coefficient staging, compute weights in
  lanes 0-17, distribute with 8 coefficient shuffles + 3 support shuffles per point. It is
  correct but much slower (uniform 19.4, clustered 28.3, sparse 2.45; 107 registers / 4
  blocks; ncu 968.4M / 1440.3M). The shuffles and fused weight computation cost more than
  the shared loads they replaced, and the batch-level fast decision/double-buffered overlap
  were lost. Closed on sm_75.

### Current profile and gate gap

- v11m uniform shares: `scan_point` 48.5%, `stage_coeff` 30.0%, `stage_raw` 5.5%; support
  placement within scan ~11.9%; inner accumulation ~16%.
- The shared loads are conflict-free/broadcast; the cost is latency/dependency (short
  scoreboard/wait), not bandwidth. The kernel is latency-bound: eligible warps/scheduler
  ~1.3-1.4, no-eligible ~34%.
- Uniform spread 14.45 vs gate target 12.76 (needs another ~1.7 ms). Clustered 21.14 vs
  12.72 (needs ~8.4 ms). Uniform total 20.69 vs 17.62; preprocessing must drop from 6.2 to
  <=3.1 ms at the current spread.
- The remaining cheap kernel micro-optimizations are exhausted: five-task coefficient pass,
  six-block occupancy, s-prefetch, per-point filters, stride changes all measured and
  rejected. The shared-memory/shuffle alternative was worse.

### Artifacts and tooling

- Source: `devel/morton_spread/exp_a_cartesian.cu` (v11m). Default binaries
  `build/gpu/exp_a_cartesian` and `build/gpu/exp_a_cartesian_lineinfo` are v11m.
- Experiments: `profile-artifacts/expA-output-owned/opt-2026-10-22/` (`v10base`, `v11a`-`v11p`,
  `README.md`, A/B logs `ab_v11i_v11k.txt`, `ab_v11i_v11m.txt`, `ab_v11m_v11n.txt`).
- ncu reports/source CSVs and details for the variants where captured; flamegraphs/calltrees
  for v11a, v11f, v11i, v11m. The profile tool `ncu_samples_to_chrome.py` has v11m ranges and
  the source-derived call hierarchy (`dim_support`/`dim_support_nowrap` under `scan_point`).
- No library code was modified; all work is in the standalone `devel/morton_spread/`
  prototype. Tracker/research notes are local.

### Lessons

- Per-point branching is expensive; batch/warp-uniform decisions pay.
- Row layout matters: padding + int support starts removed both selects and conversions.
- Occupancy cannot be bought with spills; 96 regs / 5 blocks is the current optimum.
- Shared-memory broadcasts are cheaper than warp shuffles for this access pattern on sm_75.
- The coefficient pass is not simply lane-underutilized; its latency/dependency structure
  matters.
- The current design is a local optimum: further gains need a different structure or
  preprocessing, not another micro-tweak.

### Next options

1. **Preprocessing** — highest value for the uniform total. 6.2 ms vs 0.8 ms; need <=3.1 ms
   at current spread. Radix/payload fusion, cheaper counting sort, or reuse of method 3's
   ordering.
2. **Option B: scatter/shared accumulation** — the remaining architectural alternative.
   Compute weights once per point and scatter into a shared tile; avoids the per-thread
   coefficient gather but reintroduces shared atomics/merge; likely more natural for
   clustered.
3. **Structural reread reduction** — larger warp regions or block-level union staging; bigger
   redesigns, not low-hanging.
4. **Clustered-specific split/reduction work** — clustered remains 1.33x slower.


### 2026-10-22 23:58 — exact stage_coeff micro-optimizations (v11q/v11r/v11s)

These are exact, value-preserving changes; no kernel/polynomial/coefficient approximation.

- **v11q**:
  - store the support start as an int directly (`sval = (int)ceilf(...)`), removing the
    float→int→float round trip;
  - hoist `cd`, `ci`, `nfd`, support offset, and coefficient offset out of the point loop;
  - move the `lane < 18` predicate outside the per-point coefficient loop;
  - A/B vs v11m: uniform **14.51 → 14.01 ms** (−3.4%), clustered **21.18 → 20.48 ms**
    (−3.3%), sparse **1.810 → 1.728 ms** (−4.5%);
  - ncu 720.7M uniform / 1068.6M clustered instructions; 96 registers, 5 blocks;
    `relL2` unchanged (5.49e-7 / 7.2e-7 / 6.76e-8).
- **v11r**: replaces `(int)ceilf(...)` with `__float2int_ru`. It is exact, but neutral:
  uniform 13.98, clustered 20.49, sparse 1.731 ms. Rejected; v11q remains the base.
- **v11s**: adds a pointer increment (`row += STAGE`) in the coefficient loop instead of
  recomputing `cur + b*STAGE`. Small but consistent win over v11q: uniform **13.97 → 13.87
  ms** in A/B, clustered **20.53 → 20.40 ms**, sparse **1.728 → 1.714 ms**. Full 4-run
  medians: **13.92 / 20.44 / 1.716 ms**. ncu 717.6M / 1064.0M; 96 registers / 5 blocks;
  unchanged `relL2`. **v11s is the current best.**

Current totals: uniform 20.15 ms, clustered 22.17 ms, sparse 2.130 ms. Uniform spread is
now 13% faster than method 3 (target 12.76 ms; gap ~1.16 ms); clustered is 1.28x slower
(target 12.72 ms). Uniform total is 1.20x slower; at the current spread, preprocessing
(6.24 ms) must fall to <=3.70 ms for the 5% total guard. v11s profile shares: `scan_point`
49.8%, `stage_coeff` 28.3%, `stage_raw` 5.6%.

Artifacts: `profile-artifacts/expA-output-owned/opt-2026-10-22/v11q`, `v11r`, `v11s`;
A/B logs `ab_v11m_v11q.txt`, `ab_v11q_v11r.txt`, `ab_v11q_v11s.txt`; v11s ncu reports/source
CSVs, Chrome flamegraphs, and calltrees. The profiler tool ranges/labels are updated to
v11s. The default binaries `build/gpu/exp_a_cartesian[_lineinfo]` are v11s.



### 2026-10-23 00:10 — float-ceil `zz` fix (v11t) and rejected two-point unroll (v11u)

- **v11t** keeps the float `ceilf` result (`sf`) for the `zz` computation and stores the
  int support start separately. This is exact: the original float path used the same
  integer-valued float for `zz`; v11t only removes the int→float round trip from the
  dependency chain. A/B vs v11s: uniform **13.95 → 13.83 ms**, clustered **20.43 → 20.34
  ms**, sparse **1.718 → 1.704 ms**. Full 4-run medians: **13.82 / 20.35 / 1.703 ms**
  (totals 20.06 / 22.07 / 2.118). ncu 717.6M / 1064.0M instructions; 96 registers /
  5 blocks; `relL2` unchanged. Stage_coeff share is 27.0%; scan_point 50.8%; stage_raw
  5.8%. **v11t is the current best.**
- **v11u** unrolled `stage_coeff` by two points. The ILP worked at the Horner level:
  wait samples 5,916 → 4,591 and Horner samples 16,458 → 13,814. But the two-call loop's
  control/address code grew from **6.99M to 18.88M** instructions; total uniform
  instructions rose **717.6M → 729.4M** (+1.65%), and clustered **1064.0M → 1081.7M**.
  It regressed to 14.13 / 20.78 / 1.713 ms despite staying at 96 registers / 5 blocks.
  Rejected. The simple unroll's loop/control overhead outweighs the ILP; a hand-written
  `stage_coeff2` with one shared control/address path might recover it, but it is no longer
  a quick win.

Current gate: uniform spread 13.82 ms vs target 12.76 (13% faster than method 3; needs 20%;
gap ~1.06 ms); clustered 20.35 vs 12.72 (1.28x slower). Uniform total 20.06 vs 17.62;
preprocessing 6.23 ms must fall to <=3.80 ms at the current spread.

Artifacts: `v11t/` and `v11u/`; A/B logs `ab_v11s_v11t.txt`, `ab_v11t_v11u.txt`; v11t ncu
reports/source CSVs, Chrome flamegraphs, and calltrees. Profiler ranges/labels are updated to
v11t; default binaries are v11t.



### 2026-10-23 00:21 — raw/strength-in-registers variants rejected (v11v/v11w)

- **v11v** issues the next batch's raw/strength global loads before `stage_coeff`, holds the
  five values in registers across the scan, and stores them to shared afterward. This hides
  the load latency under both passes in principle, but the five live values push the kernel
  from 96 to **101 registers**. On sm_75 the 5-block limit is effectively 96 registers after
  allocation granularity, so occupancy drops to **4 blocks/SM** (ncu active warps 4.20 →
  3.32). It regresses to 15.05 / 21.40 / 1.905 ms despite 0 spills.
- **v11w** shortens the live range: load before `stage_coeff`, store immediately after
  `stage_coeff`, before the scan. This stays at **96 registers / 5 blocks, 0 spills**, but
  is neutral to slightly worse (13.83 / 20.33 / 1.724 ms). The global-load stall is not
  hidden enough by `stage_coeff`; ncu still shows ~7.7k long-scoreboard samples, now at the
  deferred store. Rejected.
- Conclusion: the raw/strength-in-registers idea has no register headroom at the current
  96-register budget. The full shuffle version needs even more live values (raw x/y/z plus
  current/next strength) and would also fall to 4 blocks, so it is not viable without freeing
  registers elsewhere. v11t remains current best.

Artifacts: `v11v/`, `v11w/`, A/B logs `ab_v11t_v11v.txt`, `ab_v11t_v11w.txt`, ptxas logs,
and uniform ncu reports/source CSVs. v11t source/binaries restored.



### 2026-10-23 00:39 — clustered split-count tuning (v11x/v11y/v11z)

The old rule `while (nsplit < 8 && active_warps*nsplit < 512)` stopped clustered at 8
splits because its 64 active tile-warps already reached 512. Raising the target and testing
16/32/64 splits:

| split | clustered spread | total | note |
|---:|---:|---:|---|
| 8 (v11t) | 20.34 ms | 22.07 ms | baseline |
| 16 (v11x) | 19.13 ms | 20.85 ms | −6.0% |
| **32 (v11y)** | **18.75 ms** | **20.47 ms** | **−7.9%** |
| 64 (v11z) | 19.74 ms | 21.46 ms | reduction overhead |

v11y raises ncu active warps/scheduler from 4.22 to **4.70**, eligible warps from 1.47 to
**1.67**, and lowers no-eligible from 33.1% to **28.5%**. Uniform and sparse remain at
nsplit=1. Clustered `relL2` changes to ~2.6e-7 because the split summation order changes;
still well inside the noise floor. v11y is the current best.

The `active_warps=64/512` statistic is the number of tile-warps that contain points, not
runtime occupancy; nsplit turns those 64 tile-warps into 64×nsplit work warps. At 32 splits
the GPU is fed; at 64 the extra split/reduction work and empty splits outweigh the gain.

Current numbers: uniform **13.72 / 19.98 ms**, clustered **18.75 / 20.47 ms**, sparse
**1.701 / 2.119 ms**. Gate: uniform spread gap ~0.96 ms; clustered gap ~6.0 ms (1.18x slower
than method 3); uniform total gap ~2.36 ms; at the current spread preprocessing must fall
from 6.26 to <=3.90 ms.

Re-partitioning outlook: 32 splits is the knee for the current tile-based partition. The
clustered case still launches all 128 tiles × 32 splits, although only 16 tiles have points;
compacting active tiles or using a persistent work queue could recover some empty-block
overhead, but the dominant remaining cost is the **8.0× point reread**. Per-visit throughput
is already similar to uniform, so reaching the clustered gate requires reducing reread or
changing ownership (scatter/shared accumulation or block-level sharing), not just adding
more splits.

Artifacts: `v11x/`, `v11y/`, `v11z/`, A/B logs `ab_v11t_v11x.txt`, `ab_v11x_v11y.txt`;
v11y ncu reports/source CSVs, Chrome flamegraphs, and calltrees. Default binaries/source are
v11y.



### 2026-10-23 02:00 — active-tile compaction and honest metadata accounting (v11aa)

The current kernel parallelizes over **output tiles/warp regions**, not over bins. Each warp
enumerates all bins in its region+halo; empty bins are skipped at the x-run point-range check,
and empty tiles are still launched. `kernel_visits` computes per-warp visits/active flags on
device, but was outside the timed spread interval; the host copied the flags to choose
`nsplit`. That metadata was previously not counted.

v11aa makes the accounting explicit and compacts the active tile list:

- `kernel_visits` remains a separate launch-setup kernel; it cannot live inside
  `spread_cartesian_owned` because the host needs its result to choose `nsplit` and the grid.
- A device-side `kernel_compact_tiles` writes active tile IDs in deterministic tile order
  (single-threaded, no atomics).
- The spread kernel launches `active_tiles × nsplit` blocks and maps `blockIdx` through the
  tile list.
- `meta_ms` now times `kernel_visits` + compaction + the small D2H copies; `total` includes it:
  `total = preprocess + meta + spread`.

Results:

- clustered active tiles **16/128**; spread **18.75 → 18.07 ms**, total **20.47 → 19.87 ms**
  (using the honest setup accounting);
- uniform active tiles 128/128, sparse 1024/1024; grid unchanged;
- meta cost **0.51 ms uniform**, **0.08 clustered**, **0.19 sparse**. Uniform `kernel_visits`
  dominates because qb=1 makes each warp enumerate a much larger bin cube; this is now a
  visible optimization target.
- v11aa: 95 registers / 5 blocks; clustered `relL2` ~3.0e-7; no atomics in the spread kernel
  (compaction is a one-time metadata pass, single-threaded).

Honest current totals: uniform **13.83 / 20.57 ms**, clustered **18.07 / 19.87 ms**, sparse
**1.716 / 2.330 ms**. Gate: uniform spread gap ~1.07 ms; uniform total gap ~2.95 ms and
`preprocess + meta = 6.74 ms` must fall to <=3.79 ms; clustered spread gap ~5.35 ms, total
gap ~2.25 ms. The old v11y totals understated setup by ~0.51 ms uniform / ~0.19 ms sparse.

Artifacts: `v11aa/` source/bench, ncu reports/source CSVs, Chrome flamegraphs, and
calltrees; default binaries updated; opt README and main README updated. v11aa ncu:
uniform 13.76 ms / 717.6M instructions; clustered **17.21 ms / 1060.4M** (v11y 18.07 /
1086.8M), active warps/scheduler 4.73, eligible 1.71, no-eligible 27.6%. Clustered kernel
shares: stage_coeff 31.2%, scan_point 53.3%, stage_raw 3.1%.



### 2026-10-23 02:12 — parallel kernel_visits and metadata visibility (v11ab)

`kernel_visits` was profiled separately: uniform took **472 µs** with only **2 blocks / 512
threads**, active warps/scheduler 2.00, no-eligible 76.5%. The bin enumeration is not
intrinsically too expensive; it was parallelism-starved.

v11ab changes it to one **warp per warp region**: all 32 lanes split the flattened bin cube,
then warp-reduce the visits. Uniform `kernel_visits` drops to **0.313 ms**; compaction is
0.015 ms; total metadata **0.38 ms** (was 0.51). `visits`, `compact`, `meta`, `setup`, and
`total` are now printed separately, and NVTX ranges around both metadata kernels are captured
in nsys Chrome timelines.

Current v11ab medians: uniform **13.70 / 20.35 ms**, clustered **18.07 / 19.87 ms**,
sparse **1.710 / 2.337 ms**. Gate: uniform spread gap ~0.94 ms; uniform total gap ~2.73 ms
and `preprocess+meta=6.65 ms` must fall to <=3.92; clustered spread gap ~5.35 ms, total gap
~2.25 ms.

Remaining bin-enumeration ideas: use a 3D prefix/summed-area table of bin counts so each
warp cube is a few lookups instead of O(bins); use a block (not one warp) per region for more
parallelism; skip compaction entirely when all tiles are active. Uniform has essentially no
empty bins, so non-empty-bin skipping will not help there.

Artifacts: `v11ab/` source/bench; ncu source-level flamegraphs/calltrees
(`uniform_qb1_flamegraph_chrome.json`, `clustered_qb4_flamegraph_chrome.json`); nsys Chrome
timelines (`uniform_nsys_chrome.json`, `clustered_nsys_chrome.json`) that include
`kernel_visits`, `kernel_compact_tiles`, preprocessing, and spread; nsys-rep files; default
binaries updated. v11ab ncu: uniform 13.77 ms / 717.6M, clustered 17.37 ms / 1060.4M;
source shares uniform stage_coeff 27.0% / scan_point 50.7% / stage_raw 5.7%, clustered
31.4% / 53.1% / 3.1%.



### 2026-10-23 02:22 — 3D prefix/summed-area table rejected (v11ac/v11ad)

Tried replacing the O(bins) visits scan with a 3D summed-area table of bin counts:

- Query each warp cube with at most 8 periodic boxes → 64 prefix lookups.
- v11ac built the prefix with one thread per line: uniform `sat_build=0.242 ms`,
  `visits=0.019 ms`, `compact=0.015 ms`, meta **0.414 ms** (v11ab 0.38). Clustered
  sat=0.047, visits=0.029, meta **0.145 ms** (v11ab 0.08).
- v11ad used `cub::BlockScan` for each prefix line: uniform sat=0.286 ms,
  visits=0.018 ms, meta **0.471 ms** — worse; thousands of tiny scan blocks cost more
  than the serial lines.
- Conclusion: the **query** is fast (0.018 ms), but building the prefix dominates.
  SAT is not a win unless the prefix can be built during preprocessing or with a fused
  scan. Rejected; v11ab remains current.

Uniform metadata is now **0.38 ms** (parallel `kernel_visits` 0.31 + compaction 0.015 +
copies); the remaining uniform setup gap is the **6.2 ms Cartesian preprocessing**, not the
bin enumeration. v11ab source/binaries restored; nsys Chrome timelines include
`kernel_visits`, `kernel_compact_tiles`, preprocessing, and spread.



### 2026-10-23 02:28 — block-per-region kernel_visits sweep rejected (v11ae)

Swept one block per warp region at 32/64/128/256 threads (v11ab uses one warp per region):

| workload | warp/region | 64 threads | 128 threads |
|---|---:|---:|---:|
| uniform `visits` | 0.316 ms | **0.293 ms** | 0.296 ms |
| uniform `meta` | 0.385 ms | **0.363 ms** | 0.364 ms |
| clustered `visits` | 0.023 ms | 0.022 ms | 0.031 ms |
| sparse `visits` | **0.055 ms** | 0.079 ms | 0.108 ms |

The uniform gain is only ~0.02 ms; clustered is neutral; sparse regresses because 1024
regions each get a block and the reduction/launch overhead dominates. Rejected; v11ab’s
warp-per-region path remains current. The sweep is a dead end for reducing the metadata
further; the remaining uniform setup gap is the 6.2 ms preprocessing.




### 2026-10-23 03:03 — block-wide coefficient-sharing prototype rejected (v11ag)

Implemented the block-wide coefficient-sharing design: block-shared staged batch, distributed
`stage_coeff`, 4-bit per-warp relevance mask, `__syncthreads` between stage and scan. It is
correct (`relL2` unchanged), but much slower:

| workload | v11ab | v11ag block-wide |
|---|---:|---:|
| uniform spread | 13.7 ms | **52.0 ms** |
| clustered spread | 17.9 ms | **37.2 ms** |

Uniform ncu: 51.9 ms, 1.882B instructions (v11ab 717.6M), active warps/scheduler 4.41,
eligible 0.64, no-eligible 55.7%. Even clustered, where all four warps are relevant and the
mask is all-valid, the block-wide phase/synchronization pipeline dominates. The v11af
block-reread measurement still shows potential, but realizing it needs a warp-specialized,
double-buffered producer-consumer pipeline rather than a naive block-wide sequential batch.
Rejected; v11ab source/binaries restored.



### 2026-10-23 04:47 — scan metadata vectorization rejected (v11ah)

Replaced `scan_point`'s `int2` sxy + scalar sz + two strength loads with one aligned
`int4` load (`sx,sy,sz,strength.x` bitcast) plus one scalar `strength.y`. Correct and
95 regs / 5 blocks, but A/B medians are neutral: uniform 13.73 vs 13.69 ms, clustered
18.06 vs 18.07, sparse 1.709 vs 1.711. The spread-kernel SASS is effectively identical:
both versions contain 7 `LDS.U.128`, and the only difference is a commuted `IADD3`.
ptxas had already coalesced the metadata loads, so the source change cannot help.

Rejected; v11ab source/binaries restored. Artifact: `v11ah/` (`ab_v11ab_v11ah.txt`,
`ptxas_v11ah.txt`, `sass_note.txt`).



### 2026-10-23 05:00 — warp-local relevance mask rejected (v11ai/v11aj)

Implemented the mask at warp granularity, with no block barriers and no atomics. It is
correct and finds real work: **mask0 fraction 0.1996 uniform, 0.0773 clustered, 0.4634
sparse**. But all implementations are slower:

| variant | uniform | clustered | sparse |
|---|---:|---:|---:|
| v11ab reference | 13.69 | 18.07 | 1.711 |
| v11ai ballot mask, stage-only (scan unmasked) | 19.7 | 24.5 | 2.82 |
| v11aj transposed staging + full mask | 15.54 | 18.50 | 1.897 |
| v11aj transposed staging, no mask | 13.94 | 17.97 | 1.772 |
| v11aj transposed staging + stage mask, unmasked scan | 14.17 | 18.41 | 1.797 |

- v11ai initially needed 121 registers; simplifying the overlap test to an exact-interior /
  conservative-edge test brought it to 96 registers / 5 blocks with no spills, but the
  per-point ballot and control still cost more than the filtered work.
- v11aj moved `s/zz` and the mask to one point per lane and kept Horner one point per warp,
  parking zz in the row pads. This removed the ballot and cut fold/ceil/zz sequences from
  one per point to three per batch, but the staging rewrite alone is neutral and the mask
  branch/scan filtering still loses.
- The saved work is not large enough: mask0 is rare in clustered (7.7%), and the mask
  control/extra sync/zero-pad work is paid on every point. Closed; v11ab restored.

Artifacts: `v11ai/` (ballot variant, mask diagnostic source) and `v11aj/` (masked, no-mask,
stage-mask variants; quick results; ptxas).



### 2026-10-23 05:20 — per-batch compact valid-point list (v11ak)

Replaced the per-point mask branch with a batch-level compaction: one lane per point
computes the mask, one `__ballot_sync`/`__popc` per batch builds a dense valid-index list,
and Horner/scan iterate that list. All-valid batches keep the contiguous fast path.

| variant | regs / blocks | uniform | clustered | sparse |
|---|---:|---:|---:|---:|
| v11ab | 95 / 5 | 13.69 | 18.07 | 1.711 |
| v11ak one-point compact | **96 / 5** | 14.17 | **18.03** | **1.569** |
| v11ak two-point compact | 102 / 4 | — | — | — |
| v11ak two-point compact, capped 96 | 96 / 5 (16 B spills) | 14.92 | 18.37 | 1.887 |
| v11ak always compact | 119 / 4 | 15.03 | 20.09 | 1.745 |

This is the best mask implementation and validates the compaction idea: the per-point mask
branch is gone, sparse improves by 8.3%, and clustered is neutral. It still loses uniform
because the one-point compact list gives up two-point ILP; the two-point list exceeds the
96-register budget. Rejected as an overall replacement; v11ab restored. Sparse-only variant
could be selected if sparse latency were the priority.

Artifact: `v11ak/` (one-point/two-point sources, ptxas, quick results).



### 2026-10-23 05:35 — in-place row compaction / second-buffer check (v11al)

Tried to recover two-point ILP without an index-indirected hot loop by compacting the
valid rows to the front of the existing stage buffer. In-place ascending copies are safe
because `valid[v] >= v`; no second buffer is required.

| variant | regs / blocks | uniform | clustered | sparse |
|---|---:|---:|---:|---:|
| v11ab | 95 / 5 | 13.69 | 18.07 | 1.711 |
| v11al metadata-only copy | 104 / 4 | not timed | — | — |
| v11al full-row copy | 100 / 4 | 13.87 | 19.38 | 1.815 |

The copy/list path costs the 5-block occupancy. A dedicated second buffer is worse:
current shared use is 12,656 B/block; 5 blocks already occupy 63,280 B of the 64 KB pool,
leaving only ~2.2 KB/SM. A second full stage buffer would force 4 blocks immediately.
Rejected; v11ab restored.

Artifact: `v11al/` (metadata/full-row sources, ptxas, quick results).



### 2026-10-23 — mechanical v11ab source refactor (no performance change)

Renamed the v11ab kernel identifiers to match the explanatory pseudocode:
`stage_raw_points`, `prepare_point_coefficients`, `place_support`,
`place_support_nowrap`, `scan_one_staged_point`, `staged_point_values`,
`current_batch_values`/`next_batch_values`, `output_cell_accumulators`, etc.
No layout, arithmetic, launch, or occupancy change. ptxas remains **95 registers /
12,544 B shared / 5 blocks**, and the spread-kernel SASS is identical to pre-refactor
v11ab. The v11ab ncu source captures and flamegraphs were regenerated against the
refactored line numbers.



### 2026-10-23 — direct SoA / padding-free layout rejected (v11am)

Tested `staged_point_values[warp][buffer][value][point]` with the five zero-pad slots
removed: 23 values per point, 10,304 B shared/block, 96 regs, 5 blocks.

| variant | shared/block | uniform | clustered | sparse |
|---|---:|---:|---:|---:|
| v11ab AoS, padded | 12,544 B | **13.69** | **18.07** | **1.711** |
| v11am SoA, no pads | **10,304 B** | 14.92 | 19.72 | 1.806 |
| v11am SoA, BATCH=16 | 11,776 B | 16.50 | 22.99 | 2.040 |

Correctness holds (`relL2` unchanged). The shared reduction is real, but the hot scan
loses the 128-bit metadata load and needs scalar/guarded coefficient loads; staging
gains do not offset it. BATCH=16 is worse. Rejected; v11ab restored.

Artifact: `v11am/` (BATCH14/BATCH16 sources, ptxas, quick results).



### 2026-10-23 — SoA pair-prefetch rejected (v11an)

On top of the SoA/no-padding layout, added a two-point metadata prefetch: support
starts via aligned `int2` and strengths via `float2`, then scan both points.

| variant | regs / blocks | uniform | clustered | sparse |
|---|---:|---:|---:|---:|
| v11ab AoS | 95 / 5 | **13.69** | **18.07** | **1.711** |
| v11am SoA no prefetch | 96 / 5 | 14.92 | 19.72 | 1.806 |
| v11an SoA pair prefetch | 104 / 4 | 15.40 | 20.15 | 1.927 |
| v11an pair, capped 96 | 96 / 5, 16 B spills | 14.65 | 19.49 | 1.782 |

The vector pair load saves metadata LDS instructions but needs 10 live metadata
registers across the two scans, pushing the kernel to 4 blocks. Capping spills and is
still slower. Four-point prefetch would require even more metadata registers and is not
viable at 5-block occupancy.

`cp.async` is sm_80+; this target is sm_75 (Quadro T1000). It cannot be used here, and
it is global→shared only, not shared→shared, so it cannot asynchronously load staged
metadata. Rejected; v11ab restored.

Artifact: `v11an/` (source, ptxas, quick results).



### 2026-10-23 — 8-cell geometry / 24-warp occupancy rejected (v11ap)

Tried the register-reduction path: 8 warps/block, tile `16×16×8`, warp region
`8×8×4`, lane group `2×2×2`, 8 accumulators, and a hybrid metadata/coefficient shared
layout. This hits the hardware target exactly: **79 registers, no spills, 3 blocks ×
8 warps = 24 warps/SM**.

| case | v11ab spread | v11ap spread | v11ab visits | v11ap visits |
|---|---:|---:|---:|---:|
| uniform | **13.79** | 14.29 | 5.36M | 7.65M (+43%) |
| clustered | **18.06** | 21.21 | 8.00M | 12.00M (+50%) |
| sparse | **1.711** | 1.984 | 0.80M | 1.20M (+50%) |

The z-half warp region increases staging/re-read enough to overwhelm the 20% occupancy
gain. This closes the simple “8 cells/lane” geometry; a smaller z region (or x/y split)
trades the same support/re-read problem. Rejected; v11ab restored.

Artifact: `v11ap/` (source, ptxas, quick results).



### 2026-10-23 — 8-cell geometry on pure SoA (v11aq)

Applied the `8×8×4` warp region / `2×2×2` lane group / 8-accumulator geometry to
the pure SoA/no-padding layout.

| variant | regs / blocks | uniform | clustered | sparse | visits |
|---|---:|---:|---:|---:|---:|
| v11ab AoS | 95 / 5 | **13.79** | **18.06** | **1.711** | 5.36M / 8.0M / 0.80M |
| v11ap hybrid geometry | 79 / 3×8 | 14.29 | 21.21 | 1.984 | 7.65M / 12.0M / 1.20M |
| v11aq pure SoA geometry | **76 / 3×8** | 14.75 | 21.71 | 2.033 | 7.65M / 12.0M / 1.20M |

The pure SoA geometry lowers registers to 76 and preserves the 24-warp occupancy, but
the point visits are identical to v11ap because the warp geometry is identical. The
extra z-half reread remains the dominant loss; pure SoA adds scalar metadata and
coefficient-store bank costs on top. Rejected; v11ab restored.

Artifact: `v11aq/` (source, ptxas, quick results).



### 2026-10-23 — SoA register-recompute sequence (v11ar)

Tested the planned liveness/recompute steps on the 8×8×8 SoA layout:

| step | registers | blocks | uniform quick |
|---|---:|---:|---:|
| SoA baseline | 96 | 5 | 14.913 |
| 1. coefficient preload scoped per batch | **91** | 5 | 14.819 |
| 1b. direct KER6 table loads | 91 | 5 | 15.163 |
| 2. pack support offset/local bits | 93 | 5 | 15.603 |
| 3. recompute xy weights (from step 1) | 91 | 5 | 14.961 |

Coefficient scoping frees only 5 registers, not 9. Direct table loads do not reduce
further; support packing is counterproductive; weight recomputation does not lower
registers. None reaches 80, so occupancy remains 5 blocks. The fixed 32-register
accumulator plus scan working set is the hard limit for this layout. Rejected;
v11ab restored. Artifact: `v11ar/`.


## Final comprehensive summary — v11ab baseline, 2026-10-23 (superseded by v11as addendum below)

This supersedes the earlier v11m snapshot. Current best is **v11ab**; all totals now include
launch-setup metadata (`preprocess + meta + spread`), matching the baseline’s timed `setpts +
execute`.

### Current result

| workload | spread | preprocess | meta | setup | total | method 3 total |
|---|---:|---:|---:|---:|---:|---:|
| uniform `1e6` `64^3` qb=1 radix | **13.70 ms** | 6.26 | 0.38 | 6.65 | **20.35 ms** | 16.78 |
| clustered `1e6` `64^3` qb=4 counting split 32 | **18.07 ms** | 1.72 | 0.08 | 1.80 | **19.87 ms** | 16.79 |
| sparse `1e5` `128^3` qb=4 radix | **1.710 ms** | 0.42 | 0.21 | 0.63 | **2.337 ms** | 2.95 |

- 95 registers / 5 blocks; no spills.
- `relL2`: 5.49e-7 uniform, ~2.6e-7 clustered, 6.76e-8 sparse.
- Uniform ncu: 13.77 ms / 717.6M instructions; clustered: 17.37 ms / 1060.4M.
- Uniform source shares: `scan_point` 50.7%, `stage_coeff` 27.0%, `stage_raw` 5.7%.
- Clustered source shares: `scan_point` 53.1%, `stage_coeff` 31.4%, `stage_raw` 3.1%.
- Clustered active tiles: **16/128**; active warps/scheduler 4.73, eligible 1.71,
  no-eligible 27.6%.

### What changed after v11t/v11y

- **v11q** (exact micro-opts): store `s` as int, hoist `nfd`/support/coefficient offsets,
  move the `lane<18` predicate outside the point loop. −3.4% uniform, −3.3% clustered,
  −4.5% sparse vs v11m. Kept.
- **v11r**: `__float2int_ru`; neutral; rejected.
- **v11s**: pointer increment in the coefficient loop; small additional win; kept.
- **v11t**: use the float `ceilf` result for `zz` while storing the int support start;
  removes the int→float dependency. Kept; current base before clustering.
- **v11x/v11y/v11z**: raise the clustered split target. 16 splits 19.13 ms; **32 splits
  18.75 ms**; 64 splits 19.74 ms (reduction overhead). v11y = 32 splits, current clustered
  schedule.
- **v11aa**: active-tile compaction using the host-visible active flags; device-side
  deterministic tile list; launch only active tiles. Clustered grid 4096 → 512 blocks;
  clustered spread 18.75 → 18.07 ms. Uniform/sparse all tiles active. Metadata is now timed:
  `kernel_visits` + compaction + D2H copies = 0.51 uniform, 0.08 clustered, 0.19 sparse.
  Honest totals were introduced here.
- **v11ab** (current): parallelize `kernel_visits` — one warp per warp region instead of one
  thread per region. Uniform metadata 0.51 → 0.38 ms (`visits` 0.313, compaction 0.015);
  totals 20.57 → 20.35 ms. Added separate `visits=`, `compact=`, `meta=`, `setup=`,
  `total=` output and NVTX ranges. nsys Chrome timelines now include `kernel_visits`.

### Rejected variants (recent)

- **v11u** two-point `stage_coeff` unroll: Horner wait dropped, but loop/control
  instructions rose 6.99M → 18.88M; total instructions +1.65%; regressed.
- **v11v** raw/strength in registers across scan: 101 registers → **4 blocks/SM**; regressed
  to 15.05 / 21.40 / 1.905 ms despite no spills.
- **v11w** raw registers live only across `stage_coeff`: 96 registers / 5 blocks, but
  neutral/slightly worse; the global-load stall moved to the deferred store rather than
  being hidden.
- **v11ah** source-level `int4` scan metadata load: neutral; ptxas already emitted the same
  128-bit shared loads (identical SASS apart from a commuted `IADD3`).
- **v11ai/v11aj** warp-local relevance mask: mask0 20% uniform / 7.7% clustered / 46% sparse,
  but every variant is slower (ballot 19.7/24.5/2.82; transposed+mask 15.54/18.50/1.897;
  transposed no-mask 13.94/17.97/1.772). Control overhead exceeds saved work. Closed.
- **v11ak** per-batch compact valid list: best mask form, one-point 14.17/18.03/**1.569** at
  96 regs; sparse-only win, but uniform loses ILP and two-point compact is 102 regs.
  v11ab remains overall best.
- **v11al** in-place compacted rows: restores two-point ILP but 100–104 regs / 4 blocks;
  second buffer would also blow the ~2.2 KB/SM shared headroom. Rejected.
- **v11am** true SoA/no-padding: 23 values/point and 10,304 B shared, but scan scalar
  metadata + guarded coefficients lose; 14.92/19.72/1.806; BATCH16 worse. Rejected.
- **v11an** SoA + two-point metadata prefetch: 104 regs / 4 blocks; 15.40/20.15/1.927;
  capped 96 spills and is 14.65/19.49/1.782. cp.async unavailable on sm_75. Rejected.
- **v11ao** SoA with odd value stride 15: bank conflicts drop 5.42M -> 0.062M, but LDS
  instructions stay 63.0M; 14.79/19.69/1.798. Rejected.
- **v11ap** 8-cell geometry: 79 regs/no spills/24 warps, but z-half reread +43–50%;
  14.29/21.21/1.984. Rejected.
- **v11aq** same geometry on pure SoA: 76 regs/no spills/24 warps, same visits, but
  14.75/21.71/2.033. Geometry reread is the blocker. Rejected.
- **v11ar** SoA recompute/liveness steps: best 91 regs/5 blocks; coefficient scoping
  frees only 5, direct table loads/support packing/weight recompute do not reach 80.
  Rejected.
- **v11ac** 3D summed-area table, one thread per prefix line: query 0.019 ms but build
  0.242 ms; uniform meta 0.414 ms; worse than v11ab 0.38.
- **v11ad** SAT with `cub::BlockScan` prefix lines: build 0.286 ms, meta 0.471 ms; worse.
- **v11ae** block-per-region `kernel_visits` 32/64/128/256 threads: uniform 64/128 saves
  only ~0.02 ms; clustered neutral; sparse visits 0.055 → 0.074/0.110 ms. Rejected.

### Block-level coefficient-sharing measurement (v11af)

A diagnostic block-region visits counter measured the point stream a `16×16×8` block would
stage, versus the current warp-level stream:

| workload | block reread | warp reread | coefficient-work saving | extra scan if all 4 warps scan |
|---|---:|---:|---:|---:|
| uniform | **3.31** | 5.36 | 38.2% | +147% |
| clustered | **2.00** | 8.00 | **75.0%** | **0%** |
| sparse | **4.49** | 8.00 | 43.9% | +124% |

The earlier ~5.3 block-reread estimate was too pessimistic. Clustered is the standout:
`4 warps × 2.00 block reread = 8.0 warp scans`, exactly the current work, so block-level
coefficient sharing can remove ~75% of clustered `stage_coeff`/stage_raw work without
additional scan work. Uniform saves 38% of coefficient work but needs a per-warp filter to
avoid ~1.47× extra early-out scans. Sparse saves 44% but also needs filtering.

This made **block-wide coefficient sharing the top structural candidate**, especially for
clustered, and it remains atomics-free. The v11ag naive block-wide prototype was still
2–4× slower from block synchronization/phase serialization; realizing the measurement
needs a warp-specialized, double-buffered pipeline. The cheaper warp-local mask was
subsequently tried (v11ai/v11aj) and also lost to control overhead, so block sharing is not
viable without a much larger pipeline redesign.

### Metadata / baseline accounting

`kernel_visits` is a device kernel, but it cannot be inside `spread_cartesian_owned`: the
host needs its result to choose `nsplit` and the grid. It is a launch-setup pass, analogous
to baseline method 3’s `setpts` work:

- `setpts` → `indexSort_subprob_and_OD` builds bin sizes, scans `binstart`, computes
  `subprob`, scans/copies `totalnumsubprob`, maps `subprob_to_bin`.
- The harness times `setpts` and `execute` separately; baseline total includes both.
- Our fair total is `preprocess + meta + spread`.
- Uniform `kernel_visits` single-thread was 472 µs with **2 blocks / 512 threads** and 76.5%
  no-eligible. Warp-per-region v11ab is 313 µs.
- Baseline `setpts` is ~0.8 ms; our `preprocess + meta` is **6.65 ms**. The dominant setup gap
  is the **6.26 ms Cartesian binning/radix preprocessing**, not the 0.38 ms metadata.

### Traces

- ncu source flamegraphs/calltrees for v11ab:
  `v11ab/uniform_qb1_flamegraph_chrome.json`,
  `v11ab/clustered_qb4_flamegraph_chrome.json`, and matching `*_calltree.txt`.
  ncu captures only `spread_cartesian_owned`; widths are PC samples scaled to kernel time.
- nsys wall-clock Chrome timelines for v11ab:
  `v11ab/uniform_nsys_chrome.json`, `v11ab/clustered_nsys_chrome.json`.
  These include `kernel_visits`, `kernel_compact_tiles`, preprocessing, spread, memcopies,
  and NVTX ranges. Use these for metadata and launch/sync overhead.
- `ncu_samples_to_chrome.py` ranges/labels are current for v11ab.

### Gate status

- uniform spread 13.70 vs 12.76 target: gap **0.94 ms**; only 14% faster than method 3,
  needs 20%.
- uniform total 20.35 vs 17.62: gap **2.73 ms**; at current spread, `preprocess + meta`
  (6.65 ms) must fall to **≤3.92 ms**.
- clustered spread 18.07 vs 12.72: gap **5.35 ms**; total 19.87 vs 17.62: gap **2.25 ms**;
  1.14x slower than method 3 on spread, 1.18x on total.
- sparse 1.710 / 2.337 vs 2.82 / 2.95: ahead.

### What remains

1. **Preprocessing**: 6.26 ms vs baseline setpts ~0.8 ms. Device-side bin starts, cheaper
   counting/radix, fusing `kernel_visits` into preprocessing, or reusing baseline ordering.
   This is the main uniform-total gap.
2. **Clustered reread**: 8.0×. The v11af measurement showed promising block reread
   (2.00 clustered, 3.31 uniform, 4.49 sparse), but the v11ag block-wide prototype was
   2–4× slower because of block synchronization and phase serialization, even for clustered.
   Block-level coefficient sharing would need a warp-specialized/double-buffered pipeline
   to be viable; the naive implementation is rejected. Reducing scan reread still requires
   shared/global atomics or a merge.
3. **Exact 3-point `stage_coeff` packing**: not implemented. 9 lanes/point, 3 points/warp,
   existing `KER6` sets, second set from shared to stay at 96 registers. Potential ~1–2 ms
   uniform, but a substantial change.
4. **Stop tuning `kernel_visits`**: parallel warp-per-region is the sweet spot; SAT build,
   block-per-region, and further metadata work are exhausted.

### Artifacts

- Current source/binary: `devel/morton_spread/exp_a_cartesian.cu`,
  `build/gpu/exp_a_cartesian[_lineinfo]` = **v11ab**.
- `profile-artifacts/expA-output-owned/opt-2026-10-22/v11aa/` .. `v11ae/`.
- v11ab has bench, ncu flamegraphs/calltrees, nsys Chrome timelines, nsys-rep files.
- A/B logs include `ab_v11y_v11aa.txt`, `ab_v11t_v11u.txt`, `ab_v11t_v11v.txt`,
  `ab_v11t_v11w.txt`, `ab_v11x_v11y.txt`, and `v11ae/sweep_repeats.txt`.
- Tracker/research notes are local/untracked; no commits for `docs/issues/`.


## Answer

The prototype is built and measured; full evidence, commands, and counters are in
[ISSUE-009-exp-a-results.md](../research/ISSUE-009-exp-a-results.md). Artifact:
[`devel/morton_spread/`](../../../devel/morton_spread/) (`exp_a_cartesian.cu`, `README.md`,
`run_exp_a.sh`; built with `run_exp_a.sh`, binary `build/gpu/exp_a_cartesian`).

**Result: no benefit.** Output-cell ownership is correct but slower than method 3 on every
mandatory workload, so it fails the success gate:

| Case | method 3 spread (ms) | prototype spread (ms) | ratio | `relL2` |
|---|---:|---:|---:|---:|
| `1e6` uniform `64^3` | 20.2 (15.9–24.4 across runs) | 73.0 | 3.6–4.6x slower | 5.5e-07 |
| `1e6` clustered `64^3` | 15.9 | 209.1 | 13.1x slower | 5.6e-06 |
| `1e5` uniform `128^3` | 2.80 | 6.80 | 2.43x slower | 6.8e-08 |

Effective spread throughput is 13.7 Mpts/s (uniform `1e6`), 4.8 Mpts/s (clustered) and
14.7 Mpts/s (sparse `1e5`) versus method 3's 49.6–62.7, 62.9 and 35.7 Mpts/s. Preprocessing
(query-bin counting sort, `qb=4` best of the sweep) is 0.7–5.7 ms, not the problem. The kernel is instruction-bound: 2.12e9 warp instructions for 12.0e6 warp-level
point reads at `1e6` uniform (~176 instructions per read), because each point is read by ~12
warp regions and each read pays the fold/support-test/kernel-value/8-cell path with heavy
branch divergence (56.6% uniform branch targets). Occupancy is 50% (64 registers, 12 KB
shared) versus method 3's 25%, and it does not help. The clustered case additionally exposes
the expected load imbalance: only 96 of 1,024 warp regions are active, which is the 13x
regression.

**Recommendation for Experiment B.** Morton ordering is not worth testing as scoped. It does
not change the 12x reread or the ~176-instruction per-read cost, and on clustered data
Morton-sorted points concentrate in the same few tiles, so the imbalance remains; it also
adds radix-sort time. Keep the warp-cooperative staging, one-writer-per-cell stores, and the
no-merge output; change the gather first (16+ cells per thread, lane-level candidate
filtering, active-tile compaction/work stealing). Even at ~40 instructions per read a
redesigned gather only reaches parity on this GPU. The go/no-go ticket should treat this as
evidence against integration and either drop Experiment B or re-scope it behind that redesign.
- 2026-10-22 17:30 — Revised after profiling. The v1 kernel was dominated by divergent staging
  control flow and divergent constant loads, not by output ownership. Rewritten
  (register-resident Horner coefficients, batched coalesced raw loads, branchless
  `dim_support`, single early-out) and re-measured: uniform `1e6` spread 73.0 -> 24.4 ms
  (1.53x slower than method 3), clustered 209.1 -> 69.7 ms (4.38x, load imbalance), sparse
  6.80 -> 2.98 ms (1.06x). Still fails the >=20% gate. Evidence:
  `docs/issues/research/ISSUE-009-exp-a-results.md` (v3 revision) and
  `profile-artifacts/expA-output-owned/v3/`.

## Answer addendum (2026-10-22)

The Answer above is the v1 result and is superseded by the v3 revision. After the
profiler-driven rewrite the prototype is correct (`relL2` 6.8e-8..5.6e-6) and:

- uniform `1e6` spread 24.4 ms vs method 3 15.95 ms (1.53x slower; total 1.79x);
- sparse `128^3` `1e5` spread 2.98 ms vs 2.82 ms (1.06x slower);
- clustered `1e6` spread 69.7 ms vs 15.90 ms (4.38x slower — 96 of 1024 warp regions active).

It still misses the >=20%-faster gate, so the no-go stands as implemented. The uniform gap
is now per-point candidate work (`dim_support` 29% of the kernel, staging 29%, support tests
20%, accumulation 13%); the clustered gap is active-tile compaction/work stealing, which was
deliberately out of scope. Morton ordering (Experiment B) addresses neither.
- 2026-10-22 17:34 — v8. Added a two-point ILP scan and, for clustered inputs, split warp
  regions with per-split partial grids plus a reduction kernel (nsplit chosen from the
  preprocessing visit count). Warm spread: uniform 21.68 ms (1.36x slower than method 3),
  clustered 26.35 ms (1.66x, was 4.38x), sparse 2.53 ms (1.11x faster); `relL2`
  5.5e-7/7.1e-7/6.8e-8. The gate is still missed on both mandatory `1e6` cases. Profile:
  `scan_point` 31.4%, `stage_coeff` 27.9%, `dim_support` 25.6%; clustered ncu active
  warps/scheduler 1.62 -> 4.26. Artifacts: `profile-artifacts/expA-output-owned/v8/`.

## Answer addendum 2 (v8, 2026-10-22)

Supersedes the v3 addendum for the numbers; the no-go still stands. The clustered regression
is fixed by split warp regions (4.38x -> 1.66x). The remaining gap is uniform: 1.36x slower,
spent in `scan_point`/`stage_coeff`/`dim_support` at 4.23 active warps per scheduler. The
review's masked `dim_support` was measured again on the v8 structure and is still slower than
the select form (see the results note).
- 2026-10-22 18:05 — v10. Double-buffered staging, contiguous x-run batching (enables
  `qb=1`, reread 5.36), and a CUB radix-sort preprocessing option. Best configs: uniform
  `qb=1` radix spread 16.7 ms / total 22.9 (method 3: 15.95 / 16.78); clustered `qb=4`
  counting split-8 spread 25.9 / total 27.6; sparse `128^3` spread 2.08 / total 2.49
  (1.18x faster total). Fold-broadcast (B1) measured +31% and was discarded. The spread
  gate still fails (uniform parity, clustered 1.63x) and the uniform total is 1.37x.
  `relL2` 5.5e-7 / 7.0e-7 / 6.8e-8.

## Answer addendum 3 (v10, 2026-10-22)

Supersedes addendum 2's numbers. Uniform spread is now at parity (1.05x slower, was 1.36x)
and sparse is 1.36x faster; clustered is 1.63x slower (split fixed scheduler starvation but
not per-point work). The total gate fails because preprocessing is 6.2 ms at `qb=1` versus
method 3's 0.9 ms `setpts`; even free preprocessing would only reach parity. Gate remains a
no-go as implemented.


## Session log — 2026-10-23 (comprehensive)

Earlier v11a–v11p work is recorded in the 2026-10-22 session log. This log covers the
v11q–v11ar optimization sequence and the latest SoA/geometry/recompute studies.

### Current result

**Current best: v11ab** (exact, no atomics). All totals include launch metadata
(`preprocess + meta + spread`), matching method 3’s timed `setpts + execute`.

| workload | spread | preprocess | meta | setup | total | method 3 total |
|---|---:|---:|---:|---:|---:|---:|
| uniform `1e6` `64^3` qb=1 radix | **13.70** | 6.26 | 0.38 | 6.65 | **20.35** | 16.78 |
| clustered `1e6` `64^3` qb=4 counting split 32 | **18.07** | 1.72 | 0.08 | 1.80 | **19.87** | 16.79 |
| sparse `1e5` `128^3` qb=4 radix | **1.710** | 0.42 | 0.21 | 0.63 | **2.337** | 2.95 |

- 95 registers / 5 blocks / no spills; 12,544 B shared.
- `relL2` 5.49e-7 / ~2.6e-7 / 6.76e-8.
- Spread shares: uniform scan 50.7%, stage_coeff 27.0%, stage_raw 5.7%; clustered
  scan 53.1%, stage_coeff 31.4%, stage_raw 3.1%.
- Clustered: 16/128 active tiles, nsplit 32, active warps/scheduler 4.73,
  eligible 1.71, no-eligible 27.6%.
- Source identifiers are now descriptive; the spread-kernel SASS is identical to the
  pre-refactor v11ab. v11ab ncu source captures/line ranges were regenerated.

### Chronology of experiments

| variant | change | result | status |
|---|---|---|---|
| v11q | int support start, hoisted lane geometry, predicate outside point loop | −3.4% uniform / −3.3% clustered / −4.5% sparse | kept |
| v11r | `__float2int_ru` | neutral | rejected |
| v11s | pointer increment in coefficient loop | small win | kept |
| v11t | float `ceilf` for `zz`, int support store | small win; base before clustering | kept |
| v11u | two-point `stage_coeff` unroll | Horner waits down, loop/control instructions up; regressed | rejected |
| v11v | raw+strength in registers across scan | 101 regs → 4 blocks; 15.05/21.40/1.905 | rejected |
| v11w | raw registers only across `stage_coeff` | 96 regs/5 blocks, neutral/worse | rejected |
| v11x/v11y/v11z | clustered split target 16/32/64 | 19.13 / **18.75** / 19.74 ms; 32 best | v11y kept |
| v11aa | active-tile compaction + device tile list | clustered spread 18.75 → 18.07; metadata timed; honest totals | kept |
| v11ab | parallel `kernel_visits`, one warp per region | uniform metadata 0.51 → 0.38; total 20.57 → 20.35 | **current** |
| v11ac | 3D summed-area table, thread per line | query 0.019, build 0.242; metadata 0.414 | rejected |
| v11ad | SAT with `cub::BlockScan` | build 0.286; metadata 0.471 | rejected |
| v11ae | block-per-region `kernel_visits` 32/64/128/256 | saves ~0.02 ms uniform; sparse worsens | rejected |
| v11af | block-reread diagnostic | block/warp reread 3.31/2.00/4.49; savings 38/75/44% | diagnostic |
| v11ag | naive block-wide coefficient sharing + mask | correct but 52.0/37.2 ms; block sync/phase serialization | rejected |
| v11ah | source-level `int4` scan metadata load | identical SASS/LDS schedule; neutral | rejected |
| v11ai | ballot-based warp relevance mask | mask0 20/7.7/46%; 96 regs after overlap simplification, but 19.7/24.5/2.82 | rejected |
| v11aj | transposed s/zz staging + per-point mask | full mask 15.54/18.50/1.897; no-mask 13.94/17.97/1.772 | rejected |
| v11ak | per-batch compact valid list, one-point scan | 96 regs; 14.17/18.03/**1.569**; sparse-only win, uniform loses ILP | rejected overall |
| v11al | compact list + in-place row compaction | 100–104 regs → 4 blocks; full-row 13.87/19.38/1.815 | rejected |
| v11am | true SoA, no padding, 23 values/point | 10,304 B shared, 96 regs; 14.92/19.72/1.806; BATCH16 worse | rejected |
| v11an | SoA + two-point metadata prefetch | 104 regs → 4 blocks; 15.40/20.15/1.927; capped 96 spills, 14.65/19.49/1.782 | rejected |
| v11ao | SoA odd value stride 15 | store conflicts 5.42M → 0.062M, but LDS stays 63M; 14.79/19.69/1.798 | rejected |
| v11ap | 8-cell geometry, hybrid layout | **79 regs/no spills/24 warps**, but visits +43–50%; 14.29/21.21/1.984 | rejected |
| v11aq | 8-cell geometry, pure SoA | **76 regs/no spills/24 warps**, same visits; 14.75/21.71/2.033 | rejected |
| v11ar | SoA recompute/liveness steps | coefficient scope 96→91; direct table 91; support pack 93; weight recompute 91 | rejected; target 80 not reached |
| two-pass 8×8×8 acc8 | analysis only | staged batch lifetime makes per-batch two-pass incorrect; full two-pass restages all points | not viable |

### Register/occupancy findings

- 5 blocks / 20 warps is the v11ab point: 95–96 registers.
- The 6-block threshold is **80 registers**, not 86: sm_75 allocation granularity
  rounds 81–88 to 88; only ≤80 fits 768 threads.
- AoS cannot reach 6 blocks even at 80 because 12,544 × 6 > 64 KB. SoA no-pad
  (10,304 B) can; pure SoA stride15 (11,040 B) cannot.
- Forced SoA at 80 registers spills 48 B/thread and is slower (14.76/19.33/1.92).
- Recompute/liveness steps bottom out at 91 registers. The hard block is the
  16×float2 accumulator set (32 registers) plus the scan working set.

### SoA findings

- Padding removal is real: 28 → 23 values/point, 12,544 → 10,304 B shared.
- Pure SoA loses the AoS `int4` metadata load and adds guarded coefficient loads.
- Coefficient stores with value stride 14 exploded shared store bank conflicts
  (0.43M → 5.42M); stride 15 fixes them (0.062M), but shared LDS stays 63.0M
  (vs 49.1M AoS), so the layout still loses.
- BATCH=16 was much worse: stride 16 is pathological for storage banks.
- Pair prefetch reduced metadata LDS but required 104 registers / 4 blocks.

### Geometry findings

- Halving z to 8×8×4 reaches 24 warps (79 hybrid / 76 pure SoA registers),
  but point visits rise 43–50% (5.36M→7.65M uniform, 8M→12M clustered); occupancy
  gain is much smaller.
- A 6×6×6 candidate would have 216 cells, exact only with 27 active lanes (5 idle),
  non-divisible 6/12 tiles and the same boundary/reread problem; not implemented.
- The proposed “stage once, scan z halves twice” 8×8×8/acc8 design is not valid
  per batch: staged data is batch-local while accumulators persist across all batches.
  A correct full two-pass restages the point stream and doubles visits (~10.7M), worse
  than the already-rejected z-half geometry.

### Remaining open paths

1. **Preprocessing** is the main uniform-total gap: 6.26 ms vs baseline `setpts` ~0.8 ms.
   Device bin starts, cheaper counting/radix, fusing metadata, or reusing baseline ordering.
2. **Clustered structural work**: block-level point/coefficient sharing would need a
   warp-specialized, double-buffered pipeline; the naive block-wide prototype failed.
3. **Exact 3-point `stage_coeff` packing** remains unimplemented.
4. Metadata tuning is exhausted: `kernel_visits` warp-per-region is the sweet spot.

### Artifacts

- Current: `devel/morton_spread/exp_a_cartesian.cu`, `build/gpu/exp_a_cartesian[_lineinfo]`.
- `profile-artifacts/expA-output-owned/opt-2026-10-22/v11ab/`: current ncu flamegraphs,
  calltrees, nsys timelines, refactored/pre-refactor source.
- `v11aa/`–`v11ar/`: variant sources, ptxas logs, A/B and quick results; `v11af/`
  blockdiag; `v11ar/` recompute steps.
- `soa_verify/`: side-by-side AoS/SoA/stride ncu reports and metrics.
- Tracker/research notes are local/untracked; no commits for `docs/issues/`.

## v11as addendum — block-shared staging (2026-10-23)

The clustered path now uses a block-wide, double-buffered batch. Each of the four warps
stages a contiguous quarter of the batch; all four scan the block union for their own
`8×8×8` regions. One `__syncthreads()` per batch, no relevance mask,
`BLOCK_POINTS_PER_BATCH=56`, and active-tile-only split reduction. Block sharing is enabled
when measured `4 × block_visits ≤ 1.05 × warp_visits`; this selects clustered (8.0M vs
8.0M) and leaves uniform (13.24M vs 5.36M) and sparse (1.80M vs 0.80M) on the v11ab
per-warp kernel. `nsplit` remains only a performance knob; the block-shared path uses 56
and uniform/sparse stay at 1. The block-visit metadata pass costs ~0.018/0.040/0.030 ms
clustered/uniform/sparse. Flamegraphs are in
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11as/`; the clustered graph is rooted
at `spread_cartesian_owned_block_shared` and contains `stage_block_quarter`.

| workload | v11ab spread / total | v11as spread / total |
|---|---:|---:|
| uniform `1e6` `64^3` qb=1 radix | 13.85 / 20.52 ms | 13.85 / 20.53 ms |
| clustered `1e6` `64^3` qb=4 counting | 18.07 / 19.87 ms | **14.03 / 15.87 ms** |
| sparse `1e5` `128^3` qb=4 radix | 1.71 / 2.38 ms | 1.71 / 2.41 ms |

Clustered is now faster than method 3 on both spread and total (~15.90 / 16.79 ms), but the
strict 20%-faster spread gate (12.72 ms) still misses by 1.31 ms. Resources: 94 registers,
12,544 B shared, 5 blocks/SM, no spills. Clustered ncu instructions fall 1060.4M → 886.6M;
`stage_coeff` instructions fall 4×, scan body unchanged. Remaining cost is the scan itself,
support placement, and barrier/loop overhead. Details and tuning notes:
`docs/issues/research/ISSUE-009-exp-a-results.md` and
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11as/`.

Committed as **ee88a2fc**. The session state is now: clustered is below method-3 total
(15.87 vs 16.79 ms) and below method-3 spread (14.03 vs 15.90 ms), but still 1.31 ms above
the strict 12.72 ms spread target; uniform/sparse have no kernel regression. The next
blocking issue for the uniform total is preprocessing: 6.2 ms radix binning vs method 3's
~0.8 ms `setpts`. Next: replace the radix-path host suffix scan with a GPU histogram +
`cub::DeviceScan` for `binstart`, then reduce the 4.47 ms `kernel_gather_sorted` by packing
original points into one record before the permutation gather (or sorting a packed payload
directly). Both target the uniform total; clustered already clears method 3. Detailed plan
and nsys breakdown: `docs/issues/research/ISSUE-009-exp-a-results.md`.

## v11at addendum — preprocessing rewrite (2026-10-23)

The spread kernels are unchanged from v11as. Preprocessing now uses a fused qbin+pack pass,
a 32-byte packed payload gather, and GPU binstart construction (radix: mark-starts +
reverse exclusive scan; counting: GPU exclusive sum). No host suffix scan or large
D2H/H2D bin-table copies remain.

| workload | v11as pre / spread / total | v11at pre / spread / total |
|---|---:|---:|
| uniform `1e6` `64^3` qb=1 radix | 6.22 / 13.72 / 20.38 ms | **3.09 / 13.84 / 17.37 ms** |
| clustered `1e6` `64^3` qb=4 counting | 1.72 / 14.03 / 15.86 ms | **1.46 / 14.03 / 15.62 ms** |
| sparse `1e5` `128^3` qb=4 radix | 0.41 / 1.71 / 2.40 ms | **0.29 / 1.71 / 2.27 ms** |

Uniform total now clears the 17.62 ms total gate. The strict 20%-faster spread gate still
fails: uniform 13.84 vs 12.76 ms, clustered 14.03 vs 12.72 ms. The remaining work is
spread-side (32-cells/lane geometry or warp-specialized staging); preprocessing is no
longer the uniform blocker. Details:
`docs/issues/research/ISSUE-009-exp-a-results.md` and
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11at/`.

Committed as **92014e4b**. The current state is: uniform total 17.37 ms (inside the
17.62 ms total gate), clustered total 15.62 ms (faster than method 3's 16.79 ms), sparse
total 2.27 ms. The remaining gate failure is spread-side only: uniform spread 13.84 vs
12.76 ms target, clustered 14.03 vs 12.72 ms target.

## v11au addendum — three-point coefficient packing (2026-10-23)

The per-warp staging path processes three points per warp iteration: 9 lanes per point,
27 active lanes, each lane computing two coefficients. The first KER6 set stays in
registers; the second uses a shared KER6 copy. Shared memory is 12,760 B, 96 registers,
5 blocks/SM, no spills. Block-shared clustered staging is unchanged.

| workload | v11at pre / spread / total | v11au pre / spread / total |
|---|---:|---:|
| uniform `1e6` `64^3` qb=1 radix | 3.08 / 13.86 / 17.37 ms | 3.09 / **12.59 / 16.11 ms** |
| clustered `1e6` `64^3` qb=4 counting | 1.46 / 14.04 / 15.63 ms | 1.47 / 14.03 / 15.62 ms |
| sparse `1e5` `128^3` qb=4 radix | 0.28 / 1.71 / 2.26 ms | 0.29 / **1.54 / 2.09 ms** |

Uniform now passes both the strict 12.76 ms spread gate and the 17.62 ms total gate.
Clustered total is faster than method 3, but clustered spread is still 14.03 vs 12.72 ms.
Details and flamegraphs:
`docs/issues/research/ISSUE-009-exp-a-results.md` and
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11au/`.

Committed as **674ad159**. Current state: uniform spread/total gates pass; clustered
total passes but clustered spread remains 14.03 vs 12.72 ms. The remaining work is
clustered scan-side only.
