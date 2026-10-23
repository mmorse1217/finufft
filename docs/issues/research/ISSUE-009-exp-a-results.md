# ISSUE-009 Experiment A: output-cell ownership with Cartesian bins — results

**Ticket:** [ISSUE-009](../in-progress/ISSUE-009-output-cell-ownership-cartesian-prototype.md)
(map: [cuFINUFFT Morton output-owned spread go/no-go](../todo/ISSUE-001-cufinufft-morton-spread-go-no-go.md))
**Date:** 2026-10-22 (v10 revision)
**Revision:** `e3c48641` plus the uncommitted prototype under `devel/morton_spread/`
**Device:** Quadro T1000, sm_75, 4 GB, driver 580.178.04
**Toolkit:** CUDA 12.8 at `~/cuda-12.8` (`nvcc` 12.8.61)
**Scope:** 3D type 1, FP32, `ns=6`, `sigma=2`, `gpu_spreadinterponly=1`, `gpu_kerevalmeth=1`,
tile `16x16x8`, 256 threads/block, `2x2x2` cells per thread.

Prototype source, build command, and design: [`devel/morton_spread/`](../../../devel/morton_spread/).
This note is the evidence; it does not modify the library.

## v10 revision (current)

Three further steps after v8 (each measured separately):

1. **Double-buffered staging** (next batch's global loads overlap the current scan):
   uniform 21.65 -> 21.41 ms (-1.1%), clustered 26.45 -> 25.90 (-2.1%).
2. **Fold-once-per-dimension + broadcast**: measured 28.0 vs 21.4 ms (+31%) and discarded —
   the six lanes of a dimension already fold in lockstep, so the broadcast only adds a
   dependency stall.
3. **Contiguous x-run batching**: each warp walks its bin cube as contiguous x-runs, so
   batches are 16 points regardless of the query-bin size. This makes `qb=1` (reread 5.36)
   usable: uniform spread 21.36 -> 16.78 ms (-21.4%). Sparse `qb=4` 2.53 -> 2.08 (-17.8%).
4. **Radix-sort preprocessing option** (CUB `DeviceRadixSort::SortPairs` of (bin, point
   index) + payload gather + start-mark/scan, selectable against the counting sort):
   at `qb=1` uniform 11.4 -> 6.2 ms. At `qb=4` 5.3 ms; clustered is faster with the
   counting sort (1.7 vs 2.5 ms), so the best preprocessing is distribution-dependent.

Current results (warm event timing, best query-bin/preprocessing per workload):

| Case | method 3 spread / total | prototype spread / total | spread ratio | total ratio | `relL2` |
|---|---:|---:|---:|---:|---:|
| `1e6` uniform `64^3` (`qb=1`, radix) | 15.95 / 16.78 ms | 16.7 / 22.9 ms | 1.05x slower | 1.37x slower | 5.5e-07 |
| `1e6` clustered `64^3` (`qb=4`, counting, split 8) | 15.90 / 16.79 ms | 25.9 / 27.6 ms | 1.63x slower | 1.64x slower | 7.0e-07 |
| `1e5` uniform `128^3` (`qb=4`, radix) | 2.82 / 2.95 ms | 2.08 / 2.49 ms | 1.36x faster | 1.18x faster | 6.8e-08 |

**Gate: still fails.** The spread is at parity on uniform and 1.63x slower on clustered, so
the >=20%-faster requirement is not met; the uniform total is 1.37x because preprocessing is
still 6.2 ms (method 3's `setpts` is 0.9 ms). Even free preprocessing would only bring the
uniform total to parity. The clustered case remains the structural gap: the split removed
the scheduler starvation (active warps/scheduler 1.62 -> 4.26) but not the per-point work.

## v8 revision (superseded by v10)

After v6 the review comments were applied (mask arithmetic in `floordiv`, qbin clamp and
bin wraps; the stats-kernel atomics removed; per-line masks for `dim_support` measured and
reverted because they were 11-17% slower). The kernel was then optimised from the v8 call-tree
profile. Current numbers, warm event timing, `qb=4`:

| Case | method 3 spread | v8 spread | ratio | v8 total (incl. preprocessing) | total ratio | `relL2` |
|---|---:|---:|---:|---:|---:|---:|
| `1e6` uniform `64^3` | 15.95 ms | 21.68 ms | 1.36x slower | 27.28 ms | 1.63x | 5.5e-07 |
| `1e6` clustered `64^3` | 15.90 ms | 26.35 ms (split 8) | 1.66x slower | 28.07 ms | 1.67x | 7.1e-07 |
| `1e5` uniform `128^3` | 2.82 ms | 2.53 ms | 1.11x faster | 3.18 ms | 1.08x | 6.8e-08 |

What changed after v3:

- **Split warp regions for clustered inputs.** Each tile is launched `nsplit` times; split
  `s` takes batches `start+s*BATCH, start+(s+1+nsplit)*BATCH, ...` of every bin, so all
  points are covered exactly once and hot bins are shared. Each split writes a full partial
  grid (no atomics, no zeroing); `kernel_sum_splits` adds them. `nsplit` is the smallest
  power of two with `active_warps * nsplit >= 512` (cap 8): uniform 512 -> 1, clustered
  64 -> 8. Clustered spread went 71.5-72.7 -> 26.35 ms; ncu active warps/scheduler
  1.62 -> 4.26, no-eligible 57.0% -> 26.4%, compute 19.5% -> 68.1%.
- **Two-point ILP scan** (`scan_point`, two calls per iteration): ~2% in isolation, within
  noise in the final build.
- **Vectorised `sx,sy,sz,strength` shared load** and hoisted xy weight product: neutral
  (the sampled stall moved from the load line to the next instruction).
- **Review masks**: `dim_support` masks cost 11-17% and were reverted to selects, which
  ptxas already lowers to branchless SEL (identical SASS branch count). Remaining
  `atomicAdd`s are only in `kernel_hist`/`kernel_scatter`, the Cartesian counting sort.

Call-tree profile (ncu, v8): `scan_point` 31.4%, `stage_coeff` 27.9%, `dim_support` 25.6%,
scan calls 6.1%, `stage_raw` 4.6%, staging 3.6%, bin loop 0.2%. Uniform runs at 4.23 active
warps/scheduler, 29.2% no-eligible, 65.6% compute; the gate (>20% faster than method 3 on
both `1e6` cases) is still missed.

## v3 revision (superseded by v8)

The first version (v1, below) was measured, then profiled with Nsight Systems and Nsight
Compute. The profile showed the conclusion was premature: the kernel was dominated by
divergent staging control flow and divergent constant loads, not by output ownership. The
kernel was rewritten and re-measured. **These are the current numbers; the v1 numbers
below are kept as history.**

Changes: Horner coefficients preloaded into registers (no `KER6[t][i]` constant load in the
loop); support-start store written by all six lanes of a dimension (no `i==0` branch);
batched coalesced raw loads for a whole 16-point batch before the coefficient pass;
branchless `dim_support`; one combined early-out instead of three. Rejected after
measurement: `__shfl_sync` broadcast of the fold (8.4 ms dependency stall), `BATCH=8/32`.

`qb = 4`, event timing, median of 5 after two warm-ups:

| Case | method 3 spread | v3 spread | ratio | v3 total (incl. preprocessing) | total ratio | `relL2` |
|---|---:|---:|---:|---:|---:|---:|
| `1e6` uniform `64^3` | 15.95 ms | 24.36 ms | 1.53x slower | 29.96 ms | 1.79x | 5.5e-07 |
| `1e6` clustered `64^3` | 15.90 ms | 69.66 ms | 4.38x slower | 71.37 ms | 4.24x | 5.6e-06 |
| `1e5` uniform `128^3` | 2.82 ms | 2.98 ms | 1.06x slower | 3.65 ms | 1.23x | 6.8e-08 |
| `1e5` uniform `64^3` | 1.74 ms | 2.50 ms | 1.44x slower | 2.97 ms | 1.58x | 1.6e-07 |

With `qb=1` (bin = cell, reread factor 7.65 instead of 12.0) the uniform spread improves to
22.93 ms (1.44x slower) but preprocessing grows to 11.4 ms and the sparse `128^3` case
degrades to 19.6 ms, so `qb=4` is the better total configuration.

Profiler counters, v1 -> v3 (uniform `1e6`, one ncu launch):

| | v1 | v3 |
|---|---:|---:|
| Duration | 72.82 ms | 24.11 ms |
| Warp instructions | 2.115e9 | 1.469e9 |
| Branch instructions | 342M (16%) | 74.8M (5%) |
| Branch efficiency | 56.6% | 71.8% |
| Avg active threads/warp | 16.2 | 22.5 |
| Issued warps/scheduler | 0.36 | 0.78 |
| Cycles with no eligible warp | 63.8% | 22.5% |
| Warp cycles/instruction | 19.6 | 7.03 |
| Registers / blocks per SM | 64 / 4 | 72 / 3 |

The v3 flame graph (`profile-artifacts/expA-output-owned/v3/spread_flamegraph_chrome.json`)
splits the remaining 24.1 ms into `dim_support` 29.2%, staging fold+Horner 20.9%, support
tests/early-out 19.8%, 8-cell accumulation 13.5%, fold helper 7.9%, batch loads/syncs 8.3%.
Stalls are now `wait`, `math`, `not_selected` and `short_sb`; branch-resolving is 4%. The
cluster case is unchanged in nature: only 96 of 1024 warp regions are active (load
imbalance, 4.38x), which no ordering change fixes.

**Revised conclusion.** The gather is no longer 4x off; it is 1.53x off on uniform and 1.06x
on sparse, with a 4.38x clustered regression that is the deliberately unimplemented
load-balancing case. It still fails the >=20% success gate, so Experiment A remains a no-go
*as implemented*. The remaining uniform gap is mostly per-point candidate work
(`dim_support`, staging arithmetic) and the clustered gap is active-tile compaction /
work stealing. Experiment B (Morton ordering) does not address either; it should only be run
if re-scoped around the clustered load-balancing and reread questions.

Details, commands, and all profiling artifacts:
[`profile-artifacts/expA-output-owned/`](../../../profile-artifacts/expA-output-owned/)
(`v1/` and `v3/`).

## What was built

A standalone CUDA program that links the existing `build/gpu` cuFINUFFT static library and
adds nothing to the library build:

1. times pinned cuFINUFFT method 3 (gate), and methods 4 and 1 (context), spread-only;
2. counting-sorts the points by Cartesian query bin `qb` (dense `bin -> start` table) and
   times that preprocessing separately;
3. runs the output-owned spread: one CUDA threadblock per `16x16x8` tile, one warp per
   `8x8x4` sub-region, one lane per `2x2x2` cell group. Each warp enumerates the query bins
   covering its region expanded by `ceil(ns/2)=3` cells (periodic wrap), stages the points
   cooperatively (18 lanes evaluate one Horner coefficient each into shared memory), and
   every lane accumulates its own eight cells and stores them exactly once — no atomics, no
   merge;
4. checks every fine-grid cell against the method-3 output (`relL2`) and reports kernel
   attributes, occupancy, candidate point reads, and active warp regions.

Kernel values and coordinates are cuFINUFFT's own (`fold_rescale`, and the ns=6 sigma=2
Horner coefficients from `include/cufinufft/contrib/ker_horner_allw_loop.inc`), so the
prototype differs from method 3 only in the accumulation/ownership strategy.

## Commands

```bash
export PATH=$HOME/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-12.8/lib64:$LD_LIBRARY_PATH
./devel/morton_spread/run_exp_a.sh all
# or, per case: build/gpu/exp_a_cartesian <M> <N1> <N2> <N3> <cluster> <qb> <nrun>
```

Timing is CUDA events, median of 5 runs after 2 warm-ups, as the success protocol requires.
The GPU clock is not locked; method-3 numbers moved by up to ~25% between cold and warm
runs (see the uniform row below), while the prototype's numbers were stable.

## Mandatory workloads

The query-bin edge `qb` is a per-workload tuning knob, so both measured configurations are
reported. `qb=4` was used for the ticket matrix; `qb=1` is the best configuration for the
dense `64^3` grid but is far worse for the sparse `128^3` grid (many empty bins).

`qb = 4`:

| Case | method 3 `setpts` / `exec` / `total` (ms) | prototype preprocessing / spread / total (ms) | spread ratio | total ratio | `relL2` |
|---|---:|---:|---:|---:|---:|
| `1e6` uniform, `64^3` | 1.04 / 20.16 / 21.20 | 5.66 / 73.01 / 78.67 | 3.62x slower | 3.71x slower | 5.5e-07 |
| `1e6` clustered, `64^3` | 0.90 / 15.91 / 16.80 | 1.72 / 209.14 / 210.85 | 13.15x slower | 12.55x slower | 5.6e-06 |
| `1e5` uniform, `128^3` | 0.13 / 2.80 / 2.93 | 0.67 / 6.80 / 7.47 | 2.43x slower | 2.55x slower | 6.8e-08 |

`qb = 1`:

| Case | method 3 `exec` (ms) | prototype preprocessing / spread / total (ms) | spread ratio | total ratio | `relL2` |
|---|---:|---:|---:|---:|---:|
| `1e6` uniform, `64^3` | 22.90 | 11.34 / 49.37 / 60.71 | 2.16x slower | 2.53x slower | 5.5e-07 |
| `1e6` clustered, `64^3` | 27.98 (15.9 earlier) | 1.91 / 209.31 / 211.22 | 7.5-13.1x slower | 7.2-12.6x slower | 5.8e-06 |
| `1e5` uniform, `128^3` | 2.80 | 4.25 / 19.57 / 23.82 | 6.99x slower | 8.14x slower | 7.2e-08 |

`qb=1` cuts the point reread factor from 12.0 to 7.65 (the geometric minimum for an `8x8x4`
warp region with a 3-cell halo) and is 1.5x faster than `qb=4` on the dense `64^3` grid, but
it makes preprocessing 11.3 ms at `1e6` and is ~3x slower on the sparse `128^3` grid. The
best ratio anywhere in the mandatory set is still 2.16x slower than method 3; the gate needs
>=20% faster. The conclusion does not depend on the query-bin choice.

Effective spread throughput (`M / spread`): uniform `1e6` 13.7 Mpts/s (method 3: 49.6 Mpts/s
at 20.16 ms, 62.7 Mpts/s at 15.95 ms), clustered `1e6` 4.8 Mpts/s (method 3: 62.9 Mpts/s),
sparse `1e5` 14.7 Mpts/s (method 3: 35.7 Mpts/s). The optional `N=1e7` and `ns=4/8` cases were
not run: the mandatory result already decides the gate.

Reference points from the baseline note: method 3 spread is 18.67 ms (uniform `1e6`),
15.75 ms (clustered `1e6`), 2.79 ms (sparse `1e5`). Across repeated runs this session the
uniform method-3 spread measured 15.95–24.40 ms, so the uniform ratio is 3.6–4.6x.

Context (not gates): method 4 measured 106 ms (uniform) and 179 ms (clustered) spread at
`1e6`; method 1 measured 52.6 ms and 70.6 ms. Method 3 is clearly the right baseline on this
device, matching the baseline note.

Query-bin sweep, spread only, on the final kernel: at `1e5` uniform `64^3`, `qb=1` 6.59 ms,
`qb=2` 7.58 ms, `qb=4` 7.34 ms, `qb=8` 21.31 ms; at `1e6` uniform `64^3`, `qb=1` 49.37 ms,
`qb=2` 73.10 ms, `qb=4` 73.01 ms. Finer bins cut candidate reads until the bin bookkeeping
and empty-bin enumeration dominate; the crossover moves with point density.

## Correctness

`relL2` is 6.8e-08 (sparse), 5.5e-07 (uniform `1e6`), 5.6e-06 (clustered `1e6`) against the
pinned method-3 spread-only output. All are inside the reordering noise floor of the
correctness contract (`~1e-7..1e-5` in FP32, growing like `eps*sqrt(K)`); every cell is
written exactly once by construction. Sums of `|new|` and `|ref|` matched exactly on the
`1e5` case. The clustered case has the largest error because cells receive many more
contributions; it is still ~500x below the 1e-5 tolerance.

## Occupancy, registers, memory (uniform `1e6`)

| Metric | Prototype | Method 3 (baseline note) |
|---|---:|---:|
| Registers/thread | 64 | 51 |
| Dynamic/static shared per block | 12,288 B | 65,150 B |
| Blocks/SM (occupancy API) | 4 (32 warps, 50%) | 1 (8 warps, 25%) |
| Achieved warp occupancy (ncu) | 88.7% of active | 25% |
| Warp instructions executed | 2.115e9 | n/a (not gathered) |
| DRAM bytes | 227.9 MB | n/a |
| SM throughput (ncu) | 82.3% of peak | n/a |
| Branch targets uniform | 56.6% | n/a |
| Average warp latency/instruction | 19.6 cycles | n/a |

Doubling occupancy and removing the 65 KB local subgrid did not help: the kernel is
instruction-throughput bound, not occupancy/barrier bound.

## Why it loses

1. **Candidate rereads.** Each point is read by ~12 warp regions (measured
   `reread_factor = 12`; the geometric minimum for an `8x8x4` region with a 3-cell halo is
   ~7.7, and the 4-cell query bin adds the rest). Method 3 reads each point in one block.
2. **Per-read cost.** At `1e6` uniform the kernel executes 2.115e9 warp instructions for
   12.0e6 warp-level point reads — ~176 instructions per read, ~9.8 per grid contribution
   (216e6 contributions). Every read pays the coordinate fold, the three support tests, the
   shared kernel-value reads, and the eight-cell weight products, and warp divergence makes
   the whole path run whenever any lane hits, which is effectively always.
3. **Divergence.** Only 56.6% of branch targets are warp-uniform; the support test and
   accumulation are per-lane, so the warp executes the union of lane paths.
4. **Load imbalance.** On the clustered case only 96 of 1,024 warp regions are active (all
   points fall into eight tight blobs covered by a few tiles), so those warps scan ~1.2e5
   points each while the rest idle. There is no active-tile compaction by design (ISSUE-008),
   and the measured result is the 13x regression.
5. **Preprocessing is not the problem.** Query-bin counting sort is 5.7 ms at `1e6` uniform
   (vs method 3 `setpts` 1.0 ms); the spread itself is the whole story.

The structural difference from method 3: method 3 spreads each point's 216 contributions
across 256 threads (~7 warp instructions per point) and pays for it with 25% occupancy and a
global merge; the gather serialises those contributions inside each of ~12 warps that read
the point and pays ~176 instructions per read. On sm_75 the second bill is much larger.

## Recommendation for Experiment B

**Do not run Experiment B (Morton ordering) as-is.** The measured gap is not caused by
Cartesian ordering:

- Morton order changes *which* points are adjacent, not how many warp regions read a point
  (7.65-12 here) nor the ~176 instructions each read costs. It cannot close a 2.2x gap (best
  query-bin configuration), let alone the 3.6x of the ticket matrix.
- On clustered data, Morton-sorted points are *more* spatially contiguous, so they
  concentrate in the same few tiles: the load-imbalance regression (13x) would remain or
  worsen. The sketch's own mitigation (active-tile compaction / heavy-tile splitting,
  section 21-22) is the real fix.
- It adds radix-sort time to preprocessing, which is already at the `setpts` budget.

What is worth keeping if the gather is revisited: warp-cooperative staging, one writer per
cell, shared-memory kernel values with one coefficient per lane, and the no-merge store.
What must change first: cells per thread (16+ so each candidate read feeds more
contributions), lane-level rather than warp-level candidate filtering (the 12x reread is the
dominant cost), and active-tile compaction or work stealing for clustered inputs. A
redesigned gather that reaches ~40 instructions per read would still only reach parity with
method 3 on this GPU.

Consequently the go/no-go ticket should see Experiment A as evidence that output-cell
ownership alone does not pay on this device, and either close Experiment B as not worth
running or re-scope it behind a gather redesign.

## Artifacts

- [`devel/morton_spread/exp_a_cartesian.cu`](../../../devel/morton_spread/exp_a_cartesian.cu) — prototype.
- [`devel/morton_spread/README.md`](../../../devel/morton_spread/README.md) — build, run, kernel shape.
- [`devel/morton_spread/run_exp_a.sh`](../../../devel/morton_spread/run_exp_a.sh) — build + matrix script.
- `build/gpu/exp_a_cartesian` — the binary used for the runs above.

## Reopen verification (2026-10-22 18:12)

Experiment A was reopened after the working session was closed by accident. Fresh warm
re-measurement with the current binary (`build/gpu/exp_a_cartesian`) and the best
configuration per workload:

| Case | method 3 spread / total | prototype spread / total | spread ratio | total ratio | `relL2` |
|---|---:|---:|---:|---:|---:|
| `1e6` uniform `64^3` (`qb=1`, radix) | 15.96 / 16.77 ms | 16.77 / 22.99 ms | 1.05x slower | 1.37x slower | 5.5e-07 |
| `1e6` clustered `64^3` (`qb=4`, counting, split 8) | 15.89 / 16.80 ms | 25.76 / 27.48 ms | 1.62x slower | 1.64x slower | 6.9e-07 |
| `1e5` uniform `128^3` (`qb=4`, radix) | 2.82 / 2.95 ms | 2.06 / 2.46 ms | 1.37x faster | 1.18x faster | 6.8e-08 |

These match the v10 numbers above within noise. Two of three distributions are competitive
on the spread stage: uniform is at parity and sparse is faster; clustered remains 1.62x
slower. The uniform **total** is still 1.37x because preprocessing is 6.2 ms versus method
3's 0.8 ms `setpts`. The success gate (>=20% faster spread on both `1e6` cases, total within
5%) is therefore still not met.

## Reopen profile and corrected call hierarchy (2026-10-22 18:35)

Fresh Nsight Compute captures of the current `-lineinfo` build confirm v10: uniform `qb=1`
kernel 16.59 ms / 0.966e9 instructions; clustered `qb=4` 25.79 ms / 1.514e9 instructions;
scheduler and stall counters match the v10 table. Artifacts and commands:
[`profile-artifacts/expA-output-owned/reopen-2026-10-22/`](../../../profile-artifacts/expA-output-owned/reopen-2026-10-22/).

`devel/morton_spread/ncu_samples_to_chrome.py` now applies the **static source call graph**
instead of putting every inlined function at the top level. The corrected hierarchy is
`spread_cartesian_owned -> batch loop -> {stage_raw -> __ldg, stage_coeff -> fold_rescale ->
cudaFMA, scan_point -> dim_support, __syncwarp}`. This is reconstructed from the source, not a
captured device stack: ncu PC sampling is flat, and the functions are inlined.

Corrected shares (own + inlined callees, uniform / clustered):

| Frame | uniform | clustered |
|---|---:|---:|
| `scan_point` total | **58.6%** | **62.0%** |
| — own body | 37.3% | 40.8% |
| — `dim_support` | 21.4% | 21.2% |
| `stage_coeff` total | **26.4%** | **26.8%** |
| — own body | 19.1% | 19.3% |
| — `fold_rescale` | 7.3% | 7.4% |
| — — `cudaFMA` | 3.4% | 3.5% |
| `stage_raw` total | 4.0% | 2.5% |
| batch-loop own body | 9.7% | 8.0% |

Line-level hotspots in the uniform case:

- `dim_support` (21.4%): periodic-wrap arithmetic `d<0` 6.1% + `d>=nf` 3.5% = 9.6%;
  `j0` select 3.7%, `n` select 2.8%, `nh` 1.9%, low/high tests 2.6%.
- `scan_point` body (37.3%): accumulation `acc y` 7.0% + `acc x` 3.3% + weight product 3.7%
  = 14.0%; shared-load/unpack 7.1%; kernel-value selects 14.8%.
- `stage_coeff` (26.4%): Horner chain 6.6%, fold 7.3% (cudaFMA 3.4%, `fsub_rd` 2.6%,
  `fmul_rd` 1.2%), support-start `ceilf` 3.4%, coefficient write 2.6%, `zz` rescale 2.4%.
- `stage_raw` (4.0%): the `row[21]` strength store alone is 2.8%. ncu reports 1.386e6 shared
  wavefronts and 0.995e6 excessive (about 72%) on the staging stores; `STAGE=24` floats makes
  the row stride 24, so the 16 lanes hit only four banks (4-way conflict). The staging loads
  from `stage_coeff`/`scan_point` are broadcast and show zero excessive wavefronts.

## Scan optimization experiments (2026-10-22 18:47)

Four variants were built and measured one at a time, with ncu captures under
[`profile-artifacts/expA-output-owned/opt-2026-10-22/`](../../../profile-artifacts/expA-output-owned/opt-2026-10-22/).
Benchmark medians are 4 process runs (2 warm-ups + 5 timed runs each) at the best config;
ncu instructions are for the uniform `qb=1` spread kernel.

| variant | change | uniform ms | clustered ms | sparse ms | uniform ncu inst |
|---|---|---:|---:|---:|---:|
| v10 baseline | — | 16.77 | 25.88 | 2.076 | 965.5M |
| v11a | delete dead second wrap correction | **15.91** | 24.88 | 1.989 | 917.3M |
| v11b | v11a + non-wrapping min/max support | 16.14 | **24.49** | **1.932** | 877.4M |
| v11c | v11b + min/max slow path | 16.33 | 24.23 | 1.938 | 881.3M |
| v11d | v11a + warp relevance filter | 17.43 | 28.06 | 2.039 | 972.6M |
| v11e | v11b, fast only when `nsplit > 1` | 17.24 | 24.31 | 2.172 | — |

The support-placement instruction count (all `dim_support*` source lines) fell from 262.4M
(v11a) to 138.5M (v11b), a 47% cut. The scan-point body and control flow grew from 485.3M to
562.4M, so uniform wall time did not improve; eligible warps/scheduler fell from 1.59 to 1.49
and no-eligible rose from 27.8% to 31.2%. v11e showed that the per-point branch/duplicated
code path costs even when it is disabled at runtime. v11a (the dead-line deletion) is the
only clean win and is the current source. The fast-path instruction reduction should be
revisited with warp/tile-level specialization, not a per-point branch.


## Per-batch fast-path specialization (v11f, 2026-10-22 20:32)

v11b's fast path cut support instructions but lost uniform time to a per-point branch. v11f
moves the decision outside the point loop: after `stage_coeff`, the warp tests the 16 staged
support starts with one `__all_sync`, then branches once to `scan_point<true>` (min/max
support) or `scan_point<false>` (general periodic support). The check requires the warp
region and every staged support to be interior; mixed batches take the general path.

| Case | v10 baseline | v11a | v11b | **v11f** |
|---|---:|---:|---:|---:|
| uniform spread | 16.77 ms | 15.91 | 16.14 | **15.65** |
| clustered spread | 25.88 | 24.88 | 24.49 | **22.99** |
| sparse spread | 2.076 | 1.989 | 1.932 | **1.948** |
| uniform ncu instructions | 965.5M | 917.3M | 877.4M | **828.0M** |
| clustered ncu instructions | 1514.0M | 1442.0M | 1327.8M | **1248.4M** |

Support-placement instructions: uniform 262.4M (v11a) -> 153.3M (v11f); clustered 392.0M ->
181.5M. The `scan_point` body stays ~488M uniform / ~806M clustered, so unlike v11b the
branch does not inflate the hot loop. `relL2` unchanged. v11f is the current working source.


## STAGE tests and int support starts (v11i, 2026-10-22 20:50)

| Variant | Change | uniform | clustered | sparse | occupancy |
|---|---|---:|---:|---:|---:|
| v11f | per-batch fast path | 15.65 | 22.99 | 1.948 | 5 blocks |
| v11g | STAGE=26 | 17.36 | 24.93 | 2.171 | 4 blocks |
| v11h | STAGE=25 | invalid (misaligned `float2`) | — | — | — |
| v11i | STAGE=24, int support starts | **15.53** | **22.61** | **1.902** | 5 blocks |

`STAGE=26` gives 16 lanes distinct banks but costs one block of occupancy, which is worse than
the conflict. `STAGE=25` is conflict-free and keeps 5 blocks but the existing `float2` loads
fault; a scalar-load variant would be needed. v11i stores `sx,sy,sz` as ints in the stage row
and reads them directly in both the batch check and `scan_point`, removing the
`LDS -> F2I -> dim_support` chain. The F2I is gone, but the stall moves to the next consumer
(`dim_support`/coefficient selects), so shared-load latency and the coefficient dependency
chain remain the main target. Uniform ncu 15.39 ms / 812.8M instructions; clustered 22.46 ms
/ 1223.0M. v11i is the current source.


## Padded coefficient layout (v11m, 2026-10-22 22:02)

| Variant | Change | uniform | clustered | sparse | occupancy |
|---|---|---:|---:|---:|---:|
| v11i | int support starts | 15.53 | 22.61 | 1.902 | 5 blocks |
| v11l | BATCH=14 control | 15.46 | 23.14 | 1.921 | 5 blocks |
| v11m | STAGE=28 padded coefficients + BATCH=14 | **14.45** | **21.14** | **1.81** | 5 blocks |

The v11m row layout is `[sx,sy,sz][strength re,im][x0..x5,0][y0..y5,0][z0..z5,0,0,0]`
= 28 floats. The zero slots are initialized once per warp buffer; `scan_point` can then load
`kx1`, `ky1`, `kz1..kz3` without the `nx/ny/nz` selects. BATCH=14 keeps 5 blocks/SM
(12,544 B/block); the BATCH=14 control alone regressed clustered, but the dependency-chain
shortening recovered more than the batch-size loss. Uniform ncu 14.54 ms / 749.2M
instructions; clustered 20.64 ms / 1111.4M. `relL2` unchanged. v11m is current.


## Five-task stage_coeff pass (v11n, 2026-10-22 22:22)

v11n processed five `(point, dimension)` tasks per coefficient-pass iteration (30 lanes)
instead of one point (18 lanes). It reduced registers 96 -> 94 but added task-mapping
instructions (uniform 749.2M -> 760.9M, clustered 1111.4M -> 1127.4M). Interleaved A/B vs
v11m: uniform 14.515 vs 14.505 ms (flat), clustered 21.359 vs 21.119 (1.1% worse), sparse
1.760 vs 1.815 (3.1% better). Rejected; reverted to v11m. The coefficient pass is not
purely lane-utilization-bound, and the lower register count does not lift occupancy because
shared memory still caps at 5 blocks.


## Occupancy experiment (v11o, 2026-10-22 22:30)

`BATCH=12` with `__launch_bounds__(128, 6)` fits 6 blocks/SM (10,752 B/block) at 80
registers, but ptxas spills 56 B stores / 60 B loads per thread. Active warps rose 4.20 ->
4.74, eligible warps stayed ~1.42, and no-eligible rose 34.3% -> 35.9%. Benchmark: uniform
14.95, clustered 22.05, sparse 1.93 ms, all worse than v11m (14.45 / 21.14 / 1.81).
Rejected; v11m remains current. The extra occupancy does not pay for the spills and smaller
BATCH.


## Option A: register weights + warp shuffles (v11p, 2026-10-22 23:17)

Kept raw-point staging, removed coefficient staging. Lanes 0..17 compute each point's
weights in registers; the scan lanes fetch the 2/2/4 weights with 8 `__shfl_sync`, plus 3
support shuffles and strength from the raw row. The per-point fast/slow branch replaces the
batch-level `__all_sync` decision. Correct (`relL2` unchanged) but much slower: uniform 19.4,
clustered 28.3, sparse 2.45 ms; 107 registers, 4 blocks/SM; ncu uniform 968.4M instructions
(v11m 749.2M), clustered 1440.3M (v11m 1111.4M). The shuffle distribution and fused weight
computation cost more than the shared loads they replaced, and the batch-level fast path plus
the double-buffered coefficient/scan overlap were lost. Rejected; v11m remains current.

## Session log — 2026-10-22 (comprehensive)

Full optimization record for Experiment A. The working source is v11m; all variants are
under `profile-artifacts/expA-output-owned/opt-2026-10-22/`.

### Baseline and gate

- v10: uniform 16.77, clustered 25.88, sparse 2.076 ms spread; ncu 965.5M / 1514.0M;
  95 registers, 5 blocks. Method 3 spread/total: 15.95/16.78 uniform, 15.90/16.79
  clustered, 2.82/2.95 sparse.
- Gate: >=20% faster spread on both `1e6` cases (<=12.76 uniform, <=12.72 clustered);
  total preprocessing+spread <=17.62 ms uniform.

### Variant table

| Variant | Change | Uniform | Clustered | Sparse | Regs/blocks | ncu inst U/C | Verdict |
|---|---|---|---:|---:|---:|---|---|
| v10 | baseline | 16.77 | 25.88 | 2.076 | 95/5 | 965.5/1514.0 | baseline |
| v11a | delete dead second wrap correction | 15.91 | 24.88 | 1.989 | 95/5 | 917.3/1442.0 | kept |
| v11b | per-point fast path | 16.14 | 24.49 | 1.932 | 95/5 | 877.4/1327.8 | rejected; uniform worse |
| v11c | v11b + min/max slow path | 16.33 | 24.23 | 1.938 | 95/5 | 881.3/1327.8 | rejected |
| v11d | per-point relevance filter | 17.43 | 28.06 | 2.039 | 96/5 | 972.6/1575.8 | rejected |
| v11e | fast path only when `nsplit>1` | 17.24 | 24.31 | 2.172 | 95/5 | 990.7/1327.8 | rejected |
| v11f | per-batch `__all_sync` + template scan | 15.65 | 22.99 | 1.948 | 96/5 | 828.0/1248.4 | kept |
| v11g | `STAGE=26` | 17.36 | 24.93 | 2.171 | 96/4 | — | rejected; occupancy |
| v11h | `STAGE=25` | invalid (misaligned) | — | — | — | — | rejected |
| v11i | int support starts | 15.53 | 22.61 | 1.902 | 96/5 | 812.8/1223.0 | kept |
| v11j | prefetch support starts | 16.55 | 23.90 | 2.078 | 102/4 | — | rejected |
| v11k | v11j + launch_bounds(128,5) | 15.42 A/B | 22.78 A/B | — | 96/5 | 816.0/1231.2 | rejected; v11i faster |
| v11l | BATCH=14 control | 15.46 | 23.14 | 1.921 | 96/5 | — | rejected; clustered worse |
| **v11m** | padded coefficients, STAGE=28/BATCH=14 | **14.45** | **21.14** | **1.81** | **96/5** | **749.2/1111.4** | **current best** |
| v11n | five-task `stage_coeff` | 14.515 A/B | 21.359 A/B | 1.760 A/B | 94/5 | 760.9/1127.4 | rejected; flat/worse clustered |
| v11o | BATCH=12 + launch_bounds(128,6) | 14.95 | 22.05 | 1.93 | 80/6 + spills | 756.0/1120.5 | rejected; spills |
| v11p | register weights + warp shuffles | 19.4 | 28.3 | 2.45 | 107/4 | 968.4/1440.3 | rejected |

### Key outcomes

- v11a: removed a dead wrap correction (~48M instructions), first clean win.
- v11b: non-wrapping min/max support cut support instructions 262M -> 139M but the
  per-point branch inflated the scan body; uniform regressed 1.4% vs v11a.
- v11f: moved the decision outside the point loop (one `__all_sync` per batch, then
  `scan_point<true/false>`); kept the support-instruction cut and improved uniform to 15.65.
- v11i: int support starts removed `LDS -> F2I`; 15.53 / 22.61 / 1.902.
- v11m: padded coefficient rows (STAGE=28, BATCH=14) removed the `nx/ny/nz` selects and
  shortened the dependency chain; 14.45 / 21.14 / 1.81, ncu 749.2M / 1111.4M.
- v11n/v11o/v11p: five-task coefficient pass, six-block occupancy, and register/shuffle
  distribution all rejected. The coefficient pass is not purely lane-bound; occupancy is
  lost to spills; shuffles cost more than shared loads on sm_75.
- v11g/h/j/k/l: stride, prefetch, and control variants rejected for occupancy, alignment,
  or neutral/worse A/B results.

### Current profile and gap

- v11m uniform shares: scan_point 48.5%, stage_coeff 30.0%, stage_raw 5.5%; support ~11.9%
  of the kernel; inner accumulation ~16%.
- Shared loads are conflict-free/broadcast; the remaining cost is latency/dependency
  (`short_sb`/`wait`), with eligible warps ~1.3-1.4 and no-eligible ~34%.
- Current gate gap: uniform spread 14.45 vs 12.76 target (10% faster than method 3, needs
  20%); clustered 21.14 vs 12.72 (1.33x slower); uniform total 20.69 vs 17.62 because
  preprocessing is 6.2 ms vs 0.8 ms.
- At current spread, preprocessing must fall to <=3.1 ms for the total guard.

### Artifacts

- `devel/morton_spread/exp_a_cartesian.cu` = v11m; default binaries are v11m.
- `profile-artifacts/expA-output-owned/opt-2026-10-22/` has v10base, v11a-v11p, README,
  and A/B logs (`ab_v11i_v11k.txt`, `ab_v11i_v11m.txt`, `ab_v11m_v11n.txt`).
- ncu reports/source CSVs for captured variants; flamegraphs/calltrees for v11a, v11f,
  v11i, v11m. The profile tool has v11m ranges and the corrected source call tree.
- No library code changed; all prototypes are standalone under `devel/morton_spread/`.

### Lessons and next

- Batch/warp-uniform decisions beat per-point branching.
- Layout/padding and int support starts were the highest-value micro-optimizations.
- 96 registers / 5 blocks is the occupancy sweet spot; forcing 6 blocks spills and loses.
- Warp shuffles are not a substitute for shared broadcasts here.
- Remaining options: preprocessing (uniform total blocker), Option B scatter/shared
  accumulation (architectural alternative, likely clustered-oriented), larger warp regions
  or block-level union staging (structural reread reduction), and clustered split/reduction
  work.

## Exact stage_coeff micro-optimizations (v11q/v11r/v11s, 2026-10-22 23:58)

These variants do not change the kernel, polynomial, or any coefficient values.

- **v11q**: store `s` as an int directly, hoist lane geometry (`cd`, `ci`, `nfd`, support
  offset, coefficient offset) out of the point loop, and move the `lane < 18` predicate
  outside the per-point loop. A/B vs v11m: uniform 14.51 → 14.01 ms, clustered 21.18 →
  20.48 ms, sparse 1.810 → 1.728 ms. ncu 720.7M / 1068.6M instructions; 96 registers,
  5 blocks; `relL2` unchanged (5.49e-7 / 7.2e-7 / 6.76e-8).
- **v11r**: `__float2int_ru` instead of `(int)ceilf`; exact but neutral (13.98 / 20.49 /
  1.731 ms). Rejected.
- **v11s**: pointer increment in the coefficient loop (`row += STAGE`) instead of
  `cur + b*STAGE`. A/B vs v11q: uniform 13.97 → 13.87 ms, clustered 20.53 → 20.40 ms,
  sparse 1.728 → 1.714 ms. Full medians: **13.92 / 20.44 / 1.716 ms**. ncu 717.6M /
  1064.0M; 96 registers / 5 blocks; unchanged `relL2`. Current best.

Current totals: uniform 20.15, clustered 22.17, sparse 2.130 ms. Uniform spread is 13%
faster than method 3 (target 12.76; gap ~1.16 ms); clustered is 1.28x slower. Uniform total
is 1.20x slower; at current spread preprocessing (6.24 ms) must fall to <=3.70 ms. v11s
profile shares: `scan_point` 49.8%, `stage_coeff` 28.3%, `stage_raw` 5.6%.

Artifacts: `opt-2026-10-22/v11q`, `v11r`, `v11s`; A/B logs `ab_v11m_v11q.txt`,
`ab_v11q_v11r.txt`, `ab_v11q_v11s.txt`; v11s ncu reports/source CSVs, Chrome flamegraphs,
and calltrees. Profiler ranges are updated to v11s; default binaries are v11s.

## Float-ceil `zz` fix (v11t) and two-point unroll (v11u, 2026-10-23 00:10)

- **v11t** uses the float `ceilf` result for `zz` and stores the int support start
  separately. This preserves the original float arithmetic and removes the int→float
  dependency. A/B vs v11s: uniform 13.95 → 13.83 ms, clustered 20.43 → 20.34 ms, sparse
  1.718 → 1.704 ms. Full medians: **13.82 / 20.35 / 1.703 ms** (totals 20.06 / 22.07 /
  2.118). ncu 717.6M / 1064.0M; 96 registers / 5 blocks; unchanged `relL2`. Stage_coeff
  27.0%, scan_point 50.8%, stage_raw 5.8%. Current best.
- **v11u** unrolls `stage_coeff` by two points. Horner wait samples fell 5,916 → 4,591 and
  Horner samples 16,458 → 13,814, but the two-call loop's control/address instructions grew
  from 6.99M to 18.88M. Total instructions rose 717.6M → 729.4M (+1.65%), clustered
  1064.0M → 1081.7M, and the kernel regressed to 14.13 / 20.78 / 1.713 ms despite 96
  registers / 5 blocks. Rejected; the simple unroll is control-bound. A hand-written
  `stage_coeff2` with one shared control/address path could recover the ILP, but it is not
  a quick win.

Current gate: uniform spread 13.82 vs 12.76 target (gap ~1.06 ms); clustered 20.35 vs 12.72
(1.28x slower); uniform total 20.06 vs 17.62; preprocessing 6.23 ms must fall to <=3.80 ms.
Artifacts: `v11t/`, `v11u/`, A/B logs, v11t ncu/Chrome traces/calltrees; profiler ranges
updated to v11t; default binaries v11t.

## Raw/strength-in-registers variants (v11v/v11w, 2026-10-23 00:21)

- **v11v** loads the next batch's raw x/y/z/strength before `stage_coeff`, holds the five
  values across the scan, and stores them afterward. It compiles to **101 registers, 0
  spills**, which drops occupancy to **4 blocks/SM** (the 5-block limit is effectively 96
  registers after allocation granularity). ncu active warps 4.20 → 3.32; spread regresses
  to 15.05 / 21.40 / 1.905 ms.
- **v11w** loads before `stage_coeff` and stores right after it, before the scan, shortening
  the live range. It stays at **96 registers / 5 blocks, 0 spills**, but is neutral/slightly
  worse: 13.83 / 20.33 / 1.724 ms. The global-load stall moves to the deferred store instead
  of being hidden; ncu still shows ~7.7k long-scoreboard samples.

Conclusion: raw/strength-in-registers has no register headroom at the current 96-register
budget. The full shuffle version needs more live values and would also fall to 4 blocks, so
it is not viable without freeing registers elsewhere. v11t remains current best; v11t source
and default binaries restored.

## Method-3 baseline profile at v11as (2026-10-23)

Fresh profile of the comparison baseline. In this checkout the method-3 spread kernel is
`spread_output_driven<float, 1, 3, 6>` (there is no symbol named `spread_cartesian`); the
prototype kernel is `spread_cartesian_owned` / `spread_cartesian_owned_block_shared`.
Full artifacts and commands:
[`profile-artifacts/baseline-method3-2026-10-23/`](../../../profile-artifacts/baseline-method3-2026-10-23/README.md).

Fresh head-to-head, second process run, 2 warm-ups + 5 timed runs:

| Case | method 3 spread | v11as spread | v11as / m3 | method 3 total | v11as total |
|---|---:|---:|---:|---:|---:|
| uniform `1e6` `64^3` (`qb=1`, radix, per-warp) | 15.944 | **13.683** | **0.858 (1.17x faster)** | 16.761 | 20.322 |
| clustered `1e6` `64^3` (`qb=4`, counting, block-shared, split 56) | 15.892 | **14.038** | **0.883 (1.13x faster)** | 16.814 | **15.878** |
| sparse `1e5` `128^3` (`qb=4`, radix, per-warp) | 2.823 | **1.700** | **0.602 (1.66x faster)** | 2.957 | 2.356 |

This reproduces the recorded v11as medians (13.85 / 14.03 / 1.712 ms spread; totals
20.53 / 15.87 / 2.41 ms), so v11as remains current best. It keeps the v11ab per-warp
kernel for uniform/sparse and switches to block-shared staging for clustered, selected by
measured block/warp visits. Clustered spread falls 18.07 -> 14.03 ms and total 19.87 ->
15.87 ms; uniform/sparse are unchanged apart from the block-visit metadata pass
(~0.03-0.04 ms).

ncu: method 3 uniform 15.75 ms / 380.2M warp instructions / 51 regs / 65.15 KB shared /
25% occupancy / 2.00 active warps per scheduler / 71.9% no-eligible. v11as uniform is
13.77 ms / 717.6M instructions / 95 regs / 12.54 KB / 5 blocks / 4.20 active warps /
33.3% no-eligible. v11as clustered is 13.86 ms / 890.2M instructions / 94 regs /
4.87 active warps / 24.9% no-eligible. It executes ~1.9x more instructions on uniform and
~2.35x more on clustered, but wins uniform by raising occupancy/removing the per-point
barrier and wins clustered by sharing the staged block batch (stage_coeff instructions
fall roughly 4x). The remaining clustered cost is the scan, support placement, and
block-barrier/loop overhead.

Method 3 stall shares (uniform / clustered / sparse): `wait` 32.5 / 32.6 / 37.3%, barrier
25.4 / 25.5 / 26.0%, shared-memory dependency 21.6 / 21.6 / 14.3%. Dense/clustered is
barrier + shared-latency bound at 25% occupancy (65 KB local subgrid). Sparse adds merge
index/global-merge work: the merge index modulo line is 15.9% of sparse source samples,
and the global merge loop another 4.7%; that is the region v11as removes, which is why
sparse is the strongest win. Uniform v11as is 13.7 ms vs the 12.76 ms gate target
(~7.2% above), and the uniform total is still blocked by 6.21 ms of preprocessing plus
0.43 ms of metadata vs method 3's 0.81 ms `setpts`.

## Clustered split-count tuning (v11x/v11y/v11z, 2026-10-23 00:39)

The old split rule stopped clustered at 8 because 64 active tile-warps × 8 reached the
512-warp target. Raising the target:

| split | clustered spread | total |
|---:|---:|---:|
| 8 (v11t) | 20.34 ms | 22.07 ms |
| 16 (v11x) | 19.13 ms | 20.85 ms |
| **32 (v11y)** | **18.75 ms** | **20.47 ms** |
| 64 (v11z) | 19.74 ms | 21.46 ms |

v11y raises ncu active warps/scheduler 4.22 → **4.70**, eligible warps 1.47 → **1.67**, and
lowers no-eligible 33.1% → **28.5%**. Uniform/sparse stay nsplit=1. Clustered `relL2` is
~2.6e-7 with the changed split summation order; still safe. v11y is current best.

Current: uniform 13.72 / 19.98 ms, clustered 18.75 / 20.47 ms, sparse 1.701 / 2.119 ms.
Gate: uniform spread gap ~0.96 ms; clustered gap ~6.0 ms (1.18x slower than method 3);
uniform total gap ~2.36 ms; preprocessing 6.26 ms must fall to <=3.90 ms.

The `active_warps=64/512` figure is tile-warps with points, not runtime occupancy. 32 splits
is the knee; 64 splits loses to split reduction and empty splits. All 128 tiles ×32 blocks
still launch even though only 16 tiles have points; active-tile compaction or a persistent
work queue could recover some overhead, but the dominant clustered cost is the **8.0× point
reread**. Per-visit throughput is already similar to uniform, so closing the clustered gate
needs reduced reread/ownership change (scatter/shared accumulation or block-level sharing).

Artifacts: `v11x/`, `v11y/`, `v11z/`; A/B logs; v11y ncu reports, Chrome traces, calltrees;
default binaries/source v11y.

## Active-tile compaction and metadata accounting (v11aa, 2026-10-23 02:00)

The kernel parallelizes over output tiles/warp regions, not bins. Each warp enumerates all
bins in its region+halo; empty bins are skipped by the x-run point-range check, and empty
tiles are still launched. `kernel_visits` computes per-warp visits/active flags on device,
but was outside the timed spread interval; the host used the copied flags for the nsplit
decision.

v11aa:

- keeps `kernel_visits` as a separate launch-setup pass (it cannot be inside the spread
  kernel because the host needs its result for nsplit/grid);
- adds a device-side `kernel_compact_tiles` that writes active tile IDs in tile order
  (single-threaded, no atomics);
- launches `active_tiles × nsplit` blocks and maps through the tile list;
- times `kernel_visits` + compaction + small D2H copies as `meta_ms`;
- reports `total = preprocess + meta + spread` (honest total).

Results: clustered active tiles 16/128, spread 18.75 → **18.07 ms**, total 20.47 → **19.87
ms**. Uniform/sparse have all tiles active, so grid is unchanged. Metadata costs **0.51 ms
uniform**, **0.08 clustered**, **0.19 sparse**; uniform `kernel_visits` dominates because qb=1
makes each warp enumerate a large bin cube. 95 registers / 5 blocks; clustered relL2 ~3.0e-7;
no atomics in the spread kernel.

Honest totals: uniform 13.83 / 20.57 ms, clustered 18.07 / 19.87 ms, sparse 1.716 / 2.330 ms.
Gate: uniform spread gap ~1.07 ms; uniform total gap ~2.95 ms and `preprocess+meta=6.74 ms`
must fall to <=3.79 ms; clustered spread gap ~5.35 ms, total gap ~2.25 ms. Old v11y totals
understated setup by ~0.51 ms uniform / ~0.19 ms sparse.

Artifacts: `v11aa/` source/bench, ncu reports/source CSVs, Chrome flamegraphs, and
calltrees; default binaries; READMEs updated. v11aa ncu: uniform 13.76 ms / 717.6M;
clustered 17.21 ms / 1060.4M, active warps/scheduler 4.73, eligible 1.71, no-eligible
27.6%; clustered stage_coeff 31.2%, scan_point 53.3%, stage_raw 3.1%.

## Parallel kernel_visits / metadata visibility (v11ab, 2026-10-23 02:12)

`kernel_visits` uniform was 472 µs with only 2 blocks / 512 threads, active warps/scheduler
2.00, no-eligible 76.5%: it was parallelism-starved, not fundamentally bin-bound. v11ab uses
one warp per warp region; the 32 lanes split the flattened bin cube, then warp-reduce. Uniform
`kernel_visits` drops to **0.313 ms**, compaction 0.015 ms, metadata **0.38 ms** (was 0.51).
`visits`, `compact`, `meta`, `setup`, `total` are printed separately; NVTX ranges around the
metadata kernels appear in nsys Chrome timelines.

Current v11ab medians: uniform 13.70 / 20.35 ms, clustered 18.07 / 19.87 ms, sparse
1.710 / 2.337 ms. Gate: uniform spread gap 0.94 ms; uniform total gap 2.73 ms and
preprocess+meta=6.65 ms must fall to <=3.92; clustered spread gap 5.35 ms, total gap 2.25 ms.

Further bin-enumeration options: 3D prefix/summed-area table (few lookups per warp cube),
one block per region for more parallelism, or skip compaction when all tiles are active.
Uniform has essentially no empty bins, so skipping empty bins will not help it.

Artifacts: `v11ab/` source/bench; ncu source-level flamegraphs/calltrees (uniform 13.77 ms /
717.6M, clustered 17.37 ms / 1060.4M; stage_coeff 27.0%/31.4%, scan_point 50.7%/53.1%,
stage_raw 5.7%/3.1%); nsys Chrome timelines including `kernel_visits`; nsys-rep files;
default binaries updated.

## 3D prefix/summed-area table rejected (v11ac/v11ad, 2026-10-23 02:22)

- v11ac: 3D prefix of bin counts, one thread per prefix line. Uniform sat_build 0.242 ms,
  visits 0.019 ms, compact 0.015 ms, meta 0.414 ms (v11ab 0.38). Clustered sat 0.047,
  visits 0.029, meta 0.145 (v11ab 0.08).
- v11ad: `cub::BlockScan` per prefix line. Uniform sat 0.286 ms, visits 0.018 ms, meta
  0.471 ms — worse. Thousands of tiny scan blocks cost more than serial lines.

The SAT query is fast (0.018 ms for all uniform warp regions), but prefix construction
dominates. SAT is not a win unless the prefix can be built during preprocessing or with a
fused scan. Rejected; v11ab remains current. Uniform metadata is now 0.38 ms; the remaining
uniform setup gap is the 6.2 ms Cartesian preprocessing. v11ab source/binaries restored;
nsys Chrome timelines include kernel_visits/compaction/preprocessing/spread.

## Block-per-region kernel_visits sweep (v11ae, 2026-10-23 02:28)

Swept one block per warp region at 32/64/128/256 threads against v11ab's one-warp-per-region
path:

| workload | warp/region | 64 threads | 128 threads |
|---|---:|---:|---:|
| uniform visits | 0.316 ms | **0.293 ms** | 0.296 ms |
| uniform meta | 0.385 ms | **0.363 ms** | 0.364 ms |
| clustered visits | 0.023 ms | 0.022 ms | 0.031 ms |
| sparse visits | **0.055 ms** | 0.079 ms | 0.108 ms |

Uniform gains only ~0.02 ms; clustered neutral; sparse regresses due to block launch and
reduction overhead for 1024 regions. Rejected; v11ab remains current. Further metadata
tuning is not worthwhile; the remaining uniform setup gap is the 6.2 ms preprocessing.

# Final comprehensive summary — v11ab baseline, 2026-10-23 (superseded by v11as addendum below)

Current best: **v11ab**. Totals now include launch-setup metadata, matching the baseline's
timed `setpts + execute`.

| workload | spread | preprocess | meta | setup | total | method 3 total |
|---|---:|---:|---:|---:|---:|---:|
| uniform 1e6 64^3 qb=1 radix | **13.70 ms** | 6.26 | 0.38 | 6.65 | **20.35 ms** | 16.78 |
| clustered 1e6 64^3 qb=4 counting split 32 | **18.07 ms** | 1.72 | 0.08 | 1.80 | **19.87 ms** | 16.79 |
| sparse 1e5 128^3 qb=4 radix | **1.710 ms** | 0.42 | 0.21 | 0.63 | **2.337 ms** | 2.95 |

- 95 registers / 5 blocks; no spills.
- relL2 5.49e-7 uniform, ~2.6e-7 clustered, 6.76e-8 sparse.
- Uniform ncu 13.77 ms / 717.6M; clustered 17.37 ms / 1060.4M.
- Uniform source shares: scan_point 50.7%, stage_coeff 27.0%, stage_raw 5.7%.
- Clustered source shares: scan_point 53.1%, stage_coeff 31.4%, stage_raw 3.1%.
- Clustered active tiles 16/128; active warps/scheduler 4.73, eligible 1.71, no-eligible 27.6%.

## Changes after v11t/v11y

- v11q: int `s`, hoisted lane geometry, predicate outside point loop; −3.4% uniform,
  −3.3% clustered, −4.5% sparse vs v11m. Kept.
- v11r: `__float2int_ru`; neutral; rejected.
- v11s: pointer increment in the coefficient loop; small win; kept.
- v11t: float `ceilf` result used for `zz`, int support store; removes int→float dependency;
  kept.
- v11x/v11y/v11z: clustered split sweep. 16 splits 19.13 ms; **32 splits 18.75 ms**;
  64 splits 19.74 ms (reduction overhead). v11y = 32 splits.
- v11aa: active-tile compaction; device-side deterministic tile list; launch only active
  tiles. Clustered grid 4096 → 512 blocks; clustered spread 18.75 → 18.07 ms. Metadata now
  timed: kernel_visits + compaction + D2H = 0.51 uniform, 0.08 clustered, 0.19 sparse.
  Honest totals introduced here.
- v11ab (current): parallel `kernel_visits` (one warp per region). Uniform metadata
  0.51 → 0.38 ms (visits 0.313, compaction 0.015); total 20.57 → 20.35 ms. Added separate
  visits/compact/meta/setup/total output and NVTX ranges; nsys Chrome traces include
  kernel_visits.

## Rejected recent variants

- v11u: two-point stage_coeff unroll; Horner waits fell but loop/control instructions rose
  6.99M → 18.88M; +1.65% total instructions; regressed.
- v11v: raw/strength in registers across scan; 101 regs → 4 blocks/SM; 15.05 / 21.40 /
  1.905 ms despite no spills.
- v11w: raw registers live only across stage_coeff; 96 regs / 5 blocks but neutral/slightly
  worse; load stall moved to deferred store.
- v11ac: 3D SAT, one thread per prefix line; query 0.019 ms but build 0.242 ms; meta
  0.414 ms; worse than v11ab 0.38.
- v11ad: SAT with cub::BlockScan prefix lines; build 0.286 ms, meta 0.471 ms; worse.
- v11ae: block-per-region kernel_visits 32/64/128/256. Uniform 64/128 saves only ~0.02 ms;
  clustered neutral; sparse visits 0.055 → 0.074/0.110 ms. Rejected.

## Block-level coefficient-sharing measurement (v11af)

Diagnostic block-region visits for the `16×16×8` block:

| workload | block reread | warp reread | coefficient saving | extra scan if all 4 warps scan |
|---|---:|---:|---:|---:|
| uniform | **3.31** | 5.36 | 38.2% | +147% |
| clustered | **2.00** | 8.00 | **75.0%** | **0%** |
| sparse | **4.49** | 8.00 | 43.9% | +124% |

The earlier ~5.3 estimate was too pessimistic. Clustered is the standout: 4 warps × 2.00
block reread = 8.0 warp scans, exactly the current work, so block-level coefficient sharing
can remove ~75% of clustered stage_coeff/stage_raw work without extra scans. Uniform needs
a per-warp filter for the +147% early-out scans. This makes block-wide coefficient sharing
the top structural candidate, especially clustered.

## Metadata / baseline accounting

`kernel_visits` is a device kernel but cannot be inside `spread_cartesian_owned`: the host
needs its result to choose nsplit and the grid. It is a launch-setup pass analogous to
baseline method 3 setpts (`indexSort_subprob_and_OD`: bin sizes, binstart scan, subprob
counts, D2H totalnumsubprob, subprob_to_bin map). The harness times setpts and execute
separately; baseline total includes both. Our fair total is preprocess + meta + spread.

Uniform kernel_visits single-thread was 472 µs with 2 blocks / 512 threads and 76.5%
no-eligible. Warp-per-region v11ab is 313 µs. Baseline setpts is ~0.8 ms; our
preprocess + meta is 6.65 ms. The dominant gap is the 6.26 ms Cartesian
binning/radix preprocessing, not the 0.38 ms metadata.

## Traces

- ncu source flamegraphs/calltrees (v11ab):
  `v11ab/uniform_qb1_flamegraph_chrome.json`,
  `v11ab/clustered_qb4_flamegraph_chrome.json`, and matching `*_calltree.txt`.
  ncu captures only spread_cartesian_owned; widths are PC samples scaled to kernel time.
- nsys wall-clock Chrome timelines (v11ab):
  `v11ab/uniform_nsys_chrome.json`, `v11ab/clustered_nsys_chrome.json`.
  Include kernel_visits, kernel_compact_tiles, preprocessing, spread, memcopies, NVTX.
- `ncu_samples_to_chrome.py` ranges/labels are current for v11ab.

## Gate status

- uniform spread 13.70 vs 12.76: gap 0.94 ms; 14% faster than method 3, needs 20%.
- uniform total 20.35 vs 17.62: gap 2.73 ms; at current spread preprocess + meta (6.65 ms)
  must fall to <=3.92 ms.
- clustered spread 18.07 vs 12.72: gap 5.35 ms; total 19.87 vs 17.62: gap 2.25 ms;
  1.14x slower on spread, 1.18x on total.
- sparse 1.710 / 2.337 vs 2.82 / 2.95: ahead.

## What remains

1. Preprocessing: 6.26 ms vs baseline setpts ~0.8 ms. Device-side bin starts, cheaper
   counting/radix, fusing kernel_visits into preprocessing, or reusing baseline ordering.
   This is the main uniform-total gap.
2. Clustered reread 8.0x. v11af showed promising block reread (2.00 clustered, 3.31
   uniform, 4.49 sparse), but the v11ag block-wide prototype was 2–4x slower because of
   block synchronization and phase serialization, even for clustered. Block-level
   coefficient sharing would need a warp-specialized/double-buffered pipeline; the naive
   implementation is rejected. Reducing scan reread still requires atomics or a merge.
3. Exact 3-point stage_coeff packing: not implemented. 9 lanes/point, 3 points/warp,
   existing KER6 sets, second set from shared to stay at 96 registers. Potential ~1–2 ms
   uniform, but substantial.
4. Stop tuning kernel_visits: warp-per-region is the sweet spot; SAT build,
   block-per-region, and further metadata work are exhausted.

## Artifacts

- Current source/binary: devel/morton_spread/exp_a_cartesian.cu,
  build/gpu/exp_a_cartesian[_lineinfo] = v11ab.
- profile-artifacts/expA-output-owned/opt-2026-10-22/v11aa .. v11ae.
- v11ab has bench, ncu flamegraphs/calltrees, nsys Chrome timelines, nsys-rep files.
- A/B logs include ab_v11y_v11aa.txt, ab_v11t_v11u.txt, ab_v11t_v11v.txt,
  ab_v11t_v11w.txt, ab_v11x_v11y.txt, and v11ae/sweep_repeats.txt.
- Tracker/research notes are local/untracked.

## Block-wide coefficient-sharing prototype rejected (v11ag, 2026-10-23 03:03)

Implemented the block-wide design: block-shared staged batch, distributed stage_coeff,
4-bit per-warp relevance mask, __syncthreads between stage and scan. Correct (relL2
unchanged) but much slower:

| workload | v11ab | v11ag block-wide |
|---|---:|---:|
| uniform spread | 13.7 ms | 52.0 ms |
| clustered spread | 17.9 ms | 37.2 ms |

Uniform ncu: 51.9 ms, 1.882B instructions (v11ab 717.6M), active warps/scheduler 4.41,
eligible 0.64, no-eligible 55.7%. Even clustered, where all four warps are relevant and the
mask is all-valid, the block-wide phase/synchronization pipeline dominates. The v11af
block-reread measurement still shows potential, but realizing it requires a warp-specialized,
double-buffered producer-consumer pipeline, not a naive block-wide sequential batch.
Rejected; v11ab restored.

## Scan metadata vectorization rejected (v11ah, 2026-10-23 04:47)

Replaced the scan's `int2` sxy + scalar sz + two strength loads with one aligned int4
load (`sx,sy,sz,strength.x`) plus one scalar strength.y. Correct, 95 regs / 5 blocks,
but neutral: uniform 13.73 vs 13.69 ms, clustered 18.06 vs 18.07, sparse 1.709 vs 1.711.
SASS is effectively identical to v11ab: both have 7 LDS.U.128 in the spread kernel; only a
commuted IADD3 differs. ptxas had already coalesced the metadata accesses. Rejected;
v11ab restored.

## Warp-local relevance mask rejected (v11ai/v11aj, 2026-10-23 05:00)

The mask is correct and filters real work: mask0 fraction 0.1996 uniform, 0.0773
clustered, 0.4634 sparse. All implementations are slower than v11ab:

| variant | uniform | clustered | sparse |
|---|---:|---:|---:|
| v11ab | 13.69 | 18.07 | 1.711 |
| v11ai ballot mask, stage-only | 19.7 | 24.5 | 2.82 |
| v11aj transposed staging + mask | 15.54 | 18.50 | 1.897 |
| v11aj transposed staging, no mask | 13.94 | 17.97 | 1.772 |
| v11aj stage mask, unmasked scan | 14.17 | 18.41 | 1.797 |

v11ai needed 121 regs before simplifying the overlap test; at 96 regs the ballot/control
still dominated. v11aj removed the ballot by computing s/zz/mask one point per lane and
parking zz in the row pads, but the transpose alone is neutral and the mask branch/scan
filtering is a net loss. Closed; v11ab restored.

## Per-batch compact valid list (v11ak, 2026-10-23 05:20)

One lane per point computes the mask; one ballot/popcount per batch builds a dense
valid-index list; Horner and scan iterate the list with no per-point mask branch.
All-valid batches keep the contiguous fast path.

| variant | regs / blocks | uniform | clustered | sparse |
|---|---:|---:|---:|---:|
| v11ab | 95 / 5 | 13.69 | 18.07 | 1.711 |
| v11ak one-point compact | 96 / 5 | 14.17 | 18.03 | **1.569** |
| v11ak two-point compact | 102 / 4 | — | — | — |
| v11ak two-point capped 96 | 96 / 5, 16 B spills | 14.92 | 18.37 | 1.887 |
| v11ak always compact | 119 / 4 | 15.03 | 20.09 | 1.745 |

Best mask implementation: sparse improves 8.3%, clustered neutral, uniform loses because
the compact list gives up two-point ILP; the two-point list exceeds 96 registers. Rejected
as overall; v11ab restored.

## In-place row compaction / second-buffer check (v11al, 2026-10-23 05:35)

To keep two-point scan ILP, valid rows were compacted to the front of the existing stage
buffer. In-place ascending copies are safe because valid[v] >= v. Results:

| variant | regs / blocks | uniform | clustered | sparse |
|---|---:|---:|---:|---:|
| v11ab | 95 / 5 | 13.69 | 18.07 | 1.711 |
| v11al metadata-only copy | 104 / 4 | not timed | — | — |
| v11al full-row copy | 100 / 4 | 13.87 | 19.38 | 1.815 |

The copy/list path pushes the kernel off the 5-block register cliff. A dedicated second
buffer is worse: 5 blocks already use 63,280 B of the 64 KB shared pool; only ~2.2 KB/SM
remains. Rejected; v11ab restored.

## Mechanical v11ab source refactor (2026-10-23)

Renamed identifiers to the explanatory names (`stage_raw_points`,
`prepare_point_coefficients`, `place_support`, `scan_one_staged_point`,
`staged_point_values`, `current_batch_values`, `output_cell_accumulators`, etc.).
No layout or arithmetic changes. ptxas: 95 registers / 12,544 B / 5 blocks; spread
SASS identical to pre-refactor v11ab. v11ab ncu source artifacts regenerated against
the new line numbers.

## Direct SoA / padding-free layout rejected (v11am, 2026-10-23)

Layout: staged_point_values[warp][buffer][value][point], 23 values per point
(3 support starts + 2 strengths + 18 coefficients), no zero-pad slots.
Shared: 10,304 B/block (vs 12,544 AoS), 96 regs, 5 blocks.

| variant | uniform | clustered | sparse |
|---|---:|---:|---:|
| v11ab AoS padded | **13.69** | **18.07** | **1.711** |
| v11am SoA no pads, BATCH14 | 14.92 | 19.72 | 1.806 |
| v11am SoA no pads, BATCH16 | 16.50 | 22.99 | 2.040 |

The shared reduction is real, but the scan loses the 128-bit metadata load and needs
scalar/guarded coefficient loads; staging gains do not offset it. Rejected; v11ab restored.

## SoA pair-prefetch rejected (v11an, 2026-10-23)

Added aligned int2 support-start / float2 strength loads for two consecutive SoA
points, then scanned both. The pair load saves metadata LDS instructions but keeps
10 metadata registers live across the two scans:

| variant | regs / blocks | uniform | clustered | sparse |
|---|---:|---:|---:|---:|
| v11ab AoS | 95 / 5 | 13.69 | 18.07 | 1.711 |
| v11am SoA no prefetch | 96 / 5 | 14.92 | 19.72 | 1.806 |
| v11an pair prefetch | 104 / 4 | 15.40 | 20.15 | 1.927 |
| v11an capped 96 | 96 / 5, 16 B spills | 14.65 | 19.49 | 1.782 |

Four-point prefetch would need more metadata registers. cp.async is sm_80+ and
global→shared only; unavailable on sm_75 and unusable for shared metadata. Rejected;
v11ab restored.

## SoA odd-stride check (v11ao, 2026-10-23)

Set VALUE_ARRAY_STRIDE = MAX_POINTS_PER_BATCH + 1 = 15 while keeping 23 values/point.
This removes the coefficient-store bank-conflict explosion, but not the LDS count:

| variant | duration (uniform ncu) | LDS inst | store conflicts | store wavefronts |
|---|---:|---:|---:|---:|
| v11ab AoS | 13.90 ms | 49.1M | 0.43M | 12.54M |
| v11am SoA stride14 | 14.86 ms | 63.0M | 5.42M | 18.46M |
| v11ao SoA stride15 | 14.99 ms | 63.0M | 0.062M | 13.11M |

Benchmark medians (5 runs): v11ab 13.79/18.06/1.711; v11ao 14.79/19.69/1.798.
The bank fix recovers only ~1%; extra scalar metadata loads and guarded coefficient
loads dominate. Rejected; v11ab restored.

## 8-cell geometry / 24-warp occupancy rejected (v11ap, 2026-10-23)

Geometry: 8 warps/block (256 threads), tile 16x16x8, warp region 8x8x4, lane group
2x2x2, acc[8], hybrid metadata/coefficient layout. Resources: 79 regs, no spills,
21,760 B shared/block, 3 blocks = 24 warps/SM.

| case | v11ab | v11ap | v11ab visits | v11ap visits |
|---|---:|---:|---:|---:|
| uniform | 13.79 | 14.29 | 5.36M | 7.65M (+43%) |
| clustered | 18.06 | 21.21 | 8.00M | 12.00M (+50%) |
| sparse | 1.711 | 1.984 | 0.80M | 1.20M (+50%) |

The occupancy target was met, but the z-half warp region increased staged point reread
enough to lose. Rejected; v11ab restored.

## 8-cell geometry on pure SoA (v11aq, 2026-10-23)

Same geometry as v11ap but pure SoA/no-padding: 8 warps/block, warp region 8x8x4,
lane group 2x2x2, acc[8], 23 values/point. Resources: 76 regs, no spills, 20,608 B
shared, 3 blocks = 24 warps.

| variant | uniform | clustered | sparse | visits |
|---|---:|---:|---:|---:|
| v11ab | **13.79** | **18.06** | **1.711** | 5.36M / 8.0M / 0.80M |
| v11ap hybrid | 14.29 | 21.21 | 1.984 | 7.65M / 12.0M / 1.20M |
| v11aq SoA | 14.75 | 21.71 | 2.033 | 7.65M / 12.0M / 1.20M |

Pure SoA saves registers but is slower than the hybrid; the dominant cost is the
z-half reread (+43–50%), which is identical across both 8-cell variants. Rejected;
v11ab restored.

## SoA register-recompute sequence (v11ar, 2026-10-23)

On the 8x8x8 SoA layout, tested coefficient preload scoping, direct KER6 table
loads, support bit-packing, and xy-weight recomputation.

| step | registers | blocks | uniform |
|---|---:|---:|---:|
| SoA baseline | 96 | 5 | 14.913 |
| coefficient scope per batch | **91** | 5 | 14.819 |
| direct KER6 loads | 91 | 5 | 15.163 |
| support packing | 93 | 5 | 15.603 |
| recompute xy weights | 91 | 5 | 14.961 |

Coefficient scoping frees only five registers. Direct table loads do not reduce
further; support packing increases registers and slows the kernel; weight
recomputation does not change the register count. No step reaches 80/6 blocks.
Rejected; v11ab restored.

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

## Block-shared staging (v11as, 2026-10-23)

Implemented the block-wide, double-buffered staging variant for the clustered path.
Each block stages one `16×16×8` batch cooperatively: each of the four warps stages a
contiguous quarter of the batch (raw + Horner coefficients), then all four warps scan the
block union for their own `8×8×8` regions. The next batch is staged before scanning the
current one; one `__syncthreads()` per batch publishes it and makes the current buffer safe
to overwrite. No per-point mask. Selected by the measured visit gate
`4 × block_visits ≤ 1.05 × warp_visits` (clustered), not by `nsplit`. `BLOCK_POINTS_PER_BATCH=56`. The split
reduction now sums only active tiles after a `cudaMemsetAsync` of the final grid, instead of
summing the full grid.

Benchmark medians of 3 process runs, 5 timed runs each:

| workload | v11ab spread | v11as spread | v11ab total | v11as total |
|---|---:|---:|---:|---:|
| uniform `1e6` `64^3` qb=1 radix | 13.851 | 13.854 | 20.522 | 20.528 |
| clustered `1e6` `64^3` qb=4 counting | 18.065 | **14.030** | 19.871 | **15.871** |
| sparse `1e5` `128^3` qb=4 radix | 1.710 | 1.712 | 2.378 | 2.410 |

Canonical method-3 clustered is ~15.90 / 16.79 ms, so v11as is now faster on both clustered
spread and total. The strict 20%-faster spread gate (12.72 ms) still misses by 1.31 ms.
`relL2`: 5.49e-7 uniform, 2.5–2.8e-7 clustered, 6.76e-8 sparse. Forced block sharing on
uniform/sparse matches the oracle; a `72^3` partial-tile clustered run is correct
(`relL2` 1.83e-7).

Resources and counters:

- Block-shared kernel: **94 registers, 12,544 B shared, 5 blocks/SM, no spills**.
- Per-warp kernel: 95 registers, 12,544 B, 5 blocks/SM.
- Clustered ncu: **13.94 ms / 886.6M instructions** vs v11ab 17.38 ms / 1060.4M.
- Shared stores: 4.29M vs 17.15M (−75%); shared loads: 73.3M vs 79.7M.
- Barrier stalls: 5.99% (v11ab 0%); eligible warps/cycle: 1.81 (v11ab 1.71).
- Source instruction split: `stage_coeff` 157.1M → 42.8M (4×), scan body essentially
  unchanged. Remaining cost is the scan body (~70% of instructions), support placement
  (~13%), and block barrier/loop overhead.

Tuning: BATCH 14 → 28 → 42 → 56 improved clustered spread 15.62 → 14.94 → 14.77 → 14.64 ms
with full reduction; BATCH=64 loses 5-block occupancy. Active-tile reduction took
14.64 → 14.05 ms. With active reduction, `nsplit` 32 gives 14.05, 40 gives 13.98, 48 gives
13.95, 56 gives 14.03 (final), and 64 regresses; the block-shared path now selects 56,
while uniform/sparse remain at nsplit=1. The remaining serious options are the
32-cells/lane geometry (fewer point reads per output work) or a warp-specialized
producer/consumer pipeline to overlap staging and reduce barrier skew.
Stage_coeff is now only ~4.4% of instructions, so further coefficient-pass work has little
headroom.

Dispatch is now based on measured candidate reads: `kernel_visits` gives `warp_visits`
(5.36M uniform, 8.0M clustered, 0.80M sparse) and a new `kernel_block_visits` over the
active tile list gives `block_visits` (3.31M uniform, 2.00M clustered, 0.45M sparse).
The host selects block sharing only when `WARPS_PER_BLOCK * block_visits <= 1.05 *
warp_visits`; this enables clustered (8.0M vs 8.0M) and disables uniform (13.24M vs 5.36M)
and sparse (1.80M vs 0.80M). `nsplit` is now only a performance knob. The block-visit pass
costs ~0.018/0.040/0.030 ms clustered/uniform/sparse; uniform/sparse kernel time is
unchanged.

Committed as **ee88a2fc** (`devel/morton_spread/`). Artifact:
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11as/` (source, ptxas, bench, ncu
metrics, flamegraphs, README). Flamegraphs:
`uniform_qb1_flamegraph_chrome.json` rooted at `spread_cartesian_owned`, and
`clustered_qb4_flamegraph_chrome.json` rooted at `spread_cartesian_owned_block_shared`
(with `stage_block_quarter`); matching `*_calltree.txt` files are alongside them.

## Preprocessing plan (2026-10-23; implemented as v11at below)

Uniform `1e6` `64^3` qb=1 radix preprocessing is still ~6.2 ms vs method 3's ~0.8 ms
`setpts`, and is the reason uniform total is 20.5 ms instead of the 17.62 ms total gate.
An nsys capture of one preprocessing pass shows the async attribution:

| component | time |
|---|---:|
| `kernel_gather_sorted` | **4.47 ms** |
| radix sort (`SortPairs` + onesweep/histogram) | ~0.6 ms |
| `kernel_compute_qbin` | ~0.18 ms |
| `kernel_mark_starts` | ~0.07 ms |
| memsets/memcpys | ~0.3 ms |
| host suffix scan + launch gaps | ~0.6 ms |

The NVTX `preprocess_starts` range is ~6 ms, but that is misleading: the range only
contains host work; the preceding async gather/radix work is charged there when the D2H
copy waits. The two real costs are the **random payload gather** and the **host-side
suffix scan**.

Next experiments, in order:

1. **GPU binstart scan.** Replace `kernel_mark_starts` + host suffix fill + D2H/H2D with:
   histogram of sorted keys into `counts[nbins+1]` (last element zero), then
   `cub::DeviceScan::ExclusiveSum` into `binstart`. Empty bins then fall out naturally,
   and `binstart[nbins] = M`. Removes the host scan, the sentinel fill, and two 1 MB
   copies. Expected ~0.5-1 ms saved; low risk.
2. **Packed payload gather.** For radix sort, pack the original point into one
   `PackedPoint` record (x,y,z,strength re/im) before sorting, then gather one record per
   `perm[i]` instead of four random array loads. The current gather makes four random
   accesses per point (~4.47 ms); one 24/32-byte random sector should cut it substantially.
   Expected preprocessing ~3-4 ms, which would put uniform total near or below the 17.62 ms
   gate even with the current 13.85 ms spread.
3. **Alternative: sort the payload directly.** Test `cub::DeviceRadixSort::SortPairs` with
   a packed payload value type. This removes the gather entirely but may lose occupancy or
   increase sort traffic for a 20-24 byte value. Benchmark against (2).
4. **Counting-sort variant.** With GPU binstart, the counting path can drop the unused
   `sorted_idx` write and may become competitive at qb=1; currently the radix path is much
   faster, but the gap should be re-measured after (1).

Clustered preprocessing is already 1.7 ms (counting, 4096 bins) and sparse is ~0.4 ms; this
work targets the uniform total. The spread gate for clustered (12.72 ms) still needs a
separate scan-side change (32-cells/lane geometry or warp-specialized staging).

## Preprocessing rewrite (v11at, 2026-10-23)

Implemented the preprocessing plan; the spread kernels are unchanged from v11as.

- **Fused qbin + pack**: the radix path computes qbin and packs x/y/z/strength into one
  32-byte record in a single pass.
- **Packed payload gather**: after radix sort, gather one random 32-byte record per
  `perm[i]` instead of four random array reads. `kernel_gather_sorted` 4.47 ms ->
  `kernel_gather_sorted_packed` 1.35 ms.
- **GPU binstart**: radix uses `mark_starts` + reverse-array CUB exclusive scan with
  `next[b] = starts[b] if non-empty else next[b+1]`, then fills `binstart`; counting uses
  the histogram + GPU exclusive sum already added. No host suffix scan or 1 MB D2H/H2D
  bin-table copies remain.

Final medians (3 process runs, 5 timed runs each):

| workload | v11as pre | v11at pre | v11as spread | v11at spread | v11as total | v11at total |
|---|---:|---:|---:|---:|---:|---:|
| uniform `1e6` `64^3` qb=1 radix | 6.218 | **3.088** | 13.716 | 13.840 | 20.381 | **17.371** |
| clustered `1e6` `64^3` qb=4 counting | 1.720 | **1.462** | 14.027 | 14.027 | 15.862 | **15.617** |
| sparse `1e5` `128^3` qb=4 radix | 0.413 | **0.285** | 1.708 | 1.706 | 2.397 | **2.267** |

Uniform total is now **17.371 ms**, inside the 17.62 ms total gate (method 3 ~16.79 ms).
The strict spread gate still fails: uniform 13.84 vs 12.76, clustered 14.03 vs 12.72.
`relL2`: 5.49e-7 uniform, 2.3e-7 clustered, 6.76e-8 sparse; radix-path correctness also
checked on `72^3` uniform/clustered and `qb=2` uniform.

Uniform preprocessing nsys breakdown after the rewrite:

| component | time |
|---|---:|
| `kernel_gather_sorted_packed` | 1.35 ms |
| radix sort | ~0.9 ms |
| `kernel_compute_qbin_pack` | ~0.58 ms |
| mark starts + reverse scan/fill | ~0.2 ms |
| other | ~0.06 ms |

The remaining gap is now entirely spread-side. Uniform needs ~1.08 ms of spread
improvement for the strict spread gate; clustered needs ~1.31 ms. Next options are the
32-cells/lane geometry (fewer point reads per unit of output work) or a warp-specialized
producer/consumer pipeline to overlap staging and reduce barrier skew.

Committed as **92014e4b**. Artifact:
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11at/` (source, ptxas, bench, nsys
preprocessing summary, README, ncu captures and flamegraphs). v11at flamegraphs:
`uniform_qb1_flamegraph_chrome.json` (root `spread_cartesian_owned`) and
`clustered_qb4_flamegraph_chrome.json` (root `spread_cartesian_owned_block_shared`, with
`stage_block_quarter`), plus matching calltrees and source CSVs.

## Three-point coefficient packing (v11au, 2026-10-23)

The per-warp staging path now processes three points per warp iteration. Nine lanes cover
one point's 18 coefficients (three dimensions x two coefficients per lane); 27 lanes are
active and five are idle. Coefficient index `k` uses the register-resident KER6 set, and
`k+3` uses a shared copy of KER6 (216 B). Shared memory grows 12,544 -> 12,760 B, still
five blocks/SM; registers are 96 with no spills. The block-shared clustered path is
unchanged.

Median results (3 process runs, 5 timed runs each):

| workload | v11at pre | v11au pre | v11at spread | v11au spread | v11at total | v11au total |
|---|---:|---:|---:|---:|---:|---:|
| uniform `1e6` `64^3` qb=1 radix | 3.082 | 3.087 | 13.855 | **12.586** | 17.373 | **16.112** |
| clustered `1e6` `64^3` qb=4 counting | 1.462 | 1.466 | 14.035 | 14.029 | 15.630 | **15.624** |
| sparse `1e5` `128^3` qb=4 radix | 0.284 | 0.285 | 1.708 | **1.536** | 2.259 | **2.092** |

- Uniform spread **passes** the strict 12.76 ms gate at 12.586 ms.
- Uniform total passes the 17.62 ms gate at 16.112 ms.
- Clustered total is faster than method 3 (15.624 vs 16.79 ms), but clustered spread is
  still 14.03 vs 12.72 ms.
- `relL2`: 5.49e-7 uniform, 2.29e-7 clustered, 6.76e-8 sparse. Radix-path correctness was
  also checked on `72^3` uniform/clustered, `qb=2` uniform, and forced block-shared uniform.

Uniform flamegraph shares after packing: scan body 61.5%, `prepare_two_coefficients` 12.5%
(down from ~27%), `stage_raw_points` 6.8%. Clustered remains scan-dominated: scan 70.8%,
`stage_block_quarter` 14.3%.

Committed as **674ad159**. Artifact:
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11au/` (source, ptxas, bench, ncu
captures, flamegraphs, README). Remaining gate failure is clustered spread only; next
candidates are the 32-cells/lane geometry or a warp-specialized producer/consumer pipeline.
