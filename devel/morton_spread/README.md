# Experiment A prototype: output-cell-owned spreading with Cartesian bins

Throwaway prototype for the wayfinder ticket
[Prototype output-cell ownership with Cartesian bins](../../docs/issues/in-progress/ISSUE-009-output-cell-ownership-cartesian-prototype.md)
(Experiment A of the design sketch in
[`docs/issues/assets/ISSUE-001-destination-owned-nufft-design.md`](../../docs/issues/assets/ISSUE-001-destination-owned-nufft-design.md)).

It is **not** part of the library build: nothing under `src/`, `include/`, or the CMake files
is changed. The v8 results and profiler captures are in
[`profile-artifacts/expA-output-owned/v8/`](../../profile-artifacts/expA-output-owned/v8/).

## What it tests

Whether inverting ownership in a 3D type-1 FP32 spread — every fine-grid cell written by
exactly one thread, once, with no global atomics in the spread and no merge — beats
cuFINUFFT `gpu_method=3` (pinned, `gpu_spreadinterponly=1`, `gpu_kerevalmethod=1`) at the
success threshold of
[ISSUE-006](../../docs/issues/done/ISSUE-006-prototype-success-protocol.md), using Cartesian
(not Morton) point bins so the result is not confounded with the ordering experiment.

## Kernel shape (v8)

- Tile `16x16x8`, one CUDA threadblock of **128 threads (4 warps)** per tile.
- Each warp owns an `8x8x8` sub-region; each lane owns a `2x2x4` group (16 cells) and keeps
  16 accumulators in registers. Each cell is stored exactly once, non-atomically.
- Points are sorted by Cartesian query bin `qb` into a dense `bin -> start` table. Two
  preprocessing paths are selectable (`prep` argument): `0` = counting sort (atomic
  histogram + per-bin atomic scatter), `1` = CUB `DeviceRadixSort::SortPairs` of
  (bin, point index) followed by a payload gather and a start-mark/scan pass (default).
- The warp walks its bin cube as **contiguous x-runs**: bins along x are consecutive in the
  sorted point table, so a run is one contiguous point range and is batched in 16s whatever
  the query-bin size (this is what makes `qb=1`, reread 5.36, usable).
- Per 16-point batch: one lane per point issues the coalesced `x,y,z,strength` loads into
  shared memory (double-buffered: the next batch's loads are issued before the current batch
  is scanned); 18 lanes evaluate one Horner coefficient each (the nine polynomial
  coefficients for a lane's index are preloaded into registers once per kernel); the scan
  (`scan_point`) runs two points per iteration for ILP, tests the support branchlessly and
  accumulates the 16 owned cells.
- **Split for clustered inputs**: each tile is launched `nsplit` times; split `s` processes
  batches `start + s*BATCH, start + (s+1+nsplit)*BATCH, ...` of every bin, so all points are
  covered once and hot bins are shared evenly. Each split writes a full partial grid into
  `fw + split*nf` (no atomics, no zeroing); `kernel_sum_splits` sums the partials.
  `nsplit` is picked from the preprocessing visit count (smallest power of two with
  `active_warps * nsplit >= 512`, capped at 8): uniform 512 -> 1, clustered 64 -> 8.
- Kernel values come from cuFINUFFT itself: `fold_rescale` and the ns=6, sigma=2 Horner
  coefficients from `include/cufinufft/contrib/ker_horner_allw_loop.inc`, copied into a
  `__constant__` table and then into per-lane registers.

## Build and run

Needs the CUDA build already configured in `build/gpu` (see
[the baseline note](../../docs/issues/research/ISSUE-002-baseline.md)):

```bash
export PATH=$HOME/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-12.8/lib64:$LD_LIBRARY_PATH

./devel/morton_spread/run_exp_a.sh          # build + mandatory uniform 1e6 case
./devel/morton_spread/run_exp_a.sh all      # mandatory matrix + qb sweep
```

or directly:

```bash
nvcc -O3 -std=c++17 -arch=sm_75 \
  -Iinclude -I/tmp/cccl-3.4.2/thrust -I/tmp/cccl-3.4.2/libcudacxx/include \
  -I/tmp/cccl-3.4.2/cub -isystem=$HOME/cuda-12.8/include \
  -isystem=build/gpu/_deps/poet-src/include \
  devel/morton_spread/exp_a_cartesian.cu -o build/gpu/exp_a_cartesian \
  build/gpu/libcufinufft.a build/gpu/src/common/libfinufft_common.a \
  -lcudart -lcufft -lnvToolsExt -lpthread -ldl -lm

build/gpu/exp_a_cartesian <M> <N1> <N2> <N3> <cluster> <qb> <nrun> [prep]
```

Arguments: `cluster=1` uses the 8 tight Gaussian blobs of the baseline harness, `qb` is the
Cartesian query-bin edge in cells (`qb=1` is best for the uniform spread, `qb=4` for
clustered/sparse), `nrun` is the number of timed runs after two warm-ups (median reported),
`prep` selects counting sort (0) or radix sort (1, default). Each case is run twice by the script because the
first process run has low clocks; read the second. The program times pinned method 3 /
method 4 / method 1 with CUDA events, times preprocessing and the spread (including the
partial reduction when `nsplit > 1`) separately, checks the fine-grid output against method 3
(`relL2`), and reports registers, shared memory, occupancy, candidate point reads, and active
warp regions.

## Readable atomic-free baseline

The readable baseline is split into three files: `atomic_free_spread.cuh` declares the
point/grid types and launch API, `atomic_free_spread.cu` holds the kernels and launch
wrappers, and `atomic_free_spread_bench.cu` is the entry point. It is a standalone
baseline for the same type-1 correctness contract. It is not built by `run_exp_a.sh` and
is not part of the library. Each thread owns one output cell, and each block enumerates the
Cartesian bins around its tile and writes its cells once. Binning/sorting are on the host;
there are no splits, partial grids, active-tile lists, or spreading atomics. It calls
`eval_kernel_vec_horner` directly and reduces support offsets modulo the grid, which makes
it an independent check on the optimized staged scan.

### Staged weight batches

The per-cell loop does **not** evaluate the kernel. A point's weights depend on the point and
the cell offset, not on the thread, so evaluating them per cell repeats the same 18 Horner
polynomials (and the same 18-way coefficient selection) 256 times per block. Instead each
block stages a batch of `STAGE_POINTS = 64` candidate points into shared memory as
**tile-local, zero-filled vectors**: `kx[j][local]` is the weight of staged point `j` for cell
`tile_x + local`, and likewise for y and z, with `0.0` where the point does not cover that
cell (including cells past the end of the grid).

- Three staging lanes per point, one per dimension: `fold_rescale`, `ceilf`, one
  `eval_kernel_vec_horner`, then eight `stage_tile_weights` slots that call the same
  `coefficient_index()` the per-cell loop used to call. 192 of the 256 threads stage; the
  batch costs 5.67 KB of shared memory and two `__syncthreads()`.
- The cell loop then has no support test, no branch, and no coefficient selection: three
  shared loads, two multiplies, two fused multiply-adds per staged point. A zero weight adds
  an exact zero, so the output matches the per-cell form (a non-finite point strength would
  now reach cells the point does not cover).
- The early return for cells outside a partial tile became a store predicate, because a
  thread may not exit while the block still has barriers to reach.

That removes both wastes at once: the Horner work is amortized over the tile instead of
repeated per cell, and the 112-instruction register-select tree becomes one indexed shared
load. Per candidate point per block the kernel drops from ~1355 to ~95 warp instructions.

Measured on the T1000 in this worktree (median of 5, `nrun=5`):

| case | before (`17c05838`) | after | speedup | vs method-3 execute |
|---|---:|---:|---:|---:|
| uniform 1e6, 64^3 | 185.9 ms | **34.0 ms** | 5.5x | 1.79x slower |
| clustered 1e6, 64^3 | 426.9 ms | **36.3 ms** | 11.8x | 1.90x slower |
| sparse 1e5, 128^3 | 19.3 ms | **12.7 ms** | 1.5x | 3.04x slower |
| dense cluster 1e5, 64^3 | 111 ms | **9.3 ms** | 11.9x | 3.16x slower |

The staged z row is pre-scaled by the point strength, so a cell reads two weights and one
`float2` (`kx[x]`, `ky[y]`, `kz*strength[z]`) and does one multiply and two fused
multiply-adds. That cut the cell loop from four shared loads to three and bought 43.8 ->
34.0 ms uniform, 46.1 -> 36.3 clustered, 11.0 -> 9.3 dense.

The full Nsight Compute capture for the four canonical shapes is
`build/gpu/traces/af_profile_20261110_043510/` (`summary.md` plus the CSVs). The kernel is no
longer instruction- or FMA-bound: instructions fell from 16.4B to 0.99B on uniform while IPC
fell from 3.35 to 1.55. It is now bound by the shared-memory **request** rate: LSUIN requests
are at 94% of peak on uniform with the data pipe only 48% busy, so three loads per cell-visit
(96M warp-visits, ~290M warp loads) set a floor of ~30 ms and the kernel runs at 33.7 ms.
`__syncthreads()` is the next term: 18% of warp state on uniform, 46% on sparse, where a
3-point bin forces a batch smaller than `STAGE_POINTS`.

Loop-level ILP does not help at this point: processing two points per iteration with a second
accumulator pair left IPC at 1.55 and LSUIN at 93%, so it was reverted. The remaining lever is
fewer shared loads per cell, which needs several cells per thread (the `exp_a_cartesian.cu`
scan reads ~0.5 weights per cell from a 16-cell lane group and runs the same workload in
12.6 ms).

The staged rows are padded to an odd number of floats (`TILE_X + 1`, `TILE_Y + 1`,
`TILE_Z + 1`). Without the padding the staging lanes write the same `local` of 32 different
points at once, and an 8-float row puts four points on each bank: 46.4M bank conflicts,
6.6 store wavefronts per store instruction. With odd strides the conflicts are exactly 0 and
store wavefronts fall 55.2M -> 8.8M (-10.7% of all shared wavefronts on uniform), which
bought 48.5 -> 43.8 ms uniform, 50.3 -> 46.1 ms clustered, 11.4 -> 11.0 ms dense, and nothing
on sparse (it never had the conflicts, and it is barrier-bound).

It is still **not a performance candidate**: it is 2.3-3.7x slower than method 3, and it
recomputes weights for all 256 cells per point when only ~18 are covered. It is, however, no
longer dominated by redundant point work.

```bash
./devel/morton_spread/run_atomic_free_spread.sh        # build + correctness matrix
./devel/morton_spread/run_atomic_free_spread.sh all    # + bin-size/seed sweep
# or: BUILD_DIR=/path/to/build/gpu ./devel/morton_spread/run_atomic_free_spread.sh
# nrun defaults to 5; override with NRUN=10 ./devel/morton_spread/run_atomic_free_spread.sh
build/gpu/atomic_free_spread_bench M N1 N2 N3 cluster bin_size [pattern [seed [nrun]]]
```

`pattern` selects point placement: `0` = uniform / eight tight clusters (the default),
`1` = exact seam and out-of-domain coordinates, `2` = exact grid nodes and near-node
offsets, `3` = one dense cluster. The binary compares the full fine grid against pinned
method 3 and exits non-zero when `relL2 >= 1e-4` or any structural/consistency check
fails. `run_atomic_free_spread.sh` covers the mandatory shapes plus the boundary patterns;
current results on the T1000:

| case | pattern | relL2 |
|---|---:|---:|
| uniform 1e6, 64^3, qb=4 | 0 | 5.27e-7 |
| clustered 1e6, 64^3, qb=4 | 0 | 6.39e-6 |
| sparse 1e5, 128^3, qb=4 | 0 | 7.00e-8 |
| seam 1e5, 64^3, qb=4 | 1 | 2.58e-6 |
| grid nodes 1e5, 64^3, qb=4 | 2 | 1.50e-6 |
| dense cluster 1e5, 64^3, qb=4 | 3 | 7.03e-6 |
| seam 4096, 6^3, qb=6 | 1 | 7.55e-7 |
| grid nodes 4096, 6x9x13, qb=2 | 2 | 6.21e-7 |
| grid nodes 2e4, 33x35x31, qb=4 | 2 | 2.10e-7 |

The benchmark times host point generation, the two host-to-device uploads, the GPU bin
assignment, the host counting sort, and the spread kernel. The spread time is the median of
`nrun` CUDA-event runs after two warm-ups (min/max are printed too); method 3 `setpts` and
`execute` are timed separately. `prep total` is the sum of the upload/bin/copy/sort phases,
`setup` adds point generation, and `end_to_end` is `setup + spread`; reference work is
validation only and is not included.

Correctness instrumentation checks valid bin IDs, a monotonic offset table with
`offsets[num_bins] == M`, that every output cell was written (the grid is poisoned with a
0xFF NaN sentinel before the final run), finiteness, the global sum against method 3, and
`relL2`. The process exits non-zero if any check fails. The `total` check is not absolute
mass conservation: `gpu_spreadinterponly` returns the raw spread grid, whose total is
scaled by the non-unit kernel-weight sum, so the check compares against method 3's total
instead of the input strength sum.

The seam pattern found a real bug in the optimized `exp_a_cartesian.cu`: its
`place_support()` reduced the support offset by only one grid period, which leaves a
negative offset for lane groups near the far x/y edge when a point sits exactly at the
left seam. That path gave `relL2 = 6.4e-2`. The library owned-spread implementation already
had the multi-period fix; it is migrated back into `exp_a_cartesian.cu` (an `if`/`while`
after the branchless first correction). The atomic baseline was unaffected because its
`coefficient_index()` reduces `cell - support_start` explicitly, and that difference is
bounded by one grid period for a single cell. The fix leaves the mandatory
uniform/clustered/sparse `relL2` values unchanged (5.63e-7 / 2.34e-7 / 7.34e-8 here), but
it is not free: an interleaved A/B on the T1000 moved spread time 12.34 -> 13.13 ms
uniform (+6.4%), 12.69 -> 12.77 ms clustered (+0.6%), and 1.575 -> 1.644 ms sparse
(+4.4%). The **optimized `exp_a_cartesian.cu`** spread speedups against method 3 remain
1.37x / 1.41x / 1.91x, so the mandatory gates still pass. A staging-time normalization
variant measured ~2-3% instead, but this source keeps the same reduction as the library so
the two implementations match.

## Profiling the atomic-free baseline

`profile_atomic_free_spread.py` rebuilds the line-info binary, runs the benchmark,
captures Nsight Compute reports for four canonical shapes (`uniform`, `clustered`,
`sparse`, `dense`), exports raw/detail/source views, and writes a digest plus CSVs:

```bash
./devel/morton_spread/profile_atomic_free_spread.py
# re-analyze an existing capture without touching the GPU:
./devel/morton_spread/profile_atomic_free_spread.py --analyze-only build/gpu/traces/af_profile_<stamp>
# choose cases or skip stages:
./devel/morton_spread/profile_atomic_free_spread.py --cases uniform,clustered --skip-build
```

The default output is `build/gpu/traces/af_profile_<timestamp>/`. Start with
`summary.md`; `summary.json` and `summary_metrics.csv` are the machine-readable
counterparts. `reports/<case>.ncu-rep` is the original report, `ncu/<case>.raw.csv`
contains every raw metric, and `analysis/<case>/` holds the derived tables:

- `source_hotspots.csv` — every source line by instructions and PC samples, with its
  source-file/line, SASS-PC count, and top stall reason;
- `source_stalls.csv` — source line x stall reason;
- `source_groups.csv` — heuristic grouping into support-index/modulo, Horner/FMA,
  weight/select/accumulate, fold, ceil, bin iteration, point load, and branch;
- `sass_opcodes.csv` / `sass_opcode_families.csv` — exact and family-level SASS mix;
- `stalls.csv` — warp-state cycles per issued instruction and shares;
- `metrics_details.csv` — every detail-section metric in long form.

The profiler targets exactly `spread` from `atomic_free_spread.cu`; it deliberately
does not match the reference library's `spread_output_driven`. NCU uses unlocked
clocks and kernel replay, so its duration is not a benchmark median. Source
instructions/samples are deduplicated by `(kernel, SASS PC)`, keeping the first
source correlation, because Nsight repeats PCs under inline definitions and call
sites. The semantic groups are a source-text heuristic, not compiler metadata.

## Source naming

The current v11ab source uses descriptive identifiers. Historical notes and older
artifacts use the original short names; the mapping is:

| old | current |
|---|---|
| `stage_raw` | `stage_raw_points` |
| `stage_coeff` | `prepare_point_coefficients` |
| `dim_support` / `dim_support_nowrap` | `place_support` / `place_support_nowrap` |
| `scan_point` | `scan_one_staged_point` |
| `stage` | `staged_point_values` |
| `row` / `cur` / `nxt` | `staged_point` / `current_batch_values` / `next_batch_values` |
| `acc` | `output_cell_accumulators` |
| `BATCH` / `STAGE` | `MAX_POINTS_PER_BATCH` / `VALUES_PER_POINT` |

This was a mechanical rename only: the spread-kernel SASS is identical to the
pre-refactor v11ab (95 registers, 12,544 B shared, 5 blocks).

## Profiling

Timeline (nsys), source-level (ncu) and call-tree captures of the current kernel:

```bash
export PATH=$HOME/cuda-12.8/bin:$PATH
# NVTX phase markers are compiled in; link -lnvToolsExt.
nsys profile -t nvtx,cuda,osrt --force-overwrite=true \
  -o build/gpu/traces/exp_a_uniform_qb4 \
  build/gpu/exp_a_cartesian 1000000 64 64 64 0 4 1
nsys export --type=sqlite --force-overwrite=true \
  -o build/gpu/traces/exp_a_uniform_qb4.sqlite \
  build/gpu/traces/exp_a_uniform_qb4.nsys-rep
python3 devel/morton_spread/nsys_to_chrome.py \
  build/gpu/traces/exp_a_uniform_qb4.sqlite \
  build/gpu/traces/exp_a_uniform_qb4_chrome.json   # chrome://tracing or ui.perfetto.dev
```

Source-level profile of one launch (requires `-lineinfo`):

```bash
nvcc ... -lineinfo ... -o build/gpu/exp_a_cartesian_lineinfo
ncu --clock-control none -k regex:spread_cartesian_owned -c 1 \
  --section SpeedOfLight --section SchedulerStats --section WarpStateStats \
  --section SourceCounters --section InstructionStats \
  -o build/gpu/traces/exp_a_spread_ncu \
  build/gpu/exp_a_cartesian_lineinfo 1000000 64 64 64 0 4 1
ncu --import build/gpu/traces/exp_a_spread_ncu.ncu-rep --page details
ncu --import build/gpu/traces/exp_a_spread_ncu.ncu-rep --page source --print-source cuda,sass --csv
```

Call-tree flame graph (kernel -> source call hierarchy -> source line) from that CSV:

```bash
python3 devel/morton_spread/ncu_samples_to_chrome.py \
  spread_ncu_source_cuda_sass.csv spread_flamegraph_chrome.json \
  --kernel-us 21650 --text spread_calltree.txt
```

For the clustered block-shared kernel add
`--root spread_cartesian_owned_block_shared`; the uniform/per-warp capture keeps the
default root. v11as uniform/clustered captures and flamegraphs are in
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11as/`.

`--kernel-us` is required: use the duration of the **same ncu launch**, not a benchmark
median (21650 above is an example). Widths are deduplicated **PC-sample shares**, not
wall-clock self time; Chrome's time coordinates are only a synthetic layout scaled to
that duration. The duration-reference track is not a GPU timeline.

The converter counts each `(kernel, PC)` once, keeping its first source correlation, and
reconstructs the **static source call graph** from `exp_a_cartesian.cu` beside the CSV.
Use `--cuda-source <snapshot.cu>` if the matching source lives elsewhere. Source-line
mismatches fail rather than silently applying current line numbers to older captures.
Unknown source remains visible under `unmapped source`; ambiguous shared callees attach
to the root instead of being counted under multiple callers. Flat ncu samples contain
no device call stack. Stall reasons and instruction counts are event args; short-scoreboard
samples describe waiting consumers, not necessarily the triggering loads.

The v11av and v11ay uniform/clustered `*_flamegraph_chrome.json` and `*_calltree.txt`
files have been regenerated with these fixes. CPU-only regression tests:

```bash
python3 -m unittest discover -s devel/morton_spread -p 'test_ncu_samples_to_chrome.py'
```

The complete captures
(v1, v3, v5, v6, v7, v8, v10, `reopen-2026-10-22/`, and `opt-2026-10-22/`) are under
`profile-artifacts/expA-output-owned/`; `opt-2026-10-22/v11ay/` is the current version and
carries the contiguous coefficient rows and strength-scaled accumulation.

## Verdict (current: v11ay contiguous coefficient rows + strength-scaled accumulation)

**v11ay** targets the scan's 16-cell body, which the v11av profile analysis showed was the
real cost (not shared-memory bandwidth): ~6.4 uniform / 4.1 clustered lanes per
accumulation instruction and one `xy*z` multiply plus two FFMAs per cell. Two source
changes:

1. the z row gets three zero pre-pads and uses a contiguous base
   `21 + support_coefficient_offset_z - local_cell_offset_z`, removing the nested
   `local_cell_offset_z` `FSEL` chain; the existing x post-pad doubles as the y pre-pad,
   so y becomes contiguous at no extra shared-memory cost;
2. the scan forms four real and four imaginary `strength*z` factors once per lane group,
   then does one FMA per cell: 8 inner `FMUL`s instead of 16, 40 rather than 48
   weight/accumulate instructions per point.

`VALUES_PER_POINT` is 30, so the 5-block batches are 13 (per-warp) / 53 (block-shared).
Measured medians (4 process runs, 5 timed runs): spread **10.91 / 11.23 / 1.393 ms**,
total **14.45 / 12.83 / 1.956 ms** uniform / clustered / sparse. Both mandatory `1e6`
spread gates now pass: uniform is **31.6%** and clustered **29.4%** faster than method 3;
totals are 13.9% and 23.6% faster. `relL2` is unchanged within the FP32 noise floor
(5.64e-7 / 2.26e-7 / 7.34e-8). Resources: 96 registers (per-warp) / 95 (block-shared),
12,696 / 12,936 B shared, 5 blocks/SM, no spills. ncu warp instructions fall
671.9M -> **560.7M** uniform and 868.6M -> **686.1M** clustered. Details:
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11ay/`.

**v11av** applied the v11au three-point packing to the block-shared clustered staging path
too. Clustered spread fell **14.03 -> 13.71 ms** and total **15.62 -> 15.31 ms**; uniform
and sparse were unchanged at 12.58 / 1.534 ms spread. It remains the base for the
v11ax/v11ay experiments. Details:
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11av/`.

**v11au** introduced three-point coefficient packing:

**v11at** rewrote preprocessing: fused qbin+pack, a
32-byte packed payload gather, and GPU binstart construction (radix: mark-starts + reverse
exclusive scan; counting: GPU exclusive sum). Uniform preprocessing is **6.22 -> 3.09 ms**
and uniform total is **20.38 -> 17.37 ms**, inside the 17.62 ms total gate. Clustered
preprocessing is 1.72 -> 1.46 ms and sparse 0.41 -> 0.29 ms. Spread times are unchanged.
Details: `profile-artifacts/expA-output-owned/opt-2026-10-22/v11at/`.

**v11as** introduced the current clustered path. It keeps the v11ab per-warp kernel for
uniform/sparse and switches to a block-shared
double-buffered batch when measured `4 × block_visits <= 1.05 × warp_visits`. Each of the four warps stages a
contiguous quarter of the batch; all four scan the block union for their own regions. One
`__syncthreads()` per batch, no relevance mask, `BLOCK_POINTS_PER_BATCH=56`, and
active-tile-only split reduction. Clustered spread is **18.07 -> 14.03 ms**, total
**19.87 -> 15.87 ms** (method 3: 15.90 / 16.79 ms). Resources: 94 registers, 12,544 B
shared, 5 blocks/SM, no spills. `relL2` 2.1-2.8e-7. The strict 20%-faster spread gate
(12.72 ms) still misses by 1.31 ms; clustered total is now faster than method 3. Details:
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11as/`.

The v11ab/v11aa history below is the per-warp baseline that v11as keeps for uniform/sparse.

## Previous verdict: v11ab per-warp baseline

This is the per-warp result kept below as the base that v11as uses for uniform/sparse.
Correct (`relL2` 5.49e-7 uniform, 6.8e-7 clustered, 6.76e-8 sparse). Benchmark medians of 4
process runs, best query-bin size and preprocessing per workload:

| case | method 3 (spread / total) | prototype (spread / total) | spread ratio | total ratio |
|---|---:|---:|---:|---:|
| uniform `1e6` `64^3` (qb=1, radix) | 15.95 / 16.78 ms | **13.70 / 20.35 ms** | **1.16x faster** | 1.21x slower |
| clustered `1e6` `64^3` (qb=4, counting, split 32) | 15.90 / 16.79 ms | **18.07 / 19.87 ms** | 1.14x slower | 1.18x slower |
| sparse `1e5` `128^3` (qb=4, radix) | 2.82 / 2.95 ms | **1.710 / 2.337 ms** | **1.65x faster** | **1.26x faster** |

v11f uses one `__all_sync` check per staged batch, then branches once to
`scan_point<true>` (non-wrapping min/max support) or `scan_point<false>` (general periodic
support). v11i stores the support starts as ints. v11m pads the coefficient rows so the
`nx/ny/nz` range selects disappear, at the cost of BATCH=14 (STAGE=28, 12,544 B/block).
v11q then stores the support start as an int directly, hoists the lane geometry out of the
point loop, and moves the `lane < 18` predicate outside it; v11s increments the stage-row
pointer instead of recomputing `cur + b*STAGE`. v11t keeps the float `ceilf` result for `zz`
while still storing the int support start, removing the int→float dependency. All are exact
value-preserving changes. A two-point unroll of `stage_coeff` (v11u) reduced Horner stalls
but added more loop/control instructions and regressed, so it was rejected. Deferring the
next batch's raw/strength loads into registers (v11v/v11w) either dropped occupancy
(101 registers -> 4 blocks) or merely moved the global-load stall to the deferred store,
so that path was also rejected. The warp-local relevance mask was tried in v11ai/v11aj:
mask0 is real (0.20 uniform, 0.077 clustered, 0.46 sparse), but the per-point mask/control
overhead exceeded the filtered work. Per-batch compaction (v11ak) removed the per-point
branch and is the best mask form — sparse 1.711 -> 1.569 ms, clustered neutral — but the
compact list gives up two-point ILP and loses uniform. Compacting the rows in place to
restore ILP (v11al) costs 100–104 registers and loses 5-block occupancy; a dedicated second
buffer would exceed the ~2.2 KB/SM shared-memory headroom. A true SoA/no-padding layout
(v11am) reduced values from 28 to 23 and shared memory to 10,304 B/block, but lost the
128-bit scan metadata load and needed guarded coefficient loads; it was slower
(14.92/19.72/1.806). A two-point SoA metadata prefetch (v11an) raised registers to 104
and dropped occupancy; it was 15.40/20.15/1.927, or 14.65/19.49/1.782 when capped at 96
with spills. `cp.async` is sm_80+ and unavailable on this sm_75 target. An odd SoA
value stride of 15 removed the coefficient-store bank-conflict explosion but only
recovered ~1%, because the scalar metadata LDS count remained; 14.79/19.69/1.798.
An 8-cell geometry/hybrid layout reached 79 regs and 24 warps but increased point
rereads by 43–50%; 14.29/21.21/1.984. The same geometry on pure SoA reached 76 regs
but was slower still (14.75/21.71/2.033). A recompute/liveness pass on the 8x8x8
SoA layout (coefficient scoping, direct KER6 loads, support packing, weight
recomputation) bottomed out at 91 registers / 5 blocks; reaching 80 was not possible
without reducing the accumulator state. v11ab was the best per-warp variant before v11as. v11y raises the clustered split count from 8 to 32; 16 splits
gave 19.13 ms and 64 splits regressed to 19.74 ms, so 32 is the knee. v11aa compacts the active
tile list on device (16/128 tiles for clustered) and times the `kernel_visits`/compaction
metadata separately. It reduces clustered spread to 18.07 ms and total to 19.87 ms. The
metadata is not free: 0.38 ms uniform (after parallelizing `kernel_visits`), 0.08 clustered,
0.21 sparse; `total` now includes it (`total = preprocess + meta + spread`). The original
single-thread `kernel_visits` was 0.51 ms uniform with only 2 blocks / 512 threads; v11ab
gives each warp region a full warp, cutting uniform metadata to 0.38 ms.
Uniform ncu instructions are 965.5M (v10) -> 828.0M (v11f) -> 812.8M (v11i) -> 749.2M
(v11m) -> 717.6M (v11t/v11y). Clustered instructions are 1514.0M -> 1248.4M -> 1223.0M ->
1111.4M -> 1064.0M (v11t), then 1086.8M with 32 splits; the extra split/reduction work is
outweighed by better occupancy (active warps/scheduler 4.22 -> 4.70, no-eligible 33.1% ->
28.5%).

The spread gate (>=20% faster on both mandatory `N=1e6` cases) still fails: uniform is
14% faster, and clustered is 1.14x slower. The uniform total is still 1.21x because
`preprocess + meta` is 6.65 ms vs method 3's 0.8 ms `setpts`; at the current spread it would
need to fall to <=3.92 ms. The metadata pass is now 0.38 ms uniform (parallel `kernel_visits`
0.31 + compaction 0.02 + copies); the remaining setup gap is the 6.2 ms Cartesian
preprocessing. Full trajectory and profiler evidence:
[`profile-artifacts/expA-output-owned/README.md`](../../profile-artifacts/expA-output-owned/README.md)
and the issue research note.
