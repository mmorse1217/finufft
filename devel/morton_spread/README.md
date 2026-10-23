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

Widths are PC-sample shares rescaled to the measured kernel time. The hierarchy is
reconstructed from the **static source call graph** (`dim_support` under `scan_point`,
`fold_rescale`/`cudaFMA` under `stage_coeff`); ncu PC sampling is flat and inlined functions
have no runtime call stack. Stall reasons are event args, not frames. The complete captures
(v1, v3, v5, v6, v7, v8, v10, `reopen-2026-10-22/`, and `opt-2026-10-22/`) are under
`profile-artifacts/expA-output-owned/`; `opt-2026-10-22/v11au/` is the current version and
carries the three-point packing on top of the v11at preprocessing and block-shared path.

## Verdict (current: v11au three-point packing + v11at preprocessing + v11as block-shared staging)

**v11au** keeps the v11at preprocessing and v11as block-shared clustered path, and changes
the per-warp coefficient staging to process three points per warp iteration (9 lanes per
point, two coefficients per lane; the second KER6 set comes from shared). Uniform spread
falls **13.86 -> 12.59 ms**, passing the strict 12.76 ms gate; uniform total is **16.11 ms**
and passes the 17.62 ms total gate. Sparse spread is 1.536 ms. Clustered is unchanged at
14.03 ms spread / 15.62 ms total. Details:
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11au/`.

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
