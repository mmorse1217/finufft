# Experiment A prototype: output-cell-owned spreading with Cartesian bins

Throwaway prototype for the wayfinder ticket
[Prototype output-cell ownership with Cartesian bins](../../docs/issues/done/ISSUE-009-output-cell-ownership-cartesian-prototype.md)
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
- Points are counting-sorted by Cartesian query bin `qb` into a dense `bin -> start` table
  (`kernel_compute_qbin`, `kernel_hist`, `kernel_scatter`: the Cartesian preprocessing).
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

build/gpu/exp_a_cartesian <M> <N1> <N2> <N3> <cluster> <qb> <nrun>
```

Arguments: `cluster=1` uses the 8 tight Gaussian blobs of the baseline harness, `qb` is the
Cartesian query-bin edge in cells (best measured: `qb=4`), `nrun` is the number of timed
runs after two warm-ups (median reported). Each case is run twice by the script because the
first process run has low clocks; read the second. The program times pinned method 3 /
method 4 / method 1 with CUDA events, times preprocessing and the spread (including the
partial reduction when `nsplit > 1`) separately, checks the fine-grid output against method 3
(`relL2`), and reports registers, shared memory, occupancy, candidate point reads, and active
warp regions.

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

Call-tree flame graph (kernel -> inlined function -> helper -> source line) from that CSV:

```bash
python3 devel/morton_spread/ncu_samples_to_chrome.py \
  spread_ncu_source_cuda_sass.csv spread_flamegraph_chrome.json --kernel-us 21650
```

Widths are PC-sample shares rescaled to the measured kernel time; stall reasons are event
args, not frames. The complete captures (v1, v3, v5, v6, v7, v8) are under
`profile-artifacts/expA-output-owned/`; v8 is current.

## Verdict (current: v9 = v8 + double-buffered staging + contiguous x-run batching)

Correct (`relL2` 5.5e-7 uniform, 7e-7 clustered, 7e-8 sparse). Warm event timing, best
query-bin size per workload:

| case | method 3 | prototype | ratio |
|---|---:|---:|---:|
| uniform `1e6` `64^3` (qb=1, reread 5.36) | 15.95 ms | 16.78 ms | 1.05x slower |
| clustered `1e6` `64^3` (qb=4, nsplit=8) | 15.90 ms | 26.00 ms | 1.63x slower |
| sparse `1e5` `128^3` (qb=4) | 2.82 ms | 2.08 ms | **1.36x faster** |

The uniform spread is now at parity; the remaining blocker for the total-time gate is the
Cartesian counting-sort preprocessing (11.4 ms at `qb=1` over 262k bins, 5.6 ms at `qb=4`).
The clustered case is still 1.63x: the split fixed the scheduler starvation (active
warps/scheduler 1.62 -> 4.26) but not the per-point work. Full trajectory and profiler
evidence: [`profile-artifacts/expA-output-owned/README.md`](../../profile-artifacts/expA-output-owned/README.md)
and the issue research note.
