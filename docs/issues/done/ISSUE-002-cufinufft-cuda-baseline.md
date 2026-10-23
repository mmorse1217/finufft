# ISSUE-002: Establish the cuFINUFFT CUDA baseline and profiling protocol

**Created:** 2026-10-21
**Priority:** P1
**Status:** Done
**Labels:** wayfinder:task
**Started:** 2026-10-22
**Completed:** 2026-10-22

**Related commits:**
- _None yet_

**Dependencies:**
- _None_

## Description

Build `cufinufft` from this worktree with the CUDA 12.8 toolkit at `~/cuda-12.8` and the
CMake `gpu` preset (native architecture = sm_75), run the existing 3D type-1 tests, and
establish a reproducible spread-stage baseline for `gpu_method=3` (plus methods 1 and 4 for
context). Pin `gpu_method=3` explicitly — auto is method 2 for type 1 — and use
`gpu_spreadinterponly=1` with matching `gpu_kerevalmeth` so the comparison isolates the
spread stage. Confirm `nsys` and `ncu` can profile the spread kernel on the Quadro T1000.

Record:

- exact configure/build commands, compiler and library versions, and any required
  environment (for example `PATH=$HOME/cuda-12.8/bin:$PATH`, `LD_LIBRARY_PATH`,
  `CMAKE_CUDA_COMPILER`);
- a benchmark invocation for 3D type 1 FP32 with `gpu_spreadinterponly=1`, `ns=6`,
  `N=1e5` and `1e6` (and `1e7` only if 4 GB allows), uniform points and one clustered
  distribution;
- spread-stage time and total time for `gpu_method=3`, the kernel names, and launch
  geometry;
- the atomic-related and memory metrics `ncu` can report for that kernel (global/shared
  atomic throughput, L2 hit rate, DRAM throughput, achieved occupancy, registers/thread,
  shared memory per CUDA threadblock);
- container/toolkit gotchas (driver 580 vs toolkit 12.8, `NVIDIA_VISIBLE_DEVICES=void`,
  the 4 GB memory ceiling, FFTW/CPM only if the CPU path is also built).

Write the commands and results to `docs/issues/research/ISSUE-002-baseline.md` and append a
short `## Answer` to this ticket. Do not modify library code.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
- 2026-10-22 15:15 — Claimed by the wayfinder session to establish the cuFINUFFT CUDA baseline and profiling protocol.
- 2026-10-22 15:45 — Baseline established and reconciled with the existing profile artifacts; ticket closed.

## Answer

Baseline established and reconciled with the existing profile artifacts. Full note:
[ISSUE-002-baseline.md](../research/ISSUE-002-baseline.md).

- Build: CUDA 12.8 at `~/cuda-12.8`, CMake GPU build in `build/gpu` with
  `CPM_CCCL_SOURCE=/tmp/cccl-3.4.2`; Quadro T1000, sm_75, 4 GB, driver 580.178.04.
- Event-timed method 3, spread-only: dense `64³` `M=1e6` uniform 18.67 ms execute / 19.81
  ms total; clustered `64³` `M=1e6` 15.75 / 16.63 ms; sparse `128³` `M=1e5` uniform 2.79 /
  2.92 ms; `M=1e7` 156.7 ms uniform and 157.3 ms clustered.
- Launch geometry: dense 1099 CUDA threadblocks × 256 threads; clustered 992 × 256; sparse
  2197 × 256. 65.15 KB dynamic shared memory per CUDA threadblock, 51 registers/thread,
  25% occupancy, one CUDA threadblock per SM.
- Atomics: method 3 has zero shared atomics. The global merge is the only atomic site and
  compiles to `RED`, not `ATOM`. Nsight Compute reports 90.03 MB (`RED` bytes) on dense,
  81.26 MB on clustered, and 179.98 MB on sparse, all at 0% sector hit rate.
- The kernel is barrier/shared-memory-latency bound: global merge is about 2% of dense
  stall samples but about 24% of sparse samples. The 65.15 KB local subgrid is the
  occupancy limit.
- Gotchas: `nsys` flag differences, Nsight Compute timing overhead, `RED` vs `ATOM`, and
  the 4 GB ceiling. The largest tested case is `M=1e7`.
- Implication: the prototype must beat 18.67 ms (uniform dense `1e6`) and 15.75 ms
  (clustered `1e6`), with preprocessing+spread within the success protocol.
