# ISSUE-002: Establish the cuFINUFFT CUDA baseline and profiling protocol

**Created:** 2026-10-21
**Priority:** P1
**Status:** Todo
**Labels:** wayfinder:task
**Started:** _not started_
**Completed:** _not completed_

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
  shared memory/block);
- container/toolkit gotchas (driver 580 vs toolkit 12.8, `NVIDIA_VISIBLE_DEVICES=void`,
  the 4 GB memory ceiling, FFTW/CPM only if the CPU path is also built).

Write the commands and results to `docs/issues/research/ISSUE-002-baseline.md` and append a
short `## Answer` to this ticket. Do not modify library code.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
