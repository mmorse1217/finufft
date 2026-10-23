# ISSUE-010: Prototype Morton-sorted output-cell-owned spreading

**Created:** 2026-10-21
**Priority:** P1
**Status:** Todo
**Labels:** wayfinder:prototype
**Started:** _not started_
**Completed:** _not completed_

**Related commits:**
- _None yet_

**Dependencies:**
- ISSUE-003
- ISSUE-005
- ISSUE-008
- ISSUE-009

## Description

Add Morton key computation, a CCCL radix sort of `(key, point index)`, construction of the
contiguous sorted points for each bin, and tile-to-bin lookup to the Experiment A prototype. This is the
design sketch's Experiment B: the first test of the complete Morton-sorted output-owned
spread.

The prototype should:

- compute integer bin coordinates from the true point coordinates, form the Morton
  key, and sort keys with their point payload using
  `cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>` with `cub::DoubleBuffer` and
  `end_bit` set to the used Morton bits, per
  [Verify CCCL radix-sort support for Morton keys on sm_75](../done/ISSUE-005-cccl-radix-sort-sm75.md);
- build the compact table mapping each occupied bin to the start offset and count of its points, and the chosen lookup
  from [Choose the bin size, tile size, and sorted-point lookup](../done/ISSUE-008-bin-tile-and-sorted-point-lookup.md);
- let each tile enumerate the bins whose points' kernel support can reach it
  (the tile expanded by `ceil(ns/2)`), including periodic wrap, stage point batches, and
  accumulate owned cells without global atomics;
- measure preprocessing (key computation, sort, run construction) separately from the
  spread kernel, and report total time as well;
- validate against Experiment A and `gpu_method=3` on the mandatory workloads, reporting
  relative L2 error, spread-stage and total time, effective throughput, point reread factor,
  active tile count, occupancy, registers/thread, and shared memory per CUDA threadblock;
- answer whether Morton ordering adds enough over Cartesian bins to justify its
  preprocessing cost, and whether any workload shows a pathological regression.

Link the prototype artifact, commands, results, and recommendation from `## Answer`.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
- 2026-10-22 15:50 — Experiment A resolved with a no-benefit finding (see
  [its answer](../in-progress/ISSUE-009-output-cell-ownership-cartesian-prototype.md) and
  [results note](../research/ISSUE-009-exp-a-results.md)): the Cartesian gather is 3.6-4.6x
  slower than method 3 on uniform `1e6` and 13x on clustered (96/1024 warp regions active),
  and is instruction-bound (~176 instructions per warp-level point read, ~12 reads per
  point). Recommendation: do not run this ticket as scoped. Before it is claimed, settle with
  the human whether Morton ordering is worth testing at all, or re-scope it behind a gather
  redesign (16+ cells per thread, lane-level candidate filtering, active-tile compaction).
