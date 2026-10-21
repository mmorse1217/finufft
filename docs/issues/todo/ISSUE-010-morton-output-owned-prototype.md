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

Add Morton key computation, a CCCL radix sort of `(key, point index)`, occupied-bin run
construction, and output-tile source-bin lookup to the Experiment A prototype. This is the
design sketch's Experiment B: the first test of the complete Morton-sorted output-owned
spread.

The prototype should:

- compute integer source-bin coordinates from the true point coordinates, form the Morton
  key, and sort keys with their point payload using
  `cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>` with `cub::DoubleBuffer` and
  `end_bit` set to the used Morton bits, per
  [Verify CCCL radix-sort support for Morton keys on sm_75](../done/ISSUE-005-cccl-radix-sort-sm75.md);
- build the compact occupied-bin run table (key, begin, count) and the chosen run lookup
  from [Choose the source-bin, output-tile, and run-lookup geometry](ISSUE-008-spatial-geometry-and-run-lookup.md);
- let each output tile enumerate intersecting source bins (including the kernel halo and
  periodic wrap), stage point batches, and accumulate owned cells without global atomics;
- measure preprocessing (key computation, sort, run construction) separately from the
  spread kernel, and report total time as well;
- validate against Experiment A and `gpu_method=3` on the mandatory workloads, reporting
  relative L2 error, spread-stage and total time, effective throughput, point reread factor,
  active tile count, occupancy, registers/thread, and shared memory/block;
- answer whether Morton ordering adds enough over Cartesian bins to justify its
  preprocessing cost, and whether any workload shows a pathological regression.

Link the prototype artifact, commands, results, and recommendation from `## Answer`.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
