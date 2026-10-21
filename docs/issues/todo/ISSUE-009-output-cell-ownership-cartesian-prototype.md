# ISSUE-009: Prototype output-cell ownership with Cartesian bins

**Created:** 2026-10-21
**Priority:** P1
**Status:** Todo
**Labels:** wayfinder:prototype
**Started:** _not started_
**Completed:** _not completed_

**Related commits:**
- _None yet_

**Dependencies:**
- ISSUE-002
- ISSUE-004
- ISSUE-006
- ISSUE-007
- ISSUE-008

## Description

Build the smallest standalone CUDA 3D type-1 FP32 prototype that assigns each fine-grid
cell to exactly one thread/workgroup and accumulates contributions from spatially binned
points, eliminating the global atomic merge. Method 3 already avoids shared atomics (it uses
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
  [Choose the source-bin, output-tile, and run-lookup geometry](ISSUE-008-spatial-geometry-and-run-lookup.md);
- validate on the mandatory workloads from
  [Settle the prototype success protocol and scope](ISSUE-006-prototype-success-protocol.md)
  and report relative L2 error, spread-stage time, effective throughput, occupancy,
  registers/thread, shared memory/block, and where time goes;
- benchmark against `gpu_method=3` and record whether the atomic-free output ownership is
  already enough to move the needle;
- end with a short recommendation for Experiment B: what to keep, what to change, and
  whether Morton ordering is still worth testing.

Link the prototype artifact, commands, results, and recommendation from `## Answer`.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
