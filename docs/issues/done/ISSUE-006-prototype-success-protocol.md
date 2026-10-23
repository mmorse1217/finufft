# ISSUE-006: Settle the prototype success protocol and scope

**Created:** 2026-10-21
**Priority:** P1
**Status:** Done
**Labels:** wayfinder:grilling
**Started:** 2026-10-21
**Completed:** 2026-10-21

**Related commits:**
- _None yet_

**Dependencies:**
- _None_

## Description

What exactly must the prototype demonstrate before this effort commits to integration?
Settle, with the human:

- **Baseline.** Is `gpu_method=3` the only comparison, or are methods 1 and 4 also required?
  Is the metric spread-stage time in `gpu_spreadinterponly=1` mode, end-to-end type-1 time,
  or both?
- **Evidence bar.** Is the Quadro T1000 enough for the go decision, or must another GPU
  (A100/H100/Ada) be involved? If only the T1000, what caveat goes in the decision record?
- **Workloads.** Which parts of the design sketch's matrix are mandatory: `N=1e4/1e5/1e6/
  1e7`, uniform/clustered/shell/concentrated distributions, `ns=4/6/8`, tile sizes? Which
  are optional?
- **Success threshold.** The sketch proposes a consistent >20% spread-stage speedup with no
  major pathological regressions. What is the exact threshold, the averaging/aggregation
  rule, and the maximum acceptable regression on any mandatory workload?
- **Experiment sequence.** Run Experiment A (Cartesian bins + output-cell ownership) before
  Experiment B (Morton + output-cell ownership), or go straight to B? If A is skipped, the
  ticket for it is ruled out of scope.
- **Scope of this map.** Type 1 only, or must the type-2 inverse be part of the go/no-go?
  FP32 only, or must FP64 be represented in the decision?
- **No-go.** What result counts as no-go, and what evidence would reopen the decision?

Note from the resolved research: method 3 already avoids shared atomics, so the measured
benefit of output ownership is expected to come from removing the per-tile global merge and
the barrier-per-point serialization, plus the preprocessing histogram atomic. Calibrate the
threshold against that corrected cost model.

Record the decision in `## Answer`; if it changes the map's Destination or scope, update the
map's Destination, Not yet specified, and Out of scope sections.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
- 2026-10-21 20:40 — Claimed by the wayfinder session to settle the prototype success protocol and scope.
- 2026-10-21 21:18 — Resolved with the human; success protocol recorded in Answer; ticket closed.
- 2026-10-23 17:22 — Gate result recorded after Experiment A reached v11ay: both mandatory
  `1e6` spread gates and both totals pass. See `## Gate result (2026-10-23)` below. The
  settled protocol is unchanged; this is an append-only result note.

## Answer

Settled with the human.

**Gate.** Success is measured on the spread stage of pinned `gpu_method=3` with
`gpu_spreadinterponly=1`, same points, strengths, `ns`, `sigma`, `gpu_kerevalmeth`, and fine
grid. The prototype goes if it is at least 20% faster on **both** mandatory `N=1e6`
workloads (uniform and clustered), no mandatory workload regresses more than 5%, and total
preprocessing + spread (`setpts` key/sort/run + spread) is no more than 5% worse than
method 3's corresponding total. Method 4 and method 1 are reported context, not gates.

**Measurement.** Median of 5 runs after 2 warm-ups; CUDA events around the spread kernel;
identical inputs and grid across methods; report min, median, and variance. Nsight and
Nsight Compute are for counters, memory, occupancy, and stall reasons only, not timing.

**Workloads.** Mandatory and acceptance-bearing: 3D type-1 FP32, `ns=6`, `N=1e6` uniform and
one clustered distribution, plus `N=1e5` uniform as a smaller case. Optional and reported
only: `N=1e7` if 4 GB allows, `N=1e4`, `ns=4/8`, shell/concentrated distributions, and the
tile-size sweep. Small-`N` and optional results do not gate.

**Experiment sequence.** Experiment A (Cartesian bins + output-cell ownership) first, then
Experiment B (Morton sort + output-cell ownership), both time-boxed. A isolates output
ownership; B isolates the Morton preprocessing. If A shows no benefit, B is still worth one
run, but the attribution changes.

**Scope.** Type 1, FP32 only for this go/no-go. Type 2 and FP64 are integration-scope
decisions after a go, not prototype success criteria.

**Evidence bar.** The Quadro T1000 is sufficient for the prototype go/no-go. Record the
generalization risk: on sm_75 every complex global atomic is two scalar atomics, and the
4 GB/small-L2 part can inflate the relative win versus sm_90+; a data-center GPU benchmark
remains a post-go recommendation, not a gate.

**No-go / conditional go.** A failure of either `N=1e6` gate, a >5% mandatory regression, or
a >5% total preprocessing+spread regression is no-go. A conditional go is allowed only when
the failure is traced to a named, fixable implementation issue and a re-run passes.
End-to-end type-1 must not regress more than 2%; a larger regression blocks default-on even
if the spread gate passes.


## Gate result (2026-10-23)

Experiment A's final implementation, **v11ay**, clears the settled gate. Medians of 4
process runs with 5 timed runs each; method 3 is the pinned spread-only reference
(15.95 / 16.78 ms uniform, 15.90 / 16.79 ms clustered, 2.82 / 2.95 ms sparse).

| Gate | Result |
|---|---|
| ≥20% faster spread on mandatory uniform `1e6` | **10.912 ms vs 15.95 ms = 31.6% faster** |
| ≥20% faster spread on mandatory clustered `1e6` | **11.232 ms vs 15.90 ms = 29.4% faster** |
| No mandatory workload regresses >5% | none regressed; sparse spread 1.393 vs 2.82 ms |
| Preprocessing + spread total within 5% of method 3 | uniform **14.446 vs 16.78 ms** (13.9% faster); clustered **12.834 vs 16.79 ms** (23.6% faster) |
| Correctness within the reordering noise floor | `relL2` 5.64e-7 / 2.26e-7 / 7.34e-8 |

The `N=1e6` uniform and clustered spread gates are therefore **met**, not waived. The
early v11av result (12.575 / 13.714 ms) already passed uniform and missed the clustered
stretch; v11ay's scan change (contiguous coefficient rows + strength-scaled accumulation)
closes the clustered gap. Evidence: `ISSUE-009` acceptance/verification sections,
`docs/issues/research/ISSUE-009-exp-a-results.md`, and
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11ay/`.
