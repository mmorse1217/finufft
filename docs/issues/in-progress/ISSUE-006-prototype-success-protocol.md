# ISSUE-006: Settle the prototype success protocol and scope

**Created:** 2026-10-21
**Priority:** P1
**Status:** In progress
**Labels:** wayfinder:grilling
**Started:** 2026-10-21
**Completed:** _not completed_

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
