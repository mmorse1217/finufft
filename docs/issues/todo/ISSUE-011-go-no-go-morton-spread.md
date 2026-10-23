# ISSUE-011: Decide go/no-go on the Morton output-owned spread

**Created:** 2026-10-21
**Priority:** P1
**Status:** Todo
**Labels:** wayfinder:grilling
**Started:** _not started_
**Completed:** _not completed_

**Related commits:**
- _None yet_

**Dependencies:**
- ISSUE-006
- ISSUE-009
- ISSUE-010

## Description

Review the Experiment A and Experiment B evidence and decide whether to integrate the new
spread path into cuFINUFFT. With the human:

- compare the measured spread-stage and total times against the success protocol in
  [Settle the prototype success protocol and scope](../done/ISSUE-006-prototype-success-protocol.md);
- weigh the preprocessing cost, point reread factor, memory footprint, occupancy, and
  correctness against the atomic-elimination benefit;
- identify any workload that is a pathological regression and decide whether it is a
  blocker, a documented fallback case, or acceptable;
- if go, state the evidence and the conditions/risks that integration must handle;
- if no-go, state why and what result would reopen the decision.

Record the decision in `## Answer`. If no-go, close
[Settle the cuFINUFFT integration design](ISSUE-012-cufinufft-integration-design.md) as out
of scope and add it to the map's Out of scope section. If go, leave the integration ticket
unblocked for the next session.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
