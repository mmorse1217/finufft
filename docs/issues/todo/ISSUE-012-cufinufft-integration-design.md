# ISSUE-012: Settle the cuFINUFFT integration design

**Created:** 2026-10-21
**Priority:** P2
**Status:** Todo
**Labels:** wayfinder:grilling
**Started:** _not started_
**Completed:** _not completed_

**Related commits:**
- _None yet_

**Dependencies:**
- ISSUE-011

## Description

If the go/no-go decision is go, settle how the new spread path enters cuFINUFFT. With the
human:

- **Method identity.** A new `gpu_method` id, or a replacement/extension of method 3? What
  is the fallback policy, and how do auto-method heuristics choose between them?
- **Lifecycle.** Where do Morton keys, the sorted permutation, and the occupied-bin run
  table live in `makeplan`/`setpts`/`execute`? How are they reused across `ntr` and repeated
  executions, and when are they invalidated?
- **Memory and allocation.** Budget and allocation strategy on 4 GB and larger GPUs,
  including temporary sort storage and the run table.
- **Option surface.** Reuse `gpu_binsize*`, `gpu_maxsubprobsize`, `gpu_np`, and
  `gpu_kerevalmeth`, or add options? Which changes are additive and ABI-safe?
- **Scope.** Confirm 3D-only first, with 1D/2D and type 3 left to follow-on efforts (reopen
  the map's destination if the evidence argues otherwise). FP32 then FP64? Is the type-2
  inverse part of the first integration or a follow-on? What is the order of implementation
  issues?
- **Correctness and tests.** Tolerances, comparison against methods 1/3/4, regression
  tests, and whether `gpu_spreadinterponly` is the primary oracle.
- **Benchmark and rollback.** What benchmark gate must be met, and under what condition is
  the new method disabled or removed?

Record the design in `## Answer`. Graduate the implementation work into new issues (or the
roadmap's issue-creation order) and mark anything past the destination out of scope. Do not
implement the method in this ticket.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
