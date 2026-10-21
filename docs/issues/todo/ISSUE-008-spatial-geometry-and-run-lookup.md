# ISSUE-008: Choose the source-bin, output-tile, and run-lookup geometry

**Created:** 2026-10-21
**Priority:** P1
**Status:** Todo
**Labels:** wayfinder:grilling
**Started:** _not started_
**Completed:** _not completed_

**Related commits:**
- _None yet_

**Dependencies:**
- ISSUE-003
- ISSUE-004
- ISSUE-005
- ISSUE-006
- ISSUE-007

## Description

Using the atomic inventory, the correctness contract, and the radix-sort findings, settle
the prototype's spatial parameters with the human:

- source-bin size per dimension, and whether bins are cubic;
- output-tile size, cells per thread, and block/workgroup geometry;
- Morton key width and bit interleaving, including how many bits per dimension are needed
  and how periodic wrapping maps to keys;
- occupied-bin run lookup: dense `bin -> run` table, hash table, or binary search; the
  memory budget on a 4 GB T1000;
- how an output tile enumerates the source bins intersecting its kernel halo, including
  periodic wrap;
- whether the first prototype uses one workgroup per output tile with no active-tile
  compaction, as the sketch recommends, or compacts from the start;
- whether to base the output-owned kernel on method 3's output-driven structure or on
  method 4's ghost-bin partition — method 4 already gives each output bin exactly one
  owner and duplicates ghost index entries, so it is the closest existing starting point;
- how the source-bin width relates to the output-tile width and the kernel support, and
  what average point-reread factor the choice implies.

Record the choices, rejected alternatives, and the resulting parameters in `## Answer`.
Graduate any load-balancing question that becomes sharp into a new ticket or the map's Not
yet specified section.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
