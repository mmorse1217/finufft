# ISSUE-007: Settle the ownership and Morton vocabulary

**Created:** 2026-10-21
**Priority:** P2
**Status:** Done
**Labels:** wayfinder:grilling
**Started:** 2026-10-21
**Completed:** 2026-10-21

**Related commits:**
- _None yet_

**Dependencies:**
- _None_

## Description

The design sketch calls the ownership rule "destination-owned", but `docs/GLOSSARY.md`
reserves "destination" for wayfinder maps (under `target`). Settle the canonical vocabulary
the map, prototype, and eventual code should use:

- output-cell-owned versus destination-owned versus target-owned (or another precise term);
- `source bin`, `output tile`, `Morton code`, `occupied-bin run`, `active tile`, and
  `point reread factor` — define each in the glossary's term/definition/avoid format;
- whether the existing terms `spreading`, `fine grid`, `bin sort`, `kernel width`, and
  `upsampling factor` are used consistently with the sketch.

Update `docs/GLOSSARY.md` with the resolved terms, keeping it free of implementation
detail. Record the choice and rejected alternatives in `## Answer`.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
- 2026-10-21 21:20 — Claimed by the wayfinder session to settle the ownership and Morton vocabulary.
- 2026-10-21 21:35 — Resolved with the human; glossary updated; ticket closed.
- 2026-10-21 22:43 — Vocabulary revised in *Choose the bin size, tile size, and sorted-point lookup*: bin = point grouping, tile = output region, local subgrid = shared-memory buffer, CUDA threadblock; see `docs/GLOSSARY.md`.

## Answer

Settled with the human.

- Canonical ownership term: **output-cell ownership**; short form **output-owned**.
  **Destination-owned** is recorded as the design sketch's historical synonym, not the
  glossary term. **Target-owned** rejected because `target` already means type-2/type-3
  output points, and `destination` is reserved for wayfinder maps.
- Method name: **Morton-sorted output-owned spreading**.
- **`bin sort`** stays specific to the existing Cartesian source-bin histogram/permutation.
  **`Morton sort`** is a distinct reordering by Morton code; the Morton ordering is not a bin
  sort.
- **`tile`** is defined as a contiguous Cartesian block of fine-grid cells; an **output
  tile** is the accumulation unit in output-owned spreading.
- Added to `docs/GLOSSARY.md`: source bin, bin sort (updated), tile, output tile, Morton
  code, Morton sort, occupied-bin run, active tile, point reread factor, output-cell
  ownership, Morton-sorted output-owned spreading.
- Existing terms `spreading`, `fine grid`, `kernel width`, and `upsampling factor` are
  unchanged; `target` keeps its "destination is reserved for wayfinder maps" note.
