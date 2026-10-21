# ISSUE-007: Settle the ownership and Morton vocabulary

**Created:** 2026-10-21
**Priority:** P2
**Status:** Todo
**Labels:** wayfinder:grilling
**Started:** _not started_
**Completed:** _not completed_

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
