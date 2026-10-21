# Issue tracker: repo-native Markdown

This repo tracks issues as plain Markdown files under `docs/issues/`. The canonical schema,
field semantics, and workflow live in [`docs/issues/README.md`](../issues/README.md) — read
that before creating or moving any file. This doc only tells the engineering skills how to
translate their tracker operations onto this repo's conventions.

The `/issue-*` pi commands (`/issue-create`, `/issue-design`, `/issue-impl-plan`,
`/issue-start`, `/issue-update`, `/issue-done`, `/issue-sprint-spec`) implement the full
interview pipeline against these files; prefer them when the user wants that pipeline.

## Conventions

- One issue per file: `docs/issues/<stage>/ISSUE-NNN-<slug>.md`, where `NNN` is the next
  free number across all three stage directories, zero-padded to at least three digits.
- Stages are directories, mirrored by the `**Status:**` field: `todo/` ↔ `Todo`,
  `in-progress/` ↔ `In progress`, `done/` ↔ `Done`. The two must agree.
- `**Dependencies:**` lists blocking issues by `ISSUE-NNN`. An issue is unblocked when
  every listed issue is `Done` (its file is in `done/`).
- There are no tracker labels, assignees, or native blocking links. Label-like values are
  recorded as text (see Translations).
- `## Notes` is required and append-only: each entry is dated and never rewrites earlier
  ones.
- The `/issue-*` commands commit each state change when the tracker paths are tracked. If
  the paths are intentionally untracked or ignored, the commit step reports a failure and
  the file change remains in place.

## Translations

Where a skill's inline template conflicts with the schema or conventions in
[`docs/issues/README.md`](../issues/README.md), the README wins.

| Skill operation | Do this here |
| --- | --- |
| publish a spec | Create `docs/issues/todo/ISSUE-NNN-<slug>.md` from the README schema. Put Problem Statement / Solution / User Stories under `## Description`. Synthesize Implementation Decisions / Testing Decisions into the README's `## Design` shape — approach; interfaces and data; failure modes and edge cases; decisions with rationale and rejected alternatives; open questions — as written prose, never pasted as raw bullets. Start the section with `_Draft — <reason>._` when open questions remain; write it with no marker when the discussion settled the design. Leave `## Implementation Plan` and `## Subtasks` as placeholders for `/issue-impl-plan`. Commit the file alone. |
| publish a ticket | When the source is an issue with an `## Implementation Plan`, one ticket per plan step or per step-group that must land together, created in the plan's dependency order (blockers first) and numbered in that order; do not re-slice the work. Otherwise slice the work as the skill describes. Each ticket is a full README-schema file in `todo/` whose `**Dependencies:**` lists its blockers by `ISSUE-NNN`. The child's `## Design` is `_Inherited from ISSUE-NNN (parent)._`, and its `## Description` carries its slice's steps and points at the parent's plan. Append one dated `## Notes` entry to the parent naming the tickets; never rewrite the parent's other sections, and do not close the parent until every child is `Done`. Commit each file alone. |
| "ready-for-agent" | `**Status:** Todo` in `docs/issues/todo/`. There is no triage label vocabulary; skip labels. |
| claim / assign | Move the file to `in-progress/`, set `**Status:** In progress` and `**Started:**` to today, and append a dated `## Notes` entry recording the claim. Prefer `/issue-start` when the issue has a settled `## Design`. |
| close work | Move the file to `done/`, set `**Status:** Done` and `**Completed:**` to today, check off the criteria, add commit hashes to `**Related commits:**`, and append a dated `## Notes` entry. Prefer `/issue-done`. |
| comment or resolution | Append a dated `## Notes` entry. Wayfinder answers also get a `## Answer` section. |
| a label `X` (e.g. `wayfinder:research`) | Record it on a `**Labels:** X` line directly under `**Status:**`; labels are plain text here. |
| fetch a ticket | Find it by `ISSUE-NNN` under any of the three stage directories and read the whole file. |
| find the originating spec | Look for `ISSUE-NNN` references in commit messages and `## Notes`; open the matching file under `docs/issues/{done,in-progress,todo}/`. Commit messages never use GitHub `#NNN` syntax. |
| native blocking links | Unavailable. Use `**Dependencies:**` and resolve it as described above. |

Do not invent header fields beyond the optional `**Labels:**` line. The schema in
`docs/issues/README.md` is authoritative.

## Wayfinding operations

Used by the `wayfinder` skill. The map and every ticket are ordinary issue files.

- **Map**: one issue file for the effort, titled `<destination> (wayfinder map)`, in
  `todo/`. Its body carries Destination, Notes, Decisions so far, Not yet specified, and
  Out of scope as `##` sections. Label it `wayfinder:map`.
- **Child ticket**: one issue file per ticket; the question is its `## Description`. Record
  `**Labels:** wayfinder:<type>` (`research` / `prototype` / `grilling` / `task`).
- **Blocking**: `**Dependencies:**` lists the blocking `ISSUE-NNN`s. A ticket is unblocked
  when every dependency is `Done`.
- **Frontier**: open tickets in `todo/` whose dependencies are all `Done` and which carry
  no claim note; lowest `ISSUE-NNN` first.
- **Claim**: move the file to `in-progress/`, set `**Status:** In progress`, and append a
  dated `## Notes` entry. Save it before any work.
- **Resolve**: append a `## Answer` section, set `**Status:** Done`, move the file to
  `done/`, then add a one-line gist plus a link for this ticket to the map's Decisions so
  far.
