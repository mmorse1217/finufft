# Issue Tracker

This directory is the repository's issue tracker. It is intentionally simple: plain
Markdown files move through three state directories as work proceeds.

| Directory | Meaning |
|---|---|
| `docs/issues/todo/` | Outstanding issues that nobody is working on yet. |
| `docs/issues/in-progress/` | Issues currently being worked on. |
| `docs/issues/done/` | Completed issues. |
| `docs/issues/assets/` | Wayfinder design sketches and other linked assets. Not an issue stage. |
| `docs/issues/research/` | Wayfinder research notes. Not an issue stage. |

See [`../ROADMAP.md`](../ROADMAP.md) for the ordered plan that decides which issues are
created and worked next. The roadmap is populated from the decisions recorded by the
`wayfinder` skill; this directory holds the tracker mechanics. Wayfinder maps and tickets
are ordinary issue files in the stage directories; their design sketches live in `assets/`
and their research notes in `research/`, both outside the issue lifecycle.

## Issue file schema

Every issue is a single Markdown file named `ISSUE-NNN-<short-title>.md`, where `NNN` is the
next free number across all three stage directories, zero-padded to at least three digits.
Use this structure exactly:

```markdown
# ISSUE-NNN: <Short imperative title>

**Created:** YYYY-MM-DD
**Priority:** P0 | P1 | P2 | P3
**Status:** Todo | In progress | Done
**Started:** YYYY-MM-DD | _not started_
**Completed:** YYYY-MM-DD | _not completed_

**Related commits:**
- _None yet_

**Dependencies:**
- _None_

## Description

The problem, context, and intended outcome.

## Design

_Not yet written. Settled by the design interview before implementation starts._

## Implementation Plan

_Not yet written. Ordered step-by-step plan from the implementation-plan interview._

## Subtasks

- [ ] Concrete implementation step
- [ ] Any further steps required for completion

## Acceptance criteria

- [ ] A user-visible / behavioral requirement that must hold when the issue is done

## Verification criteria

- [ ] A concrete check with evidence, e.g. the exact build/test commands run and their result

## Completion override

_Leave blank unless closing with unmet criteria._

**Approved by:**
**Rationale:**
**Unmet criteria:**

## Notes

- YYYY-MM-DD — Issue created.
```

Field semantics:

- **Status** is the single source of truth for lifecycle: `Todo` → `In progress` → `Done`.
  It must match the directory the file lives in.
- **Priority** is `P0` (urgent) through `P3` (nice to have).
- **Design** is the synthesized design document. A settled design contains the approach;
  interfaces and data; failure modes and edge cases; decisions with rationale and rejected
  alternatives; and open questions (resolved, or explicitly deferred with a reason). It is
  written while the issue is still in `todo/`, either by the design interview
  (`/issue-design`, using `grill-me`) or, when the discussion already settled it, by the
  `to-spec` skill. The section has three states: **placeholder** (the template text
  `_Not yet written. Settled by the design interview before implementation starts._`),
  **draft** (the section begins with a line `_Draft — <reason>._`), and **settled**
  (anything else). `/issue-impl-plan` refuses a placeholder or draft design; a draft must
  be settled by `/issue-design`, or replaced by a settled `to-spec` design, first. Do not
  start the issue while an **Open** item remains.
- **Implementation Plan** is the ordered, step-by-step plan produced by a second interview
  after the design is settled. Each step must be separately implementable and verifiable;
  `## Subtasks` mirrors the steps as checkboxes.
- **Subtasks** are the checklist of work required for completion. **Acceptance criteria**
  are the requirements that define done. **Verification criteria** are the concrete checks
  and evidence (commands run, tests, observations) that prove acceptance — record the
  actual command, not just "tests pass".
- **Related commits** starts empty and accumulates short commit hashes as work lands.
- **Dependencies** lists issues (by `ISSUE-NNN` id) that must be `Done` before this one can
  start. `/issue-start` refuses to start an issue with unmet blockers; record a waiver and
  its rationale in Notes if one is genuinely needed.
- **Completion override** stays blank for normal closures. It is filled only when an issue
  is closed with unmet criteria, recording who approved, why, and which criteria were not
  met.
- **Notes** is an append-only log. Each entry is dated (`YYYY-MM-DD HH:MM`), summarizes
  what happened since the previous entry (implemented work, problems and fixes, decisions
  and rationale), and never rewrites earlier entries.
- **Labels** is optional and is used only by `wayfinder` maps and tickets, directly under
  `**Status:**`. Ordinary issues do not carry labels.

## Workflow

Work on one issue per session. The `issue-tracker` commands write and move the files at
every transition.

An issue normally moves through five stages:

1. **File it.** `/issue-create [P0-P3] <topic>` runs the `grill-me` interview about the
   topic and writes the issue to `docs/issues/todo/`. The interview settles the problem
   and the intended outcome; `## Design` and `## Implementation Plan` are left as
   placeholders. To spec a whole sprint at once, `/issue-sprint-spec [sprint]` runs the
   same interview for each work item in `docs/ROADMAP.md` in sequence, writing one issue
   file per item and updating the roadmap as it goes.
   When the discussion already happened, the `to-spec` skill files the issue instead of
   `/issue-create`: the spec goes under `## Description` and a synthesized design under
   `## Design`. The later stages are unchanged.
2. **Settle the design** (issue still in `todo/`). Run `/issue-design <todo-file>`: a
   `grill-me` interview whose output is a synthesized design document, not a transcript of
   the questions and answers and not a bare list of decisions. The section contains the
   approach; interfaces and data; failure modes and edge cases; decisions with rationale
   and rejected alternatives; and open questions. Resolve every open question or mark it
   explicitly deferred with a reason. When the interview ends early with unresolved items
   the section is written as a **draft**, begun with `_Draft — <reason>._`; otherwise it
   is **settled**.
3. **Write the implementation plan** (issue still in `todo/`). Run
   `/issue-impl-plan <todo-file>`: a second `grill-me` interview that turns the settled
   design into an ordered, step-by-step plan in `## Implementation Plan`. Each step must be
   implementable and verifiable on its own. The command mirrors the steps into
   `## Subtasks`, refines the acceptance and verification criteria if the plan surfaces
   gaps, then commits. It refuses until `## Design` is settled.
4. **Start it.** `/issue-start <md-file>` moves the issue from `todo/` to `in-progress/`,
   sets `Status: In progress` and `Started`, appends a dated note, and commits. Do not
   start before the design and plan are settled.
5. **Implement one step at a time.** For each plan step, run the implementation command
   scoped to that step alone, review and verify the change, check the step off in
   `## Subtasks`, and run `/issue-update` to append the dated note. Do not begin the next
   step until the current one is verified. Close with `/issue-done <md-file>` when every
   checklist item is checked.

### Command reference

| Command | Effect |
|---|---|
| `/issue-create [P0-P3] <topic>` | Runs the `grill-me` interview about the topic, then writes the resulting issue to `docs/issues/todo/` (with `## Design` and `## Implementation Plan` as placeholders) and commits it. |
| `/issue-sprint-spec [sprint]` | Specs a whole sprint in one run: reads `docs/ROADMAP.md`, then conducts a `grill-me` interview per work item in sequence and writes each settled spec into its issue file. Subdivides an item when its estimated delta exceeds ~350 lines, and updates `docs/ROADMAP.md` as it goes. No interview document is kept. |
| `/issue-design [md-file]` | Runs the `grill-me` design interview for a `todo/` issue and writes the synthesized design into `## Design`: settled when the interview completes, otherwise begun with `_Draft — <reason>._`. Then commits. Refuses when the issue has no `## Design` section. |
| `/issue-impl-plan [md-file]` | Runs the `grill-me` implementation-plan interview for a `todo/` issue: writes the ordered plan into `## Implementation Plan`, mirrors the steps into `## Subtasks`, refines the criteria, commits. Refuses until `## Design` is settled. |
| `/issue-start <md-file>` | Moves the issue from `todo/` to `in-progress/`, sets `Status: In progress` and `Started`, adds a dated Notes entry, commits. Refuses to start while any **Dependencies** issue is not yet `Done`. |
| `/issue-update [md-file]` | Reviews the session's work since the last Notes entry and updates the in-progress issue: checks off completed Subtasks/Acceptance/Verification items, adds new commit ids, appends a dated Notes summary, commits. |
| `/issue-done [md-file]` | Final update, then close: requires every Subtask/Acceptance/Verification item checked (or an explicitly approved Completion override), sets `Status: Done` and `Completed`, appends a verification log note, moves the file to `docs/issues/done/`, commits. |

Without an explicit `<md-file>`, the commands target the issue the current session started,
falling back to the single issue in the relevant directory, then an interactive picker.

### Wayfinder maps

The `wayfinder` skill uses this tracker for planning work that is too large for one
session. A map and its tickets are ordinary issue files:

- The **map** is a `todo/` issue carrying `**Labels:** wayfinder:map`. Its body has
  `## Destination`, `## Notes`, `## Decisions so far`, `## Not yet specified`, and
  `## Out of scope` sections.
- A **ticket** is a child issue carrying `**Labels:** wayfinder:research`,
  `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task`. Its question is the
  `## Description`.
- **Dependencies** list blocking tickets by `ISSUE-NNN`. A ticket is unblocked when every
  dependency is `Done`.
- A ticket is **claimed** by moving it to `in-progress/` and appending a dated note. It is
  **resolved** by appending a `## Answer` section, moving it to `done/`, and adding a
  one-line gist plus a link to the map's `## Decisions so far`.

Wayfinder maps and tickets are planning artifacts, not sprint work items. Do not feed them
to `/issue-sprint-spec`; use their settled decisions to populate `docs/ROADMAP.md` and then
file the implementation issues.

## Conventions

- Dates are `YYYY-MM-DD`; Notes timestamps are `YYYY-MM-DD HH:MM` (local time).
- Only record what actually happened. Do not check off items that are not done, and do not
  invent work, commits, or verification evidence in Notes.
- Do not rewrite history in an issue file: earlier Notes entries, commit ids, and checklist
  state stay intact; new information is appended.
- `## Design` and `## Implementation Plan` are living sections: refine them as decisions
  land, unlike the append-only Notes log. Record decisions in the issue file, not only in
  chat.
- Implement one plan step per run; a step is complete only when its subtask is checked with
  verification evidence.
- Commit messages for tracker changes use `docs(issues): <verb> ISSUE-NNN <title>` where
  the verb is one of `add`, `design`, `plan`, `start`, `update`, `close`.
