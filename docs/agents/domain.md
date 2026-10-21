# Domain Docs

Where this repo keeps its domain glossary and architecture decision records, and how the
engineering skills should consume them. This file overrides the skill defaults (a repo-root
`GLOSSARY.md` and `docs/adr/`).

## Read these

- **`docs/GLOSSARY.md`** — the domain glossary. Use its vocabulary whenever your output
  names a domain concept: issue titles, refactor proposals, hypotheses, and test names.
  Don't drift to synonyms the glossary explicitly avoids.
- **`docs/adr/`** — ADRs that touch the area you are about to work in.

If a file doesn't exist, **proceed silently**: don't flag its absence and don't suggest
creating it upfront. The `domain-modeling` skill creates `docs/GLOSSARY.md` and
`docs/adr/` lazily, when a term or decision actually gets resolved.

This is a **single-context** repo: one glossary and one ADR directory. There is no
`GLOSSARY-MAP.md`.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently
overriding it:

> _Contradicts ADR-0004 (do X), but worth reopening because…_
