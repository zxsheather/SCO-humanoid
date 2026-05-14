# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repo root
- `CONTEXT-MAP.md` at the repo root if it exists
- `docs/adr/` if it exists

If any of these files don't exist, proceed silently.

## Layout

This repo is single-context:
- root `CONTEXT.md`
- optional root `docs/adr/`

## Vocabulary rule

Use the glossary terms from `CONTEXT.md` in issues, plans, hypotheses, and test names.

## ADR rule

If a proposal contradicts an existing ADR, surface the conflict explicitly.
