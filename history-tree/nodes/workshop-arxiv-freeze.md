# Workshop/arXiv Manuscript Freeze

- **Date**: 2026-05-26
- **Type**: milestone
- **Outcome**: success
- **Tags**: paper, workshop, freeze

## Timeline and Background

After the SC-PPO-centered post-freeze diagnostics, the project assembled a
workshop/arXiv manuscript package around the hard-constraint mechanism line.
This was the last stable paper package before the full-paper branch changed the
claim from "SC-PPO-centered" to "mechanism comparison."

## Technical Details

- Freeze commit: [`08fcb99`](https://github.com/zxsheather/SCO-humanoid/commit/08fcb99)
  `Tighten manuscript citations and bibliography`.
- Frozen branch/tag context:
  `freeze/workshop-arxiv-2026-05-27` and
  `workshop-arxiv-freeze-2026-05-27`.
- Main source at the time: `docs/paper/arxiv-workshop-manuscript.md`.

## Decision Process

The project preserved this package as a stable archival paper baseline rather
than continuing to mutate the workshop submission while the evidence base was
changing.

## Results and Impact

The freeze made it safe to start `full-paper/extended-seeds` without rewriting
the older workshop/arXiv story. Later full-paper work supersedes the
SC-PPO-centered narrative but does not invalidate the frozen historical package.
