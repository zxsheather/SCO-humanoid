# Full-Paper LaTeX Build

This directory now contains a venue-neutral, buildable LaTeX source for the
full-paper mechanism-comparison draft.

## Source Files

- `full-paper.tex`: root manuscript source.
- `references.bib`: bibliography used by the manuscript.
- `full-paper-red-team-notes.md`: reviewer-risk pass over the assembled source.
- `Makefile`: source-level convenience wrapper around `latexmk`.

Generated PDFs and auxiliary LaTeX files are intentionally ignored by git. Do
not commit compiled submission packages, PDFs, or archive bundles.

## Canonical Local Build

The current project convention is to build submission artifacts under the
ignored local workspace, not in `docs/paper`:

```bash
make -C .local/paper-submission/full-paper clean all
```

When the local package is present, this produces both:

- `.local/paper-submission/full-paper/build/full-paper.pdf`
- `.local/paper-submission/full-paper/build/full-paper_wcbm.pdf`

The `.local/paper-submission/full-paper` package is intentionally not tracked by
git. It may contain venue/template files such as `wcbm.sty`,
`wcbmabbrvnat.bst`, copied figures, and local-only WCBM source generated from
`docs/paper/full-paper.tex`.

## Source-Only Smoke Build

Use the tracked Makefile only when deliberately checking the venue-neutral
source in place:

```bash
make -C docs/paper
make -C docs/paper clean
```

Do not commit PDFs or LaTeX auxiliary files from either build path.

## Manuscript Boundary

The source follows the full-paper mechanism-comparison framing:

- Primary rows: LCP-style soft Jacobian/Lipschitz penalty, SC-PPO 3.8
  PID-Lagrangian, and the revised heuristic action-rate anchor.
- Policy perturbation audit: mechanism diagnostic for local policy-output
  amplification.
- OmniSafe PPO-Lag and local CPO-style work: diagnostic or future-work material
  only, not promoted baselines.

If a target venue is chosen later, replace the article class and author block
while preserving the claim boundaries above.
