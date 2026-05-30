# Full-Paper LaTeX Build

This directory now contains a venue-neutral, buildable LaTeX source for the
full-paper mechanism-comparison draft.

## Source Files

- `full-paper.tex`: root manuscript source.
- `references.bib`: bibliography used by the manuscript.
- `Makefile`: convenience wrapper around `latexmk`.

Generated PDFs and auxiliary LaTeX files are intentionally ignored by git. Do
not commit compiled submission packages, PDFs, or archive bundles.

## Build

```bash
cd docs/paper
make
```

Equivalent direct command:

```bash
cd docs/paper
latexmk -pdf -interaction=nonstopmode -halt-on-error full-paper.tex
```

## Clean

```bash
cd docs/paper
make clean
```

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
