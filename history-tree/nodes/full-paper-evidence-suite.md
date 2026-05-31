# Full-Paper Evidence Suite

- **Date**: 2026-05-29
- **Type**: analysis
- **Outcome**: success
- **Tags**: statistics, mujoco, claim-boundary

## Timeline and Background

Once the LCP-style baseline became the main full-paper comparison row, the
project consolidated evidence needed to defend the mechanism-comparison claim
against reviewer objections.

## Technical Details

- Mechanism draft: `docs/paper/full-paper-mechanism-comparison-draft.md`.
- Matched MuJoCo anchors: `docs/full-paper/matched-mujoco-anchor-results.md`.
- Statistical robustness:
  `docs/full-paper/statistical-robustness-results.md`.
- Selected-vs-final checkpoint robustness:
  `docs/full-paper/selected-vs-final-checkpoint-robustness.md`.
- Mixed-evidence mechanism note:
  `docs/full-paper/mujoco-mixed-evidence-mechanism.md`.
- Related-work and claim-boundary map:
  `docs/full-paper/related-work-claim-boundary-map.md`.

## Decision Process

The paper needed a defensible claim that did not overstate SC-PPO, LCP, or the
heuristic. The evidence suite therefore focused on metric-specific comparisons,
uncertainty, checkpoint dependence, and explicit claim boundaries.

## Results and Impact

The full-paper thesis became stable: policy-local-sensitivity regularization is
useful for policy-output smoothness, soft LCP-style enforcement is strongest in
the same-task audit, and the revised heuristic remains a serious downstream
dynamics anchor.
