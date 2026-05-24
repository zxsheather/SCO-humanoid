# Revised Heuristic Formal Anchor

- **Date**: 2026-05-21
- **Type**: milestone
- **Outcome**: success
- **Tags**: protocol-revision, baseline, 3-seed

## Timeline and Background

The repaired-budget probe proved that a mere patch was not enough. The repo then asked a stricter question: under the same 512-env regime, does extending the run produce a genuine 3-seed baseline anchor with no `checkpoint 0` dependence?

## Technical Details

- Canonical notes:
  - [docs/baselines/rough-terrain-formal-protocol-revision-decision.md](../../docs/baselines/rough-terrain-formal-protocol-revision-decision.md)
  - [docs/baselines/rough-terrain-formal-protocol-revision-long-budget.md](../../docs/baselines/rough-terrain-formal-protocol-revision-long-budget.md)
- Revised regime:
  - `512 envs x 400 iterations`
  - same `action_rate = -0.0050`
- Success outcome:
  - selected checkpoints `350 / 300 / 350`
  - all three seeds survived
  - no row depended on `checkpoint 0`

## Decision Process

- The repo adopted a minimum revision rule:
  - all three seeds must be non-collapsed
  - the row must be task-valid
  - the row must not survive only through `checkpoint 0`
- This replaced the frozen formal compare as the baseline-side anchor and re-closed issue `#5`.

## Results and Impact

- This node restored the repo's ability to defend an Isaac-side three-way comparison.
- It redefined the main rough-terrain closure as:
  - `Vanilla PPO` raw reference
  - revised heuristic anchor
  - `SC-PPO 3.8`
- It also forced the aligned reread of [mujoco-aligned-mixed](../nodes/mujoco-aligned-mixed.md).
