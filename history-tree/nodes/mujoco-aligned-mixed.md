# MuJoCo Aligned Mixed Reading

- **Date**: 2026-05-21
- **Type**: experiment
- **Outcome**: mixed
- **Tags**: mujoco, external-validation, mixed

## Timeline and Background

The repo had earlier used a looser MuJoCo interpretation. Once the heuristic baseline was replaced by the revised long-budget anchor, that older wording had to be retested rather than carried forward.

## Technical Details

- Canonical note: [docs/sc-ppo-mujoco-revised-anchor-aligned-comparison.md](../../docs/sc-ppo-mujoco-revised-anchor-aligned-comparison.md)
- Comparable replay:
  - terrain mode `isaac_mainline`
  - heuristic selected checkpoints `350 / 300 / 350`
  - SC-PPO selected checkpoints `300 / 300 / 400`
- New aligned read:
  - revised heuristic is better on task stability, velocity tracking, episode length, and joint acceleration
  - `SC-PPO 3.8` is only better on action jitter

## Decision Process

- The repo retired the older "partial transfer advantage" wording.
- It kept MuJoCo as external validation, but narrowed the safe claim to **mixed evidence**.
- This decision is also reflected in [docs/sc-ppo-current-summary.md](../../docs/sc-ppo-current-summary.md) and [docs/sc-ppo-next-step-direction.md](../../docs/sc-ppo-next-step-direction.md).

## Results and Impact

- This node prevented overclaiming.
- It locked the repo into a defensible stance: Isaac-side win, but no cross-engine SC-PPO win.
- That reinterpretation is one reason the later [research-delivery-freeze](../nodes/research-delivery-freeze.md) was possible.
