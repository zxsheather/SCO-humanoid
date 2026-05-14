# Method Split For Rough-Terrain Locomotion

This repo now distinguishes three method-layer configs for the `速度跟踪行走` study on `粗糙平面`:

- `configs/methods/vanilla_ppo.json`
- `configs/methods/heuristic_smoothing.json`
- `configs/methods/sc_ppo.json`

## Current split

`Vanilla PPO` is the raw reference for the study's `三组正式对比`.
It disables the upstream smoothness-oriented reward terms:

- `rewards.scales.action_smoothness`
- `rewards.scales.dof_acc`
- `rewards.scales.base_acc`
- `rewards.scales.dof_vel`

`Heuristic smoothing baseline` keeps the upstream smoothness-oriented reward shaping active.

`SC-PPO` currently shares the same smoothness reward disablement as `Vanilla PPO`, because the hard constraint
mechanism has not yet been implemented. This keeps the method config aligned with the intended
`完全替换对比` setup.

The heuristic side now has a separate bounded sweep document at
`docs/baselines/heuristic-action-rate-sweep.md`, so the final heuristic baseline can be selected through
`启发式小范围调参基线` rather than a single fixed weight.

## Shared entrypoints

Training:

```bash
python scripts/baseline/train_vanilla_ppo.py --config configs/methods/vanilla_ppo.json
python scripts/baseline/train_vanilla_ppo.py --config configs/methods/heuristic_smoothing.json
```

Export:

```bash
python scripts/baseline/export_policy.py --config configs/methods/vanilla_ppo.json
python scripts/baseline/export_policy.py --config configs/methods/heuristic_smoothing.json
```

Evaluation:

```bash
python scripts/baseline/evaluate_policy.py --config configs/methods/vanilla_ppo.json
python scripts/baseline/evaluate_policy.py --config configs/methods/heuristic_smoothing.json
```
