# Issue #1: Vanilla PPO Rough-Terrain Baseline

This document defines the reproducible path for issue `#1`: train and evaluate a `Vanilla PPO` `速度跟踪行走` policy on `粗糙平面`, save a checkpoint, export a reloadable policy, and write a reusable evaluation artifact.

## Upstream contract

- Upstream project: `https://github.com/roboterax/humanoid-gym`
- Pinned ref: `ae46e201c85a2b17e7f2cea59a441dae7ea88a8f`
- Default checkout path in this repo: `.external/humanoid-gym`
- Upstream task name: `humanoid_ppo`

## Files added for issue #1

- `configs/baselines/vanilla_ppo.json`
- `scripts/baseline/bootstrap_humanoid_gym.sh`
- `scripts/baseline/check_env.py`
- `scripts/baseline/train_vanilla_ppo.py`
- `scripts/baseline/export_policy.py`
- `scripts/baseline/evaluate_policy.py`

## Expected prerequisites

- Python `3.8`
- PyTorch `1.13.x`
- Isaac Gym Preview 4 installed in the active environment
- A working `Humanoid-Gym` checkout at the pinned ref

## Setup

Bootstrap the upstream checkout and install it editable:

```bash
./scripts/baseline/bootstrap_humanoid_gym.sh
```

Validate the environment:

```bash
python scripts/baseline/check_env.py
```

## Documented training command

Dry-run the exact training command:

```bash
python scripts/baseline/train_vanilla_ppo.py --dry-run
```

Run the baseline training:

```bash
python scripts/baseline/train_vanilla_ppo.py
```

This wraps the upstream command equivalent to:

```bash
python .external/humanoid-gym/humanoid/scripts/train.py \
  --task=humanoid_ppo \
  --experiment_name=ecolab_humanoid_ppo \
  --run_name=vanilla_ppo_rough_terrain \
  --headless \
  --num_envs=4096 \
  --rl_device=cuda:0 \
  --sim_device=cuda:0
```

## Export and evaluation

Export a reloadable JIT policy from the latest checkpoint:

```bash
python scripts/baseline/export_policy.py
```

Evaluate the latest checkpoint on `粗糙平面` and write the reusable artifact:

```bash
python scripts/baseline/evaluate_policy.py
```

## Output layout

The scripts populate:

- `artifacts/baselines/vanilla_ppo/vanilla_ppo_rough_terrain/manifest.json`
- `artifacts/baselines/vanilla_ppo/vanilla_ppo_rough_terrain/exported/policies/policy_1.pt`
- `artifacts/baselines/vanilla_ppo/vanilla_ppo_rough_terrain/metrics.json`

`metrics.json` contains the minimum issue `#1` comparison fields:

- `velocity_tracking_error_mean` for `速度跟踪误差主指标`
- `fall_rate` for `跌倒率底线指标`
- `episode_return_mean` for `总回报补充指标`
