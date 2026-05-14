# Heuristic Action-Rate Sweep

Issue `#3` strengthens the `启发式平滑基线` before any `SC-PPO` claim is made.

The bounded sweep keeps the upstream smoothness bundle fixed except for the primary `动作时间变化率`
penalty weight:

- `action_rate_0005` -> `rewards.scales.action_smoothness = -0.0005`
- `action_rate_0020` -> `rewards.scales.action_smoothness = -0.0020`
- `action_rate_0050` -> `rewards.scales.action_smoothness = -0.0050`

All candidates keep the current upstream values for:

- `rewards.scales.dof_acc = -1e-7`
- `rewards.scales.base_acc = 0.2`
- `rewards.scales.dof_vel = -5e-4`

Sweep config:

- `configs/sweeps/heuristic_action_rate_rough_terrain.json`

Candidate configs:

- `configs/methods/heuristic_smoothing_action_rate_0005.json`
- `configs/methods/heuristic_smoothing.json`
- `configs/methods/heuristic_smoothing_action_rate_0050.json`

## Selection rule

The sweep follows the repo glossary:

- `先过底线再取最平滑`
- `速度误差10%退化上限`
- `跌倒率5个百分点退化上限`

Operationally:

1. Pick the heuristic candidate with the best `velocity_tracking_error_mean` as the in-family task reference.
2. Keep only candidates whose:
   `velocity_tracking_error_mean <= reference * 1.10`
3. Also require:
   `fall_rate <= reference_fall_rate + 0.05`
4. Among the surviving candidates, choose the lowest `joint_acceleration_l2_mean`.
5. Break ties with `action_jitter_l2_mean`, then `velocity_tracking_error_mean`.

## Commands

Inspect candidate status and resolved commands:

```bash
python scripts/baseline/run_method_sweep.py --stage plan --skip-completed
```

Run the full long-budget bounded sweep while reusing completed candidates:

```bash
python scripts/baseline/run_method_sweep.py \
  --stage all \
  --skip-completed \
  --train-num-envs 512 \
  --max-iterations 200 \
  --eval-num-envs 32 \
  --episodes 20
```

Select the heuristic baseline after the sweep finishes:

```bash
python scripts/baseline/select_heuristic_sweep.py
```

The selection artifact is written to:

- `artifacts/analysis/heuristic_action_rate_rough_terrain/selection.json`

## Reuse note

`action_rate_0020` reuses the current long-run artifact from:

- `configs/methods/heuristic_smoothing.json`

This means `--skip-completed` only needs to train and evaluate the new `-0.0005` and `-0.0050` candidates.
