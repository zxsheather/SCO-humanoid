# SCO-humanoid

SCO-humanoid is a research-oriented repository for validating constrained reinforcement learning
methods for smoother humanoid locomotion.

`SCO` stands for `Smooth-Constrained-Optimization`.

The current codebase uses `Humanoid-Gym` as the training backbone, evaluates methods under a shared
metric schema, and compares:

- `Vanilla PPO`
- `PPO + heuristic smoothing` via action-rate reward shaping
- `SC-PPO` with a Jacobian-style smoothness constraint

The project is organized as an experiment repo, not as a general-purpose framework. The main goal
is reproducible evidence, comparative baselines, and report-grade analysis.

## What this repo is for

Use this repo if you want to:

- reproduce the current rough-terrain smooth-control experiments
- inspect the evidence chain behind the current `SC-PPO` line
- rerun baseline sweeps, checkpoint selection, and MuJoCo replay under the same metric schema

This repo is not designed as:

- a reusable locomotion RL framework
- a pip-installable package
- a generalized benchmark suite for many robots or tasks

## Research question

The core question is:

`Can a constrained smooth-control PPO variant produce smoother humanoid locomotion than heuristic reward shaping while keeping velocity-tracking performance usable?`

The current main task is rough-terrain velocity-tracking locomotion in Isaac Gym, with MuJoCo
sim-to-sim replay used as external validation.

## Current status

The strongest completed method line in the repo is:

- `SC-PPO threshold = 3.8`
- `PID-Lagrangian`
- `pid_integral_mode = lower_bound_clamp`
- `cost_aggregation = quantile(0.90)`

Current selected-checkpoint aggregate over seeds `11`, `17`, and `23`:

- `velocity_tracking_error_mean = 0.6412 +- 0.0554`
- `joint_acceleration_l2_mean = 115.9079 +- 6.9386`
- `action_jitter_l2_mean = 0.2205 +- 0.0017`
- `episode_return_mean = 100.2838 +- 2.7150`
- `fall_rate = 0.1000 +- 0.0000`

Important boundary:

- the method line above is currently the strongest completed line in the repo
- the frozen baseline refresh showed that `Vanilla PPO` and the bounded heuristic action-rate
  family (`-0.0005`, `-0.0020`, `-0.0050`) all collapse to `checkpoint 0` under the original
  `64 envs x 400 iterations` formal-compare regime
- the baseline-side protocol repair line then produced a revised long-budget heuristic anchor:
  `action_rate = -0.0050`, `512 envs x 400 iterations`, selected checkpoints `350 / 300 / 350`
- against that revised heuristic anchor, `SC-PPO 3.8` remains better on the shared Isaac
  rough-terrain velocity-tracking, fall-rate, joint-acceleration, and action-jitter metrics
- the aligned `MuJoCo isaac_mainline` replay is mixed external-validation evidence rather than an
  `SC-PPO` cross-engine win: the revised heuristic is better on task-side metrics, while `SC-PPO
  3.8` is only slightly better on action jitter
- the `PID有限消融` follow-up is now closed as a limited mechanism diagnostic: a matched
  `普通对偶上升` probe at `threshold = 3.8` collapses (`fall_rate = 1.0`), so it supports keeping
  `PID-Lagrangian` as the formal `SC-PPO` line without expanding into full component attribution

So the repo currently supports an Isaac-side `方法优于启发式` result, with `MuJoCo` reported as a
`混合外部验证结论`.

`main` should now be read as a `冻结主档案分支`:

- it preserves the completed internal research delivery package
- it may only absorb bounded backports for `冻结边界章节` updates and reusable evaluation or
  diagnostic infrastructure
- it should not reopen training, replay, or mechanism-specific branch work on `main`

For the most current interpretation, read:

- [docs/sc-ppo-report-status.md](docs/sc-ppo-report-status.md)
- [docs/sc-ppo-current-summary.md](docs/sc-ppo-current-summary.md)
- [docs/baselines/rough-terrain-formal-comparison.md](docs/baselines/rough-terrain-formal-comparison.md)
- [docs/sc-ppo-pid-limited-ablation.md](docs/sc-ppo-pid-limited-ablation.md)

## Fastest current handoff

If you only need the current frozen-mainline answer, read:

1. [docs/sc-ppo-current-summary.md](docs/sc-ppo-current-summary.md)
2. [docs/sc-ppo-report-status.md](docs/sc-ppo-report-status.md)
3. [docs/reproduction/final-research-delivery-checklist.md](docs/reproduction/final-research-delivery-checklist.md)

Minimal frozen-package validation:

```bash
cd /home/zhuoxiang/SCO-humanoid
export PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python
$PYTHON_BIN scripts/baseline/check_env.py
$PYTHON_BIN -m unittest \
  tests.test_baseline_common \
  tests.test_behavior_trace_metrics \
  tests.test_checkpoint_sweep_recovery \
  tests.test_baseline_protocol_failfast
git diff --check
```

Use [docs/reproduction/final-research-delivery-checklist.md](docs/reproduction/final-research-delivery-checklist.md)
for the broader frozen-package validation and canonical artifact index.

## Quick start

### 1. Bootstrap the local dependency layout

This repo expects a local `Humanoid-Gym` checkout under `.external/humanoid-gym`.

```bash
bash scripts/baseline/bootstrap_humanoid_gym.sh
```

### 2. Check the runtime environment

```bash
python scripts/baseline/check_env.py
```

### 3. Read the current frozen-mainline package

Start from:

- [docs/sc-ppo-current-summary.md](docs/sc-ppo-current-summary.md)
- [docs/sc-ppo-report-status.md](docs/sc-ppo-report-status.md)
- [docs/reproduction/final-research-delivery-checklist.md](docs/reproduction/final-research-delivery-checklist.md)

### 4. Historical experiment entrypoint

This is a historical reproduction entrypoint, not part of routine frozen-package validation on
`main`:

```bash
python scripts/baseline/train_vanilla_ppo.py \
  --config configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json
```

For shared-server or headless runs, prefer clearing `DISPLAY`:

```bash
env -u DISPLAY python scripts/baseline/train_vanilla_ppo.py \
  --config configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json
```

## Repository layout

```text
configs/     Experiment and sweep configs
docs/        Experiment notes, status docs, and report-facing summaries
scripts/     Training, evaluation, sweep, and analysis entrypoints
artifacts/   Generated runtime outputs and analysis summaries
tests/       Focused regression tests for experiment utilities
```

Main script entrypoints:

- `scripts/baseline/train_vanilla_ppo.py`
- `scripts/baseline/evaluate_policy.py`
- `scripts/baseline/evaluate_checkpoint_sweep.py`
- `scripts/baseline/evaluate_mujoco_sim2sim.py`
- `scripts/baseline/run_method_sweep.py`
- `scripts/baseline/select_heuristic_sweep.py`
- `scripts/baseline/run_formal_comparison.py`

## Requirements

This repo assumes:

- Linux
- CUDA GPU access
- an Isaac Gym compatible environment
- a local checkout of `Humanoid-Gym` under `.external/humanoid-gym`
- the MuJoCo evaluation stack used by the repo's sim-to-sim scripts

The training/evaluation scripts are written around the local environment used in this project and
are not packaged as a one-command install.

Useful bootstrap/helper script:

- `scripts/baseline/bootstrap_humanoid_gym.sh`

Typical local runtime assumptions:

- Python environment already provisioned with the repo's RL dependencies
- working NVIDIA driver / CUDA stack
- writable `artifacts/` directory
- writable `TORCH_EXTENSIONS_DIR` for Isaac Gym / PyTorch extension builds

## Shared metrics

All main comparisons use the repo's shared metric schema:

- `velocity_tracking_error_mean`
- `fall_rate`
- `joint_acceleration_l2_mean`
- `action_jitter_l2_mean`
- `episode_return_mean`
- optional `constraint_metrics`

See:

- [docs/evaluation/shared-smooth-control-metrics.md](docs/evaluation/shared-smooth-control-metrics.md)
- [artifacts/README.md](artifacts/README.md)

## Common workflows

### 1. Train one method

```bash
python scripts/baseline/train_vanilla_ppo.py \
  --config configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json
```

### 2. Evaluate one checkpoint

```bash
python scripts/baseline/evaluate_policy.py \
  --config configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json \
  --load-run <run_dir_or_run_name> \
  --checkpoint <checkpoint_id>
```

### 3. Run checkpoint sweep selection

```bash
python scripts/baseline/evaluate_checkpoint_sweep.py \
  --config configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json \
  --run-name <artifact_run_name> \
  --load-run <run_dir_or_run_name>
```

### 4. Run the bounded heuristic action-rate sweep

```bash
python scripts/baseline/run_method_sweep.py --stage all --skip-completed
python scripts/baseline/select_heuristic_sweep.py
```

### 5. Run the rough-terrain formal comparison wrapper

```bash
env -u DISPLAY CUDA_VISIBLE_DEVICES=1 \
  /home/zhuoxiang/miniconda3/envs/ecolab-isaacgym/bin/python -u \
  scripts/baseline/run_formal_comparison.py --stage all --skip-completed
```

The wrapper now:

- resolves real `run_dir` paths from manifests when needed
- tolerates the common Isaac/X11 cleanup case where artifacts are written before a non-zero exit
  code on process teardown

If a run writes `manifest.json` and then exits during Isaac cleanup, treat that as a teardown issue
first and inspect the recorded artifacts before assuming the experiment itself failed.

## Key documents

Project language and decision rules:

- [CONTEXT.md](CONTEXT.md)

Baseline and comparison docs:

- [docs/baselines/vanilla-ppo-rough-terrain.md](docs/baselines/vanilla-ppo-rough-terrain.md)
- [docs/baselines/heuristic-action-rate-sweep.md](docs/baselines/heuristic-action-rate-sweep.md)
- [docs/baselines/rough-terrain-formal-comparison.md](docs/baselines/rough-terrain-formal-comparison.md)

Current status and next step:

- [docs/sc-ppo-report-status.md](docs/sc-ppo-report-status.md)
- [docs/sc-ppo-current-summary.md](docs/sc-ppo-current-summary.md)
- [docs/sc-ppo-current-blockers.md](docs/sc-ppo-current-blockers.md)
- [docs/sc-ppo-next-step-direction.md](docs/sc-ppo-next-step-direction.md)
- [docs/reproduction/final-research-delivery-checklist.md](docs/reproduction/final-research-delivery-checklist.md)

Post-freeze reusable diagnostics on `main`:

- [docs/sc-ppo-objective-mismatch-diagnostic.md](docs/sc-ppo-objective-mismatch-diagnostic.md)
- [docs/sc-ppo-behavior-smoothness-metric-diagnostic.md](docs/sc-ppo-behavior-smoothness-metric-diagnostic.md)

## Reproducibility notes

- Runtime outputs are written under `artifacts/`
- Main experiment records use `manifest.json` files to capture config, run path, and selected
  evaluation artifacts
- For longer-budget comparisons, the repo prefers `checkpoint-sweep` selection over blindly using
  the final checkpoint

## Reading order

If you are new to the repo, the fastest way to build context is:

1. [CONTEXT.md](CONTEXT.md)
2. [docs/sc-ppo-current-summary.md](docs/sc-ppo-current-summary.md)
3. [docs/sc-ppo-report-status.md](docs/sc-ppo-report-status.md)
4. [docs/baselines/rough-terrain-formal-comparison.md](docs/baselines/rough-terrain-formal-comparison.md)
5. [docs/sc-ppo-next-step-direction.md](docs/sc-ppo-next-step-direction.md)
6. [docs/sc-ppo-pid-limited-ablation.md](docs/sc-ppo-pid-limited-ablation.md)
7. [docs/sc-ppo-sn-feasibility-diagnostic.md](docs/sc-ppo-sn-feasibility-diagnostic.md)
8. [docs/sc-ppo-sn-prototype.md](docs/sc-ppo-sn-prototype.md)
9. [docs/random-stairs-selected-checkpoint-stress.md](docs/random-stairs-selected-checkpoint-stress.md)
10. [docs/reproduction/final-research-delivery-checklist.md](docs/reproduction/final-research-delivery-checklist.md)
11. [docs/adr/0001-freeze-research-delivery-before-new-protocol-repair.md](docs/adr/0001-freeze-research-delivery-before-new-protocol-repair.md)

## Current repo state

The repo has completed `科研交付冻结 / 仓库内科研交付包`.

`main` should now be read as the frozen internal research delivery package rather than as an active
experiment branch:

- Isaac rough-terrain: `SC-PPO 3.8` supports the current `方法优于启发式` claim against the revised
  heuristic anchor
- `MuJoCo isaac_mainline`: aligned replay is `混合外部验证结论`
- `PID有限消融`: closed as limited mechanism support for `PID-Lagrangian正式方案`, not a new mainline
- `SN`: the bounded `替代机制可行性诊断` is closed negative; full-actor `smoke`, `short`,
  `medium`, hidden-layer-only `medium`, hidden-layer-only `coeff = 2.0`, and first-hidden-only
  `medium` reduced-budget runs completed but collapsed, so the repo should not continue blind
  SN-only architecture toggles
- `#7 随机阶梯`: closed as a selected-checkpoint stress test; all three methods collapse under the
  first stairs-only random-stairs protocol, so it records direct transfer failure rather than a
  task-valid method ranking
- post-freeze reusable diagnostics now included on `main`: objective-mismatch checkpoint alignment
  and behavior-smoothness trace summaries
- future moderated random-stairs, terrain protocol repair, or task-stabilized SN work should be
  opened as separate post-freeze branches rather than modifying the frozen package on `main`

Tracked freeze references:

- [artifacts/analysis/final_research_delivery_freeze/summary.json](artifacts/analysis/final_research_delivery_freeze/summary.json)
- [docs/reproduction/final-research-delivery-checklist.md](docs/reproduction/final-research-delivery-checklist.md)
- [docs/sc-ppo-objective-mismatch-diagnostic.md](docs/sc-ppo-objective-mismatch-diagnostic.md)
- [docs/sc-ppo-behavior-smoothness-metric-diagnostic.md](docs/sc-ppo-behavior-smoothness-metric-diagnostic.md)
