# SCO-humanoid

SCO-humanoid is a research-oriented repository for validating constrained reinforcement learning
methods for smoother humanoid locomotion.

`SCO` stands for `Smooth-Constrained-Optimization`.

The current full-paper branch uses `Humanoid-Gym` as the training backbone, evaluates methods under
a shared metric schema, and compares:

- `LCP-style soft Jacobian/Lipschitz penalty`
- `SC-PPO 3.8` with a hard policy-local-sensitivity constraint and `PID-Lagrangian` enforcement
- `PPO + heuristic smoothing` via a revised action-rate reward-shaping anchor
- bounded diagnostics for `OmniSafe PPO-Lag`, local `CPO-style` updates, and historical alternative
  smoothness mechanisms

The project is organized as an experiment repo, not as a general-purpose framework. The main goal
is reproducible evidence, comparative baselines, and report-grade analysis.

## What this repo is for

Use this repo if you want to:

- reproduce the current rough-terrain smooth-control experiments
- inspect the evidence chain behind the current full-paper mechanism-comparison line
- rerun baseline sweeps, checkpoint selection, and MuJoCo replay under the same metric schema
- build the current venue-neutral full-paper LaTeX manuscript source

This repo is not designed as:

- a reusable locomotion RL framework
- a pip-installable package
- a generalized benchmark suite for many robots or tasks

## Research question

The core question is:

`Is policy-local-sensitivity regularization a useful smooth-control mechanism for humanoid locomotion, and how do hard constraints, soft penalties, and reward-shaping anchors trade off under the same Isaac/MuJoCo protocol?`

The current main task is rough-terrain velocity-tracking locomotion in Isaac Gym, with MuJoCo
sim-to-sim replay used as external validation.

## Current status

The active branch is `full-paper/extended-seeds`. The current paper package is a
mechanism-comparison result, not an `SC-PPO beats all baselines` result.

Current thesis:

`Policy-local-sensitivity regularization is a useful smooth-control lens, but enforcement details matter and no single row dominates every metric.`

Primary five-seed Isaac selected-checkpoint comparison over seeds `11/17/23/29/31`:

| Method | Selected ckpts | Fall | Vel. err | Jnt acc | Jitter | Return | Sens. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | `300/400/400/400/400` | `0.000` | `0.490` | `117.331` | `0.212` | `118.420` | `1.890` |
| SC-PPO 3.8 PID | `300/300/400/400/400` | `0.170` | `0.606` | `142.955` | `0.277` | `99.349` | `3.630` |
| Revised heuristic | `350/300/350/400/400` | `0.150` | `0.705` | `115.317` | `0.260` | `105.326` | `7.331` |

Matched five-seed MuJoCo selected replay:

| Method | Fall | Vel. err | Jnt acc | Jitter | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | `0.000` | `0.406` | `117.425` | `0.195` | `-599.108` |
| SC-PPO 3.8 PID | `0.010` | `0.471` | `159.718` | `0.322` | `-627.238` |
| Revised heuristic | `0.000` | `0.406` | `111.615` | `0.226` | `-456.370` |

Important boundaries:

- `LCP-style` is a same-task local adaptation, not official LCP code/checkpoint parity.
- `SC-PPO` is the hard-constraint/PID-Lagrangian mechanism row; it is useful for understanding
  enforcement and seed/checkpoint sensitivity, not because it is the strongest row.
- The revised heuristic remains a strong reward-shaping anchor and wins matched MuJoCo aggregate
  joint acceleration and return.
- MuJoCo evidence is mixed but interpretable as a control-path metric split: LCP is cleanest on
  action jitter, while downstream joint acceleration and return depend on the full closed loop.
- `OmniSafe PPO-Lag` and local `CPO-style` work are diagnostic/future-work material only; neither
  is promoted to a main baseline.
- There is no hardware validation, official LCP parity, broad hyperparameter sweep, or
  multi-robot/multi-terrain evidence.

The buildable full-paper source is [docs/paper/full-paper.tex](docs/paper/full-paper.tex).

The frozen `main` branch remains a `冻结主档案分支` for the older internal research delivery package.
Historical SC-PPO-centered docs remain useful context, but they no longer carry the current
full-paper claim.

## Fastest current handoff

If you only need the current full-paper answer, read:

1. [docs/paper/full-paper.tex](docs/paper/full-paper.tex)
2. [docs/paper/full-paper-build.md](docs/paper/full-paper-build.md)
3. [docs/paper/full-paper-red-team-notes.md](docs/paper/full-paper-red-team-notes.md)
4. [docs/full-paper/full-paper-narrative-integration.md](docs/full-paper/full-paper-narrative-integration.md)
5. [docs/paper/reviewer-risk-checklist.md](docs/paper/reviewer-risk-checklist.md)
6. [docs/full-paper/related-work-claim-boundary-map.md](docs/full-paper/related-work-claim-boundary-map.md)

Build the venue-neutral manuscript source:

```bash
cd docs/paper
make
make clean
```

Generated PDFs and LaTeX auxiliary files are intentionally ignored by git.
Do not commit compiled submission packages.

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

### 3. Read or build the current full-paper package

Start from:

- [docs/paper/full-paper.tex](docs/paper/full-paper.tex)
- [docs/paper/full-paper-build.md](docs/paper/full-paper-build.md)
- [docs/paper/full-paper-red-team-notes.md](docs/paper/full-paper-red-team-notes.md)

To build:

```bash
cd docs/paper
make
make clean
```

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
- resolves repo-relative `configs/...json` paths against the repo root, so train/evaluate wrappers
  do not depend on the caller's current working directory
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

Full-paper manuscript and claim boundary:

- [docs/paper/full-paper.tex](docs/paper/full-paper.tex)
- [docs/paper/full-paper-build.md](docs/paper/full-paper-build.md)
- [docs/paper/full-paper-red-team-notes.md](docs/paper/full-paper-red-team-notes.md)
- [docs/paper/full-paper-mechanism-comparison-draft.md](docs/paper/full-paper-mechanism-comparison-draft.md)
- [docs/paper/manuscript-skeleton.md](docs/paper/manuscript-skeleton.md)
- [docs/paper/reviewer-risk-checklist.md](docs/paper/reviewer-risk-checklist.md)
- [docs/full-paper/full-paper-narrative-integration.md](docs/full-paper/full-paper-narrative-integration.md)
- [docs/full-paper/related-work-claim-boundary-map.md](docs/full-paper/related-work-claim-boundary-map.md)
- [docs/paper/references.bib](docs/paper/references.bib)

Historical workshop/frozen-mainline docs:

- [docs/sc-ppo-report-status.md](docs/sc-ppo-report-status.md)
- [docs/sc-ppo-current-summary.md](docs/sc-ppo-current-summary.md)
- [docs/sc-ppo-current-blockers.md](docs/sc-ppo-current-blockers.md)
- [docs/sc-ppo-next-step-direction.md](docs/sc-ppo-next-step-direction.md)
- [docs/sc-ppo-cross-engine-degradation.md](docs/sc-ppo-cross-engine-degradation.md)
- [docs/paper/arxiv-workshop-manuscript.md](docs/paper/arxiv-workshop-manuscript.md)
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
2. [docs/paper/full-paper.tex](docs/paper/full-paper.tex)
3. [docs/paper/full-paper-red-team-notes.md](docs/paper/full-paper-red-team-notes.md)
4. [docs/full-paper/full-paper-narrative-integration.md](docs/full-paper/full-paper-narrative-integration.md)
5. [docs/paper/reviewer-risk-checklist.md](docs/paper/reviewer-risk-checklist.md)
6. [docs/full-paper/related-work-claim-boundary-map.md](docs/full-paper/related-work-claim-boundary-map.md)
7. [docs/full-paper/statistical-robustness-results.md](docs/full-paper/statistical-robustness-results.md)
8. [docs/full-paper/mujoco-mixed-evidence-mechanism.md](docs/full-paper/mujoco-mixed-evidence-mechanism.md)
9. [docs/full-paper/policy-perturbation-audit.md](docs/full-paper/policy-perturbation-audit.md)
10. [docs/reproduction/final-research-delivery-checklist.md](docs/reproduction/final-research-delivery-checklist.md)
11. [docs/adr/0001-freeze-research-delivery-before-new-protocol-repair.md](docs/adr/0001-freeze-research-delivery-before-new-protocol-repair.md)

## Current repo state

The repo completed `科研交付冻结 / 仓库内科研交付包` and then expanded onto
`full-paper/extended-seeds`. The current branch now has:

- a buildable venue-neutral LaTeX manuscript package;
- five-seed Isaac and matched five-seed MuJoCo comparisons for LCP-style, SC-PPO, and heuristic
  rows;
- paired bootstrap and selected-vs-final robustness audits;
- a policy perturbation audit that supports the local policy-output mechanism;
- bounded OmniSafe PPO-Lag and local CPO-style diagnostics kept out of the main baseline table;
- a reviewer-risk pass recording the claim boundaries and remaining gaps.

The paper should be read as a mechanism comparison:

- positive: policy-local-sensitivity regularization is a useful lens for smooth humanoid control;
- positive: LCP-style soft regularization is the strongest local-sensitivity row in this same-task
  audit;
- positive: the revised heuristic is a strong reward-shaping anchor and remains best on matched
  MuJoCo aggregate joint acceleration and return;
- diagnostic: SC-PPO exposes the hard-constraint/PID-Lagrangian enforcement trade-off;
- diagnostic: OmniSafe/CPO paths clarify interface and future-work boundaries;
- limitation: no official LCP parity, hardware validation, broad hyperparameter search,
  multi-robot evidence, or multi-terrain evidence.

`main` operates as a `冻结主档案分支`; `full-paper/extended-seeds` is the current full-paper work
branch.
