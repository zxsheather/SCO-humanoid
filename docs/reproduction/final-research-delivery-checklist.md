# Final Research Delivery Checklist

This checklist defines the already completed `仓库内科研交付包`. It is an operational handoff index
for the completed evidence, not a request to generate new results on `main`.

## Scope

Freeze target:

- Isaac rough-terrain main result: `SC-PPO 3.8 + PID-Lagrangian` beats the revised heuristic anchor
  under the shared metric schema.
- `MuJoCo isaac_mainline`: aligned replay is `混合外部验证结论`, not a cross-engine `SC-PPO` win.
- `PID有限消融`: matched plain-dual probe collapses and only supports the PID-Lagrangian boundary.
- `SN-only`: replacement-mechanism diagnostic is operational but negative.
- `随机阶梯`: selected rough-terrain checkpoints collapse under the first stairs-only stress test.

Out of scope for this frozen package:

- rerunning Isaac training
- rerunning MuJoCo replay
- opening moderated random-stairs protocol repair
- continuing blind SN-only architecture toggles
- turning the repo into an engineering product or paper-submission package

Tracked freeze summary:

- `artifacts/analysis/final_research_delivery_freeze/summary.json`

## Main-Branch Posture

`main` is now a `冻结主档案分支`.

Allowed bounded backports on `main`:

- `冻结边界章节` updates that refine freeze-boundary wording without reopening the main narrative
- reusable evaluation or diagnostic infrastructure that strengthens artifact interpretation across
  completed lines

Disallowed on `main`:

- rerunning training or replay to generate new evidence
- backporting mechanism-specific negative branches as if they were part of the frozen mainline
- reopening moderated stairs, SN recipe redesign, or new algorithm lines inside the frozen package

## Freeze Validation

Use the project Python environment:

```bash
cd /home/zhuoxiang/SCO-humanoid
export PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python
```

Lightweight checks:

```bash
$PYTHON_BIN scripts/baseline/check_env.py
$PYTHON_BIN -m unittest discover -s tests
$PYTHON_BIN - <<'PY'
import json
from pathlib import Path

roots = [Path("configs"), Path("artifacts/analysis")]
for root in roots:
    for path in root.rglob("*.json"):
        with path.open("r", encoding="utf-8") as fh:
            json.load(fh)
print("json_valid")
PY
git diff --check
```

These checks validate environment wiring, unit-level experiment utilities, JSON parseability, and
patch hygiene. They do not validate new scientific evidence.

Minimal targeted regression for the post-freeze reusable backports:

```bash
$PYTHON_BIN -m unittest \
  tests.test_baseline_common \
  tests.test_behavior_trace_metrics \
  tests.test_checkpoint_sweep_recovery \
  tests.test_formal_comparison_runner \
  tests.test_baseline_protocol_failfast
```

These targeted tests cover the shared `load_run` resolution fix, repo-root config-path resolution
for the formal comparison wrapper, plus the merged objective-mismatch and behavior-smoothness
checkpoint diagnostics now carried on `main`.

## Bounded Post-Freeze Diagnostic Backports on `main`

These merged diagnostics are part of the current frozen package as reusable evaluation or
diagnostic infrastructure, not as a reopened algorithm mainline:

- `objective mismatch diagnostic`:
  - canonical doc: `docs/sc-ppo-objective-mismatch-diagnostic.md`
  - canonical derived output: `checkpoint_diagnostic_alignment.json`
  - checkpoint-sweep support: eval/train constraint ranges, correlations, and collapsed-task-floor
    summaries
- `behavior smoothness diagnostic`:
  - canonical doc: `docs/sc-ppo-behavior-smoothness-metric-diagnostic.md`
  - canonical derived outputs: `episode_traces_checkpoint_<N>.json`,
    `behavior_smoothness_metrics_checkpoint_<N>.json`,
    `behavior_smoothness_metrics_selected.json`
  - checkpoint-sweep support: per-episode trace capture plus selected-checkpoint smoothness
    summaries
- `shared run-dir cleanup`:
  - canonical support file: `scripts/baseline/_common.py`
  - purpose: accept historical repo-relative `logs/...` style `load_run` paths under the configured
    `Humanoid-Gym` root

## Evidence Layer 1: Isaac Rough-Terrain Mainline

Current reading:

- `Vanilla PPO` is a collapsed raw reference.
- Revised heuristic anchor: `action_rate = -0.0050`, `512 envs x 400 iterations`, selected
  checkpoints `350 / 300 / 350`.
- Formal mainline: `SC-PPO threshold = 3.8`, `PID-Lagrangian`, selected checkpoints
  `300 / 300 / 400`.
- Isaac rough-terrain supports `方法优于启发式` for `SC-PPO 3.8` against the revised heuristic anchor.

Historical reproduction commands:

```bash
$PYTHON_BIN -u scripts/baseline/run_formal_comparison.py \
  --sweep-config configs/sweeps/rough_terrain_formal_comparison.json \
  --stage all \
  --skip-completed

$PYTHON_BIN -u scripts/baseline/run_formal_comparison.py \
  --sweep-config configs/sweeps/rough_terrain_formal_protocol_revision_long_budget.json \
  --stage all \
  --skip-completed
```

Canonical artifacts:

- `artifacts/analysis/final_research_delivery_freeze/summary.json`
- `artifacts/analysis/sc_ppo_report_figures/figure_isaac_main_result.png`
- `artifacts/analysis/sc_ppo_report_figures/manifest.json`

Local raw output locations, when this workspace's ignored runtime artifacts are available:

- `artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/metrics_selected.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/metrics_selected.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/metrics_selected.json`

Reference docs:

- `docs/sc-ppo-report-status.md`
- `docs/sc-ppo-current-summary.md`
- `report.md`
- `report.zh.md`

## Evidence Layer 2: MuJoCo Aligned Replay

Current reading:

- Revised heuristic is stronger on task stability, velocity tracking, episode length, and joint
  acceleration.
- `SC-PPO 3.8` is only slightly stronger on action jitter.
- This is `混合外部验证结论`, not a cross-engine `SC-PPO` win.

Historical reproduction commands:

```bash
PYTHON_BIN=$PYTHON_BIN bash scripts/baseline/run_mujoco_revised_heuristic_parallel.sh
PYTHON_BIN=$PYTHON_BIN bash scripts/baseline/run_mujoco_scppo38_parallel.sh
```

Canonical artifacts:

- `artifacts/analysis/final_research_delivery_freeze/summary.json`
- `docs/sc-ppo-mujoco-revised-anchor-aligned-comparison.md`
- `artifacts/analysis/sc_ppo_report_figures/figure_mujoco_aligned_replay.png`
- `artifacts/analysis/sc_ppo_report_figures/manifest.json`

Local raw output locations, when this workspace's ignored runtime artifacts are available:

- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`

## Evidence Layer 3: PID-Limited Ablation

Current reading:

- Matched `普通对偶上升` at `threshold = 3.8` collapses at checkpoint `100`.
- Lower action jitter does not count as a usable smooth-control win when `fall_rate = 1.0`.
- The result supports `PID-Lagrangian正式方案` as the formal SC-PPO update, but not broad
  component attribution.

Historical reproduction command template:

```bash
$PYTHON_BIN -u scripts/baseline/train_vanilla_ppo.py \
  --config configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_dual_001.json \
  --run-name sc_ppo_threshold_38_lambda_05_quantile_090_dual_001_rough_terrain_iter100 \
  --num-envs 512 \
  --max-iterations 100 \
  --seed 123145

$PYTHON_BIN scripts/baseline/evaluate_checkpoint_sweep.py \
  --config configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_dual_001.json \
  --run-name sc_ppo_threshold_38_lambda_05_quantile_090_dual_001_rough_terrain_iter100 \
  --load-run May14_10-44-08_sc_ppo_threshold_38_lambda_05_quantile_090_dual_001_rough_terrain_iter100 \
  --checkpoints 100 \
  --num-envs 32 \
  --episodes 20 \
  --seed 123145
```

Canonical artifacts:

- `artifacts/analysis/final_research_delivery_freeze/summary.json`
- `docs/sc-ppo-pid-limited-ablation.md`
- `artifacts/analysis/sc_ppo_pid_limited_ablation/summary.json`

Local raw output locations, when this workspace's ignored runtime artifacts are available:

- `artifacts/methods/sc_ppo_dual_probe/sc_ppo_threshold_38_lambda_05_quantile_090_dual_001_rough_terrain_iter100/metrics.json`
- `artifacts/methods/sc_ppo_dual_probe/sc_ppo_threshold_38_lambda_05_quantile_090_dual_001_rough_terrain_iter100/checkpoint_sweep_summary.json`

## Evidence Layer 4: SN-Only Diagnostic

Current reading:

- Actor-side spectral normalization is wired and visible in checkpoints.
- Full-actor, hidden-layer-only, `coeff = 2.0`, and first-hidden-only reduced-budget diagnostics
  all remain not task-valid.
- The branch is closed negative and should not consume formal seeds or MuJoCo budget.

Historical reproduction command:

```bash
$PYTHON_BIN scripts/baseline/run_sn_diagnostic.py \
  --config configs/methods/sn_ppo_first_hidden_rough_terrain.json \
  --stage all \
  --preset medium \
  --skip-completed
```

Canonical artifacts:

- `artifacts/analysis/final_research_delivery_freeze/summary.json`
- `docs/sc-ppo-sn-feasibility-diagnostic.md`
- `docs/sc-ppo-sn-prototype.md`
- `artifacts/analysis/sn_replacement_diagnostic/sn_ppo_rough_terrain_smoke_seed123145_summary.json`
- `artifacts/analysis/sn_replacement_diagnostic/sn_ppo_rough_terrain_short_seed123145_summary.json`
- `artifacts/analysis/sn_replacement_diagnostic/sn_ppo_rough_terrain_medium_seed123145_summary.json`
- `artifacts/analysis/sn_replacement_diagnostic/sn_ppo_hidden_only_rough_terrain_medium_seed123145_summary.json`
- `artifacts/analysis/sn_replacement_diagnostic/sn_ppo_hidden_only_coeff_2_rough_terrain_medium_seed123145_summary.json`
- `artifacts/analysis/sn_replacement_diagnostic/sn_ppo_first_hidden_rough_terrain_medium_seed123145_summary.json`

## Evidence Layer 5: Random-Stairs Selected-Checkpoint Stress Test

Current reading:

- The first stairs-only random-stairs protocol evaluates selected rough-terrain checkpoints only.
- Vanilla PPO, the revised heuristic anchor, and `SC-PPO 3.8` all collapse with `fall_rate = 1.0`.
- The result is direct selected-checkpoint transfer failure, not a task-valid random-stairs method
  ranking.

Historical reproduction command:

```bash
$PYTHON_BIN scripts/baseline/run_random_stairs_stress_test.py \
  --stage all \
  --skip-completed
```

Canonical artifacts:

- `artifacts/analysis/final_research_delivery_freeze/summary.json`
- `docs/random-stairs-selected-checkpoint-stress.md`
- `configs/sweeps/random_stairs_selected_checkpoint_stress.json`
- `artifacts/analysis/random_stairs_selected_checkpoint_stress/comparison_summary.json`

Local raw output locations, when this workspace's ignored runtime artifacts are available:

- `artifacts/methods/random_stairs_stress/vanilla_ppo_random_stairs_stress_seed11/metrics_selected.json`
- `artifacts/methods/random_stairs_stress/vanilla_ppo_random_stairs_stress_seed17/metrics_selected.json`
- `artifacts/methods/random_stairs_stress/vanilla_ppo_random_stairs_stress_seed23/metrics_selected.json`
- `artifacts/methods/random_stairs_stress/heuristic_smoothing_action_rate_0050_random_stairs_stress_seed11/metrics_selected.json`
- `artifacts/methods/random_stairs_stress/heuristic_smoothing_action_rate_0050_random_stairs_stress_seed17/metrics_selected.json`
- `artifacts/methods/random_stairs_stress/heuristic_smoothing_action_rate_0050_random_stairs_stress_seed23/metrics_selected.json`
- `artifacts/methods/random_stairs_stress/sc_ppo_threshold_38_pid_random_stairs_stress_seed11/metrics_selected.json`
- `artifacts/methods/random_stairs_stress/sc_ppo_threshold_38_pid_random_stairs_stress_seed17/metrics_selected.json`
- `artifacts/methods/random_stairs_stress/sc_ppo_threshold_38_pid_random_stairs_stress_seed23/metrics_selected.json`

## Freeze Decision Record

The delivery decision is recorded in:

- `docs/adr/0001-freeze-research-delivery-before-new-protocol-repair.md`

Any future moderated-stairs protocol repair, SN recipe redesign, or engineering-product direction
should start from a new post-freeze issue rather than changing this frozen evidence package.
