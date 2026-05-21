# SC-PPO SN Feasibility Diagnostic

This note defines the minimal first-stage plan for the repo's post-mainline
`架构级平滑优化线`.

It is intentionally narrower than a formal candidate-line plan.

## Purpose

This branch is a `替代机制可行性诊断`.

It does not ask whether `SN` immediately becomes the new formal mainline.
It asks whether `SN` can become a credible replacement path for the current
Jacobian-penalty mechanism.

## Core question

The first-stage question is:

`Can an actor-side Spectral Normalization path train stably and produce interpretable smoothness evidence without relying on the current Jacobian-penalty loss?`

## Comparison rule

This branch must follow `同尺比较`.

That means:

- same task family: `速度跟踪行走`
- same primary terrain: `粗糙平面`
- same shared behavior metrics
- same checkpoint-sweep habit if the branch proves worth continuing
- same external reporting distinction between behavior-level evidence and
  mechanism-side evidence

## First comparison target

The first comparison is:

`SN diagnostic branch` vs `current SC-PPO threshold = 3.8 diagnostic config`

This is a one-to-one mechanism comparison.

It is not yet:

- `SN` vs heuristic
- `SN` vs Vanilla PPO
- full formal comparison matrix

Current implementation note:

- [SC-PPO SN Prototype](./sc-ppo-sn-prototype.md)

## Minimal implementation tasks

1. Add a new method-family path or configuration switch that enables actor-side
   `Spectral Normalization` without mutating the current `SC-PPO` mainline path.
2. Add a repeatable reduced-budget diagnostic launcher:
   - `scripts/baseline/run_sn_diagnostic.py`
3. Keep the existing evaluation entrypoints usable:
   - `scripts/baseline/train_vanilla_ppo.py`
   - `scripts/baseline/evaluate_policy.py`
   - `scripts/baseline/evaluate_checkpoint_sweep.py`
4. Preserve current artifact hygiene:
   - `manifest.json`
   - evaluation metrics JSON
   - reportable config identity
5. Keep current `policy_local_sensitivity_cost_*` evaluation available wherever
   feasible so the replacement branch still emits comparable mechanism-side
   evidence.

Current one-command diagnostic entry:

```bash
python scripts/baseline/run_sn_diagnostic.py --stage all --preset smoke --skip-completed
```

For a slightly larger but still non-formal pass:

```bash
python scripts/baseline/run_sn_diagnostic.py --stage all --preset short --skip-completed
```

For the next single-seed diagnostic after matched-control isolation:

```bash
python scripts/baseline/run_sn_diagnostic.py --stage all --preset medium --skip-completed
```

For the output-layer isolation diagnostic:

```bash
python scripts/baseline/run_sn_diagnostic.py \
  --config configs/methods/sn_ppo_hidden_only_rough_terrain.json \
  --stage all \
  --preset medium \
  --skip-completed
```

All presets remain `替代机制可行性诊断`; none should be reported as a formal mainline challenge.

Current runner status:

- `smoke` has completed once with `run_name = sn_ppo_rough_terrain_smoke_seed123145`
- the run wrote a compact summary to
  `artifacts/analysis/sn_replacement_diagnostic/sn_ppo_rough_terrain_smoke_seed123145_summary.json`
- the result is operationally useful but not task-valid:
  `selection_status = all_checkpoints_collapsed`, `fall_rate = 1.0000`
- `short` has also completed once with `run_name = sn_ppo_rough_terrain_short_seed123145`
- `short` remains not task-valid:
  `selection_status = all_checkpoints_collapsed`, `fall_rate = 1.0000`,
  `episode_return_mean = 3.5569`
- `medium` has completed once with `run_name = sn_ppo_rough_terrain_medium_seed123145`
- `medium` remains not task-valid:
  `selection_status = all_checkpoints_collapsed`, `fall_rate = 1.0000`,
  `velocity_tracking_error_mean = 1.1948`, `episode_return_mean = 3.0827`

Mechanism check:

- the `short` checkpoint contains actor spectral-normalization state
  (`actor.*.weight_orig`, `actor.*.weight_u`, `actor.*.weight_v`)
- so the current failure is not explained by the SN switch being absent from the training graph

Next isolation control:

- config:
  `configs/methods/ppo_no_smoothness_rough_terrain_diagnostic.json`
- purpose: compare the same no-smoothness-reward PPO path without actor-side SN
- result: the matched `short` control also collapses with `fall_rate = 1.0000`
- interpretation: the immediate issue is likely the reduced-budget no-Jacobian/no-heuristic path,
  not SN being uniquely broken
- next action: do not promote or scale seeds; revise the SN parameterization or training recipe
  before spending more diagnostic budget

Output-layer isolation:

- config:
  `configs/methods/sn_ppo_hidden_only_rough_terrain.json`
- purpose: test whether full-actor SN collapse is caused by constraining the actor output layer
- result: hidden-layer-only `medium` also collapses with `fall_rate = 1.0000`
- selected checkpoint: `100`
- `velocity_tracking_error_mean = 1.4903`
- `joint_acceleration_l2_mean = 90.0006`
- `action_jitter_l2_mean = 0.1019`
- `episode_return_mean = 3.4648`
- `policy_local_sensitivity_cost_mean = 1.3343`
- mechanism check: `actor.0`, `actor.2`, and `actor.4` contain SN state, while output layer
  `actor.6` contains normal `weight` and `bias`
- interpretation: output-layer SN is not the blocker; hidden-only SN is operational but still not
  task-valid

Current decision:

- SN is operational and emits comparable mechanism-side evidence
- SN is not task-valid under the current reduced-budget diagnostic presets
- actor-side SN is confirmed present in the checkpoint, so the failure is not a missing-config bug
- hidden-layer-only SN is also confirmed active, so the failure is not explained by the output layer
  being SN-constrained
- this branch should stay open only for mechanism tuning; it should not consume formal-comparison or
  MuJoCo budget in the current form

## Minimal experiment tasks

The first batch should stay diagnostic and low-budget.

Recommended sequence:

1. one smoke training run to verify the branch trains and writes artifacts
2. one short diagnostic comparison against current `SC-PPO threshold = 3.8`
3. one checkpoint sweep only if the short diagnostic run shows nontrivial signal

The branch should not consume:

- `主实验三种子`
- `MuJoCo关键两组终验`

until the mechanism replacement question is answered positively.

## First-stage success criterion

The branch clears first-stage feasibility only if all of the following hold:

- training is operational without breaking artifact generation
- the branch produces readable behavior-level metrics
- the branch still produces interpretable mechanism-side evidence under the
  current schema
- short-budget results are not obviously degenerate relative to the current
  `3.8` diagnostic reference

## First-stage failure criterion

The branch should be considered a failed first-stage diagnostic if any of the
following dominate:

- training becomes unstable or operationally brittle
- the branch cannot be compared under the existing evidence schema
- behavior-level metrics collapse even before formal budgeting is considered
- the branch requires changing too many assumptions at once to preserve
  interpretability

## Promotion rule

Only after first-stage feasibility is established should the repo discuss
whether `SN` deserves promotion from `替代机制可行性诊断` into a real
`诊断支线` that can challenge the current mainline.
