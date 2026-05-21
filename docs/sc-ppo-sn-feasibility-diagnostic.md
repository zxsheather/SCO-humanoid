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

Both presets remain `替代机制可行性诊断`; neither should be reported as a formal mainline challenge.

Current runner status:

- `smoke` has completed once with `run_name = sn_ppo_rough_terrain_smoke_seed123145`
- the run wrote a compact summary to
  `artifacts/analysis/sn_replacement_diagnostic/sn_ppo_rough_terrain_smoke_seed123145_summary.json`
- the result is operationally useful but not task-valid:
  `selection_status = all_checkpoints_collapsed`, `fall_rate = 1.0000`
- the next implementation step should use the `short` preset only as another diagnostic, not as a
  formal promotion attempt

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
