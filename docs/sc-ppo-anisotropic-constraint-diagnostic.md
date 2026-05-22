# SC-PPO Anisotropic Constraint Diagnostic

This note records the completed post-freeze `诊断支线` that tested anisotropic
policy-local-sensitivity constraints on Isaac rough terrain.

This branch is not a `正式候选线`. It exists to answer one mechanism question:

`can anisotropic support-set weighting repair short-budget rough-terrain collapse without creating misleading constraint evidence?`

## Scope

The branch kept the same shared task and evaluation frame:

- task: `速度跟踪行走`
- condition: Isaac rough terrain
- budget: reduced diagnostic runs, not `主实验三种子`
- comparison rule: `同尺比较`

The branch added:

- anisotropic action-group slices, labels, and weights
- `penalty_mode`
- `update_error_mode`
- optional `legacy_guard_mode = max_with_legacy`

The guard was added only after the first proximal-only masking result showed a clear reporting
loophole.

## Completed reduced-budget runs

Key completed runs under `artifacts/methods/sc_ppo_anisotropic_probe/`:

- `anisotropic_constraint_short25_seed11`
- `anisotropic_constraint_t08_short25_seed11`
- `anisotropic_constraint_t08_pospen_short25_seed11`
- `anisotropic_constraint_t08_pospen_posupd_short25_seed11`
- `anisotropic_constraint_t055_proxonly_pospen_short25_seed11`
- `anisotropic_constraint_t055_proxonly_pospen_legacyguard_short25_seed11`

## Main findings

### 1. Positive-part penalty mattered more than positive-part multiplier update

The largest mechanism confounder was the signed penalty path, not the signed multiplier update.

Current reading:

- `penalty_mode = positive_part` materially reduced reported smoothness cost relative to the
  signed-penalty variants
- switching `update_error_mode` to `positive_part` changed the multiplier trace, but did not
  recover task-valid rough-terrain behavior in the available short budget

### 2. Proximal-only masking can under-report whole-policy sensitivity

The proximal-only masking probe produced a misleadingly low constrained update cost:

- unguarded `policy_local_sensitivity_cost_update = 0.4425`
- unguarded `policy_local_sensitivity_legacy_cost_mean = 0.8775`
- unguarded `constraint_legacy_violation_rate = 1.0`

Interpretation:

- the masked support set made the active constrained cost look clean
- distal chains absorbed sensitivity outside the masked support set
- the whole policy remained highly sensitive under the legacy scalar reading

So the unguarded proximal-only branch should not be interpreted as a valid smoothness win.

### 3. The honesty guard fixes the reporting loophole, not the task collapse

The guarded proximal-only rerun forced update-time cost accounting to stay honest:

- `policy_local_sensitivity_effective_cost_update = 0.4148`
- `policy_local_sensitivity_legacy_cost_update = 0.4858`
- guarded `policy_local_sensitivity_cost_update = 0.4858`
- guarded `lagrange_multiplier = 0.4094`

This confirms the mechanism:

- the guard prevents anisotropic support-set masking from claiming a cheaper update cost than the
  whole-policy legacy scalar metric
- the old loophole was real

But the guarded branch still did not clear the task floor:

- eval `episode_return_mean = 4.0833`
- eval `joint_acceleration_l2_mean = 86.9346`
- eval `velocity_tracking_error_mean = 1.4552`
- eval `fall_rate = 1.0`

So the honesty repair removes the misleading evidence path, but it does not turn the branch into a
task-valid candidate.

## Branch decision

This `诊断支线` is closed with a negative promotion result.

What it supports:

- anisotropic weighting is implementable and testable inside the current `SC-PPO` path
- support-set masking needs whole-policy guardrails if it is allowed at all
- reduced constrained cost alone is not enough evidence when the legacy scalar sensitivity and task
  floor disagree

What it does not support:

- promotion to `正式候选线`
- spending `主实验三种子` budget on this anisotropic masking line
- claiming that proximal-only masking repaired rough-terrain collapse

## Next useful question

The next useful post-freeze mechanism question is no longer:

`which anisotropic weights should we try next?`

It is:

`why can train-time local-sensitivity improvement coexist with rough-terrain fall_rate = 1.0, and what objective mismatch should be tested next?`

That next step should be opened as a separate post-freeze issue rather than continued as blind
anisotropic weight search.
