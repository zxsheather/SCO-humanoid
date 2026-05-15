# SC-PPO Full-Batch Threshold Promotion Protocol

This note records how the repo should evaluate whether `threshold = 3.6 + full_batch` is allowed to
challenge the current formal `threshold = 3.8` mainline.

It is a comparison protocol for the current experiment stage, not an ADR and not a final result
summary.

## Current roles

- current formal mainline:
  `SC-PPO threshold = 3.8`
- current promotion candidate:
  `SC-PPO threshold = 3.6 + full_batch`
- frozen diagnostic branch:
  `SC-PPO threshold = 3.7 + full_batch`

The `3.6 + full_batch` line is now a `正式候选线`, not just a `诊断支线`.
However, it is not allowed to replace the current mainline until it clears the promotion gates
below.

## Comparison target

The current promotion question is:

`3.6 + full_batch` 是否足够稳定且足够有价值，值得作为新的正式主线候选去挑战当前 `3.8` 主线？

This is not a new repo-wide `方法优于启发式` question.
That claim is already represented by the current `3.8` mainline evidence.

So the primary comparison for this promotion stage is:

- `3.6 + full_batch` vs current `3.8` mainline

The heuristic anchor remains only as a floor-check reference, not as the first comparison target.

## Execution scope

Promotion work should stay narrow:

- reuse the completed `seed11` result
- only add `seed17` and `seed23`
- run `400 iteration + checkpoint sweep`
- do not rely on the final checkpoint alone
- do not expand `3.7 + full_batch` into the same promotion batch

## Validation order

The validation order is fixed:

1. run Isaac `400 iteration + checkpoint sweep` for `seed17`
2. run Isaac `400 iteration + checkpoint sweep` for `seed23`
3. check promotion gates on both seeds
4. only if both seeds pass, run `MuJoCo isaac_mainline`

If either seed fails the Isaac-side promotion gates, stop the promotion line before spending new
`MuJoCo` budget.

## Promotion gates

### Gate 1: per-seed hard gate

Each new seed must clear a `逐种子硬门槛` before aggregate evidence is allowed to count.

For each seed:

- selected checkpoint must come from `checkpoint sweep`
- selected checkpoint must not collapse to a pathological early or null result
- Isaac selected-checkpoint metrics must preserve the current `任务守底线`

If one seed fails here, the line does not advance to new `MuJoCo` evaluation.

### Gate 2: early-checkpoint failure rule

The repo applies an `早期checkpoint失效规则` during promotion.

If a seed selects an overly early checkpoint such as `50` or `100`, treat that as promotion
failure even if the metrics at that checkpoint look locally attractive.

The reason is that promotion requires a credible long-budget operating point rather than a fragile
mid-training spike.

### Gate 3: cross-engine challenge gate

Only after both new seeds pass the Isaac-side gates may the repo run `MuJoCo isaac_mainline` for
those seeds.

The promotion reading is still narrow:

- preserve task-valid behavior
- preserve strong `fall_rate` behavior
- require `joint_acceleration_l2_mean` to remain no worse than the current `3.8` mainline reading
- treat `action_jitter_l2_mean` only as supporting evidence rather than a standalone rescue metric

## Out of scope

The following are not part of the current promotion protocol:

- `hfield_moderate`
- `hfield_stress`
- another threshold-neighborhood expansion
- rerunning the completed `seed11`
- promoting `3.7 + full_batch`

Those remain separate questions and should not be mixed into the `3.6` promotion decision.

## Naming and artifact retention

During promotion, the candidate line should stay inside the existing
`sc_ppo_fullbatch_threshold_probe` naming and artifact namespace.

Do not rename the experiment family or move completed `seed11` artifacts into a new mainline-style
directory before promotion succeeds.

The repo should only consider a namespace migration after `3.6 + full_batch` actually clears the
promotion protocol and is ready to replace the current `3.8` mainline.

## Current outcome

The current promotion attempt has now been evaluated on:

- reused `seed11`
- new `seed17`
- new `seed23`

Observed Isaac-side result:

- `seed11 -> checkpoint 350`
- `seed17 -> checkpoint 350`
- `seed23 -> checkpoint 0`

This means the current promotion attempt fails at the Isaac stage.

Reason:

- `seed23` triggers the repo's `早期checkpoint失效规则`
- so the candidate line does not clear the `逐种子硬门槛`
- therefore the promotion protocol stops before any new `MuJoCo isaac_mainline` budget is spent

Current reading:

- `3.6 + full_batch` produced a meaningful single-seed diagnostic improvement
- but it does not currently qualify as a stable `正式候选线` that can replace the `3.8` mainline
- after this outcome, the line should be read again as an informative `诊断支线`, not as an active
  promotion candidate
