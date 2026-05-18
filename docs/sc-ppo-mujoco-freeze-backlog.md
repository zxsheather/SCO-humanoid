# SC-PPO MuJoCo Freeze Backlog

This note records the concrete work needed to finish the current
`主线证据闭环` for the repo's `MuJoCo` reading.

It separates report-freeze work from the separate `MuJoCo terrain`
`协议修复线`.

## Part 1: Report-freeze items

These items belong to the current mainline closure work.

### Required frozen reading

The repo's current main report must align on:

- Isaac `粗糙平面` as the main result
- `MuJoCo isaac_mainline` as a bounded `部分迁移结论`
- no claim that smoothness fully transfers across engines
- no claim that current `MuJoCo terrain` is report-grade

### Required document checks

The following documents should agree on the same wording boundary:

- `report.md`
- `report.zh.md`
- `docs/sc-ppo-report-status.md`
- `docs/sc-ppo-current-summary.md`
- `docs/sc-ppo-current-blockers.md`

### Required artifact checks

The frozen citation set should continue to point at:

- heuristic `metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `SC-PPO threshold = 3.8` representative
  `metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- current `3-seed` Isaac checkpoint-sweep summaries

### Completion condition

This report-freeze segment is complete when a reader cannot plausibly confuse:

- `部分迁移结论`
with
- `全面跨引擎转优`

## Part 2: MuJoCo terrain 协议修复线

This work is not part of the current mainline claim.

It is a separate bounded repair line.

### Current split

The repo should maintain the explicit terrain split:

- `terrain_mode = isaac_mainline`
- `terrain_mode = hfield_moderate`
- `terrain_mode = hfield_stress`

### Current reading

- `isaac_mainline` is the minimal comparable replay line
- `hfield_moderate` is the current repair-stage intermediate line
- `hfield_stress` is the transfer-pressure line

### Repair questions

The next terrain-side questions should be framed as protocol questions first:

1. Is the current terrain protocol discriminative enough to separate methods
   cleanly?
2. Is the current terrain protocol aligned with the repo's intended external
   validation semantics?
3. Does terrain repair need protocol adjustment first, or algorithm-robustness
   work first?

### Non-goal

The repo should not currently fold any terrain-side result into the main report
headline as if it were already report-grade external validation.

## Recommended order

The recommended order is:

1. freeze `MuJoCo isaac_mainline` wording and citations
2. lock the report boundary around `部分迁移结论`
3. keep `hfield_moderate` and `hfield_stress` visible as blocked or repair-stage
   protocol lines
4. only then decide whether to spend additional budget on terrain protocol
   repair
