# Output-Side Scaling Line

- **Date**: 2026-05-23
- **Type**: experiment
- **Outcome**: failure
- **Tags**: architecture-line, output-scaling, replacement

## Timeline and Background

Issue [`#33`](https://github.com/zxsheather/SCO-humanoid/issues/33) opens after the canonical action-side scaling line had already closed negative. The repo decided the old mean-only `action_scaling_ppo` prototype was not a clean output-side comparison, so it opened a replacement line.

## Technical Details

- Issue body defines the scope: a distinct `output_scaling_ppo` path, canonical config, sweep, and test coverage.
- The closing issue comments record the canonical 3-seed run on `11 / 17 / 23`, `512 envs x 400 iterations`.
- Reported aggregate:
  - `fall_rate_mean = 0.4333`
  - `velocity_tracking_error_mean_mean = 0.7689`
  - `joint_acceleration_l2_mean_mean = 121.36`
  - `action_jitter_l2_mean_mean = 0.2164`
- Result: still outside the shared entry gate because `seed17` remained clearly task-invalid.

## Decision Process

- The repo treated this as a clean same-question failure, not as proof that nearby output-scale schedules should be reopened immediately.
- Branch nuance:
  - the local branch name `output-scaling-line` now points at `main`
  - the durable historical record is therefore mostly in issue `#33` and the later umbrella probe branch commit `82d6032`

## Results and Impact

- This negative result pushed the repo away from scaling families and into architecture-line candidates.
- Issue [`#34`](https://github.com/zxsheather/SCO-humanoid/issues/34) explicitly cites both action-side and output-side scaling as already closed negative when it proposes the orthogonal actor line.
- Related follow-up: [orthogonal-actor-line](../nodes/orthogonal-actor-line.md).
