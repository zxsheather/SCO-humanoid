# LayerNorm Actor Line

- **Date**: 2026-05-24
- **Type**: experiment
- **Outcome**: mixed
- **Tags**: architecture-line, layernorm, open-issue

## Timeline and Background

Issue [`#35`](https://github.com/zxsheather/SCO-humanoid/issues/35) opens immediately after the orthogonal actor line closes. The rationale is explicit: scaling families failed, orthogonal actor failed even after gain isolation, so the next distinct candidate should target hidden activation scale directly.

## Technical Details

- Issue body defines the first canonical entry:
  - actor-side LayerNorm support
  - removal of the active Jacobian/double-backward path
  - canonical `11 / 17 / 23`, `512 envs x 400 iterations` sweep
- Issue comments record two phases:
  - initial implementation and blocked smoke due GPU contention
  - completed canonical formal run
- Canonical read:
  - `fall_rate_mean = 0.8667`
  - `episode_return_mean_mean = 26.0529`
  - `velocity_tracking_error_mean_mean = 1.0676`
  - `joint_acceleration_l2_mean_mean = 185.1820`
  - `action_jitter_l2_mean_mean = 0.3338`

## Decision Process

- The repo did **not** promote the line, but it also did not discard it as completely as orthogonal actor.
- Current interpretation from the comments:
  - partial task-valid rescue relative to orthogonal actor
  - failure mode looks "too aggressive / too rough," not simply "too conservative"
  - keep only one bounded follow-up: lower `actor_output_gain` from `1.00` to `0.75`

## Results and Impact

- This is the only open issue in the current history tree.
- It marks the current frontier of post-freeze architecture exploration.
- Branch nuance:
  - there is no separate current local `layernorm` branch ref
  - the surviving code evidence appears bundled into local branch tip `82d6032` on `orthogonal-actor-line`
