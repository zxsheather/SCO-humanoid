# LayerNorm Actor Line

- **Date**: 2026-05-25
- **Type**: experiment
- **Outcome**: failure
- **Tags**: architecture-line, layernorm, reliability-challenge, mujoco-replay, closed-negative

## Timeline and Background

Issue [`#35`](https://github.com/zxsheather/SCO-humanoid/issues/35) opened immediately after the
orthogonal actor line closed. The rationale was explicit: scaling families failed, orthogonal actor
failed even after gain isolation, so the next distinct candidate should target hidden activation scale
directly.

## Technical Details

- Issue body defines the first canonical entry:
  - actor-side LayerNorm support
  - removal of the active Jacobian/double-backward path
  - canonical `11 / 17 / 23`, `512 envs x 400 iterations` sweep
- Initial canonical `actor_output_gain = 1.00` read:
  - `fall_rate_mean = 0.8667`
  - `episode_return_mean_mean = 26.0529`
  - `velocity_tracking_error_mean_mean = 1.0676`
  - `joint_acceleration_l2_mean_mean = 185.1820`
  - `action_jitter_l2_mean_mean = 0.3338`
- Bounded `actor_output_gain = 0.75` repair later cleared the entry gate but still had checkpoint
  selection dependence.
- The `num_learning_epochs = 3` reliability lever became the first architecture-side candidate to pass
  the full Isaac internal challenge with `selected = final = 400` on all three seeds.
- The completed MuJoCo replay then showed large dynamic-smoothness degradation:
  - `joint_acceleration_l2_mean_mean = 602.5776`
  - `action_jitter_l2_mean_mean = 3.3285`
  - Isaac-to-MuJoCo joint-acceleration inflation ~= `3.5x`

## Decision Process

- The repo did promote the repaired LayerNorm recipe far enough to spend MuJoCo replay budget.
- The line still closed negatively as a smoothness replacement because passing the Isaac task challenge
  did not preserve dynamic smoothness across engines.
- The durable interpretation is task-vs-smoothness trade-off, not a new live architecture frontier.

## Results and Impact

- LayerNorm actor is now closed under the current same-question boundary.
- It is the first architecture-side line to show a `3/3 selected=final=400` Isaac result, but not a
  clean replacement for `SC-PPO 3.8`.
- Related closure nodes:
  - [layernorm-reliability-probes](../nodes/layernorm-reliability-probes.md)
  - [layernorm-mujoco-replay](../nodes/layernorm-mujoco-replay.md)
  - [layernorm-line-closure](../nodes/layernorm-line-closure.md)
