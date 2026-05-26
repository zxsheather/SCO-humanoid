# LayerNorm MuJoCo Cross-Engine Replay

- **Date**: 2026-05-25
- **Type**: experiment
- **Outcome**: failure
- **Tags**: mujoco, cross-engine, layernorm, smoothness-collapse

## Timeline and Background

The LayerNorm epochs=3 candidate was replayed in `MuJoCo isaac_mainline` after it passed the Isaac
internal challenge. This was the decisive test of whether actor-side normalization could replace the
Jacobian/double-backward path as a cross-engine smoothness mechanism.

## Technical Details

- Replay target: LayerNorm actor with `actor_output_gain = 0.75` and `num_learning_epochs = 3`.
- MuJoCo aggregate:
  - `fall_rate_mean = 0.0000`
  - `velocity_tracking_error_mean_mean = 0.4467`
  - `joint_acceleration_l2_mean_mean = 602.5776`
  - `action_jitter_l2_mean_mean = 3.3285`
  - `episode_return_mean_mean = -543.5494`
- Compared with `SC-PPO 3.8` in MuJoCo, joint acceleration is about `5x` worse and action jitter is
  about `14x` worse.
- Compared with its own Isaac result, the joint-acceleration degradation factor is about `3.5x`.

## Decision Process

- The replay preserved task survival but failed the dynamic-smoothness goal.
- The repo therefore rejected the reading that LayerNorm is a clean smoothness replacement.

## Results and Impact

- This replay closed the architecture-side LayerNorm line as a negative replacement result.
- It directly supports the paper-direction claim that the Jacobian constraint path provides
  cross-engine dynamic-smoothness robustness that LayerNorm does not replicate.
- Related follow-up: [layernorm-line-closure](../nodes/layernorm-line-closure.md).
