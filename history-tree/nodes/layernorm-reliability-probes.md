# LayerNorm Reliability Probes

- **Date**: 2026-05-25
- **Type**: experiment
- **Outcome**: mixed
- **Tags**: reliability, layernorm, ablation

## Timeline and Background

After the bounded `actor_output_gain = 0.75` LayerNorm repair cleared the entry gate, the remaining
question was final-checkpoint reliability. The line still depended on checkpoint selection, so the repo
tested a small set of bounded reliability levers rather than opening a broad architecture sweep.

## Technical Details

- Four levers were tested on the canonical `11 / 17 / 23`, `512 envs x 400 iterations` entry:
  - fixed learning-rate schedule: collapsed
  - lower `policy.init_noise_std`: collapsed
  - lower `desired_kl = 0.005`: collapsed
  - `num_learning_epochs = 3`: passed
- The epochs=3 candidate achieved `selected = final = 400` on all three seeds.
- Its Isaac aggregate was task-strong but dynamically rougher than `SC-PPO 3.8`.

## Decision Process

- The failed levers closed as local negatives rather than warm retries.
- The epochs=3 lever earned MuJoCo replay budget because it passed the full Isaac internal challenge.
- Passing that challenge was not treated as proof of smoothness replacement.

## Results and Impact

- This node explains why the LayerNorm line advanced to cross-engine replay.
- It also separates task reliability from dynamic smoothness robustness.
- Related follow-up: [layernorm-mujoco-replay](../nodes/layernorm-mujoco-replay.md).
