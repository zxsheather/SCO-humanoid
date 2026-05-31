# Unitree H1 Feasibility Slice

This branch adds a narrow Unitree H1 task as the first multi-robot probe. The
goal is to verify that SCO-humanoid can support a second humanoid morphology
without changing the PPO, SC-PPO, or LCP-style algorithm implementations.

## Asset Boundary

The H1 task targets the leg-only Unitree H1 asset layout from the official
`unitreerobotics/unitree_rl_gym` repository. In that asset, the torso and arm
joints are fixed and the policy controls ten leg joints. This is intentionally
smaller than the full Unitree ROS H1 model and is better suited for the first
smoke test because it matches the legged-locomotion control path.

Robot assets are not committed to this repository. Install them locally with:

```bash
scripts/setup/fetch_unitree_h1_assets.sh
```

The script copies the asset source license into the ignored local asset
directory. The GitHub repository metadata reports `unitree_rl_gym` as
BSD-3-Clause licensed.

## Task Entry Point

- Task name: `h1_ppo`
- Smoke config: `configs/methods/h1_vanilla_ppo_smoke.json`
- Asset path expected by Humanoid-Gym:
  `.external/humanoid-gym/resources/robots/h1/urdf/h1.urdf`

The smoke config is intentionally short and should not be used as claim-grade
training evidence. It only verifies task registration, observation/action
dimensions, rollout compatibility, and metric-pipeline portability.

## Baseline Validity Sweep

Issue #102 trained one H1 vanilla PPO run beyond smoke length to test whether
the task can reach non-collapsed locomotion before running any H1 method
comparison.

- Training run:
  `h1_baseline_validity_seed5_iter200_env512`
- Budget: 512 environments, 200 PPO iterations, seed 5, Isaac Gym GPU pipeline.
- Run directory:
  `.external/humanoid-gym/logs/ecolab_h1_ppo_smoke/May31_13-17-15_h1_baseline_validity_seed5_iter200_env512`
- Manifest:
  `artifacts/methods/h1_vanilla_ppo_smoke/h1_baseline_validity_seed5_iter200_env512/manifest.json`

Training did not crash and improved from early short episodes to mean episode
lengths around 1300 steps near iteration 200. Two checkpoints were evaluated
with the standard metric schema, 16 evaluation environments, and 20 completed
episodes per checkpoint:

| Checkpoint | Fall rate | Vel. err | Return | Jnt acc | Jitter | Sens. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.000 | 1.291 | 8.068 | 61.570 | 0.152 | 4.441 |
| 200 | 0.300 | 1.029 | 61.047 | 19.565 | 0.164 | 9.787 |

The result is a positive viability signal but not claim-grade H1 evidence.
Checkpoint 200 shows the task can produce non-collapsed locomotion under a
modest budget, but a 30% fall rate is too high for a fair smooth-control method
comparison. The next H1 step should adjust or extend the baseline task before
training H1 LCP-style, heuristic, or SC-PPO variants.

Issue #108 tested the simplest stabilization hypothesis: rerun the same H1
vanilla PPO setup for a longer 400-iteration budget and evaluate later
checkpoints. This did not pass the method-probe gate:

| Checkpoint | Fall rate | Vel. err | Return | Jnt acc | Jitter | Sens. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.300 | 1.011 | 62.169 | 18.610 | 0.164 | 9.829 |
| 300 | 0.350 | 0.995 | 66.513 | 23.093 | 0.195 | 11.460 |
| 400 | 0.350 | 0.961 | 65.641 | 22.061 | 0.206 | 13.428 |

Longer training improved velocity tracking and return relative to the #102
checkpoint-200 probe, but it did not improve fall rate and it increased policy
local sensitivity, action jitter, and joint acceleration. H1 should therefore
not proceed to LCP-style, heuristic, or SC-PPO method probes under this
unchanged task/config. The next H1 work should tune the task or baseline
configuration and re-run this gate before unblocking method comparisons.

## Design Choices

- Reuse the existing XBot-L humanoid environment logic for the first vertical
  slice.
- Override only the XBot-specific parts needed for H1: reference-state joint
  indices, observation-noise layout, and default-joint-position reward indices.
- Keep algorithm classes unchanged; any H1-specific tuning should remain in
  config overrides.

After the #102 baseline validity sweep, H1 should remain gated on baseline
stabilization before any method comparison. If more robots are added after H1,
the duplicated humanoid logic should then be extracted into a shared
`HumanoidEnv` base class.
