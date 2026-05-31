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

## Design Choices

- Reuse the existing XBot-L humanoid environment logic for the first vertical
  slice.
- Override only the XBot-specific parts needed for H1: reference-state joint
  indices, observation-noise layout, and default-joint-position reward indices.
- Keep algorithm classes unchanged; any H1-specific tuning should remain in
  config overrides.

If the smoke slice works, the next step is a small H1 comparison with the same
method families used in the main paper. If more robots are added after H1, the
duplicated humanoid logic should then be extracted into a shared `HumanoidEnv`
base class.
