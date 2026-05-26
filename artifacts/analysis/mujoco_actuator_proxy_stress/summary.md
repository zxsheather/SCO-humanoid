# MuJoCo Actuator-Proxy Stress Summary

Protocol: `isaac_mainline`, `20 episodes x 20s`, `joint_reset_noise = 0.1`, command `(vx=0.4, vy=0.0, dyaw=0.0)`, with a first-order action low-pass proxy before PD target generation.

Actuator proxy:

- mode: `action_lowpass`
- low-pass time constant: `0.05` seconds
- nominal control timestep: `0.01` seconds
- low-pass alpha: `0.16666666666666666`

## Aggregate Metrics

| Method | Nominal fall | Proxy fall | Nominal jnt acc | Proxy jnt acc | Jnt acc factor | Nominal raw jitter | Proxy raw jitter | Proxy applied jitter | Proxy lag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scppo38 | 0.017 | 0.067 | 125.541 | 104.028 | 0.829 | 0.231 | 0.256 | 0.170 | 0.851 |
| heuristic | 0.000 | 0.250 | 120.734 | 126.616 | 1.049 | 0.245 | 0.295 | 0.194 | 0.972 |
| layernorm_ep3 | 0.000 | 0.333 | 602.578 | 138.986 | 0.231 | 3.328 | 1.083 | 0.297 | 1.484 |

## Task Degradation

| Method | Nominal vel err | Proxy vel err | Vel err delta | Nominal steps | Proxy steps | Steps delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scppo38 | 0.491 | 0.483 | -0.008 | 1984.783 | 1934.167 | -50.617 |
| heuristic | 0.419 | 0.578 | 0.160 | 2000.000 | 1767.567 | -232.433 |
| layernorm_ep3 | 0.447 | 0.622 | 0.176 | 2000.000 | 1571.983 | -428.017 |

## Source Artifacts

- `scppo38` seed `11` proxy: `artifacts/methods/sc_ppo_pid_probe/scppo38_mujoco_actuator_lowpass_tau005_seed11/metrics_mujoco_actuator_lowpass_tau005_20ep_20s_noise01.json`
- `scppo38` seed `11` nominal: `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `scppo38` seed `17` proxy: `artifacts/methods/sc_ppo_pid_probe/scppo38_mujoco_actuator_lowpass_tau005_seed17/metrics_mujoco_actuator_lowpass_tau005_20ep_20s_noise01.json`
- `scppo38` seed `17` nominal: `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `scppo38` seed `23` proxy: `artifacts/methods/sc_ppo_pid_probe/scppo38_mujoco_actuator_lowpass_tau005_seed23/metrics_mujoco_actuator_lowpass_tau005_20ep_20s_noise01.json`
- `scppo38` seed `23` nominal: `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `heuristic` seed `11` proxy: `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_mujoco_actuator_lowpass_tau005_seed11/metrics_mujoco_actuator_lowpass_tau005_20ep_20s_noise01.json`
- `heuristic` seed `11` nominal: `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `heuristic` seed `17` proxy: `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_mujoco_actuator_lowpass_tau005_seed17/metrics_mujoco_actuator_lowpass_tau005_20ep_20s_noise01.json`
- `heuristic` seed `17` nominal: `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `heuristic` seed `23` proxy: `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_mujoco_actuator_lowpass_tau005_seed23/metrics_mujoco_actuator_lowpass_tau005_20ep_20s_noise01.json`
- `heuristic` seed `23` nominal: `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `layernorm_ep3` seed `11` proxy: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_ep3_mujoco_actuator_lowpass_tau005_seed11/metrics_mujoco_actuator_lowpass_tau005_20ep_20s_noise01.json`
- `layernorm_ep3` seed `11` nominal: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `layernorm_ep3` seed `17` proxy: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_ep3_mujoco_actuator_lowpass_tau005_seed17/metrics_mujoco_actuator_lowpass_tau005_20ep_20s_noise01.json`
- `layernorm_ep3` seed `17` nominal: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed17/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `layernorm_ep3` seed `23` proxy: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_ep3_mujoco_actuator_lowpass_tau005_seed23/metrics_mujoco_actuator_lowpass_tau005_20ep_20s_noise01.json`
- `layernorm_ep3` seed `23` nominal: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed23/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
