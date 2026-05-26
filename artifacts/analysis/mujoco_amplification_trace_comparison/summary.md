# MuJoCo Amplification Trace Comparison

Matched replay protocol: `isaac_mainline`, `20 episodes x 20s`, `joint_reset_noise = 0.1`, command `(vx=0.4, vy=0.0, dyaw=0.0)`, capturing the first 3 episodes per seed with up to 1024 control steps each.

## Aggregate Trace Metrics

| Method | Trace steps | Fall rate | MuJoCo jnt acc | MuJoCo jitter | Trace jitter p95 | Trace jnt acc p95 | Contact force p95 | Tau p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scppo38 | 9216 | 0.017 | 125.541 | 0.231 | 0.529 | 339.112 | 461.511 | 137.686 |
| layernorm_ep3 | 9216 | 0.000 | 602.578 | 3.328 | 6.251 | 1319.939 | 617.249 | 220.986 |
| action_scaling | 2248 | 1.000 | 1835.590 | 8.305 | 16.386 | 3552.513 | 1418.324 | 381.449 |

## Spike Coupling

| Method | Joint-spike threshold | Spike contact fraction | Spike mean jitter | Spike mean contact force | Spike mean tau | corr(jitter,jnt_acc) | corr(contact,jnt_acc) | corr(tau,jnt_acc) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scppo38 | 339.112 | 0.952 | 0.393 | 424.704 | 109.496 | 0.419 | 0.401 | 0.221 |
| layernorm_ep3 | 1319.939 | 0.844 | 5.975 | 308.629 | 148.097 | 0.708 | 0.063 | 0.329 |
| action_scaling | 3552.513 | 0.239 | 15.599 | 486.538 | 317.721 | 0.659 | -0.035 | 0.372 |

## Top Joint-Acceleration Spikes

| Method | Seed | Episode | Step | Time | Fell | Joint acc | Action jitter | Contact force | Contact count | Tau |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| scppo38 | 11 | 0 | 977 | 9.77 | true | 1480.764 | 0.269 | 1547.181 | 2 | 112.890 |
| scppo38 | 11 | 2 | 723 | 7.23 | false | 1239.839 | 0.332 | 627.662 | 1 | 144.048 |
| scppo38 | 11 | 0 | 978 | 9.78 | true | 1235.326 | 0.345 | 572.322 | 1 | 130.158 |
| layernorm_ep3 | 11 | 1 | 9 | 0.09 | false | 3114.836 | 5.877 | 0.000 | 0 | 92.946 |
| layernorm_ep3 | 11 | 1 | 8 | 0.08 | false | 2572.759 | 5.053 | 0.000 | 0 | 120.095 |
| layernorm_ep3 | 11 | 0 | 9 | 0.09 | false | 2385.268 | 5.103 | 0.000 | 0 | 100.480 |
| action_scaling | 11 | 1 | 188 | 1.88 | true | 5972.111 | 16.904 | 0.000 | 0 | 267.426 |
| action_scaling | 11 | 0 | 112 | 1.12 | true | 5795.272 | 20.681 | 0.000 | 0 | 323.924 |
| action_scaling | 17 | 1 | 62 | 0.62 | true | 5698.753 | 9.990 | 0.000 | 0 | 277.038 |

## Source Artifacts

- `scppo38` seed `11` metrics: `artifacts/methods/sc_ppo_pid_probe/scppo38_mujoco_amp_trace_seed11/metrics_mujoco_amplification_trace_20ep_20s_noise01.json`
- `scppo38` seed `11` trace: `artifacts/methods/sc_ppo_pid_probe/scppo38_mujoco_amp_trace_seed11/mujoco_amplification_trace_3ep_20s_noise01.json`
- `scppo38` seed `17` metrics: `artifacts/methods/sc_ppo_pid_probe/scppo38_mujoco_amp_trace_seed17/metrics_mujoco_amplification_trace_20ep_20s_noise01.json`
- `scppo38` seed `17` trace: `artifacts/methods/sc_ppo_pid_probe/scppo38_mujoco_amp_trace_seed17/mujoco_amplification_trace_3ep_20s_noise01.json`
- `scppo38` seed `23` metrics: `artifacts/methods/sc_ppo_pid_probe/scppo38_mujoco_amp_trace_seed23/metrics_mujoco_amplification_trace_20ep_20s_noise01.json`
- `scppo38` seed `23` trace: `artifacts/methods/sc_ppo_pid_probe/scppo38_mujoco_amp_trace_seed23/mujoco_amplification_trace_3ep_20s_noise01.json`
- `layernorm_ep3` seed `11` metrics: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_ep3_mujoco_amp_trace_seed11/metrics_mujoco_amplification_trace_20ep_20s_noise01.json`
- `layernorm_ep3` seed `11` trace: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_ep3_mujoco_amp_trace_seed11/mujoco_amplification_trace_3ep_20s_noise01.json`
- `layernorm_ep3` seed `17` metrics: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_ep3_mujoco_amp_trace_seed17/metrics_mujoco_amplification_trace_20ep_20s_noise01.json`
- `layernorm_ep3` seed `17` trace: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_ep3_mujoco_amp_trace_seed17/mujoco_amplification_trace_3ep_20s_noise01.json`
- `layernorm_ep3` seed `23` metrics: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_ep3_mujoco_amp_trace_seed23/metrics_mujoco_amplification_trace_20ep_20s_noise01.json`
- `layernorm_ep3` seed `23` trace: `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_ep3_mujoco_amp_trace_seed23/mujoco_amplification_trace_3ep_20s_noise01.json`
- `action_scaling` seed `11` metrics: `artifacts/methods/action_scaling_probe/action_scaling_mujoco_amp_trace_seed11/metrics_mujoco_amplification_trace_20ep_20s_noise01.json`
- `action_scaling` seed `11` trace: `artifacts/methods/action_scaling_probe/action_scaling_mujoco_amp_trace_seed11/mujoco_amplification_trace_3ep_20s_noise01.json`
- `action_scaling` seed `17` metrics: `artifacts/methods/action_scaling_probe/action_scaling_mujoco_amp_trace_seed17/metrics_mujoco_amplification_trace_20ep_20s_noise01.json`
- `action_scaling` seed `17` trace: `artifacts/methods/action_scaling_probe/action_scaling_mujoco_amp_trace_seed17/mujoco_amplification_trace_3ep_20s_noise01.json`
- `action_scaling` seed `23` metrics: `artifacts/methods/action_scaling_probe/action_scaling_mujoco_amp_trace_seed23/metrics_mujoco_amplification_trace_20ep_20s_noise01.json`
- `action_scaling` seed `23` trace: `artifacts/methods/action_scaling_probe/action_scaling_mujoco_amp_trace_seed23/mujoco_amplification_trace_3ep_20s_noise01.json`
