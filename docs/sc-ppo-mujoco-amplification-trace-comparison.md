# MuJoCo Amplification Trace Comparison

This note records the issue #49 follow-up to the cross-engine degradation
analysis: matched MuJoCo per-timestep traces for `SC-PPO 3.8`, `LayerNorm
actor epochs=3`, and `Action Scaling`.

## Scope

The goal is not to add a new training result. The goal is to test whether the
existing policy-level amplification interpretation survives a trace-level check:

```text
high policy sensitivity -> jittery MuJoCo command stream -> elevated joint acceleration
```

The comparison uses selected representative checkpoints from the already frozen
experiment families:

| Method | Seeds | Checkpoints |
| --- | --- | --- |
| `scppo38` | 11, 17, 23 | 300, 300, 400 |
| `layernorm_ep3` | 11, 17, 23 | 400, 400, 400 |
| `action_scaling` | 11, 17, 23 | 400, 400, 400 |

## Protocol

All replays use the matched MuJoCo `isaac_mainline` protocol from the
cross-engine degradation table:

- `20` episodes x `20s`
- `joint_reset_noise = 0.1`
- `base_xy_noise = 0.0`
- command `(vx=0.4, vy=0.0, dyaw=0.0)`
- MuJoCo reset seed `12345`
- trace capture: first `3` episodes per training seed, capped at `1024`
  control steps per episode

Execution script:

```bash
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/baseline/run_mujoco_amplification_traces.py
```

Analysis script:

```bash
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/analyze_mujoco_amplification_traces.py
```

The MuJoCo replay process exits with `-11` during cleanup in this environment
after writing valid JSON. The runner treats this as a soft success only when
both the metrics JSON and trace JSON parse successfully.

## Compact Results

Tracked compact summaries:

- `artifacts/analysis/mujoco_amplification_trace_comparison/summary.json`
- `artifacts/analysis/mujoco_amplification_trace_comparison/summary.md`

Aggregate trace metrics:

| Method | Trace steps | Fall rate | MuJoCo jnt acc | MuJoCo jitter | Trace jitter p95 | Trace jnt acc p95 | Contact force p95 | Tau p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scppo38` | 9216 | 0.017 | 125.541 | 0.231 | 0.529 | 339.112 | 461.511 | 137.686 |
| `layernorm_ep3` | 9216 | 0.000 | 602.578 | 3.328 | 6.251 | 1319.939 | 617.249 | 220.986 |
| `action_scaling` | 2248 | 1.000 | 1835.590 | 8.305 | 16.386 | 3552.513 | 1418.324 | 381.449 |

Spike coupling:

| Method | Joint-spike threshold | Spike contact fraction | Spike mean jitter | Spike mean contact force | Spike mean tau | corr(jitter,jnt_acc) | corr(contact,jnt_acc) | corr(tau,jnt_acc) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scppo38` | 339.112 | 0.952 | 0.393 | 424.704 | 109.496 | 0.419 | 0.401 | 0.221 |
| `layernorm_ep3` | 1319.939 | 0.844 | 5.975 | 308.629 | 148.097 | 0.708 | 0.063 | 0.329 |
| `action_scaling` | 3552.513 | 0.239 | 15.599 | 486.538 | 317.721 | 0.659 | -0.035 | 0.372 |

## Interpretation

The trace evidence supports the bounded `policy -> physics` amplification
reading, with one refinement: the strongest signal is policy-output and control
stream amplification, not a contact-only explanation.

High-degradation methods show much larger action-jitter and joint-acceleration
tails. `Action Scaling` and `LayerNorm` have high jitter/joint-acceleration
correlation (`0.659` and `0.708`), while their contact-force correlation is weak
(`-0.035` and `0.063`). Their largest joint-acceleration spikes can occur with
zero recorded contacts, so the data does not support a narrow claim that contact
impacts alone explain the degradation.

`SC-PPO 3.8` remains the dynamically smooth case in MuJoCo. Its jitter tail is
an order of magnitude smaller than `LayerNorm` and `Action Scaling`, and its
joint-acceleration spikes are much smaller. Its spike set is contact-heavy
(`0.952` contact fraction), which is consistent with normal locomotion contact
events rather than unstable policy command amplification.

The defensible conclusion is therefore:

```text
lower policy sensitivity -> lower MuJoCo action jitter -> lower joint acceleration amplification
```

This strengthens the paper's mechanism discussion but still is not a formal
time-series causal proof. It also does not change the external-validation
boundary: there is no real-robot result, and MuJoCo replay remains sim-to-sim
evidence only.

## Raw Trace Artifacts

The raw replay outputs are local runtime artifacts under ignored
`artifacts/methods/` directories. Exact per-seed paths are listed in the tracked
summary markdown. The path pattern is:

- `artifacts/methods/sc_ppo_pid_probe/scppo38_mujoco_amp_trace_seed{11,17,23}/mujoco_amplification_trace_3ep_20s_noise01.json`
- `artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_ep3_mujoco_amp_trace_seed{11,17,23}/mujoco_amplification_trace_3ep_20s_noise01.json`
- `artifacts/methods/action_scaling_probe/action_scaling_mujoco_amp_trace_seed{11,17,23}/mujoco_amplification_trace_3ep_20s_noise01.json`

The corresponding metrics files use:

- `metrics_mujoco_amplification_trace_20ep_20s_noise01.json`

## Limitations

- This trace slice compares three methods. The heuristic baseline and Output
  Scaling remain represented by aggregate MuJoCo degradation artifacts, not by
  this trace comparison.
- Only the first three episodes per seed are traced; aggregate fall and smoothness
  metrics still come from the full 20-episode replay.
- Correlation is computed within the replayed traces. It identifies coupling, not
  intervention-level causality.
- Contact force comes from MuJoCo contact instrumentation and should be treated as
  a simulator-side diagnostic, not a hardware force measurement.
