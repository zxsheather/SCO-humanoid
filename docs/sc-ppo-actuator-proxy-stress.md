# MuJoCo Actuator-Proxy Stress Test

This note records issue #54: a bounded simulator-side actuator-bandwidth proxy
stress test for the current paper package. It is a sim-to-real-motivated
diagnostic only. It is not real-hardware validation.

Current full-paper status: this is a historical three-row proxy test. The
paper-facing actuator evidence is now the five-seed #97 actuator-bandwidth
sweep over LCP-style soft penalty, SC-PPO 3.8 PID, and the revised heuristic:
`artifacts/analysis/actuator_latency_robustness/summary.md`. Do not use this
#54 note to claim that SC-PPO is the strongest actuator-robust row in the
current full-paper comparison.

## Protocol

The test reuses selected checkpoints and does not retrain any method. It adds a
first-order low-pass filter between the policy action and the PD target:

```text
applied_action_t = alpha * policy_action_t + (1 - alpha) * applied_action_{t-1}
```

Parameters:

- MuJoCo protocol: `isaac_mainline`
- episodes: `20 x 20s`
- reset: `joint_reset_noise = 0.1`, `base_xy_noise = 0.0`
- command: `(vx=0.4, vy=0.0, dyaw=0.0)`
- low-pass time constant: `0.05s`
- control timestep: `0.01s`
- low-pass alpha: `0.1667`

Rows:

| Method | Seeds | Checkpoints |
| --- | --- | --- |
| `scppo38` | 11, 17, 23 | 300, 300, 400 |
| `heuristic` | 11, 17, 23 | 350, 300, 350 |
| `layernorm_ep3` | 11, 17, 23 | 400, 400, 400 |

Execution:

```bash
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/baseline/run_mujoco_actuator_proxy_stress.py
/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/analyze_mujoco_actuator_proxy_stress.py
```

The evaluator records both raw policy-action jitter and filtered applied-action
jitter:

- `action_jitter_l2_mean`: raw policy command changes
- `applied_action_jitter_l2_mean`: low-pass filtered command changes applied to
  the PD target
- `action_lag_l2_mean`: action-space gap introduced by the low-pass proxy

## Compact Results

Tracked summaries:

- `artifacts/analysis/mujoco_actuator_proxy_stress/summary.json`
- `artifacts/analysis/mujoco_actuator_proxy_stress/summary.md`

Aggregate metrics:

| Method | Nominal fall | Proxy fall | Nominal jnt acc | Proxy jnt acc | Jnt acc factor | Nominal raw jitter | Proxy raw jitter | Proxy applied jitter | Proxy lag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scppo38` | 0.017 | 0.067 | 125.541 | 104.028 | 0.829 | 0.231 | 0.256 | 0.170 | 0.851 |
| `heuristic` | 0.000 | 0.250 | 120.734 | 126.616 | 1.049 | 0.245 | 0.295 | 0.194 | 0.972 |
| `layernorm_ep3` | 0.000 | 0.333 | 602.578 | 138.986 | 0.231 | 3.328 | 1.083 | 0.297 | 1.484 |

Task degradation:

| Method | Nominal vel err | Proxy vel err | Vel err delta | Nominal steps | Proxy steps | Steps delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `scppo38` | 0.491 | 0.483 | -0.008 | 1984.783 | 1934.167 | -50.617 |
| `heuristic` | 0.419 | 0.578 | 0.160 | 2000.000 | 1767.567 | -232.433 |
| `layernorm_ep3` | 0.447 | 0.622 | 0.176 | 2000.000 | 1571.983 | -428.017 |

## Interpretation

The actuator proxy supports the paper's sim-to-real-motivated smoothness
argument, with an important caveat: the low-pass filter can mechanically reduce
joint acceleration, so lower joint acceleration alone is not enough to claim
robustness under this proxy.

The task-stability metrics are the decisive part of this stress test. Under the
same low-pass actuator path, `SC-PPO 3.8` has the lowest proxy fall rate
(`0.067`), smallest episode-length loss (`-50.6` steps), and no velocity-tracking
penalty. The revised heuristic and LayerNorm rows degrade more strongly:
heuristic fall rises to `0.250`, and LayerNorm fall rises to `0.333`.

The action-level metrics also favor `SC-PPO 3.8` under the proxy. It has the
lowest raw policy jitter (`0.256`), lowest filtered applied-action jitter
(`0.170`), and lowest action lag (`0.851`). LayerNorm still carries much larger
raw policy jitter (`1.083`) even after the low-pass proxy attenuates the command
stream.

The bounded conclusion is:

```text
When the idealized action-to-PD-target path is perturbed by a 50 ms first-order
low-pass proxy, SC-PPO 3.8 is the most stable of the three replayed rows.
```

This strengthens the paper's sim-to-real-motivated discussion because the
Jacobian-constrained row is less sensitive to a non-ideal actuator command path.
It does not establish hardware transfer, actuator system identification, or a
general latency-robust controller.

## Raw Artifacts

Raw per-seed proxy metrics are local runtime artifacts under ignored
`artifacts/methods/` directories. Exact per-seed paths are listed in the tracked
summary markdown.

The proxy metrics filename is:

- `metrics_mujoco_actuator_lowpass_tau005_20ep_20s_noise01.json`

## Limitations

- This is a single actuator proxy, not a calibrated actuator model.
- The low-pass time constant is a bounded stress parameter, not a measured H1
  actuator bandwidth.
- The proxy is applied to the policy action before PD target generation; it does
  not model motor current limits, backlash, encoder latency, or communication
  jitter.
- Results remain MuJoCo sim-to-sim evidence. They must not be described as
  real-robot validation.
