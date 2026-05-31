# Second-Setting Scout

Issues: `#91`, prerequisite decision for `#92`.

## Goal

Find one lightweight setting that tests whether the mechanism-comparison evidence is specific to
the current rough-terrain protocol, without turning the project into a broad benchmark.

Selection criteria:

- same H1 morphology
- existing selected checkpoints or low retraining cost
- compatible metrics: fall rate, velocity error, joint acceleration, action jitter, return, and
  policy local sensitivity where available
- task validity before metric ranking
- clear claim boundary for a full-paper generality check

## Candidate Inventory

| Candidate | Existing support | Feasibility | Decision |
| --- | --- | --- | --- |
| random-stairs Isaac selected-checkpoint stress | `scripts/baseline/run_random_stairs_stress_test.py`, `configs/sweeps/random_stairs_selected_checkpoint_stress.json` | Complete for seeds `11/17/23`; all methods collapse with `fall_rate = 1.0` | No-go as a method-comparison validation; useful negative transfer result only |
| MuJoCo `hfield_stress` | `evaluate_mujoco_sim2sim.py --terrain-mode hfield_stress` | Too severe; existing docs treat it as a terrain stress probe, not report-grade | No-go for current paper-facing validation |
| MuJoCo `hfield_moderate` | `evaluate_mujoco_sim2sim.py --terrain-mode hfield_moderate` | Same morphology, no retraining, metric-compatible, moderated terrain difficulty | Selected |
| command-distribution sweep | existing command arguments in MuJoCo evaluator | Feasible but changes task distribution and lacks prior calibration | Defer until terrain check is closed |
| actuator low-pass proxy | `scripts/baseline/run_mujoco_actuator_proxy_stress.py` | Already completed as actuator-path stress, not second terrain/task | Do not reuse for `#92` |
| observation-noise sweep | `scripts/baseline/run_observation_noise_robustness_sweep.py` | Covered by `#89`; useful mechanism robustness, not second terrain/task | Keep separate from `#92` |

## Random-Stairs Probe

The random-stairs protocol has already been run as a selected-checkpoint pressure test:

- note: `docs/random-stairs-selected-checkpoint-stress.md`
- artifact: `artifacts/analysis/random_stairs_selected_checkpoint_stress/comparison_summary.json`
- seeds: `11/17/23`
- result: Vanilla PPO, revised heuristic, and SC-PPO all report `fall_rate = 1.0`

This is a valid negative transfer finding, but it cannot support a task-valid mechanism ranking.
The next useful random-stairs work would be protocol repair, such as reduced stair height or mixed
rough/stairs proportions, before spending full method-comparison budget.

## Selected Candidate

Use MuJoCo `hfield_moderate` as the narrow second setting.

Rationale:

- it reuses the same H1 selected checkpoints for LCP-style soft penalty, SC-PPO 3.8 PID, and the
  revised heuristic
- it uses the same MuJoCo evaluator and smooth-control metric schema as the matched replay
- it is less degenerate than `hfield_stress` and random stairs
- it is still clearly outside the flat `isaac_mainline` replay
- it should be described as a repair-stage terrain generality check, not a benchmark result

Planned run:

```bash
python scripts/baseline/run_hfield_moderate_second_setting.py --cuda-visible-devices 1
```

Bounded smoke command:

```bash
python scripts/baseline/run_hfield_moderate_second_setting.py \
  --methods lcp \
  --seeds 11 \
  --episodes 1 \
  --sim-duration 2 \
  --cuda-visible-devices 1
```

Expected full runtime is roughly one MuJoCo replay batch over `15` jobs:

- `3` methods
- `5` seeds
- `20 episodes x 20 seconds`

## Go/No-Go Decision

Go for `#92` with MuJoCo `hfield_moderate`.

The paper-facing interpretation should be decided after the full run:

- if all methods collapse, report it as a negative second-setting result and do not rank methods
- if only some methods remain task-valid, compare smooth-control metrics only within task-valid rows
- if all methods remain task-valid, use it as a compact terrain generality check for the mechanism
  argument

## Completed hfield_moderate Result

The selected second-setting run is complete:

- runner: `scripts/baseline/run_hfield_moderate_second_setting.py`
- summary: `artifacts/analysis/hfield_moderate_second_setting/summary.json`
- markdown: `artifacts/analysis/hfield_moderate_second_setting/summary.md`
- rows: `15/15` completed

Aggregate over seeds `11/17/23/29/31`:

| Method | Fall | Vel. err | Jnt acc | Jitter | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| LCP-style soft penalty | `0.350` | `0.832` | `321.113` | `0.277` | `-566.367` |
| SC-PPO 3.8 PID | `0.500` | `0.981` | `386.647` | `0.380` | `-513.230` |
| Revised heuristic | `0.400` | `1.003` | `322.360` | `0.358` | `-666.633` |

Reading:

- the LCP-style row is best on fall rate, velocity error, joint acceleration, and action jitter
- SC-PPO has the highest return but also the highest fall rate among the three rows
- the result supports the policy-output and moderated-terrain robustness part of the mechanism
  conclusion
- because all fall rates remain materially higher than in the primary matched replay, this should
  remain a repair-stage terrain generality check rather than a promoted multi-terrain benchmark
