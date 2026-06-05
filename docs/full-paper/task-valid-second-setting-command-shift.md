# Task-Valid Second-Setting Command Shift (#126)

Status: `complete`.

## Goal

Add one bounded same-morphology second-setting slice that is actually
task-valid for the current `LCP-style` main row, so the paper has one stronger
supplementary robustness read than the existing `hfield_moderate` high-fall
terrain replay and the mixed-terrain retrain no-go.

## Rejected Candidate

A plain MuJoCo `plane` replay was checked first and rejected as the selected
issue slice.

Reason:

- the resulting aggregate exactly reproduced the current matched MuJoCo replay
  values;
- plain `plane` was therefore not a distinct second setting for the current
  replay path and could not close the issue.

## Selected Slice

Use a no-retraining MuJoCo command shift:

- terrain path: `plane`
- command change: `command_vx = 0.6` instead of the main replay `0.4`
- same selected checkpoints as the five-seed main rows
- same five seeds `11/17/23/29/31`
- same metric schema as the matched MuJoCo replay

Why this was selected:

- it is a genuine second setting, unlike plain `plane`;
- it keeps the morphology, selected checkpoints, and replay bridge fixed;
- it changes only one protocol variable; and
- it has a realistic chance to remain task-valid across all three methods.

## Protocol

- sweep:
  `configs/sweeps/plane_command_vx06_second_setting.json`
- runner:
  `scripts/baseline/run_hfield_moderate_second_setting.py`
- summary:
  `artifacts/analysis/plane_command_vx06_second_setting/summary.json`
- markdown:
  `artifacts/analysis/plane_command_vx06_second_setting/summary.md`

## Result

Aggregate over seeds `11/17/23/29/31`:

| Method | Fall | Vel. err | Jnt acc | Jitter | Return | Sens. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Revised heuristic | `0.000` | `0.538` | `118.274` | `0.245` | `-512.075` | `7.331` |
| LCP-style soft penalty | `0.000` | `0.562` | `119.555` | `0.198` | `-784.939` | `1.890` |
| SC-PPO 3.8 PID | `0.040` | `0.598` | `176.182` | `0.352` | `-657.114` | `3.630` |

Read:

- the slice is task-valid for all three rows, with `LCP-style` and the revised
  heuristic both at `0.000` fall and SC-PPO only slightly worse at `0.040`;
- `LCP-style` remains best on action jitter and measured sensitivity;
- the revised heuristic remains best on velocity error, joint acceleration, and
  return; and
- SC-PPO remains the weakest aggregate row under this command shift.

## Interpretation

This is the cleanest task-valid second-setting result currently available in the
main morphology.

The correct paper-facing read is:

- it supports the same mechanism split already seen in the matched MuJoCo
  replay;
- it strengthens the claim that the `LCP-style` advantage is concentrated in
  policy-output smoothness rather than in every downstream closed-loop metric;
- it does **not** close the `one terrain only` limitation, because this is a
  command-distribution shift on the plane replay path rather than a terrain
  benchmark.

## Recommendation

Promote this slice as **supplementary robustness evidence**, not as a new main
results block.

Recommended usage:

- appendix table in the paper;
- one short sentence in Discussion noting that the metric split persists under a
  task-valid command-shift second setting; and
- one sentence in Limitations clarifying that this improves bounded generality
  evidence without becoming multi-terrain validation.
