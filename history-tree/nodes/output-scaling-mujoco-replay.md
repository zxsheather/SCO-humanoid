# Output Scaling MuJoCo Replay

- **Date**: 2026-05-25
- **Type**: experiment
- **Outcome**: failure
- **Tags**: mujoco, cross-engine, output-scaling, collapse

## Timeline and Background

Output-side scaling was tested separately from action-side scaling so the repo would not mix two
different replacement mechanisms. After the Isaac-side line missed the shared entry gate, selected
checkpoints were replayed in `MuJoCo isaac_mainline`.

## Technical Details

- Replay checkpoints: selected `400 / 400 / 400`.
- MuJoCo result: all three seeds collapsed with `fall_rate = 1.0`.
- Aggregate MuJoCo joint acceleration: about `500.5`.
- Isaac-to-MuJoCo joint-acceleration degradation factor: about `4.1x`.

## Decision Process

- The repo closed output-side scaling as a replacement direction under the current same-question
  boundary.
- The result was not treated as a prompt for nearby scale-floor or schedule tuning.

## Results and Impact

- Output-side scaling became another non-Jacobian negative point in the cross-engine degradation table.
- Its failure helped separate the Jacobian constraint path from generic output modulation mechanisms.
