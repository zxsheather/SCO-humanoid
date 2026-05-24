# Protocol Repair Probe

- **Date**: 2026-05-21
- **Type**: experiment
- **Outcome**: mixed
- **Tags**: protocol-repair, baseline, 512-envs

## Timeline and Background

After the frozen collapse, the repo ran a tracer-bullet repair instead of changing everything at once. The question was narrow: if the previous heuristic winner is rerun under repaired selector semantics and a larger budget, does it stop collapsing?

## Technical Details

- Canonical note: [docs/baselines/rough-terrain-formal-protocol-repair-probe.md](../../docs/baselines/rough-terrain-formal-protocol-repair-probe.md)
- Probe regime:
  - candidate `action_rate = -0.0050`
  - `512 envs x 200 iterations`
  - repaired selector `先过底线再取最平滑`
- Result:
  - selected checkpoints `0 / 0 / 200`
  - `seed11` and `seed17` still collapsed
  - only `seed23` survived, and still weakly

## Decision Process

- The probe ruled out the strongest "universal collapse under every variant" claim.
- It did **not** justify calling the baseline fixed.
- That pushed the repo from "repair" language to "revision" language: the baseline protocol itself needed replacement.

## Results and Impact

- This node is the bridge between failure and closure.
- It directly triggered [revised-heuristic-anchor](../nodes/revised-heuristic-anchor.md).
- The project stopped debating heuristic weights and started debating formal anchor rules.
