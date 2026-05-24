# Objective Mismatch Diagnostic

- **Date**: 2026-05-22
- **Type**: experiment
- **Outcome**: mixed
- **Tags**: post-freeze, diagnostic, alignment

## Timeline and Background

Issue [`#19`](https://github.com/zxsheather/SCO-humanoid/issues/19) opens after the anisotropic constraint line closed negative. The question shifts from "what anisotropic weight next?" to "is the current local-sensitivity objective pulling in the wrong direction relative to task validity?"

## Technical Details

- Canonical note: [docs/sc-ppo-objective-mismatch-diagnostic.md](../../docs/sc-ppo-objective-mismatch-diagnostic.md)
- Merged via PR [`#28`](https://github.com/zxsheather/SCO-humanoid/pull/28) on 2026-05-22.
- Main code change:
  - checkpoint-level alignment summaries inside `scripts/baseline/evaluate_checkpoint_sweep.py`
  - comparison between train-side local-sensitivity signals and eval-side task/behavior metrics
- Multi-seed mainline replay found a stable sign pattern:
  - higher local-sensitivity cost correlated with better task validity and better tracking
  - the same higher cost also correlated with rougher action jitter and joint acceleration

## Decision Process

- The branch rejected the simple "metric is noise" story.
- It also rejected the easy "constraint objective is obviously helping smoothness" story.
- The more defensible reading became: the current objective is behavior-relevant, but it is in tension with the task-valid regime the policy actually learns.

## Results and Impact

- This branch did not create a new method line.
- It did sharpen the next question, which led directly to [behavior-trace-metrics](../nodes/behavior-trace-metrics.md).
- It also gave the repo a stronger justification for post-freeze architecture experiments rather than tiny threshold tweaks.
