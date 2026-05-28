# OmniSafe Policy Evaluation Bridge (#62)

## Status

The #62 compatibility slice passes. The repository can now load an
OmniSafe-style Gaussian actor checkpoint, run deterministic rough-terrain
rollouts, emit the shared metric schema, and summarize checkpoint rows with the
same task-floor-then-smoothest selection rule used by the existing checkpoint
sweeps.

This is not a training result. The smoke uses fixture checkpoints with the same
loader format expected from the downstream PPO-Lag training path.

## Added Path

- Policy loader: `scripts/baseline/_omnisafe_policy_loader.py`
- Evaluator: `scripts/baseline/evaluate_omnisafe_policy.py`
- Sweep summarizer: `scripts/baseline/summarize_omnisafe_checkpoint_sweep.py`
- Smoke wrapper: `scripts/baseline/run_omnisafe_eval_smoke.sh`
- Config: `configs/methods/omnisafe_ppolag_eval_smoke.json`

The evaluator writes the same top-level metrics used by rough-terrain checkpoint
sweeps:

- velocity tracking error;
- fall rate;
- joint acceleration;
- action jitter;
- episode return;
- constraint/local-sensitivity diagnostics when enabled.

## Smoke Result

Command:

```bash
PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python \
  scripts/baseline/run_omnisafe_eval_smoke.sh
```

Artifacts:

- `artifacts/methods/omnisafe_ppolag_eval_smoke/omnisafe_ppolag_eval_smoke_seed23/metrics_checkpoint_0.json`
- `artifacts/methods/omnisafe_ppolag_eval_smoke/omnisafe_ppolag_eval_smoke_seed23/metrics_checkpoint_100.json`
- `artifacts/methods/omnisafe_ppolag_eval_smoke/omnisafe_ppolag_eval_smoke_seed23/checkpoint_sweep_summary.json`

Selected-checkpoint summary:

| Field | Value |
| --- | ---: |
| selection status | `all_checkpoints_collapsed` |
| selected checkpoint | `0` |
| velocity tracking error | `1.5975` |
| fall rate | `1.0000` |
| joint acceleration | `97.0577` |
| action jitter | `0.0288` |
| episode return | `2.9920` |
| local sensitivity mean | `0.5971` |
| violation rate | `0.0000` |

The fixture policy is randomly initialized, so collapse is expected and should
not be interpreted as a PPO-Lag result.

As with the other Isaac Gym smoke paths, the direct Python evaluation exits with
the known teardown segmentation fault after writing complete artifacts. The
wrapper recovers only when the per-checkpoint `omnisafe_evaluation.json` reports
`status=complete` for the matching checkpoint.

## Consequence

#62 unlocks #63: the next slice can produce real OmniSafe PPO-Lag checkpoints
using the #65 update hook, evaluate them with this bridge, and aggregate seeds
`23 / 29 / 31` with the shared selection rule.
