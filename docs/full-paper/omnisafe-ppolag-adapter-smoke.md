# OmniSafe PPO-Lag Adapter Smoke

**Issue:** #60
**Branch:** `issue-60-omnisafe-adapter-smoke`
**Run:** `omnisafe_ppolag_adapter_smoke_seed23_verify`

## Status

The Humanoid-Gym to OmniSafe tuple smoke completed and wrote:

- `artifacts/methods/omnisafe_ppolag_adapter_smoke/omnisafe_ppolag_adapter_smoke_seed23_verify/omnisafe_adapter_smoke.json`
- `artifacts/methods/omnisafe_ppolag_adapter_smoke/omnisafe_ppolag_adapter_smoke_seed23_verify/manifest.json`

The direct Python process segfaulted during Isaac Gym teardown after writing the
complete artifact, matching the repo's known Isaac cleanup pattern. The shell
wrapper verifies `status=complete` in the artifact and recovers the command to a
successful exit.

## Command

```bash
PYTHON_BIN=/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python \
  scripts/baseline/run_omnisafe_adapter_smoke.sh \
  --run-name=omnisafe_ppolag_adapter_smoke_seed23_verify \
  --num-envs=1 \
  --steps=1 \
  --seed=23 \
  --rl-device=cuda:0 \
  --sim-device=cuda:0 \
  --write-failure-artifact
```

## Result

The adapter successfully instantiated the rough-terrain humanoid task, reset it,
and stepped once through the OmniSafe-style CMDP tuple:

`observation, reward, cost, terminated, truncated, info`

Observed shapes:

| Field | Shape | Device | Dtype |
| --- | ---: | --- | --- |
| observation | `[1, 705]` | `cuda:0` | `torch.float32` |
| action | `[1, 12]` | `cuda:0` | `torch.float32` |
| reward | `[1]` | inherited | `torch.float32` |
| cost | `[1]` | inherited | `torch.float32` |
| terminated | `[1]` | inherited | bool |
| truncated | `[1]` | inherited | bool |

Cost status:

- `cost_source = non_canonical_zero_smoke`
- `cost_is_canonical = false`

This cost is intentionally a zero smoke field. It only verifies the OmniSafe
custom-environment tuple contract and must not be used as a #53 baseline cost.
The faithful policy-local-sensitivity/Jacobian cost bridge remains #61.

## Consequence

#60 passes. The simulator/runtime boundary is viable enough to proceed to #61:
bridge the actual SC-PPO local-sensitivity cost into the PPO-Lag path or close
the external baseline as infeasible if that bridge becomes too invasive.
