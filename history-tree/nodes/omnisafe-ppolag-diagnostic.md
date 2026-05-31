# OmniSafe PPO-Lag Diagnostic

- **Date**: 2026-05-28
- **Type**: experiment
- **Outcome**: failure
- **Tags**: omnisafe, ppo-lag, interface-boundary

## Timeline and Background

The project investigated whether a more standard constrained-RL implementation
could provide a stronger external baseline than the local SC-PPO implementation.
OmniSafe PPO-Lag was chosen as the first framework migration target.

## Technical Details

- Adapter smoke path: `docs/full-paper/omnisafe-ppolag-adapter-smoke.md`.
- Cost bridge result: `docs/full-paper/omnisafe-cost-bridge-result.md`.
- Update-hook smoke: `docs/full-paper/omnisafe-ppolag-update-hook.md`.
- Bounded diagnostic result:
  `docs/full-paper/omnisafe-ppolag-diagnostic-results.md`.

## Decision Process

The key constraint is actor-internal: it depends on the current policy
Jacobian with respect to observations during the update. Standard safe-RL
framework adapters usually consume environment-side rollout costs, which is not
faithful to the policy-local-sensitivity question.

## Results and Impact

The bounded PPO-Lag diagnostic collapsed and was not promoted to a main
baseline. The correct interpretation is an interface boundary, not evidence
that OmniSafe, PPO-Lag, or constrained RL broadly fails.
