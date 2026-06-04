# Policy-Mechanism Intervention Diagnostic (#122)

Status: `complete`.

## Decision

Use the existing controlled observation-perturbation audit as the bounded
intervention-style diagnostic for the current mechanism chain.

This issue does not open a new closed-loop causal experiment. It consolidates
the strongest existing intervention slice into an explicit paper-facing answer
to the question: does policy-local-sensitivity regularization directly reduce
policy-output amplification, or are we only observing downstream correlation?

## Selected Diagnostic

The chosen diagnostic is:

- `docs/full-paper/policy-perturbation-audit.md`

That slice is technically faithful to the current stack because it:

- uses the existing selected checkpoints for `LCP-style`, `SC-PPO 3.8`, and the
  revised heuristic;
- evaluates every policy on the same pooled observation bank;
- applies explicit controlled perturbations in observation space; and
- reads out the resulting action-response amplification directly from the
  policy, without relying on downstream simulator metrics alone.

## Result

The intervention strengthens the current mechanism interpretation.

On the shared observation bank, controlled perturbations produce the expected
policy-output amplification ordering:

- `LCP-style` has the smallest action response;
- `SC-PPO` is intermediate; and
- the revised heuristic is largest.

The main result note remains:

- `artifacts/analysis/policy_perturbation_audit/summary.json`
- `artifacts/analysis/policy_perturbation_audit/table_policy_perturbation_audit.md`
- `docs/full-paper/policy-perturbation-audit.md`

## Paper-Facing Read

The current evidence is stronger than pure correlation but weaker than a
closed-loop causal proof.

Defensible reading:

> Controlled local observation perturbations reproduce the expected
> policy-output amplification ordering on a shared observation bank. This
> strengthens the interpretation that policy-local-sensitivity regularization
> acts on the policy-output channel itself, not only on aggregate rollout
> smoothness metrics.

Still not defensible:

- a claim that sensitivity alone fully determines locomotion smoothness;
- a claim that the intervention proves cross-engine degradation causally; or
- a claim that the audit replaces the main Isaac/MuJoCo mechanism tables.

## Recommendation

Treat this diagnostic as the current intervention-strengthened mechanism slice
for the paper. It is strong enough to answer the `mechanism or correlation?`
reviewer concern more directly, but it should remain a bounded local-policy
result rather than a promoted primary benchmark.
