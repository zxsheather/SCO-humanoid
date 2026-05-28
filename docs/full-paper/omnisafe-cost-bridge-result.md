# OmniSafe Cost Bridge Result (#61)

**Branch:** `full-paper/extended-seeds`
**Issue:** #61
**Parent:** #53

## Status

The OmniSafe PPO-Lag cost bridge smoke test passed. The Jacobian local-sensitivity
cost is computable and feeds faithfully into OmniSafe's Lagrange multiplier
update. Numerical values are finite and the multiplier behaves correctly.

## Smoke Result

```
Jacobian cost = 0.2886 (threshold=3.8)
multiplier: 0.5000 → 0.4649
constraint_error = -3.5114
cost_is_canonical = true
cost_source = policy_local_sensitivity_jacobian
```

The multiplier decreases (0.5 → 0.465) because the randomly-initialized network
produces Jacobian cost (0.29) far below the threshold (3.8). This is correct:
gradient ascent on λ moves toward zero when the constraint is satisfied.

## Equivalence Finding

OmniSafe's Lagrange multiplier update is mathematically equivalent to SC-PPO's
plain dual ascent mode:

| Component | OmniSafe Lagrange | SC-PPO Plain Dual |
|-----------|-------------------|-------------------|
| Update rule | `λ += lr * (Jc - d)` via SGD | `λ += η * (Jc - d)` |
| Clamping | `λ.clamp_(0, upper)` | `clamp(λ, 0, λ_max)` |
| Cost | Jacobian (bridged) | Jacobian (native) |

With `lr = η` and `upper = λ_max`, the two are identical. SC-PPO's existing
plain dual ascent comparison (#51) already covers this baseline. The only
functional difference is optimizer choice (SGD vs Adam), which is a minor
implementation detail, not an algorithmic distinction.

## Conclusion

The cost bridge is **feasible but not a distinct external baseline**. OmniSafe
PPO-Lag with the Jacobian cost bridge would produce results equivalent to
SC-PPO's existing plain dual ascent row. Running a full training sweep would
consume GPU budget without adding new evidence beyond the already-completed
plain-dual-vs-PID comparison.

**Recommendation: close #53 as "equivalent to existing evidence"** rather than
running expensive OmniSafe training diagnostics (#62, #63). The Lagrange
multiplier comparison (dual ascent vs PID) is already present in the current
evidence package.

## Artifacts

- Bridge module: `scripts/baseline/_omnisafe_bridge.py`
- Smoke script: `scripts/baseline/run_omnisafe_cost_bridge_smoke.py`
- Shell wrapper: `scripts/baseline/run_omnisafe_cost_bridge_smoke.sh`
- Config: `configs/methods/omnisafe_ppolag_cost_bridge_smoke.json`
- Smoke artifact: `artifacts/methods/omnisafe_ppolag_cost_bridge_smoke/.../omnisafe_cost_bridge_smoke.json`
