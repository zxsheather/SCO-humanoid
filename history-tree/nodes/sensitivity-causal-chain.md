# Jacobian Sensitivity Causal Chain

- **Date**: 2026-05-25
- **Type**: analysis
- **Outcome**: success
- **Tags**: sensitivity, causal-chain, checkpoint-sweep, paper

## Timeline and Background

The cross-engine degradation table needed a mechanism-level explanation rather than only outcome
statistics. The repo therefore compared policy local-sensitivity readings across the successful
SC-PPO and LayerNorm paths.

## Technical Details

- `SC-PPO 3.8` keeps policy local sensitivity near `3.6`.
- LayerNorm epochs=3 reaches task validity with sensitivity near `10.7`.
- The higher LayerNorm sensitivity aligns with the observed `3.5x` Isaac-to-MuJoCo dynamic-smoothness
  degradation.
- The SC-PPO PID-Lagrangian path keeps the sensitivity level bounded across the checkpoint sweep.

## Decision Process

- The repo treated this as support for a narrow causal-chain claim:
  constraining policy Jacobian sensitivity acts as implicit sim-to-sim regularization for dynamic
  smoothness.
- It did not claim all smoothness metrics agree; LDLJ/SPARC later show a separate kinematic axis.

## Results and Impact

- This node supplies the mechanistic bridge between the degradation table and the paper thesis.
- Related analysis:
  - [cross-engine-degradation-table](../nodes/cross-engine-degradation-table.md)
  - [trace-ldlj-sparc-comparison](../nodes/trace-ldlj-sparc-comparison.md)
