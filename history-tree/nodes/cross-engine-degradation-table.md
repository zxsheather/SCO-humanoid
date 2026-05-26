# Cross-Engine Degradation Analysis

- **Date**: 2026-05-25
- **Type**: analysis
- **Outcome**: success
- **Tags**: cross-engine, degradation, paper, signature-figure

## Timeline and Background

After the post-freeze replacement mechanisms closed, the repo had enough comparable Isaac and MuJoCo
evidence to summarize cross-engine dynamic-smoothness degradation across method families.

## Technical Details

| Method | Isaac jnt_acc | MuJoCo jnt_acc | Degradation |
| --- | ---: | ---: | ---: |
| Heuristic baseline | 120 | 121 | x1.01 |
| SC-PPO 3.8 | 116 | 126 | x1.08 |
| LayerNorm epochs=3 | 172 | 603 | x3.5 |
| Action Scaling | 144 | 1836 | x12.7 |
| Output Scaling | 121 | 500 | x4.1 |

## Decision Process

- The repo moved the paper direction from a simple method-vs-heuristic framing toward a cross-engine
  smoothness degradation framing.
- The table is descriptive evidence, not a claim of real-robot validation.

## Results and Impact

- Jacobian constraint and heuristic action-rate penalty are the only tested mechanisms that preserve
  dynamic smoothness across engines.
- Non-Jacobian replacement mechanisms show large degradation.
- Related follow-up: [sensitivity-causal-chain](../nodes/sensitivity-causal-chain.md).
