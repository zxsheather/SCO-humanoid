# LDLJ/SPARC Trace Comparison

- **Date**: 2026-05-26
- **Type**: analysis
- **Outcome**: mixed
- **Tags**: trace-metrics, ldlj, sparc, kinematic-smoothness

## Timeline and Background

The cross-engine degradation argument focuses on dynamic smoothness metrics such as joint acceleration
and action jitter. A trace-level comparison added LDLJ/SPARC-style kinematic smoothness metrics to test
whether all smoothness readings tell the same story.

## Technical Details

| Method | jnt_acc | jitter | LDLJ | SPARC |
| --- | ---: | ---: | ---: | ---: |
| SC-PPO 3.8 | 115.9 | 0.22 | -28.35 | -25.54 |
| LayerNorm epochs=3 | 172.0 | 0.52 | -29.69 | -32.28 |

- LayerNorm is better on LDLJ and SPARC.
- SC-PPO is better on joint acceleration and action jitter.

## Decision Process

- The repo interpreted smoothness as at least two-dimensional:
  kinematic trajectory smoothness and dynamic actuation smoothness.
- This prevents overclaiming that one metric family fully summarizes policy quality.

## Results and Impact

- The result complicates the story but improves paper defensibility.
- The core paper claim should stay narrow: Jacobian sensitivity constraints preserve cross-engine
  dynamic smoothness better than the tested non-Jacobian replacements.
