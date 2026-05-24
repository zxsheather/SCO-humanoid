# Orthogonal Actor Line

- **Date**: 2026-05-24
- **Type**: experiment
- **Outcome**: failure
- **Tags**: architecture-line, orthogonal, output-gain

## Timeline and Background

Issue [`#34`](https://github.com/zxsheather/SCO-humanoid/issues/34) is the repo's first explicit **架构级平滑优化线** candidate after scaling families closed negative. It tests whether actor-side orthogonal parametrization plus bounded output gain can replace the current Jacobian path.

## Technical Details

- Issue comments record two stages:
  - canonical orthogonal low-gain actor probe with `output_gain = 0.50`
  - gain-isolation follow-up with `output_gain = 1.00`
- Both remained collapsed with `fall_rate = 1.0`.
- Branch detail from git:
  - local branch `orthogonal-actor-line` is the only branch ahead of `main`
  - tip commit [`82d6032`](https://github.com/zxsheather/SCO-humanoid/commit/82d6032) touches 36 files and is broader than the branch name suggests
  - it bundles formal probe infrastructure for action scaling, output scaling, orthogonal actor, and LayerNorm actor

## Decision Process

- The gain-isolation follow-up matters: the repo explicitly ruled out the cheap explanation that low gain alone caused the collapse.
- That let the project close the orthogonal actor line more confidently instead of reopening parameter nibbling inside the same branch.

## Results and Impact

- Orthogonal actor closed negative.
- The architectural search did not stop, but it changed hypothesis: hidden activation normalization became the next candidate.
- Related follow-up: [layernorm-actor-line](../nodes/layernorm-actor-line.md).
