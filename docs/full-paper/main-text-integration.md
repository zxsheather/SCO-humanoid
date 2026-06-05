# Main-Text Integration (#125)

Status: `complete`.

## Decision

The manuscript main text should center on exactly four evidence blocks:

1. five-seed Isaac mechanism comparison;
2. matched five-seed MuJoCo replay;
3. mechanism-chain diagnostics built around policy perturbation, with
   observation-noise stress kept in the same block; and
4. robustness and reliability analyses.

This keeps the paper focused on the mechanism-comparison claim rather than
turning the main text into a repository-wide evidence dump.

## Main-Text Promotion

Promoted into the venue-neutral manuscript source:

- five-seed Isaac selected-checkpoint comparison
- matched five-seed MuJoCo replay
- mechanism-chain analysis with perturbation amplification
- observation-noise stress as supporting policy-output evidence
- robustness and reliability summary:
  paired bootstrap + IQM read, selected-vs-final dependence, local LCP
  coefficient boundary, and SC-PPO threshold/plain-dual boundary text

Primary source:

- `docs/paper/full-paper.tex`

## Main-Text Demotion

Removed from the main comparison story and kept as appendix or limitations
material:

- OmniSafe PPO-Lag feasibility check
- local CPO-style feasibility path
- `hfield_moderate` second-setting replay
- task-valid command-shift second setting at `plane + command_vx=0.6`
- actuator-bandwidth stress
- bounded mixed-terrain retrain no-go
- LCP + heuristic hybrid pilot
- closer-to-official full-batch LCP scout
- bounded Unitree H1 cross-morphology probe

These slices remain scientifically useful, but they are boundary-setting
evidence rather than primary comparison blocks.

## Paper-Facing Read

The resulting manuscript is now better aligned with the current claim boundary:

- the paper is about policy-local-sensitivity regularization as a mechanism
  lens for smooth humanoid control;
- `LCP-style` is the strongest current same-task local-sensitivity row;
- `SC-PPO` is the hard-constraint enforcement comparison, not a SOTA claim; and
- the revised heuristic remains a strong reward-shaping anchor rather than a
  strawman.

## Recommendation

Keep future manuscript edits inside this structure unless a new evidence line is
strong enough to displace one of the four promoted blocks. Bounded diagnostics
should default to appendix or limitations placement.
