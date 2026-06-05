# Appendix And Limitations Consolidation (#127)

Status: `complete`.

## Goal

Make the paper easier to read by separating:

- main evidence,
- supplementary diagnostics,
- negative-result boundaries, and
- future-work gaps.

This note is the placement map behind the updated venue-neutral manuscript.

## Main Text

Keep only the four evidence blocks that directly support the core claim:

1. five-seed Isaac mechanism comparison;
2. matched five-seed MuJoCo replay;
3. mechanism-chain diagnostics:
   policy perturbation plus observation-noise stress; and
4. robustness and reliability analyses.

Primary source:

- `docs/paper/full-paper.tex`
- `docs/full-paper/main-text-integration.md`

## Appendix Placement

Appendix or supplementary diagnostics should carry the following slices:

- official-LCP parity boundary table
- local LCP coefficient table
- MuJoCo physical secondary metrics
- `hfield_moderate` second-setting replay
- task-valid `plane + command_vx=0.6` command-shift replay
- actuator-bandwidth stress
- bounded Unitree H1 single-seed diagnostic
- OmniSafe PPO-Lag feasibility table

The appendix should also be the default home for any later bounded same-question
diagnostic that does not clearly strengthen the main mechanism claim.

## Limitations Placement

The limitations section should carry the recurring boundary statements once,
rather than repeating them defensively throughout the paper body:

- no official LCP code/checkpoint/task parity
- one primary rough-terrain task
- no claim-grade multi-terrain comparison
- task-valid same-morphology command-shift evidence does not substitute for
  multi-terrain validation
- the current no-heights terrain-observation contract is a likely bottleneck
  for terrain-aware retraining
- no stable multi-robot ranking
- no hardware validation
- selected-checkpoint dependence, especially for SC-PPO and the heuristic
- five-seed uncertainty is descriptive, not a large-sample significance test
- local mechanism evidence is stronger than pure correlation but weaker than a
  closed-loop causal intervention proof
- external constrained-RL coverage remains bounded and diagnostic only

## Negative-Result Boundaries

These lines should be mentioned as bounded evidence, not promoted as main
comparison rows:

- OmniSafe PPO-Lag migration
- local CPO-style training path
- LCP + heuristic hybrid pilot
- mixed rough/stairs retrain no-go
- no-stairs `trimesh + curriculum` retrain no-go
- closer-to-official full-batch LCP scout

The correct role of these results is to sharpen the paper's claim boundary and
show which obvious follow-up branches were tested and did not promote.
Only the OmniSafe feasibility result needs an appendix table in the current
manuscript; the other negative-result lines can remain summarized through the
appendix map, limitations text, and supporting notes.

## Future-Work Bucket

Items that should remain future work rather than partial promises:

- official LCP parity
- height-aware terrain-observation support for task-valid multi-terrain
  retraining at paper-ready quality
- broad multi-robot validation
- hardware transfer
- larger hyperparameter programs

## Manuscript Outcome

The updated `docs/paper/full-paper.tex` reflects this placement policy:

- bounded diagnostics were moved out of the main results flow;
- the limitations section now holds the recurring scope statements; and
- the appendix now carries an explicit placement table for supplementary and
  negative-result lines, including the task-valid command-shift second setting.
