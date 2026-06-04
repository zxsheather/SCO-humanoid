# Official LCP Parity-Gap Tightening Audit (#120)

Status: `complete`.

This note re-audits how far the current `LCP-style` row can be tightened toward
official LCP without reopening the full benchmark stack. The goal is not to
claim official parity. The goal is to identify the single next alignment axis
that would most reduce reviewer ambiguity at bounded local cost.

## Decision

The single highest-ROI next alignment axis is **penalty sampling detail**.

More specifically: if one closer-to-official scout is opened next, it should
target the current `algorithm.lcp.subsample_obs = 64` gap before attempting
architecture matching, official-checkpoint matching, or task/robot alignment.

## Axis-by-Axis Re-Audit

| Axis | Current local read | Residual gap | Can this be tightened locally without changing the paper question? | ROI read |
| --- | --- | --- | --- | --- |
| Objective | Aligned at the mechanism-family level: fixed soft `grad_obs log pi(a \| obs)` penalty, coefficient anchor `0.002`, and actor-side backprop with `create_graph=True` | Not official-loss equivalence; still embedded in Humanoid-Gym PPO rather than the official MimicKit stack | Mostly already tightened enough for the current claim boundary | Medium-low |
| Sampling | Partial alignment: PPO minibatch state-action samples match the intended rollout-sample family, but the local row randomly subsamples up to `64` actor observations before computing the penalty | This is the clearest remaining method-fidelity deviation inside the current stack | Yes | **Highest** |
| Architecture | Same broad PPO actor-critic family, but not the official task-stack architecture path | Even a closer MLP-width match would still leave observation shaping, action head, and task-stack differences unresolved | Only superficially | Low-medium |
| Checkpoint / evaluation path | Internally fair within this paper: same local checkpoint sweep, Isaac metrics, and MuJoCo replay bridge as the other rows | Not the official LCP checkpoint or evaluation pipeline | Not without leaving the current paper's shared protocol | Low |
| Task / robot | Explicitly different: local H1-class Humanoid-Gym rough-terrain row, not the official task/robot stack | This is a benchmark-scope gap, not a small implementation gap | No | Low for this paper, high for a future reproduction paper |

## Why Sampling Comes First

Penalty sampling detail is the strongest next scout because it is the only
remaining gap that is simultaneously:

- reviewer-visible as a genuine method-fidelity question;
- local to the current training implementation; and
- informative about whether the current row is merely `LCP-style` in wording or
  also materially close in how the penalty is estimated during training.

The present implementation already aligns on the core mechanism:

- fixed soft penalty rather than a dual update;
- the published coefficient anchor `0.002`;
- actor-update participation through second-order autograd; and
- state-action-sample dependence through PPO rollout storage.

What remains visibly local is that the penalty estimate is formed from a random
subsample of at most `64` actor observations per minibatch. That choice is easy
for a reviewer to question because it is neither a task-level limitation nor a
paper-level protocol decision. It is an implementation detail that can, in
principle, be tightened inside the present stack.

## Why the Other Axes Are Not the Next Scout

### Objective

The residual objective gap is real but smaller. The current row already matches
the paper-facing mechanism family, the penalty sign, the fixed coefficient
anchor, and the actor-update path. Tightening the exact PPO constants or other
training details inside Humanoid-Gym would likely consume effort while reducing
little ambiguity compared with the sampling gap.

### Architecture

Architecture is not the right next scout because matching layer widths or other
surface-level network details would not yield defensible official parity while
the observation pipeline, task stack, and checkpoint route remain different.
This would risk spending effort on a change that looks closer without actually
closing the most meaningful reviewer concern.

### Checkpoint / Evaluation Path

Official-checkpoint or official-eval alignment is not locally available under
the current question. The paper's comparison depends on one shared H1
Humanoid-Gym checkpoint-sweep and MuJoCo replay pipeline across SC-PPO, the
revised heuristic, and LCP-style. Replacing that with official LCP checkpoints
would stop being a same-task mechanism comparison and turn into a different
benchmarking exercise.

### Task / Robot

Task and robot alignment are the most obvious non-parity dimensions, but they
are also the least locally tighten-able. Changing them would reopen the
scientific question, not merely harden the current claim boundary. That belongs
to a future official reproduction or cross-stack paper, not to the next bounded
credibility scout for this manuscript.

## Recommended Next Scout

Recommendation for the blocked follow-up line (`#124`):

- keep the current method, seed protocol, and evaluation pipeline fixed;
- remove or materially relax the `subsample_obs = 64` cap;
- prefer `subsample_obs = all` if memory permits, with one larger capped value
  as a fallback only if the full-batch path is infeasible;
- read out both mechanism fidelity and practicality:
  task validity, selected checkpoint, Isaac smoothness metrics, MuJoCo replay
  metrics if the Isaac gate passes, plus wall-clock and peak-memory cost.

This scout would answer the most important remaining local question:

> Does the current five-seed LCP-style result depend materially on the
> `64`-sample penalty estimator, or does it survive a closer-to-official
> penalty-sampling path?

Either outcome is useful:

- if the closer-sampling variant is feasible and qualitatively similar, the
  paper can say the main local method-fidelity gap has been reduced;
- if it is unstable or too expensive, the paper can document a concrete reason
  for retaining the bounded-subsample implementation without pretending that
  the gap is gone.

## Scout Outcome

The follow-up full-batch scout is now complete:

- `docs/full-paper/lcp-full-batch-alignment-scout.md`

Outcome:

- the sampling axis was indeed locally tighten-able;
- the full-batch variant was technically feasible to train in the local stack;
- but the bounded seed-23 `256 env x 200 iter` pilot remained
  `all_checkpoints_collapsed`, so it did not produce a task-valid replacement
  for the current `subsample_obs = 64` paper row.

This sharpens the recommendation further: the main paper should keep the
current `LCP-style` row and continue to describe official parity as unachieved.

## Paper-Facing Outcome

After this audit, the paper-facing boundary is sharper:

- the current row is still defensible as a same-task `LCP-style` adaptation;
- official parity is still not claimed; and
- the next bounded credibility upgrade is now explicit and reviewable:
  tighten penalty sampling detail before opening any broader alignment line.

## Pointers

- Existing fidelity boundary:
  `docs/full-paper/lcp-style-fidelity-audit.md`
- Local baseline recipe:
  `docs/full-paper/lcp-style-baseline-recipe.md`
- Method config:
  `configs/methods/lcp_soft_jacobian_penalty_diagnostic.json`
- Training implementation:
  `.external/humanoid-gym/humanoid/algo/ppo/lcp_ppo.py`
