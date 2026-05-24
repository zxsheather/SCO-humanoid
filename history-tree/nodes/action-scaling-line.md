# Action-Side Scaling Line

- **Date**: 2026-05-23
- **Type**: experiment
- **Outcome**: failure
- **Tags**: replacement-line, action-scaling, non-architectural

## Timeline and Background

After the repo's post-freeze diagnostics sharpened the `objective tension` reading, one same-question
follow-up was to replace the active Jacobian-style smoothness mechanism with a constraint-aware
action-side scaling path rather than an actor-side architecture change.

## Technical Details

- This line is the canonical `action-side scaling` precursor later referenced by issues
  [`#33`](https://github.com/zxsheather/SCO-humanoid/issues/33),
  [`#34`](https://github.com/zxsheather/SCO-humanoid/issues/34), and
  [`#35`](https://github.com/zxsheather/SCO-humanoid/issues/35).
- The later issue context establishes the durable project reading:
  - the canonical action-side scaling recipe had already closed negative under the true rough-terrain
    `11 / 17 / 23`, `512 envs x 400 iterations` entry
  - the historical `action_scaling_ppo` prototype scaled actor output mean only and left exploration
    `std` unchanged
- That semantic narrowness is exactly why the repo then opened a separate
  [output-scaling-line](../nodes/output-scaling-line.md) instead of silently reusing the same line as
  both action-side and output-side evidence.

## Decision Process

- The repo treated the first canonical action-side scaling attempt as a clean same-question negative
  result, not as permission to wander immediately through nearby gain or schedule neighborhoods.
- It also refused to misclassify the mean-only prototype as an output-side result.

## Results and Impact

- This line closed negatively and did not clear the shared entry gate.
- Its main historical role is structural:
  - it established that one non-architectural replacement family had already failed
  - it directly set up the cleaner [output-scaling-line](../nodes/output-scaling-line.md)
  - together with output-side failure, it later pushed the repo into
    [orthogonal-actor-line](../nodes/orthogonal-actor-line.md)
