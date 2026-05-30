# CPO evidence decision

Issue: [#84](https://github.com/zxsheather/SCO-humanoid/issues/84)
Parent: [#80](https://github.com/zxsheather/SCO-humanoid/issues/80)
Date: 2026-05-30
Status: **Decision accepted - CPO remains diagnostic/future work**

## Decision

Do not promote CPO to a paper baseline in the current full-paper manuscript.
Keep the result as a **local CPO-style diagnostic** and describe CPO as future
work or limitation material only.

No multi-seed CPO training issue should be opened from the current evidence.

## Evidence Readout

| Issue | Result | What it proves | What it does not prove |
| --- | --- | --- | --- |
| [#81](https://github.com/zxsheather/SCO-humanoid/issues/81) | Autograd/HVP smoke passed | The actor-internal Jacobian cost can produce finite parameter gradients, and KL Fisher-vector products are shape-consistent and finite. | No rollout, no constrained update, no training stability, and no official CPO parity. |
| [#82](https://github.com/zxsheather/SCO-humanoid/issues/82) | One-update CPO-style prototype passed | A local reward gradient, constraint gradient, CG solve, dual decision, and line search can complete once. | No repeated-update stability, no checkpoint quality, and no task-valid policy. |
| [#83](https://github.com/zxsheather/SCO-humanoid/issues/83) | Bounded seed-23 diagnostic ran but collapsed | The CPO-style update can run inside the Humanoid-Gym training loop, produce checkpoints, and log finite accepted updates. | It does not produce a usable baseline: every evaluated checkpoint had `fall_rate=1.000`. |

## Manuscript Boundary

Use this label:

- `local CPO-style diagnostic`

Avoid these labels:

- `official CPO`;
- `OmniSafe CPO`;
- `CPO parity`;
- `CPO baseline`;
- `CPO failure`;
- `external constrained RL failure`.

The correct paper-facing statement is:

> A local CPO-style path for the actor-internal Jacobian cost is technically
> possible at the tensor, one-update, and tiny training-loop levels, but the
> bounded training diagnostic collapsed on all evaluated checkpoints. We
> therefore keep CPO as future work rather than promoting it to a baseline.

## Follow-Up Policy

For the current mechanism-comparison paper:

- do not add CPO to the main result tables;
- do not run a multi-seed CPO expansion;
- mention CPO only in limitations/future work or as appendix diagnostic context;
- rely on LCP-style soft Jacobian/Lipschitz regularization as the closest
  literature-aligned policy-sensitivity baseline;
- keep OmniSafe PPO-Lag as a framework-interface diagnostic, not as a promoted
  external constrained-RL baseline.

For a future revision, a new CPO issue is justified only if it explicitly
allocates a larger bounded budget, preserves exact wall-time and memory
measurements, keeps a single-seed gate before any multi-seed expansion, and does
not replace the actor-internal Jacobian cost with an environment-side proxy.
