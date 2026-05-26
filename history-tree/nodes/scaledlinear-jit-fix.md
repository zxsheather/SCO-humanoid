# ScaledLinear TorchScript Fix

- **Date**: 2026-05-25
- **Type**: fix
- **Outcome**: success
- **Tags**: torchscript, jit, export, infrastructure

## Timeline and Background

The replacement-mechanism MuJoCo replay path exposed a TorchScript export issue in models that used
scaled actor output layers. The fix was needed so trained replacement-mechanism checkpoints could be
exported and replayed consistently.

## Technical Details

- The implementation changed `ScaledLinear.forward` from a `super().forward()` call to an explicit
  `F.linear()` call.
- This made the layer compatible with `torch.jit.script`.
- The change was infrastructure-only and did not alter the experimental claim boundary.

## Decision Process

- The repo treated this as a reusable diagnostic/export backport, which is allowed under the
  `冻结主档案分支` rule.
- It was not counted as new mechanism-specific evidence.

## Results and Impact

- MuJoCo export became available for replacement-mechanism models using scaled output layers.
- This enabled the later cross-engine replay records for scaling and LayerNorm candidates.
