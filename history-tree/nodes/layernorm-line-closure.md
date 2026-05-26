# LayerNorm Actor Line Closure

- **Date**: 2026-05-25
- **Type**: decision
- **Outcome**: failure
- **Tags**: architecture-line, closure, negative-boundary

## Timeline and Background

LayerNorm actor became the strongest architecture-side replacement candidate in the post-freeze
same-question exploration. It was the first architecture candidate to pass the full Isaac internal
challenge, but the MuJoCo replay changed the interpretation.

## Technical Details

- Positive evidence:
  - `num_learning_epochs = 3` achieved `selected = final = 400` on all three Isaac seeds
  - task metrics improved relative to `SC-PPO 3.8` on the Isaac side
- Negative evidence:
  - Isaac dynamic smoothness was worse than `SC-PPO 3.8`
  - MuJoCo replay had about `5x` worse joint acceleration and about `14x` worse action jitter than
    `SC-PPO 3.8`
  - Isaac-to-MuJoCo joint-acceleration degradation was about `3.5x`

## Decision Process

- The repo closed the line as a task-valid but smoothness-negative mechanism.
- It did not reopen nearby LayerNorm tuning as the next frontier under the same issue.

## Results and Impact

- LayerNorm actor is no longer an open history-tree frontier.
- Its negative closure strengthens the paper's cross-engine degradation argument.
- The stable conclusion is that architecture-side normalization can repair task reliability but does
  not preserve cross-engine dynamic smoothness like the Jacobian constraint path.
