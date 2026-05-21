# Report-Grade Results for Smooth-Constrained PPO on Rough-Terrain Humanoid Locomotion

## Abstract

This report fixes the current report-grade claim for the repo's smooth-control study. On the Isaac
rough-terrain main experiment, repaired PID-Lagrangian `SC-PPO` with `threshold = 3.8` beats the
revised heuristic smoothing baseline under a `3-seed + checkpoint-sweep` protocol. On `MuJoCo
isaac_mainline`, the aligned replay is mixed external-validation evidence rather than a
cross-engine `SC-PPO` win. A tighter `3.6 + full_batch` challenger does not replace the `3.8`
mainline because its formal promotion attempt fails at the Isaac stage.

## 1. Research Question and Current Claim

This project is a research-validation delivery rather than a framework productization effort. The
current question is whether repaired PID-Lagrangian `SC-PPO` can beat a strong heuristic
smoothness baseline on the repo's rough-terrain humanoid velocity-tracking task.

The current report-grade claim is narrow:

1. On the Isaac rough-terrain main experiment, repaired PID-Lagrangian `SC-PPO` with
   `threshold = 3.8` supports a defensible method-over-heuristic result.
2. On `MuJoCo isaac_mainline`, the aligned replay supports mixed external-validation evidence: the
   revised heuristic anchor is stronger on task-side metrics, while `SC-PPO 3.8` is only slightly
   stronger on action jitter.
3. The tighter `3.6 + full_batch` line does not replace the `3.8` mainline, because its formal
   promotion attempt failed at the Isaac stage.

## 2. Protocol and Evidence Boundary

Current canonical configs:

- Raw reference: `configs/methods/vanilla_ppo.json`
- Heuristic anchor:
  `configs/methods/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget.json`
- Formal mainline: `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json`
- Failed promotion candidate: `configs/methods/sc_ppo_threshold_36_lambda_05_quantile_090_pid_lower_bound_clamp_full_batch.json`

Evidence rules:

- The Isaac main result is based on `3-seed + checkpoint-sweep` selected checkpoints, not on the
  final checkpoint alone.
- The selected checkpoints for the current `3.8` mainline are `300`, `300`, and `400` for seeds
  `11`, `17`, and `23`.
- The `MuJoCo isaac_mainline` result is now aligned to the revised heuristic anchor with a
  `3-seed` selected-checkpoint replay for both key methods.
- `hfield_moderate` and `hfield_stress` are excluded from the main result narrative. They remain
  repair-stage protocol lines, not report-grade headline evidence.
- Any follow-up work on `hfield_moderate` or `hfield_stress` belongs to the terrain-side protocol
  repair backlog, not to a new algorithm-promotion claim.

## 3. Isaac Rough-Terrain Main Result

`Vanilla PPO` remains useful as the raw reference that shows the unsmoothed starting point under
the current shared protocol. The main claim, however, is carried by the comparison between the
selected heuristic anchor and the current `SC-PPO 3.8` mainline.

### Table 1. Isaac rough-terrain comparison

| Method | Evidence scope | `velocity_tracking_error_mean` | `joint_acceleration_l2_mean` | `action_jitter_l2_mean` | `episode_return_mean` | `fall_rate` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Vanilla PPO` | raw reference, frozen formal compare (`ckpt 0/0/0`) | `1.3321 +/- 0.1181` | `83.7179 +/- 13.3692` | `0.0161 +/- 0.0008` | `4.0002 +/- 0.4323` | `1.0000 +/- 0.0000` |
| `PPO + heuristic smoothing (action_rate=-0.0050)` | revised heuristic anchor (`ckpt 350/300/350`) | `0.7549 +/- 0.1068` | `119.8639 +/- 2.1966` | `0.2711 +/- 0.0084` | `100.9327 +/- 11.2711` | `0.1500 +/- 0.0816` |
| `SC-PPO 3.8` | `3-seed` selected-checkpoint aggregate (`ckpt 300/300/400`) | `0.6412 +/- 0.0554` | `115.9079 +/- 6.9386` | `0.2205 +/- 0.0017` | `100.2838 +/- 2.7150` | `0.1000 +/- 0.0000` |

Table note: the `Vanilla PPO` row is a raw-reference collapse record. The heuristic and `SC-PPO`
rows are selected-checkpoint aggregates, not final-checkpoint-only summaries.

Interpretation:

- The raw `Vanilla PPO` reference remains fully collapsed and is useful only as a raw reference.
- The revised heuristic baseline is task-valid and is now a meaningful formal anchor.
- The repaired PID-Lagrangian `SC-PPO 3.8` mainline beats that heuristic anchor on all current
  shared Isaac-side comparison metrics, while also reducing collapse.

This is the current strongest result in the repo.

![Figure 1. Isaac rough-terrain main result](artifacts/analysis/sc_ppo_report_figures/figure_isaac_main_result.png)

Figure 1. Shared-metric Isaac comparison. `SC-PPO 3.8` uses the selected-checkpoint aggregate over
seeds `11`, `17`, and `23`.

## 4. MuJoCo `isaac_mainline` Aligned Replay

The current `MuJoCo` reading must stay weaker than the Isaac main result. After aligning the replay
to the revised heuristic anchor, the comparable scope is:

- Heuristic anchor: checkpoints `350 / 300 / 350`
- `SC-PPO 3.8`: checkpoints `300 / 300 / 400`
- Protocol: `terrain_mode = isaac_mainline`, `joint_reset_noise = 0.1`, `20 episodes`,
  `20 seconds`

### Table 2. `MuJoCo isaac_mainline` aligned comparison

| Method | Evidence scope | `velocity_tracking_error_mean` | `joint_acceleration_l2_mean` | `action_jitter_l2_mean` | `fall_rate` | `episode_steps_mean` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `PPO + heuristic smoothing (action_rate=-0.0050)` | revised anchor, `3-seed` selected-checkpoint replay | `0.4188 +/- 0.0398` | `120.7339 +/- 2.6413` | `0.2452 +/- 0.0288` | `0.0000 +/- 0.0000` | `2000.0 +/- 0.0` |
| `SC-PPO 3.8` | `3-seed` selected-checkpoint replay | `0.4910 +/- 0.0944` | `125.5411 +/- 21.1683` | `0.2313 +/- 0.0351` | `0.0167 +/- 0.0236` | `1984.8 +/- 21.5` |

Interpretation:

- The revised heuristic anchor is stronger on task stability, velocity tracking, episode length,
  and joint acceleration.
- `SC-PPO 3.8` is only slightly stronger on `action_jitter_l2_mean`.
- This does not preserve the Isaac-side ordering.

So the correct reading is mixed external-validation evidence, not an `SC-PPO` cross-engine win.

Headline-safe wording:

`On the current Isaac rough-terrain main experiment, repaired PID-Lagrangian SC-PPO (threshold = 3.8) supports the repo's main result against the revised heuristic baseline; on the aligned MuJoCo isaac_mainline replay, that ordering does not transfer, so the external-validation result should be reported as mixed external-validation evidence rather than as a cross-engine SC-PPO win.`

![Figure 2. MuJoCo isaac_mainline aligned replay](artifacts/analysis/sc_ppo_report_figures/figure_mujoco_aligned_replay.png)

Figure 2. Aligned `MuJoCo isaac_mainline` replay against the revised heuristic anchor.

## 5. Why `3.8` Remains the Formal Mainline

The `3.6 + full_batch` line was evaluated only as a formal challenger to the current `3.8`
mainline. It was not a fresh repo-wide claim against the heuristic anchor.

Promotion rule:

- reuse `seed11`
- add `seed17` and `seed23`
- require each seed to clear the Isaac-side hard gate
- reject pathological early or null checkpoint selection

### Table 3. `3.6 + full_batch` Isaac promotion outcome

| Seed | Selected checkpoint | `velocity_tracking_error_mean` | `joint_acceleration_l2_mean` | `action_jitter_l2_mean` | `fall_rate` | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `11` | `350` | `0.5735` | `116.0164` | `0.2107` | `0.1000` | locally strong |
| `17` | `350` | `0.6770` | `129.7265` | `0.2289` | `0.6500` | unstable but still non-null |
| `23` | `0` | `1.2863` | `80.8770` | `0.0115` | `1.0000` | pathological null selection |

Interpretation:

- `seed11` looked promising.
- `seed17` was materially weaker.
- `seed23` selected `checkpoint 0`, which triggers the repo's early-checkpoint failure rule.

So the promotion stops at the Isaac stage. No new `MuJoCo isaac_mainline` budget should be spent
for this line, and `3.6 + full_batch` returns to a completed diagnostic branch rather than a new
formal mainline.

PID-limited mechanism note:

- A matched plain-dual diagnostic at `threshold = 3.8` was evaluated at checkpoint `100`.
- It collapsed under the minimal Isaac evaluation: `fall_rate = 1.0000`,
  `episode_return_mean = 4.7101`, and `velocity_tracking_error_mean = 1.1646`.
- Its lower `action_jitter_l2_mean = 0.1661` is not a usable smooth-control win because the policy
  is not task-valid.
- This supports keeping `PID-Lagrangian` as the formal `SC-PPO` update, but it is only a
  `PID有限消融`, not a full component-attribution study.

![Figure 3. Failed promotion of 3.6 plus full batch](artifacts/analysis/sc_ppo_report_figures/figure_threshold36_promotion_failure.png)

Figure 3. The formal `3.6 + full_batch` promotion line fails at the Isaac stage because `seed23`
selects `checkpoint 0`.

## 6. Established Result and Remaining Boundary

Established:

- Repaired PID-Lagrangian `SC-PPO` with `threshold = 3.8` is now the formal mainline.
- On Isaac rough terrain, that line supports a defensible method-over-heuristic claim.
- The result survives `3 seeds` and explicitly depends on checkpoint sweep selection.
- The nearest tighter challenger `3.6 + full_batch` does not replace the current mainline.
- `MuJoCo isaac_mainline` is now aligned to the revised heuristic anchor and supports mixed
  external-validation evidence.
- The matched plain-dual diagnostic collapses, so the limited PID ablation supports the current
  `PID-Lagrangian` algorithm boundary without adding a new mainline result.

Not established:

- that the final checkpoint alone is sufficient for the current `SC-PPO` branch
- that the Isaac-side ordering transfers to `MuJoCo`
- that smoothness superiority fully transfers to `MuJoCo`
- that `hfield_moderate` or `hfield_stress` are ready to serve as report-grade terrain results
- that a broader neighborhood of tighter thresholds is interchangeable with the `3.8` mainline
- that the PID-limited diagnostic proves independent necessity of every PID term

Any remaining terrain-side work should therefore be read as protocol repair for external-validation
semantics, not as evidence that a different algorithm line is ready to replace the current mainline.

## 7. Canonical Artifacts

Isaac main result:

- `artifacts/analysis/rough_terrain_formal_comparison/comparison_summary.json`
- `artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json`
- heuristic selected metrics:
  `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/*/metrics_selected.json`
- `SC-PPO 3.8` selected metrics and checkpoint sweeps:
  `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed*/`

`MuJoCo isaac_mainline` aligned replay:

- heuristic:
  `artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/*/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `SC-PPO 3.8`:
  `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed*/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`

Failed promotion line:

- `artifacts/methods/sc_ppo_fullbatch_threshold_probe/sc_ppo_fullbatch_threshold_36_iter400_seed*/checkpoint_sweep_summary.json`

PID-limited mechanism diagnostic:

- `docs/sc-ppo-pid-limited-ablation.md`
- `artifacts/analysis/sc_ppo_pid_limited_ablation/summary.json`

Generated figures:

- `scripts/analysis/generate_sc_ppo_report_figures.py`
- `artifacts/analysis/sc_ppo_report_figures/figure_isaac_main_result.png`
- `artifacts/analysis/sc_ppo_report_figures/figure_mujoco_aligned_replay.png`
- `artifacts/analysis/sc_ppo_report_figures/figure_threshold36_promotion_failure.png`
- `artifacts/analysis/sc_ppo_report_figures/manifest.json`

## 8. Post-Mainline Diagnostics and Freeze Boundary

After the mainline rough-terrain result was fixed, the repo completed three bounded diagnostics.
They clarify the delivery boundary but do not create new headline results:

- The `PID有限消融` supports keeping `PID-Lagrangian` as the formal `SC-PPO` update. The matched
  plain-dual probe collapses, so the result is mechanism support rather than full component
  attribution.
- The `SN-only` replacement diagnostic is operational but negative. Full-actor, hidden-layer-only,
  `coeff = 2.0`, and first-hidden-only reduced-budget variants all collapse, so blind SN-only
  toggles should stop.
- The random-stairs selected-checkpoint stress test is closed as direct transfer failure. Vanilla
  PPO, the revised heuristic anchor, and `SC-PPO 3.8` all have `fall_rate = 1.0` under the first
  stairs-only protocol, so the result is not a task-valid random-stairs method ranking.

The repo therefore enters `科研交付冻结`: the output is a `仓库内科研交付包` with aligned reports,
tracker state, artifact pointers, and reproduction entrypoints. Freeze validation is limited to
tests, JSON/path sanity, Markdown consistency, and git hygiene. New moderated-stairs protocols,
task-stabilized SN recipes, or additional terrain repair should be opened as post-freeze branches
rather than folded into this report.

Freeze references:

- `artifacts/analysis/final_research_delivery_freeze/summary.json`
- `docs/reproduction/final-research-delivery-checklist.md`
- `docs/adr/0001-freeze-research-delivery-before-new-protocol-repair.md`
- `docs/sc-ppo-sn-feasibility-diagnostic.md`
- `docs/random-stairs-selected-checkpoint-stress.md`
- `artifacts/analysis/sn_replacement_diagnostic/sn_ppo_first_hidden_rough_terrain_medium_seed123145_summary.json`
- `artifacts/analysis/random_stairs_selected_checkpoint_stress/comparison_summary.json`
