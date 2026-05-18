# SC-PPO `粗糙平面` 结果报告

## 摘要

本文记录仓库当前可辩护的 report-grade 结果。在 Isaac `粗糙平面` 主实验上，修复后的
PID-Lagrangian `SC-PPO` 且 `threshold = 3.8`，已经在 `3-seed + checkpoint-sweep` 协议下
形成对已选 heuristic baseline 的主结果。在 `MuJoCo isaac_mainline` 上，当前最小可比 replay
只支持有边界的 `partial transfer`：`SC-PPO` 在任务稳定性和速度跟踪上更强，但当前行为层
平滑指标还没有转优。更紧的 `3.6 + full_batch` 候选线则因为 Isaac 升格失败，没有取代当前
`3.8` 主线。

## 1. 研究问题与当前回答

本项目当前是 `科研验证型交付`，不是框架产品化工作。当前要回答的问题是：修复后的
PID-Lagrangian `SC-PPO`，是否能在仓库定义的 `粗糙平面` 人形机器人速度跟踪任务上，
稳定优于强启发式平滑基线。

当前可成立的回答很窄，也必须写得很硬：

1. 在 Isaac `粗糙平面` 主实验中，修复后的 `SC-PPO` 且 `threshold = 3.8` 已经形成真实的
   `method beats heuristic` 主结果。
2. 在 `MuJoCo isaac_mainline` 上，目前只能支持 `partial transfer` 结论：任务稳定性和速度跟踪
   更强，但当前行为层平滑指标还没有转优。
3. 更紧的 `3.6 + full_batch` 线没有取代当前 `3.8` 主线，因为它的正式升格尝试在 Isaac 阶段失败。

## 2. 协议与证据边界

当前 canonical configs:

- 原始参照组: `configs/methods/vanilla_ppo.json`
- 启发式锚点: `configs/methods/heuristic_smoothing_action_rate_0050.json`
- 正式主线: `configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json`
- 升格失败候选线: `configs/methods/sc_ppo_threshold_36_lambda_05_quantile_090_pid_lower_bound_clamp_full_batch.json`

证据规则必须写死：

- Isaac 主结果基于 `3-seed + checkpoint-sweep` 的 selected checkpoints，不是 final checkpoint。
- 当前 `3.8` 主线的 selected checkpoints 分别是 `seed11 -> 300`、`seed17 -> 300`、
  `seed23 -> 400`。
- `MuJoCo isaac_mainline` 结果只是 representative first-pass external validation，不是对齐到
  `3-seed` 的统计性结论。
- `hfield_moderate` 和 `hfield_stress` 不进入主结果叙事。它们仍然是 repair-stage protocol lines
  和当前阻塞项，不是 report-grade headline evidence。

## 3. Isaac `粗糙平面` 主结果

`Vanilla PPO` 仍然有价值，因为它给出了当前共享协议下的原始未平滑参照。但真正承载主命题的，
是启发式锚点与当前 `SC-PPO 3.8` 主线之间的对比。

### 表 1. Isaac `粗糙平面` 对比结果

| Method | Evidence scope | `velocity_tracking_error_mean` | `joint_acceleration_l2_mean` | `action_jitter_l2_mean` | `episode_return_mean` | `fall_rate` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Vanilla PPO` | 单次结果, `20 episodes` | `1.2835` | `164.3631` | `0.2969` | `8.2694` | `1.0000` |
| `PPO + heuristic smoothing (action_rate=-0.0050)` | 已选启发式锚点, 单次结果, `20 episodes` | `1.1381` | `140.6399` | `0.2457` | `11.9674` | `1.0000` |
| `SC-PPO 3.8` | `3-seed` selected-checkpoint aggregate (`ckpt 300/300/400`) | `0.6412 +/- 0.0554` | `115.9079 +/- 6.9386` | `0.2205 +/- 0.0017` | `100.2838 +/- 2.7150` | `0.1000 +/- 0.0000` |

表注：`SC-PPO 3.8` 这一行是 selected-checkpoint aggregate，不是 final-checkpoint-only
summary。

这里的正确解读是：

- 原始 `Vanilla PPO` 在当前共享指标下明显更粗糙、更不稳定。
- 调好的 heuristic baseline 相比原始 PPO 已经明显更强，所以它是有效的强锚点。
- 修复后的 PID-Lagrangian `SC-PPO 3.8` 主线，在当前 Isaac 侧共享比较指标上已经整体优于
  这个强启发式锚点，并且 collapse 明显更少。

这就是当前仓库里最强的主结果。

![Figure 1. Isaac rough-terrain main result](artifacts/analysis/sc_ppo_report_figures/figure_isaac_main_result.png)

图 1. 当前共享指标下的 Isaac 主结果图。`SC-PPO 3.8` 使用 seeds `11/17/23` 的
selected-checkpoint aggregate。

## 4. `MuJoCo isaac_mainline` First Pass

当前 `MuJoCo` 结论必须弱于 Isaac 主结果，因为它基于 representative checkpoint，而不是多种子聚合：

- heuristic anchor: `checkpoint 200`
- `SC-PPO 3.8`: `seed11`, `checkpoint 300`
- protocol: `terrain_mode = isaac_mainline`, `joint_reset_noise = 0.1`, `20 episodes`,
  `20 seconds`

### 表 2. `MuJoCo isaac_mainline` representative first-pass 对比

| Method | Evidence scope | `velocity_tracking_error_mean` | `joint_acceleration_l2_mean` | `action_jitter_l2_mean` | `fall_rate` | `episode_steps_mean` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `PPO + heuristic smoothing (action_rate=-0.0050)` | representative checkpoint, `20 episodes` | `0.6811 +/- 0.1113` | `110.2715 +/- 13.0420` | `0.2005 +/- 0.0158` | `0.7000` | `962.90` |
| `SC-PPO 3.8` | representative checkpoint (`seed11`, `ckpt 300`), `20 episodes` | `0.6206 +/- 0.0458` | `154.4672 +/- 12.0365` | `0.2785 +/- 0.0150` | `0.0500` | `1954.35` |

这里必须严格按当前证据说话：

- `SC-PPO 3.8` 在这个 first pass replay 上，任务稳定性和 survival 明显更强。
- `SC-PPO 3.8` 在 `velocity_tracking_error_mean` 上也更强。
- 但当前行为层平滑指标仍然是 heuristic anchor 更好，因为
  `joint_acceleration_l2_mean` 和 `action_jitter_l2_mean` 都更低。

所以当前正确口径只能是 `partial transfer`，不能写成完整的跨引擎平滑性胜利。

标题级安全表述：

`在当前 Isaac 粗糙平面主实验中，修复后的 PID-Lagrangian SC-PPO（threshold = 3.8）已经形成对已选 heuristic baseline 的主结果；在 MuJoCo isaac_mainline first pass 中，该方法显示出任务稳定性与速度跟踪的有边界 partial transfer，但当前行为层平滑指标尚未完成跨引擎转优。`

![Figure 2. MuJoCo isaac_mainline representative first pass](artifacts/analysis/sc_ppo_report_figures/figure_mujoco_first_pass.png)

图 2. `MuJoCo isaac_mainline` 的 representative first-pass replay。它的证据强度故意弱于
Isaac 主结果，因为这里不是 matched multi-seed aggregate。

## 5. 为什么当前正式主线仍然是 `3.8`

`3.6 + full_batch` 这条线的定位很窄：它只是用来挑战当前 `3.8` 主线，不是重新对 heuristic
做一轮全新的 repo-wide claim。

当前升格规则是：

- 复用 `seed11`
- 新增 `seed17` 和 `seed23`
- 每个 seed 都必须先通过 Isaac-side hard gate
- pathological early or null checkpoint selection 直接判失败

### 表 3. `3.6 + full_batch` Isaac 升格结果

| Seed | Selected checkpoint | `velocity_tracking_error_mean` | `joint_acceleration_l2_mean` | `action_jitter_l2_mean` | `fall_rate` | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `11` | `350` | `0.5735` | `116.0164` | `0.2107` | `0.1000` | 局部很强 |
| `17` | `350` | `0.6770` | `129.7265` | `0.2289` | `0.6500` | 明显不稳, 但非空 |
| `23` | `0` | `1.2863` | `80.8770` | `0.0115` | `1.0000` | pathological null selection |

当前结论非常明确：

- `seed11` 看起来很有希望；
- `seed17` 已经明显更弱；
- `seed23` 直接选到 `checkpoint 0`，触发仓库的 early-checkpoint failure rule。

因此这条线在 Isaac 阶段就停止，不能再消费新的 `MuJoCo isaac_mainline` 预算。
`3.6 + full_batch` 应回读为已完成的 `诊断支线`，而不是新的正式主线。

![Figure 3. Failed promotion of 3.6 plus full batch](artifacts/analysis/sc_ppo_report_figures/figure_threshold36_promotion_failure.png)

图 3. 正式 `3.6 + full_batch` 升格线在 Isaac 阶段失败，核心原因是 `seed23`
选到了 `checkpoint 0`。

## 6. 当前已成立的结果与剩余边界

当前已成立：

- 修复后的 PID-Lagrangian `SC-PPO` 且 `threshold = 3.8` 是当前正式主线。
- 在 Isaac `粗糙平面` 主实验上，这条线已经支持可辩护的 `method beats heuristic` 结论。
- 这个主结果经过了 `3 seeds`，并且显式依赖 checkpoint sweep selection。
- 最近的更紧候选线 `3.6 + full_batch` 没能取代当前主线。
- `MuJoCo isaac_mainline` 已经支持一个有边界的 external-validation 结论，重点在任务稳定性与速度跟踪。

当前还不能成立：

- final checkpoint 单独就足够支撑当前 `SC-PPO` 主线
- 平滑性优势已经完整迁移到 `MuJoCo`
- `hfield_moderate` 或 `hfield_stress` 已经可以当作 report-grade terrain 结果
- 一整片更紧 threshold 邻域都能等价替代当前 `3.8` 主线

## 7. Canonical Artifacts

Isaac 主结果:

- `artifacts/methods/vanilla_ppo/vanilla_ppo_rough_terrain/metrics.json`
- `artifacts/analysis/heuristic_action_rate_rough_terrain/selection.json`
- `artifacts/methods/heuristic_smoothing_sweep/heuristic_smoothing_action_rate_0050_rough_terrain/metrics.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_selected.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17/metrics_selected.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23/metrics_selected.json`

`MuJoCo isaac_mainline` first pass:

- `artifacts/methods/heuristic_smoothing_sweep/heuristic_smoothing_action_rate_0050_rough_terrain/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`
- `artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json`

升格失败候选线:

- `artifacts/methods/sc_ppo_fullbatch_threshold_probe/sc_ppo_fullbatch_threshold_36_iter400_seed11/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_fullbatch_threshold_probe/sc_ppo_fullbatch_threshold_36_iter400_seed17/checkpoint_sweep_summary.json`
- `artifacts/methods/sc_ppo_fullbatch_threshold_probe/sc_ppo_fullbatch_threshold_36_iter400_seed23/checkpoint_sweep_summary.json`

生成图表:

- `scripts/analysis/generate_sc_ppo_report_figures.py`
- `artifacts/analysis/sc_ppo_report_figures/figure_isaac_main_result.png`
- `artifacts/analysis/sc_ppo_report_figures/figure_mujoco_first_pass.png`
- `artifacts/analysis/sc_ppo_report_figures/figure_threshold36_promotion_failure.png`
- `artifacts/analysis/sc_ppo_report_figures/manifest.json`
