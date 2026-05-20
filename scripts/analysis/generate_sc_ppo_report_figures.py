#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "analysis" / "sc_ppo_report_figures"

VANILLA_METRICS = REPO_ROOT / "artifacts" / "methods" / "vanilla_ppo" / "vanilla_ppo_rough_terrain" / "metrics.json"
HEURISTIC_SELECTION = REPO_ROOT / "artifacts" / "analysis" / "heuristic_action_rate_rough_terrain" / "selection.json"
SC_MAINLINE_SEED_METRICS = {
    11: REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_pid_probe"
    / "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11"
    / "metrics_selected.json",
    17: REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_pid_probe"
    / "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17"
    / "metrics_selected.json",
    23: REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_pid_probe"
    / "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23"
    / "metrics_selected.json",
}
SC_MAINLINE_SEED_SWEEPS = {
    11: REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_pid_probe"
    / "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11"
    / "checkpoint_sweep_summary.json",
    17: REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_pid_probe"
    / "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17"
    / "checkpoint_sweep_summary.json",
    23: REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_pid_probe"
    / "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23"
    / "checkpoint_sweep_summary.json",
}
SC_MUJOCO_REPRESENTATIVE = (
    REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_pid_probe"
    / "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11"
    / "metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"
)
PROMOTION_SWEEPS = {
    11: REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_fullbatch_threshold_probe"
    / "sc_ppo_fullbatch_threshold_36_iter400_seed11"
    / "checkpoint_sweep_summary.json",
    17: REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_fullbatch_threshold_probe"
    / "sc_ppo_fullbatch_threshold_36_iter400_seed17"
    / "checkpoint_sweep_summary.json",
    23: REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_fullbatch_threshold_probe"
    / "sc_ppo_fullbatch_threshold_36_iter400_seed23"
    / "checkpoint_sweep_summary.json",
}

COLORS = {
    "vanilla": "#7f8c8d",
    "heuristic": "#4c78a8",
    "scppo": "#54a24b",
    "warning": "#e45756",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def annotate_bars(ax: plt.Axes, bars: Any, values: list[float], extra_y: float = 0.02) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * extra_y
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            format_value(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def load_heuristic_metrics() -> tuple[dict[str, Any], dict[str, Any], Path]:
    selection = read_json(HEURISTIC_SELECTION)
    selected_candidate = selection.get("selected_candidate")
    if not isinstance(selected_candidate, dict):
        raise RuntimeError(
            "Heuristic selection does not contain a valid selected_candidate; "
            f"selection_status={selection.get('selection_status')!r}"
        )
    metrics_rel = selected_candidate["metrics_path"]
    metrics_path = REPO_ROOT / metrics_rel
    return selection, read_json(metrics_path), metrics_path


def aggregate_sc_mainline() -> tuple[dict[str, float], dict[int, int]]:
    seed_metrics = {seed: read_json(path) for seed, path in SC_MAINLINE_SEED_METRICS.items()}
    seed_sweeps = {seed: read_json(path) for seed, path in SC_MAINLINE_SEED_SWEEPS.items()}
    metric_keys = [
        "velocity_tracking_error_mean",
        "joint_acceleration_l2_mean",
        "action_jitter_l2_mean",
        "episode_return_mean",
        "fall_rate",
    ]
    aggregate: dict[str, float] = {}
    for key in metric_keys:
        values = [float(seed_metrics[seed][key]) for seed in sorted(seed_metrics)]
        aggregate[f"{key}_mean"] = mean(values)
        aggregate[f"{key}_std"] = pstdev(values)
    selected_checkpoints = {
        seed: int(seed_sweeps[seed]["best_checkpoint"]) for seed in sorted(seed_sweeps)
    }
    return aggregate, selected_checkpoints


def build_isaac_main_figure(output_dir: Path) -> Path:
    vanilla = read_json(VANILLA_METRICS)
    _, heuristic, _ = load_heuristic_metrics()
    sc_aggregate, selected_checkpoints = aggregate_sc_mainline()

    methods = ["Vanilla PPO", "Heuristic", "SC-PPO 3.8"]
    colors = [COLORS["vanilla"], COLORS["heuristic"], COLORS["scppo"]]
    metrics = [
        ("velocity_tracking_error_mean", "Velocity Tracking Error", "lower is better"),
        ("joint_acceleration_l2_mean", "Joint Acceleration L2", "lower is better"),
        ("action_jitter_l2_mean", "Action Jitter L2", "lower is better"),
        ("fall_rate", "Fall Rate", "lower is better"),
    ]

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title, subtitle) in zip(axes.flat, metrics):
        values = [
            float(vanilla[key]),
            float(heuristic[key]),
            float(sc_aggregate[f"{key}_mean"]),
        ]
        bars = ax.bar(methods, values, color=colors, width=0.65)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.tick_params(axis="x", labelrotation=12)
        ymax = max(values) * 1.22 if max(values) > 0 else 1.0
        ax.set_ylim(0.0, ymax)
        annotate_bars(ax, bars, values)

    fig.suptitle(
        "Isaac rough-terrain main result\n"
        "SC-PPO 3.8 uses 3-seed selected-checkpoint aggregate (checkpoints 300 / 300 / 400)",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Vanilla PPO and heuristic anchor are single-run shared-protocol results. "
        "SC-PPO aggregate is computed from seed11 / seed17 / seed23 selected checkpoints.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.92])

    output_path = output_dir / "figure_isaac_main_result.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def build_mujoco_first_pass_figure(output_dir: Path) -> Path:
    selection, _, heuristic_metrics_path = load_heuristic_metrics()
    heuristic_mujoco = read_json(heuristic_metrics_path.with_name("metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"))
    sc_mujoco = read_json(SC_MUJOCO_REPRESENTATIVE)

    metrics = [
        ("velocity_tracking_error_mean", "Velocity Tracking Error"),
        ("joint_acceleration_l2_mean", "Joint Acceleration L2"),
        ("action_jitter_l2_mean", "Action Jitter L2"),
        ("fall_rate", "Fall Rate"),
        ("mujoco_eval.episode_steps_mean", "Episode Steps Mean"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(15, 4.8))
    labels = ["Heuristic", "SC-PPO 3.8"]
    colors = [COLORS["heuristic"], COLORS["scppo"]]

    for ax, (key, title) in zip(axes.flat, metrics):
        if key == "mujoco_eval.episode_steps_mean":
            heuristic_value = float(heuristic_mujoco["mujoco_eval"]["episode_steps_mean"])
            sc_value = float(sc_mujoco["mujoco_eval"]["episode_steps_mean"])
        else:
            heuristic_value = float(heuristic_mujoco[key])
            sc_value = float(sc_mujoco[key])
        values = [heuristic_value, sc_value]
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=12)
        ymax = max(values) * 1.22 if max(values) > 0 else 1.0
        ax.set_ylim(0.0, ymax)
        annotate_bars(ax, bars, values)

    fig.suptitle(
        "MuJoCo isaac_mainline representative first pass\n"
        "Heuristic checkpoint 200 vs SC-PPO seed11 checkpoint 300",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "This figure is a representative first-pass replay, not a matched multi-seed aggregate.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.01, 0.05, 0.99, 0.90])

    output_path = output_dir / "figure_mujoco_first_pass.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def build_promotion_failure_figure(output_dir: Path) -> Path:
    summaries = {seed: read_json(path) for seed, path in PROMOTION_SWEEPS.items()}
    seeds = [11, 17, 23]
    selected_checkpoints = [int(summaries[seed]["best_checkpoint"]) for seed in seeds]
    selected_rows = []
    for seed in seeds:
        summary = summaries[seed]
        best_checkpoint = int(summary["best_checkpoint"])
        row = next(item for item in summary["rows"] if int(item["checkpoint"]) == best_checkpoint)
        selected_rows.append(row)
    fall_rates = [float(row["fall_rate"]) for row in selected_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    seed_labels = [f"seed{seed}" for seed in seeds]
    checkpoint_colors = [COLORS["scppo"], COLORS["scppo"], COLORS["warning"]]

    ax0 = axes[0]
    bars0 = ax0.bar(seed_labels, selected_checkpoints, color=checkpoint_colors, width=0.62)
    ax0.axhspan(-5, 100, color=COLORS["warning"], alpha=0.12)
    ax0.text(1.45, 45, "early/null failure zone", color=COLORS["warning"], fontsize=9, ha="right")
    ax0.set_title("Selected checkpoint", fontsize=11)
    ax0.set_ylim(0, 420)
    annotate_bars(ax0, bars0, [float(v) for v in selected_checkpoints], extra_y=0.01)

    ax1 = axes[1]
    bars1 = ax1.bar(seed_labels, fall_rates, color=checkpoint_colors, width=0.62)
    ax1.set_title("Selected-checkpoint fall_rate", fontsize=11)
    ax1.set_ylim(0.0, 1.1)
    annotate_bars(ax1, bars1, fall_rates, extra_y=0.03)

    fig.suptitle(
        "3.6 + full_batch promotion outcome\n"
        "seed23 selects checkpoint 0 and fails the Isaac-side promotion gate",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Current selected checkpoints: seed11 -> 350, seed17 -> 350, seed23 -> 0.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.90])

    output_path = output_dir / "figure_threshold36_promotion_failure.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def write_manifest(output_dir: Path, outputs: dict[str, Path]) -> Path:
    payload = {
        "figure_paths": {name: str(path.relative_to(REPO_ROOT)) for name, path in outputs.items()},
        "source_artifacts": {
            "vanilla_metrics": str(VANILLA_METRICS.relative_to(REPO_ROOT)),
            "heuristic_selection": str(HEURISTIC_SELECTION.relative_to(REPO_ROOT)),
            "sc_mainline_seed_metrics": {
                str(seed): str(path.relative_to(REPO_ROOT))
                for seed, path in SC_MAINLINE_SEED_METRICS.items()
            },
            "sc_mainline_seed_sweeps": {
                str(seed): str(path.relative_to(REPO_ROOT))
                for seed, path in SC_MAINLINE_SEED_SWEEPS.items()
            },
            "sc_mujoco_representative": str(SC_MUJOCO_REPRESENTATIVE.relative_to(REPO_ROOT)),
            "promotion_sweeps": {
                str(seed): str(path.relative_to(REPO_ROOT))
                for seed, path in PROMOTION_SWEEPS.items()
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate report-grade static figures for the SC-PPO result draft.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated figures and manifest.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir).resolve())

    outputs = {
        "isaac_main_result": build_isaac_main_figure(output_dir),
        "mujoco_first_pass": build_mujoco_first_pass_figure(output_dir),
        "threshold36_promotion_failure": build_promotion_failure_figure(output_dir),
    }
    manifest_path = write_manifest(output_dir, outputs)

    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
