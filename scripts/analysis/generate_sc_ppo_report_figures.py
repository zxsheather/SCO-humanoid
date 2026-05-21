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

HEURISTIC_MUJOCO_PATHS = [
    REPO_ROOT
    / "artifacts"
    / "methods"
    / "heuristic_smoothing_formal_protocol_revision_long_budget"
    / f"heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed{seed}"
    / "metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"
    for seed in (11, 17, 23)
]
SC_MUJOCO_PATHS = [
    REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_pid_probe"
    / f"sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed{seed}"
    / "metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"
    for seed in (11, 17, 23)
]

ISAAC_ROWS = {
    "Vanilla PPO\nraw reference": {
        "velocity_tracking_error_mean": 1.3321,
        "joint_acceleration_l2_mean": 83.7179,
        "action_jitter_l2_mean": 0.0161,
        "fall_rate": 1.0,
    },
    "Revised\nheuristic": {
        "velocity_tracking_error_mean": 0.7549,
        "joint_acceleration_l2_mean": 119.8639,
        "action_jitter_l2_mean": 0.2711,
        "fall_rate": 0.15,
    },
    "SC-PPO 3.8": {
        "velocity_tracking_error_mean": 0.6412,
        "joint_acceleration_l2_mean": 115.9079,
        "action_jitter_l2_mean": 0.2205,
        "fall_rate": 0.1,
    },
}

PROMOTION_ROWS = {
    11: {"selected_checkpoint": 350, "fall_rate": 0.1},
    17: {"selected_checkpoint": 350, "fall_rate": 0.65},
    23: {"selected_checkpoint": 0, "fall_rate": 1.0},
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


def aggregate_json_metrics(paths: list[Path]) -> dict[str, float]:
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(read_json(path))

    aggregate: dict[str, float] = {}
    for key in [
        "velocity_tracking_error_mean",
        "joint_acceleration_l2_mean",
        "action_jitter_l2_mean",
        "fall_rate",
        "episode_return_mean",
    ]:
        values = [float(row[key]) for row in rows]
        aggregate[f"{key}_mean"] = mean(values)
        aggregate[f"{key}_std"] = pstdev(values)

    step_values = [float(row["mujoco_eval"]["episode_steps_mean"]) for row in rows]
    aggregate["episode_steps_mean_mean"] = mean(step_values)
    aggregate["episode_steps_mean_std"] = pstdev(step_values)
    return aggregate


def build_isaac_main_figure(output_dir: Path) -> Path:
    methods = list(ISAAC_ROWS)
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
        values = [float(ISAAC_ROWS[method][key]) for method in methods]
        bars = ax.bar(methods, values, color=colors, width=0.65)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.tick_params(axis="x", labelrotation=0)
        ymax = max(values) * 1.22 if max(values) > 0 else 1.0
        ax.set_ylim(0.0, ymax)
        annotate_bars(ax, bars, values)

    fig.suptitle(
        "Isaac rough-terrain main comparison\n"
        "Vanilla raw reference, revised heuristic anchor, and SC-PPO 3.8",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "SC-PPO 3.8 and revised heuristic use 3-seed selected-checkpoint aggregates. "
        "Vanilla PPO records the raw-reference collapse row.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.92])

    output_path = output_dir / "figure_isaac_main_result.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def build_mujoco_aligned_figure(output_dir: Path) -> Path:
    heuristic = aggregate_json_metrics(HEURISTIC_MUJOCO_PATHS)
    sc = aggregate_json_metrics(SC_MUJOCO_PATHS)
    metrics = [
        ("velocity_tracking_error_mean_mean", "Velocity Tracking Error"),
        ("joint_acceleration_l2_mean_mean", "Joint Acceleration L2"),
        ("action_jitter_l2_mean_mean", "Action Jitter L2"),
        ("fall_rate_mean", "Fall Rate"),
        ("episode_steps_mean_mean", "Episode Steps Mean"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(15, 4.8))
    labels = ["Revised\nheuristic", "SC-PPO 3.8"]
    colors = [COLORS["heuristic"], COLORS["scppo"]]

    for ax, (key, title) in zip(axes.flat, metrics):
        values = [float(heuristic[key]), float(sc[key])]
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=0)
        ymax = max(values) * 1.22 if max(values) > 0 else 1.0
        ax.set_ylim(0.0, ymax)
        annotate_bars(ax, bars, values)

    fig.suptitle(
        "MuJoCo isaac_mainline aligned replay\n"
        "3-seed selected-checkpoint comparison against revised heuristic anchor",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "This aligned replay is mixed: revised heuristic is stronger on task-side metrics; "
        "SC-PPO 3.8 is slightly better on action jitter.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.01, 0.05, 0.99, 0.90])

    output_path = output_dir / "figure_mujoco_aligned_replay.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def build_promotion_failure_figure(output_dir: Path) -> Path:
    seeds = [11, 17, 23]
    seed_labels = [f"seed{seed}" for seed in seeds]
    selected_checkpoints = [PROMOTION_ROWS[seed]["selected_checkpoint"] for seed in seeds]
    fall_rates = [PROMOTION_ROWS[seed]["fall_rate"] for seed in seeds]
    checkpoint_colors = [COLORS["scppo"], COLORS["scppo"], COLORS["warning"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

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
    annotate_bars(ax1, bars1, [float(v) for v in fall_rates], extra_y=0.03)

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
            "heuristic_mujoco": [str(path.relative_to(REPO_ROOT)) for path in HEURISTIC_MUJOCO_PATHS],
            "sc_mujoco": [str(path.relative_to(REPO_ROOT)) for path in SC_MUJOCO_PATHS],
            "isaac_rows_source": "docs/sc-ppo-report-status.md",
            "promotion_rows_source": "docs/sc-ppo-report-status.md",
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
        "mujoco_aligned_replay": build_mujoco_aligned_figure(output_dir),
        "threshold36_promotion_failure": build_promotion_failure_figure(output_dir),
    }
    manifest_path = write_manifest(output_dir, outputs)

    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
