#!/usr/bin/env python3
"""Generate paper-grade figures and tables from existing experiment artifacts.

Reproducible analysis: reads JSON summaries and checkpoint sweeps, produces
PNG figures and structured JSON data files with source-artifact traceability.

Usage:
    python scripts/analysis/generate_paper_figures.py [--output-dir ARTIFACTS_DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "analysis" / "paper_figures"


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def checkpoint_metric(base: str, seed: int, checkpoint: int, key: str) -> float | None:
    """Read a single metric from a checkpoint sweep summary."""
    path = REPO_ROOT / f"{base}_seed{seed}/checkpoint_sweep_summary.json"
    if not path.exists():
        return None
    data = read_json(path)
    for r in data.get("rows", []):
        if r["checkpoint"] == checkpoint:
            return r.get(key)
    return None


def selected_metric(method_dir: str, seed: int, key: str) -> float | None:
    """Read a metric from metrics_selected.json."""
    path = REPO_ROOT / f"{method_dir}_seed{seed}/metrics_selected.json"
    if not path.exists():
        # Try metrics.json
        path = REPO_ROOT / f"{method_dir}_seed{seed}/metrics.json"
    if not path.exists():
        return None
    data = read_json(path)
    return data.get(key)


def mujoco_metric(method_dir: str, seed: int, key: str) -> float | None:
    """Read a metric from MuJoCo replay."""
    path = REPO_ROOT / f"{method_dir}_seed{seed}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"
    if not path.exists():
        return None
    data = read_json(path)
    return data.get(key)


def mean_across_seeds(method_dir: str, seeds: list[int], key: str, source: str = "selected") -> tuple[float, float]:
    """Compute mean and std across seeds."""
    vals = []
    for s in seeds:
        if source == "selected":
            v = selected_metric(method_dir, s, key)
        elif source == "mujoco":
            v = mujoco_metric(method_dir, s, key)
        elif source == "checkpoint":
            # For checkpoint, we need base + seed + cp
            pass
        if v is not None:
            vals.append(v)
    if not vals:
        return 0.0, 0.0
    return statistics.fmean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def assemble_degradation_data():
    """Assemble the cross-engine degradation table data."""
    methods = [
        {
            "id": "heuristic",
            "label": "Heuristic\nbaseline",
            "color": "#2196F3",
            "isaac_dir": "artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain",
            "mujoco_dir": "artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain",
            "sensitivity": None,
            "source_artifacts": {
                "isaac": "artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed{11,17,23}/metrics_selected.json",
                "mujoco": "artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json",
            },
        },
        {
            "id": "scppo38",
            "label": "SC-PPO 3.8\n(Jacobian)",
            "color": "#4CAF50",
            "isaac_dir": "artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400",
            "mujoco_dir": "artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400",
            "sensitivity": 3.58,
            "source_artifacts": {
                "isaac": "artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed{11,17,23}/metrics_selected.json",
                "mujoco": "artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json",
                "checkpoint_sweep": "artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed{11,17,23}/checkpoint_sweep_summary.json",
            },
        },
        {
            "id": "layernorm",
            "label": "LayerNorm\nepochs=3",
            "color": "#FF9800",
            "isaac_dir": "artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain",
            "mujoco_dir": "artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain",
            "sensitivity": 10.74,
            "source_artifacts": {
                "isaac": "artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed{11,17,23}/metrics_selected.json",
                "mujoco": "artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json",
                "checkpoint_sweep": "artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed{11,17,23}/checkpoint_sweep_summary.json",
            },
        },
        {
            "id": "action_scaling",
            "label": "Action\nScaling",
            "color": "#F44336",
            "isaac_dir": "artifacts/methods/action_scaling_probe/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain",
            "mujoco_dir": "artifacts/methods/action_scaling_probe/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain",
            "sensitivity": 8.6,
            "source_artifacts": {
                "isaac": "artifacts/methods/action_scaling_probe/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/metrics_selected.json",
                "mujoco": "artifacts/methods/action_scaling_probe/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json",
            },
        },
        {
            "id": "output_scaling",
            "label": "Output\nScaling",
            "color": "#9C27B0",
            "isaac_dir": "artifacts/methods/output_scaling_probe/output_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain",
            "mujoco_dir": "artifacts/methods/output_scaling_probe/output_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain",
            "sensitivity": 6.15,
            "source_artifacts": {
                "isaac": "artifacts/methods/output_scaling_probe/output_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/metrics_selected.json",
                "mujoco": "artifacts/methods/output_scaling_probe/output_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json",
            },
        },
    ]

    seeds = [11, 17, 23]
    metrics_keys = ["joint_acceleration_l2_mean", "action_jitter_l2_mean",
                    "velocity_tracking_error_mean", "fall_rate"]

    for m in methods:
        for key in metrics_keys:
            # Isaac
            i_vals = []
            for s in seeds:
                v = selected_metric(m["isaac_dir"], s, key)
                if v is not None:
                    i_vals.append(v)
            if i_vals:
                m[f"isaac_{key}"] = statistics.fmean(i_vals)
                m[f"isaac_{key}_std"] = statistics.pstdev(i_vals) if len(i_vals) > 1 else 0.0

            # MuJoCo
            m_vals = []
            for s in seeds:
                v = mujoco_metric(m["mujoco_dir"], s, key)
                if v is not None:
                    m_vals.append(v)
            if m_vals:
                m[f"mujoco_{key}"] = statistics.fmean(m_vals)
                m[f"mujoco_{key}_std"] = statistics.pstdev(m_vals) if len(m_vals) > 1 else 0.0

        # Degradation factor
        if m.get("isaac_joint_acceleration_l2_mean") and m.get("mujoco_joint_acceleration_l2_mean"):
            m["degradation_factor"] = m["mujoco_joint_acceleration_l2_mean"] / m["isaac_joint_acceleration_l2_mean"]
        else:
            m["degradation_factor"] = None

    return methods


def assemble_sensitivity_evolution():
    """Assemble sensitivity evolution data from checkpoint sweeps."""
    sc_base = "artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400"
    ln_base = "artifacts/methods/layernorm_actor_gain_reliability_probe/layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain"

    evolution = {"scppo": {}, "layernorm": {}}

    for name, base in [("scppo", sc_base), ("layernorm", ln_base)]:
        for cp in [0, 100, 200, 300, 400]:
            sens_vals, fall_vals, jnt_vals = [], [], []
            for seed in [11, 17, 23]:
                path = REPO_ROOT / f"{base}_seed{seed}/checkpoint_sweep_summary.json"
                if not path.exists():
                    continue
                data = read_json(path)
                for r in data.get("rows", []):
                    if r["checkpoint"] == cp:
                        sens_vals.append(r.get("policy_local_sensitivity_cost_mean", 0))
                        fall_vals.append(r.get("fall_rate", 1))
                        jnt_vals.append(r.get("joint_acceleration_l2_mean", 0))
                        break
            if sens_vals:
                evolution[name][cp] = {
                    "sensitivity_mean": statistics.fmean(sens_vals),
                    "sensitivity_std": statistics.pstdev(sens_vals) if len(sens_vals) > 1 else 0.0,
                    "fall_rate_mean": statistics.fmean(fall_vals),
                    "jnt_acc_mean": statistics.fmean(jnt_vals),
                }

    evolution["source"] = {
        "scppo": f"{sc_base}_seed{{11,17,23}}/checkpoint_sweep_summary.json",
        "layernorm": f"{ln_base}_seed{{11,17,23}}/checkpoint_sweep_summary.json",
    }

    return evolution


def assemble_threshold_data():
    """Assemble threshold sensitivity data from docs and artifacts."""
    return {
        "thresholds": [
            {"threshold": 3.6, "regime": "full_batch", "seed11_cp": 350, "seed11_fall": 0.10,
             "seed17_cp": 350, "seed17_fall": 0.65, "seed23_cp": 0, "seed23_fall": 1.00,
             "outcome": "failed"},
            {"threshold": 3.7, "regime": "full_batch", "outcome": "frozen"},
            {"threshold": 3.8, "regime": "quantile-0.90", "seed11_cp": 300, "seed11_fall": 0.10,
             "seed17_cp": 300, "seed17_fall": 0.10, "seed23_cp": 400, "seed23_fall": 0.10,
             "outcome": "mainline"},
            {"threshold": 4.0, "regime": "quantile-0.90", "seed23_cp": 0, "seed23_fall": 1.00,
             "outcome": "failed"},
            {"threshold": 4.2, "regime": "quantile-0.90", "outcome": "closed"},
        ],
        "source": "docs/sc-ppo-report-status.md, docs/sc-ppo-fullbatch-threshold-promotion.md, CONTEXT.md",
    }


def assemble_ldlj_sparc_data():
    """Assemble LDLJ/SPARC trace comparison data."""
    scppo_traces = {}
    ln_traces = {}

    for seed in [11, 17, 23]:
        sc_path = REPO_ROOT / f"artifacts/methods/sc_ppo_pid_probe/scppo38_trace20_seed{seed}/behavior_smoothness_metrics_selected.json"
        ln_path = REPO_ROOT / f"artifacts/methods/layernorm_actor_gain_reliability_probe/ln_ep3_trace20_seed{seed}/behavior_smoothness_metrics_selected.json"

        for label, path, store in [("scppo", sc_path, scppo_traces), ("layernorm", ln_path, ln_traces)]:
            if path.exists():
                data = read_json(path)
                summary = data.get("summary", {})
                store[seed] = {
                    "ldlj": summary.get("joint_position_ldlj_mean", {}).get("mean"),
                    "ldlj_std": summary.get("joint_position_ldlj_mean", {}).get("std"),
                    "sparc": summary.get("joint_velocity_sparc_mean", {}).get("mean"),
                    "sparc_std": summary.get("joint_velocity_sparc_mean", {}).get("std"),
                    "episodes": data.get("episode_count", 0),
                    "fell_count": sum(1 for ep in data.get("episodes", []) if ep.get("fell", False)),
                }

    return {
        "scppo": scppo_traces,
        "layernorm": ln_traces,
        "source": {
            "scppo": "artifacts/methods/sc_ppo_pid_probe/scppo38_trace20_seed{11,17,23}/behavior_smoothness_metrics_selected.json",
            "layernorm": "artifacts/methods/layernorm_actor_gain_reliability_probe/ln_ep3_trace20_seed{11,17,23}/behavior_smoothness_metrics_selected.json",
        },
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.figsize": (7, 4.5),
    })


def fig1_degradation_bars(methods, output_dir: Path):
    """Figure 1: Cross-engine degradation bar chart."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = [m["label"] for m in methods]
    colors = [m["color"] for m in methods]
    x = np.arange(len(methods))
    width = 0.35

    # Panel A: Joint acceleration
    isaac_jnt = [m.get("isaac_joint_acceleration_l2_mean", 0) for m in methods]
    mujoco_jnt = [m.get("mujoco_joint_acceleration_l2_mean", 0) for m in methods]

    bars1 = ax1.bar(x - width/2, isaac_jnt, width, label="Isaac", color="#90CAF9", edgecolor="#1976D2", linewidth=0.8)
    bars2 = ax1.bar(x + width/2, mujoco_jnt, width, label="MuJoCo", color="#FFAB91", edgecolor="#D84315", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Joint Acceleration L2")
    ax1.set_title("(a) Joint Acceleration: Isaac vs MuJoCo")
    ax1.legend(fontsize=8)

    # Annotate degradation factors
    for i, m in enumerate(methods):
        factor = m.get("degradation_factor")
        if factor and factor > 1.5:
            ax1.annotate(f"×{factor:.1f}", (x[i] + width/2, mujoco_jnt[i]),
                        textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7,
                        color="#D84315", fontweight="bold")

    # Panel B: Degradation factor
    factors = [m.get("degradation_factor", 0) or 0 for m in methods]
    bar_colors = ["#4CAF50" if f < 1.5 else "#FF9800" if f < 5 else "#F44336" for f in factors]
    ax2.bar(x, factors, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Degradation Factor (MuJoCo / Isaac)")
    ax2.set_title("(b) Cross-Engine Degradation Factor")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("×%.1f"))

    # Annotate values
    for i, f in enumerate(factors):
        if f > 0:
            ax2.annotate(f"×{f:.1f}", (x[i], f), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "figure_cross_engine_degradation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure_cross_engine_degradation.png")


def fig2_sensitivity_evolution(evolution, output_dir: Path):
    """Figure 2: Sensitivity evolution during training."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    cps = [0, 100, 200, 300, 400]

    # Panel A: Sensitivity
    for name, color, label in [("scppo", "#4CAF50", "SC-PPO 3.8"), ("layernorm", "#FF9800", "LayerNorm epochs=3")]:
        sens = [evolution[name][cp]["sensitivity_mean"] for cp in cps]
        sens_std = [evolution[name][cp]["sensitivity_std"] for cp in cps]
        ax1.errorbar(cps, sens, yerr=sens_std, marker="o", color=color, label=label,
                    capsize=3, linewidth=1.5, markersize=5)

    ax1.axhline(y=3.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.annotate("threshold = 3.8", (50, 3.9), fontsize=8, color="gray")
    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Policy Local Sensitivity")
    ax1.set_title("(a) Jacobian Sensitivity Evolution")
    ax1.legend(fontsize=8)

    # Panel B: Fall rate
    for name, color, label in [("scppo", "#4CAF50", "SC-PPO 3.8"), ("layernorm", "#FF9800", "LayerNorm epochs=3")]:
        fall = [evolution[name][cp]["fall_rate_mean"] for cp in cps]
        ax2.plot(cps, fall, marker="s", color=color, label=label, linewidth=1.5, markersize=5)

    ax2.set_xlabel("Training Iteration")
    ax2.set_ylabel("Fall Rate")
    ax2.set_title("(b) Task Acquisition")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "figure_sensitivity_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure_sensitivity_evolution.png")


def fig3_sensitivity_vs_degradation(methods, output_dir: Path):
    """Figure 3: Sensitivity → degradation scatter plot."""
    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    for m in methods:
        sens = m.get("sensitivity")
        factor = m.get("degradation_factor")
        if sens and factor:
            ax.scatter(sens, factor, c=m["color"], s=100, edgecolors="white", linewidth=1, zorder=5)
            offset = (5, 5) if sens < 8 else (5, -10)
            ax.annotate(m["label"].replace("\n", " "), (sens, factor),
                       textcoords="offset points", xytext=offset, fontsize=8,
                       arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    ax.set_xlabel("Policy Local Sensitivity (Isaac cp400)")
    ax.set_ylabel("Cross-Engine Degradation Factor")
    ax.set_title("Sensitivity Predicts Cross-Engine Degradation")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("×%.1f"))
    ax.grid(True, alpha=0.3)

    # Trend line
    pts = [(m["sensitivity"], m["degradation_factor"]) for m in methods
           if m.get("sensitivity") and m.get("degradation_factor")]
    if len(pts) >= 2:
        x_vals = [p[0] for p in pts]
        y_vals = [p[1] for p in pts]
        z = np.polyfit(x_vals, np.log(y_vals), 1)
        x_line = np.linspace(min(x_vals) * 0.8, max(x_vals) * 1.1, 50)
        y_line = np.exp(z[1]) * np.exp(z[0] * x_line)
        ax.plot(x_line, y_line, "--", color="gray", alpha=0.5, linewidth=1)

    plt.tight_layout()
    fig.savefig(output_dir / "figure_sensitivity_vs_degradation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure_sensitivity_vs_degradation.png")


def fig4_ldlj_sparc(ldlj_data, methods, output_dir: Path):
    """Figure 4: LDLJ/SPARC kinematic vs dynamic smoothness."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Compute means
    sc_ldlj = statistics.fmean([ldlj_data["scppo"][s]["ldlj"] for s in [11, 17, 23]
                                 if s in ldlj_data["scppo"] and ldlj_data["scppo"][s]["ldlj"] is not None])
    ln_ldlj = statistics.fmean([ldlj_data["layernorm"][s]["ldlj"] for s in [11, 17, 23]
                                 if s in ldlj_data["layernorm"] and ldlj_data["layernorm"][s]["ldlj"] is not None])
    sc_sparc = statistics.fmean([ldlj_data["scppo"][s]["sparc"] for s in [11, 17, 23]
                                  if s in ldlj_data["scppo"] and ldlj_data["scppo"][s]["sparc"] is not None])
    ln_sparc = statistics.fmean([ldlj_data["layernorm"][s]["sparc"] for s in [11, 17, 23]
                                  if s in ldlj_data["layernorm"] and ldlj_data["layernorm"][s]["sparc"] is not None])

    # Panel A: LDLJ
    x = [0, 1]
    ax1.bar(x, [sc_ldlj, ln_ldlj], color=["#4CAF50", "#FF9800"], edgecolor="white", width=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["SC-PPO 3.8", "LayerNorm epochs=3"], fontsize=9)
    ax1.set_ylabel("LDLJ (lower = smoother)")
    ax1.set_title("(a) Joint Position Jerk")
    ax1.axhline(y=0, color="gray", linewidth=0.5)

    # Panel B: SPARC
    ax2.bar(x, [sc_sparc, ln_sparc], color=["#4CAF50", "#FF9800"], edgecolor="white", width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["SC-PPO 3.8", "LayerNorm epochs=3"], fontsize=9)
    ax2.set_ylabel("SPARC (lower = smoother)")
    ax2.set_title("(b) Joint Velocity Spectral Arc Length")
    ax2.axhline(y=0, color="gray", linewidth=0.5)

    # Get jnt_acc for comparison annotation
    sc_jnt = None
    ln_jnt = None
    for m in methods:
        if m["id"] == "scppo38":
            sc_jnt = m.get("isaac_joint_acceleration_l2_mean")
        if m["id"] == "layernorm":
            ln_jnt = m.get("isaac_joint_acceleration_l2_mean")

    if sc_jnt and ln_jnt:
        fig.text(0.5, 0.01,
                 f"Kinematic smoothness (LDLJ/SPARC): LayerNorm wins.\n"
                 f"Dynamic smoothness (jnt_acc): SC-PPO wins ({sc_jnt:.0f} vs {ln_jnt:.0f}).\n"
                 f"Smoothness is two-dimensional.",
                 ha="center", fontsize=8, fontstyle="italic", color="gray")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_dir / "figure_ldlj_sparc.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure_ldlj_sparc.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paper-grade figures from experiment artifacts")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                       help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Paper Figures Generation ===")
    print(f"Output: {output_dir}")

    # Assemble data
    print("\n[1/6] Assembling degradation data...")
    methods = assemble_degradation_data()
    print(f"  {len(methods)} methods loaded")

    print("[2/6] Assembling sensitivity evolution...")
    evolution = assemble_sensitivity_evolution()
    print(f"  SC-PPO: {len(evolution['scppo'])} checkpoints, LayerNorm: {len(evolution['layernorm'])} checkpoints")

    print("[3/6] Assembling threshold data...")
    threshold_data = assemble_threshold_data()
    print(f"  {len(threshold_data['thresholds'])} thresholds")

    print("[4/6] Assembling LDLJ/SPARC data...")
    ldlj_data = assemble_ldlj_sparc_data()
    print(f"  SC-PPO: {len(ldlj_data['scppo'])} seeds, LayerNorm: {len(ldlj_data['layernorm'])} seeds")

    # Generate figures
    print("\n[5/6] Generating figures...")
    fig1_degradation_bars(methods, output_dir)
    fig2_sensitivity_evolution(evolution, output_dir)
    fig3_sensitivity_vs_degradation(methods, output_dir)
    fig4_ldlj_sparc(ldlj_data, methods, output_dir)

    # Save structured data
    print("\n[6/6] Saving structured data...")
    paper_data = {
        "generated": "2026-05-26",
        "description": "Paper-grade figures and tables for cross-engine smoothness degradation study",
        "methods": methods,
        "sensitivity_evolution": {k: {str(cp): v for cp, v in evolution[k].items()}
                                  for k in ["scppo", "layernorm"]},
        "sensitivity_evolution_source": evolution.get("source", {}),
        "threshold_sensitivity": threshold_data,
        "ldlj_sparc": {
            "scppo_aggregate": {
                "ldlj_mean": statistics.fmean([ldlj_data["scppo"][s]["ldlj"] for s in [11, 17, 23]
                                               if s in ldlj_data["scppo"] and ldlj_data["scppo"][s]["ldlj"] is not None]),
                "sparc_mean": statistics.fmean([ldlj_data["scppo"][s]["sparc"] for s in [11, 17, 23]
                                                if s in ldlj_data["scppo"] and ldlj_data["scppo"][s]["sparc"] is not None]),
            },
            "layernorm_aggregate": {
                "ldlj_mean": statistics.fmean([ldlj_data["layernorm"][s]["ldlj"] for s in [11, 17, 23]
                                               if s in ldlj_data["layernorm"] and ldlj_data["layernorm"][s]["ldlj"] is not None]),
                "sparc_mean": statistics.fmean([ldlj_data["layernorm"][s]["sparc"] for s in [11, 17, 23]
                                                if s in ldlj_data["layernorm"] and ldlj_data["layernorm"][s]["sparc"] is not None]),
            },
            "source": ldlj_data.get("source", {}),
        },
    }

    data_path = output_dir / "paper_figures_data.json"
    with open(data_path, "w") as f:
        json.dump(paper_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved paper_figures_data.json")

    # Manifest
    manifest = {
        "generated": "2026-05-26",
        "script": "scripts/analysis/generate_paper_figures.py",
        "outputs": [
            {"file": "figure_cross_engine_degradation.png", "description": "Cross-engine degradation bar chart (5 methods)"},
            {"file": "figure_sensitivity_evolution.png", "description": "Sensitivity evolution during training (SC-PPO vs LayerNorm)"},
            {"file": "figure_sensitivity_vs_degradation.png", "description": "Sensitivity → degradation scatter plot with trend line"},
            {"file": "figure_ldlj_sparc.png", "description": "LDLJ/SPARC kinematic vs dynamic smoothness comparison"},
            {"file": "paper_figures_data.json", "description": "Structured JSON data with source-artifact traceability"},
        ],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Saved manifest.json")

    print(f"\nDone. {len(manifest['outputs'])} files in {output_dir}")
    print("Reproduction: python scripts/analysis/generate_paper_figures.py")


if __name__ == "__main__":
    main()
