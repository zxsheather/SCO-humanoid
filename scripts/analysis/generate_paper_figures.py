#!/usr/bin/env python3
"""Generate paper-grade figures and tables from existing experiment artifacts.

Reproducible analysis: reads JSON summaries and checkpoint sweeps, produces
PNG figures, Markdown/CSV tables, and structured JSON data files with source-artifact traceability.

Usage:
    /TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/generate_paper_figures.py
    python scripts/analysis/generate_paper_figures.py [--output-dir ARTIFACTS_DIR]  # if deps installed
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
except ModuleNotFoundError as exc:
    sys.stderr.write(
        "Missing plotting dependency. Use the project Python environment, e.g. "
        "`/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python "
        "scripts/analysis/generate_paper_figures.py`, or install matplotlib and numpy.\n"
    )
    raise SystemExit(2) from exc

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "analysis" / "paper_figures"
SEEDS = [11, 17, 23]
FULL_SEEDS = [11, 17, 23, 29, 31]
DIAGNOSTIC_SEEDS = [23, 29, 31]
METRIC_KEYS = [
    "joint_acceleration_l2_mean",
    "action_jitter_l2_mean",
    "velocity_tracking_error_mean",
    "fall_rate",
]

FULL_PAPER_EXTENDED_SUMMARY = (
    "artifacts/analysis/rough_terrain_extended_seeds/comparison_summary.json"
)
LCP_FORMAL_SUMMARY = (
    "artifacts/analysis/rough_terrain_lcp_soft_jacobian_formal/comparison_summary.json"
)
LCP_WEIGHT_SUMMARIES = {
    "0.001": "artifacts/analysis/rough_terrain_lcp_weight_sensitivity_diagnostic/w0001/comparison_summary.json",
    "0.002": "artifacts/analysis/rough_terrain_lcp_soft_jacobian_diagnostic/comparison_summary.json",
    "0.004": "artifacts/analysis/rough_terrain_lcp_weight_sensitivity_diagnostic/w0004/comparison_summary.json",
}
OMNISAFE_DIAGNOSTIC_SUMMARY = (
    "artifacts/analysis/rough_terrain_omnisafe_ppolag_diagnostic/comparison_summary.json"
)
LCP_MUJOCO_BASE = (
    "artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/"
    "lcp_soft_jacobian_penalty_diagnostic"
)
SCPPO_MUJOCO_BASE = (
    "artifacts/methods/sc_ppo_pid_probe/"
    "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400"
)
HEURISTIC_MUJOCO_BASE = (
    "artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/"
    "heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain"
)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def seed_path(base: str, seed: int, filename: str) -> Path:
    return REPO_ROOT / f"{base}_seed{seed}" / filename


def read_sweep(base: str, seed: int) -> dict:
    return read_json(require_path(seed_path(base, seed, "checkpoint_sweep_summary.json"),
                                  "checkpoint sweep summary"))


def row_for_checkpoint(sweep: dict, checkpoint: int) -> dict:
    for row in sweep.get("rows", []):
        if row.get("checkpoint") == checkpoint:
            return row
    raise KeyError(f"Checkpoint {checkpoint} not found in {sweep.get('run_name', '<unknown>')}")


def final_checkpoint_from_sweep(sweep: dict) -> int:
    checkpoints = sweep.get("evaluated_checkpoints") or [row["checkpoint"] for row in sweep.get("rows", [])]
    return max(checkpoints)


def checkpoint_metadata(base: str, seeds: list[int]) -> dict:
    """Return explicit selected/final checkpoint provenance for one method family."""
    selected_checkpoints: dict[str, int] = {}
    final_checkpoints: dict[str, int] = {}
    selection_statuses: dict[str, str | None] = {}
    per_seed: dict[str, dict] = {}

    for seed in seeds:
        sweep = read_sweep(base, seed)
        selected_cp = int(sweep["best_checkpoint"])
        final_cp = int(final_checkpoint_from_sweep(sweep))
        selected_row = row_for_checkpoint(sweep, selected_cp)
        final_row = row_for_checkpoint(sweep, final_cp)

        selected_checkpoints[str(seed)] = selected_cp
        final_checkpoints[str(seed)] = final_cp
        selection_statuses[str(seed)] = sweep.get("selection_status")
        per_seed[str(seed)] = {
            "checkpoint_sweep_summary": f"{base}_seed{seed}/checkpoint_sweep_summary.json",
            "selection_status": sweep.get("selection_status"),
            "selected_checkpoint": selected_cp,
            "selected_metrics_path": selected_row.get("metrics_path") or f"{base}_seed{seed}/metrics_selected.json",
            "final_checkpoint": final_cp,
            "final_metrics_path": final_row.get("metrics_path") or f"{base}_seed{seed}/metrics.json",
        }

    return {
        "selected_checkpoints": selected_checkpoints,
        "final_checkpoints": final_checkpoints,
        "selection_statuses": selection_statuses,
        "per_seed": per_seed,
    }


def aggregate_checkpoint_metric(base: str, seeds: list[int], key: str, basis: str) -> tuple[float, float]:
    vals = []
    for seed in seeds:
        sweep = read_sweep(base, seed)
        if basis == "selected":
            checkpoint = int(sweep["best_checkpoint"])
        elif basis == "final":
            checkpoint = final_checkpoint_from_sweep(sweep)
        else:
            raise ValueError(f"Unknown checkpoint basis: {basis}")
        row = row_for_checkpoint(sweep, checkpoint)
        value = row.get(key)
        if value is not None:
            vals.append(value)

    if not vals:
        return 0.0, 0.0
    return statistics.fmean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def checkpoint_map_str(value: dict | None, seeds: list[int] | None = None) -> str:
    if not value:
        return ""
    seed_order = seeds or SEEDS
    return " / ".join(str(value.get(str(seed), "")) for seed in seed_order)


def seeds_str(seeds: list[int]) -> str:
    return " / ".join(str(seed) for seed in seeds)


def compact_source_list(paths: list[str]) -> str:
    seen: set[str] = set()
    ordered = []
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)
    return "; ".join(ordered)


def source_list_from_metadata(meta: dict, basis: str) -> str:
    key = "selected_metrics_path" if basis == "selected" else "final_metrics_path"
    return "; ".join(meta["per_seed"][str(seed)][key] for seed in SEEDS)


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
            "include_in_sensitivity_plot": False,
            "summary_artifact": "artifacts/analysis/rough_terrain_formal_protocol_revision_long_budget/comparison_summary.json",
            "source_artifacts": {
                "isaac": "artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed{11,17,23}/metrics_selected.json",
                "mujoco": "artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json",
                "checkpoint_sweep": "artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed{11,17,23}/checkpoint_sweep_summary.json",
            },
        },
        {
            "id": "scppo38",
            "label": "SC-PPO 3.8\n(Jacobian)",
            "color": "#4CAF50",
            "isaac_dir": "artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400",
            "mujoco_dir": "artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400",
            "sensitivity": None,
            "include_in_sensitivity_plot": True,
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
            "sensitivity": None,
            "include_in_sensitivity_plot": True,
            "summary_artifact": "artifacts/analysis/rough_terrain_layernorm_actor_more_epochs_reliability_probe/comparison_summary.json",
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
            "sensitivity": None,
            "include_in_sensitivity_plot": True,
            "summary_artifact": "artifacts/analysis/rough_terrain_action_scaling_probe/comparison_summary.json",
            "source_artifacts": {
                "isaac": "artifacts/methods/action_scaling_probe/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/metrics_selected.json",
                "mujoco": "artifacts/methods/action_scaling_probe/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json",
                "checkpoint_sweep": "artifacts/methods/action_scaling_probe/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/checkpoint_sweep_summary.json",
            },
        },
        {
            "id": "output_scaling",
            "label": "Output\nScaling",
            "color": "#9C27B0",
            "isaac_dir": "artifacts/methods/output_scaling_probe/output_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain",
            "mujoco_dir": "artifacts/methods/output_scaling_probe/output_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain",
            "sensitivity": None,
            "include_in_sensitivity_plot": True,
            "summary_artifact": "artifacts/analysis/rough_terrain_output_scaling_probe/comparison_summary.json",
            "source_artifacts": {
                "isaac": "artifacts/methods/output_scaling_probe/output_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/metrics_selected.json",
                "mujoco": "artifacts/methods/output_scaling_probe/output_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json",
                "checkpoint_sweep": "artifacts/methods/output_scaling_probe/output_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed{11,17,23}/checkpoint_sweep_summary.json",
            },
        },
    ]

    for m in methods:
        meta = checkpoint_metadata(m["isaac_dir"], SEEDS)
        m["checkpoint_metadata"] = {
            "isaac_selected": {
                "basis": "selected",
                "description": "Isaac aggregate is computed from each seed's selected checkpoint.",
                **meta,
            },
            "isaac_final": {
                "basis": "final",
                "description": "Final-checkpoint side read for selected-vs-final audit.",
                "selected_checkpoints": meta["selected_checkpoints"],
                "final_checkpoints": meta["final_checkpoints"],
                "selection_statuses": meta["selection_statuses"],
                "per_seed": meta["per_seed"],
            },
            "mujoco": {
                "basis": "selected-checkpoint replay",
                "description": "MuJoCo aggregate replays the same selected Isaac checkpoint for each seed.",
                "selected_checkpoints": meta["selected_checkpoints"],
                "per_seed": {
                    str(seed): {
                        "checkpoint": meta["selected_checkpoints"][str(seed)],
                        "metrics_path": f"{m['mujoco_dir']}_seed{seed}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json",
                    }
                    for seed in SEEDS
                },
            },
        }
        m["selected_checkpoints"] = meta["selected_checkpoints"]
        m["final_checkpoints"] = meta["final_checkpoints"]
        m["selection_statuses"] = meta["selection_statuses"]
        m["isaac_checkpoint_basis"] = "selected"
        m["mujoco_checkpoint_basis"] = "selected-checkpoint replay"

        for key in METRIC_KEYS:
            # Isaac
            i_vals = []
            for s in SEEDS:
                v = selected_metric(m["isaac_dir"], s, key)
                if v is not None:
                    i_vals.append(v)
            if i_vals:
                m[f"isaac_{key}"] = statistics.fmean(i_vals)
                m[f"isaac_{key}_std"] = statistics.pstdev(i_vals) if len(i_vals) > 1 else 0.0

            # MuJoCo
            m_vals = []
            for s in SEEDS:
                v = mujoco_metric(m["mujoco_dir"], s, key)
                if v is not None:
                    m_vals.append(v)
            if m_vals:
                m[f"mujoco_{key}"] = statistics.fmean(m_vals)
                m[f"mujoco_{key}_std"] = statistics.pstdev(m_vals) if len(m_vals) > 1 else 0.0

            # Final-checkpoint side read
            f_mean, f_std = aggregate_checkpoint_metric(m["isaac_dir"], SEEDS, key, "final")
            m[f"final_{key}"] = f_mean
            m[f"final_{key}_std"] = f_std

        sensitivity_mean, sensitivity_std = aggregate_checkpoint_metric(
            m["isaac_dir"], SEEDS, "policy_local_sensitivity_cost_mean", "final"
        )
        if m["include_in_sensitivity_plot"]:
            m["sensitivity"] = sensitivity_mean
            m["sensitivity_std"] = sensitivity_std
        m["sensitivity_source"] = {
            "checkpoint_basis": "final",
            "metric": "policy_local_sensitivity_cost_mean",
            "mean": sensitivity_mean,
            "std": sensitivity_std,
            "per_seed": {
                str(seed): meta["per_seed"][str(seed)]["final_metrics_path"]
                for seed in SEEDS
            },
        }

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
# Full-paper mechanism-comparison data assembly
# ---------------------------------------------------------------------------

def iter_per_seed(per_seed) -> list[tuple[str, dict]]:
    if isinstance(per_seed, dict):
        return sorted(per_seed.items(), key=lambda item: int(item[0]))
    return sorted(((str(item["seed"]), item) for item in per_seed), key=lambda item: int(item[0]))


def metrics_block(seed_record: dict, basis: str) -> dict:
    return seed_record.get(basis) or seed_record.get(f"{basis}_metrics") or {}


def checkpoint_for_basis(seed_record: dict, basis: str) -> int | None:
    key = f"{basis}_checkpoint"
    if seed_record.get(key) is not None:
        return int(seed_record[key])
    metrics = metrics_block(seed_record, basis)
    checkpoint = metrics.get("checkpoint")
    return int(checkpoint) if checkpoint is not None else None


def metric_values_from_summary(summary: dict, basis: str, metric_key: str, seeds: list[int]) -> list[float]:
    values = []
    wanted = {str(seed) for seed in seeds}
    for seed, seed_record in iter_per_seed(summary.get("per_seed", {})):
        if seed not in wanted:
            continue
        value = metrics_block(seed_record, basis).get(metric_key)
        if value is not None:
            values.append(value)
    return values


def first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def policy_sensitivity_values_from_summary(summary: dict, basis: str, seeds: list[int]) -> list[float]:
    values = []
    wanted = {str(seed) for seed in seeds}
    for seed, seed_record in iter_per_seed(summary.get("per_seed", {})):
        if seed not in wanted:
            continue
        metrics = metrics_block(seed_record, basis)
        value = first_not_none(
            metrics.get("eval_policy_local_sensitivity_cost_mean"),
            metrics.get("policy_local_sensitivity_cost_mean"),
            metrics.get("constraint_metrics", {}).get("policy_local_sensitivity_cost_mean"),
        )
        if value is not None:
            values.append(value)
    return values


def violation_values_from_summary(summary: dict, basis: str, seeds: list[int]) -> list[float]:
    values = []
    wanted = {str(seed) for seed in seeds}
    for seed, seed_record in iter_per_seed(summary.get("per_seed", {})):
        if seed not in wanted:
            continue
        metrics = metrics_block(seed_record, basis)
        value = first_not_none(
            metrics.get("eval_constraint_violation_rate"),
            metrics.get("constraint_violation_rate"),
            metrics.get("constraint_metrics", {}).get("constraint_violation_rate"),
        )
        if value is not None:
            values.append(value)
    return values


def fmt_values(values: list[float], digits: int = 3) -> str:
    if not values:
        return ""
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{fmt(mean, digits)} +/- {fmt(std, digits)}"


def source_paths_from_summary(summary_path: str, summary: dict, basis: str, seeds: list[int]) -> str:
    sources = [summary_path]
    wanted = {str(seed) for seed in seeds}
    for seed, seed_record in iter_per_seed(summary.get("per_seed", {})):
        if seed not in wanted:
            continue
        metrics = metrics_block(seed_record, basis)
        sources.extend([
            seed_record.get(f"{basis}_metrics_path"),
            metrics.get("metrics_path"),
            seed_record.get("checkpoint_sweep_summary_path"),
            seed_record.get("summary_path"),
            seed_record.get("training_status_path"),
        ])
    return compact_source_list([source for source in sources if source])


def checkpoints_from_summary(summary: dict, basis: str, seeds: list[int]) -> dict[str, int]:
    checkpoints = {}
    wanted = {str(seed) for seed in seeds}
    for seed, seed_record in iter_per_seed(summary.get("per_seed", {})):
        if seed not in wanted:
            continue
        checkpoint = checkpoint_for_basis(seed_record, basis)
        if checkpoint is not None:
            checkpoints[seed] = checkpoint
    return checkpoints


def statuses_from_summary(summary: dict, seeds: list[int]) -> dict[str, str]:
    statuses = {}
    wanted = {str(seed) for seed in seeds}
    for seed, seed_record in iter_per_seed(summary.get("per_seed", {})):
        if seed in wanted and seed_record.get("selection_status") is not None:
            statuses[seed] = seed_record["selection_status"]
    return statuses


def mechanism_summary_row(
    label: str,
    summary_path: str,
    summary: dict,
    seeds: list[int],
    basis: str = "selected",
    evidence_tier: str = "primary",
    extra: dict | None = None,
) -> dict:
    row = {
        "method": label,
        "evidence_tier": evidence_tier,
        "checkpoint_basis": basis,
        "seeds": seeds_str(seeds),
        "checkpoints": checkpoint_map_str(checkpoints_from_summary(summary, basis, seeds), seeds),
        "selection_statuses": checkpoint_map_str(statuses_from_summary(summary, seeds), seeds),
        "fall_rate": fmt_values(metric_values_from_summary(summary, basis, "fall_rate", seeds)),
        "velocity_error": fmt_values(metric_values_from_summary(summary, basis, "velocity_tracking_error_mean", seeds)),
        "joint_acceleration": fmt_values(metric_values_from_summary(summary, basis, "joint_acceleration_l2_mean", seeds)),
        "action_jitter": fmt_values(metric_values_from_summary(summary, basis, "action_jitter_l2_mean", seeds)),
        "episode_return": fmt_values(metric_values_from_summary(summary, basis, "episode_return_mean", seeds)),
        "policy_sensitivity": fmt_values(policy_sensitivity_values_from_summary(summary, basis, seeds)),
        "violation_rate": fmt_values(violation_values_from_summary(summary, basis, seeds)),
        "source_artifacts": source_paths_from_summary(summary_path, summary, basis, seeds),
    }
    if extra:
        row.update(extra)
    return row


def candidate_by_id(summary: dict, candidate_id: str) -> dict:
    for candidate in summary.get("candidates", []):
        if candidate.get("id") == candidate_id:
            return candidate
    raise KeyError(f"Candidate {candidate_id} not found in {summary.get('comparison_name', '<unknown>')}")


def build_full_paper_isaac_rows() -> list[dict]:
    lcp = read_json(require_path(REPO_ROOT / LCP_FORMAL_SUMMARY, "LCP formal comparison summary"))
    extended = read_json(require_path(REPO_ROOT / FULL_PAPER_EXTENDED_SUMMARY, "extended-seed comparison summary"))
    scppo = candidate_by_id(extended, "sc_ppo")
    heuristic = candidate_by_id(extended, "heuristic_smoothing")

    return [
        mechanism_summary_row(
            "LCP-style soft Jacobian/Lipschitz penalty",
            LCP_FORMAL_SUMMARY,
            lcp,
            FULL_SEEDS,
            evidence_tier="primary_full_paper",
            extra={"method_family": "soft_policy_sensitivity_regularization"},
        ),
        mechanism_summary_row(
            "SC-PPO 3.8 PID-Lagrangian",
            FULL_PAPER_EXTENDED_SUMMARY,
            scppo,
            FULL_SEEDS,
            evidence_tier="primary_full_paper",
            extra={"method_family": "hard_policy_sensitivity_constraint"},
        ),
        mechanism_summary_row(
            "Revised heuristic action-rate penalty",
            FULL_PAPER_EXTENDED_SUMMARY,
            heuristic,
            FULL_SEEDS,
            evidence_tier="primary_full_paper",
            extra={"method_family": "reward_shaping_anchor"},
        ),
    ]


def selected_checkpoints_for_mujoco() -> dict[str, dict[str, int]]:
    lcp = read_json(require_path(REPO_ROOT / LCP_FORMAL_SUMMARY, "LCP formal comparison summary"))
    extended = read_json(require_path(REPO_ROOT / FULL_PAPER_EXTENDED_SUMMARY, "extended-seed comparison summary"))
    return {
        "lcp": checkpoints_from_summary(lcp, "selected", FULL_SEEDS),
        "scppo": candidate_by_id(extended, "sc_ppo").get("selected_checkpoints", {}),
        "heuristic": candidate_by_id(extended, "heuristic_smoothing").get("selected_checkpoints", {}),
    }


def build_mujoco_row(
    label: str,
    method_key: str,
    base: str,
    summary_path: str,
    checkpoints: dict[str, int],
    method_family: str,
) -> dict:
    metrics_by_key: dict[str, list[float]] = {
        "fall_rate": [],
        "velocity_tracking_error_mean": [],
        "joint_acceleration_l2_mean": [],
        "action_jitter_l2_mean": [],
        "episode_return_mean": [],
    }
    sources = [summary_path]
    for seed in FULL_SEEDS:
        path = REPO_ROOT / f"{base}_seed{seed}/metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"
        require_path(path, f"{label} MuJoCo metrics for seed {seed}")
        sources.append(str(path.relative_to(REPO_ROOT)))
        data = read_json(path)
        for key in metrics_by_key:
            value = data.get(key)
            if value is not None:
                metrics_by_key[key].append(value)

    return {
        "method": label,
        "method_family": method_family,
        "evidence_tier": "primary_full_paper",
        "checkpoint_basis": "selected-checkpoint MuJoCo replay",
        "seeds": seeds_str(FULL_SEEDS),
        "checkpoints": checkpoint_map_str(checkpoints, FULL_SEEDS),
        "fall_rate": fmt_values(metrics_by_key["fall_rate"]),
        "velocity_error": fmt_values(metrics_by_key["velocity_tracking_error_mean"]),
        "joint_acceleration": fmt_values(metrics_by_key["joint_acceleration_l2_mean"]),
        "action_jitter": fmt_values(metrics_by_key["action_jitter_l2_mean"]),
        "episode_return": fmt_values(metrics_by_key["episode_return_mean"]),
        "source_artifacts": compact_source_list(sources),
        "method_key": method_key,
    }


def build_matched_mujoco_rows() -> list[dict]:
    checkpoints = selected_checkpoints_for_mujoco()
    return [
        build_mujoco_row(
            "LCP-style soft Jacobian/Lipschitz penalty",
            "lcp",
            LCP_MUJOCO_BASE,
            LCP_FORMAL_SUMMARY,
            checkpoints["lcp"],
            "soft_policy_sensitivity_regularization",
        ),
        build_mujoco_row(
            "SC-PPO 3.8 PID-Lagrangian",
            "scppo",
            SCPPO_MUJOCO_BASE,
            FULL_PAPER_EXTENDED_SUMMARY,
            checkpoints["scppo"],
            "hard_policy_sensitivity_constraint",
        ),
        build_mujoco_row(
            "Revised heuristic action-rate penalty",
            "heuristic",
            HEURISTIC_MUJOCO_BASE,
            FULL_PAPER_EXTENDED_SUMMARY,
            checkpoints["heuristic"],
            "reward_shaping_anchor",
        ),
    ]


def build_lcp_weight_sensitivity_rows() -> list[dict]:
    rows = []
    for weight, summary_path in LCP_WEIGHT_SUMMARIES.items():
        summary = read_json(require_path(REPO_ROOT / summary_path, f"LCP weight={weight} summary"))
        seeds = [int(seed) for seed in summary.get("seeds", DIAGNOSTIC_SEEDS)]
        rows.append(
            mechanism_summary_row(
                f"LCP-style soft penalty weight={weight}",
                summary_path,
                summary,
                seeds,
                evidence_tier="diagnostic_local_weight_grid",
                extra={
                    "lcp_weight": weight,
                    "method_family": "soft_policy_sensitivity_regularization",
                },
            )
        )
    return rows


def build_omnisafe_diagnostic_rows() -> list[dict]:
    summary = read_json(require_path(REPO_ROOT / OMNISAFE_DIAGNOSTIC_SUMMARY, "OmniSafe diagnostic summary"))
    rows = []
    for basis in ["selected", "final"]:
        rows.append(
            mechanism_summary_row(
                "OmniSafe PPO-Lag migration diagnostic",
                OMNISAFE_DIAGNOSTIC_SUMMARY,
                summary,
                DIAGNOSTIC_SEEDS,
                basis=basis,
                evidence_tier="diagnostic_only",
                extra={
                    "method_family": "external_framework_migration",
                    "diagnostic_only": str(summary.get("diagnostic_only", True)),
                    "status": summary.get("status", ""),
                    "claim_boundary": summary.get("claim_boundary", ""),
                },
            )
        )
    return rows


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
        if m.get("include_in_sensitivity_plot", True) and sens and factor:
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
           if m.get("include_in_sensitivity_plot", True)
           and m.get("sensitivity") and m.get("degradation_factor")]
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


def fig5_task_vs_smoothness(methods, output_dir: Path):
    """Figure 5: Separate task metrics from dynamic smoothness metrics."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    labels = [m["label"] for m in methods]
    x = np.arange(len(methods))
    width = 0.35

    isaac_fall = [m.get("isaac_fall_rate", 0) for m in methods]
    mujoco_fall = [m.get("mujoco_fall_rate", 0) for m in methods]
    ax1.bar(x - width / 2, isaac_fall, width, label="Isaac selected", color="#BBDEFB",
            edgecolor="#1976D2", linewidth=0.8)
    ax1.bar(x + width / 2, mujoco_fall, width, label="MuJoCo selected replay", color="#FFCCBC",
            edgecolor="#D84315", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Fall Rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("(a) Task Metric")
    ax1.legend(fontsize=8)

    isaac_jitter = [m.get("isaac_action_jitter_l2_mean", 0) for m in methods]
    mujoco_jitter = [m.get("mujoco_action_jitter_l2_mean", 0) for m in methods]
    ax2.bar(x - width / 2, isaac_jitter, width, label="Isaac selected", color="#C8E6C9",
            edgecolor="#388E3C", linewidth=0.8)
    ax2.bar(x + width / 2, mujoco_jitter, width, label="MuJoCo selected replay", color="#FFE0B2",
            edgecolor="#F57C00", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Action Jitter L2")
    ax2.set_title("(b) Dynamic Smoothness Metric")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "figure_task_vs_smoothness.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure_task_vs_smoothness.png")


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def write_csv_table(path: Path, rows: list[dict], headers: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def write_markdown_table(path: Path, rows: list[dict], headers: list[str]) -> None:
    with open(path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")


def method_table_row(method: dict, label: str, basis: str) -> dict:
    if basis == "selected":
        prefix = "isaac_"
        checkpoints = method["selected_checkpoints"]
        source = source_list_from_metadata(method["checkpoint_metadata"]["isaac_selected"], "selected")
    elif basis == "final":
        prefix = "final_"
        checkpoints = method["final_checkpoints"]
        source = source_list_from_metadata(method["checkpoint_metadata"]["isaac_final"], "final")
    else:
        raise ValueError(f"Unknown row basis: {basis}")

    return {
        "method": label,
        "checkpoint_basis": basis,
        "checkpoints_11_17_23": checkpoint_map_str(checkpoints),
        "selection_status_11_17_23": checkpoint_map_str(method["selection_statuses"]),
        "fall_rate": fmt(method.get(f"{prefix}fall_rate")),
        "velocity_error": fmt(method.get(f"{prefix}velocity_tracking_error_mean")),
        "joint_acceleration": fmt(method.get(f"{prefix}joint_acceleration_l2_mean")),
        "action_jitter": fmt(method.get(f"{prefix}action_jitter_l2_mean")),
        "source_artifacts": source,
    }


def comparison_candidate_rows(summary_path: str, label: str) -> list[dict]:
    data = read_json(require_path(REPO_ROOT / summary_path, "comparison summary"))
    candidate = data["candidates"][0]
    rows = []

    for basis, aggregate_key, checkpoints_key in [
        ("selected", "selected_aggregate", "selected_checkpoints"),
        ("final", "final_aggregate", "final_checkpoints"),
    ]:
        aggregate = candidate.get(aggregate_key)
        if not aggregate:
            continue
        rows.append({
            "method": label,
            "checkpoint_basis": basis,
            "checkpoints_11_17_23": checkpoint_map_str(candidate.get(checkpoints_key)),
            "selection_status_11_17_23": checkpoint_map_str(candidate.get("selection_statuses")),
            "fall_rate": fmt(aggregate.get("fall_rate_mean")),
            "velocity_error": fmt(aggregate.get("velocity_tracking_error_mean_mean")),
            "joint_acceleration": fmt(aggregate.get("joint_acceleration_l2_mean_mean")),
            "action_jitter": fmt(aggregate.get("action_jitter_l2_mean_mean")),
            "source_artifacts": f"{summary_path}#candidates[0].{aggregate_key}",
        })

    return rows


def ldlj_mean(ldlj_data: dict, method_key: str, metric_key: str) -> float | None:
    vals = [
        ldlj_data[method_key][seed][metric_key]
        for seed in SEEDS
        if seed in ldlj_data[method_key] and ldlj_data[method_key][seed][metric_key] is not None
    ]
    return statistics.fmean(vals) if vals else None


def build_cross_engine_rows(methods: list[dict]) -> list[dict]:
    rows = []
    for method in methods:
        rows.append({
            "method": method["label"].replace("\n", " "),
            "isaac_checkpoint_basis": method["isaac_checkpoint_basis"],
            "mujoco_checkpoint_basis": method["mujoco_checkpoint_basis"],
            "selected_checkpoints_11_17_23": checkpoint_map_str(method["selected_checkpoints"]),
            "final_checkpoints_11_17_23": checkpoint_map_str(method["final_checkpoints"]),
            "isaac_fall_rate": fmt(method.get("isaac_fall_rate")),
            "isaac_velocity_error": fmt(method.get("isaac_velocity_tracking_error_mean")),
            "isaac_joint_acceleration": fmt(method.get("isaac_joint_acceleration_l2_mean")),
            "isaac_action_jitter": fmt(method.get("isaac_action_jitter_l2_mean")),
            "mujoco_fall_rate": fmt(method.get("mujoco_fall_rate")),
            "mujoco_velocity_error": fmt(method.get("mujoco_velocity_tracking_error_mean")),
            "mujoco_joint_acceleration": fmt(method.get("mujoco_joint_acceleration_l2_mean")),
            "mujoco_action_jitter": fmt(method.get("mujoco_action_jitter_l2_mean")),
            "degradation_factor": fmt(method.get("degradation_factor")),
            "source_artifacts": (
                source_list_from_metadata(method["checkpoint_metadata"]["isaac_selected"], "selected")
                + "; "
                + "; ".join(
                    method["checkpoint_metadata"]["mujoco"]["per_seed"][str(seed)]["metrics_path"]
                    for seed in SEEDS
                )
            ),
        })
    return rows


def build_threshold_rows(threshold_data: dict) -> list[dict]:
    rows = []
    for item in threshold_data["thresholds"]:
        rows.append({
            "threshold": fmt(item.get("threshold"), 1),
            "regime": item.get("regime", ""),
            "seed11": f"cp{item.get('seed11_cp', '')} fall={fmt(item.get('seed11_fall'), 2)}",
            "seed17": f"cp{item.get('seed17_cp', '')} fall={fmt(item.get('seed17_fall'), 2)}",
            "seed23": f"cp{item.get('seed23_cp', '')} fall={fmt(item.get('seed23_fall'), 2)}",
            "outcome": item.get("outcome", ""),
            "source_artifacts": item.get("source", threshold_data.get("source", "")),
        })
    return rows


def build_layernorm_tradeoff_rows(methods: list[dict], ldlj_data: dict) -> list[dict]:
    by_id = {m["id"]: m for m in methods}
    rows = []
    for method_id, label, ldlj_key in [
        ("scppo38", "SC-PPO 3.8", "scppo"),
        ("layernorm", "LayerNorm epochs=3", "layernorm"),
    ]:
        method = by_id[method_id]
        rows.append({
            "method": label,
            "checkpoint_basis": method["isaac_checkpoint_basis"],
            "checkpoints_11_17_23": checkpoint_map_str(method["selected_checkpoints"]),
            "isaac_fall_rate": fmt(method.get("isaac_fall_rate")),
            "isaac_velocity_error": fmt(method.get("isaac_velocity_tracking_error_mean")),
            "isaac_joint_acceleration": fmt(method.get("isaac_joint_acceleration_l2_mean")),
            "isaac_action_jitter": fmt(method.get("isaac_action_jitter_l2_mean")),
            "mujoco_fall_rate": fmt(method.get("mujoco_fall_rate")),
            "mujoco_joint_acceleration": fmt(method.get("mujoco_joint_acceleration_l2_mean")),
            "mujoco_action_jitter": fmt(method.get("mujoco_action_jitter_l2_mean")),
            "ldlj": fmt(ldlj_mean(ldlj_data, ldlj_key, "ldlj")),
            "sparc": fmt(ldlj_mean(ldlj_data, ldlj_key, "sparc")),
            "source_artifacts": (
                source_list_from_metadata(method["checkpoint_metadata"]["isaac_selected"], "selected")
                + "; "
                + method["source_artifacts"].get("mujoco", "")
                + "; "
                + ldlj_data["source"][ldlj_key]
            ),
        })
    return rows


TABLE_DESCRIPTIONS = {
    "full_paper_isaac_mechanism_comparison": (
        "T0 primary full-paper five-seed Isaac mechanism comparison: "
        "LCP-style soft penalty, SC-PPO PID-Lagrangian, and revised heuristic"
    ),
    "matched_mujoco_mechanism_comparison": (
        "T0b primary full-paper matched five-seed MuJoCo selected-checkpoint replay comparison"
    ),
    "lcp_weight_sensitivity": (
        "T0c diagnostic LCP coefficient sensitivity grid for weights 0.001, 0.002, and 0.004"
    ),
    "omnisafe_diagnostic": (
        "Diagnostic-only OmniSafe PPO-Lag migration summary; not a promoted baseline"
    ),
    "cross_engine_degradation": (
        "Historical workshop-era 3-seed five-method selected-checkpoint degradation table"
    ),
    "threshold_sensitivity": "Historical workshop-era SC-PPO threshold sensitivity table",
    "plain_dual_vs_pid": (
        "Historical workshop-era plain dual ascent vs PID-Lagrangian selected/final table"
    ),
    "scppo_epochs3_repair": (
        "Historical workshop-era SC-PPO epochs=3 reliability repair selected/final table"
    ),
    "layernorm_tradeoff_ldlj_sparc": (
        "Historical workshop-era LayerNorm trade-off and LDLJ/SPARC table"
    ),
}


def source_artifacts_from_rows(rows: list[dict]) -> list[str]:
    sources = []
    for row in rows:
        for source in str(row.get("source_artifacts", "")).split(";"):
            source = source.strip()
            if source:
                sources.append(source)
    return sorted(set(sources))


def write_paper_tables(methods: list[dict], threshold_data: dict, ldlj_data: dict, output_dir: Path) -> dict:
    """Write Markdown/CSV tables and return their structured rows."""
    table_specs = {}

    full_headers = [
        "method", "method_family", "evidence_tier", "checkpoint_basis", "seeds",
        "checkpoints", "selection_statuses", "fall_rate", "velocity_error",
        "joint_acceleration", "action_jitter", "episode_return",
        "policy_sensitivity", "violation_rate", "source_artifacts",
    ]
    table_specs["full_paper_isaac_mechanism_comparison"] = (
        build_full_paper_isaac_rows(),
        full_headers,
    )

    mujoco_headers = [
        "method", "method_family", "evidence_tier", "checkpoint_basis", "seeds",
        "checkpoints", "fall_rate", "velocity_error", "joint_acceleration",
        "action_jitter", "episode_return", "source_artifacts",
    ]
    table_specs["matched_mujoco_mechanism_comparison"] = (
        build_matched_mujoco_rows(),
        mujoco_headers,
    )

    lcp_weight_headers = [
        "method", "method_family", "evidence_tier", "lcp_weight", "checkpoint_basis",
        "seeds", "checkpoints", "selection_statuses", "fall_rate", "velocity_error",
        "joint_acceleration", "action_jitter", "episode_return", "policy_sensitivity",
        "violation_rate", "source_artifacts",
    ]
    table_specs["lcp_weight_sensitivity"] = (
        build_lcp_weight_sensitivity_rows(),
        lcp_weight_headers,
    )

    omnisafe_headers = [
        "method", "method_family", "evidence_tier", "diagnostic_only", "status",
        "checkpoint_basis", "seeds", "checkpoints", "selection_statuses",
        "fall_rate", "velocity_error", "joint_acceleration", "action_jitter",
        "episode_return", "claim_boundary", "source_artifacts",
    ]
    table_specs["omnisafe_diagnostic"] = (
        build_omnisafe_diagnostic_rows(),
        omnisafe_headers,
    )

    cross_headers = [
        "method", "isaac_checkpoint_basis", "mujoco_checkpoint_basis",
        "selected_checkpoints_11_17_23", "final_checkpoints_11_17_23",
        "isaac_fall_rate", "isaac_velocity_error", "isaac_joint_acceleration",
        "isaac_action_jitter", "mujoco_fall_rate", "mujoco_velocity_error",
        "mujoco_joint_acceleration", "mujoco_action_jitter", "degradation_factor",
        "source_artifacts",
    ]
    table_specs["cross_engine_degradation"] = (build_cross_engine_rows(methods), cross_headers)

    threshold_headers = ["threshold", "regime", "seed11", "seed17", "seed23", "outcome", "source_artifacts"]
    table_specs["threshold_sensitivity"] = (build_threshold_rows(threshold_data), threshold_headers)

    by_id = {m["id"]: m for m in methods}
    comparison_headers = [
        "method", "checkpoint_basis", "checkpoints_11_17_23", "selection_status_11_17_23",
        "fall_rate", "velocity_error", "joint_acceleration", "action_jitter", "source_artifacts",
    ]
    plain_rows = [
        method_table_row(by_id["scppo38"], "PID-Lagrangian SC-PPO 3.8", "selected"),
        method_table_row(by_id["scppo38"], "PID-Lagrangian SC-PPO 3.8", "final"),
    ] + comparison_candidate_rows(
        "artifacts/analysis/rough_terrain_plain_dual_probe/comparison_summary.json",
        "Plain dual ascent",
    )
    table_specs["plain_dual_vs_pid"] = (plain_rows, comparison_headers)

    epochs_rows = [
        method_table_row(by_id["scppo38"], "SC-PPO 3.8 epochs=2", "selected"),
        method_table_row(by_id["scppo38"], "SC-PPO 3.8 epochs=2", "final"),
    ] + comparison_candidate_rows(
        "artifacts/analysis/rough_terrain_sc_ppo_epochs3_probe/comparison_summary.json",
        "SC-PPO 3.8 epochs=3",
    )
    table_specs["scppo_epochs3_repair"] = (epochs_rows, comparison_headers)

    tradeoff_headers = [
        "method", "checkpoint_basis", "checkpoints_11_17_23", "isaac_fall_rate",
        "isaac_velocity_error", "isaac_joint_acceleration", "isaac_action_jitter",
        "mujoco_fall_rate", "mujoco_joint_acceleration", "mujoco_action_jitter",
        "ldlj", "sparc", "source_artifacts",
    ]
    table_specs["layernorm_tradeoff_ldlj_sparc"] = (
        build_layernorm_tradeoff_rows(methods, ldlj_data),
        tradeoff_headers,
    )

    outputs = {}
    for name, (rows, headers) in table_specs.items():
        csv_path = output_dir / f"table_{name}.csv"
        md_path = output_dir / f"table_{name}.md"
        write_csv_table(csv_path, rows, headers)
        write_markdown_table(md_path, rows, headers)
        outputs[name] = {
            "csv": csv_path.name,
            "markdown": md_path.name,
            "description": TABLE_DESCRIPTIONS.get(name, name.replace("_", " ")),
            "headers": headers,
            "rows": rows,
            "source_artifacts": source_artifacts_from_rows(rows),
        }
        print(f"  Saved table_{name}.csv and table_{name}.md")

    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paper-grade figures and tables from experiment artifacts")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                       help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Paper Figures Generation ===")
    print(f"Output: {output_dir}")

    # Assemble data
    print("\n[1/7] Assembling degradation data...")
    methods = assemble_degradation_data()
    print(f"  {len(methods)} methods loaded")

    print("[2/7] Assembling sensitivity evolution...")
    evolution = assemble_sensitivity_evolution()
    print(f"  SC-PPO: {len(evolution['scppo'])} checkpoints, LayerNorm: {len(evolution['layernorm'])} checkpoints")

    print("[3/7] Assembling threshold data...")
    threshold_data = assemble_threshold_data()
    print(f"  {len(threshold_data['thresholds'])} thresholds")

    print("[4/7] Assembling LDLJ/SPARC data...")
    ldlj_data = assemble_ldlj_sparc_data()
    print(f"  SC-PPO: {len(ldlj_data['scppo'])} seeds, LayerNorm: {len(ldlj_data['layernorm'])} seeds")

    # Generate figures
    print("\n[5/7] Generating figures...")
    fig1_degradation_bars(methods, output_dir)
    fig2_sensitivity_evolution(evolution, output_dir)
    fig3_sensitivity_vs_degradation(methods, output_dir)
    fig4_ldlj_sparc(ldlj_data, methods, output_dir)
    fig5_task_vs_smoothness(methods, output_dir)

    print("\n[6/7] Generating tables...")
    tables = write_paper_tables(methods, threshold_data, ldlj_data, output_dir)

    # Save structured data
    print("\n[7/7] Saving structured data...")
    paper_data = {
        "generated": "2026-05-29",
        "description": "Paper-grade figures and tables for mechanism-comparison smooth humanoid control study",
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
        "tables": tables,
    }

    data_path = output_dir / "paper_figures_data.json"
    with open(data_path, "w") as f:
        json.dump(paper_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved paper_figures_data.json")

    # Manifest
    figure_outputs = [
        {"file": "figure_cross_engine_degradation.png", "description": "Historical workshop-era cross-engine degradation bar chart (5 methods)"},
        {"file": "figure_sensitivity_evolution.png", "description": "Historical workshop-era sensitivity evolution during training (SC-PPO vs LayerNorm)"},
        {"file": "figure_sensitivity_vs_degradation.png", "description": "Historical workshop-era sensitivity-to-degradation scatter plot with trend line"},
        {"file": "figure_ldlj_sparc.png", "description": "Historical workshop-era LDLJ/SPARC kinematic vs dynamic smoothness comparison"},
        {"file": "figure_task_vs_smoothness.png", "description": "Historical workshop-era task metrics separated from dynamic smoothness metrics"},
    ]
    table_outputs = []
    for table in tables.values():
        for output_key in ["markdown", "csv"]:
            table_outputs.append({
                "file": table[output_key],
                "description": table["description"],
                "source_artifacts": table["source_artifacts"],
            })

    manifest = {
        "generated": "2026-05-29",
        "script": "scripts/analysis/generate_paper_figures.py",
        "outputs": figure_outputs + table_outputs + [
            {"file": "paper_figures_data.json", "description": "Structured JSON data with source-artifact traceability"},
        ],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Saved manifest.json")

    print(f"\nDone. {len(manifest['outputs'])} files in {output_dir}")
    print(f"Reproduction: {sys.executable} scripts/analysis/generate_paper_figures.py")


if __name__ == "__main__":
    main()
