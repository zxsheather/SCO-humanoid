#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:
    sys.stderr.write(
        "Missing plotting dependency. Use the project Python environment, e.g. "
        "`/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python "
        "scripts/analysis/generate_mechanism_chain_figure.py`.\n"
    )
    raise SystemExit(2) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "analysis" / "paper_figures"
POLICY_PERTURBATION_SUMMARY = REPO_ROOT / "artifacts" / "analysis" / "policy_perturbation_audit" / "summary.json"
OBS_NOISE_SUMMARY = REPO_ROOT / "artifacts" / "analysis" / "observation_noise_robustness" / "summary.json"
MUJOCO_FILENAME = "metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"
METHODS = [
    {
        "id": "lcp",
        "label": "LCP-style",
        "mujoco_dir": "artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/lcp_soft_jacobian_penalty_diagnostic_seed{seed}",
    },
    {
        "id": "scppo38",
        "label": "SC-PPO",
        "mujoco_dir": "artifacts/methods/sc_ppo_pid_probe/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed{seed}",
    },
    {
        "id": "heuristic",
        "label": "Heuristic",
        "mujoco_dir": "artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed{seed}",
    },
]
SEEDS = [11, 17, 23, 29, 31]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_metric(paths: list[Path], key: str) -> float | None:
    values: list[float] = []
    for path in paths:
        if not path.exists():
            continue
        value = read_json(path).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return statistics.fmean(values) if values else None


def perturbation_rows() -> dict[str, dict[str, float]]:
    summary = read_json(POLICY_PERTURBATION_SUMMARY)
    rows: dict[str, dict[str, float]] = {}
    for row in summary["aggregates"]:
        method_id = str(row["method_id"])
        rows[method_id] = {
            "policy_sensitivity": float(row["selected_policy_sensitivity"]["mean"]),
            "perturbation_amplification": float(row["amplification_mean"]["mean"]),
            "isaac_action_jitter": float(row["selected_action_jitter"]["mean"]),
            "isaac_joint_acceleration": float(row["selected_joint_acceleration"]["mean"]),
        }
    return rows


def mujoco_rows() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for method in METHODS:
        paths = [
            REPO_ROOT / method["mujoco_dir"].format(seed=seed) / MUJOCO_FILENAME
            for seed in SEEDS
        ]
        rows[method["id"]] = {
            "mujoco_action_jitter": mean_metric(paths, "action_jitter_l2_mean"),
            "mujoco_joint_acceleration": mean_metric(paths, "joint_acceleration_l2_mean"),
            "mujoco_return": mean_metric(paths, "episode_return_mean"),
        }
    return rows


def observation_noise_column() -> dict[str, float] | None:
    if not OBS_NOISE_SUMMARY.exists():
        return None
    summary = read_json(OBS_NOISE_SUMMARY)
    aggregates = summary.get("aggregates", [])
    candidates: dict[str, dict[str, float]] = {}
    for row in aggregates:
        if row.get("engine") != "isaac":
            continue
        if abs(float(row.get("noise_std", -1.0)) - 0.05) > 1e-12:
            continue
        if int(row.get("seed_count", 0)) < len(SEEDS):
            continue
        rel = row.get("relative_to_noise0", {})
        value = rel.get("action_jitter_l2_mean")
        if isinstance(value, (int, float)):
            candidates[str(row["method_id"])] = float(value)
    if all(method["id"] in candidates for method in METHODS):
        return candidates
    return None


def rank_values(values: dict[str, float], higher_is_better: bool = False) -> dict[str, int]:
    ordered = sorted(
        values.items(),
        key=lambda item: item[1],
        reverse=higher_is_better,
    )
    return {method_id: rank + 1 for rank, (method_id, _) in enumerate(ordered)}


def fmt(value: float | None, metric_id: str) -> str:
    if value is None:
        return "n/a"
    if metric_id == "mujoco_return":
        return f"{value:.0f}"
    if "joint_acceleration" in metric_id:
        return f"{value:.0f}"
    if metric_id == "policy_sensitivity":
        return f"{value:.2f}"
    if metric_id == "perturbation_amplification":
        return f"{value:.3f}"
    if metric_id == "obs_noise_jitter_factor":
        return f"{value:.2f}x"
    return f"{value:.3f}"


def assemble_metrics() -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    data = perturbation_rows()
    mujoco = mujoco_rows()
    for method_id, row in mujoco.items():
        data.setdefault(method_id, {}).update(row)

    metrics = [
        {
            "id": "policy_sensitivity",
            "label": "Policy\nsensitivity",
            "group": "Policy-output mechanism",
            "higher_is_better": False,
        },
        {
            "id": "perturbation_amplification",
            "label": "Perturb.\namplification",
            "group": "Policy-output mechanism",
            "higher_is_better": False,
        },
        {
            "id": "isaac_action_jitter",
            "label": "Isaac\naction jitter",
            "group": "Policy-output mechanism",
            "higher_is_better": False,
        },
        {
            "id": "mujoco_action_jitter",
            "label": "MuJoCo\naction jitter",
            "group": "Policy-output mechanism",
            "higher_is_better": False,
        },
    ]

    obs_noise = observation_noise_column()
    if obs_noise is not None:
        for method_id, value in obs_noise.items():
            data.setdefault(method_id, {})["obs_noise_jitter_factor"] = value
        metrics.append(
            {
                "id": "obs_noise_jitter_factor",
                "label": "Obs-noise\njitter factor",
                "group": "Policy-output mechanism",
                "higher_is_better": False,
            }
        )

    metrics.extend(
        [
            {
                "id": "mujoco_joint_acceleration",
                "label": "MuJoCo\njoint acc.",
                "group": "Closed-loop downstream",
                "higher_is_better": False,
            },
            {
                "id": "mujoco_return",
                "label": "MuJoCo\nreturn",
                "group": "Closed-loop downstream",
                "higher_is_better": True,
            },
        ]
    )
    return metrics, data


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, data = assemble_metrics()
    method_ids = [method["id"] for method in METHODS]
    rank_matrix = np.zeros((len(method_ids), len(metrics)))
    value_labels: list[list[str]] = []

    for col, metric in enumerate(metrics):
        values = {
            method_id: float(data[method_id][metric["id"]])
            for method_id in method_ids
            if data.get(method_id, {}).get(metric["id"]) is not None
        }
        ranks = rank_values(values, higher_is_better=bool(metric["higher_is_better"]))
        col_labels = []
        for row, method_id in enumerate(method_ids):
            rank = ranks.get(method_id, len(method_ids))
            rank_matrix[row, col] = rank
            value = data.get(method_id, {}).get(metric["id"])
            col_labels.append(f"#{rank}\n{fmt(value, metric['id'])}")
        value_labels.append(col_labels)

    fig_width = max(8.5, 1.15 * len(metrics) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 3.6))
    im = ax.imshow(rank_matrix, cmap="RdYlGn_r", vmin=1, vmax=len(method_ids), aspect="auto")
    ax.set_xticks(range(len(metrics)), [metric["label"] for metric in metrics], fontsize=9)
    ax.set_yticks(range(len(method_ids)), [method["label"] for method in METHODS], fontsize=10)
    ax.tick_params(axis="both", length=0)

    for col, metric_labels in enumerate(value_labels):
        for row, label in enumerate(metric_labels):
            ax.text(col, row, label, ha="center", va="center", fontsize=8, color="#202020")

    downstream_start = next(
        idx for idx, metric in enumerate(metrics) if metric["group"] == "Closed-loop downstream"
    )
    ax.axvline(downstream_start - 0.5, color="#222222", linewidth=1.2)
    ax.text(
        (downstream_start - 1) / 2,
        -0.9,
        "policy-output mechanism",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        downstream_start + (len(metrics) - downstream_start - 1) / 2,
        -0.9,
        "closed-loop downstream",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_title("Mechanism-chain evidence ranks: local policy metrics align, downstream metrics can split", fontsize=11, pad=30)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[1, 2, 3], label="rank (1 = best)")
    fig.tight_layout()

    figure_path = output_dir / "figure_mechanism_chain.png"
    data_path = output_dir / "figure_mechanism_chain_data.json"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "figure": str(figure_path.relative_to(REPO_ROOT)),
        "metrics": metrics,
        "methods": METHODS,
        "values": data,
        "rank_matrix": rank_matrix.tolist(),
        "source_artifacts": [
            str(POLICY_PERTURBATION_SUMMARY.relative_to(REPO_ROOT)),
            str(OBS_NOISE_SUMMARY.relative_to(REPO_ROOT)) if OBS_NOISE_SUMMARY.exists() else None,
            "artifacts/methods/*/*/" + MUJOCO_FILENAME,
        ],
        "claim_boundary": (
            "Ranks summarize mechanism-chain evidence for selected checkpoints. "
            "They are descriptive and do not establish hardware transfer or broad benchmark superiority."
        ),
    }
    payload["source_artifacts"] = [source for source in payload["source_artifacts"] if source]
    with data_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the full-paper mechanism-chain figure.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    payload = generate(Path(args.output_dir))
    print(f"wrote {payload['figure']}")
    print(f"wrote {Path(args.output_dir) / 'figure_mechanism_chain_data.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
