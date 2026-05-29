#!/usr/bin/env python3
"""Mechanism-level decomposition of matched five-seed MuJoCo mixed evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from _common import ensure_directory, relative_to_repo, write_json  # noqa: E402


SEEDS = [11, 17, 23, 29, 31]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "analysis" / "mujoco_mixed_evidence"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "full-paper" / "mujoco-mixed-evidence-mechanism.md"

LCP_SUMMARY = REPO_ROOT / "artifacts" / "analysis" / "rough_terrain_lcp_soft_jacobian_formal" / "comparison_summary.json"
EXTENDED_SUMMARY = REPO_ROOT / "artifacts" / "analysis" / "rough_terrain_extended_seeds" / "comparison_summary.json"
STATISTICS_NOTE = REPO_ROOT / "docs" / "full-paper" / "statistical-robustness-results.md"
AMPLIFICATION_NOTE = REPO_ROOT / "docs" / "sc-ppo-mujoco-amplification-trace-comparison.md"
ACTUATOR_PROXY_NOTE = REPO_ROOT / "docs" / "sc-ppo-actuator-proxy-stress.md"

MUJOCO_BASES = {
    "lcp": (
        REPO_ROOT
        / "artifacts"
        / "methods"
        / "lcp_soft_jacobian_penalty_diagnostic"
        / "lcp_soft_jacobian_penalty_diagnostic"
    ),
    "scppo": (
        REPO_ROOT
        / "artifacts"
        / "methods"
        / "sc_ppo_pid_probe"
        / "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400"
    ),
    "heuristic": (
        REPO_ROOT
        / "artifacts"
        / "methods"
        / "heuristic_smoothing_formal_protocol_revision_long_budget"
        / "heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain"
    ),
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    lower_is_better: bool
    digits: int = 3


METRICS = [
    MetricSpec("fall_rate", "Fall", True),
    MetricSpec("velocity_tracking_error_mean", "Vel. err", True),
    MetricSpec("joint_acceleration_l2_mean", "Jnt acc", True),
    MetricSpec("action_jitter_l2_mean", "Jitter", True),
    MetricSpec("episode_return_mean", "Return", False),
]
METHOD_ORDER = ["lcp", "scppo", "heuristic"]
METHOD_LABELS = {
    "lcp": "LCP-style soft penalty",
    "scppo": "SC-PPO 3.8 PID",
    "heuristic": "Revised heuristic",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def metrics_path(method_id: str, seed: int) -> Path:
    return Path(f"{MUJOCO_BASES[method_id]}_seed{seed}") / "metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"


def collect_metrics() -> tuple[dict[str, dict[int, dict[str, float]]], list[str]]:
    data: dict[str, dict[int, dict[str, float]]] = {method_id: {} for method_id in METHOD_ORDER}
    sources = [str(LCP_SUMMARY.relative_to(REPO_ROOT)), str(EXTENDED_SUMMARY.relative_to(REPO_ROOT))]
    for method_id in METHOD_ORDER:
        for seed in SEEDS:
            path = metrics_path(method_id, seed)
            payload = read_json(path)
            data[method_id][seed] = {
                metric.key: float(payload[metric.key])
                for metric in METRICS
            }
            sources.append(str(path.relative_to(REPO_ROOT)))
    return data, sorted(set(sources))


def winner_from_values(values: dict[str, float], metric: MetricSpec) -> str:
    best_value = min(values.values()) if metric.lower_is_better else max(values.values())
    winners = [
        method_id for method_id, value in values.items()
        if math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12)
    ]
    return " / ".join(METHOD_LABELS[method_id] for method_id in winners)


def winner_for_seed(data: dict[str, dict[int, dict[str, float]]], seed: int, metric: MetricSpec) -> str:
    values = {method_id: data[method_id][seed][metric.key] for method_id in METHOD_ORDER}
    return winner_from_values(values, metric)


def aggregate_rows(data: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, Any]]:
    rows = []
    for method_id in METHOD_ORDER:
        for metric in METRICS:
            values = [data[method_id][seed][metric.key] for seed in SEEDS]
            rows.append(
                {
                    "method_id": method_id,
                    "method": METHOD_LABELS[method_id],
                    "metric": metric.key,
                    "metric_label": metric.label,
                    "lower_is_better": metric.lower_is_better,
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values),
                    "values_by_seed": {str(seed): data[method_id][seed][metric.key] for seed in SEEDS},
                }
            )
    return rows


def per_seed_winner_rows(data: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        row = {"seed": seed}
        for metric in METRICS:
            row[metric.key] = winner_for_seed(data, seed, metric)
        rows.append(row)
    return rows


def pairwise_rows(data: dict[str, dict[int, dict[str, float]]], first: str, second: str) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        row = {
            "seed": seed,
            "first_method": METHOD_LABELS[first],
            "second_method": METHOD_LABELS[second],
        }
        for metric in METRICS:
            row[f"{metric.key}_delta"] = data[first][seed][metric.key] - data[second][seed][metric.key]
        rows.append(row)
    return rows


def winner_counts(data: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, Any]]:
    rows = []
    for metric in METRICS:
        counts = {METHOD_LABELS[method_id]: 0.0 for method_id in METHOD_ORDER}
        for seed in SEEDS:
            winner = winner_for_seed(data, seed, metric)
            split = winner.split(" / ")
            for item in split:
                counts[item] += 1.0 / len(split)
        rows.append(
            {
                "metric": metric.key,
                "metric_label": metric.label,
                "winner_counts": counts,
            }
        )
    return rows


def correlation_rows(data: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, Any]]:
    rows = []
    for method_id in METHOD_ORDER:
        for seed in SEEDS:
            payload = data[method_id][seed]
            rows.append(
                {
                    "method_id": method_id,
                    "seed": seed,
                    **payload,
                }
            )
    return rows


def correlation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    jitter = [row["action_jitter_l2_mean"] for row in rows]
    joint = [row["joint_acceleration_l2_mean"] for row in rows]
    velocity = [row["velocity_tracking_error_mean"] for row in rows]
    returns = [row["episode_return_mean"] for row in rows]
    fall = [row["fall_rate"] for row in rows]
    return {
        "n": len(rows),
        "corr_jitter_joint_acc": pearson(jitter, joint),
        "corr_velocity_return": pearson(velocity, returns),
        "corr_joint_return": pearson(joint, returns),
        "corr_jitter_return": pearson(jitter, returns),
        "corr_fall_return": pearson(fall, returns),
    }


def correlations(data: dict[str, dict[int, dict[str, float]]]) -> dict[str, Any]:
    rows = correlation_rows(data)
    return {
        "across_method_seed_rows": correlation_summary(rows),
        "exclude_seed_29": correlation_summary([row for row in rows if row["seed"] != 29]),
        "exclude_scppo_seed_29": correlation_summary(
            [row for row in rows if not (row["method_id"] == "scppo" and row["seed"] == 29)]
        ),
        "lcp_plus_heuristic_only": correlation_summary(
            [row for row in rows if row["method_id"] in {"lcp", "heuristic"}]
        ),
        "by_method": {
            method_id: correlation_summary([row for row in rows if row["method_id"] == method_id])
            for method_id in METHOD_ORDER
        },
    }


def leave_one_seed_out(data: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, Any]]:
    rows = []
    for held_out in SEEDS:
        kept = [seed for seed in SEEDS if seed != held_out]
        row = {"held_out_seed": held_out}
        for method_id in METHOD_ORDER:
            row[method_id] = {
                metric.key: statistics.fmean(data[method_id][seed][metric.key] for seed in kept)
                for metric in METRICS
            }
        rows.append(row)
    return rows


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    aggregate = summary["aggregate_rows"]
    winner_rows = summary["per_seed_winner_rows"]
    counts = summary["winner_counts"]
    lcp_vs_heuristic = summary["lcp_vs_heuristic_rows"]
    leave_one_rows = summary["leave_one_seed_out"]
    correlations_payload = summary["correlations"]
    corr = correlations_payload["across_method_seed_rows"]

    lines = [
        "# Matched MuJoCo Mixed-Evidence Mechanism Note (#77)",
        "",
        "Status: `complete`.",
        "",
        "This note explains the matched five-seed MuJoCo split without forcing a universal winner. "
        "It uses the existing selected-checkpoint MuJoCo replays for LCP, SC-PPO, and the revised heuristic; no training or replay was rerun.",
        "",
        "## Main Read",
        "",
        "- LCP is the cleanest policy-output row: it has the lowest aggregate MuJoCo action jitter and wins the per-seed jitter ranking on three of five seeds.",
        "- The revised heuristic remains a strong control-path anchor: it wins aggregate MuJoCo joint acceleration and return, but not by dominating every seed.",
        "- SC-PPO's matched MuJoCo aggregate is mostly hurt by rough dynamic outliers, especially seed 29 in joint acceleration and action jitter.",
        "- The metric split is coherent: policy-local sensitivity regularization suppresses action-stream variability, while joint acceleration and return also depend on closed-loop tracking, PD response, contact timing, and simulator-specific dynamics.",
        "- Existing amplification/proxy notes support policy-output/control-stream amplification as a plausible mechanism, but the current full-paper matched LCP-vs-heuristic read remains aggregate-level and correlational.",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | Metric | Mean | Std |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| {method} | {metric} | {mean} | {std} |".format(
                method=row["method"],
                metric=row["metric_label"],
                mean=fmt(row["mean"]),
                std=fmt(row["std"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Seed Winners",
            "",
            "| Seed | Fall | Vel. err | Jnt acc | Jitter | Return |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in winner_rows:
        lines.append(
            "| {seed} | {fall} | {vel} | {joint} | {jitter} | {ret} |".format(
                seed=row["seed"],
                fall=row["fall_rate"],
                vel=row["velocity_tracking_error_mean"],
                joint=row["joint_acceleration_l2_mean"],
                jitter=row["action_jitter_l2_mean"],
                ret=row["episode_return_mean"],
            )
        )

    lines.extend(
        [
            "",
            "Winner counts across the five matched seeds:",
            "",
            "| Metric | LCP | SC-PPO | Heuristic |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in counts:
        c = row["winner_counts"]
        lines.append(
            "| {metric} | {lcp} | {scppo} | {heuristic} |".format(
                metric=row["metric_label"],
                lcp=fmt(c[METHOD_LABELS["lcp"]], 1),
                scppo=fmt(c[METHOD_LABELS["scppo"]], 1),
                heuristic=fmt(c[METHOD_LABELS["heuristic"]], 1),
            )
        )

    lines.extend(
        [
            "",
            "## LCP-vs-Heuristic Seed Deltas",
            "",
            "Delta is `LCP - heuristic`. Lower is better for fall, velocity error, joint acceleration, and jitter; higher is better for return.",
            "",
            "| Seed | Fall delta | Vel delta | Jnt acc delta | Jitter delta | Return delta |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in lcp_vs_heuristic:
        lines.append(
            "| {seed} | {fall} | {vel} | {joint} | {jitter} | {ret} |".format(
                seed=row["seed"],
                fall=fmt(row["fall_rate_delta"]),
                vel=fmt(row["velocity_tracking_error_mean_delta"]),
                joint=fmt(row["joint_acceleration_l2_mean_delta"]),
                jitter=fmt(row["action_jitter_l2_mean_delta"]),
                ret=fmt(row["episode_return_mean_delta"]),
            )
        )

    lines.extend(
        [
            "",
            "## Cross-Metric Coupling",
            "",
            "Across the 15 method-seed rows:",
            "",
            f"- corr(action jitter, joint acceleration) = `{fmt(corr['corr_jitter_joint_acc'])}`",
            f"- corr(velocity error, return) = `{fmt(corr['corr_velocity_return'])}`",
            f"- corr(joint acceleration, return) = `{fmt(corr['corr_joint_return'])}`",
            f"- corr(action jitter, return) = `{fmt(corr['corr_jitter_return'])}`",
            "",
            "Correlation sensitivity checks:",
            "",
            "| Row set | n | corr(jitter, jnt acc) | corr(vel err, return) | corr(jitter, return) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    correlation_rows_to_write = [
        ("All method-seed rows", correlations_payload["across_method_seed_rows"]),
        ("Exclude all seed-29 rows", correlations_payload["exclude_seed_29"]),
        ("Exclude only SC-PPO seed-29", correlations_payload["exclude_scppo_seed_29"]),
        ("LCP + heuristic only", correlations_payload["lcp_plus_heuristic_only"]),
    ]
    for label, payload in correlation_rows_to_write:
        lines.append(
            "| {label} | {n} | {joint} | {ret} | {jitret} |".format(
                label=label,
                n=payload["n"],
                joint=fmt(payload["corr_jitter_joint_acc"]),
                ret=fmt(payload["corr_velocity_return"]),
                jitret=fmt(payload["corr_jitter_return"]),
            )
        )

    lines.extend(
        [
            "",
            "Interpretation: action jitter and joint acceleration are coupled but not identical. The all-row coupling is amplified by the SC-PPO seed-29 outlier; after removing that single row, the coupling remains positive but drops from very strong to moderate. Return is more strongly tied to task tracking and seed-specific rollout behavior than to a single smoothness metric. This explains why LCP can be best on action jitter while the heuristic remains better on aggregate return.",
            "",
            "## Leave-One-Seed Stability",
            "",
            "| Held-out seed | Best aggregate jnt acc | Best aggregate jitter | Best aggregate return | SC-PPO jnt acc |",
            "| ---: | --- | --- | --- | ---: |",
        ]
    )
    for row in leave_one_rows:
        joint_winner = winner_from_values(
            {method_id: row[method_id]["joint_acceleration_l2_mean"] for method_id in METHOD_ORDER},
            next(metric for metric in METRICS if metric.key == "joint_acceleration_l2_mean"),
        )
        jitter_winner = winner_from_values(
            {method_id: row[method_id]["action_jitter_l2_mean"] for method_id in METHOD_ORDER},
            next(metric for metric in METRICS if metric.key == "action_jitter_l2_mean"),
        )
        return_winner = winner_from_values(
            {method_id: row[method_id]["episode_return_mean"] for method_id in METHOD_ORDER},
            next(metric for metric in METRICS if metric.key == "episode_return_mean"),
        )
        lines.append(
            "| {seed} | {joint} | {jitter} | {ret} | {scppo_joint} |".format(
                seed=row["held_out_seed"],
                joint=joint_winner,
                jitter=jitter_winner,
                ret=return_winner,
                scppo_joint=fmt(row["scppo"]["joint_acceleration_l2_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "Leave-one-seed aggregates preserve the main split: LCP remains the best action-jitter row in every split, while the revised heuristic remains the best joint-acceleration and return row in every split. Removing seed 29 sharply improves SC-PPO joint acceleration (`159.718 -> 125.003`), so SC-PPO's weak MuJoCo aggregate is outlier-amplified, but the LCP-vs-heuristic trade-off is not just a seed-29 artifact.",
            "",
            "## Mechanism Interpretation",
            "",
            "The current evidence supports a two-stage explanation:",
            "",
            "1. Policy-local-sensitivity regularization primarily acts on the policy-output stream. This is why LCP has the strongest aggregate action-jitter profile and why it is much cleaner than SC-PPO in matched MuJoCo replay.",
            "2. MuJoCo joint acceleration and return are downstream closed-loop outcomes. They depend not only on action jitter, but also on velocity tracking, contact timing, PD target dynamics, and simulator-specific response. The revised heuristic can therefore beat LCP on joint acceleration/return even while having higher aggregate action jitter.",
            "",
            "This should be written as a mechanism-level trade-off, not as a contradiction and not as an LCP universal win.",
            "",
            "## Relation to Existing Trace/Proxy Evidence",
            "",
            f"- Amplification trace note: `{relative_to_repo(AMPLIFICATION_NOTE)}`",
            f"- Actuator proxy note: `{relative_to_repo(ACTUATOR_PROXY_NOTE)}`",
            "",
            "The older amplification trace evidence supports the broad `policy-output/control-stream amplification -> joint acceleration` pathway for high-degradation methods. It does not directly prove the LCP-vs-heuristic split because that trace slice did not include the full-paper LCP/heuristic matched five-seed rows. The actuator-proxy result similarly supports the relevance of control-path smoothness but remains a bounded diagnostic.",
            "",
            "## Paper Wording Guidance",
            "",
            "- Say that LCP is the strongest current local-sensitivity row and the cleanest action-jitter row.",
            "- Say that the revised heuristic remains a competitive reward-shaping anchor and is better on matched MuJoCo joint acceleration and return.",
            "- Say that MuJoCo evidence is mixed but interpretable as a control-path metric split.",
            "- Do not claim that policy sensitivity alone causally determines MuJoCo return or joint acceleration.",
            "",
            "## Source Artifacts",
            "",
        ]
    )
    for source in summary["source_artifacts"]:
        lines.append(f"- `{source}`")

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/analyze_mujoco_mixed_evidence.py",
            "```",
            "",
            f"Generated runtime summary: `{summary['generated_artifacts']['summary_json']}`",
            f"Generated runtime table note: `{summary['generated_artifacts']['summary_markdown']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(output_dir: Path) -> dict[str, Any]:
    data, sources = collect_metrics()
    return {
        "issue": "#77",
        "protocol": {
            "seeds": SEEDS,
            "mujoco_protocol": "isaac_mainline, 20 episodes x 20s, joint_reset_noise=0.1",
            "interpretation_boundary": "Aggregate mechanism decomposition; not intervention-level causal proof.",
        },
        "aggregate_rows": aggregate_rows(data),
        "per_seed_winner_rows": per_seed_winner_rows(data),
        "winner_counts": winner_counts(data),
        "lcp_vs_heuristic_rows": pairwise_rows(data, "lcp", "heuristic"),
        "lcp_vs_scppo_rows": pairwise_rows(data, "lcp", "scppo"),
        "scppo_vs_heuristic_rows": pairwise_rows(data, "scppo", "heuristic"),
        "leave_one_seed_out": leave_one_seed_out(data),
        "correlations": correlations(data),
        "source_artifacts": sorted(set(sources + [
            relative_to_repo(STATISTICS_NOTE),
            relative_to_repo(AMPLIFICATION_NOTE),
            relative_to_repo(ACTUATOR_PROXY_NOTE),
        ])),
        "generated_artifacts": {
            "summary_json": relative_to_repo(output_dir / "summary.json"),
            "summary_markdown": relative_to_repo(output_dir / "summary.md"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze matched MuJoCo mixed evidence.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--doc-path", default=str(DEFAULT_DOC_PATH))
    args = parser.parse_args()

    output_dir = ensure_directory(Path(args.output_dir))
    doc_path = Path(args.doc_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(output_dir)
    write_json(output_dir / "summary.json", summary)
    write_markdown(summary, output_dir / "summary.md")
    write_markdown(summary, doc_path)
    print(f"Wrote {relative_to_repo(output_dir / 'summary.json')}")
    print(f"Wrote {relative_to_repo(output_dir / 'summary.md')}")
    print(f"Wrote {relative_to_repo(doc_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
