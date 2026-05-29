#!/usr/bin/env python3
"""Statistical robustness analysis for full-paper mechanism-comparison tables.

The analysis is deliberately descriptive: five matched seeds are enough for
uncertainty and paired-delta auditing, but not enough for strong NHST claims.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
BOOTSTRAP_ITERATIONS = 20_000
RANDOM_SEED = 20260529

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "analysis" / "full_paper_statistics"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "full-paper" / "statistical-robustness-results.md"

LCP_SUMMARY = REPO_ROOT / "artifacts" / "analysis" / "rough_terrain_lcp_soft_jacobian_formal" / "comparison_summary.json"
EXTENDED_SUMMARY = REPO_ROOT / "artifacts" / "analysis" / "rough_terrain_extended_seeds" / "comparison_summary.json"
LCP_MUJOCO_BASE = (
    REPO_ROOT
    / "artifacts"
    / "methods"
    / "lcp_soft_jacobian_penalty_diagnostic"
    / "lcp_soft_jacobian_penalty_diagnostic"
)
SCPPO_MUJOCO_BASE = (
    REPO_ROOT
    / "artifacts"
    / "methods"
    / "sc_ppo_pid_probe"
    / "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400"
)
HEURISTIC_MUJOCO_BASE = (
    REPO_ROOT
    / "artifacts"
    / "methods"
    / "heuristic_smoothing_formal_protocol_revision_long_budget"
    / "heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain"
)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    lower_is_better: bool
    digits: int = 3


ISAAC_METRICS = [
    MetricSpec("fall_rate", "Fall", True),
    MetricSpec("velocity_tracking_error_mean", "Vel. err", True),
    MetricSpec("joint_acceleration_l2_mean", "Jnt acc", True),
    MetricSpec("action_jitter_l2_mean", "Jitter", True),
    MetricSpec("episode_return_mean", "Return", False),
    MetricSpec("policy_sensitivity", "Sensitivity", True),
]
MUJOCO_METRICS = [
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
COMPARISONS = [
    ("lcp", "scppo"),
    ("lcp", "heuristic"),
    ("scppo", "heuristic"),
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty values")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    return sorted_values[lower] * (upper - pos) + sorted_values[upper] * (pos - lower)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize empty values")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def bootstrap_mean_ci(
    values: list[float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot bootstrap empty values")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    return percentile(means, 0.025), percentile(means, 0.975)


def policy_sensitivity(metrics: dict[str, Any]) -> float | None:
    direct = metrics.get("eval_policy_local_sensitivity_cost_mean")
    if direct is not None:
        return float(direct)
    direct = metrics.get("policy_local_sensitivity_cost_mean")
    if direct is not None:
        return float(direct)
    nested = metrics.get("constraint_metrics", {})
    value = nested.get("policy_local_sensitivity_cost_mean") if isinstance(nested, dict) else None
    return float(value) if value is not None else None


def candidate_by_id(summary: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in summary.get("candidates", []):
        if candidate.get("id") == candidate_id:
            return candidate
    raise KeyError(f"Missing candidate {candidate_id}")


def selected_metrics_for_candidate(candidate: dict[str, Any], seed: int) -> tuple[dict[str, Any], list[str]]:
    record = candidate["per_seed"][str(seed)]
    metrics = dict(record["selected_metrics"])
    sens = policy_sensitivity(metrics)
    if sens is not None:
        metrics["policy_sensitivity"] = sens
    sources = [
        str(EXTENDED_SUMMARY.relative_to(REPO_ROOT)),
        record.get("selected_metrics_path"),
        record.get("checkpoint_sweep_summary_path"),
    ]
    return metrics, [source for source in sources if source]


def selected_metrics_for_lcp(summary: dict[str, Any], seed: int) -> tuple[dict[str, Any], list[str]]:
    record = next(row for row in summary["per_seed"] if int(row["seed"]) == seed)
    metrics = dict(record["selected"])
    sens = policy_sensitivity(metrics)
    if sens is not None:
        metrics["policy_sensitivity"] = sens
    sources = [
        str(LCP_SUMMARY.relative_to(REPO_ROOT)),
        metrics.get("metrics_path"),
        record.get("summary_path"),
    ]
    return metrics, [source for source in sources if source]


def mujoco_metrics(base: Path, seed: int, summary_path: Path) -> tuple[dict[str, Any], list[str]]:
    path = Path(f"{base}_seed{seed}") / "metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"
    metrics = read_json(path)
    return metrics, [str(summary_path.relative_to(REPO_ROOT)), str(path.relative_to(REPO_ROOT))]


def collect_data() -> tuple[dict[str, dict[str, dict[int, dict[str, float]]]], list[str]]:
    lcp_summary = read_json(LCP_SUMMARY)
    extended_summary = read_json(EXTENDED_SUMMARY)
    scppo = candidate_by_id(extended_summary, "sc_ppo")
    heuristic = candidate_by_id(extended_summary, "heuristic_smoothing")

    data: dict[str, dict[str, dict[int, dict[str, float]]]] = {
        "isaac": {method: {} for method in METHOD_ORDER},
        "mujoco": {method: {} for method in METHOD_ORDER},
    }
    sources: list[str] = []

    for seed in SEEDS:
        for method_id, loader in [
            ("lcp", lambda s=seed: selected_metrics_for_lcp(lcp_summary, s)),
            ("scppo", lambda s=seed: selected_metrics_for_candidate(scppo, s)),
            ("heuristic", lambda s=seed: selected_metrics_for_candidate(heuristic, s)),
        ]:
            metrics, method_sources = loader()
            data["isaac"][method_id][seed] = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
            sources.extend(method_sources)

        for method_id, base, summary_path in [
            ("lcp", LCP_MUJOCO_BASE, LCP_SUMMARY),
            ("scppo", SCPPO_MUJOCO_BASE, EXTENDED_SUMMARY),
            ("heuristic", HEURISTIC_MUJOCO_BASE, EXTENDED_SUMMARY),
        ]:
            metrics, method_sources = mujoco_metrics(base, seed, summary_path)
            data["mujoco"][method_id][seed] = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
            sources.extend(method_sources)

    return data, sorted(set(sources))


def values_for(data: dict[str, dict[int, dict[str, float]]], method_id: str, metric_key: str) -> list[float]:
    return [data[method_id][seed][metric_key] for seed in SEEDS]


def summarize_means(data: dict[str, dict[str, dict[int, dict[str, float]]]]) -> list[dict[str, Any]]:
    rows = []
    for dataset, metrics in [("isaac", ISAAC_METRICS), ("mujoco", MUJOCO_METRICS)]:
        for method_id in METHOD_ORDER:
            for metric in metrics:
                values = values_for(data[dataset], method_id, metric.key)
                mean, std = mean_std(values)
                ci_low, ci_high = bootstrap_mean_ci(values)
                rows.append(
                    {
                        "dataset": dataset,
                        "method_id": method_id,
                        "method": METHOD_LABELS[method_id],
                        "metric": metric.key,
                        "metric_label": metric.label,
                        "lower_is_better": metric.lower_is_better,
                        "mean": mean,
                        "std": std,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "values_by_seed": {str(seed): data[dataset][method_id][seed][metric.key] for seed in SEEDS},
                    }
                )
    return rows


def better_method(first: str, second: str, delta: float, metric: MetricSpec) -> str:
    if delta == 0.0:
        return "tie"
    first_better = delta < 0.0 if metric.lower_is_better else delta > 0.0
    return first if first_better else second


def preferred_label(first: str, second: str, delta: float, metric: MetricSpec) -> str:
    winner = better_method(first, second, delta, metric)
    return "tie" if winner == "tie" else METHOD_LABELS[winner]


def summarize_paired_deltas(data: dict[str, dict[str, dict[int, dict[str, float]]]]) -> list[dict[str, Any]]:
    rows = []
    for dataset, metrics in [("isaac", ISAAC_METRICS), ("mujoco", MUJOCO_METRICS)]:
        for first, second in COMPARISONS:
            for metric in metrics:
                deltas = [
                    data[dataset][first][seed][metric.key] - data[dataset][second][seed][metric.key]
                    for seed in SEEDS
                ]
                mean, std = mean_std(deltas)
                ci_low, ci_high = bootstrap_mean_ci(deltas)
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{METHOD_LABELS[first]} - {METHOD_LABELS[second]}",
                        "first_method_id": first,
                        "second_method_id": second,
                        "metric": metric.key,
                        "metric_label": metric.label,
                        "lower_is_better": metric.lower_is_better,
                        "mean_delta": mean,
                        "std_delta": std,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "ci_excludes_zero": ci_low > 0.0 or ci_high < 0.0,
                        "preferred_by_mean": preferred_label(first, second, mean, metric),
                        "deltas_by_seed": {str(seed): deltas[index] for index, seed in enumerate(SEEDS)},
                    }
                )
    return rows


def summarize_rank_stability(data: dict[str, dict[str, dict[int, dict[str, float]]]]) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    rows = []
    for dataset, metrics in [("isaac", ISAAC_METRICS), ("mujoco", MUJOCO_METRICS)]:
        for metric in metrics:
            winner_counts = {method_id: 0.0 for method_id in METHOD_ORDER}
            n = len(SEEDS)
            for _ in range(BOOTSTRAP_ITERATIONS):
                sampled_seeds = [SEEDS[rng.randrange(n)] for _ in range(n)]
                method_means = {
                    method_id: statistics.fmean(data[dataset][method_id][seed][metric.key] for seed in sampled_seeds)
                    for method_id in METHOD_ORDER
                }
                best = min(method_means.values()) if metric.lower_is_better else max(method_means.values())
                winners = [
                    method_id for method_id, value in method_means.items()
                    if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)
                ]
                for winner in winners:
                    winner_counts[winner] += 1.0 / len(winners)
            max_count = max(winner_counts.values())
            top_winners = [
                METHOD_LABELS[method_id] for method_id, count in winner_counts.items()
                if math.isclose(count, max_count, rel_tol=0.0, abs_tol=1e-12)
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric.key,
                    "metric_label": metric.label,
                    "lower_is_better": metric.lower_is_better,
                    "winner_frequencies": {
                        METHOD_LABELS[method_id]: winner_counts[method_id] / BOOTSTRAP_ITERATIONS
                        for method_id in METHOD_ORDER
                    },
                    "most_frequent_winner": " / ".join(top_winners),
                }
            )
    return rows


def find_delta(
    rows: list[dict[str, Any]],
    dataset: str,
    first: str,
    second: str,
    metric: str,
) -> dict[str, Any]:
    for row in rows:
        if (
            row["dataset"] == dataset
            and row["first_method_id"] == first
            and row["second_method_id"] == second
            and row["metric"] == metric
        ):
            return row
    raise KeyError((dataset, first, second, metric))


def stability_phrase(row: dict[str, Any]) -> str:
    ci = f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"
    if row["ci_excludes_zero"]:
        return f"{row['preferred_by_mean']} preferred; paired 95% CI excludes zero ({ci})"
    return f"{row['preferred_by_mean']} preferred by mean, but CI includes zero ({ci})"


def write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    mean_rows = summary["mean_ci_rows"]
    delta_rows = summary["paired_delta_rows"]
    rank_rows = summary["rank_stability_rows"]

    lines = [
        "# Full-Paper Statistical Robustness Results (#75)",
        "",
        "Status: `complete`.",
        "",
        "This note adds a descriptive statistical audit for the full-paper mechanism-comparison evidence. "
        "It uses matched seeds `11/17/23/29/31`, nonparametric bootstrap confidence intervals over seed means, "
        "paired seed-level deltas, and bootstrap rank stability. With five seeds, these intervals should be "
        "read as uncertainty evidence rather than strong null-hypothesis significance tests.",
        "",
        "## Main Read",
        "",
        "- The strongest statistically robust statement is still mechanism-level: LCP is clearly stronger than the current SC-PPO hard-constraint row on Isaac fall, velocity error, return, and sensitivity, and on MuJoCo action jitter.",
        "- LCP's joint-acceleration advantage over SC-PPO is directionally favorable in both Isaac and MuJoCo, but the paired bootstrap intervals overlap zero because seed-level variance is large.",
        "- LCP versus the revised heuristic remains metric-dependent: LCP is usually better on action jitter and return-sensitive Isaac task behavior, while the heuristic remains competitive or better on joint acceleration, especially in MuJoCo.",
        "- Several paired confidence intervals include zero. The paper should therefore report stable directions and uncertainty, not binary significance claims.",
        "",
        "Representative paired reads:",
        "",
        f"- Isaac LCP vs SC-PPO, joint acceleration: {stability_phrase(find_delta(delta_rows, 'isaac', 'lcp', 'scppo', 'joint_acceleration_l2_mean'))}.",
        f"- Isaac LCP vs heuristic, joint acceleration: {stability_phrase(find_delta(delta_rows, 'isaac', 'lcp', 'heuristic', 'joint_acceleration_l2_mean'))}.",
        f"- MuJoCo LCP vs SC-PPO, joint acceleration: {stability_phrase(find_delta(delta_rows, 'mujoco', 'lcp', 'scppo', 'joint_acceleration_l2_mean'))}.",
        f"- MuJoCo LCP vs heuristic, joint acceleration: {stability_phrase(find_delta(delta_rows, 'mujoco', 'lcp', 'heuristic', 'joint_acceleration_l2_mean'))}.",
        f"- MuJoCo LCP vs heuristic, action jitter: {stability_phrase(find_delta(delta_rows, 'mujoco', 'lcp', 'heuristic', 'action_jitter_l2_mean'))}.",
        "",
        "## Mean and Bootstrap CI",
        "",
        "| Dataset | Method | Metric | Mean | Std | 95% bootstrap CI |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in mean_rows:
        lines.append(
            "| {dataset} | {method} | {metric} | {mean} | {std} | [{low}, {high}] |".format(
                dataset=row["dataset"],
                method=row["method"],
                metric=row["metric_label"],
                mean=fmt(row["mean"]),
                std=fmt(row["std"]),
                low=fmt(row["ci95_low"]),
                high=fmt(row["ci95_high"]),
            )
        )

    lines.extend(
        [
            "",
            "## Paired Seed-Level Deltas",
            "",
            "Delta is `first method - second method`. For fall, velocity error, joint acceleration, jitter, "
            "and sensitivity, lower is better. For return, higher is better.",
            "",
            "| Dataset | Comparison | Metric | Mean delta | 95% bootstrap CI | Mean-preferred method | CI excludes zero |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in delta_rows:
        lines.append(
            "| {dataset} | {comparison} | {metric} | {delta} | [{low}, {high}] | {preferred} | {excludes} |".format(
                dataset=row["dataset"],
                comparison=row["comparison"],
                metric=row["metric_label"],
                delta=fmt(row["mean_delta"]),
                low=fmt(row["ci95_low"]),
                high=fmt(row["ci95_high"]),
                preferred=row["preferred_by_mean"],
                excludes=str(row["ci_excludes_zero"]).lower(),
            )
        )

    lines.extend(
        [
            "",
            "## Bootstrap Rank Stability",
            "",
            "Values are the fraction of bootstrap resamples in which each method is the best-ranked method for the metric.",
            "",
            "| Dataset | Metric | Most frequent winner | LCP | SC-PPO | Heuristic |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rank_rows:
        freqs = row["winner_frequencies"]
        lines.append(
            "| {dataset} | {metric} | {winner} | {lcp} | {scppo} | {heuristic} |".format(
                dataset=row["dataset"],
                metric=row["metric_label"],
                winner=row["most_frequent_winner"],
                lcp=fmt(freqs[METHOD_LABELS["lcp"]]),
                scppo=fmt(freqs[METHOD_LABELS["scppo"]]),
                heuristic=fmt(freqs[METHOD_LABELS["heuristic"]]),
            )
        )

    lines.extend(
        [
            "",
            "## Paper Wording Guidance",
            "",
            "- Use `paired bootstrap uncertainty audit` rather than `statistical significance test`.",
            "- It is defensible to say LCP is robustly stronger than SC-PPO in the current five-seed mechanism comparison.",
            "- It is not defensible to say LCP robustly dominates the revised heuristic across all metrics.",
            "- Keep the revised heuristic as a strong reward-shaping anchor; the statistics reinforce that it is not a strawman.",
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
            "/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/analyze_full_paper_statistics.py",
            "```",
            "",
            f"Generated runtime summary: `{summary['generated_artifacts']['summary_json']}`",
            f"Generated runtime table note: `{summary['generated_artifacts']['summary_markdown']}`",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(output_dir: Path) -> dict[str, Any]:
    data, source_artifacts = collect_data()
    return {
        "issue": "#75",
        "protocol": {
            "seeds": SEEDS,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "random_seed": RANDOM_SEED,
            "interpretation": "Descriptive paired bootstrap uncertainty audit, not NHST.",
        },
        "mean_ci_rows": summarize_means(data),
        "paired_delta_rows": summarize_paired_deltas(data),
        "rank_stability_rows": summarize_rank_stability(data),
        "source_artifacts": source_artifacts,
        "generated_artifacts": {
            "summary_json": relative_to_repo(output_dir / "summary.json"),
            "summary_markdown": relative_to_repo(output_dir / "summary.md"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze full-paper statistical robustness.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--doc-path", default=str(DEFAULT_DOC_PATH))
    args = parser.parse_args()

    output_dir = ensure_directory(Path(args.output_dir))
    doc_path = Path(args.doc_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(output_dir)
    write_json(output_dir / "summary.json", summary)
    write_summary_markdown(summary, output_dir / "summary.md")
    write_summary_markdown(summary, doc_path)
    print(f"Wrote {relative_to_repo(output_dir / 'summary.json')}")
    print(f"Wrote {relative_to_repo(output_dir / 'summary.md')}")
    print(f"Wrote {relative_to_repo(doc_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
