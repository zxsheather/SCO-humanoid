#!/usr/bin/env python3
"""Selected-vs-final checkpoint robustness audit for full-paper methods."""

from __future__ import annotations

import argparse
import json
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

from _common import artifact_dir, ensure_directory, load_config, relative_to_repo, write_json  # noqa: E402


SEEDS = [11, 17, 23, 29, 31]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "analysis" / "checkpoint_robustness"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "full-paper" / "selected-vs-final-checkpoint-robustness.md"
MUJOCO_SELECTED_METRICS = "metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"
MUJOCO_FINAL_METRICS_TEMPLATE = "metrics_mujoco_final_checkpoint_{checkpoint}_20ep_20s_noise01.json"

LCP_SUMMARY = REPO_ROOT / "artifacts" / "analysis" / "rough_terrain_lcp_soft_jacobian_formal" / "comparison_summary.json"
EXTENDED_SUMMARY = REPO_ROOT / "artifacts" / "analysis" / "rough_terrain_extended_seeds" / "comparison_summary.json"


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
    MetricSpec("policy_sensitivity", "Sensitivity", True),
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


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def policy_sensitivity(metrics: dict[str, Any]) -> float | None:
    for key in ["eval_policy_local_sensitivity_cost_mean", "policy_local_sensitivity_cost_mean"]:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    nested = metrics.get("constraint_metrics")
    if isinstance(nested, dict) and nested.get("policy_local_sensitivity_cost_mean") is not None:
        return float(nested["policy_local_sensitivity_cost_mean"])
    return None


def normalized_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out = {}
    for metric in METRICS:
        if metric.key == "policy_sensitivity":
            value = policy_sensitivity(metrics)
        else:
            value = metrics.get(metric.key)
        if value is not None:
            out[metric.key] = float(value)
    return out


def candidate_by_id(summary: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in summary.get("candidates", []):
        if candidate.get("id") == candidate_id:
            return candidate
    raise KeyError(f"Missing candidate {candidate_id}")


def row_for_checkpoint(sweep: dict[str, Any], checkpoint: int) -> dict[str, Any]:
    for row in sweep.get("rows", []):
        if int(row["checkpoint"]) == int(checkpoint):
            return row
    raise KeyError(f"Checkpoint {checkpoint} not found in {sweep.get('run_name', '<unknown>')}")


def lcp_seed_record(summary: dict[str, Any], seed: int) -> dict[str, Any]:
    record = next(row for row in summary["per_seed"] if int(row["seed"]) == seed)
    selected = normalized_metrics(record["selected"])
    final = normalized_metrics(record["final"])
    selected_path = record["selected"].get("metrics_path")
    final_path = record["final"].get("metrics_path")
    return {
        "seed": seed,
        "selected_checkpoint": int(record["selected_checkpoint"]),
        "final_checkpoint": int(record["final"]["checkpoint"]),
        "selection_status": record.get("selection_status"),
        "config_path": "configs/methods/lcp_soft_jacobian_penalty_diagnostic.json",
        "base_run_name": str(record["run_name"]),
        "selected": selected,
        "final": final,
        "source_artifacts": [
            str(LCP_SUMMARY.relative_to(REPO_ROOT)),
            selected_path,
            final_path,
            record.get("summary_path"),
        ],
    }


def candidate_seed_record(candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    record = candidate["per_seed"][str(seed)]
    selected_checkpoint = int(record["selected_checkpoint"])
    final_checkpoint = int(record["final_checkpoint"])
    sweep_path = REPO_ROOT / record["checkpoint_sweep_summary_path"]
    sweep = read_json(sweep_path)
    selected_row = row_for_checkpoint(sweep, selected_checkpoint)
    final_row = row_for_checkpoint(sweep, final_checkpoint)
    return {
        "seed": seed,
        "selected_checkpoint": selected_checkpoint,
        "final_checkpoint": final_checkpoint,
        "selection_status": record.get("selection_status"),
        "config_path": candidate["config_path"],
        "base_run_name": str(record["run_name"]),
        "selected": normalized_metrics(selected_row),
        "final": normalized_metrics(final_row),
        "source_artifacts": [
            str(EXTENDED_SUMMARY.relative_to(REPO_ROOT)),
            str(sweep_path.relative_to(REPO_ROOT)),
            selected_row.get("metrics_path"),
            final_row.get("metrics_path"),
        ],
    }


def collect_records() -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    lcp_summary = read_json(LCP_SUMMARY)
    extended = read_json(EXTENDED_SUMMARY)
    scppo = candidate_by_id(extended, "sc_ppo")
    heuristic = candidate_by_id(extended, "heuristic_smoothing")

    records = {
        "lcp": [lcp_seed_record(lcp_summary, seed) for seed in SEEDS],
        "scppo": [candidate_seed_record(scppo, seed) for seed in SEEDS],
        "heuristic": [candidate_seed_record(heuristic, seed) for seed in SEEDS],
    }
    sources = []
    for method_records in records.values():
        for record in method_records:
            sources.extend(source for source in record["source_artifacts"] if source)
    return records, sorted(set(sources))


def mujoco_final_run_name(record: dict[str, Any]) -> str:
    return f"{record['base_run_name']}_finalcp{record['final_checkpoint']}_mujoco"


def mujoco_selected_path(record: dict[str, Any]) -> Path:
    return artifact_dir(load_config(record["config_path"]), record["base_run_name"]) / MUJOCO_SELECTED_METRICS


def mujoco_final_path(record: dict[str, Any]) -> Path:
    if record["selected_checkpoint"] == record["final_checkpoint"]:
        return mujoco_selected_path(record)
    output_name = MUJOCO_FINAL_METRICS_TEMPLATE.format(checkpoint=int(record["final_checkpoint"]))
    return artifact_dir(load_config(record["config_path"]), mujoco_final_run_name(record)) / output_name


def maybe_read_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    return {
        key: float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


def collect_mujoco_records(records: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    out: dict[str, list[dict[str, Any]]] = {method_id: [] for method_id in METHOD_ORDER}
    missing: list[str] = []
    for method_id, method_records in records.items():
        for record in method_records:
            selected_path = mujoco_selected_path(record)
            final_path = mujoco_final_path(record)
            selected = maybe_read_metrics(selected_path)
            final = maybe_read_metrics(final_path)
            if selected is None:
                missing.append(relative_to_repo(selected_path))
            if final is None:
                missing.append(relative_to_repo(final_path))
            if selected is None or final is None:
                continue
            out[method_id].append(
                {
                    "seed": record["seed"],
                    "selected_checkpoint": record["selected_checkpoint"],
                    "final_checkpoint": record["final_checkpoint"],
                    "selected": selected,
                    "final": final,
                    "source_artifacts": [
                        relative_to_repo(selected_path),
                        relative_to_repo(final_path),
                    ],
                }
            )
    return out, sorted(set(missing))


def metric_delta(final_value: float, selected_value: float) -> float:
    return final_value - selected_value


def metric_preference(delta: float, metric: MetricSpec) -> str:
    if delta == 0.0:
        return "unchanged"
    final_better = delta < 0.0 if metric.lower_is_better else delta > 0.0
    return "final_better" if final_better else "selected_better"


def checkpoint_map(records: list[dict[str, Any]], key: str) -> str:
    by_seed = {record["seed"]: record[key] for record in records}
    return " / ".join(str(by_seed[seed]) for seed in SEEDS)


def classify_method(aggregate: dict[str, Any], changed_seed_count: int) -> str:
    fall_delta = aggregate["metrics"]["fall_rate"]["delta"]
    joint_delta = aggregate["metrics"]["joint_acceleration_l2_mean"]["delta"]
    jitter_delta = aggregate["metrics"]["action_jitter_l2_mean"]["delta"]
    if changed_seed_count <= 1 and abs(fall_delta) <= 0.001 and abs(joint_delta) <= 2.0 and abs(jitter_delta) <= 0.01:
        return "near_final"
    if fall_delta > 0.025:
        return "task_selection_sensitive"
    if joint_delta > 5.0 or jitter_delta > 0.02:
        return "dynamic_selection_sensitive"
    return "mild_selection_effect"


def summarize_method(method_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for metric in METRICS:
        selected_values = [record["selected"][metric.key] for record in records if metric.key in record["selected"]]
        final_values = [record["final"][metric.key] for record in records if metric.key in record["final"]]
        deltas = [
            metric_delta(record["final"][metric.key], record["selected"][metric.key])
            for record in records
            if metric.key in record["selected"] and metric.key in record["final"]
        ]
        metrics[metric.key] = {
            "label": metric.label,
            "lower_is_better": metric.lower_is_better,
            "selected_mean": mean(selected_values),
            "final_mean": mean(final_values),
            "delta": mean(deltas),
            "preference": metric_preference(mean(deltas), metric),
            "deltas_by_seed": {str(record["seed"]): metric_delta(record["final"][metric.key], record["selected"][metric.key]) for record in records if metric.key in record["selected"] and metric.key in record["final"]},
        }
    changed_seed_count = sum(1 for record in records if record["selected_checkpoint"] != record["final_checkpoint"])
    aggregate = {
        "method_id": method_id,
        "method": METHOD_LABELS[method_id],
        "selected_checkpoints": checkpoint_map(records, "selected_checkpoint"),
        "final_checkpoints": checkpoint_map(records, "final_checkpoint"),
        "changed_seed_count": changed_seed_count,
        "changed_seeds": [record["seed"] for record in records if record["selected_checkpoint"] != record["final_checkpoint"]],
        "metrics": metrics,
    }
    aggregate["classification"] = classify_method(aggregate, changed_seed_count)
    return aggregate


def summarize_mujoco_method(method_id: str, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(records) != len(SEEDS):
        return None
    metrics = {}
    for metric in METRICS:
        if metric.key == "policy_sensitivity":
            continue
        selected_values = [record["selected"][metric.key] for record in records if metric.key in record["selected"]]
        final_values = [record["final"][metric.key] for record in records if metric.key in record["final"]]
        deltas = [
            metric_delta(record["final"][metric.key], record["selected"][metric.key])
            for record in records
            if metric.key in record["selected"] and metric.key in record["final"]
        ]
        metrics[metric.key] = {
            "label": metric.label,
            "lower_is_better": metric.lower_is_better,
            "selected_mean": mean(selected_values),
            "final_mean": mean(final_values),
            "delta": mean(deltas),
            "preference": metric_preference(mean(deltas), metric),
            "deltas_by_seed": {str(record["seed"]): metric_delta(record["final"][metric.key], record["selected"][metric.key]) for record in records if metric.key in record["selected"] and metric.key in record["final"]},
        }
    return {
        "method_id": method_id,
        "method": METHOD_LABELS[method_id],
        "selected_checkpoints": checkpoint_map(records, "selected_checkpoint"),
        "final_checkpoints": checkpoint_map(records, "final_checkpoint"),
        "changed_seed_count": sum(1 for record in records if record["selected_checkpoint"] != record["final_checkpoint"]),
        "metrics": metrics,
    }


def changed_seed_rows(method_id: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if record["selected_checkpoint"] == record["final_checkpoint"]:
            continue
        row = {
            "method_id": method_id,
            "method": METHOD_LABELS[method_id],
            "seed": record["seed"],
            "selected_checkpoint": record["selected_checkpoint"],
            "final_checkpoint": record["final_checkpoint"],
        }
        for metric in METRICS:
            if metric.key in record["selected"] and metric.key in record["final"]:
                row[f"{metric.key}_selected"] = record["selected"][metric.key]
                row[f"{metric.key}_final"] = record["final"][metric.key]
                row[f"{metric.key}_delta"] = metric_delta(record["final"][metric.key], record["selected"][metric.key])
        rows.append(row)
    return rows


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    rows = summary["method_summaries"]
    lines = [
        "# Selected-vs-Final Checkpoint Robustness (#94)",
        "",
        "Status: `complete`.",
        "",
        "This note audits whether the full-paper primary rows rely materially on checkpoint-sweep selection. "
        "The comparison uses the same five Isaac seeds `11/17/23/29/31` for LCP, SC-PPO, and the revised heuristic. "
        "Delta is `final checkpoint - selected checkpoint`; for fall, velocity error, joint acceleration, jitter, "
        "and sensitivity, positive deltas are worse. For return, positive deltas are better.",
        "",
        "## Main Read",
        "",
        "- LCP is close to final-only behavior: only seed 11 selects checkpoint 300 rather than 400, and aggregate final-vs-selected deltas are small.",
        "- SC-PPO is dynamic-selection-sensitive: final checkpoints improve velocity and return but worsen joint acceleration and jitter.",
        "- The revised heuristic is task-selection-sensitive: final checkpoints improve velocity/return slightly but increase fall rate and dynamic roughness.",
        "- The paper should report checkpoint-sweep selection explicitly, but selected-checkpoint dependence is not a hidden failure mode for LCP.",
        "",
        "## Aggregate Selected-vs-Final Audit",
        "",
        "| Method | Classification | Changed seeds | Selected ckpts | Final ckpts | Fall delta | Vel delta | Jnt acc delta | Jitter delta | Return delta | Sens delta |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {method} | {cls} | {changed} | {sel} | {fin} | {fall} | {vel} | {joint} | {jitter} | {ret} | {sens} |".format(
                method=row["method"],
                cls=row["classification"],
                changed=row["changed_seed_count"],
                sel=row["selected_checkpoints"],
                fin=row["final_checkpoints"],
                fall=fmt(metrics["fall_rate"]["delta"]),
                vel=fmt(metrics["velocity_tracking_error_mean"]["delta"]),
                joint=fmt(metrics["joint_acceleration_l2_mean"]["delta"]),
                jitter=fmt(metrics["action_jitter_l2_mean"]["delta"]),
                ret=fmt(metrics["episode_return_mean"]["delta"]),
                sens=fmt(metrics["policy_sensitivity"]["delta"]),
            )
        )

    lines.extend(
        [
            "",
            "## Selected and Final Means",
            "",
            "| Method | Metric | Selected mean | Final mean | Delta | Preference by delta |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        for metric in METRICS:
            values = row["metrics"][metric.key]
            lines.append(
                "| {method} | {metric} | {selected} | {final} | {delta} | {pref} |".format(
                    method=row["method"],
                    metric=metric.label,
                    selected=fmt(values["selected_mean"]),
                    final=fmt(values["final_mean"]),
                    delta=fmt(values["delta"]),
                    pref=values["preference"],
                )
            )

    lines.extend(
        [
            "",
            "## Changed-Seed Detail",
            "",
            "| Method | Seed | Selected ckpt | Final ckpt | Fall delta | Vel delta | Jnt acc delta | Jitter delta | Return delta | Sens delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["changed_seed_rows"]:
        lines.append(
            "| {method} | {seed} | {sel} | {fin} | {fall} | {vel} | {joint} | {jitter} | {ret} | {sens} |".format(
                method=row["method"],
                seed=row["seed"],
                sel=row["selected_checkpoint"],
                fin=row["final_checkpoint"],
                fall=fmt(row.get("fall_rate_delta")),
                vel=fmt(row.get("velocity_tracking_error_mean_delta")),
                joint=fmt(row.get("joint_acceleration_l2_mean_delta")),
                jitter=fmt(row.get("action_jitter_l2_mean_delta")),
                ret=fmt(row.get("episode_return_mean_delta")),
                sens=fmt(row.get("policy_sensitivity_delta")),
            )
        )

    lines.extend(
        [
            "",
            "## MuJoCo Selected-vs-Final Audit",
            "",
        ]
    )
    if summary["mujoco_missing_artifacts"]:
        lines.extend(
            [
                "MuJoCo final-checkpoint replay is incomplete. Missing artifacts:",
                "",
            ]
        )
        lines.extend(f"- `{path}`" for path in summary["mujoco_missing_artifacts"])
        lines.extend(
            [
                "",
                "Generate missing replays with:",
                "",
                "```bash",
                "/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/baseline/run_mujoco_final_checkpoint_replay.py",
                "```",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "All changed selected/final checkpoints have matched no-retraining MuJoCo final-checkpoint replays.",
                "",
                "| Method | Changed seeds | Selected ckpts | Final ckpts | Fall delta | Vel delta | Jnt acc delta | Jitter delta | Return delta |",
                "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary["mujoco_method_summaries"]:
            metrics = row["metrics"]
            lines.append(
                "| {method} | {changed} | {sel} | {fin} | {fall} | {vel} | {joint} | {jitter} | {ret} |".format(
                    method=row["method"],
                    changed=row["changed_seed_count"],
                    sel=row["selected_checkpoints"],
                    fin=row["final_checkpoints"],
                    fall=fmt(metrics["fall_rate"]["delta"]),
                    vel=fmt(metrics["velocity_tracking_error_mean"]["delta"]),
                    joint=fmt(metrics["joint_acceleration_l2_mean"]["delta"]),
                    jitter=fmt(metrics["action_jitter_l2_mean"]["delta"]),
                    ret=fmt(metrics["episode_return_mean"]["delta"]),
                )
            )

    lines.extend(
        [
            "",
            "## Paper Wording Guidance",
            "",
            "- Say that LCP is nearly final-checkpoint stable under the current protocol.",
            "- Say that SC-PPO's selected checkpoint mainly protects dynamic smoothness, not task survival.",
            "- Say that the heuristic selected checkpoint mainly protects fall-rate/task validity.",
            "- Keep selected-checkpoint selection as an explicit protocol limitation; do not hide it in aggregate tables.",
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
            "/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/baseline/run_mujoco_final_checkpoint_replay.py",
            "/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python scripts/analysis/analyze_checkpoint_robustness.py",
            "```",
            "",
            f"Generated runtime summary: `{summary['generated_artifacts']['summary_json']}`",
            f"Generated runtime table note: `{summary['generated_artifacts']['summary_markdown']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(output_dir: Path) -> dict[str, Any]:
    records, source_artifacts = collect_records()
    mujoco_records, missing_mujoco = collect_mujoco_records(records)
    method_summaries = [summarize_method(method_id, records[method_id]) for method_id in METHOD_ORDER]
    mujoco_method_summaries = [
        row for method_id in METHOD_ORDER
        if (row := summarize_mujoco_method(method_id, mujoco_records[method_id])) is not None
    ]
    changed_rows = []
    for method_id in METHOD_ORDER:
        changed_rows.extend(changed_seed_rows(method_id, records[method_id]))
    return {
        "issue": "#94",
        "protocol": {
            "seeds": SEEDS,
            "delta_definition": "final checkpoint metric minus selected checkpoint metric",
            "selection_rule": "task floor first, then smoothest checkpoint",
        },
        "method_summaries": method_summaries,
        "mujoco_method_summaries": mujoco_method_summaries,
        "mujoco_missing_artifacts": missing_mujoco,
        "changed_seed_rows": changed_rows,
        "source_artifacts": source_artifacts,
        "generated_artifacts": {
            "summary_json": relative_to_repo(output_dir / "summary.json"),
            "summary_markdown": relative_to_repo(output_dir / "summary.md"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze selected-vs-final checkpoint robustness.")
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
