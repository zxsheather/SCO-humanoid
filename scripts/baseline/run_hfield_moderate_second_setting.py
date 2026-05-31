#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_observation_noise_robustness_sweep import (  # noqa: E402
    REPO_ROOT,
    EVALUATE_MUJOCO,
    METRIC_KEYS,
    SelectedRun,
    artifact_dir,
    read_json,
    relative,
    selected_runs,
    write_json,
)


DEFAULT_SWEEP = REPO_ROOT / "configs" / "sweeps" / "hfield_moderate_second_setting.json"
SENSITIVITY_KEY = "policy_local_sensitivity_cost_mean"


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def default_python_bin() -> str:
    candidate = Path("/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python")
    return str(candidate) if candidate.exists() else sys.executable


def output_path_for(sweep: dict[str, Any], run: SelectedRun) -> Path:
    run_name = f"{run.base_run_name}_hfield_moderate"
    return artifact_dir(run.config_path, run_name) / str(sweep["output_name"])


def selected_metrics_path(run: SelectedRun) -> Path:
    return artifact_dir(run.config_path, run.base_run_name) / "metrics_selected.json"


def selected_policy_sensitivity(run: SelectedRun) -> float | None:
    path = selected_metrics_path(run)
    if not path.exists():
        return None
    metrics = read_json(path)
    constraint_metrics = metrics.get("constraint_metrics")
    if not isinstance(constraint_metrics, dict):
        return None
    value = constraint_metrics.get(SENSITIVITY_KEY)
    return float(value) if isinstance(value, (int, float)) else None


def build_command(args: argparse.Namespace, sweep: dict[str, Any], run: SelectedRun) -> list[str]:
    run_name = f"{run.base_run_name}_hfield_moderate"
    return [
        args.python_bin,
        str(EVALUATE_MUJOCO),
        "--config",
        str(REPO_ROOT / run.config_path),
        "--run-name",
        run_name,
        "--load-run",
        run.load_run,
        "--checkpoint",
        str(run.checkpoint),
        "--terrain-mode",
        str(sweep["terrain_mode"]),
        "--episodes",
        str(args.episodes or sweep["episodes"]),
        "--sim-duration",
        str(args.sim_duration or sweep["sim_duration"]),
        "--joint-reset-noise",
        str(sweep["joint_reset_noise"]),
        "--base-xy-noise",
        str(sweep["base_xy_noise"]),
        "--command-vx",
        str(sweep["command_vx"]),
        "--command-vy",
        str(sweep["command_vy"]),
        "--command-dyaw",
        str(sweep["command_dyaw"]),
        "--output-name",
        str(sweep["output_name"]),
        "--manifest-slot",
        str(sweep["manifest_slot"]),
    ]


def run_job(args: argparse.Namespace, command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(command) + "\n\n")
        log_handle.flush()
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(proc.returncode)


def collect_rows(sweep: dict[str, Any], runs: list[SelectedRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics_path = output_path_for(sweep, run)
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        row = {
            "method_id": run.method_id,
            "method_label": run.method_label,
            "seed": run.seed,
            "checkpoint": run.checkpoint,
            "terrain_mode": sweep["terrain_mode"],
            "metrics_path": relative(metrics_path),
            SENSITIVITY_KEY: selected_policy_sensitivity(run),
        }
        for key in METRIC_KEYS:
            value = metrics.get(key)
            row[key] = float(value) if isinstance(value, (int, float)) else None
        rows.append(row)
    return rows


def mean_std(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["method_id"]), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for method_id, group in sorted(groups.items()):
        out: dict[str, Any] = {
            "method_id": method_id,
            "method_label": group[0]["method_label"],
            "terrain_mode": group[0]["terrain_mode"],
            "seed_count": len({row["seed"] for row in group}),
        }
        for key in [*METRIC_KEYS, SENSITIVITY_KEY]:
            values = [row[key] for row in group if isinstance(row.get(key), (int, float))]
            out[key] = mean_std(values)
        aggregates.append(out)
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "terrain_mode",
        "method_id",
        "method_label",
        "seed",
        "checkpoint",
        *METRIC_KEYS,
        SENSITIVITY_KEY,
        "metrics_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_summary_md(path: Path, sweep: dict[str, Any], rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
    reading = interpret_result(rows, aggregates)
    lines = [
        "# hfield_moderate Second-Setting Validation",
        "",
        f"Issue: {sweep.get('issue', '#92')}",
        "",
        "This is a no-retraining selected-checkpoint MuJoCo terrain validation. It replays the same H1 policies on `hfield_moderate` and should be read as a repair-stage generality check.",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | Seeds | Fall | Vel. err | Jnt acc | Jitter | Return | Sensitivity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {method} | {seeds} | {fall} | {vel} | {jacc} | {jit} | {ret} | {sens} |".format(
                method=row["method_label"],
                seeds=row["seed_count"],
                fall=fmt(row["fall_rate"]["mean"]),
                vel=fmt(row["velocity_tracking_error_mean"]["mean"]),
                jacc=fmt(row["joint_acceleration_l2_mean"]["mean"]),
                jit=fmt(row["action_jitter_l2_mean"]["mean"]),
                ret=fmt(row["episode_return_mean"]["mean"]),
                sens=fmt(row[SENSITIVITY_KEY]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Result Reading",
            "",
            reading["summary"],
            "",
            "",
            "## Interpretation Guard",
            "",
            "- Treat this as a controlled terrain generality check for selected checkpoints.",
            "- Do not describe it as a broad locomotion benchmark or hardware validation.",
            "- If fall rates are high, metric orderings are diagnostic rather than claim-grade method rankings.",
            "",
            "## Artifacts",
            "",
            f"- Per-run CSV: `{relative(path.parent / 'table_hfield_moderate_second_setting.csv')}`",
            f"- Summary JSON: `{relative(path.parent / 'summary.json')}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_method(aggregates: list[dict[str, Any]], key: str, higher_is_better: bool = False) -> str | None:
    candidates = [
        (row["method_id"], row[key]["mean"])
        for row in aggregates
        if isinstance(row.get(key), dict) and isinstance(row[key].get("mean"), (int, float))
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[1], reverse=higher_is_better)[0][0]


def interpret_result(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    if not aggregates:
        return {
            "status": "incomplete",
            "summary": "No completed hfield_moderate rows are available yet.",
        }
    collapsed = [
        row["method_id"]
        for row in aggregates
        if isinstance(row.get("fall_rate"), dict)
        and isinstance(row["fall_rate"].get("mean"), (int, float))
        and float(row["fall_rate"]["mean"]) >= 0.95
    ]
    best_jitter = best_method(aggregates, "action_jitter_l2_mean")
    best_jacc = best_method(aggregates, "joint_acceleration_l2_mean")
    best_return = best_method(aggregates, "episode_return_mean", higher_is_better=True)

    if len(collapsed) == len(aggregates):
        status = "weakens_generality"
        summary = (
            "All completed methods collapse on the moderated terrain setting. This weakens any "
            "multi-terrain generality claim and should be reported as a negative protocol result, "
            "not as a method ranking."
        )
    elif best_jitter == "lcp" and (best_jacc == "heuristic" or best_return == "heuristic"):
        status = "supports_metric_split"
        summary = (
            "The second setting supports the paper's metric-split conclusion: the LCP-style row is "
            "best on policy-output action jitter, while at least one downstream closed-loop metric "
            "still favors the heuristic."
        )
    elif best_jitter == "lcp":
        status = "supports_policy_output_mechanism"
        summary = (
            "The second setting supports the policy-output part of the mechanism conclusion because "
            "the LCP-style row has the lowest action jitter. Downstream metric ordering should be "
            "read directly from the table."
        )
    else:
        status = "mixed_or_unchanged"
        summary = (
            "The second setting is mixed and does not materially strengthen the primary mechanism "
            "claim. Use it as diagnostic generality evidence and avoid promoting a broad terrain "
            "benchmark claim."
        )
    return {
        "status": status,
        "summary": summary,
        "collapsed_method_ids": collapsed,
        "best_action_jitter_method": best_jitter,
        "best_joint_acceleration_method": best_jacc,
        "best_return_method": best_return,
        "completed_rows": len(rows),
    }


def summarize(args: argparse.Namespace, sweep: dict[str, Any], runs: list[SelectedRun]) -> None:
    output_dir = REPO_ROOT / sweep["analysis_root"]
    rows = collect_rows(sweep, runs)
    aggregates = aggregate_rows(rows)
    reading = interpret_result(rows, aggregates)
    write_csv(output_dir / "table_hfield_moderate_second_setting.csv", rows)
    payload = {
        "issue": sweep.get("issue"),
        "claim_boundary": sweep.get("claim_boundary"),
        "terrain_mode": sweep["terrain_mode"],
        "row_count": len(rows),
        "expected_row_count": len(runs),
        "rows": rows,
        "aggregates": aggregates,
        "result_reading": reading,
        "generated_artifacts": {
            "summary_json": relative(output_dir / "summary.json"),
            "summary_markdown": relative(output_dir / "summary.md"),
            "table_csv": relative(output_dir / "table_hfield_moderate_second_setting.csv"),
        },
        "rerun_command": " ".join(sys.argv),
    }
    write_json(output_dir / "summary.json", payload)
    write_summary_md(output_dir / "summary.md", sweep, rows, aggregates)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MuJoCo hfield_moderate second-setting validation.")
    parser.add_argument("--sweep-config", default=str(DEFAULT_SWEEP))
    parser.add_argument("--methods", default=None, help="Comma-separated subset of lcp,scppo38,heuristic.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed subset.")
    parser.add_argument("--python-bin", default=default_python_bin())
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--sim-duration", type=float, default=None)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    sweep = read_json(args.sweep_config)
    method_filter = set(parse_csv_strings(args.methods)) if args.methods else None
    seed_filter = {int(seed) for seed in parse_csv_strings(args.seeds)} if args.seeds else None
    runs = selected_runs(sweep)
    if method_filter:
        runs = [run for run in runs if run.method_id in method_filter]
    if seed_filter:
        runs = [run for run in runs if run.seed in seed_filter]

    analysis_root = REPO_ROOT / sweep["analysis_root"]
    logs_root = analysis_root / "logs"
    failures: list[dict[str, Any]] = []
    print(f"jobs={len(runs)} terrain={sweep['terrain_mode']} methods={sorted({run.method_id for run in runs})} seeds={sorted({run.seed for run in runs})}")

    if not args.summarize_only:
        for run in runs:
            output_path = output_path_for(sweep, run)
            log_path = logs_root / f"{run.method_id}_seed{run.seed}.log"
            if args.skip_completed and output_path.exists():
                print(f"skip existing {relative(output_path)}")
                continue
            command = build_command(args, sweep, run)
            print(f"run {run.method_id} seed={run.seed} -> {relative(output_path)}")
            if args.dry_run:
                print("  " + " ".join(command))
                continue
            exit_code = run_job(args, command, log_path)
            if exit_code != 0 and output_path.exists():
                print(f"recovered non-zero exit={exit_code}; found {relative(output_path)}")
            if not output_path.exists():
                failures.append(
                    {
                        "method_id": run.method_id,
                        "seed": run.seed,
                        "exit_code": exit_code,
                        "log_path": relative(log_path),
                        "expected_output": relative(output_path),
                    }
                )
                print(f"FAILED exit={exit_code} log={relative(log_path)}")
            else:
                print(f"wrote {relative(output_path)}")

    if args.dry_run:
        print("dry-run complete; no summary artifacts were written")
        return 0

    summarize(args, sweep, runs)
    if failures:
        write_json(analysis_root / "failures.json", failures)
        print(f"failures={len(failures)}; wrote {relative(analysis_root / 'failures.json')}")
        return 1
    failures_path = analysis_root / "failures.json"
    if failures_path.exists():
        failures_path.unlink()
    print(f"wrote {relative(analysis_root / 'summary.json')}")
    print(f"wrote {relative(analysis_root / 'summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
