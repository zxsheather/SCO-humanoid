#!/usr/bin/env python3
"""Run and summarize no-retraining MuJoCo actuator-bandwidth robustness sweeps."""

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
    EVALUATE_MUJOCO,
    METRIC_KEYS,
    REPO_ROOT,
    SelectedRun,
    artifact_dir,
    read_json,
    relative,
    selected_runs,
    write_json,
)


DEFAULT_SWEEP = REPO_ROOT / "configs" / "sweeps" / "actuator_latency_robustness.json"
NOMINAL_OUTPUT_NAME = "metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"
EXTRA_METRIC_KEYS = ["applied_action_jitter_l2_mean", "action_lag_l2_mean"]


def default_python_bin() -> str:
    candidate = Path("/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python")
    return str(candidate) if candidate.exists() else sys.executable


def tau_tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def output_name_for(tau: float) -> str:
    if abs(tau) < 1e-12:
        return NOMINAL_OUTPUT_NAME
    return f"metrics_mujoco_actuator_lowpass_tau{tau_tag(tau)}_20ep_20s_noise01.json"


def run_name_for(run: SelectedRun, tau: float) -> str:
    if abs(tau) < 1e-12:
        return run.base_run_name
    return f"{run.base_run_name}_actuator_lowpass_tau{tau_tag(tau)}"


def output_path_for(run: SelectedRun, tau: float) -> Path:
    return artifact_dir(run.config_path, run_name_for(run, tau)) / output_name_for(tau)


def artifact_ok(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        json.load(handle)
    return True


def parse_csv_strings(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def parse_float_csv(value: str | None, default: list[float]) -> list[float]:
    if value is None:
        return default
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_command(args: argparse.Namespace, sweep: dict[str, Any], run: SelectedRun, tau: float) -> list[str]:
    mujoco_cfg = sweep["mujoco"]
    mode = "none" if abs(tau) < 1e-12 else "action_lowpass"
    return [
        args.python_bin,
        str(EVALUATE_MUJOCO),
        "--config",
        str(REPO_ROOT / run.config_path),
        "--run-name",
        run_name_for(run, tau),
        "--load-run",
        run.load_run,
        "--checkpoint",
        str(run.checkpoint),
        "--terrain-mode",
        str(mujoco_cfg["terrain_mode"]),
        "--episodes",
        str(args.episodes or mujoco_cfg["episodes"]),
        "--sim-duration",
        str(args.sim_duration or mujoco_cfg["sim_duration"]),
        "--joint-reset-noise",
        str(mujoco_cfg["joint_reset_noise"]),
        "--base-xy-noise",
        str(mujoco_cfg["base_xy_noise"]),
        "--command-vx",
        str(mujoco_cfg["command_vx"]),
        "--command-vy",
        str(mujoco_cfg["command_vy"]),
        "--command-dyaw",
        str(mujoco_cfg["command_dyaw"]),
        "--output-name",
        output_name_for(tau),
        "--manifest-slot",
        f"mujoco_actuator_lowpass_tau{tau_tag(tau)}",
        "--actuator-proxy-mode",
        mode,
        "--actuator-lowpass-time-constant",
        str(tau),
    ]


def run_job(args: argparse.Namespace, command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(command) + "\n\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def collect_rows(runs: list[SelectedRun], taus: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for tau in taus:
            path = output_path_for(run, tau)
            if not path.exists():
                continue
            metrics = read_json(path)
            row = {
                "method_id": run.method_id,
                "method_label": run.method_label,
                "seed": run.seed,
                "checkpoint": run.checkpoint,
                "lowpass_time_constant": float(tau),
                "metrics_path": relative(path),
            }
            for key in [*METRIC_KEYS, *EXTRA_METRIC_KEYS]:
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


def factor(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0.0:
        return None
    return value / base


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    base_by_method: dict[str, dict[str, float | None]] = {}
    for row in rows:
        grouped.setdefault((str(row["method_id"]), float(row["lowpass_time_constant"])), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (method_id, tau), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        out: dict[str, Any] = {
            "method_id": method_id,
            "method_label": group[0]["method_label"],
            "lowpass_time_constant": tau,
            "seed_count": len({row["seed"] for row in group}),
        }
        for key in [*METRIC_KEYS, *EXTRA_METRIC_KEYS]:
            values = [row[key] for row in group if isinstance(row.get(key), (int, float))]
            out[key] = mean_std(values)
        if abs(tau) < 1e-12:
            base_by_method[method_id] = {
                key: out[key]["mean"] if isinstance(out.get(key), dict) else None
                for key in [*METRIC_KEYS, *EXTRA_METRIC_KEYS]
            }
        aggregates.append(out)

    for out in aggregates:
        base = base_by_method.get(str(out["method_id"]), {})
        out["relative_to_tau0"] = {
            key: factor(out[key]["mean"], base.get(key)) if isinstance(out.get(key), dict) else None
            for key in [
                "joint_acceleration_l2_mean",
                "action_jitter_l2_mean",
                "applied_action_jitter_l2_mean",
                "episode_return_mean",
            ]
        }
        out["delta_to_tau0"] = {
            key: (
                out[key]["mean"] - base.get(key)
                if isinstance(out.get(key), dict)
                and isinstance(out[key].get("mean"), (int, float))
                and isinstance(base.get(key), (int, float))
                else None
            )
            for key in ["fall_rate", "velocity_tracking_error_mean", "episode_return_mean"]
        }
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_id",
        "method_label",
        "seed",
        "checkpoint",
        "lowpass_time_constant",
        *METRIC_KEYS,
        *EXTRA_METRIC_KEYS,
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
    lines = [
        "# Actuator-Bandwidth Robustness Sweep",
        "",
        f"Issue: {sweep.get('issue', '#97')}",
        "",
        "This is a no-retraining MuJoCo actuator-path stress grid over first-order action low-pass time constants. It is simulator-side robustness evidence, not calibrated hardware validation or sim-to-real proof.",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | Tau | Seeds | Fall | Vel. err | Jnt acc | Jnt acc factor | Raw jitter | Raw jitter factor | Applied jitter | Action lag | Return |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        rel = row["relative_to_tau0"]
        lines.append(
            "| {method} | {tau} | {seeds} | {fall} | {vel} | {jacc} | {jfac} | {jit} | {jifac} | {appjit} | {lag} | {ret} |".format(
                method=row["method_label"],
                tau=fmt(row["lowpass_time_constant"], 2),
                seeds=row["seed_count"],
                fall=fmt(row["fall_rate"]["mean"]),
                vel=fmt(row["velocity_tracking_error_mean"]["mean"]),
                jacc=fmt(row["joint_acceleration_l2_mean"]["mean"]),
                jfac=fmt(rel["joint_acceleration_l2_mean"]),
                jit=fmt(row["action_jitter_l2_mean"]["mean"]),
                jifac=fmt(rel["action_jitter_l2_mean"]),
                appjit=fmt(row["applied_action_jitter_l2_mean"]["mean"]),
                lag=fmt(row["action_lag_l2_mean"]["mean"]),
                ret=fmt(row["episode_return_mean"]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- Treat the sweep as a bounded actuator-bandwidth stress test.",
            "- Do not describe the result as hardware transfer or calibrated actuator modeling.",
            "- Compare degradation factors to each method's own tau=0 replay; do not rank methods by low-pass metrics alone if task validity changes.",
            "",
            "## Artifacts",
            "",
            f"- Per-run CSV: `{relative(path.parent / 'table_actuator_latency_robustness.csv')}`",
            f"- Summary JSON: `{relative(path.parent / 'summary.json')}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run selected-checkpoint actuator-bandwidth robustness sweep.")
    parser.add_argument("--sweep-config", default=str(DEFAULT_SWEEP))
    parser.add_argument("--python-bin", default=default_python_bin())
    parser.add_argument("--methods", default=None, help="Comma-separated subset of lcp,scppo38,heuristic.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed subset.")
    parser.add_argument("--taus", default=None, help="Comma-separated low-pass time constants.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--sim-duration", type=float, default=None)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    sweep = read_json(args.sweep_config)
    methods = parse_csv_strings(args.methods)
    seeds = {int(seed) for seed in parse_csv_strings(args.seeds) or []} or None
    taus = parse_float_csv(args.taus, [float(value) for value in sweep["lowpass_time_constants"]])
    runs = [
        run for run in selected_runs(sweep)
        if (methods is None or run.method_id in methods)
        and (seeds is None or run.seed in seeds)
    ]

    analysis_root = REPO_ROOT / sweep["analysis_root"]
    log_dir = analysis_root / "logs"
    failures = 0
    for run in runs:
        for tau in taus:
            path = output_path_for(run, tau)
            if artifact_ok(path):
                print(
                    f"skip {run.method_id} seed={run.seed} tau={tau}: {relative(path)}",
                    flush=True,
                )
                continue
            if abs(tau) < 1e-12:
                print(f"Missing nominal tau=0 replay: {relative(path)}", file=sys.stderr)
                failures += 1
                continue
            if args.summarize_only:
                failures += 1
                continue
            command = build_command(args, sweep, run, tau)
            log_path = log_dir / f"{run.method_id}_seed{run.seed}_tau{tau_tag(tau)}.log"
            if args.dry_run:
                print("$ " + " ".join(command))
                continue
            print(
                f"run {run.method_id} seed={run.seed} tau={tau} -> {relative(path)}",
                flush=True,
            )
            exit_code = run_job(args, command, log_path)
            if exit_code != 0 and not artifact_ok(path):
                print(f"Failed {run.method_id} seed{run.seed} tau={tau}; see {relative(log_path)}", file=sys.stderr)
                failures += 1

    rows = collect_rows(runs, taus)
    aggregates = aggregate_rows(rows)
    write_csv(analysis_root / "table_actuator_latency_robustness.csv", rows)
    write_json(
        analysis_root / "summary.json",
        {
            "issue": sweep.get("issue", "#97"),
            "claim_boundary": sweep.get("claim_boundary"),
            "rows": rows,
            "aggregates": aggregates,
        },
    )
    write_summary_md(analysis_root / "summary.md", sweep, rows, aggregates)
    print(f"Wrote {relative(analysis_root / 'summary.json')}")
    print(f"Wrote {relative(analysis_root / 'summary.md')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
