#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = REPO_ROOT / "configs" / "sweeps" / "observation_noise_robustness.json"
EVALUATE_POLICY = REPO_ROOT / "scripts" / "baseline" / "evaluate_policy.py"
EVALUATE_MUJOCO = REPO_ROOT / "scripts" / "baseline" / "evaluate_mujoco_sim2sim.py"
METRIC_KEYS = [
    "fall_rate",
    "velocity_tracking_error_mean",
    "joint_acceleration_l2_mean",
    "action_jitter_l2_mean",
    "episode_return_mean",
]
SIDE_READ_KEYS = [
    "policy_local_sensitivity_cost_mean",
    "constraint_violation_rate",
]


@dataclass(frozen=True)
class SelectedRun:
    method_id: str
    method_label: str
    config_path: str
    seed: int
    base_run_name: str
    load_run: str
    checkpoint: int


def read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    with candidate.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def relative(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(candidate)


def noise_tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def load_run_from_summary(summary_path: str | Path) -> str:
    summary = read_json(summary_path)
    latest = Path(summary["latest_checkpoint_path"])
    return latest.parent.name


def selected_runs(sweep: dict[str, Any]) -> list[SelectedRun]:
    source_paths = sweep["source_summaries"]
    lcp_summary = read_json(source_paths["lcp"])
    extended_summary = read_json(source_paths["extended_seeds"])
    methods = {method["id"]: method for method in sweep["methods"]}
    runs: list[SelectedRun] = []

    lcp_method = methods["lcp"]
    for record in lcp_summary["per_seed"]:
        runs.append(
            SelectedRun(
                method_id="lcp",
                method_label=lcp_method["label"],
                config_path=lcp_method["config"],
                seed=int(record["seed"]),
                base_run_name=str(record["run_name"]),
                load_run=load_run_from_summary(record["summary_path"]),
                checkpoint=int(record["selected_checkpoint"]),
            )
        )

    candidate_by_id = {candidate["id"]: candidate for candidate in extended_summary["candidates"]}
    for method_id in ["scppo38", "heuristic"]:
        method = methods[method_id]
        candidate = candidate_by_id[method["source_candidate_id"]]
        for seed_text, record in candidate["per_seed"].items():
            runs.append(
                SelectedRun(
                    method_id=method_id,
                    method_label=method["label"],
                    config_path=method["config"],
                    seed=int(seed_text),
                    base_run_name=str(record["run_name"]),
                    load_run=Path(record["load_run"]).name,
                    checkpoint=int(record["selected_checkpoint"]),
                )
            )
    seeds = {int(seed) for seed in sweep["seeds"]}
    return [run for run in runs if run.seed in seeds]


def load_config(path: str) -> dict[str, Any]:
    return read_json(path)


def artifact_dir(config_path: str, run_name: str) -> Path:
    config = load_config(config_path)
    return REPO_ROOT / config["artifacts_root"] / run_name


def output_path_for(engine: str, run: SelectedRun, noise: float) -> Path:
    run_name = f"{run.base_run_name}_obsnoise_{noise_tag(noise)}"
    if engine == "isaac":
        return artifact_dir(run.config_path, run_name) / "metrics.json"
    if engine == "mujoco":
        return artifact_dir(run.config_path, run_name) / f"metrics_mujoco_obsnoise_{noise_tag(noise)}.json"
    raise ValueError(f"Unsupported engine: {engine}")


def build_command(args: argparse.Namespace, sweep: dict[str, Any], engine: str, run: SelectedRun, noise: float) -> list[str]:
    run_name = f"{run.base_run_name}_obsnoise_{noise_tag(noise)}"
    obs_noise_seed = int(args.obs_noise_seed_base) + run.seed * 1000 + int(round(noise * 1_000_000))
    if engine == "isaac":
        isaac_cfg = sweep["isaac"]
        return [
            args.python_bin,
            str(EVALUATE_POLICY),
            "--config",
            str(REPO_ROOT / run.config_path),
            "--run-name",
            run_name,
            "--load-run",
            run.load_run,
            "--checkpoint",
            str(run.checkpoint),
            "--episodes",
            str(args.isaac_episodes or isaac_cfg["episodes"]),
            "--num-envs",
            str(args.isaac_num_envs or isaac_cfg["num_envs"]),
            "--obs-noise-std",
            str(noise),
            "--obs-noise-seed",
            str(obs_noise_seed),
        ]
    if engine == "mujoco":
        mujoco_cfg = sweep["mujoco"]
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
            str(mujoco_cfg["terrain_mode"]),
            "--episodes",
            str(args.mujoco_episodes or mujoco_cfg["episodes"]),
            "--sim-duration",
            str(args.mujoco_sim_duration or mujoco_cfg["sim_duration"]),
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
            "--obs-noise-std",
            str(noise),
            "--obs-noise-seed",
            str(obs_noise_seed),
            "--output-name",
            f"metrics_mujoco_obsnoise_{noise_tag(noise)}.json",
            "--manifest-slot",
            f"mujoco_obsnoise_{noise_tag(noise)}",
        ]
    raise ValueError(f"Unsupported engine: {engine}")


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


def collect_rows(sweep: dict[str, Any], runs: list[SelectedRun], engines: list[str], noises: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for engine in engines:
        for run in runs:
            for noise in noises:
                metrics_path = output_path_for(engine, run, noise)
                if not metrics_path.exists():
                    continue
                metrics = read_json(metrics_path)
                row = {
                    "engine": engine,
                    "method_id": run.method_id,
                    "method_label": run.method_label,
                    "seed": run.seed,
                    "checkpoint": run.checkpoint,
                    "noise_std": noise,
                    "metrics_path": relative(metrics_path),
                }
                for key in METRIC_KEYS:
                    value = metrics.get(key)
                    row[key] = float(value) if isinstance(value, (int, float)) else None
                constraint_metrics = metrics.get("constraint_metrics")
                if not isinstance(constraint_metrics, dict):
                    constraint_metrics = {}
                for key in SIDE_READ_KEYS:
                    value = constraint_metrics.get(key)
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
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["engine"], row["method_id"], float(row["noise_std"])), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (engine, method_id, noise), group in sorted(groups.items()):
        out = {
            "engine": engine,
            "method_id": method_id,
            "method_label": group[0]["method_label"],
            "noise_std": noise,
            "seed_count": len({row["seed"] for row in group}),
        }
        for key in [*METRIC_KEYS, *SIDE_READ_KEYS]:
            values = [row[key] for row in group if isinstance(row.get(key), (int, float))]
            out[key] = mean_std(values)
        aggregates.append(out)

    baseline = {
        (row["engine"], row["method_id"], key): row[key]["mean"]
        for row in aggregates
        for key in METRIC_KEYS
        if float(row["noise_std"]) == 0.0 and row[key]["mean"] is not None
    }
    for row in aggregates:
        degradation: dict[str, Any] = {}
        for key in METRIC_KEYS:
            base = baseline.get((row["engine"], row["method_id"], key))
            current = row[key]["mean"]
            if base is None or current is None:
                degradation[key] = None
            elif abs(float(base)) > 1e-12:
                degradation[key] = float(current) / float(base)
            else:
                degradation[key] = None
        row["relative_to_noise0"] = degradation
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["engine", "method_id", "method_label", "seed", "checkpoint", "noise_std", *METRIC_KEYS, *SIDE_READ_KEYS, "metrics_path"]
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
        "# Observation-Noise Robustness Sweep",
        "",
        f"Issue: {sweep.get('issue', '#89')}",
        "",
        "This is a no-retraining selected-checkpoint stress evaluation. Gaussian noise is added only to the policy observation input at inference time.",
        "",
        "## Aggregate Metrics",
        "",
        "| Engine | Method | Noise | Fall | Vel. err | Jnt acc | Jitter | Return | Sens. | Viol. | Jitter/noise0 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {engine} | {method} | {noise} | {fall} | {vel} | {jacc} | {jit} | {ret} | {sens} | {viol} | {jit_rel} |".format(
                engine=row["engine"],
                method=row["method_label"],
                noise=fmt(row["noise_std"], 3),
                fall=fmt(row["fall_rate"]["mean"]),
                vel=fmt(row["velocity_tracking_error_mean"]["mean"]),
                jacc=fmt(row["joint_acceleration_l2_mean"]["mean"]),
                jit=fmt(row["action_jitter_l2_mean"]["mean"]),
                ret=fmt(row["episode_return_mean"]["mean"]),
                sens=fmt(row["policy_local_sensitivity_cost_mean"]["mean"]),
                viol=fmt(row["constraint_violation_rate"]["mean"]),
                jit_rel=fmt(row["relative_to_noise0"]["action_jitter_l2_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- Treat these results as input-perturbation robustness evidence for selected checkpoints.",
            "- Do not treat the sweep as hardware validation, broad simulator robustness, or SOTA superiority.",
            "- The most paper-relevant read is whether degradation under observation noise follows the policy-sensitivity ordering.",
            "",
            "## Artifacts",
            "",
            f"- Per-run CSV: `{relative(path.parent / 'table_observation_noise_robustness.csv')}`",
            f"- Summary JSON: `{relative(path.parent / 'summary.json')}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(args: argparse.Namespace, sweep: dict[str, Any], runs: list[SelectedRun], engines: list[str], noises: list[float]) -> None:
    output_dir = REPO_ROOT / sweep["analysis_root"]
    rows = collect_rows(sweep, runs, engines, noises)
    aggregates = aggregate_rows(rows)
    write_csv(output_dir / "table_observation_noise_robustness.csv", rows)
    payload = {
        "issue": sweep.get("issue"),
        "claim_boundary": sweep.get("claim_boundary"),
        "engines": engines,
        "noise_scales": noises,
        "row_count": len(rows),
        "expected_row_count": len(runs) * len(engines) * len(noises),
        "rows": rows,
        "aggregates": aggregates,
        "generated_artifacts": {
            "summary_json": relative(output_dir / "summary.json"),
            "summary_markdown": relative(output_dir / "summary.md"),
            "table_csv": relative(output_dir / "table_observation_noise_robustness.csv"),
        },
        "rerun_command": " ".join(sys.argv),
    }
    write_json(output_dir / "summary.json", payload)
    write_summary_md(output_dir / "summary.md", sweep, rows, aggregates)


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def default_python_bin() -> str:
    candidate = Path("/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python")
    return str(candidate) if candidate.exists() else sys.executable


def main() -> int:
    parser = argparse.ArgumentParser(description="Run selected-checkpoint observation-noise robustness sweeps.")
    parser.add_argument("--sweep-config", default=str(DEFAULT_SWEEP))
    parser.add_argument("--engines", default=None, help="Comma-separated subset, e.g. isaac,mujoco.")
    parser.add_argument("--noise-scales", default=None, help="Comma-separated noise scales.")
    parser.add_argument("--methods", default=None, help="Comma-separated subset of lcp,scppo38,heuristic.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed subset.")
    parser.add_argument("--python-bin", default=default_python_bin())
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--obs-noise-seed-base", type=int, default=20260531)
    parser.add_argument("--isaac-episodes", type=int, default=None)
    parser.add_argument("--isaac-num-envs", type=int, default=None)
    parser.add_argument("--mujoco-episodes", type=int, default=None)
    parser.add_argument("--mujoco-sim-duration", type=float, default=None)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    sweep = read_json(args.sweep_config)
    engines = parse_csv_strings(args.engines) if args.engines else list(sweep["engines"])
    noises = parse_csv_floats(args.noise_scales) if args.noise_scales else [float(value) for value in sweep["noise_scales"]]
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
    jobs = [(engine, run, noise) for engine in engines for run in runs for noise in noises]
    print(f"jobs={len(jobs)} engines={engines} methods={sorted({run.method_id for run in runs})} seeds={sorted({run.seed for run in runs})} noises={noises}")

    if not args.summarize_only:
        for engine, run, noise in jobs:
            output_path = output_path_for(engine, run, noise)
            log_path = logs_root / f"{engine}_{run.method_id}_seed{run.seed}_obsnoise_{noise_tag(noise)}.log"
            if args.skip_completed and output_path.exists():
                print(f"skip existing {relative(output_path)}")
                continue
            command = build_command(args, sweep, engine, run, noise)
            print(f"run {engine} {run.method_id} seed={run.seed} noise={noise} -> {relative(output_path)}")
            if args.dry_run:
                print("  " + " ".join(command))
                continue
            exit_code = run_job(args, command, log_path)
            if exit_code != 0 and output_path.exists():
                print(f"recovered non-zero exit={exit_code}; found {relative(output_path)}")
            if not output_path.exists():
                failures.append(
                    {
                        "engine": engine,
                        "method_id": run.method_id,
                        "seed": run.seed,
                        "noise_std": noise,
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

    summarize(args, sweep, runs, engines, noises)
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
