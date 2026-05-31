#!/usr/bin/env python3
"""Run missing final-checkpoint MuJoCo replays for the full-paper methods."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_observation_noise_robustness_sweep import (  # noqa: E402
    EVALUATE_MUJOCO,
    REPO_ROOT,
    SelectedRun,
    artifact_dir,
    load_run_from_summary,
    read_json,
    relative,
    write_json,
)


DEFAULT_SWEEP = REPO_ROOT / "configs" / "sweeps" / "observation_noise_robustness.json"
OUTPUT_NAME_TEMPLATE = "metrics_mujoco_final_checkpoint_{checkpoint}_20ep_20s_noise01.json"
MANIFEST_SLOT_TEMPLATE = "mujoco_final_checkpoint_{checkpoint}_20ep_20s_noise01"


@dataclass(frozen=True)
class FinalRun(SelectedRun):
    final_checkpoint: int


def default_python_bin() -> str:
    candidate = Path("/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python")
    return str(candidate) if candidate.exists() else sys.executable


def final_runs(sweep: dict[str, Any]) -> list[FinalRun]:
    source_paths = sweep["source_summaries"]
    lcp_summary = read_json(source_paths["lcp"])
    extended_summary = read_json(source_paths["extended_seeds"])
    methods = {method["id"]: method for method in sweep["methods"]}
    runs: list[FinalRun] = []

    lcp_method = methods["lcp"]
    for record in lcp_summary["per_seed"]:
        runs.append(
            FinalRun(
                method_id="lcp",
                method_label=lcp_method["label"],
                config_path=lcp_method["config"],
                seed=int(record["seed"]),
                base_run_name=str(record["run_name"]),
                load_run=load_run_from_summary(record["summary_path"]),
                checkpoint=int(record["selected_checkpoint"]),
                final_checkpoint=int(record["final"]["checkpoint"]),
            )
        )

    candidate_by_id = {candidate["id"]: candidate for candidate in extended_summary["candidates"]}
    for method_id in ["scppo38", "heuristic"]:
        method = methods[method_id]
        candidate = candidate_by_id[method["source_candidate_id"]]
        for seed_text, record in candidate["per_seed"].items():
            runs.append(
                FinalRun(
                    method_id=method_id,
                    method_label=method["label"],
                    config_path=method["config"],
                    seed=int(seed_text),
                    base_run_name=str(record["run_name"]),
                    load_run=Path(record["load_run"]).name,
                    checkpoint=int(record["selected_checkpoint"]),
                    final_checkpoint=int(record["final_checkpoint"]),
                )
            )

    seeds = {int(seed) for seed in sweep["seeds"]}
    return [run for run in runs if run.seed in seeds]


def output_name(checkpoint: int) -> str:
    return OUTPUT_NAME_TEMPLATE.format(checkpoint=int(checkpoint))


def manifest_slot(checkpoint: int) -> str:
    return MANIFEST_SLOT_TEMPLATE.format(checkpoint=int(checkpoint))


def final_run_name(run: FinalRun) -> str:
    return f"{run.base_run_name}_finalcp{run.final_checkpoint}_mujoco"


def selected_output_path(run: FinalRun) -> Path:
    return artifact_dir(run.config_path, run.base_run_name) / "metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"


def final_output_path(run: FinalRun) -> Path:
    if run.final_checkpoint == run.checkpoint:
        return selected_output_path(run)
    return artifact_dir(run.config_path, final_run_name(run)) / output_name(run.final_checkpoint)


def artifact_ok(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        json.load(handle)
    return True


def build_command(args: argparse.Namespace, sweep: dict[str, Any], run: FinalRun) -> list[str]:
    mujoco_cfg = sweep["mujoco"]
    return [
        args.python_bin,
        str(EVALUATE_MUJOCO),
        "--config",
        str(REPO_ROOT / run.config_path),
        "--run-name",
        final_run_name(run),
        "--load-run",
        run.load_run,
        "--checkpoint",
        str(run.final_checkpoint),
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
        output_name(run.final_checkpoint),
        "--manifest-slot",
        manifest_slot(run.final_checkpoint),
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


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = {
        "issue": "#94",
        "protocol": "No-retraining final-checkpoint MuJoCo replay for changed selected/final checkpoints.",
        "rows": rows,
    }
    write_json(path, payload)


def parse_csv_strings(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final-checkpoint MuJoCo replays for changed selected checkpoints.")
    parser.add_argument("--sweep-config", default=str(DEFAULT_SWEEP))
    parser.add_argument("--python-bin", default=default_python_bin())
    parser.add_argument("--methods", default=None, help="Comma-separated subset of lcp,scppo38,heuristic.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed subset.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--sim-duration", type=float, default=None)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    sweep = read_json(args.sweep_config)
    methods = parse_csv_strings(args.methods)
    seeds = {int(seed) for seed in parse_csv_strings(args.seeds) or []} or None
    runs = [
        run for run in final_runs(sweep)
        if (methods is None or run.method_id in methods)
        and (seeds is None or run.seed in seeds)
        and run.final_checkpoint != run.checkpoint
    ]

    output_root = REPO_ROOT / "artifacts" / "analysis" / "mujoco_final_checkpoint_replay"
    log_dir = output_root / "logs"
    rows: list[dict[str, Any]] = []
    failures = 0
    for run in runs:
        path = final_output_path(run)
        row = {
            "method_id": run.method_id,
            "method_label": run.method_label,
            "seed": run.seed,
            "selected_checkpoint": run.checkpoint,
            "final_checkpoint": run.final_checkpoint,
            "run_name": final_run_name(run),
            "metrics_path": relative(path),
            "status": "pending",
        }
        if artifact_ok(path):
            row["status"] = "exists"
            rows.append(row)
            continue
        if args.summarize_only:
            row["status"] = "missing"
            rows.append(row)
            failures += 1
            continue
        command = build_command(args, sweep, run)
        log_path = log_dir / f"{run.method_id}_seed{run.seed}_finalcp{run.final_checkpoint}.log"
        if args.dry_run:
            row["status"] = "dry_run"
            row["command"] = command
            row["log_path"] = relative(log_path)
            rows.append(row)
            continue
        exit_code = run_job(args, command, log_path)
        row["exit_code"] = exit_code
        row["log_path"] = relative(log_path)
        if exit_code == 0 or artifact_ok(path):
            row["status"] = "completed" if exit_code == 0 else "completed_with_nonzero_exit"
        else:
            row["status"] = "failed"
            failures += 1
        rows.append(row)

    write_summary(output_root / "summary.json", rows)
    print(f"Wrote {relative(output_root / 'summary.json')}")
    for row in rows:
        print(
            f"- {row['method_id']} seed{row['seed']} selected={row['selected_checkpoint']} "
            f"final={row['final_checkpoint']} status={row['status']} path={row['metrics_path']}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
