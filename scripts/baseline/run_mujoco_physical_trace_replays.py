#!/usr/bin/env python3
"""Run matched MuJoCo trace replays for secondary physical smoothness metrics.

This is the execution slice for issue #117. It replays the current five-seed
selected checkpoints for the three primary methods under the matched
`isaac_mainline` MuJoCo protocol and enables per-timestep trace capture.
"""

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

from _common import artifact_dir, load_config, relative_to_repo, repo_root  # noqa: E402


DEFAULT_PYTHON = Path(sys.executable)
REPO_ROOT = repo_root()
LCP_SUMMARY = REPO_ROOT / "artifacts" / "analysis" / "rough_terrain_lcp_soft_jacobian_formal" / "comparison_summary.json"
EXTENDED_SUMMARY = REPO_ROOT / "artifacts" / "analysis" / "rough_terrain_extended_seeds" / "comparison_summary.json"
LCP_CONFIG = "configs/methods/lcp_soft_jacobian_penalty_diagnostic.json"

METRICS_NAME = "metrics_mujoco_physical_trace_5ep_20s_noise01.json"
TRACE_NAME = "mujoco_physical_trace_5ep_20s_noise01.json"
MANIFEST_SLOT = "mujoco_physical_trace_5ep_20s_noise01"
RUN_SUFFIX = "mujoco_physical_trace_5ep"

SEEDS = [11, 17, 23, 29, 31]


@dataclass(frozen=True)
class TraceReplayRun:
    method_id: str
    method_label: str
    seed: int
    config_path: str
    artifact_run_name: str
    load_run: str
    checkpoint: int

    @property
    def replay_run_name(self) -> str:
        return f"{self.artifact_run_name}_{RUN_SUFFIX}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_by_id(summary: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in summary.get("candidates", []):
        if candidate.get("id") == candidate_id:
            return candidate
    raise KeyError(f"Missing candidate {candidate_id}")


def load_run_name_from_manifest(config_path: str, artifact_run_name: str) -> str:
    manifest_path = artifact_dir(load_config(config_path), artifact_run_name) / "manifest.json"
    manifest = read_json(manifest_path)
    run_dir = manifest.get("run_dir")
    if not isinstance(run_dir, str):
        raise RuntimeError(f"Manifest missing run_dir: {relative_to_repo(manifest_path)}")
    return Path(run_dir).name


def build_runs() -> list[TraceReplayRun]:
    runs: list[TraceReplayRun] = []

    lcp_summary = read_json(LCP_SUMMARY)
    for row in lcp_summary.get("per_seed", []):
        seed = int(row["seed"])
        if seed not in SEEDS:
            continue
        artifact_run_name = str(row["run_name"])
        runs.append(
            TraceReplayRun(
                method_id="lcp",
                method_label="LCP-style soft penalty",
                seed=seed,
                config_path=LCP_CONFIG,
                artifact_run_name=artifact_run_name,
                load_run=load_run_name_from_manifest(LCP_CONFIG, artifact_run_name),
                checkpoint=int(row["selected_checkpoint"]),
            )
        )

    extended_summary = read_json(EXTENDED_SUMMARY)
    for candidate_id, method_id, label in [
        ("sc_ppo", "scppo", "SC-PPO 3.8 PID"),
        ("heuristic_smoothing", "heuristic", "Revised heuristic"),
    ]:
        candidate = candidate_by_id(extended_summary, candidate_id)
        config_path = str(candidate["config_path"])
        for seed in SEEDS:
            record = candidate["per_seed"][str(seed)]
            artifact_run_name = str(record["run_name"])
            runs.append(
                TraceReplayRun(
                    method_id=method_id,
                    method_label=label,
                    seed=seed,
                    config_path=config_path,
                    artifact_run_name=artifact_run_name,
                    load_run=load_run_name_from_manifest(config_path, artifact_run_name),
                    checkpoint=int(record["selected_checkpoint"]),
                )
            )

    return sorted(runs, key=lambda run: (run.method_id, run.seed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched MuJoCo trace replays for physical secondary metrics.")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON), help="Python executable for replay runs.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sim-duration", type=float, default=20.0)
    parser.add_argument("--joint-reset-noise", type=float, default=0.1)
    parser.add_argument("--base-xy-noise", type=float, default=0.0)
    parser.add_argument("--command-vx", type=float, default=0.4)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-dyaw", type=float, default=0.0)
    parser.add_argument("--export-rl-device", default="cpu")
    parser.add_argument("--export-sim-device", default="cpu")
    parser.add_argument("--trace-max-episodes", type=int, default=None)
    parser.add_argument("--trace-max-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345, help="MuJoCo reset RNG seed.")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--only-method", choices=["heuristic", "lcp", "scppo"], default=None)
    parser.add_argument("--only-seed", type=int, choices=SEEDS, default=None)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def expected_paths(run: TraceReplayRun) -> tuple[Path, Path]:
    config = load_config(run.config_path)
    output_dir = artifact_dir(config, run.replay_run_name)
    return output_dir / METRICS_NAME, output_dir / TRACE_NAME


def artifact_ok(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        json.load(handle)
    return True


def run_one(run: TraceReplayRun, args: argparse.Namespace) -> None:
    metrics_path, trace_path = expected_paths(run)
    if args.skip_completed and artifact_ok(metrics_path) and artifact_ok(trace_path):
        print(f"[skip] {run.method_id} seed{run.seed}: {relative_to_repo(metrics_path)}")
        return

    trace_max_episodes = args.episodes if args.trace_max_episodes is None else int(args.trace_max_episodes)
    command = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "baseline" / "evaluate_mujoco_sim2sim.py"),
        "--config",
        str(REPO_ROOT / run.config_path),
        "--run-name",
        run.replay_run_name,
        "--load-run",
        run.load_run,
        "--checkpoint",
        str(run.checkpoint),
        "--export-rl-device",
        str(args.export_rl_device),
        "--export-sim-device",
        str(args.export_sim_device),
        "--terrain-mode",
        "isaac_mainline",
        "--episodes",
        str(args.episodes),
        "--sim-duration",
        str(args.sim_duration),
        "--joint-reset-noise",
        str(args.joint_reset_noise),
        "--base-xy-noise",
        str(args.base_xy_noise),
        "--command-vx",
        str(args.command_vx),
        "--command-vy",
        str(args.command_vy),
        "--command-dyaw",
        str(args.command_dyaw),
        "--seed",
        str(args.seed),
        "--output-name",
        METRICS_NAME,
        "--manifest-slot",
        MANIFEST_SLOT,
        "--capture-traces",
        "--trace-max-episodes",
        str(trace_max_episodes),
        "--trace-max-steps",
        str(args.trace_max_steps),
        "--trace-output-name",
        TRACE_NAME,
    ]

    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    print(
        f"[{run.method_id} seed{run.seed}] selected checkpoint={run.checkpoint} "
        f"load_run={run.load_run} replay_run={run.replay_run_name}"
    )
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    metrics_ok = artifact_ok(metrics_path)
    trace_ok = artifact_ok(trace_path)
    if completed.returncode != 0 and metrics_ok and trace_ok:
        print(
            f"[{run.method_id} seed{run.seed}] replay exited {completed.returncode}, "
            "but required metrics and trace artifacts were written; treating as soft success."
        )
        return
    if completed.returncode != 0:
        raise RuntimeError(f"{run.method_id} seed{run.seed} failed with exit code {completed.returncode}")
    if not metrics_ok or not trace_ok:
        raise RuntimeError(
            f"{run.method_id} seed{run.seed} missing expected artifacts: "
            f"{relative_to_repo(metrics_path)}, {relative_to_repo(trace_path)}"
        )


def main() -> int:
    args = parse_args()
    selected_runs = [
        run
        for run in build_runs()
        if (args.only_method is None or run.method_id == args.only_method)
        and (args.only_seed is None or run.seed == args.only_seed)
    ]
    for run in selected_runs:
        run_one(run, args)

    print("Completed MuJoCo physical-trace replays:")
    for run in selected_runs:
        metrics_path, trace_path = expected_paths(run)
        print(f"- {run.method_id} seed{run.seed}: {relative_to_repo(metrics_path)}")
        print(f"  trace: {relative_to_repo(trace_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
