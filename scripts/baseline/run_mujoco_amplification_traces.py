#!/usr/bin/env python3
"""Run matched MuJoCo trace replays for the policy-to-physics amplification check.

This is the execution slice for issue #49. It replays selected checkpoints with
the same MuJoCo isaac_mainline protocol used by the cross-engine degradation
table and enables the optional per-timestep trace capture added in #48.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import artifact_dir, load_config, relative_to_repo, repo_root  # noqa: E402


DEFAULT_PYTHON = Path("/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python")
METRICS_NAME = "metrics_mujoco_amplification_trace_20ep_20s_noise01.json"
TRACE_NAME = "mujoco_amplification_trace_3ep_20s_noise01.json"
MANIFEST_SLOT = "mujoco_amplification_trace_20ep_20s_noise01"


@dataclass(frozen=True)
class TraceRun:
    method_id: str
    seed: int
    config: str
    run_name: str
    load_run: str
    checkpoint: int


RUNS = [
    TraceRun(
        "scppo38",
        11,
        "configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json",
        "scppo38_mujoco_amp_trace_seed11",
        "May14_13-38-03_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11",
        300,
    ),
    TraceRun(
        "scppo38",
        17,
        "configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json",
        "scppo38_mujoco_amp_trace_seed17",
        "May14_13-38-38_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17",
        300,
    ),
    TraceRun(
        "scppo38",
        23,
        "configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json",
        "scppo38_mujoco_amp_trace_seed23",
        "May14_13-40-42_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23",
        400,
    ),
    TraceRun(
        "layernorm_ep3",
        11,
        "configs/methods/layernorm_actor_output_gain_0750_more_epochs_reliability_probe.json",
        "layernorm_ep3_mujoco_amp_trace_seed11",
        "May25_11-28-17_layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed11",
        400,
    ),
    TraceRun(
        "layernorm_ep3",
        17,
        "configs/methods/layernorm_actor_output_gain_0750_more_epochs_reliability_probe.json",
        "layernorm_ep3_mujoco_amp_trace_seed17",
        "May25_12-04-08_layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed17",
        400,
    ),
    TraceRun(
        "layernorm_ep3",
        23,
        "configs/methods/layernorm_actor_output_gain_0750_more_epochs_reliability_probe.json",
        "layernorm_ep3_mujoco_amp_trace_seed23",
        "May25_12-38-11_layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed23",
        400,
    ),
    TraceRun(
        "action_scaling",
        11,
        "configs/methods/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp.json",
        "action_scaling_mujoco_amp_trace_seed11",
        "May23_01-27-22_action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed11",
        400,
    ),
    TraceRun(
        "action_scaling",
        17,
        "configs/methods/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp.json",
        "action_scaling_mujoco_amp_trace_seed17",
        "May23_02-15-58_action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed17",
        400,
    ),
    TraceRun(
        "action_scaling",
        23,
        "configs/methods/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp.json",
        "action_scaling_mujoco_amp_trace_seed23",
        "May23_03-56-01_action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed23",
        400,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched MuJoCo amplification trace replays.")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON), help="Python executable for replay runs.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sim-duration", type=float, default=20.0)
    parser.add_argument("--joint-reset-noise", type=float, default=0.1)
    parser.add_argument("--base-xy-noise", type=float, default=0.0)
    parser.add_argument("--command-vx", type=float, default=0.4)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-dyaw", type=float, default=0.0)
    parser.add_argument("--trace-max-episodes", type=int, default=3)
    parser.add_argument("--trace-max-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345, help="MuJoCo reset RNG seed; matches existing replay scripts.")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--only-method", choices=sorted({run.method_id for run in RUNS}), default=None)
    parser.add_argument("--only-seed", type=int, choices=[11, 17, 23], default=None)
    return parser.parse_args()


def expected_paths(run: TraceRun) -> tuple[Path, Path]:
    config = load_config(run.config)
    output_dir = artifact_dir(config, run.run_name)
    return output_dir / METRICS_NAME, output_dir / TRACE_NAME


def artifact_ok(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        json.load(handle)
    return True


def remove_stale_outputs(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def run_one(run: TraceRun, args: argparse.Namespace) -> None:
    metrics_path, trace_path = expected_paths(run)
    remove_stale_outputs(metrics_path, trace_path)
    command = [
        args.python_bin,
        str(repo_root() / "scripts" / "baseline" / "evaluate_mujoco_sim2sim.py"),
        "--config",
        str(repo_root() / run.config),
        "--run-name",
        run.run_name,
        "--load-run",
        run.load_run,
        "--checkpoint",
        str(run.checkpoint),
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
        str(args.trace_max_episodes),
        "--trace-max-steps",
        str(args.trace_max_steps),
        "--trace-output-name",
        TRACE_NAME,
    ]
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    print(f"[{run.method_id} seed{run.seed}] checkpoint={run.checkpoint} run_name={run.run_name}")
    completed = subprocess.run(command, cwd=repo_root(), env=env, check=False)
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
        for run in RUNS
        if (args.only_method is None or run.method_id == args.only_method)
        and (args.only_seed is None or run.seed == args.only_seed)
    ]
    for run in selected_runs:
        run_one(run, args)

    print("Completed matched MuJoCo amplification trace replays:")
    for run in selected_runs:
        metrics_path, trace_path = expected_paths(run)
        print(f"- {run.method_id} seed{run.seed}: {relative_to_repo(metrics_path)}")
        print(f"  trace: {relative_to_repo(trace_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
