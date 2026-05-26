#!/usr/bin/env python3
"""Run matched MuJoCo actuator-proxy stress replays for issue #54."""

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
METRICS_NAME = "metrics_mujoco_actuator_lowpass_tau005_20ep_20s_noise01.json"
NOMINAL_METRICS_NAME = "metrics_mujoco_isaac_mainline_20ep_20s_noise01.json"
MANIFEST_SLOT = "mujoco_actuator_lowpass_tau005_20ep_20s_noise01"
ACTUATOR_PROXY_MODE = "action_lowpass"
ACTUATOR_LOWPASS_TAU = 0.05


@dataclass(frozen=True)
class ActuatorProxyRun:
    method_id: str
    seed: int
    config: str
    run_name: str
    nominal_run_name: str
    load_run: str
    checkpoint: int


RUNS = [
    ActuatorProxyRun(
        "scppo38",
        11,
        "configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json",
        "scppo38_mujoco_actuator_lowpass_tau005_seed11",
        "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11",
        "May14_13-38-03_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed11",
        300,
    ),
    ActuatorProxyRun(
        "scppo38",
        17,
        "configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json",
        "scppo38_mujoco_actuator_lowpass_tau005_seed17",
        "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17",
        "May14_13-38-38_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed17",
        300,
    ),
    ActuatorProxyRun(
        "scppo38",
        23,
        "configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp.json",
        "scppo38_mujoco_actuator_lowpass_tau005_seed23",
        "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23",
        "May14_13-40-42_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed23",
        400,
    ),
    ActuatorProxyRun(
        "heuristic",
        11,
        "configs/methods/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget.json",
        "heuristic_mujoco_actuator_lowpass_tau005_seed11",
        "heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11",
        "May21_03-55-49_heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed11",
        350,
    ),
    ActuatorProxyRun(
        "heuristic",
        17,
        "configs/methods/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget.json",
        "heuristic_mujoco_actuator_lowpass_tau005_seed17",
        "heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17",
        "May21_04-23-32_heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed17",
        300,
    ),
    ActuatorProxyRun(
        "heuristic",
        23,
        "configs/methods/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget.json",
        "heuristic_mujoco_actuator_lowpass_tau005_seed23",
        "heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23",
        "May21_04-51-50_heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed23",
        350,
    ),
    ActuatorProxyRun(
        "layernorm_ep3",
        11,
        "configs/methods/layernorm_actor_output_gain_0750_more_epochs_reliability_probe.json",
        "layernorm_ep3_mujoco_actuator_lowpass_tau005_seed11",
        "layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed11",
        "May25_11-28-17_layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed11",
        400,
    ),
    ActuatorProxyRun(
        "layernorm_ep3",
        17,
        "configs/methods/layernorm_actor_output_gain_0750_more_epochs_reliability_probe.json",
        "layernorm_ep3_mujoco_actuator_lowpass_tau005_seed17",
        "layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed17",
        "May25_12-04-08_layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed17",
        400,
    ),
    ActuatorProxyRun(
        "layernorm_ep3",
        23,
        "configs/methods/layernorm_actor_output_gain_0750_more_epochs_reliability_probe.json",
        "layernorm_ep3_mujoco_actuator_lowpass_tau005_seed23",
        "layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed23",
        "May25_12-38-11_layernorm_actor_output_gain_0750_more_epochs_reliability_probe_rough_terrain_seed23",
        400,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched MuJoCo actuator-proxy stress replays.")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON), help="Python executable for replay runs.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sim-duration", type=float, default=20.0)
    parser.add_argument("--joint-reset-noise", type=float, default=0.1)
    parser.add_argument("--base-xy-noise", type=float, default=0.0)
    parser.add_argument("--command-vx", type=float, default=0.4)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-dyaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345, help="MuJoCo reset RNG seed; matches existing replay scripts.")
    parser.add_argument("--actuator-lowpass-time-constant", type=float, default=ACTUATOR_LOWPASS_TAU)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--only-method", choices=sorted({run.method_id for run in RUNS}), default=None)
    parser.add_argument("--only-seed", type=int, choices=[11, 17, 23], default=None)
    return parser.parse_args()


def metrics_path(run: ActuatorProxyRun) -> Path:
    config = load_config(run.config)
    return artifact_dir(config, run.run_name) / METRICS_NAME


def nominal_metrics_path(run: ActuatorProxyRun) -> Path:
    config = load_config(run.config)
    return artifact_dir(config, run.nominal_run_name) / NOMINAL_METRICS_NAME


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


def run_one(run: ActuatorProxyRun, args: argparse.Namespace) -> None:
    output_path = metrics_path(run)
    remove_stale_outputs(output_path)
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
        "--actuator-proxy-mode",
        ACTUATOR_PROXY_MODE,
        "--actuator-lowpass-time-constant",
        str(args.actuator_lowpass_time_constant),
    ]
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    print(f"[{run.method_id} seed{run.seed}] checkpoint={run.checkpoint} run_name={run.run_name}")
    completed = subprocess.run(command, cwd=repo_root(), env=env, check=False)
    metrics_ok = artifact_ok(output_path)
    if completed.returncode != 0 and metrics_ok:
        print(
            f"[{run.method_id} seed{run.seed}] replay exited {completed.returncode}, "
            "but required metrics artifact was written; treating as soft success."
        )
        return
    if completed.returncode != 0:
        raise RuntimeError(f"{run.method_id} seed{run.seed} failed with exit code {completed.returncode}")
    if not metrics_ok:
        raise RuntimeError(f"{run.method_id} seed{run.seed} missing expected artifact: {relative_to_repo(output_path)}")


def main() -> int:
    args = parse_args()
    selected_runs = [
        run
        for run in RUNS
        if (args.only_method is None or run.method_id == args.only_method)
        and (args.only_seed is None or run.seed == args.only_seed)
    ]
    for run in selected_runs:
        if not nominal_metrics_path(run).exists():
            raise RuntimeError(f"Missing nominal MuJoCo metrics for comparison: {relative_to_repo(nominal_metrics_path(run))}")
        run_one(run, args)

    print("Completed matched MuJoCo actuator-proxy stress replays:")
    for run in selected_runs:
        print(f"- {run.method_id} seed{run.seed}: {relative_to_repo(metrics_path(run))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
