#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    BaselineError,
    artifact_dir,
    configure_runtime_env,
    default_manifest,
    ensure_humanoid_gym_checkout,
    ensure_upstream_on_syspath,
    load_config,
    read_json,
    relative_to_repo,
    resolve_humanoid_gym_root,
    resolve_run_dir,
    write_json,
)
from _overrides import apply_method_overrides  # noqa: E402


VALID_TERRAIN_MODES = ("isaac_mainline", "plane", "hfield_stress")


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def quaternion_to_euler_array(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z], dtype=np.float64)


def get_obs(data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat_wxyz = data.sensor("orientation").data.astype(np.double)
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    rot = R.from_quat(quat_xyzw)
    base_lin_vel = rot.apply(data.qvel[:3], inverse=True).astype(np.double)
    base_ang_vel = data.sensor("angular-velocity").data.astype(np.double)
    gravity_vec = rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return q, dq, quat_xyzw, base_lin_vel, base_ang_vel, gravity_vec


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def build_pd_gains(cfg: Any, actuator_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    kp = np.zeros((len(actuator_names),), dtype=np.double)
    kd = np.zeros((len(actuator_names),), dtype=np.double)
    for idx, name in enumerate(actuator_names):
        for dof_name, stiffness in cfg.control.stiffness.items():
            if dof_name in name:
                kp[idx] = float(stiffness)
                kd[idx] = float(cfg.control.damping[dof_name])
                break
    return kp, kd


def build_default_dof_pos(cfg: Any, actuator_names: list[str]) -> np.ndarray:
    return np.array([float(cfg.init_state.default_joint_angles[name]) for name in actuator_names], dtype=np.double)


def build_tau_limit(model) -> np.ndarray:
    if model.nu <= 0:
        raise RuntimeError("MuJoCo model exposes no actuators.")
    if model.actuator_ctrlrange.shape[0] != model.nu:
        raise RuntimeError("Unexpected actuator_ctrlrange shape in MuJoCo model.")
    return np.abs(model.actuator_ctrlrange[:, 1]).astype(np.double)


def actuator_joint_names(model, mujoco) -> list[str]:
    names: list[str] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    return names


def normalize_terrain_mode(terrain_mode: str) -> str:
    normalized = terrain_mode.strip().lower().replace("-", "_")
    aliases = {
        "default": "isaac_mainline",
        "isaac": "isaac_mainline",
        "terrain": "hfield_stress",
        "hfield": "hfield_stress",
    }
    return aliases.get(normalized, normalized)


def isaac_training_mesh_type(cfg: Any) -> str:
    terrain_cfg = getattr(cfg, "terrain", None)
    mesh_type = getattr(terrain_cfg, "mesh_type", None)
    if not isinstance(mesh_type, str) or not mesh_type.strip():
        raise BaselineError("Missing cfg.terrain.mesh_type; cannot align MuJoCo protocol to Isaac training condition.")
    return mesh_type.strip().lower()


def resolve_mujoco_protocol(cfg: Any, legged_gym_root_dir: str | Path, *, terrain_mode: str) -> dict[str, Any]:
    requested_mode = normalize_terrain_mode(terrain_mode)
    if requested_mode not in VALID_TERRAIN_MODES:
        raise BaselineError(
            f"Unsupported terrain_mode={terrain_mode!r}. Expected one of: {', '.join(VALID_TERRAIN_MODES)}."
        )

    isaac_mesh_type = isaac_training_mesh_type(cfg)
    root_dir = Path(legged_gym_root_dir)
    xml_dir = root_dir / "resources" / "robots" / "XBot" / "mjcf"

    if requested_mode == "isaac_mainline":
        if isaac_mesh_type != "plane":
            raise BaselineError(
                "terrain_mode='isaac_mainline' currently has no aligned MuJoCo terrain profile for "
                f"Isaac mesh_type={isaac_mesh_type!r}. Use an explicit override such as "
                "'plane' or 'hfield_stress' instead of silently mislabeling the protocol."
            )
        resolved_mode = "plane"
        purpose = "minimal_comparable_first_pass"
        comparable_to_isaac_mainline = True
        note = "Matches the current Isaac mainline training condition."
        model_path = xml_dir / "XBot-L.xml"
    elif requested_mode == "plane":
        resolved_mode = "plane"
        purpose = "minimal_comparable_first_pass" if isaac_mesh_type == "plane" else "manual_plane_override"
        comparable_to_isaac_mainline = isaac_mesh_type == "plane"
        note = (
            "Explicit MuJoCo plane replay."
            if comparable_to_isaac_mainline
            else "Explicit plane override; not guaranteed to match the Isaac training condition."
        )
        model_path = xml_dir / "XBot-L.xml"
    else:
        resolved_mode = "hfield_stress"
        purpose = "terrain_stress_probe"
        comparable_to_isaac_mainline = False
        note = "Aggressive hfield stress test. Treat as transfer-pressure probe, not Isaac-mainline-equivalent replay."
        model_path = xml_dir / "XBot-L-terrain.xml"

    if not model_path.exists():
        raise BaselineError(f"Resolved MuJoCo model does not exist: {model_path}")

    return {
        "terrain_mode_requested": requested_mode,
        "terrain_mode": resolved_mode,
        "purpose": purpose,
        "isaac_training_mesh_type": isaac_mesh_type,
        "is_isaac_mainline_comparable": comparable_to_isaac_mainline,
        "note": note,
        "mujoco_model_path": model_path,
    }


def build_policy_input(
    *,
    count_lowlevel: int,
    dt: float,
    cfg: Any,
    cmd_vx: float,
    cmd_vy: float,
    cmd_dyaw: float,
    q: np.ndarray,
    dq: np.ndarray,
    prev_action: np.ndarray,
    omega: np.ndarray,
    quat_xyzw: np.ndarray,
) -> np.ndarray:
    obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
    eu_ang = quaternion_to_euler_array(quat_xyzw)
    eu_ang[eu_ang > math.pi] -= 2.0 * math.pi

    obs[0, 0] = math.sin(2.0 * math.pi * count_lowlevel * dt / 0.64)
    obs[0, 1] = math.cos(2.0 * math.pi * count_lowlevel * dt / 0.64)
    obs[0, 2] = cmd_vx * cfg.normalization.obs_scales.lin_vel
    obs[0, 3] = cmd_vy * cfg.normalization.obs_scales.lin_vel
    obs[0, 4] = cmd_dyaw * cfg.normalization.obs_scales.ang_vel
    obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
    obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
    obs[0, 29:41] = prev_action
    obs[0, 41:44] = omega
    obs[0, 44:47] = eu_ang
    return np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)


def reset_episode_state(
    model,
    data,
    mujoco,
    *,
    default_dof_pos: np.ndarray,
    rng: np.random.Generator,
    joint_reset_noise: float,
    base_xy_noise: float,
) -> None:
    mujoco.mj_resetData(model, data)
    if base_xy_noise > 0.0:
        data.qpos[0] += float(rng.uniform(-base_xy_noise, base_xy_noise))
        data.qpos[1] += float(rng.uniform(-base_xy_noise, base_xy_noise))
    if joint_reset_noise > 0.0:
        joint_noise = rng.uniform(-joint_reset_noise, joint_reset_noise, size=default_dof_pos.shape[0])
        data.qpos[7 : 7 + default_dof_pos.shape[0]] = default_dof_pos + joint_noise
    else:
        data.qpos[7 : 7 + default_dof_pos.shape[0]] = default_dof_pos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def run_episode(
    policy,
    cfg: Any,
    mujoco,
    torch,
    rng: np.random.Generator,
    *,
    mujoco_model_path: Path,
    command_vx: float,
    command_vy: float,
    command_dyaw: float,
    joint_reset_noise: float,
    base_xy_noise: float,
):
    model = mujoco.MjModel.from_xml_path(str(mujoco_model_path))
    model.opt.timestep = cfg.sim.dt
    data = mujoco.MjData(model)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    actuator_names = actuator_joint_names(model, mujoco)
    kp, kd = build_pd_gains(cfg, actuator_names)
    target_q_default = build_default_dof_pos(cfg, actuator_names)
    tau_limit = build_tau_limit(model)
    reset_episode_state(
        model,
        data,
        mujoco,
        default_dof_pos=target_q_default,
        rng=rng,
        joint_reset_noise=joint_reset_noise,
        base_xy_noise=base_xy_noise,
    )

    target_q = np.zeros((cfg.env.num_actions,), dtype=np.double)
    action = np.zeros((cfg.env.num_actions,), dtype=np.double)
    target_dq = np.zeros((cfg.env.num_actions,), dtype=np.double)
    hist_obs: deque[np.ndarray] = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.float32))

    returns = 0.0
    tracking_error_sum = 0.0
    joint_accel_sum = 0.0
    action_jitter_sum = 0.0
    step_count = 0
    count_lowlevel = 0
    prev_action = np.zeros_like(action)
    fall = False

    action_scale = cfg.control.action_scale
    control_dt = cfg.control.decimation * cfg.sim.dt
    total_control_steps = int(cfg.eval.sim_duration / control_dt)

    for _ in range(total_control_steps):
        q_all, dq_all, quat_xyzw, base_lin_vel, base_ang_vel, _ = get_obs(data)
        q = q_all[-cfg.env.num_actions :]
        dq = dq_all[-cfg.env.num_actions :]
        dq_before = dq.copy()
        obs = build_policy_input(
            count_lowlevel=count_lowlevel,
            dt=cfg.sim.dt,
            cfg=cfg,
            cmd_vx=command_vx,
            cmd_vy=command_vy,
            cmd_dyaw=command_dyaw,
            q=q,
            dq=dq,
            prev_action=action,
            omega=base_ang_vel,
            quat_xyzw=quat_xyzw,
        )
        hist_obs.append(obs)
        hist_obs.popleft()

        policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
        for i in range(cfg.env.frame_stack):
            start = i * cfg.env.num_single_obs
            end = (i + 1) * cfg.env.num_single_obs
            policy_input[0, start:end] = hist_obs[i][0, :]

        with torch.inference_mode():
            action = policy(torch.tensor(policy_input))[0].detach().cpu().numpy()
        action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
        target_q = action * action_scale + target_q_default

        for _ in range(cfg.control.decimation):
            q_all, dq_all, _, _, _, _ = get_obs(data)
            q = q_all[-cfg.env.num_actions :]
            dq = dq_all[-cfg.env.num_actions :]
            tau = pd_control(target_q, q, kp, target_dq, dq, kd)
            tau = np.clip(tau, -tau_limit, tau_limit)
            data.ctrl = tau
            mujoco.mj_step(model, data)
            count_lowlevel += 1

        _, dq_after_all, _, base_lin_vel_after, base_ang_vel_after, _ = get_obs(data)
        dq_after = dq_after_all[-cfg.env.num_actions :]
        linear_error = abs(command_vx - float(base_lin_vel_after[0]))
        yaw_error = abs(command_dyaw - float(base_ang_vel_after[2]))
        tracking_error_sum += linear_error + yaw_error
        joint_accel_sum += float(np.linalg.norm((dq_after - dq_before) / control_dt))
        action_jitter_sum += float(np.linalg.norm(action - prev_action))
        returns += float(base_lin_vel_after[0]) - (linear_error + yaw_error)
        step_count += 1

        prev_action = action.copy()

        base_height = float(data.xpos[base_body_id][2])
        if base_height < cfg.eval.fall_height_threshold:
            fall = True
            break

    denom = max(step_count, 1)
    return {
        "episode_return": returns,
        "velocity_tracking_error": tracking_error_sum / denom,
        "joint_acceleration_l2": joint_accel_sum / denom,
        "action_jitter_l2": action_jitter_sum / denom,
        "fell": fall,
        "steps": step_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one exported policy through Humanoid-Gym MuJoCo sim2sim.")
    parser.add_argument("--config", default=None, help="Path to the method config JSON.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    parser.add_argument("--run-name", default=None, help="Override the configured run_name.")
    parser.add_argument("--load-run", default=None, help="Explicit upstream run directory name.")
    parser.add_argument("--checkpoint", type=int, default=None, help="Specific checkpoint number to export and evaluate.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of MuJoCo episodes.")
    parser.add_argument("--command-vx", type=float, default=0.4, help="Forward velocity command used in MuJoCo rollout.")
    parser.add_argument("--command-vy", type=float, default=0.0, help="Lateral velocity command used in MuJoCo rollout.")
    parser.add_argument("--command-dyaw", type=float, default=0.0, help="Yaw-rate command used in MuJoCo rollout.")
    parser.add_argument(
        "--terrain-mode",
        default="isaac_mainline",
        help=(
            "MuJoCo replay protocol. "
            "Use 'isaac_mainline' for the current comparable replay, 'plane' for an explicit plane override, "
            "or 'hfield_stress' for the separate terrain stress probe."
        ),
    )
    parser.add_argument(
        "--terrain",
        action="store_true",
        help="Deprecated alias for --terrain-mode=hfield_stress. Kept only for backward compatibility.",
    )
    parser.add_argument("--sim-duration", type=float, default=20.0, help="Episode duration in seconds.")
    parser.add_argument("--fall-height-threshold", type=float, default=0.55, help="Base height threshold below which the rollout counts as a fall.")
    parser.add_argument(
        "--joint-reset-noise",
        type=float,
        default=0.0,
        help="Uniform joint-position reset noise magnitude, matching Isaac-style default_dof_pos +/- noise.",
    )
    parser.add_argument(
        "--base-xy-noise",
        type=float,
        default=0.0,
        help="Uniform base x/y reset noise magnitude in meters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used for MuJoCo episode resets when reset noise is enabled.",
    )
    parser.add_argument(
        "--output-name",
        default="metrics_mujoco.json",
        help="Artifact filename for the MuJoCo metrics JSON.",
    )
    parser.add_argument(
        "--manifest-slot",
        default=None,
        help="Optional manifest key under mujoco_metrics_runs for preserving multiple MuJoCo eval variants.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    ensure_humanoid_gym_checkout(humanoid_gym_root)
    configure_runtime_env()
    ensure_upstream_on_syspath(humanoid_gym_root)

    importlib = __import__("importlib")
    importlib.import_module("humanoid.envs")
    import torch
    import mujoco
    from humanoid import LEGGED_GYM_ROOT_DIR
    from humanoid.utils import task_registry

    run_name = args.run_name or config["run_name"]

    # Reuse the existing export pipeline so MuJoCo always consumes a checked JIT policy.
    export_policy_module = importlib.import_module("export_policy")
    export_argv = [
        "export_policy.py",
        f"--config={args.config}" if args.config else "",
        f"--run-name={run_name}",
    ]
    if args.load_run:
        export_argv.append(f"--load-run={args.load_run}")
    if args.checkpoint is not None:
        export_argv.append(f"--checkpoint={args.checkpoint}")
    export_argv = [arg for arg in export_argv if arg]

    original_argv = sys.argv[:]
    try:
        sys.argv = export_argv
        export_policy_module.main()
    finally:
        sys.argv = original_argv

    output_dir = artifact_dir(config, run_name)
    manifest_path = output_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else default_manifest(config, humanoid_gym_root)
    exported_policy_path = output_dir / "exported" / "policies" / "policy_1.pt"
    if not exported_policy_path.exists():
        raise RuntimeError(f"Expected exported policy not found: {exported_policy_path}")

    policy = torch.jit.load(str(exported_policy_path))
    policy.eval()

    env_cfg, train_cfg = task_registry.get_cfgs(name=config["task"])
    cfg, train_cfg = apply_method_overrides(env_cfg, train_cfg, config)
    cfg.LEGGED_GYM_ROOT_DIR = str(LEGGED_GYM_ROOT_DIR)
    cfg.eval = type("EvalCfg", (), {})()
    cfg.eval.sim_duration = float(args.sim_duration)
    cfg.eval.fall_height_threshold = float(args.fall_height_threshold)
    requested_terrain_mode = normalize_terrain_mode(args.terrain_mode)
    if args.terrain:
        if requested_terrain_mode != "isaac_mainline":
            raise BaselineError("Do not combine --terrain with an explicit non-default --terrain-mode.")
        requested_terrain_mode = "hfield_stress"
    protocol = resolve_mujoco_protocol(cfg, LEGGED_GYM_ROOT_DIR, terrain_mode=requested_terrain_mode)
    rng = np.random.default_rng(int(args.seed))

    episode_returns: list[float] = []
    tracking_errors: list[float] = []
    joint_accels: list[float] = []
    action_jitters: list[float] = []
    fell_flags: list[bool] = []
    episode_lengths: list[int] = []

    for _ in range(args.episodes):
        result = run_episode(
            policy,
            cfg,
            mujoco,
            torch,
            rng,
            mujoco_model_path=protocol["mujoco_model_path"],
            command_vx=float(args.command_vx),
            command_vy=float(args.command_vy),
            command_dyaw=float(args.command_dyaw),
            joint_reset_noise=float(args.joint_reset_noise),
            base_xy_noise=float(args.base_xy_noise),
        )
        episode_returns.append(result["episode_return"])
        tracking_errors.append(result["velocity_tracking_error"])
        joint_accels.append(result["joint_acceleration_l2"])
        action_jitters.append(result["action_jitter_l2"])
        fell_flags.append(bool(result["fell"]))
        episode_lengths.append(int(result["steps"]))

    return_mean, return_std = summarize(episode_returns)
    tracking_mean, tracking_std = summarize(tracking_errors)
    joint_accel_mean, joint_accel_std = summarize(joint_accels)
    action_jitter_mean, action_jitter_std = summarize(action_jitters)

    method_cfg = config.get("method", {})
    metrics = {
        "metric_schema_version": int(config["evaluation"].get("metric_schema_version", 2)),
        "method_id": method_cfg.get("id"),
        "method_label": method_cfg.get("label"),
        "evaluation_backend": "mujoco_sim2sim",
        "episodes_evaluated": int(args.episodes),
        "velocity_tracking_error_mean": tracking_mean,
        "velocity_tracking_error_std": tracking_std,
        "fall_rate": sum(1 for flag in fell_flags if flag) / max(len(fell_flags), 1),
        "joint_acceleration_l2_mean": joint_accel_mean,
        "joint_acceleration_l2_std": joint_accel_std,
        "action_jitter_l2_mean": action_jitter_mean,
        "action_jitter_l2_std": action_jitter_std,
        "episode_return_mean": return_mean,
        "episode_return_std": return_std,
        "mujoco_eval": {
            "checkpoint": int(args.checkpoint) if args.checkpoint is not None else None,
            "load_run": args.load_run,
            "terrain": protocol["terrain_mode"] == "hfield_stress",
            "terrain_mode_requested": protocol["terrain_mode_requested"],
            "terrain_mode": protocol["terrain_mode"],
            "protocol_purpose": protocol["purpose"],
            "isaac_training_mesh_type": protocol["isaac_training_mesh_type"],
            "is_isaac_mainline_comparable": protocol["is_isaac_mainline_comparable"],
            "protocol_note": protocol["note"],
            "mujoco_model_path": relative_to_repo(protocol["mujoco_model_path"]),
            "command_vx": float(args.command_vx),
            "command_vy": float(args.command_vy),
            "command_dyaw": float(args.command_dyaw),
            "sim_duration": float(args.sim_duration),
            "fall_height_threshold": float(args.fall_height_threshold),
            "joint_reset_noise": float(args.joint_reset_noise),
            "base_xy_noise": float(args.base_xy_noise),
            "seed": int(args.seed),
            "episode_steps_mean": statistics.fmean(episode_lengths) if episode_lengths else None,
        },
    }

    metrics_path = output_dir / args.output_name
    write_json(metrics_path, metrics)

    run_dir = resolve_run_dir(humanoid_gym_root, config, run_name=run_name, load_run=args.load_run)
    manifest["run_name"] = run_name
    manifest["run_dir"] = relative_to_repo(run_dir)
    manifest["exported_policy_path"] = relative_to_repo(exported_policy_path)
    manifest["mujoco_metrics_path"] = relative_to_repo(metrics_path)
    manifest["mujoco_metrics"] = metrics
    manifest["mujoco_protocol"] = {
        "terrain": protocol["terrain_mode"] == "hfield_stress",
        "terrain_mode_requested": protocol["terrain_mode_requested"],
        "terrain_mode": protocol["terrain_mode"],
        "protocol_purpose": protocol["purpose"],
        "isaac_training_mesh_type": protocol["isaac_training_mesh_type"],
        "is_isaac_mainline_comparable": protocol["is_isaac_mainline_comparable"],
        "protocol_note": protocol["note"],
        "mujoco_model_path": relative_to_repo(protocol["mujoco_model_path"]),
    }
    manifest_slot = args.manifest_slot or Path(args.output_name).stem
    mujoco_metrics_runs = manifest.get("mujoco_metrics_runs")
    if not isinstance(mujoco_metrics_runs, dict):
        mujoco_metrics_runs = {}
    mujoco_metrics_runs[manifest_slot] = {
        "metrics_path": relative_to_repo(metrics_path),
        "metrics": metrics,
    }
    manifest["mujoco_metrics_runs"] = mujoco_metrics_runs
    write_json(manifest_path, manifest)

    print(f"Wrote {relative_to_repo(metrics_path)}")
    print(f"Updated {relative_to_repo(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
