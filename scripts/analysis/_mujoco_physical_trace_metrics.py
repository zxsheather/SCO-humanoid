from __future__ import annotations

import math
from typing import Any

import numpy as np


def _as_vector(values: Any) -> np.ndarray:
    if values is None:
        return np.zeros((0,), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"Expected flat vector, got shape {array.shape}")
    return array


def control_vector(step: dict[str, Any]) -> np.ndarray:
    values = step.get("applied_control")
    if values is None or (isinstance(values, list) and not values):
        values = step.get("control_tau")
    return _as_vector(values)


def joint_velocity_vector(step: dict[str, Any]) -> np.ndarray:
    return _as_vector(step.get("joint_velocity"))


def base_forward_velocity(step: dict[str, Any]) -> float | None:
    values = step.get("base_linear_velocity")
    vector = _as_vector(values)
    if vector.size == 0:
        return None
    return float(vector[0])


def paired_control_and_velocity(step: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    tau = control_vector(step)
    dq = joint_velocity_vector(step)
    if tau.size == 0 or dq.size == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    n = min(tau.size, dq.size)
    return tau[:n], dq[:n]


def mechanical_power(step: dict[str, Any]) -> np.ndarray:
    tau, dq = paired_control_and_velocity(step)
    if tau.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return tau * dq


def compute_episode_physical_metrics(episode: dict[str, Any], *, control_dt: float) -> dict[str, float | int | bool | None]:
    control_dt = float(control_dt)
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError(f"Expected positive finite control_dt, got {control_dt}")

    trace = episode.get("trace", [])
    if not trace:
        return {
            "step_count": 0,
            "fell": bool(episode.get("fell")),
            "control_torque_rms": None,
            "control_torque_l2_mean": None,
            "mechanical_power_abs_mean": None,
            "mechanical_power_positive_mean": None,
            "mechanical_energy_abs": None,
            "forward_speed_abs_mean": None,
            "forward_distance_abs": None,
            "mechanical_cot_proxy": None,
        }

    torque_square_values: list[float] = []
    torque_l2_values: list[float] = []
    abs_power_values: list[float] = []
    pos_power_values: list[float] = []
    forward_speed_values: list[float] = []

    for step in trace:
        tau = control_vector(step)
        if tau.size:
            torque_square_values.extend(float(value * value) for value in tau.tolist())
            torque_l2_values.append(float(np.linalg.norm(tau)))

        power = mechanical_power(step)
        if power.size:
            abs_power_values.append(float(np.sum(np.abs(power))))
            pos_power_values.append(float(np.sum(np.maximum(power, 0.0))))

        forward_speed = base_forward_velocity(step)
        if forward_speed is not None:
            forward_speed_values.append(abs(float(forward_speed)))

    control_torque_rms = math.sqrt(float(np.mean(torque_square_values))) if torque_square_values else None
    control_torque_l2_mean = float(np.mean(torque_l2_values)) if torque_l2_values else None
    mechanical_power_abs_mean = float(np.mean(abs_power_values)) if abs_power_values else None
    mechanical_power_positive_mean = float(np.mean(pos_power_values)) if pos_power_values else None
    forward_speed_abs_mean = float(np.mean(forward_speed_values)) if forward_speed_values else None

    mechanical_energy_abs = None
    forward_distance_abs = None
    mechanical_cot_proxy = None
    if mechanical_power_abs_mean is not None:
        mechanical_energy_abs = float(np.sum(abs_power_values) * control_dt)
    if forward_speed_abs_mean is not None:
        forward_distance_abs = float(np.sum(forward_speed_values) * control_dt)
    if mechanical_energy_abs is not None and forward_distance_abs is not None and forward_distance_abs > 1e-12:
        mechanical_cot_proxy = mechanical_energy_abs / forward_distance_abs

    return {
        "step_count": len(trace),
        "fell": bool(episode.get("fell")),
        "control_torque_rms": control_torque_rms,
        "control_torque_l2_mean": control_torque_l2_mean,
        "mechanical_power_abs_mean": mechanical_power_abs_mean,
        "mechanical_power_positive_mean": mechanical_power_positive_mean,
        "mechanical_energy_abs": mechanical_energy_abs,
        "forward_speed_abs_mean": forward_speed_abs_mean,
        "forward_distance_abs": forward_distance_abs,
        "mechanical_cot_proxy": mechanical_cot_proxy,
    }
