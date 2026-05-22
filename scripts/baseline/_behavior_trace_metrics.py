from __future__ import annotations

import math
from typing import Any

import numpy as np


def trace_capture_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "enabled": False,
        "max_episodes": 0,
        "max_steps_per_episode": 2000,
        "filename": "episode_traces.json",
    }
    merged = defaults.copy()
    merged.update(config.get("evaluation", {}).get("trace_capture", {}))
    merged["max_episodes"] = int(merged["max_episodes"])
    merged["max_steps_per_episode"] = int(merged["max_steps_per_episode"])
    return merged


def should_capture_traces(trace_cfg: dict[str, Any]) -> bool:
    return bool(trace_cfg.get("enabled")) and int(trace_cfg.get("max_episodes", 0)) > 0


def _as_2d_series(values: list[list[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D trace, got shape {array.shape}")
    return array


def _valid_dt(dt: float) -> float:
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Expected positive finite dt, got {dt}")
    return dt


def compute_log_dimensionless_jerk(position_trace: list[list[float]] | np.ndarray, dt: float) -> float | None:
    dt = _valid_dt(dt)
    trace = _as_2d_series(position_trace)
    if trace.shape[0] < 4:
        return None

    metrics: list[float] = []
    duration = dt * float(trace.shape[0] - 1)
    eps = 1e-12

    for dim in range(trace.shape[1]):
        series = trace[:, dim]
        amplitude = float(np.ptp(series))
        if amplitude <= eps:
            continue
        jerk = np.diff(series, n=3) / (dt**3)
        if jerk.size == 0:
            continue
        jerk_energy = float(np.sum(np.square(jerk)) * dt)
        scaled = (duration**5 * jerk_energy) / max(amplitude**2, eps)
        metrics.append(-math.log(max(scaled, eps)))

    if not metrics:
        return None
    return float(np.mean(metrics))


def compute_sparc(
    velocity_trace: list[list[float]] | np.ndarray,
    dt: float,
    *,
    cutoff_hz: float | None = None,
    amplitude_threshold: float = 0.05,
) -> float | None:
    dt = _valid_dt(dt)
    trace = _as_2d_series(velocity_trace)
    if trace.shape[0] < 4:
        return None

    sampling_rate = 1.0 / dt
    nyquist_hz = sampling_rate / 2.0
    effective_cutoff = nyquist_hz if cutoff_hz is None else min(float(cutoff_hz), nyquist_hz)
    if effective_cutoff <= 0.0:
        return None

    metrics: list[float] = []
    padlevel = 4
    eps = 1e-12

    for dim in range(trace.shape[1]):
        series = trace[:, dim]
        centered = series - np.mean(series)
        if np.max(np.abs(centered)) <= eps:
            continue

        nfft = int(2 ** math.ceil(math.log2(max(len(centered), 2)) + padlevel))
        freqs = np.fft.rfftfreq(nfft, d=dt)
        spectrum = np.abs(np.fft.rfft(centered, n=nfft))
        max_amp = float(np.max(spectrum))
        if max_amp <= eps:
            continue
        spectrum /= max_amp

        mask = (freqs <= effective_cutoff) & (spectrum >= float(amplitude_threshold))
        indices = np.flatnonzero(mask)
        if indices.size < 2:
            continue

        freq_slice = freqs[indices]
        amp_slice = spectrum[indices]
        freq_scale = max(float(freq_slice[-1]), eps)
        freq_norm = freq_slice / freq_scale
        arc = np.sum(np.sqrt(np.diff(freq_norm) ** 2 + np.diff(amp_slice) ** 2))
        metrics.append(float(-arc))

    if not metrics:
        return None
    return float(np.mean(metrics))


def compute_episode_smoothness_metrics(
    episode_trace: dict[str, Any],
    *,
    sparc_cutoff_hz: float | None = None,
    sparc_amplitude_threshold: float = 0.05,
) -> dict[str, Any]:
    dt = _valid_dt(float(episode_trace["dt"]))
    dof_pos = episode_trace.get("dof_pos")
    dof_vel = episode_trace.get("dof_vel")

    metrics = {
        "joint_position_ldlj_mean": (
            compute_log_dimensionless_jerk(dof_pos, dt) if dof_pos is not None else None
        ),
        "joint_velocity_sparc_mean": (
            compute_sparc(
                dof_vel,
                dt,
                cutoff_hz=sparc_cutoff_hz,
                amplitude_threshold=sparc_amplitude_threshold,
            )
            if dof_vel is not None
            else None
        ),
    }
    return metrics
