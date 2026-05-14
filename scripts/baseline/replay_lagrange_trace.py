#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import load_config, read_json


def nested_get(payload: dict[str, Any], dotted_key: str, default: Any) -> Any:
    if dotted_key in payload:
        return payload[dotted_key]
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def pid_integral_update(
    integral: float,
    error: float,
    *,
    lagrange_multiplier: float,
    lambda_min: float,
    epsilon: float,
    integral_min: float,
    integral_max: float,
    integral_mode: str,
    integral_decay: float,
) -> float:
    lower_bound_active = lagrange_multiplier <= (lambda_min + epsilon)

    if integral_mode == "standard":
        updated = integral + error
    elif integral_mode == "lower_bound_clamp":
        if lower_bound_active:
            integral = max(integral, 0.0)
        updated = integral + error
    elif integral_mode == "lower_bound_decay":
        if lower_bound_active and error < 0.0:
            updated = integral * integral_decay
        else:
            updated = integral + error
    else:
        raise ValueError(f"Unsupported integral_mode: {integral_mode}")

    return min(max(updated, integral_min), integral_max)


def replay_trace(
    trace: list[dict[str, Any]],
    *,
    update_mode: str,
    dual_lr: float,
    pid_kp: float,
    pid_ki: float,
    pid_kd: float,
    integral_mode: str,
    integral_decay: float,
    lambda_init: float,
    lambda_min: float,
    lambda_max: float,
    integral_min: float,
    integral_max: float,
    epsilon: float,
) -> list[dict[str, Any]]:
    lagrange_multiplier = lambda_init
    integral_error = 0.0
    previous_error = 0.0
    rows: list[dict[str, Any]] = []

    for entry in trace:
        threshold = float(entry["constraint_threshold"])
        error = float(entry["policy_local_sensitivity_cost_update"]) - threshold
        if update_mode == "pid":
            integral_error = pid_integral_update(
                integral_error,
                error,
                lagrange_multiplier=lagrange_multiplier,
                lambda_min=lambda_min,
                epsilon=epsilon,
                integral_min=integral_min,
                integral_max=integral_max,
                integral_mode=integral_mode,
                integral_decay=integral_decay,
            )
            derivative = error - previous_error
            delta = pid_kp * error + pid_ki * integral_error + pid_kd * derivative
            previous_error = error
        elif update_mode == "dual":
            delta = dual_lr * error
        else:
            raise ValueError(f"Unsupported update_mode: {update_mode}")

        lagrange_multiplier = min(max(lagrange_multiplier + delta, lambda_min), lambda_max)
        rows.append(
            {
                "iteration": int(entry["iteration"]),
                "constraint_error": error,
                "constraint_integral_error": integral_error,
                "lagrange_delta": delta,
                "lagrange_multiplier": lagrange_multiplier,
            }
        )

    return rows


def summarize_rows(rows: list[dict[str, Any]], tail: int) -> dict[str, Any]:
    positive_delta_iterations = [row["iteration"] for row in rows if row["lagrange_delta"] > 0.0]
    positive_multiplier_iterations = [row["iteration"] for row in rows if row["lagrange_multiplier"] > 0.0]
    threshold_cross_iterations = [row["iteration"] for row in rows if row["constraint_error"] > 0.0]
    return {
        "iterations": len(rows),
        "positive_delta_count": len(positive_delta_iterations),
        "positive_delta_iterations": positive_delta_iterations,
        "positive_multiplier_count": len(positive_multiplier_iterations),
        "positive_multiplier_iterations": positive_multiplier_iterations,
        "threshold_cross_count": len(threshold_cross_iterations),
        "threshold_cross_iterations": threshold_cross_iterations,
        "tail": rows[-tail:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay one Lagrange-multiplier trace under PID or dual update rules.")
    parser.add_argument("--trace", required=True, help="Path to lagrange_multiplier_trace.json.")
    parser.add_argument("--config", default=None, help="Optional method config JSON used to seed replay parameters.")
    parser.add_argument("--update-mode", default=None, help="Override update mode: pid or dual.")
    parser.add_argument("--dual-lr", type=float, default=None, help="Override dual learning rate.")
    parser.add_argument("--pid-kp", type=float, default=None, help="Override PID proportional gain.")
    parser.add_argument("--pid-ki", type=float, default=None, help="Override PID integral gain.")
    parser.add_argument("--pid-kd", type=float, default=None, help="Override PID derivative gain.")
    parser.add_argument("--pid-integral-mode", default=None, help="Override PID integral mode.")
    parser.add_argument("--pid-integral-decay", type=float, default=None, help="Override PID integral decay.")
    parser.add_argument("--lambda-init", type=float, default=None, help="Override initial multiplier.")
    parser.add_argument("--lambda-min", type=float, default=None, help="Override multiplier min.")
    parser.add_argument("--lambda-max", type=float, default=None, help="Override multiplier max.")
    parser.add_argument("--integral-min", type=float, default=None, help="Override integral min.")
    parser.add_argument("--integral-max", type=float, default=None, help="Override integral max.")
    parser.add_argument("--epsilon", type=float, default=None, help="Override epsilon used for lower-bound checks.")
    parser.add_argument("--tail", type=int, default=12, help="How many final rows to print.")
    args = parser.parse_args()

    config = load_config(args.config)
    overrides = config.get("overrides", {}).get("train", {})
    trace_payload = read_json(Path(args.trace))
    trace = trace_payload["trace"]

    update_mode = args.update_mode or nested_get(overrides, "algorithm.constraint.update_mode", "pid")
    dual_lr = float(args.dual_lr if args.dual_lr is not None else nested_get(overrides, "algorithm.constraint.dual_lr", 0.01))
    pid_kp = float(args.pid_kp if args.pid_kp is not None else nested_get(overrides, "algorithm.constraint.pid_kp", 0.05))
    pid_ki = float(args.pid_ki if args.pid_ki is not None else nested_get(overrides, "algorithm.constraint.pid_ki", 0.001))
    pid_kd = float(args.pid_kd if args.pid_kd is not None else nested_get(overrides, "algorithm.constraint.pid_kd", 0.01))
    integral_mode = args.pid_integral_mode or nested_get(overrides, "algorithm.constraint.pid_integral_mode", "standard")
    integral_decay = float(
        args.pid_integral_decay
        if args.pid_integral_decay is not None
        else nested_get(overrides, "algorithm.constraint.pid_integral_decay", 1.0)
    )
    lambda_init = float(
        args.lambda_init if args.lambda_init is not None else nested_get(overrides, "algorithm.constraint.lambda_init", 0.0)
    )
    lambda_min = float(
        args.lambda_min if args.lambda_min is not None else nested_get(overrides, "algorithm.constraint.lambda_min", 0.0)
    )
    lambda_max = float(
        args.lambda_max if args.lambda_max is not None else nested_get(overrides, "algorithm.constraint.lambda_max", 5.0)
    )
    integral_min = float(
        args.integral_min if args.integral_min is not None else nested_get(overrides, "algorithm.constraint.integral_min", -10.0)
    )
    integral_max = float(
        args.integral_max if args.integral_max is not None else nested_get(overrides, "algorithm.constraint.integral_max", 10.0)
    )
    epsilon = float(args.epsilon if args.epsilon is not None else nested_get(overrides, "algorithm.constraint.epsilon", 1e-6))

    rows = replay_trace(
        trace,
        update_mode=update_mode,
        dual_lr=dual_lr,
        pid_kp=pid_kp,
        pid_ki=pid_ki,
        pid_kd=pid_kd,
        integral_mode=integral_mode,
        integral_decay=integral_decay,
        lambda_init=lambda_init,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        integral_min=integral_min,
        integral_max=integral_max,
        epsilon=epsilon,
    )
    summary = summarize_rows(rows, args.tail)

    print(f"update_mode={update_mode}")
    if update_mode == "pid":
        print(
            "pid="
            f"(kp={pid_kp}, ki={pid_ki}, kd={pid_kd}, integral_mode={integral_mode}, integral_decay={integral_decay})"
        )
    else:
        print(f"dual_lr={dual_lr}")
    print(f"lambda_init={lambda_init} lambda_range=[{lambda_min}, {lambda_max}] integral_range=[{integral_min}, {integral_max}]")
    print(f"positive_delta_iterations={summary['positive_delta_iterations']}")
    print(f"positive_multiplier_count={summary['positive_multiplier_count']}")
    print(f"threshold_cross_iterations={summary['threshold_cross_iterations']}")
    print("tail=")
    for row in summary["tail"]:
        print(
            "  "
            f"iter={row['iteration']} "
            f"error={row['constraint_error']:.4f} "
            f"integral={row['constraint_integral_error']:.4f} "
            f"delta={row['lagrange_delta']:.4f} "
            f"lambda={row['lagrange_multiplier']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
