#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    artifact_dir,
    ensure_directory,
    latest_checkpoint,
    load_config,
    read_json,
    relative_to_repo,
    repo_root,
    run_command,
    write_json,
)

REPO_ROOT = repo_root()
DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "methods" / "logprob_constraint_tau_60_mean_pid_lower_bound_clamp.json"
)
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "baseline" / "train_vanilla_ppo.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "baseline" / "evaluate_checkpoint_sweep.py"
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "artifacts" / "analysis" / "logprob_constraint_diagnostic"

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "train_num_envs": 1,
        "max_iterations": 1,
        "eval_num_envs": 1,
        "episodes": 1,
        "seed": 23,
    },
    "calibration": {
        "train_num_envs": 512,
        "max_iterations": 100,
        "save_interval": 50,
        "eval_num_envs": 32,
        "episodes": 20,
        "seed": 23,
        "checkpoints": "0,50,100",
    },
}


def resolve_config_path(raw: str | Path | None) -> Path:
    if raw is None:
        return DEFAULT_CONFIG
    path = Path(raw)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def preset_value(args: argparse.Namespace, key: str) -> int:
    override = getattr(args, key)
    if override is not None:
        return int(override)
    return int(PRESETS[args.preset][key])


def preset_optional_value(args: argparse.Namespace, key: str, default: Any = None) -> Any:
    override = getattr(args, key)
    if override is not None:
        return override
    return PRESETS[args.preset].get(key, default)


def run_name_for(config: dict[str, Any], args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    seed = preset_value(args, "seed")
    return f"{config['run_name']}_{args.preset}_seed{seed}"


def train_manifest_path(config: dict[str, Any], run_name: str) -> Path:
    return artifact_dir(config, run_name) / "manifest.json"


def eval_summary_path(config: dict[str, Any], run_name: str) -> Path:
    return artifact_dir(config, run_name) / "checkpoint_sweep_summary.json"


def constraint_sidecar_path(config: dict[str, Any], run_name: str) -> Path:
    constraint_logging = config.get("evaluation", {}).get("constraint_logging", {})
    sidecar_filename = str(constraint_logging.get("sidecar_metrics_filename", "constraint_metrics.json"))
    return artifact_dir(config, run_name) / sidecar_filename


def constraint_trace_path(config: dict[str, Any], run_name: str) -> Path:
    constraint_logging = config.get("evaluation", {}).get("constraint_logging", {})
    trace_filename = str(constraint_logging.get("multiplier_trace_filename", "lagrange_multiplier_trace.json"))
    return artifact_dir(config, run_name) / trace_filename


def train_complete(config: dict[str, Any], run_name: str) -> bool:
    manifest_path = train_manifest_path(config, run_name)
    if not manifest_path.exists():
        return False
    try:
        manifest = read_json(manifest_path)
    except json.JSONDecodeError:
        return False
    checkpoint_path = manifest.get("checkpoint_path")
    if not isinstance(checkpoint_path, str):
        return False
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()
    return checkpoint.exists()


def evaluate_complete(config: dict[str, Any], run_name: str) -> bool:
    return eval_summary_path(config, run_name).exists()


def resolve_load_run(config: dict[str, Any], run_name: str) -> str:
    manifest_path = train_manifest_path(config, run_name)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        run_dir = manifest.get("run_dir")
        if isinstance(run_dir, str):
            return run_dir
    return run_name


def latest_checkpoint_id(config: dict[str, Any], run_name: str) -> int | None:
    manifest_path = train_manifest_path(config, run_name)
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    run_dir = manifest.get("run_dir")
    if not isinstance(run_dir, str):
        return None
    path = Path(run_dir)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        return None
    return int(latest_checkpoint(path).stem.split("_")[-1])


def build_train_command(config_path: Path, run_name: str, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        f"--config={relative_to_repo(config_path)}",
        f"--run-name={run_name}",
        f"--num-envs={preset_value(args, 'train_num_envs')}",
        f"--max-iterations={preset_value(args, 'max_iterations')}",
        f"--seed={preset_value(args, 'seed')}",
    ]
    save_interval = preset_optional_value(args, "save_interval")
    if save_interval is not None:
        command.append(f"--save-interval={int(save_interval)}")
    if args.humanoid_gym_root:
        command.append(f"--humanoid-gym-root={args.humanoid_gym_root}")
    if args.rl_device:
        command.append(f"--rl-device={args.rl_device}")
    if args.sim_device:
        command.append(f"--sim-device={args.sim_device}")
    return command


def build_evaluate_command(
    config: dict[str, Any], config_path: Path, run_name: str, args: argparse.Namespace
) -> list[str]:
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        f"--config={relative_to_repo(config_path)}",
        "--run-name",
        run_name,
        "--load-run",
        resolve_load_run(config, run_name),
        "--num-envs",
        str(preset_value(args, "eval_num_envs")),
        "--episodes",
        str(preset_value(args, "episodes")),
        "--seed",
        str(preset_value(args, "seed")),
    ]
    checkpoints = preset_optional_value(args, "checkpoints")
    if checkpoints:
        command.extend(["--checkpoints", str(checkpoints)])
    else:
        checkpoint = latest_checkpoint_id(config, run_name)
        if checkpoint is not None:
            command.extend(["--checkpoints", str(checkpoint)])
    if args.humanoid_gym_root:
        command.extend(["--humanoid-gym-root", args.humanoid_gym_root])
    if args.rl_device:
        command.extend(["--rl-device", args.rl_device])
    if args.sim_device:
        command.extend(["--sim-device", args.sim_device])
    return command


def is_finite_json(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(is_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(is_finite_json(item) for item in value.values())
    return False


def calibration_recommendation(
    constraint_metrics: dict[str, Any] | None,
    trace_entries: list[dict[str, Any]] | None,
    sweep: dict[str, Any] | None,
) -> dict[str, Any]:
    assessment: dict[str, Any] = {
        "available": bool(constraint_metrics and trace_entries and sweep),
        "recommendation": "stop_branch_here",
        "decision_boundary": [
            "Request three-seed budget only if the initial tau=6.0 recipe stays finite, avoids early sustained lambda saturation, and keeps train-side cost on the same rough scale as tau.",
            "Stop the branch here if the calibration reads as effectively inactive, obviously over-tight, or behaviorally collapsed by checkpoint 100.",
        ],
    }
    if not assessment["available"]:
        assessment["reason"] = "missing_trace_or_sweep_artifacts"
        return assessment

    tau = float(constraint_metrics.get("constraint_threshold", 0.0) or 0.0)
    lambda_max = float(constraint_metrics.get("lagrange_multiplier_max", 0.0) or 0.0)
    lambda_values = [
        float(entry["lagrange_multiplier"])
        for entry in trace_entries
        if isinstance(entry.get("lagrange_multiplier"), (int, float))
    ]
    cost_values = [
        float(entry["logprob_gradient_cost_update"])
        for entry in trace_entries
        if isinstance(entry.get("logprob_gradient_cost_update"), (int, float))
    ]
    finite = is_finite_json(trace_entries) and is_finite_json(constraint_metrics)
    assessment["finite_trace"] = finite
    if not finite or not lambda_values or not cost_values or tau <= 0.0:
        assessment["reason"] = "non_finite_or_incomplete_trace"
        return assessment

    saturation_threshold = 0.95 * lambda_max if lambda_max > 0.0 else None
    saturated_indices = (
        [idx for idx, value in enumerate(lambda_values) if value >= saturation_threshold]
        if saturation_threshold is not None
        else []
    )
    first_saturation_iteration = saturated_indices[0] if saturated_indices else None
    sustained_saturation_fraction = 0.0
    if first_saturation_iteration is not None:
        sustained_window = lambda_values[first_saturation_iteration:]
        sustained_saturation_fraction = sum(
            1 for value in sustained_window if value >= saturation_threshold
        ) / max(len(sustained_window), 1)
    early_sustained_saturation = (
        first_saturation_iteration is not None
        and first_saturation_iteration < max(5, len(lambda_values) // 10)
        and sustained_saturation_fraction >= 0.8
    )

    recent_window = max(5, len(cost_values) // 5)
    recent_cost_mean = statistics.fmean(cost_values[-recent_window:])
    overall_cost_mean = statistics.fmean(cost_values)
    max_cost = max(cost_values)
    recent_cost_ratio = recent_cost_mean / tau
    max_cost_ratio = max_cost / tau
    max_lambda_seen = max(lambda_values)

    if max_cost_ratio < 0.5 and max_lambda_seen <= 0.1 * lambda_max:
        cost_scale_reading = "too_loose_below_tau"
    elif early_sustained_saturation and recent_cost_ratio > 1.5:
        cost_scale_reading = "too_tight_saturating"
    elif 0.5 <= recent_cost_ratio <= 1.5:
        cost_scale_reading = "on_scale"
    elif recent_cost_ratio > 1.5:
        cost_scale_reading = "high_drift_above_tau"
    else:
        cost_scale_reading = "low_drift_below_tau"

    rows = sweep.get("rows") if isinstance(sweep, dict) else None
    final_row = None
    if isinstance(rows, list) and rows:
        final_row = max(rows, key=lambda row: int(row.get("checkpoint", -1)))
    final_checkpoint_fall_rate = None if final_row is None else final_row.get("fall_rate")
    final_checkpoint_collapsed = (
        isinstance(final_checkpoint_fall_rate, (int, float)) and float(final_checkpoint_fall_rate) >= 0.95
    )
    selection_status = sweep.get("selection_status") if isinstance(sweep, dict) else None
    all_checkpoints_collapsed = selection_status == "all_checkpoints_collapsed"

    assessment.update(
        {
            "tau": tau,
            "lambda_max": lambda_max,
            "max_lambda_seen": max_lambda_seen,
            "first_saturation_iteration": first_saturation_iteration,
            "early_sustained_saturation": early_sustained_saturation,
            "sustained_saturation_fraction_after_first_hit": sustained_saturation_fraction,
            "overall_cost_mean": overall_cost_mean,
            "recent_cost_mean": recent_cost_mean,
            "recent_cost_ratio_to_tau": recent_cost_ratio,
            "max_cost_ratio_to_tau": max_cost_ratio,
            "cost_scale_reading": cost_scale_reading,
            "selection_status": selection_status,
            "final_checkpoint": None if final_row is None else final_row.get("checkpoint"),
            "final_checkpoint_fall_rate": final_checkpoint_fall_rate,
            "final_checkpoint_collapsed": final_checkpoint_collapsed,
        }
    )

    request_three_seed_budget = (
        cost_scale_reading == "on_scale"
        and not early_sustained_saturation
        and not final_checkpoint_collapsed
        and not all_checkpoints_collapsed
    )
    assessment["recommendation"] = (
        "request_three_seed_budget" if request_three_seed_budget else "stop_branch_here"
    )

    if request_three_seed_budget:
        assessment["reason"] = (
            "seed23 calibration stayed finite, kept train-side cost on the tau scale, "
            "and did not obviously collapse by checkpoint 100."
        )
    elif cost_scale_reading == "too_loose_below_tau":
        assessment["reason"] = (
            "Under the initial tau=6.0 recipe, train-side cost stayed far below tau and lambda stayed near the floor, "
            "so the hard constraint was effectively inactive."
        )
    elif cost_scale_reading == "too_tight_saturating" or early_sustained_saturation:
        assessment["reason"] = (
            "The multiplier hit the upper region too early and stayed there, so the initial recipe reads as over-tight "
            "rather than decision-ready."
        )
    elif final_checkpoint_collapsed or all_checkpoints_collapsed:
        assessment["reason"] = (
            "Even with finite training traces, the checkpoint sweep remained behaviorally collapsed, so the branch "
            "does not currently justify three-seed budget."
        )
    else:
        assessment["reason"] = (
            "The calibration remained finite but did not keep train-side cost near tau in a way that is clear enough "
            "to justify canonical three-seed expansion."
        )
    return assessment


def collect_summary(config: dict[str, Any], run_name: str, args: argparse.Namespace) -> dict[str, Any]:
    output_dir = artifact_dir(config, run_name)
    summary_path = eval_summary_path(config, run_name)
    manifest_path = train_manifest_path(config, run_name)
    sidecar_path = constraint_sidecar_path(config, run_name)
    trace_path = constraint_trace_path(config, run_name)

    payload: dict[str, Any] = {
        "diagnostic": (
            "logprob_gradient_hard_constraint_scale_calibration"
            if args.preset == "calibration"
            else "logprob_gradient_hard_constraint_smoke"
        ),
        "scope": "Cell D seed23 calibration" if args.preset == "calibration" else "Cell D post-freeze smoke",
        "run_name": run_name,
        "preset": args.preset,
        "train_num_envs": preset_value(args, "train_num_envs"),
        "max_iterations": preset_value(args, "max_iterations"),
        "eval_num_envs": preset_value(args, "eval_num_envs"),
        "episodes": preset_value(args, "episodes"),
        "seed": preset_value(args, "seed"),
        "save_interval": (
            int(preset_optional_value(args, "save_interval"))
            if preset_optional_value(args, "save_interval") is not None
            else None
        ),
        "requested_checkpoints": preset_optional_value(args, "checkpoints"),
        "train_manifest_path": relative_to_repo(manifest_path) if manifest_path.exists() else None,
        "checkpoint_sweep_summary_path": relative_to_repo(summary_path) if summary_path.exists() else None,
        "constraint_metrics_path": relative_to_repo(sidecar_path) if sidecar_path.exists() else None,
        "constraint_trace_path": relative_to_repo(trace_path) if trace_path.exists() else None,
        "status": "evaluated" if summary_path.exists() else "train_only" if manifest_path.exists() else "planned",
        "interpretation_boundary": (
            [
                "This calibration asks whether the initial tau=6.0 hard-constraint recipe is on-scale enough to justify canonical three-seed budget.",
                "It is still a bounded post-freeze diagnostic and not a formal candidate result.",
                "The main question is whether multiplier behavior, train-side cost, and checkpoint viability are interpretable on seed23.",
            ]
            if args.preset == "calibration"
            else [
                "This smoke verifies that the logprob-gradient hard-constraint path can train, checkpoint, reload, and emit comparable artifacts.",
                "It does not claim task-validity or formal candidate status.",
                "The main smoke question is artifact integrity and finite multiplier / constraint traces.",
            ]
        ),
    }

    sidecar_metrics = None
    if sidecar_path.exists():
        sidecar_metrics = read_json(sidecar_path)
        payload["constraint_metrics"] = sidecar_metrics
        payload["constraint_metrics_finite"] = is_finite_json(sidecar_metrics)

    trace_entries = None
    if trace_path.exists():
        trace_payload = read_json(trace_path)
        trace_entries = trace_payload.get("trace")
        if isinstance(trace_entries, list):
            payload["constraint_trace_length"] = len(trace_entries)
            payload["constraint_trace_finite"] = is_finite_json(trace_entries)
            if trace_entries:
                payload["latest_trace_entry"] = trace_entries[-1]

    sweep = None
    if summary_path.exists():
        sweep = read_json(summary_path)
        payload["selection_status"] = sweep.get("selection_status")
        payload["selected_checkpoint"] = sweep.get("best_checkpoint")
        payload["selected_metrics_path"] = sweep.get("selected_metrics_path")
        rows = sweep.get("rows")
        if isinstance(rows, list) and rows:
            payload["selected_row"] = next(
                (row for row in rows if row.get("checkpoint") == sweep.get("best_checkpoint")),
                rows[0],
            )
        if args.preset == "calibration":
            payload["calibration_assessment"] = calibration_recommendation(sidecar_metrics, trace_entries, sweep)

    return payload


def write_analysis_summary(config: dict[str, Any], run_name: str, args: argparse.Namespace) -> Path:
    output_root = ensure_directory(Path(args.analysis_root).expanduser().resolve())
    output_path = output_root / f"{run_name}_summary.json"
    write_json(output_path, collect_summary(config, run_name, args))
    return output_path


def print_plan(config: dict[str, Any], config_path: Path, run_name: str, args: argparse.Namespace) -> None:
    print(f"Logprob hard-constraint diagnostic preset: {args.preset}")
    print(f"config: {relative_to_repo(config_path)}")
    print(f"run_name: {run_name}")
    print(f"train: {'complete' if train_complete(config, run_name) else 'pending'}")
    print("command: " + " ".join(build_train_command(config_path, run_name, args)))
    print(f"evaluate: {'complete' if evaluate_complete(config, run_name) else 'pending'}")
    print("command: " + " ".join(build_evaluate_command(config, config_path, run_name, args)))
    print(f"analysis summary: {relative_to_repo(Path(args.analysis_root) / f'{run_name}_summary.json')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a bounded logprob-gradient hard-constraint smoke diagnostic."
    )
    parser.add_argument("--config", default=None, help="Path to the logprob hard-constraint method config JSON.")
    parser.add_argument("--run-name", default=None, help="Override the artifact and upstream run name.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke", help="Diagnostic budget preset.")
    parser.add_argument("--stage", choices=("plan", "train", "evaluate", "all"), default="plan")
    parser.add_argument("--skip-completed", action="store_true", help="Skip completed stages.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Optional upstream checkout override.")
    parser.add_argument("--train-num-envs", type=int, default=None, help="Override preset training num_envs.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override preset training iterations.")
    parser.add_argument("--save-interval", type=int, default=None, help="Override preset checkpoint save interval.")
    parser.add_argument("--eval-num-envs", type=int, default=None, help="Override preset evaluation num_envs.")
    parser.add_argument("--episodes", type=int, default=None, help="Override preset evaluation episode count.")
    parser.add_argument("--checkpoints", default=None, help="Override preset checkpoint sweep list, e.g. 0,50,100.")
    parser.add_argument("--seed", type=int, default=None, help="Override preset seed.")
    parser.add_argument("--rl-device", default=None, help="Optional RL device override.")
    parser.add_argument("--sim-device", default=None, help="Optional sim device override.")
    parser.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_ROOT), help="Directory for compact summaries.")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    run_name = run_name_for(config, args)

    if args.stage == "plan":
        print_plan(config, config_path, run_name, args)
        return 0

    if args.stage in {"train", "all"}:
        if not (args.skip_completed and train_complete(config, run_name)):
            train_status = run_command(
                build_train_command(config_path, run_name, args),
                cwd=REPO_ROOT,
                dry_run=args.dry_run,
            )
            if train_status != 0:
                if train_complete(config, run_name):
                    print(
                        "train exited with code "
                        f"{train_status} after writing "
                        f"{relative_to_repo(train_manifest_path(config, run_name))}; continuing.",
                        file=sys.stderr,
                    )
                else:
                    return int(train_status)

    if args.stage in {"evaluate", "all"}:
        if not (args.skip_completed and evaluate_complete(config, run_name)):
            eval_status = run_command(
                build_evaluate_command(config, config_path, run_name, args),
                cwd=REPO_ROOT,
                dry_run=args.dry_run,
            )
            if eval_status != 0:
                if evaluate_complete(config, run_name):
                    print(
                        "evaluate exited with code "
                        f"{eval_status} after writing "
                        f"{relative_to_repo(eval_summary_path(config, run_name))}; continuing.",
                        file=sys.stderr,
                    )
                else:
                    return int(eval_status)

    summary_path = write_analysis_summary(config, run_name, args)
    print(f"Wrote analysis summary to {relative_to_repo(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
