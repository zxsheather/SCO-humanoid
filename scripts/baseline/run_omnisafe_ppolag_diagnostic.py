#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    artifact_dir,
    ensure_directory,
    load_config,
    read_json,
    relative_to_repo,
    repo_root,
    runtime_env,
    write_json,
)


REPO_ROOT = repo_root()
DEFAULT_SWEEP_CONFIG = REPO_ROOT / "configs" / "sweeps" / "rough_terrain_omnisafe_ppolag_diagnostic.json"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "baseline" / "train_omnisafe_ppolag_diagnostic.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "baseline" / "evaluate_omnisafe_policy.py"
SUMMARIZE_SCRIPT = REPO_ROOT / "scripts" / "baseline" / "summarize_omnisafe_checkpoint_sweep.py"
RESULT_NOTE = REPO_ROOT / "docs" / "full-paper" / "omnisafe-ppolag-diagnostic-results.md"
COMPARISON_METRIC_KEYS = [
    "velocity_tracking_error_mean",
    "joint_acceleration_l2_mean",
    "action_jitter_l2_mean",
    "episode_return_mean",
    "fall_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or summarize the bounded OmniSafe PPO-Lag diagnostic (#63).")
    parser.add_argument("--sweep-config", default=str(DEFAULT_SWEEP_CONFIG))
    parser.add_argument("--stage", choices=("plan", "train", "evaluate", "summary", "all"), default="plan")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--humanoid-gym-root", default=None)
    parser.add_argument("--train-num-envs", type=int, default=None)
    parser.add_argument("--eval-num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--rl-device", default=None)
    parser.add_argument("--sim-device", default=None)
    return parser.parse_args()


def load_sweep_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def candidate_run_name(config: dict[str, Any], seed: int) -> str:
    return f"{config['run_name']}_seed{seed}"


def train_status_path(config: dict[str, Any], run_name: str) -> Path:
    return artifact_dir(config, run_name) / "omnisafe_training.json"


def eval_status_path(config: dict[str, Any], run_name: str) -> Path:
    return artifact_dir(config, run_name) / "omnisafe_evaluation.json"


def eval_summary_path(config: dict[str, Any], run_name: str) -> Path:
    return artifact_dir(config, run_name) / "checkpoint_sweep_summary.json"


def metrics_checkpoint_path(config: dict[str, Any], run_name: str, checkpoint: int) -> Path:
    return artifact_dir(config, run_name) / f"metrics_checkpoint_{checkpoint}.json"


def checkpoint_path(config: dict[str, Any], run_name: str, checkpoint: int) -> Path:
    return artifact_dir(config, run_name) / "checkpoints" / f"model_{checkpoint}.pt"


def checkpoints_from_sweep(sweep_cfg: dict[str, Any]) -> list[int]:
    return [int(value) for value in sweep_cfg["checkpoints"]]


def train_complete(config: dict[str, Any], run_name: str, checkpoints: list[int]) -> bool:
    status_path = train_status_path(config, run_name)
    if not status_path.exists():
        return False
    try:
        status = read_json(status_path)
    except json.JSONDecodeError:
        return False
    if status.get("status") != "complete":
        return False
    return all(checkpoint_path(config, run_name, checkpoint).exists() for checkpoint in checkpoints)


def evaluation_checkpoint_complete(config: dict[str, Any], run_name: str, checkpoint: int) -> bool:
    if not metrics_checkpoint_path(config, run_name, checkpoint).exists():
        return False
    status_path = eval_status_path(config, run_name)
    if not status_path.exists():
        return True
    try:
        status = read_json(status_path)
    except json.JSONDecodeError:
        return False
    return status.get("status") == "complete" or int(status.get("checkpoint", checkpoint)) != checkpoint


def evaluate_complete(config: dict[str, Any], run_name: str) -> bool:
    return eval_summary_path(config, run_name).exists()


def build_train_command(
    config_path: Path,
    config: dict[str, Any],
    run_name: str,
    seed: int,
    sweep_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        f"--config={relative_to_repo(config_path)}",
        f"--run-name={run_name}",
        f"--seed={seed}",
        "--write-failure-artifact",
    ]
    if args.humanoid_gym_root:
        command.append(f"--humanoid-gym-root={args.humanoid_gym_root}")
    train_num_envs = args.train_num_envs if args.train_num_envs is not None else sweep_cfg.get("train_num_envs")
    max_iterations = args.max_iterations if args.max_iterations is not None else sweep_cfg.get("max_iterations")
    if train_num_envs is not None:
        command.append(f"--num-envs={int(train_num_envs)}")
    if max_iterations is not None:
        command.append(f"--max-iterations={int(max_iterations)}")
    if args.rl_device:
        command.append(f"--rl-device={args.rl_device}")
    elif config.get("training", {}).get("rl_device"):
        command.append(f"--rl-device={config['training']['rl_device']}")
    if args.sim_device:
        command.append(f"--sim-device={args.sim_device}")
    elif config.get("training", {}).get("sim_device"):
        command.append(f"--sim-device={config['training']['sim_device']}")
    return command


def build_eval_command(
    config_path: Path,
    config: dict[str, Any],
    run_name: str,
    seed: int,
    checkpoint: int,
    sweep_cfg: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        f"--config={relative_to_repo(config_path)}",
        f"--run-name={run_name}",
        f"--checkpoint={checkpoint}",
        f"--checkpoint-path={relative_to_repo(checkpoint_path(config, run_name, checkpoint))}",
        f"--seed={seed}",
    ]
    if args.humanoid_gym_root:
        command.append(f"--humanoid-gym-root={args.humanoid_gym_root}")
    eval_num_envs = args.eval_num_envs if args.eval_num_envs is not None else sweep_cfg.get("eval_num_envs")
    episodes = args.episodes if args.episodes is not None else sweep_cfg.get("episodes")
    if eval_num_envs is not None:
        command.append(f"--num-envs={int(eval_num_envs)}")
    if episodes is not None:
        command.append(f"--episodes={int(episodes)}")
    if args.rl_device:
        command.append(f"--rl-device={args.rl_device}")
    elif config.get("evaluation", {}).get("rl_device"):
        command.append(f"--rl-device={config['evaluation']['rl_device']}")
    if args.sim_device:
        command.append(f"--sim-device={args.sim_device}")
    elif config.get("evaluation", {}).get("sim_device"):
        command.append(f"--sim-device={config['evaluation']['sim_device']}")
    return command


def build_summarize_command(config_path: Path, run_name: str, checkpoints: list[int]) -> list[str]:
    return [
        sys.executable,
        str(SUMMARIZE_SCRIPT),
        f"--config={relative_to_repo(config_path)}",
        f"--run-name={run_name}",
        f"--checkpoints={','.join(str(checkpoint) for checkpoint in checkpoints)}",
    ]


def run_command(
    command: list[str],
    *,
    dry_run: bool,
    recovery_check=None,
    recovery_label: str = "artifact",
) -> int:
    print(" ".join(command), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(REPO_ROOT), env=runtime_env(), check=False)
    if completed.returncode == 0:
        return 0
    if recovery_check is not None and recovery_check():
        print(
            f"Recovered non-zero Isaac exit ({completed.returncode}) because {recovery_label} is complete.",
            file=sys.stderr,
            flush=True,
        )
        return 0
    return completed.returncode


def aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    for key in COMPARISON_METRIC_KEYS:
        values = [float(metrics[key]) for metrics in metrics_list if metrics.get(key) is not None]
        if not values:
            continue
        aggregate[f"{key}_mean"] = statistics.fmean(values)
        aggregate[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return aggregate


def latest_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: int(row["checkpoint"]))


def metrics_snapshot(row: dict[str, Any] | None) -> dict[str, float] | None:
    if row is None:
        return None
    if any(row.get(key) is None for key in COMPARISON_METRIC_KEYS):
        return None
    return {key: float(row[key]) for key in COMPARISON_METRIC_KEYS}


def collect_anchor_summary(sweep_cfg: dict[str, Any]) -> dict[str, Any] | None:
    anchor_path_raw = sweep_cfg.get("_comparison_anchors", {}).get("extended_seed_summary")
    if not anchor_path_raw:
        return None
    anchor_path = Path(anchor_path_raw)
    if not anchor_path.is_absolute():
        anchor_path = REPO_ROOT / anchor_path
    if not anchor_path.exists():
        return None

    anchor_payload = read_json(anchor_path)
    seed_slice = [str(seed) for seed in sweep_cfg.get("_comparison_anchors", {}).get("anchor_seed_slice", [])]
    anchors: dict[str, Any] = {
        "source_path": relative_to_repo(anchor_path),
        "full_seed_set": anchor_payload.get("seeds"),
        "seed_slice": [int(seed) for seed in seed_slice],
        "candidates": {},
    }
    for candidate in anchor_payload.get("candidates", []):
        candidate_id = candidate.get("id")
        per_seed = candidate.get("per_seed", {})
        slice_metrics = [
            per_seed[seed]["selected_metrics"]
            for seed in seed_slice
            if seed in per_seed and isinstance(per_seed[seed].get("selected_metrics"), dict)
        ]
        anchors["candidates"][candidate_id] = {
            "label": candidate.get("label"),
            "status": candidate.get("status"),
            "full_selected_aggregate": candidate.get("selected_aggregate") or candidate.get("aggregate"),
            "seed_slice_selected_aggregate": aggregate_metrics(slice_metrics) if slice_metrics else None,
            "seed_slice_selected_checkpoints": {
                seed: candidate.get("selected_checkpoints", {}).get(seed) for seed in seed_slice
            },
        }
    return anchors


def collect_summary(
    sweep_cfg: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    checkpoints: list[int],
) -> dict[str, Any]:
    selected_metrics_list: list[dict[str, Any]] = []
    final_metrics_list: list[dict[str, float]] = []
    per_seed: dict[str, Any] = {}
    missing_seeds: list[int] = []
    selected_checkpoints: dict[str, int] = {}
    final_checkpoints: dict[str, int] = {}
    selection_statuses: dict[str, str] = {}

    for seed in [int(value) for value in sweep_cfg["seeds"]]:
        run_name = candidate_run_name(config, seed)
        summary_path = eval_summary_path(config, run_name)
        if not summary_path.exists():
            missing_seeds.append(seed)
            continue
        summary = read_json(summary_path)
        selected = summary.get("selected_metrics")
        rows = summary.get("rows", [])
        if not isinstance(selected, dict) or not isinstance(rows, list):
            missing_seeds.append(seed)
            continue
        final = metrics_snapshot(latest_row(rows))
        selected_snapshot = metrics_snapshot(selected)
        if selected_snapshot is None:
            missing_seeds.append(seed)
            continue
        selected_checkpoint = int(summary.get("selected_checkpoint", summary.get("best_checkpoint")))
        final_row = latest_row(rows)
        per_seed[str(seed)] = {
            "run_name": run_name,
            "training_status_path": relative_to_repo(train_status_path(config, run_name)),
            "checkpoint_sweep_summary_path": relative_to_repo(summary_path),
            "selection_status": summary.get("selection_status"),
            "selected_checkpoint": selected_checkpoint,
            "selected_metrics_path": selected.get("metrics_path") or summary.get("selected_metrics_path"),
            "selected_metrics": selected_snapshot,
            "final_checkpoint": int(final_row["checkpoint"]) if final_row is not None else None,
            "final_metrics": final,
        }
        selected_metrics_list.append(selected_snapshot)
        if final is not None:
            final_metrics_list.append(final)
        selected_checkpoints[str(seed)] = selected_checkpoint
        if final_row is not None:
            final_checkpoints[str(seed)] = int(final_row["checkpoint"])
        selection_statuses[str(seed)] = str(summary.get("selection_status"))

    status = "complete" if not missing_seeds and len(selected_metrics_list) == len(sweep_cfg["seeds"]) else "incomplete"
    if status == "complete" and selection_statuses and all(value != "selected" for value in selection_statuses.values()):
        status = "collapsed"

    payload: dict[str, Any] = {
        "comparison_name": sweep_cfg["name"],
        "issue": "#63",
        "diagnostic_only": True,
        "config_path": relative_to_repo(config_path),
        "method": config.get("method"),
        "seeds": sweep_cfg["seeds"],
        "checkpoints": checkpoints,
        "missing_seeds": missing_seeds,
        "status": status,
        "selected_checkpoints": selected_checkpoints,
        "final_checkpoints": final_checkpoints,
        "selection_statuses": selection_statuses,
        "per_seed": per_seed,
        "selected_aggregate": aggregate_metrics(selected_metrics_list) if selected_metrics_list else None,
        "final_aggregate": aggregate_metrics(final_metrics_list) if final_metrics_list else None,
        "comparison_anchors": collect_anchor_summary(sweep_cfg),
        "claim_boundary": (
            "Bounded three-seed external-baseline diagnostic only; does not replace the #51 "
            "five-seed SC-PPO record or establish MuJoCo transfer."
        ),
    }
    return payload


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def metric_table_row(label: str, aggregate: dict[str, Any] | None) -> str:
    aggregate = aggregate or {}
    return (
        f"| {label} | {fmt(aggregate.get('fall_rate_mean'))} | "
        f"{fmt(aggregate.get('velocity_tracking_error_mean_mean'))} | "
        f"{fmt(aggregate.get('joint_acceleration_l2_mean_mean'))} | "
        f"{fmt(aggregate.get('action_jitter_l2_mean_mean'))} | "
        f"{fmt(aggregate.get('episode_return_mean_mean'))} |"
    )


def write_result_note(summary: dict[str, Any], summary_path: Path) -> None:
    per_seed = summary.get("per_seed", {})
    anchors = summary.get("comparison_anchors") or {}
    anchor_candidates = anchors.get("candidates", {})
    lines = [
        "# OmniSafe PPO-Lag Diagnostic Result (#63)",
        "",
        "## Status",
        "",
        f"Status: `{summary['status']}`.",
        "",
        "This is a bounded three-seed diagnostic for the external constrained-RL baseline path. "
        "It uses the #65 Jacobian update hook and #62 OmniSafe policy evaluator; it does not "
        "replace the #51 five-seed SC-PPO record.",
        "",
        "Artifacts:",
        "",
        f"- Summary: `{relative_to_repo(summary_path)}`",
        f"- Config: `{summary['config_path']}`",
        "",
        "Because every evaluated checkpoint collapsed (`fall_rate = 1.0`), the low joint-acceleration "
        "and jitter numbers are not evidence of smooth locomotion; they are artifacts of policies that "
        "fall rather than move successfully.",
        "",
        "## Selected-Checkpoint Aggregate",
        "",
        "| Method / anchor | Fall | Vel. err | Jnt acc | Jitter | Ep. return |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        metric_table_row("OmniSafe PPO-Lag diagnostic, seeds 23/29/31", summary.get("selected_aggregate")),
    ]
    for candidate_id, candidate in anchor_candidates.items():
        label = candidate.get("label") or candidate_id
        lines.append(metric_table_row(f"{label}, same 3-seed slice", candidate.get("seed_slice_selected_aggregate")))

    lines.extend(
        [
            "",
            "## Final-Checkpoint Aggregate",
            "",
            "| Method | Fall | Vel. err | Jnt acc | Jitter | Ep. return |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            metric_table_row("OmniSafe PPO-Lag diagnostic, final checkpoint 400", summary.get("final_aggregate")),
            "",
            "## Per-Seed Selection",
            "",
            "| Seed | Status | Selected ckpt | Fall | Vel. err | Jnt acc | Jitter | Ep. return |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in summary["seeds"]:
        row = per_seed.get(str(seed), {})
        metrics = row.get("selected_metrics") or {}
        lines.append(
            f"| {seed} | `{row.get('selection_status', 'missing')}` | "
            f"{row.get('selected_checkpoint', 'n/a')} | "
            f"{fmt(metrics.get('fall_rate'))} | "
            f"{fmt(metrics.get('velocity_tracking_error_mean'))} | "
            f"{fmt(metrics.get('joint_acceleration_l2_mean'))} | "
            f"{fmt(metrics.get('action_jitter_l2_mean'))} | "
            f"{fmt(metrics.get('episode_return_mean'))} |"
        )

    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- The result is diagnostic-only and should not be described as a full OmniSafe PPO-Lag replacement for SC-PPO.",
            "- No MuJoCo replay or five-seed expansion is implied unless all selected checkpoints are task-valid and the user approves expanded budget.",
            "- The bridge uses the same policy-local-sensitivity/Jacobian cost family; it does not substitute action-rate, torque, fall, or other proxy costs.",
            "",
        ]
    )
    ensure_directory(RESULT_NOTE.parent)
    RESULT_NOTE.write_text("\n".join(lines), encoding="utf-8")


def write_summary(sweep_cfg: dict[str, Any], config: dict[str, Any], config_path: Path, checkpoints: list[int]) -> Path:
    output_root = ensure_directory((REPO_ROOT / sweep_cfg["analysis_root"]).resolve())
    payload = collect_summary(sweep_cfg, config, config_path, checkpoints)
    output_path = output_root / "comparison_summary.json"
    write_json(output_path, payload)
    write_result_note(payload, output_path)
    return output_path


def print_plan(sweep_cfg: dict[str, Any], config: dict[str, Any], config_path: Path, checkpoints: list[int], args: argparse.Namespace) -> None:
    print(f"[{sweep_cfg['candidate']['id']}] {sweep_cfg['candidate']['label']}")
    print(f"  config: {relative_to_repo(config_path)}")
    print(f"  checkpoints: {','.join(str(checkpoint) for checkpoint in checkpoints)}")
    for seed in [int(value) for value in sweep_cfg["seeds"]]:
        run_name = candidate_run_name(config, seed)
        train_status = "complete" if train_complete(config, run_name, checkpoints) else "pending"
        eval_status = "complete" if evaluate_complete(config, run_name) else "pending"
        print(f"  seed {seed}")
        print(f"    train: {train_status}")
        print(
            "    command: "
            + " ".join(build_train_command(config_path, config, run_name, seed, sweep_cfg, args))
        )
        print(f"    evaluate: {eval_status}")
        for checkpoint in checkpoints:
            print(
                "    command: "
                + " ".join(build_eval_command(config_path, config, run_name, seed, checkpoint, sweep_cfg, args))
            )
        print("    command: " + " ".join(build_summarize_command(config_path, run_name, checkpoints)))


def run_train_stage(sweep_cfg: dict[str, Any], config: dict[str, Any], config_path: Path, checkpoints: list[int], args: argparse.Namespace) -> int:
    for seed in [int(value) for value in sweep_cfg["seeds"]]:
        run_name = candidate_run_name(config, seed)
        if args.skip_completed and train_complete(config, run_name, checkpoints):
            print(f"Skipping train seed {seed}: {relative_to_repo(train_status_path(config, run_name))} already exists")
            continue
        command = build_train_command(config_path, config, run_name, seed, sweep_cfg, args)
        exit_code = run_command(
            command,
            dry_run=args.dry_run,
            recovery_check=lambda c=config, r=run_name: train_complete(c, r, checkpoints),
            recovery_label=f"training artifact for seed {seed}",
        )
        if exit_code != 0:
            return exit_code
    return 0


def run_evaluate_stage(sweep_cfg: dict[str, Any], config: dict[str, Any], config_path: Path, checkpoints: list[int], args: argparse.Namespace) -> int:
    for seed in [int(value) for value in sweep_cfg["seeds"]]:
        run_name = candidate_run_name(config, seed)
        if args.skip_completed and evaluate_complete(config, run_name):
            print(f"Skipping evaluate seed {seed}: {relative_to_repo(eval_summary_path(config, run_name))} already exists")
            continue
        for checkpoint in checkpoints:
            if args.skip_completed and metrics_checkpoint_path(config, run_name, checkpoint).exists():
                print(f"Skipping eval seed {seed} checkpoint {checkpoint}: metrics already exists")
                continue
            command = build_eval_command(config_path, config, run_name, seed, checkpoint, sweep_cfg, args)
            exit_code = run_command(
                command,
                dry_run=args.dry_run,
                recovery_check=lambda c=config, r=run_name, ckpt=checkpoint: evaluation_checkpoint_complete(c, r, ckpt),
                recovery_label=f"evaluation artifact for seed {seed} checkpoint {checkpoint}",
            )
            if exit_code != 0:
                return exit_code
        summarize_command = build_summarize_command(config_path, run_name, checkpoints)
        exit_code = run_command(summarize_command, dry_run=args.dry_run)
        if exit_code != 0:
            return exit_code
    return 0


def main() -> int:
    args = parse_args()
    sweep_cfg = load_sweep_config(args.sweep_config)
    config_path = resolve_config_path(sweep_cfg["candidate"]["config"])
    config = load_config(config_path)
    checkpoints = checkpoints_from_sweep(sweep_cfg)

    if args.stage == "plan":
        print_plan(sweep_cfg, config, config_path, checkpoints, args)
        return 0

    if args.stage in {"train", "all"}:
        exit_code = run_train_stage(sweep_cfg, config, config_path, checkpoints, args)
        if exit_code != 0:
            return exit_code

    if args.stage in {"evaluate", "all"}:
        exit_code = run_evaluate_stage(sweep_cfg, config, config_path, checkpoints, args)
        if exit_code != 0:
            return exit_code

    if args.stage in {"summary", "all"} and not args.dry_run:
        summary_path = write_summary(sweep_cfg, config, config_path, checkpoints)
        print(f"Wrote {relative_to_repo(summary_path)}")
        print(f"Wrote {relative_to_repo(RESULT_NOTE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
