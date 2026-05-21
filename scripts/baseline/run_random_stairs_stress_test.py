#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
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
    run_command,
    write_json,
)

REPO_ROOT = repo_root()
DEFAULT_SWEEP_CONFIG = REPO_ROOT / "configs" / "sweeps" / "random_stairs_selected_checkpoint_stress.json"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "baseline" / "evaluate_checkpoint_sweep.py"

METRIC_KEYS = [
    "velocity_tracking_error_mean",
    "joint_acceleration_l2_mean",
    "action_jitter_l2_mean",
    "episode_return_mean",
    "fall_rate",
]


def load_sweep_config(config_path: str | Path | None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_SWEEP_CONFIG
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def selected_checkpoint(candidate: dict[str, Any], seed: int) -> int:
    checkpoints = candidate.get("selected_checkpoints", {})
    try:
        return int(checkpoints[str(seed)])
    except KeyError as exc:
        raise KeyError(f"Candidate {candidate['id']} has no selected checkpoint for seed {seed}") from exc


def load_run(candidate: dict[str, Any], seed: int) -> str:
    load_runs = candidate.get("load_runs", {})
    try:
        return str(load_runs[str(seed)])
    except KeyError as exc:
        raise KeyError(f"Candidate {candidate['id']} has no load_run for seed {seed}") from exc


def candidate_run_name(candidate: dict[str, Any], seed: int, suffix: str | None = None) -> str:
    template = candidate.get("run_name_template")
    if isinstance(template, str):
        base = template.format(seed=seed, id=candidate["id"])
    else:
        base = f"{candidate['id']}_random_stairs_stress_seed{seed}"
    return f"{base}_{suffix}" if suffix else base


def eval_summary_path(config: dict[str, Any], run_name: str) -> Path:
    return artifact_dir(config, run_name) / "checkpoint_sweep_summary.json"


def evaluate_complete(config: dict[str, Any], run_name: str) -> bool:
    return eval_summary_path(config, run_name).exists()


def selected_metrics_path(config: dict[str, Any], run_name: str) -> Path:
    return artifact_dir(config, run_name) / "metrics_selected.json"


def arg_or_sweep(args: argparse.Namespace, sweep_cfg: dict[str, Any], key: str) -> int:
    value = getattr(args, key)
    if value is not None:
        return int(value)
    return int(sweep_cfg[key])


def analysis_root(sweep_cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    root = Path(args.analysis_root or sweep_cfg["analysis_root"]).expanduser()
    if args.analysis_root is None and args.run_suffix:
        root = root.with_name(f"{root.name}_{args.run_suffix}")
    return root


def build_evaluate_command(
    *,
    sweep_cfg: dict[str, Any],
    candidate: dict[str, Any],
    config_path: Path,
    run_name: str,
    seed: int,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        f"--config={relative_to_repo(config_path)}",
        "--run-name",
        run_name,
        "--load-run",
        load_run(candidate, seed),
        "--checkpoints",
        str(selected_checkpoint(candidate, seed)),
        "--num-envs",
        str(arg_or_sweep(args, sweep_cfg, "eval_num_envs")),
        "--episodes",
        str(arg_or_sweep(args, sweep_cfg, "episodes")),
        "--seed",
        str(seed),
    ]
    if args.reuse_existing_metrics:
        command.append("--reuse-existing-metrics")
    if args.humanoid_gym_root:
        command.extend(["--humanoid-gym-root", args.humanoid_gym_root])
    if args.rl_device:
        command.extend(["--rl-device", args.rl_device])
    if args.sim_device:
        command.extend(["--sim-device", args.sim_device])
    return command


def aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    for key in METRIC_KEYS:
        values = [float(metrics[key]) for metrics in metrics_list if metrics.get(key) is not None]
        if not values:
            continue
        aggregate[f"{key}_mean"] = mean(values)
        aggregate[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return aggregate


def collect_candidate_summary(candidate: dict[str, Any], seeds: list[int], suffix: str | None = None) -> dict[str, Any]:
    config = load_config(resolve_config_path(candidate["config"]))
    per_seed: dict[str, Any] = {}
    metrics_list: list[dict[str, Any]] = []
    missing_seeds: list[int] = []
    selection_statuses: dict[str, str] = {}

    for seed in seeds:
        run_name = candidate_run_name(candidate, seed, suffix)
        summary_path = eval_summary_path(config, run_name)
        metrics_path = selected_metrics_path(config, run_name)
        if not summary_path.exists() or not metrics_path.exists():
            missing_seeds.append(seed)
            continue
        summary = read_json(summary_path)
        metrics = read_json(metrics_path)
        status = str(summary.get("selection_status", "selected"))
        selection_statuses[str(seed)] = status
        per_seed[str(seed)] = {
            "run_name": run_name,
            "load_run": load_run(candidate, seed),
            "selected_checkpoint": selected_checkpoint(candidate, seed),
            "checkpoint_sweep_summary_path": relative_to_repo(summary_path),
            "selected_metrics_path": relative_to_repo(metrics_path),
            "selection_status": status,
            "selected_metrics": metrics,
        }
        metrics_list.append(metrics)

    payload: dict[str, Any] = {
        "id": candidate["id"],
        "label": candidate["label"],
        "config_path": candidate["config"],
        "seeds": seeds,
        "missing_seeds": missing_seeds,
        "selection_statuses": selection_statuses,
        "per_seed": per_seed,
        "status": "complete" if not missing_seeds and len(metrics_list) == len(seeds) else "incomplete",
    }
    if metrics_list:
        payload["aggregate"] = aggregate_metrics(metrics_list)
    if metrics_list and selection_statuses and all(status != "selected" for status in selection_statuses.values()):
        payload["status"] = "collapsed"
    return payload


def compare_metric(sc_value: float, heuristic_value: float, *, higher_is_better: bool = False) -> str:
    if sc_value == heuristic_value:
        return "tie"
    if higher_is_better:
        return "sc_ppo_better" if sc_value > heuristic_value else "heuristic_better"
    return "sc_ppo_better" if sc_value < heuristic_value else "heuristic_better"


def build_interpretation(candidate_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {summary["id"]: summary for summary in candidate_summaries}
    sc = by_id.get("sc_ppo", {}).get("aggregate")
    heuristic = by_id.get("heuristic_smoothing", {}).get("aggregate")
    interpretation: dict[str, Any] = {
        "claim_boundary": (
            "Random stairs is a 复杂地形条件 pressure test of selected rough-terrain checkpoints; "
            "it does not rewrite the Isaac rough-terrain main claim."
        ),
        "status": "incomplete",
    }
    if not isinstance(sc, dict) or not isinstance(heuristic, dict):
        return interpretation

    metric_comparison: dict[str, Any] = {}
    for key in METRIC_KEYS:
        sc_key = f"{key}_mean"
        heuristic_key = f"{key}_mean"
        if sc_key not in sc or heuristic_key not in heuristic:
            continue
        higher_is_better = key == "episode_return_mean"
        metric_comparison[key] = {
            "sc_ppo": sc[sc_key],
            "heuristic": heuristic[heuristic_key],
            "ordering": compare_metric(sc[sc_key], heuristic[heuristic_key], higher_is_better=higher_is_better),
        }

    interpretation["status"] = "complete"
    interpretation["sc_ppo_vs_revised_heuristic"] = metric_comparison
    return interpretation


def write_summary(
    sweep_cfg: dict[str, Any],
    candidate_summaries: list[dict[str, Any]],
    seeds: list[int],
    args: argparse.Namespace,
) -> Path:
    output_root = ensure_directory(analysis_root(sweep_cfg, args).resolve())
    payload = {
        "comparison_name": sweep_cfg["name"],
        "scope": "复杂地形条件 pressure test",
        "seeds": seeds,
        "eval_num_envs": arg_or_sweep(args, sweep_cfg, "eval_num_envs"),
        "episodes": arg_or_sweep(args, sweep_cfg, "episodes"),
        "terrain_protocol": sweep_cfg.get("terrain_protocol", {}),
        "candidates": candidate_summaries,
        "interpretation": build_interpretation(candidate_summaries),
    }
    output_path = output_root / "comparison_summary.json"
    write_json(output_path, payload)
    return output_path


def selected_candidates(sweep_cfg: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    candidates = list(sweep_cfg["candidates"])
    if not requested:
        return candidates
    wanted = set(requested)
    filtered = [candidate for candidate in candidates if candidate["id"] in wanted]
    missing = wanted - {candidate["id"] for candidate in filtered}
    if missing:
        raise SystemExit(f"Unknown candidate ids: {sorted(missing)}")
    return filtered


def selected_seeds(sweep_cfg: dict[str, Any], requested: list[int] | None) -> list[int]:
    seeds = [int(seed) for seed in sweep_cfg["seeds"]]
    if not requested:
        return seeds
    wanted = set(int(seed) for seed in requested)
    missing = wanted - set(seeds)
    if missing:
        raise SystemExit(f"Unknown seeds for this sweep: {sorted(missing)}")
    return [seed for seed in seeds if seed in wanted]


def print_plan(sweep_cfg: dict[str, Any], candidates: list[dict[str, Any]], seeds: list[int], args: argparse.Namespace) -> None:
    print(f"Random-stairs stress test: {sweep_cfg['name']}")
    print("scope: evaluation-only selected-checkpoint pressure test")
    print(f"eval_num_envs: {arg_or_sweep(args, sweep_cfg, 'eval_num_envs')}")
    print(f"episodes: {arg_or_sweep(args, sweep_cfg, 'episodes')}")
    print(f"analysis summary: {relative_to_repo(analysis_root(sweep_cfg, args) / 'comparison_summary.json')}")
    for candidate in candidates:
        config_path = resolve_config_path(candidate["config"])
        config = load_config(config_path)
        print(f"[{candidate['id']}] {candidate['label']}")
        for seed in seeds:
            run_name = candidate_run_name(candidate, seed, args.run_suffix)
            status = "complete" if evaluate_complete(config, run_name) else "pending"
            command = build_evaluate_command(
                sweep_cfg=sweep_cfg,
                candidate=candidate,
                config_path=config_path,
                run_name=run_name,
                seed=seed,
                args=args,
            )
            print(f"  seed {seed} checkpoint {selected_checkpoint(candidate, seed)}: {status}")
            print("    command: " + " ".join(command))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate selected rough-terrain checkpoints on random stairs.")
    parser.add_argument("--sweep-config", default=None, help="Path to the random-stairs stress sweep JSON.")
    parser.add_argument("--candidate", action="append", default=None, help="Optional candidate id filter.")
    parser.add_argument("--seed", action="append", type=int, default=None, help="Optional training/evaluation seed filter.")
    parser.add_argument("--stage", choices=("plan", "evaluate", "summarize", "all"), default="plan")
    parser.add_argument("--skip-completed", action="store_true", help="Skip completed evaluation artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--reuse-existing-metrics", action="store_true", help="Reuse existing per-checkpoint metrics.")
    parser.add_argument("--run-suffix", default=None, help="Optional suffix to keep smoke artifacts separate.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Optional upstream checkout override.")
    parser.add_argument("--eval-num-envs", type=int, default=None, help="Override evaluation num_envs.")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count.")
    parser.add_argument("--rl-device", default=None, help="Optional RL device override.")
    parser.add_argument("--sim-device", default=None, help="Optional sim device override.")
    parser.add_argument("--analysis-root", default=None, help="Directory for comparison_summary.json.")
    args = parser.parse_args()

    sweep_cfg = load_sweep_config(args.sweep_config)
    candidates = selected_candidates(sweep_cfg, args.candidate)
    seeds = selected_seeds(sweep_cfg, args.seed)

    if args.stage == "plan":
        print_plan(sweep_cfg, candidates, seeds, args)
        return 0

    if args.stage in {"evaluate", "all"}:
        for candidate in candidates:
            config_path = resolve_config_path(candidate["config"])
            config = load_config(config_path)
            print(f"[{candidate['id']}] {candidate['label']}")
            for seed in seeds:
                run_name = candidate_run_name(candidate, seed, args.run_suffix)
                marker = eval_summary_path(config, run_name)
                if args.skip_completed and evaluate_complete(config, run_name):
                    print(f"Skipping evaluate: {relative_to_repo(marker)} already exists")
                    continue
                command = build_evaluate_command(
                    sweep_cfg=sweep_cfg,
                    candidate=candidate,
                    config_path=config_path,
                    run_name=run_name,
                    seed=seed,
                    args=args,
                )
                exit_code = run_command(command, cwd=REPO_ROOT, dry_run=args.dry_run)
                if exit_code != 0:
                    if evaluate_complete(config, run_name):
                        print(
                            f"evaluate exited with code {exit_code} after writing {relative_to_repo(marker)}; continuing.",
                            file=sys.stderr,
                        )
                        continue
                    return exit_code
            print()

    if args.stage in {"summarize", "all"} and not args.dry_run:
        candidate_summaries = [collect_candidate_summary(candidate, seeds, args.run_suffix) for candidate in candidates]
        summary_path = write_summary(sweep_cfg, candidate_summaries, seeds, args)
        print(f"Wrote {relative_to_repo(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
