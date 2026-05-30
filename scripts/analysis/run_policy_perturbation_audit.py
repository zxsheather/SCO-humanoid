#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from _common import (  # noqa: E402
    configure_runtime_env,
    ensure_directory,
    ensure_humanoid_gym_checkout,
    ensure_upstream_on_syspath,
    load_config,
    read_json,
    relative_to_repo,
    repo_root,
    resolve_humanoid_gym_root,
    write_json,
)
from _overrides import apply_method_overrides  # noqa: E402


def load_sweep(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root() / candidate
    return read_json(candidate)


def method_entry(sweep: dict[str, Any], method_id: str) -> dict[str, Any]:
    for method in sweep["methods"]:
        if method["id"] == method_id:
            return method
    raise KeyError(f"Unknown method id: {method_id}")


def seed_run_name(config: dict[str, Any], seed: int) -> str:
    return f"{config['run_name']}_seed{seed}"


def seed_artifact_dir(config: dict[str, Any], seed: int) -> Path:
    return repo_root() / config["artifacts_root"] / seed_run_name(config, seed)


def checkpoint_summary(config: dict[str, Any], seed: int) -> dict[str, Any]:
    path = seed_artifact_dir(config, seed) / "checkpoint_sweep_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint sweep summary: {path}")
    return read_json(path)


def selected_checkpoint(summary: dict[str, Any]) -> int:
    if "best_checkpoint" not in summary:
        raise KeyError("checkpoint_sweep_summary.json is missing best_checkpoint")
    return int(summary["best_checkpoint"])


def selected_run_dir(summary: dict[str, Any]) -> Path:
    latest = summary.get("latest_checkpoint_path")
    if not isinstance(latest, str):
        raise KeyError("checkpoint_sweep_summary.json is missing latest_checkpoint_path")
    path = Path(latest)
    if not path.is_absolute():
        path = repo_root() / path
    return path.parent.resolve()


def selected_checkpoint_path(summary: dict[str, Any]) -> Path:
    run_dir = selected_run_dir(summary)
    checkpoint = selected_checkpoint(summary)
    path = run_dir / f"model_{checkpoint}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected checkpoint: {path}")
    return path


def selected_metrics(config: dict[str, Any], seed: int) -> dict[str, Any]:
    path = seed_artifact_dir(config, seed) / "metrics_selected.json"
    if path.exists():
        payload = read_json(path)
        payload["_path"] = relative_to_repo(path)
        return payload
    summary = checkpoint_summary(config, seed)
    metrics_path = summary.get("selected_checkpoint_metrics_path")
    if not isinstance(metrics_path, str):
        return {"_path": None}
    path = Path(metrics_path)
    if not path.is_absolute():
        path = repo_root() / path
    payload = read_json(path)
    payload["_path"] = relative_to_repo(path)
    return payload


def build_upstream_args(get_args, config: dict[str, Any], run_name: str, rl_device: str, sim_device: str, seed: int):
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "policy_perturbation_audit.py",
            f"--task={config['task']}",
            f"--experiment_name={config['experiment_name']}",
            f"--run_name={run_name}",
            f"--rl_device={rl_device}",
            f"--sim_device={sim_device}",
            f"--seed={seed}",
            "--headless",
        ]
        return get_args()
    finally:
        sys.argv = original_argv


def infer_actor_critic_kwargs(state_dict: dict[str, Any]) -> dict[str, Any]:
    actor_weights = [
        (int(key.split(".")[1]), value)
        for key, value in state_dict.items()
        if key.startswith("actor.") and key.endswith(".weight") and key.split(".")[1].isdigit()
    ]
    critic_weights = [
        (int(key.split(".")[1]), value)
        for key, value in state_dict.items()
        if key.startswith("critic.") and key.endswith(".weight") and key.split(".")[1].isdigit()
    ]
    if not actor_weights or not critic_weights:
        raise ValueError("Cannot infer ActorCritic dimensions from checkpoint state dict")

    actor_shapes = [tensor.shape for _, tensor in sorted(actor_weights)]
    critic_shapes = [tensor.shape for _, tensor in sorted(critic_weights)]
    return {
        "num_actor_obs": int(actor_shapes[0][1]),
        "num_critic_obs": int(critic_shapes[0][1]),
        "num_actions": int(actor_shapes[-1][0]),
        "actor_hidden_dims": [int(shape[0]) for shape in actor_shapes[:-1]],
        "critic_hidden_dims": [int(shape[0]) for shape in critic_shapes[:-1]],
    }


def load_actor(checkpoint_path: Path, humanoid_gym_root: Path, device: str):
    import torch
    import torch.nn as nn

    ensure_upstream_on_syspath(humanoid_gym_root)
    from humanoid.algo.ppo.actor_critic import ActorCritic

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = loaded["model_state_dict"]
    kwargs = infer_actor_critic_kwargs(state_dict)
    actor_critic = ActorCritic(
        kwargs["num_actor_obs"],
        kwargs["num_critic_obs"],
        kwargs["num_actions"],
        actor_hidden_dims=kwargs["actor_hidden_dims"],
        critic_hidden_dims=kwargs["critic_hidden_dims"],
        activation=nn.ELU(),
    )
    incompatible = actor_critic.load_state_dict(state_dict, strict=False)
    allowed_missing = {"action_output_scale", "action_std_scale"}
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected or missing:
        raise RuntimeError(
            f"Unexpected checkpoint compatibility issue: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    actor_critic.to(device)
    actor_critic.eval()
    return actor_critic


def collect_observation_bank(args: argparse.Namespace) -> int:
    import importlib

    sweep = load_sweep(args.sweep_config)
    method = method_entry(sweep, args.method)
    config = load_config(method["config"])
    summary = checkpoint_summary(config, args.seed)
    checkpoint_path = selected_checkpoint_path(summary)
    run_dir = selected_run_dir(summary)

    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    ensure_humanoid_gym_checkout(humanoid_gym_root)
    configure_runtime_env()
    ensure_upstream_on_syspath(humanoid_gym_root)

    importlib.import_module("humanoid.envs")
    import torch
    from humanoid.utils import get_args, task_registry

    bank_cfg = sweep["observation_bank"]
    eval_cfg = config["evaluation"]
    rl_device = args.rl_device or eval_cfg["rl_device"]
    sim_device = args.sim_device or eval_cfg["sim_device"]
    run_name = seed_run_name(config, args.seed)
    upstream_args = build_upstream_args(get_args, config, run_name, rl_device, sim_device, int(eval_cfg.get("seed", 123145)))
    upstream_args.num_envs = int(args.num_envs or bank_cfg["num_envs"])

    env_cfg, train_cfg = task_registry.get_cfgs(name=config["task"])
    env_cfg, train_cfg = apply_method_overrides(env_cfg, train_cfg, config)
    env_cfg.env.num_envs = upstream_args.num_envs
    env_cfg.terrain.curriculum = False
    env, _ = task_registry.make_env(name=config["task"], args=upstream_args, env_cfg=env_cfg)

    actor_critic = load_actor(checkpoint_path, humanoid_gym_root, device=env.device)
    observations = []
    obs = env.get_observations()
    max_observations = int(args.max_observations_per_seed or bank_cfg["max_observations_per_seed"])
    sample_every = max(int(args.sample_every or bank_cfg["sample_every"]), 1)
    steps = int(args.steps or bank_cfg["steps"])

    for step in range(steps):
        if step % sample_every == 0 and sum(chunk.shape[0] for chunk in observations) < max_observations:
            remaining = max_observations - sum(chunk.shape[0] for chunk in observations)
            observations.append(obs[:remaining].detach().cpu())
        with torch.inference_mode():
            actions = actor_critic.act_inference(obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())

    if not observations:
        raise RuntimeError("No observations collected")

    output_dir = repo_root() / sweep["analysis_root"] / "observation_banks"
    ensure_directory(output_dir)
    output_path = output_dir / f"{args.method}_seed{args.seed}_obs.pt"
    payload = {
        "observations": torch.cat(observations, dim=0)[:max_observations],
        "source": {
            "method_id": args.method,
            "method_label": method["label"],
            "seed": args.seed,
            "checkpoint": selected_checkpoint(summary),
            "checkpoint_path": relative_to_repo(checkpoint_path),
            "run_dir": relative_to_repo(run_dir),
            "steps": steps,
            "sample_every": sample_every,
        },
    }
    torch.save(payload, output_path)
    print(f"Wrote {relative_to_repo(output_path)}")
    return 0


def tensor_stats(values) -> dict[str, Any]:
    import torch

    if values.numel() == 0:
        return {"count": 0}
    values = values.detach().float().cpu()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def compute_local_sensitivity(actor_critic, observations, sample_count: int):
    import torch

    sample = observations[: min(sample_count, observations.shape[0])].detach().clone().requires_grad_(True)
    action_mean = actor_critic.act_inference(sample)
    squared_norm = torch.zeros(sample.shape[0], device=sample.device)
    for action_idx in range(action_mean.shape[1]):
        grad_outputs = torch.zeros_like(action_mean)
        grad_outputs[:, action_idx] = 1.0
        grads = torch.autograd.grad(
            outputs=action_mean,
            inputs=sample,
            grad_outputs=grad_outputs,
            retain_graph=action_idx + 1 < action_mean.shape[1],
            create_graph=False,
            allow_unused=False,
        )[0]
        squared_norm += torch.sum(torch.square(grads), dim=1)
    return torch.sqrt(torch.clamp(squared_norm, min=0.0)).detach()


def load_observation_bank(sweep: dict[str, Any]):
    import torch

    bank_dir = repo_root() / sweep["analysis_root"] / "observation_banks"
    chunks = []
    sources = []
    for path in sorted(bank_dir.glob("*_obs.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        chunks.append(payload["observations"])
        sources.append(payload["source"])
    if not chunks:
        raise FileNotFoundError(f"No observation banks found in {bank_dir}")
    return torch.cat(chunks, dim=0), sources


def score_policy(args: argparse.Namespace) -> int:
    import torch

    sweep = load_sweep(args.sweep_config)
    method = method_entry(sweep, args.method)
    config = load_config(method["config"])
    summary = checkpoint_summary(config, args.seed)
    checkpoint_path = selected_checkpoint_path(summary)
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    configure_runtime_env()

    perturb_cfg = sweep["perturbation"]
    device = args.device or "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    observations, bank_sources = load_observation_bank(sweep)
    max_observations = int(args.max_score_observations or perturb_cfg["max_observations"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(perturb_cfg["seed"]))
    if observations.shape[0] > max_observations:
        indices = torch.randperm(observations.shape[0], generator=generator)[:max_observations]
        observations = observations.index_select(0, indices)
    observations = observations.to(device)

    actor_critic = load_actor(checkpoint_path, humanoid_gym_root, device=device)
    with torch.inference_mode():
        base_actions = actor_critic.act_inference(observations)

    by_epsilon: dict[str, Any] = {}
    for epsilon in perturb_cfg["epsilons"]:
        epsilon = float(epsilon)
        amplifications = []
        action_deltas = []
        for direction_idx in range(int(perturb_cfg["directions_per_observation"])):
            direction_generator = torch.Generator(device=device)
            direction_generator.manual_seed(int(perturb_cfg["seed"]) + 1009 * direction_idx + 17 * args.seed)
            direction = torch.randn(observations.shape, device=device, generator=direction_generator)
            direction_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-12)
            delta_obs = epsilon * direction / direction_norm
            with torch.inference_mode():
                perturbed_actions = actor_critic.act_inference(observations + delta_obs)
            delta_action = torch.linalg.vector_norm(perturbed_actions - base_actions, dim=1)
            amplifications.append(delta_action / epsilon)
            action_deltas.append(delta_action)
        by_epsilon[str(epsilon)] = {
            "epsilon": epsilon,
            "amplification": tensor_stats(torch.cat(amplifications)),
            "action_delta_l2": tensor_stats(torch.cat(action_deltas)),
        }

    sensitivity = compute_local_sensitivity(
        actor_critic,
        observations,
        sample_count=int(args.jacobian_sample_count or perturb_cfg["jacobian_sample_count"]),
    )
    metrics = selected_metrics(config, args.seed)
    output_dir = repo_root() / sweep["analysis_root"] / "per_policy"
    ensure_directory(output_dir)
    output_path = output_dir / f"{args.method}_seed{args.seed}.json"
    payload = {
        "issue": sweep.get("issue"),
        "method_id": args.method,
        "method_label": method["label"],
        "seed": args.seed,
        "checkpoint": selected_checkpoint(summary),
        "checkpoint_path": relative_to_repo(checkpoint_path),
        "observation_bank_sources": bank_sources,
        "observation_count": int(observations.shape[0]),
        "perturbation": {
            "epsilons": perturb_cfg["epsilons"],
            "primary_epsilon": perturb_cfg["primary_epsilon"],
            "directions_per_observation": perturb_cfg["directions_per_observation"],
        },
        "by_epsilon": by_epsilon,
        "local_sensitivity_on_bank": tensor_stats(sensitivity),
        "selected_metrics": {
            "metrics_path": metrics.get("_path"),
            "fall_rate": metrics.get("fall_rate"),
            "velocity_tracking_error_mean": metrics.get("velocity_tracking_error_mean"),
            "joint_acceleration_l2_mean": metrics.get("joint_acceleration_l2_mean"),
            "action_jitter_l2_mean": metrics.get("action_jitter_l2_mean"),
            "episode_return_mean": metrics.get("episode_return_mean"),
            "policy_local_sensitivity_cost_mean": (
                metrics.get("constraint_metrics", {}).get("policy_local_sensitivity_cost_mean")
                if isinstance(metrics.get("constraint_metrics"), dict)
                else metrics.get("policy_local_sensitivity_cost_mean")
            ),
        },
        "claim_boundary": sweep["claim_boundary"],
    }
    write_json(output_path, payload)
    print(f"Wrote {relative_to_repo(output_path)}")
    return 0


def mean_std(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_vals, y_vals = zip(*pairs)
    mx = statistics.fmean(x_vals)
    my = statistics.fmean(y_vals)
    dx = [x - mx for x in x_vals]
    dy = [y - my for y in y_vals]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def summarize_results(args: argparse.Namespace) -> int:
    sweep = load_sweep(args.sweep_config)
    primary_epsilon = str(float(sweep["perturbation"]["primary_epsilon"]))
    rows = []
    for path in sorted((repo_root() / sweep["analysis_root"] / "per_policy").glob("*.json")):
        payload = read_json(path)
        primary = payload["by_epsilon"][primary_epsilon]["amplification"]
        selected = payload["selected_metrics"]
        rows.append(
            {
                "method_id": payload["method_id"],
                "method_label": payload["method_label"],
                "seed": payload["seed"],
                "checkpoint": payload["checkpoint"],
                "amplification_mean": primary.get("mean"),
                "amplification_p90": primary.get("p90"),
                "amplification_p95": primary.get("p95"),
                "bank_local_sensitivity_mean": payload["local_sensitivity_on_bank"].get("mean"),
                "selected_policy_sensitivity": selected.get("policy_local_sensitivity_cost_mean"),
                "selected_action_jitter": selected.get("action_jitter_l2_mean"),
                "selected_joint_acceleration": selected.get("joint_acceleration_l2_mean"),
                "selected_fall_rate": selected.get("fall_rate"),
                "selected_return": selected.get("episode_return_mean"),
                "source_path": relative_to_repo(path),
            }
        )

    if not rows:
        raise FileNotFoundError("No per-policy perturbation results found")

    aggregates = []
    for method in sweep["methods"]:
        method_rows = [row for row in rows if row["method_id"] == method["id"]]
        aggregates.append(
            {
                "method_id": method["id"],
                "method_label": method["label"],
                "seed_count": len(method_rows),
                "amplification_mean": mean_std([row["amplification_mean"] for row in method_rows]),
                "amplification_p90": mean_std([row["amplification_p90"] for row in method_rows]),
                "bank_local_sensitivity_mean": mean_std(
                    [row["bank_local_sensitivity_mean"] for row in method_rows]
                ),
                "selected_policy_sensitivity": mean_std(
                    [row["selected_policy_sensitivity"] for row in method_rows]
                ),
                "selected_action_jitter": mean_std([row["selected_action_jitter"] for row in method_rows]),
                "selected_joint_acceleration": mean_std(
                    [row["selected_joint_acceleration"] for row in method_rows]
                ),
                "selected_fall_rate": mean_std([row["selected_fall_rate"] for row in method_rows]),
            }
        )

    correlations = {
        "amplification_mean_vs_selected_policy_sensitivity": pearson(
            [row["amplification_mean"] for row in rows],
            [row["selected_policy_sensitivity"] for row in rows],
        ),
        "amplification_mean_vs_selected_action_jitter": pearson(
            [row["amplification_mean"] for row in rows],
            [row["selected_action_jitter"] for row in rows],
        ),
        "amplification_mean_vs_selected_joint_acceleration": pearson(
            [row["amplification_mean"] for row in rows],
            [row["selected_joint_acceleration"] for row in rows],
        ),
    }

    output_dir = repo_root() / sweep["analysis_root"]
    ensure_directory(output_dir)
    summary_path = output_dir / "summary.json"
    table_path = output_dir / "table_policy_perturbation_audit.md"
    write_json(
        summary_path,
        {
            "issue": sweep.get("issue"),
            "name": sweep["name"],
            "primary_epsilon": float(primary_epsilon),
            "claim_boundary": sweep["claim_boundary"],
            "rows": rows,
            "aggregates": aggregates,
            "correlations": correlations,
        },
    )
    lines = [
        "| Method | Seeds | Amplification mean | Amplification p90 | Bank sensitivity | Selected sensitivity | Selected jitter | Selected jnt acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {label} | {n} | {amp:.3f} | {p90:.3f} | {bank:.3f} | {sens:.3f} | {jitter:.3f} | {jacc:.3f} |".format(
                label=row["method_label"],
                n=row["seed_count"],
                amp=row["amplification_mean"]["mean"],
                p90=row["amplification_p90"]["mean"],
                bank=row["bank_local_sensitivity_mean"]["mean"],
                sens=row["selected_policy_sensitivity"]["mean"],
                jitter=row["selected_action_jitter"]["mean"],
                jacc=row["selected_joint_acceleration"]["mean"],
            )
        )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {relative_to_repo(summary_path)}")
    print(f"Wrote {relative_to_repo(table_path)}")
    return 0


def recoverable_run(command: list[str], marker: Path) -> None:
    completed = subprocess.run(command, cwd=repo_root(), check=False)
    if completed.returncode == 0:
        return
    if marker.exists():
        print(f"Recovered non-zero Isaac exit {completed.returncode}; found {relative_to_repo(marker)}")
        return
    raise SystemExit(completed.returncode)


def run_all(args: argparse.Namespace) -> int:
    sweep = load_sweep(args.sweep_config)
    source_methods = set(sweep["observation_bank"]["source_method_ids"])
    python_bin = sys.executable
    for method in sweep["methods"]:
        if method["id"] not in source_methods:
            continue
        for seed in sweep["seeds"]:
            marker = (
                repo_root()
                / sweep["analysis_root"]
                / "observation_banks"
                / f"{method['id']}_seed{seed}_obs.pt"
            )
            if marker.exists() and not args.force:
                print(f"Skipping existing bank {relative_to_repo(marker)}")
                continue
            command = [
                python_bin,
                str(Path(__file__).resolve()),
                "collect",
                "--sweep-config",
                args.sweep_config,
                "--method",
                method["id"],
                "--seed",
                str(seed),
            ]
            if args.humanoid_gym_root:
                command.extend(["--humanoid-gym-root", args.humanoid_gym_root])
            if args.rl_device:
                command.extend(["--rl-device", args.rl_device])
            if args.sim_device:
                command.extend(["--sim-device", args.sim_device])
            recoverable_run(command, marker)

    for method in sweep["methods"]:
        for seed in sweep["seeds"]:
            marker = repo_root() / sweep["analysis_root"] / "per_policy" / f"{method['id']}_seed{seed}.json"
            if marker.exists() and not args.force:
                print(f"Skipping existing score {relative_to_repo(marker)}")
                continue
            command = [
                python_bin,
                str(Path(__file__).resolve()),
                "score",
                "--sweep-config",
                args.sweep_config,
                "--method",
                method["id"],
                "--seed",
                str(seed),
            ]
            if args.humanoid_gym_root:
                command.extend(["--humanoid-gym-root", args.humanoid_gym_root])
            if args.device:
                command.extend(["--device", args.device])
            recoverable_run(command, marker)

    return summarize_results(args)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a policy perturbation audit on existing selected checkpoints.")
    parser.add_argument("stage", choices=["collect", "score", "summarize", "all"])
    parser.add_argument("--sweep-config", default="configs/sweeps/policy_perturbation_audit.json")
    parser.add_argument("--method", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--humanoid-gym-root", default=None)
    parser.add_argument("--rl-device", default=None)
    parser.add_argument("--sim-device", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=None)
    parser.add_argument("--max-observations-per-seed", type=int, default=None)
    parser.add_argument("--max-score-observations", type=int, default=None)
    parser.add_argument("--jacobian-sample-count", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stage in {"collect", "score"} and (args.method is None or args.seed is None):
        parser.error("collect and score require --method and --seed")
    if args.stage == "collect":
        return collect_observation_bank(args)
    if args.stage == "score":
        return score_policy(args)
    if args.stage == "summarize":
        return summarize_results(args)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
