#!/usr/bin/env python3
"""
OmniSafe cost bridge smoke test (#61).

Loads a freshly-initialized ActorCritic from the rough-terrain task config,
computes Jacobian cost on a dummy observation batch, runs one OmniSafe Lagrange
multiplier update, and writes a smoke artifact.

This is a non-training diagnostic. It verifies:
- OmniSafe import and Lagrange instantiation.
- Jacobian cost computation (faithful to SC-PPO definition).
- Cost bridge: OmniSafe multiplier updates from Jacobian cost.
- Finite, non-NaN numerical values throughout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# ── repo path setup ──────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, os.path.join(_REPO_ROOT, ".external", "humanoid-gym"))

# CRITICAL: isaacgym must be imported before torch (including torch
# imported transitively through omnisafe or humanoid-gym modules).
import isaacgym          # noqa: F401 — side-effect: registers gym_38 bindings
import torch             # noqa: F401 — now safe after isaacgym

from humanoid.algo.ppo.actor_critic import ActorCritic

from scripts.baseline._common import load_config, artifact_dir
from scripts.baseline._omnisafe_bridge import run_bridge_smoke


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OmniSafe cost bridge smoke test (#61)")
    p.add_argument("--config", required=True, help="Path to method config JSON")
    p.add_argument("--artifacts-root", default=None, help="Override artifacts root dir")
    p.add_argument("--obs-dim", type=int, default=705, help="Observation dimension")
    p.add_argument("--act-dim", type=int, default=12, help="Action dimension")
    p.add_argument("--batch-size", type=int, default=256, help="Dummy obs batch size")
    p.add_argument("--threshold", type=float, default=3.8, help="Cost threshold")
    p.add_argument("--multiplier-init", type=float, default=0.5)
    p.add_argument("--lambda-lr", type=float, default=0.01)
    p.add_argument("--lambda-optimizer", type=str, default="SGD")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config for provenance only
    method_cfg = load_config(args.config)
    run_name = method_cfg.get("run_name", "omnisafe_cost_bridge_smoke")
    artifacts_root = str(artifact_dir(method_cfg, run_name))
    os.makedirs(artifacts_root, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    result: dict = {
        "run_name": run_name,
        "timestamp": timestamp,
        "config_path": args.config,
        "artifacts_root": artifacts_root,
        "issue": "#61",
        "status": "running",
    }

    # ── 1. Create dummy ActorCritic ─────────────────────────────
    actor_critic = ActorCritic(
        num_actor_obs=args.obs_dim,
        num_critic_obs=args.obs_dim,
        num_actions=args.act_dim,
    )
    result["actor_critic"] = {
        "obs_dim": args.obs_dim,
        "act_dim": args.act_dim,
        "num_params": sum(p.numel() for p in actor_critic.parameters()),
    }

    # ── 2. Create dummy observation batch ───────────────────────
    obs_batch = torch.randn(args.batch_size, args.obs_dim)
    result["obs_batch"] = {"shape": list(obs_batch.shape), "dtype": str(obs_batch.dtype)}

    # ── 3. Run bridge smoke ────────────────────────────────────
    bridge_result = run_bridge_smoke(
        actor_critic,
        obs_batch,
        threshold=args.threshold,
        multiplier_init=args.multiplier_init,
        lambda_lr=args.lambda_lr,
        lambda_optimizer=args.lambda_optimizer,
    )

    # Validate numerical sanity
    cost = bridge_result["cost"]
    upd = bridge_result["update"]
    assert cost["cost_for_update"] > 0, f"Expected positive Jacobian cost, got {cost['cost_for_update']}"
    assert np.isfinite(cost["cost_for_update"]), "Cost is non-finite"
    assert np.isfinite(upd["multiplier"]), "Multiplier is non-finite"
    assert upd["multiplier"] >= 0, f"Multiplier should be >= 0, got {upd['multiplier']}"

    result["bridge"] = bridge_result
    result["status"] = "complete"
    result["summary"] = (
        f"Jacobian cost={cost['cost_for_update']:.4f} (threshold={args.threshold}), "
        f"multiplier: {upd['multiplier_before']:.4f} → {upd['multiplier']:.4f}, "
        f"constraint_error={upd['constraint_error']:.4f}"
    )

    # ── 4. Write artifact ──────────────────────────────────────
    artifact_path = os.path.join(artifacts_root, "omnisafe_cost_bridge_smoke.json")
    with open(artifact_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"SMOKE OK — artifact: {artifact_path}")
    print(result["summary"])


if __name__ == "__main__":
    main()
