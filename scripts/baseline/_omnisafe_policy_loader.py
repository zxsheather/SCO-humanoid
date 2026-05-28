from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor


DEFAULT_ACTIVATION = "tanh"
DEFAULT_HIDDEN_SIZES = [64, 64]


class OmniSafeDeterministicPolicy:
    """Deterministic evaluation adapter for OmniSafe Gaussian actors."""

    def __init__(self, actor: GaussianLearningActor) -> None:
        self.actor = actor

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.predict(obs, deterministic=True)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        distribution = self.actor(obs)
        return distribution.mean


def create_gaussian_actor(
    *,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: list[int] | None = None,
    activation: str = DEFAULT_ACTIVATION,
    device: str | torch.device = "cpu",
) -> GaussianLearningActor:
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(int(obs_dim),), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(int(act_dim),), dtype=np.float32)
    actor = GaussianLearningActor(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=list(hidden_sizes or DEFAULT_HIDDEN_SIZES),
        activation=activation,
    )
    return actor.to(device)


def save_omnisafe_policy_checkpoint(
    path: str | Path,
    actor: GaussianLearningActor,
    *,
    checkpoint: int,
    seed: int,
    cost_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "sco_omnisafe_gaussian_actor_v1",
        "checkpoint": int(checkpoint),
        "seed": int(seed),
        "actor": {
            "obs_dim": int(actor._obs_dim),
            "act_dim": int(actor._act_dim),
            "hidden_sizes": list(actor._hidden_sizes),
            "activation": str(actor._activation),
        },
        "cost_config": cost_config or {},
        "metadata": metadata or {},
        "actor_state_dict": actor.state_dict(),
    }
    torch.save(payload, output_path)
    return output_path


def load_omnisafe_policy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[OmniSafeDeterministicPolicy, dict[str, Any]]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_cfg = payload.get("actor")
    if not isinstance(actor_cfg, dict):
        raise ValueError(f"Missing actor metadata in OmniSafe checkpoint: {checkpoint_path}")
    actor = create_gaussian_actor(
        obs_dim=int(actor_cfg["obs_dim"]),
        act_dim=int(actor_cfg["act_dim"]),
        hidden_sizes=[int(value) for value in actor_cfg.get("hidden_sizes", DEFAULT_HIDDEN_SIZES)],
        activation=str(actor_cfg.get("activation", DEFAULT_ACTIVATION)),
        device=device,
    )
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    metadata = {
        "format": payload.get("format"),
        "checkpoint": payload.get("checkpoint"),
        "seed": payload.get("seed"),
        "actor": actor_cfg,
        "cost_config": payload.get("cost_config", {}),
        "metadata": payload.get("metadata", {}),
        "checkpoint_path": str(checkpoint_path),
    }
    return OmniSafeDeterministicPolicy(actor), metadata
