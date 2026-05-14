from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent

from _common import apply_overrides_object, config_overrides  # noqa: E402


def apply_method_overrides(env_cfg: Any, train_cfg: Any, config: dict[str, Any]) -> tuple[Any, Any]:
    env_cfg = deepcopy(env_cfg)
    train_cfg = deepcopy(train_cfg)
    overrides = config_overrides(config)
    env_overrides = overrides.get("env", {})
    train_overrides = overrides.get("train", {})

    if env_overrides:
        env_cfg = apply_overrides_object(env_cfg, env_overrides)
    if train_overrides:
        train_cfg = apply_overrides_object(train_cfg, train_overrides)
    return env_cfg, train_cfg
