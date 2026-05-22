from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]
HUMANOID_GYM_ROOT = REPO_ROOT / ".external" / "humanoid-gym"
HUMANOID_ROOT = HUMANOID_GYM_ROOT / "humanoid"
ALGO_ROOT = HUMANOID_ROOT / "algo"
PPO_ROOT = ALGO_ROOT / "ppo"


def _ensure_package(module_name: str, package_path: Path):
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__path__ = [str(package_path)]
        sys.modules[module_name] = module
    return module


def _load_module(module_name: str, module_path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_sc_ppo_module():
    humanoid_root_str = str(HUMANOID_GYM_ROOT)
    if humanoid_root_str not in sys.path:
        sys.path.insert(0, humanoid_root_str)

    _ensure_package("humanoid", HUMANOID_ROOT)
    _ensure_package("humanoid.algo", ALGO_ROOT)
    _ensure_package("humanoid.algo.ppo", PPO_ROOT)
    _load_module("humanoid.algo.ppo.actor_critic", PPO_ROOT / "actor_critic.py")
    _load_module("humanoid.algo.ppo.rollout_storage", PPO_ROOT / "rollout_storage.py")
    _load_module("humanoid.algo.ppo.ppo", PPO_ROOT / "ppo.py")
    return _load_module("humanoid.algo.ppo.sc_ppo", PPO_ROOT / "sc_ppo.py")


def load_actor_critic_module():
    humanoid_root_str = str(HUMANOID_GYM_ROOT)
    if humanoid_root_str not in sys.path:
        sys.path.insert(0, humanoid_root_str)

    _ensure_package("humanoid", HUMANOID_ROOT)
    _ensure_package("humanoid.algo", ALGO_ROOT)
    _ensure_package("humanoid.algo.ppo", PPO_ROOT)
    return _load_module("humanoid.algo.ppo.actor_critic", PPO_ROOT / "actor_critic.py")
