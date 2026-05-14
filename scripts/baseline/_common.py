from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baselines" / "vanilla_ppo.json"
DEFAULT_TORCH_EXTENSIONS_DIR = Path("/tmp/torch_extensions")
DEFAULT_MPLCONFIGDIR = Path("/tmp/matplotlib")
DEFAULT_XDG_CACHE_HOME = Path("/tmp/xdg-cache")


class BaselineError(RuntimeError):
    pass


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def repo_root() -> Path:
    return REPO_ROOT


def resolve_humanoid_gym_root(config: dict[str, Any], override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env_override = os.environ.get("HUMANOID_GYM_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    checkout_dir = config["upstream"]["checkout_dir"]
    return (REPO_ROOT / checkout_dir).resolve()


def ensure_humanoid_gym_checkout(root: Path) -> None:
    expected = [
        root / "README.md",
        root / "humanoid" / "scripts" / "train.py",
        root / "humanoid" / "scripts" / "play.py",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise BaselineError(
            "Humanoid-Gym checkout is incomplete or missing. "
            f"Expected files not found: {missing}"
        )


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifacts_root(config: dict[str, Any]) -> Path:
    return (REPO_ROOT / config["artifacts_root"]).resolve()


def artifact_dir(config: dict[str, Any], run_name: str | None = None) -> Path:
    return artifacts_root(config) / (run_name or config["run_name"])


def experiment_root(humanoid_gym_root: Path, experiment_name: str) -> Path:
    return humanoid_gym_root / "logs" / experiment_name


def run_dirs_for_name(log_root: Path, run_name: str) -> list[Path]:
    if not log_root.exists():
        return []
    suffix = f"_{run_name}"
    return sorted(
        [path for path in log_root.iterdir() if path.is_dir() and path.name.endswith(suffix)],
        key=lambda path: path.stat().st_mtime,
    )


def resolve_run_dir(
    humanoid_gym_root: Path,
    config: dict[str, Any],
    run_name: str | None = None,
    load_run: str | None = None,
) -> Path:
    experiment_name = config["experiment_name"]
    log_root = experiment_root(humanoid_gym_root, experiment_name)
    if load_run:
        explicit = log_root / load_run
        if not explicit.exists():
            raise BaselineError(f"Requested load_run does not exist: {explicit}")
        return explicit

    candidates = run_dirs_for_name(log_root, run_name or config["run_name"])
    if not candidates:
        raise BaselineError(f"No run directories found in {log_root}")
    return candidates[-1]


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(
        run_dir.glob("model_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not checkpoints:
        raise BaselineError(f"No checkpoints found in {run_dir}")
    return checkpoints[-1]


def ensure_upstream_on_syspath(humanoid_gym_root: Path) -> None:
    root_str = str(humanoid_gym_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def build_upstream_command(
    humanoid_gym_root: Path,
    script_name: str,
    *,
    task: str,
    experiment_name: str,
    run_name: str,
    rl_device: str,
    sim_device: str,
    headless: bool,
    num_envs: int | None = None,
    seed: int | None = None,
    max_iterations: int | None = None,
    resume: bool = False,
    load_run: str | None = None,
    checkpoint: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(humanoid_gym_root / "humanoid" / "scripts" / script_name),
        f"--task={task}",
        f"--experiment_name={experiment_name}",
        f"--run_name={run_name}",
        f"--rl_device={rl_device}",
        f"--sim_device={sim_device}",
    ]
    if headless:
        command.append("--headless")
    if num_envs is not None:
        command.append(f"--num_envs={num_envs}")
    if seed is not None:
        command.append(f"--seed={seed}")
    if max_iterations is not None:
        command.append(f"--max_iterations={max_iterations}")
    if resume:
        command.append("--resume")
    if load_run is not None:
        command.append(f"--load_run={load_run}")
    if checkpoint is not None:
        command.append(f"--checkpoint={checkpoint}")
    return command


def run_command(command: list[str], cwd: Path, dry_run: bool = False) -> int:
    rendered = " ".join(command)
    print(rendered)
    if dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("TORCH_EXTENSIONS_DIR", str(DEFAULT_TORCH_EXTENSIONS_DIR))
    env.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))
    env.setdefault("XDG_CACHE_HOME", str(DEFAULT_XDG_CACHE_HOME))
    env.setdefault("WANDB_MODE", "disabled")
    DEFAULT_TORCH_EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    return completed.returncode


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def default_manifest(config: dict[str, Any], humanoid_gym_root: Path) -> dict[str, Any]:
    return {
        "baseline": config["name"],
        "task": config["task"],
        "experiment_name": config["experiment_name"],
        "run_name": config["run_name"],
        "upstream": {
            "repo_url": config["upstream"]["repo_url"],
            "ref": config["upstream"]["ref"],
            "checkout_dir": relative_to_repo(humanoid_gym_root),
        },
    }
