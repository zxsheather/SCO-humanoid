#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    BaselineError,
    ensure_humanoid_gym_checkout,
    ensure_upstream_on_syspath,
    load_config,
    relative_to_repo,
    resolve_humanoid_gym_root,
)


def probe_module(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, str(exc)
    version = getattr(module, "__version__", "imported")
    return True, str(version)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the issue #1 Humanoid-Gym baseline environment.")
    parser.add_argument("--config", default=None, help="Path to the baseline config JSON.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    args = parser.parse_args()

    config = load_config(args.config)
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)

    print(f"Python: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"Humanoid-Gym root: {relative_to_repo(humanoid_gym_root)}")
    print(f"Expected upstream ref: {config['upstream']['ref']}")
    python_ok = sys.version_info[:2] == (3, 8)
    print(f"python_3_8: {'ok' if python_ok else 'mismatch'}")

    try:
        ensure_humanoid_gym_checkout(humanoid_gym_root)
        print("checkout: ok")
    except BaselineError as exc:
        print(f"checkout: missing ({exc})")
        return 1

    ensure_upstream_on_syspath(humanoid_gym_root)

    missing_modules: list[str] = []
    # Isaac Gym must be imported before torch in the same process.
    for module_name in ("isaacgym", "torch", "humanoid"):
        ok, detail = probe_module(module_name)
        status = "ok" if ok else "missing"
        print(f"{module_name}: {status} ({detail})")
        if not ok:
            missing_modules.append(module_name)

    try:
        completed = subprocess.run(
            ["git", "-C", str(humanoid_gym_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            print(f"checkout_head: {completed.stdout.strip()}")
        else:
            print("checkout_head: unavailable")
    except FileNotFoundError:
        print("checkout_head: git unavailable")

    if not python_ok or missing_modules:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
