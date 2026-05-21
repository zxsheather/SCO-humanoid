#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    BaselineError,
    STAIR_DIFFICULTY_CAP_ENV,
    STAIR_DIFFICULTY_MAX_ENV,
    STAIR_DIFFICULTY_MIN_ENV,
    load_config,
    relative_to_repo,
    resolve_humanoid_gym_root,
)
import patch_humanoid_gym_stair_difficulty_env as cap_patch  # noqa: E402


STAIR_DIFFICULTY_BAND_PATCH_MARKER = (
    "SCO-humanoid local patch: configurable HumanoidTerrain decoupled stair difficulty band."
)
OLD_CAP_BLOCK = cap_patch.CAP_BLOCK
NEW_BAND_BLOCK = (
    f"        # {STAIR_DIFFICULTY_BAND_PATCH_MARKER}\n"
    f'        stair_difficulty_cap = float(os.environ.get("{STAIR_DIFFICULTY_CAP_ENV}", "1.0"))\n'
    f'        stair_difficulty_min_raw = os.environ.get("{STAIR_DIFFICULTY_MIN_ENV}")\n'
    f'        stair_difficulty_max_raw = os.environ.get("{STAIR_DIFFICULTY_MAX_ENV}")\n'
    "        if stair_difficulty_min_raw is None and stair_difficulty_max_raw is None:\n"
    "            stair_difficulty = min(difficulty, stair_difficulty_cap)\n"
    "        else:\n"
    '            stair_difficulty_min = float(stair_difficulty_min_raw or "0.0")\n'
    "            stair_difficulty_max = float(stair_difficulty_max_raw or stair_difficulty_min_raw or \"1.0\")\n"
    "            if stair_difficulty_max < stair_difficulty_min:\n"
    "                stair_difficulty_min, stair_difficulty_max = stair_difficulty_max, stair_difficulty_min\n"
    "            stair_difficulty = np.random.uniform(stair_difficulty_min, stair_difficulty_max)\n"
    "        stair_step_height = stair_difficulty * 0.04 * stair_height_scale\n"
)


def terrain_source_path(humanoid_gym_root: Path) -> Path:
    return humanoid_gym_root / "humanoid" / "utils" / "terrain.py"


def patch_present(source: str) -> bool:
    return STAIR_DIFFICULTY_MIN_ENV in source and STAIR_DIFFICULTY_MAX_ENV in source


def patch_terrain_source(source: str) -> tuple[str, bool]:
    if NEW_BAND_BLOCK in source and cap_patch.STAIR_UP_CAP in source and cap_patch.STAIR_DOWN_CAP in source:
        return source, False

    updated, changed = cap_patch.patch_terrain_source(source)

    if NEW_BAND_BLOCK in updated:
        return updated, changed

    if OLD_CAP_BLOCK not in updated:
        raise BaselineError("Unable to apply stair-difficulty-band patch; stair difficulty cap block missing.")
    updated = updated.replace(OLD_CAP_BLOCK, NEW_BAND_BLOCK, 1)
    changed = True
    return updated, changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the local Humanoid-Gym stair-difficulty-band patch for decoupled random-stairs evaluation."
    )
    parser.add_argument("--config", default=None, help="Path to any config JSON that resolves the upstream checkout.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return success only if the stair-difficulty-band patch is already present.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing the upstream file.")
    args = parser.parse_args()

    config = load_config(args.config)
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    path = terrain_source_path(humanoid_gym_root)
    source = path.read_text(encoding="utf-8")

    if args.check:
        status = "ok" if patch_present(source) else "missing"
        print(f"stair_difficulty_band_patch: {status} ({relative_to_repo(path)})")
        return 0 if patch_present(source) else 1

    updated, changed = patch_terrain_source(source)
    if not changed:
        print(f"Patch already present in {relative_to_repo(path)}")
        return 0
    if args.dry_run:
        print(
            f"Would patch {relative_to_repo(path)} to read "
            f"{STAIR_DIFFICULTY_MIN_ENV} and {STAIR_DIFFICULTY_MAX_ENV}"
        )
        return 0

    path.write_text(updated, encoding="utf-8")
    print(
        f"Patched {relative_to_repo(path)} to read "
        f"{STAIR_DIFFICULTY_MIN_ENV} and {STAIR_DIFFICULTY_MAX_ENV}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
