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
    STAIR_HEIGHT_SCALE_ENV,
    STAIR_HEIGHT_SCALE_PATCH_MARKER,
    load_config,
    relative_to_repo,
    resolve_humanoid_gym_root,
)

IMPORT_NEEDLE = "import numpy as np\n"
IMPORT_REPLACEMENT = "import os\nimport numpy as np\n"
SCALE_BLOCK_NEEDLE = """        discrete_obstacles_height = difficulty * 0.04
        r_height = difficulty * 0.07
        h_slope = difficulty * 0.15
"""
SCALE_BLOCK_REPLACEMENT = f"""        discrete_obstacles_height = difficulty * 0.04
        r_height = difficulty * 0.07
        h_slope = difficulty * 0.15
        # {STAIR_HEIGHT_SCALE_PATCH_MARKER}
        stair_height_scale = float(os.environ.get("{STAIR_HEIGHT_SCALE_ENV}", "1.0"))
"""
STAIR_UP_NEEDLE = (
    "            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, "
    "step_height=discrete_obstacles_height, platform_size=1.)\n"
)
STAIR_UP_REPLACEMENT = (
    "            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, "
    "step_height=discrete_obstacles_height * stair_height_scale, platform_size=1.)\n"
)
STAIR_DOWN_NEEDLE = (
    "            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, "
    "step_height=-discrete_obstacles_height, platform_size=1.)\n"
)
STAIR_DOWN_REPLACEMENT = (
    "            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, "
    "step_height=-(discrete_obstacles_height * stair_height_scale), platform_size=1.)\n"
)


def terrain_source_path(humanoid_gym_root: Path) -> Path:
    return humanoid_gym_root / "humanoid" / "utils" / "terrain.py"


def patch_present(source: str) -> bool:
    return STAIR_HEIGHT_SCALE_PATCH_MARKER in source


def patch_terrain_source(source: str) -> tuple[str, bool]:
    if patch_present(source):
        return source, False

    updated = source
    replacements = [
        (IMPORT_NEEDLE, IMPORT_REPLACEMENT),
        (SCALE_BLOCK_NEEDLE, SCALE_BLOCK_REPLACEMENT),
        (STAIR_UP_NEEDLE, STAIR_UP_REPLACEMENT),
        (STAIR_DOWN_NEEDLE, STAIR_DOWN_REPLACEMENT),
    ]
    for needle, replacement in replacements:
        if needle not in updated:
            raise BaselineError(f"Unable to apply stair-height patch; expected snippet not found: {needle!r}")
        updated = updated.replace(needle, replacement, 1)
    return updated, True


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the local Humanoid-Gym stair-height patch for moderated random-stairs evaluation.")
    parser.add_argument("--config", default=None, help="Path to any config JSON that resolves the upstream checkout.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    parser.add_argument("--check", action="store_true", help="Return success only if the patch is already present.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing the upstream file.")
    args = parser.parse_args()

    config = load_config(args.config)
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    path = terrain_source_path(humanoid_gym_root)
    source = path.read_text(encoding="utf-8")

    if args.check:
        status = "ok" if patch_present(source) else "missing"
        print(f"stair_height_scale_patch: {status} ({relative_to_repo(path)})")
        return 0 if patch_present(source) else 1

    updated, changed = patch_terrain_source(source)
    if not changed:
        print(f"Patch already present in {relative_to_repo(path)}")
        return 0
    if args.dry_run:
        print(f"Would patch {relative_to_repo(path)} to read {STAIR_HEIGHT_SCALE_ENV}")
        return 0

    path.write_text(updated, encoding="utf-8")
    print(f"Patched {relative_to_repo(path)} to read {STAIR_HEIGHT_SCALE_ENV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
