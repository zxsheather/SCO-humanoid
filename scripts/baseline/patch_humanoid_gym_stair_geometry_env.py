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
    STAIR_WIDTH_SCALE_ENV,
    load_config,
    relative_to_repo,
    resolve_humanoid_gym_root,
)

STAIR_WIDTH_SCALE_PATCH_MARKER = "SCO-humanoid local patch: configurable HumanoidTerrain stair width scale."
IMPORT_NEEDLE = "import numpy as np\n"
IMPORT_REPLACEMENT = "import os\nimport numpy as np\n"
SLOPE_LINE = "        h_slope = difficulty * 0.15\n"
HEIGHT_BLOCK = (
    f"        # {STAIR_HEIGHT_SCALE_PATCH_MARKER}\n"
    f'        stair_height_scale = float(os.environ.get("{STAIR_HEIGHT_SCALE_ENV}", "1.0"))\n'
)
WIDTH_BLOCK = (
    f"        # {STAIR_WIDTH_SCALE_PATCH_MARKER}\n"
    f'        stair_width_scale = float(os.environ.get("{STAIR_WIDTH_SCALE_ENV}", "1.0"))\n'
)
STAIR_UP_CANDIDATES = [
    "            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=discrete_obstacles_height, platform_size=1.)\n",
    "            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=discrete_obstacles_height * stair_height_scale, platform_size=1.)\n",
]
STAIR_DOWN_CANDIDATES = [
    "            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=-discrete_obstacles_height, platform_size=1.)\n",
    "            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=-(discrete_obstacles_height * stair_height_scale), platform_size=1.)\n",
]
STAIR_UP_FULL = (
    "            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4 * stair_width_scale, "
    "step_height=discrete_obstacles_height * stair_height_scale, platform_size=1.)\n"
)
STAIR_DOWN_FULL = (
    "            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4 * stair_width_scale, "
    "step_height=-(discrete_obstacles_height * stair_height_scale), platform_size=1.)\n"
)


def terrain_source_path(humanoid_gym_root: Path) -> Path:
    return humanoid_gym_root / "humanoid" / "utils" / "terrain.py"


def patch_present(source: str) -> bool:
    return STAIR_HEIGHT_SCALE_ENV in source and STAIR_WIDTH_SCALE_ENV in source


def replace_first_match(source: str, candidates: list[str], replacement: str) -> tuple[str, bool]:
    if replacement in source:
        return source, False
    for candidate in candidates:
        if candidate in source:
            return source.replace(candidate, replacement, 1), True
    raise BaselineError(f"Unable to apply stair-geometry patch; expected one of: {candidates!r}")


def patch_terrain_source(source: str) -> tuple[str, bool]:
    updated = source
    changed = False

    if "import os\n" not in updated:
        if IMPORT_NEEDLE not in updated:
            raise BaselineError(f"Unable to apply stair-geometry patch; expected snippet not found: {IMPORT_NEEDLE!r}")
        updated = updated.replace(IMPORT_NEEDLE, IMPORT_REPLACEMENT, 1)
        changed = True

    if HEIGHT_BLOCK not in updated:
        if SLOPE_LINE not in updated:
            raise BaselineError(f"Unable to apply stair-geometry patch; expected snippet not found: {SLOPE_LINE!r}")
        updated = updated.replace(SLOPE_LINE, SLOPE_LINE + HEIGHT_BLOCK, 1)
        changed = True

    if WIDTH_BLOCK not in updated:
        if HEIGHT_BLOCK not in updated:
            raise BaselineError("Unable to apply stair-geometry patch; height block missing after insertion.")
        updated = updated.replace(HEIGHT_BLOCK, HEIGHT_BLOCK + WIDTH_BLOCK, 1)
        changed = True

    updated, up_changed = replace_first_match(updated, STAIR_UP_CANDIDATES, STAIR_UP_FULL)
    changed = changed or up_changed
    updated, down_changed = replace_first_match(updated, STAIR_DOWN_CANDIDATES, STAIR_DOWN_FULL)
    changed = changed or down_changed
    return updated, changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the local Humanoid-Gym stair-geometry patch for moderated random-stairs evaluation."
    )
    parser.add_argument("--config", default=None, help="Path to any config JSON that resolves the upstream checkout.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    parser.add_argument("--check", action="store_true", help="Return success only if the geometry patch is already present.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing the upstream file.")
    args = parser.parse_args()

    config = load_config(args.config)
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    path = terrain_source_path(humanoid_gym_root)
    source = path.read_text(encoding="utf-8")

    if args.check:
        status = "ok" if patch_present(source) else "missing"
        print(f"stair_geometry_patch: {status} ({relative_to_repo(path)})")
        return 0 if patch_present(source) else 1

    updated, changed = patch_terrain_source(source)
    if not changed:
        print(f"Patch already present in {relative_to_repo(path)}")
        return 0
    if args.dry_run:
        print(
            f"Would patch {relative_to_repo(path)} to read "
            f"{STAIR_HEIGHT_SCALE_ENV} and {STAIR_WIDTH_SCALE_ENV}"
        )
        return 0

    path.write_text(updated, encoding="utf-8")
    print(
        f"Patched {relative_to_repo(path)} to read "
        f"{STAIR_HEIGHT_SCALE_ENV} and {STAIR_WIDTH_SCALE_ENV}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
