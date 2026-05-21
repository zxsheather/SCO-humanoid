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
    STAIR_DIFFICULTY_MAX_ENV,
    STAIR_DIFFICULTY_MIN_ENV,
    STRUCTURED_STAIR_LANDING_M_ENV,
    STRUCTURED_STAIR_RUNWAY_M_ENV,
    STRUCTURED_STAIR_STEP_COUNT_ENV,
    load_config,
    relative_to_repo,
    resolve_humanoid_gym_root,
)
import patch_humanoid_gym_stair_difficulty_band_env as band_patch  # noqa: E402


STRUCTURED_STAIRS_PATCH_MARKER = (
    "SCO-humanoid local patch: structured HumanoidTerrain stairs with flat spawn runway and landing."
)
STRUCTURED_STAIRS_HELPER_MARKER = (
    "SCO-humanoid local patch: structured stair helper with flat spawn runway and landing."
)
CLASS_NEEDLE = "class Terrain:\n"
STRUCTURED_STAIRS_HELPER_BLOCK = (
    f"\n\n# {STRUCTURED_STAIRS_HELPER_MARKER}\n"
    "def sco_structured_stairs_terrain(terrain, *, step_width, step_height, step_count, runway_m, landing_m):\n"
    "    step_count = max(1, int(step_count))\n"
    "    step_width_px = max(1, int(round(step_width / terrain.horizontal_scale)))\n"
    "    runway_px = max(0, int(round(runway_m / terrain.horizontal_scale)))\n"
    "    landing_px = max(0, int(round(landing_m / terrain.horizontal_scale)))\n"
    "    spawn_pad_px = max(1, int(round(0.8 / terrain.horizontal_scale)))\n"
    "    step_height_units = 0\n"
    "    if abs(step_height) > 0.0:\n"
    "        step_height_units = int(round(abs(step_height) / terrain.vertical_scale))\n"
    "        step_height_units = max(1, step_height_units)\n"
    "        step_height_units = step_height_units if step_height > 0.0 else -step_height_units\n"
    "    stair_span_px = step_count * step_width_px\n"
    "    stair_start_x = terrain.length // 2 + spawn_pad_px + runway_px\n"
    "    max_start_x = max(0, terrain.length - stair_span_px - landing_px)\n"
    "    stair_start_x = min(max_start_x, stair_start_x)\n"
    "    stair_end_x = min(terrain.length, stair_start_x + stair_span_px)\n"
    "    current_height_units = 0\n"
    "    cursor = stair_start_x\n"
    "    for _ in range(step_count):\n"
    "        next_cursor = min(stair_end_x, cursor + step_width_px)\n"
    "        current_height_units += step_height_units\n"
    "        terrain.height_field_raw[cursor:next_cursor, :] = current_height_units\n"
    "        cursor = next_cursor\n"
    "        if cursor >= stair_end_x:\n"
    "            break\n"
    "    landing_end_x = min(terrain.length, stair_end_x + landing_px)\n"
    "    terrain.height_field_raw[stair_end_x:landing_end_x, :] = current_height_units\n"
    "    terrain.height_field_raw[landing_end_x:, :] = current_height_units\n"
)
STRUCTURED_STAIRS_BLOCK = (
    f"        # {STRUCTURED_STAIRS_PATCH_MARKER}\n"
    f'        structured_stair_step_count = int(os.environ.get("{STRUCTURED_STAIR_STEP_COUNT_ENV}", "4"))\n'
    f'        structured_stair_runway_m = float(os.environ.get("{STRUCTURED_STAIR_RUNWAY_M_ENV}", "1.0"))\n'
    f'        structured_stair_landing_m = float(os.environ.get("{STRUCTURED_STAIR_LANDING_M_ENV}", "1.0"))\n'
)
STRUCTURED_STAIR_UP = (
    "            sco_structured_stairs_terrain(\n"
    "                terrain,\n"
    "                step_width=0.4 * stair_width_scale,\n"
    "                step_height=stair_step_height,\n"
    "                step_count=structured_stair_step_count,\n"
    "                runway_m=structured_stair_runway_m,\n"
    "                landing_m=structured_stair_landing_m,\n"
    "            )\n"
)
STRUCTURED_STAIR_DOWN = (
    "            sco_structured_stairs_terrain(\n"
    "                terrain,\n"
    "                step_width=0.4 * stair_width_scale,\n"
    "                step_height=-stair_step_height,\n"
    "                step_count=structured_stair_step_count,\n"
    "                runway_m=structured_stair_runway_m,\n"
    "                landing_m=structured_stair_landing_m,\n"
    "            )\n"
)


def terrain_source_path(humanoid_gym_root: Path) -> Path:
    return humanoid_gym_root / "humanoid" / "utils" / "terrain.py"


def patch_present(source: str) -> bool:
    required_markers = (
        STAIR_DIFFICULTY_MIN_ENV,
        STAIR_DIFFICULTY_MAX_ENV,
        STRUCTURED_STAIR_STEP_COUNT_ENV,
        STRUCTURED_STAIR_RUNWAY_M_ENV,
        STRUCTURED_STAIR_LANDING_M_ENV,
        STRUCTURED_STAIRS_HELPER_MARKER,
    )
    return all(marker in source for marker in required_markers)


def replace_first_match(source: str, candidates: list[str], replacement: str) -> tuple[str, bool]:
    if replacement in source:
        return source, False
    for candidate in candidates:
        if candidate in source:
            return source.replace(candidate, replacement, 1), True
    raise BaselineError(f"Unable to apply structured-stairs patch; expected one of: {candidates!r}")


def patch_terrain_source(source: str) -> tuple[str, bool]:
    if (
        STRUCTURED_STAIRS_HELPER_MARKER in source
        and STRUCTURED_STAIRS_BLOCK in source
        and STRUCTURED_STAIR_UP in source
        and STRUCTURED_STAIR_DOWN in source
    ):
        return source, False

    updated, changed = band_patch.patch_terrain_source(source)

    if STRUCTURED_STAIRS_HELPER_MARKER not in updated:
        if CLASS_NEEDLE not in updated:
            raise BaselineError(f"Unable to apply structured-stairs patch; expected snippet not found: {CLASS_NEEDLE!r}")
        updated = updated.replace(CLASS_NEEDLE, STRUCTURED_STAIRS_HELPER_BLOCK + "\n" + CLASS_NEEDLE, 1)
        changed = True

    if STRUCTURED_STAIRS_BLOCK not in updated:
        if band_patch.NEW_BAND_BLOCK not in updated:
            raise BaselineError("Unable to apply structured-stairs patch; stair-difficulty-band block missing.")
        updated = updated.replace(band_patch.NEW_BAND_BLOCK, band_patch.NEW_BAND_BLOCK + STRUCTURED_STAIRS_BLOCK, 1)
        changed = True

    updated, up_changed = replace_first_match(updated, [band_patch.cap_patch.STAIR_UP_CAP], STRUCTURED_STAIR_UP)
    changed = changed or up_changed
    updated, down_changed = replace_first_match(
        updated,
        [band_patch.cap_patch.STAIR_DOWN_CAP],
        STRUCTURED_STAIR_DOWN,
    )
    changed = changed or down_changed
    return updated, changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the local Humanoid-Gym structured-stairs patch for random-stairs redesign evaluation."
    )
    parser.add_argument("--config", default=None, help="Path to any config JSON that resolves the upstream checkout.")
    parser.add_argument("--humanoid-gym-root", default=None, help="Path to the Humanoid-Gym checkout.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return success only if the structured-stairs patch is already present.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing the upstream file.")
    args = parser.parse_args()

    config = load_config(args.config)
    humanoid_gym_root = resolve_humanoid_gym_root(config, args.humanoid_gym_root)
    path = terrain_source_path(humanoid_gym_root)
    source = path.read_text(encoding="utf-8")

    if args.check:
        status = "ok" if patch_present(source) else "missing"
        print(f"structured_stairs_patch: {status} ({relative_to_repo(path)})")
        return 0 if patch_present(source) else 1

    updated, changed = patch_terrain_source(source)
    if not changed:
        print(f"Patch already present in {relative_to_repo(path)}")
        return 0
    if args.dry_run:
        print(
            f"Would patch {relative_to_repo(path)} to read "
            f"{STRUCTURED_STAIR_STEP_COUNT_ENV}, {STRUCTURED_STAIR_RUNWAY_M_ENV}, and "
            f"{STRUCTURED_STAIR_LANDING_M_ENV}"
        )
        return 0

    path.write_text(updated, encoding="utf-8")
    print(
        f"Patched {relative_to_repo(path)} to read "
        f"{STRUCTURED_STAIR_STEP_COUNT_ENV}, {STRUCTURED_STAIR_RUNWAY_M_ENV}, and "
        f"{STRUCTURED_STAIR_LANDING_M_ENV}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
