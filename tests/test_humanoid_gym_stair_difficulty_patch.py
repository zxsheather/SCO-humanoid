from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import _common  # noqa: E402
import patch_humanoid_gym_stair_difficulty_env as stair_patch  # noqa: E402


GEOMETRY_PATCHED_SOURCE = """import os
import numpy as np

class HumanoidTerrain:
    def make_terrain(self, choice, difficulty):
        discrete_obstacles_height = difficulty * 0.04
        r_height = difficulty * 0.07
        h_slope = difficulty * 0.15
        # SCO-humanoid local patch: configurable HumanoidTerrain stair height scale.
        stair_height_scale = float(os.environ.get("SCO_HUMANOID_STAIR_HEIGHT_SCALE", "1.0"))
        # SCO-humanoid local patch: configurable HumanoidTerrain stair width scale.
        stair_width_scale = float(os.environ.get("SCO_HUMANOID_STAIR_WIDTH_SCALE", "1.0"))
        if choice < self.proportions[5]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4 * stair_width_scale, step_height=discrete_obstacles_height * stair_height_scale, platform_size=1.)
        elif choice < self.proportions[6]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4 * stair_width_scale, step_height=-(discrete_obstacles_height * stair_height_scale), platform_size=1.)
"""


class HumanoidGymStairDifficultyPatchTests(unittest.TestCase):
    def test_patch_terrain_source_inserts_difficulty_cap_runtime_control(self) -> None:
        patched, changed = stair_patch.patch_terrain_source(GEOMETRY_PATCHED_SOURCE)
        self.assertTrue(changed)
        self.assertIn(_common.STAIR_DIFFICULTY_CAP_ENV, patched)
        self.assertIn("stair_difficulty = min(difficulty, stair_difficulty_cap)", patched)
        self.assertIn("step_height=stair_step_height", patched)
        self.assertIn("step_height=-stair_step_height", patched)

    def test_patch_terrain_source_is_idempotent(self) -> None:
        patched, changed = stair_patch.patch_terrain_source(GEOMETRY_PATCHED_SOURCE)
        self.assertTrue(changed)
        repatched, changed_again = stair_patch.patch_terrain_source(patched)
        self.assertFalse(changed_again)
        self.assertEqual(repatched, patched)

    def test_verify_required_local_patches_rejects_missing_difficulty_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            terrain_path = root / "humanoid" / "utils" / "terrain.py"
            terrain_path.parent.mkdir(parents=True, exist_ok=True)
            terrain_path.write_text(
                f"{_common.STAIR_HEIGHT_SCALE_ENV}\n{_common.STAIR_WIDTH_SCALE_ENV}\n",
                encoding="utf-8",
            )
            config = {"runtime_env": {_common.STAIR_DIFFICULTY_CAP_ENV: "0.5"}}

            with self.assertRaises(_common.BaselineError):
                _common.verify_required_local_patches(config, root)

            terrain_path.write_text(
                (
                    f"{_common.STAIR_HEIGHT_SCALE_ENV}\n"
                    f"{_common.STAIR_WIDTH_SCALE_ENV}\n"
                    f"{_common.STAIR_DIFFICULTY_CAP_ENV}\n"
                ),
                encoding="utf-8",
            )
            _common.verify_required_local_patches(config, root)


if __name__ == "__main__":
    unittest.main()
