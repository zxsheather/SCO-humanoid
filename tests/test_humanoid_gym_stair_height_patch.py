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
import patch_humanoid_gym_stair_height_scale as stair_patch  # noqa: E402


UNPATCHED_SOURCE = """import numpy as np

class HumanoidTerrain:
    def make_terrain(self, choice, difficulty):
        discrete_obstacles_height = difficulty * 0.04
        r_height = difficulty * 0.07
        h_slope = difficulty * 0.15
        if choice < self.proportions[5]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=discrete_obstacles_height, platform_size=1.)
        elif choice < self.proportions[6]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4, step_height=-discrete_obstacles_height, platform_size=1.)
"""


class HumanoidGymStairHeightPatchTests(unittest.TestCase):
    def test_patch_terrain_source_inserts_runtime_scale(self) -> None:
        patched, changed = stair_patch.patch_terrain_source(UNPATCHED_SOURCE)
        self.assertTrue(changed)
        self.assertIn("import os", patched)
        self.assertIn(_common.STAIR_HEIGHT_SCALE_PATCH_MARKER, patched)
        self.assertIn(_common.STAIR_HEIGHT_SCALE_ENV, patched)
        self.assertIn("discrete_obstacles_height * stair_height_scale", patched)
        self.assertIn("-(discrete_obstacles_height * stair_height_scale)", patched)

    def test_patch_terrain_source_is_idempotent(self) -> None:
        patched, changed = stair_patch.patch_terrain_source(UNPATCHED_SOURCE)
        self.assertTrue(changed)
        repatched, changed_again = stair_patch.patch_terrain_source(patched)
        self.assertFalse(changed_again)
        self.assertEqual(repatched, patched)

    def test_verify_required_local_patches_rejects_missing_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            terrain_path = root / "humanoid" / "utils" / "terrain.py"
            terrain_path.parent.mkdir(parents=True, exist_ok=True)
            terrain_path.write_text("unpatched\n", encoding="utf-8")
            config = {"runtime_env": {_common.STAIR_HEIGHT_SCALE_ENV: "0.5"}}

            with self.assertRaises(_common.BaselineError):
                _common.verify_required_local_patches(config, root)

            terrain_path.write_text(f"{_common.STAIR_HEIGHT_SCALE_ENV}\n", encoding="utf-8")
            _common.verify_required_local_patches(config, root)


if __name__ == "__main__":
    unittest.main()
