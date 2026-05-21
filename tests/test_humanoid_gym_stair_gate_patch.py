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
import patch_humanoid_gym_stair_gate_env as stair_patch  # noqa: E402


BAND_PATCHED_SOURCE = """import os
import numpy as np

class Terrain:
    pass

class HumanoidTerrain:
    def make_terrain(self, choice, difficulty):
        discrete_obstacles_height = difficulty * 0.04
        r_height = difficulty * 0.07
        h_slope = difficulty * 0.15
        # SCO-humanoid local patch: configurable HumanoidTerrain stair height scale.
        stair_height_scale = float(os.environ.get("SCO_HUMANOID_STAIR_HEIGHT_SCALE", "1.0"))
        # SCO-humanoid local patch: configurable HumanoidTerrain stair width scale.
        stair_width_scale = float(os.environ.get("SCO_HUMANOID_STAIR_WIDTH_SCALE", "1.0"))
        # SCO-humanoid local patch: configurable HumanoidTerrain decoupled stair difficulty band.
        stair_difficulty_cap = float(os.environ.get("SCO_HUMANOID_STAIR_DIFFICULTY_CAP", "1.0"))
        stair_difficulty_min_raw = os.environ.get("SCO_HUMANOID_STAIR_DIFFICULTY_MIN")
        stair_difficulty_max_raw = os.environ.get("SCO_HUMANOID_STAIR_DIFFICULTY_MAX")
        if stair_difficulty_min_raw is None and stair_difficulty_max_raw is None:
            stair_difficulty = min(difficulty, stair_difficulty_cap)
        else:
            stair_difficulty_min = float(stair_difficulty_min_raw or "0.0")
            stair_difficulty_max = float(stair_difficulty_max_raw or stair_difficulty_min_raw or "1.0")
            if stair_difficulty_max < stair_difficulty_min:
                stair_difficulty_min, stair_difficulty_max = stair_difficulty_max, stair_difficulty_min
            stair_difficulty = np.random.uniform(stair_difficulty_min, stair_difficulty_max)
        stair_step_height = stair_difficulty * 0.04 * stair_height_scale
        if choice < self.proportions[5]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4 * stair_width_scale, step_height=stair_step_height, platform_size=1.)
        elif choice < self.proportions[6]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.4 * stair_width_scale, step_height=-stair_step_height, platform_size=1.)
"""

STRUCTURED_PATCHED_SOURCE = """import os
import numpy as np

# SCO-humanoid local patch: structured stair helper with flat spawn runway and landing.
def sco_structured_stairs_terrain(terrain, *, step_width, step_height, step_count, runway_m, landing_m):
    pass

class Terrain:
    pass

class HumanoidTerrain:
    def make_terrain(self, choice, difficulty):
        discrete_obstacles_height = difficulty * 0.04
        r_height = difficulty * 0.07
        h_slope = difficulty * 0.15
        # SCO-humanoid local patch: configurable HumanoidTerrain stair height scale.
        stair_height_scale = float(os.environ.get("SCO_HUMANOID_STAIR_HEIGHT_SCALE", "1.0"))
        # SCO-humanoid local patch: configurable HumanoidTerrain stair width scale.
        stair_width_scale = float(os.environ.get("SCO_HUMANOID_STAIR_WIDTH_SCALE", "1.0"))
        # SCO-humanoid local patch: configurable HumanoidTerrain decoupled stair difficulty band.
        stair_difficulty_cap = float(os.environ.get("SCO_HUMANOID_STAIR_DIFFICULTY_CAP", "1.0"))
        stair_difficulty_min_raw = os.environ.get("SCO_HUMANOID_STAIR_DIFFICULTY_MIN")
        stair_difficulty_max_raw = os.environ.get("SCO_HUMANOID_STAIR_DIFFICULTY_MAX")
        if stair_difficulty_min_raw is None and stair_difficulty_max_raw is None:
            stair_difficulty = min(difficulty, stair_difficulty_cap)
        else:
            stair_difficulty_min = float(stair_difficulty_min_raw or "0.0")
            stair_difficulty_max = float(stair_difficulty_max_raw or stair_difficulty_min_raw or "1.0")
            if stair_difficulty_max < stair_difficulty_min:
                stair_difficulty_min, stair_difficulty_max = stair_difficulty_max, stair_difficulty_min
            stair_difficulty = np.random.uniform(stair_difficulty_min, stair_difficulty_max)
        stair_step_height = stair_difficulty * 0.04 * stair_height_scale
        # SCO-humanoid local patch: structured HumanoidTerrain stairs with flat spawn runway and landing.
        structured_stair_step_count = int(os.environ.get("SCO_HUMANOID_STRUCTURED_STAIR_STEP_COUNT", "4"))
        structured_stair_runway_m = float(os.environ.get("SCO_HUMANOID_STRUCTURED_STAIR_RUNWAY_M", "1.0"))
        structured_stair_landing_m = float(os.environ.get("SCO_HUMANOID_STRUCTURED_STAIR_LANDING_M", "1.0"))
        if choice < self.proportions[5]:
            sco_structured_stairs_terrain(
                terrain,
                step_width=0.4 * stair_width_scale,
                step_height=stair_step_height,
                step_count=structured_stair_step_count,
                runway_m=structured_stair_runway_m,
                landing_m=structured_stair_landing_m,
            )
        elif choice < self.proportions[6]:
            sco_structured_stairs_terrain(
                terrain,
                step_width=0.4 * stair_width_scale,
                step_height=-stair_step_height,
                step_count=structured_stair_step_count,
                runway_m=structured_stair_runway_m,
                landing_m=structured_stair_landing_m,
            )
"""


class HumanoidGymStairGatePatchTests(unittest.TestCase):
    def test_patch_terrain_source_inserts_gate_controls(self) -> None:
        patched, changed = stair_patch.patch_terrain_source(BAND_PATCHED_SOURCE)
        self.assertTrue(changed)
        self.assertIn(_common.STAIR_GATE_STEP_COUNT_ENV, patched)
        self.assertIn(_common.STAIR_GATE_RUNWAY_M_ENV, patched)
        self.assertIn(_common.STAIR_GATE_PLATFORM_M_ENV, patched)
        self.assertIn(_common.STAIR_GATE_EXIT_M_ENV, patched)
        self.assertIn("def sco_structured_stair_gate_terrain", patched)
        self.assertIn("current_height_units -= step_height_units", patched)

    def test_patch_terrain_source_is_idempotent(self) -> None:
        patched, changed = stair_patch.patch_terrain_source(BAND_PATCHED_SOURCE)
        self.assertTrue(changed)
        repatched, changed_again = stair_patch.patch_terrain_source(patched)
        self.assertFalse(changed_again)
        self.assertEqual(repatched, patched)

    def test_patch_terrain_source_upgrades_structured_patch(self) -> None:
        patched, changed = stair_patch.patch_terrain_source(STRUCTURED_PATCHED_SOURCE)
        self.assertTrue(changed)
        self.assertIn("def sco_structured_stair_gate_terrain", patched)
        self.assertIn("platform_m=stair_gate_platform_m", patched)
        self.assertIn("exit_m=stair_gate_exit_m", patched)

    def test_verify_required_local_patches_rejects_missing_gate_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            terrain_path = root / "humanoid" / "utils" / "terrain.py"
            terrain_path.parent.mkdir(parents=True, exist_ok=True)
            terrain_path.write_text(
                (
                    f"{_common.STAIR_HEIGHT_SCALE_ENV}\n"
                    f"{_common.STAIR_WIDTH_SCALE_ENV}\n"
                    f"{_common.STAIR_DIFFICULTY_CAP_ENV}\n"
                    f"{_common.STAIR_DIFFICULTY_MIN_ENV}\n"
                    f"{_common.STAIR_DIFFICULTY_MAX_ENV}\n"
                ),
                encoding="utf-8",
            )
            config = {
                "runtime_env": {
                    _common.STAIR_DIFFICULTY_MIN_ENV: "0.0",
                    _common.STAIR_DIFFICULTY_MAX_ENV: "0.35",
                    _common.STAIR_GATE_STEP_COUNT_ENV: "3",
                    _common.STAIR_GATE_RUNWAY_M_ENV: "1.0",
                    _common.STAIR_GATE_PLATFORM_M_ENV: "0.6",
                    _common.STAIR_GATE_EXIT_M_ENV: "1.0",
                }
            }

            with self.assertRaises(_common.BaselineError):
                _common.verify_required_local_patches(config, root)

            terrain_path.write_text(
                (
                    f"{_common.STAIR_HEIGHT_SCALE_ENV}\n"
                    f"{_common.STAIR_WIDTH_SCALE_ENV}\n"
                    f"{_common.STAIR_DIFFICULTY_CAP_ENV}\n"
                    f"{_common.STAIR_DIFFICULTY_MIN_ENV}\n"
                    f"{_common.STAIR_DIFFICULTY_MAX_ENV}\n"
                    f"{_common.STAIR_GATE_STEP_COUNT_ENV}\n"
                    f"{_common.STAIR_GATE_RUNWAY_M_ENV}\n"
                    f"{_common.STAIR_GATE_PLATFORM_M_ENV}\n"
                    f"{_common.STAIR_GATE_EXIT_M_ENV}\n"
                ),
                encoding="utf-8",
            )
            _common.verify_required_local_patches(config, root)


if __name__ == "__main__":
    unittest.main()
