import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import evaluate_mujoco_sim2sim as mujoco_eval  # noqa: E402


class _PlaneTerrainCfg:
    mesh_type = "plane"


class _PlaneCfg:
    terrain = _PlaneTerrainCfg()


class _TrimeshTerrainCfg:
    mesh_type = "trimesh"


class _TrimeshCfg:
    terrain = _TrimeshTerrainCfg()


class MuJoCoProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.legged_gym_root = REPO_ROOT / ".external" / "humanoid-gym"

    def test_alias_hfield_mode_is_explicit(self) -> None:
        self.assertEqual(mujoco_eval.normalize_terrain_mode("terrain"), "hfield_stress")
        self.assertEqual(mujoco_eval.normalize_terrain_mode("hfield"), "hfield_stress")

    def test_isaac_mainline_plane_resolves_to_comparable_plane_protocol(self) -> None:
        protocol = mujoco_eval.resolve_mujoco_protocol(
            _PlaneCfg(),
            self.legged_gym_root,
            terrain_mode="isaac_mainline",
        )
        self.assertEqual(protocol["terrain_mode"], "plane")
        self.assertTrue(protocol["is_isaac_mainline_comparable"])
        self.assertEqual(protocol["purpose"], "minimal_comparable_first_pass")
        self.assertEqual(protocol["mujoco_model_path"].name, "XBot-L.xml")

    def test_hfield_stress_is_never_marked_as_mainline_comparable(self) -> None:
        protocol = mujoco_eval.resolve_mujoco_protocol(
            _PlaneCfg(),
            self.legged_gym_root,
            terrain_mode="hfield_stress",
        )
        self.assertEqual(protocol["terrain_mode"], "hfield_stress")
        self.assertFalse(protocol["is_isaac_mainline_comparable"])
        self.assertEqual(protocol["purpose"], "terrain_stress_probe")
        self.assertEqual(protocol["mujoco_model_path"].name, "XBot-L-terrain.xml")

    def test_unaligned_isaac_mainline_raises_instead_of_silent_hfield_fallback(self) -> None:
        with self.assertRaises(mujoco_eval.BaselineError) as context:
            mujoco_eval.resolve_mujoco_protocol(
                _TrimeshCfg(),
                self.legged_gym_root,
                terrain_mode="isaac_mainline",
            )
        self.assertIn("no aligned MuJoCo terrain profile", str(context.exception))


if __name__ == "__main__":
    unittest.main()
