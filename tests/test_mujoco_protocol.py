import sys
import unittest
from pathlib import Path

import numpy as np


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


class _FakeContact:
    geom1 = 1
    geom2 = 2
    dist = 0.012
    pos = np.array([0.1, 0.2, 0.3], dtype=np.double)


class _FakeData:
    ncon = 1
    contact = [_FakeContact()]
    xpos = np.array([[0.0, 0.0, 0.9]], dtype=np.double)
    ctrl = np.array([0.7, -0.4], dtype=np.double)


class _FakeMujoco:
    class mjtObj:
        mjOBJ_GEOM = "geom"

    @staticmethod
    def mj_id2name(_model, _obj_type, geom_id):
        return {1: "left_foot", 2: "floor"}.get(geom_id)

    @staticmethod
    def mj_contactForce(_model, _data, _contact_id, out):
        out[:] = np.array([3.0, 4.0, 0.0, 0.1, 0.2, 0.3], dtype=np.double)


class MuJoCoProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.legged_gym_root = REPO_ROOT / ".external" / "humanoid-gym"

    def test_alias_hfield_mode_is_explicit(self) -> None:
        self.assertEqual(mujoco_eval.normalize_terrain_mode("terrain"), "hfield_stress")
        self.assertEqual(mujoco_eval.normalize_terrain_mode("hfield"), "hfield_stress")
        self.assertEqual(mujoco_eval.normalize_terrain_mode("soft"), "hfield_moderate")
        self.assertEqual(mujoco_eval.normalize_terrain_mode("moderate"), "hfield_moderate")

    def test_isaac_mainline_plane_resolves_to_comparable_plane_protocol(self) -> None:
        protocol = mujoco_eval.resolve_mujoco_protocol(
            _PlaneCfg(),
            self.legged_gym_root,
            terrain_mode="isaac_mainline",
        )
        self.assertEqual(protocol["terrain_mode"], "plane")
        self.assertTrue(protocol["is_isaac_mainline_comparable"])
        self.assertEqual(protocol["purpose"], "minimal_comparable_first_pass")
        self.assertEqual(protocol["mujoco_model_template_path"].name, "XBot-L.xml")
        self.assertIsNone(protocol["hfield_size_override"])

    def test_hfield_stress_is_never_marked_as_mainline_comparable(self) -> None:
        protocol = mujoco_eval.resolve_mujoco_protocol(
            _PlaneCfg(),
            self.legged_gym_root,
            terrain_mode="hfield_stress",
        )
        self.assertEqual(protocol["terrain_mode"], "hfield_stress")
        self.assertFalse(protocol["is_isaac_mainline_comparable"])
        self.assertEqual(protocol["purpose"], "terrain_stress_probe")
        self.assertEqual(protocol["mujoco_model_template_path"].name, "XBot-L-terrain.xml")
        self.assertIsNone(protocol["hfield_size_override"])

    def test_hfield_moderate_uses_softened_hfield_override(self) -> None:
        protocol = mujoco_eval.resolve_mujoco_protocol(
            _PlaneCfg(),
            self.legged_gym_root,
            terrain_mode="hfield_moderate",
        )
        self.assertEqual(protocol["terrain_mode"], "hfield_moderate")
        self.assertFalse(protocol["is_isaac_mainline_comparable"])
        self.assertEqual(protocol["purpose"], "terrain_repair_probe")
        self.assertEqual(protocol["mujoco_model_template_path"].name, "XBot-L-terrain.xml")
        self.assertEqual(protocol["hfield_size_override"], mujoco_eval.HFIELD_MODERATE_SIZE)

    def test_unaligned_isaac_mainline_raises_instead_of_silent_hfield_fallback(self) -> None:
        with self.assertRaises(mujoco_eval.BaselineError) as context:
            mujoco_eval.resolve_mujoco_protocol(
                _TrimeshCfg(),
                self.legged_gym_root,
                terrain_mode="isaac_mainline",
            )
        self.assertIn("no aligned MuJoCo terrain profile", str(context.exception))

    def test_materialize_mujoco_model_rewrites_hfield_size_and_assets(self) -> None:
        protocol = mujoco_eval.resolve_mujoco_protocol(
            _PlaneCfg(),
            self.legged_gym_root,
            terrain_mode="hfield_moderate",
        )
        model_path = mujoco_eval.materialize_mujoco_model(protocol)
        self.assertTrue(model_path.exists())
        xml_text = model_path.read_text(encoding="utf-8")
        self.assertIn('meshdir="', xml_text)
        self.assertIn('file="', xml_text)
        self.assertIn('size="50 50 0.06 0.02"', xml_text)
        self.assertIn('uneven.png', xml_text)

    def test_trace_step_limit_is_inclusive_of_requested_count(self) -> None:
        self.assertTrue(mujoco_eval.should_record_trace_step(1, 1))
        self.assertTrue(mujoco_eval.should_record_trace_step(1024, 1024))
        self.assertFalse(mujoco_eval.should_record_trace_step(1025, 1024))
        self.assertFalse(mujoco_eval.should_record_trace_step(1, 0))

    def test_action_lowpass_proxy_filters_policy_action(self) -> None:
        proxy = mujoco_eval.build_actuator_proxy_config(
            mode="action-lowpass",
            lowpass_time_constant=0.03,
            control_dt=0.01,
        )
        self.assertEqual(proxy["mode"], "action_lowpass")
        self.assertAlmostEqual(proxy["lowpass_alpha"], 0.25)

        applied = mujoco_eval.apply_actuator_proxy_action(
            np.array([1.0, -1.0], dtype=np.double),
            np.array([0.2, -0.2], dtype=np.double),
            proxy,
        )
        np.testing.assert_allclose(applied, np.array([0.4, -0.4], dtype=np.double))

    def test_no_actuator_proxy_applies_policy_action_directly(self) -> None:
        proxy = mujoco_eval.build_actuator_proxy_config(
            mode="none",
            lowpass_time_constant=0.05,
            control_dt=0.01,
        )
        policy_action = np.array([0.7, -0.1], dtype=np.double)
        applied = mujoco_eval.apply_actuator_proxy_action(
            policy_action,
            np.array([0.0, 0.0], dtype=np.double),
            proxy,
        )
        np.testing.assert_allclose(applied, policy_action)

    def test_build_trace_step_records_control_and_contact_schema(self) -> None:
        step = mujoco_eval.build_trace_step(
            step_count=1,
            action=np.array([0.1, -0.2], dtype=np.double),
            prev_action=np.array([0.0, 0.1], dtype=np.double),
            applied_action=np.array([0.05, -0.1], dtype=np.double),
            prev_applied_action=np.array([0.0, 0.05], dtype=np.double),
            dq_before=np.array([1.0, 2.0], dtype=np.double),
            dq_after=np.array([1.4, 1.0], dtype=np.double),
            control_dt=0.2,
            target_q=np.array([0.3, -0.6], dtype=np.double),
            tau=np.array([0.7, -0.4], dtype=np.double),
            data=_FakeData(),
            base_body_id=0,
            base_lin_vel_after=np.array([0.4, 0.0, 0.0], dtype=np.double),
            base_ang_vel_after=np.array([0.0, 0.0, 0.1], dtype=np.double),
            model=object(),
            mujoco=_FakeMujoco(),
        )

        self.assertEqual(step["control_tau"], [0.7, -0.4])
        self.assertEqual(step["applied_control"], [0.7, -0.4])
        self.assertEqual(step["applied_action"], [0.05, -0.1])
        self.assertEqual(step["target_joint_position"], [0.3, -0.6])
        self.assertAlmostEqual(step["action_jitter_l2"], float(np.linalg.norm([0.1, -0.3])))
        self.assertAlmostEqual(step["applied_action_jitter_l2"], float(np.linalg.norm([0.05, -0.15])))
        self.assertAlmostEqual(step["action_lag_l2"], float(np.linalg.norm([0.05, -0.1])))
        self.assertEqual(step["contact_count"], 1)
        self.assertEqual(len(step["contacts"]), 1)
        contact = step["contacts"][0]
        self.assertEqual(contact["geom1_name"], "left_foot")
        self.assertEqual(contact["geom2_name"], "floor")
        self.assertEqual(contact["force_torque"], [3.0, 4.0, 0.0, 0.1, 0.2, 0.3])
        self.assertAlmostEqual(contact["force_norm"], 5.0)
        self.assertEqual(contact["position"], [0.1, 0.2, 0.3])

    def test_update_trace_manifest_clears_stale_path_when_disabled(self) -> None:
        manifest = {"mujoco_trace_path": "old_trace.json", "other": 1}
        mujoco_eval.update_trace_manifest(manifest, trace_path=None)
        self.assertNotIn("mujoco_trace_path", manifest)
        self.assertEqual(manifest["other"], 1)


if __name__ == "__main__":
    unittest.main()
