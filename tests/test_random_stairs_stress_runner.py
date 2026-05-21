from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import run_random_stairs_stress_test as random_stairs  # noqa: E402


class RandomStairsStressRunnerTests(unittest.TestCase):
    def protocol_stub(self, **overrides):
        terrain_protocol = {
            "terrain_condition": "random_stairs",
            "scope": "复杂地形条件 pressure test",
            "claim_boundary": "stress test only; not retraining or a new headline method line",
        }
        terrain_protocol.update(overrides)
        return {"name": "test_protocol", "terrain_protocol": terrain_protocol}

    def build_args(self, **overrides):
        defaults = {
            "reuse_existing_metrics": False,
            "humanoid_gym_root": None,
            "eval_num_envs": None,
            "episodes": None,
            "rl_device": None,
            "sim_device": None,
            "analysis_root": None,
            "run_suffix": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_random_stairs_protocol_preserves_actor_observation_contract(self) -> None:
        config_path = REPO_ROOT / "configs" / "methods" / "sc_ppo_threshold_38_pid_random_stairs_eval.json"
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        env_overrides = config["overrides"]["env"]
        self.assertEqual(env_overrides["terrain.mesh_type"], "trimesh")
        self.assertFalse(env_overrides["terrain.measure_heights"])
        self.assertEqual(env_overrides["terrain.terrain_proportions"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5])

    def test_build_evaluate_command_targets_selected_checkpoint(self) -> None:
        sweep_cfg = random_stairs.load_sweep_config(random_stairs.DEFAULT_SWEEP_CONFIG)
        candidate = next(item for item in sweep_cfg["candidates"] if item["id"] == "sc_ppo")
        config_path = random_stairs.resolve_config_path(candidate["config"])
        args = self.build_args()

        command = random_stairs.build_evaluate_command(
            sweep_cfg=sweep_cfg,
            candidate=candidate,
            config_path=config_path,
            run_name=random_stairs.candidate_run_name(candidate, 23),
            seed=23,
            args=args,
        )

        self.assertIn("--checkpoints", command)
        self.assertEqual(command[command.index("--checkpoints") + 1], "400")
        self.assertIn("--load-run", command)
        self.assertIn("seed23", command[command.index("--load-run") + 1])
        self.assertIn("--seed", command)
        self.assertEqual(command[command.index("--seed") + 1], "23")

    def test_aggregate_metrics_uses_population_std(self) -> None:
        aggregate = random_stairs.aggregate_metrics(
            [
                {"velocity_tracking_error_mean": 1.0, "fall_rate": 0.0},
                {"velocity_tracking_error_mean": 3.0, "fall_rate": 1.0},
            ]
        )

        self.assertEqual(aggregate["velocity_tracking_error_mean_mean"], 2.0)
        self.assertEqual(aggregate["velocity_tracking_error_mean_std"], 1.0)
        self.assertEqual(aggregate["fall_rate_mean"], 0.5)

    def test_interpretation_compares_scppo_against_heuristic(self) -> None:
        interpretation = random_stairs.build_interpretation(
            self.protocol_stub(terrain_condition="random_stairs_moderated"),
            [
                {
                    "id": "heuristic_smoothing",
                    "status": "complete",
                    "aggregate": {
                        "velocity_tracking_error_mean_mean": 0.7,
                        "episode_return_mean_mean": 100.0,
                    },
                },
                {
                    "id": "sc_ppo",
                    "status": "complete",
                    "aggregate": {
                        "velocity_tracking_error_mean_mean": 0.6,
                        "episode_return_mean_mean": 90.0,
                    },
                },
            ]
        )

        comparison = interpretation["sc_ppo_vs_revised_heuristic"]
        self.assertEqual(comparison["velocity_tracking_error_mean"]["ordering"], "sc_ppo_better")
        self.assertEqual(comparison["episode_return_mean"]["ordering"], "heuristic_better")
        self.assertEqual(interpretation["task_validity_outcome"], "mixed_or_incomplete")
        self.assertEqual(interpretation["terrain_condition"], "random_stairs_moderated")

    def test_interpretation_flags_all_methods_collapsed(self) -> None:
        interpretation = random_stairs.build_interpretation(
            self.protocol_stub(),
            [
                {
                    "id": "heuristic_smoothing",
                    "status": "collapsed",
                    "aggregate": {"fall_rate_mean": 1.0},
                },
                {
                    "id": "sc_ppo",
                    "status": "collapsed",
                    "aggregate": {"fall_rate_mean": 1.0},
                },
            ]
        )

        self.assertEqual(interpretation["task_validity_outcome"], "all_methods_collapsed")
        self.assertEqual(interpretation["collapsed_candidate_ids"], ["heuristic_smoothing", "sc_ppo"])

    def test_moderated_random_stairs_protocol_preserves_actor_observation_contract(self) -> None:
        config_path = REPO_ROOT / "configs" / "methods" / "sc_ppo_threshold_38_pid_random_stairs_moderated_eval.json"
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        env_overrides = config["overrides"]["env"]
        protocol = config["evaluation_protocol"]
        self.assertEqual(protocol["id"], "random_stairs_moderated_selected_checkpoint_stress")
        self.assertEqual(protocol["terrain_condition"], "random_stairs_moderated")
        self.assertFalse(env_overrides["terrain.measure_heights"])
        self.assertEqual(env_overrides["terrain.terrain_proportions"], [0.0, 0.0, 0.5, 0.0, 0.0, 0.25, 0.25])

    def test_moderated_sweep_reuses_selected_checkpoints_with_new_run_names(self) -> None:
        sweep_cfg = random_stairs.load_sweep_config(
            REPO_ROOT / "configs" / "sweeps" / "random_stairs_moderated_selected_checkpoint_stress.json"
        )
        sc_candidate = next(item for item in sweep_cfg["candidates"] if item["id"] == "sc_ppo")
        self.assertEqual(sc_candidate["selected_checkpoints"], {"11": 300, "17": 300, "23": 400})
        self.assertIn("moderated", sc_candidate["run_name_template"])

    def test_halfheight_protocol_declares_patch_dependency_and_runtime_env(self) -> None:
        config_path = (
            REPO_ROOT / "configs" / "methods" / "sc_ppo_threshold_38_pid_random_stairs_moderated_halfheight_eval.json"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        self.assertEqual(config["runtime_env"]["SCO_HUMANOID_STAIR_HEIGHT_SCALE"], "0.5")
        protocol = config["evaluation_protocol"]
        self.assertEqual(protocol["terrain_condition"], "random_stairs_moderated_halfheight")
        self.assertEqual(protocol["stair_height_scale"], 0.5)
        self.assertEqual(
            protocol["local_patch_dependency"],
            "scripts/baseline/patch_humanoid_gym_stair_height_scale.py",
        )

    def test_halfheight_sweep_reuses_selected_checkpoints(self) -> None:
        sweep_cfg = random_stairs.load_sweep_config(
            REPO_ROOT / "configs" / "sweeps" / "random_stairs_moderated_halfheight_selected_checkpoint_stress.json"
        )
        protocol = sweep_cfg["terrain_protocol"]
        self.assertEqual(protocol["stair_height_scale"], 0.5)
        sc_candidate = next(item for item in sweep_cfg["candidates"] if item["id"] == "sc_ppo")
        self.assertEqual(sc_candidate["selected_checkpoints"], {"11": 300, "17": 300, "23": 400})
        self.assertIn("halfheight", sc_candidate["run_name_template"])

    def test_widestep_protocol_declares_patch_dependency_and_runtime_env(self) -> None:
        config_path = (
            REPO_ROOT / "configs" / "methods" / "sc_ppo_threshold_38_pid_random_stairs_moderated_widestep_eval.json"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        self.assertEqual(config["runtime_env"]["SCO_HUMANOID_STAIR_WIDTH_SCALE"], "2.0")
        protocol = config["evaluation_protocol"]
        self.assertEqual(protocol["terrain_condition"], "random_stairs_moderated_widestep")
        self.assertEqual(protocol["stair_width_scale"], 2.0)
        self.assertEqual(
            protocol["local_patch_dependency"],
            "scripts/baseline/patch_humanoid_gym_stair_geometry_env.py",
        )

    def test_widestep_sweep_reuses_selected_checkpoints(self) -> None:
        sweep_cfg = random_stairs.load_sweep_config(
            REPO_ROOT / "configs" / "sweeps" / "random_stairs_moderated_widestep_selected_checkpoint_stress.json"
        )
        protocol = sweep_cfg["terrain_protocol"]
        self.assertEqual(protocol["stair_width_scale"], 2.0)
        sc_candidate = next(item for item in sweep_cfg["candidates"] if item["id"] == "sc_ppo")
        self.assertEqual(sc_candidate["selected_checkpoints"], {"11": 300, "17": 300, "23": 400})
        self.assertIn("widestep", sc_candidate["run_name_template"])

    def test_difficultycap_protocol_declares_patch_dependency_and_runtime_env(self) -> None:
        config_path = (
            REPO_ROOT / "configs" / "methods" / "sc_ppo_threshold_38_pid_random_stairs_moderated_difficultycap_eval.json"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        self.assertEqual(config["runtime_env"]["SCO_HUMANOID_STAIR_DIFFICULTY_CAP"], "0.5")
        protocol = config["evaluation_protocol"]
        self.assertEqual(protocol["terrain_condition"], "random_stairs_moderated_difficultycap")
        self.assertEqual(protocol["stair_difficulty_cap"], 0.5)
        self.assertEqual(
            protocol["local_patch_dependency"],
            "scripts/baseline/patch_humanoid_gym_stair_difficulty_env.py",
        )

    def test_difficultycap_sweep_reuses_selected_checkpoints(self) -> None:
        sweep_cfg = random_stairs.load_sweep_config(
            REPO_ROOT / "configs" / "sweeps" / "random_stairs_moderated_difficultycap_selected_checkpoint_stress.json"
        )
        protocol = sweep_cfg["terrain_protocol"]
        self.assertEqual(protocol["stair_difficulty_cap"], 0.5)
        sc_candidate = next(item for item in sweep_cfg["candidates"] if item["id"] == "sc_ppo")
        self.assertEqual(sc_candidate["selected_checkpoints"], {"11": 300, "17": 300, "23": 400})
        self.assertIn("difficultycap", sc_candidate["run_name_template"])

    def test_decoupled_stairband_protocol_declares_patch_dependency_and_runtime_env(self) -> None:
        config_path = (
            REPO_ROOT / "configs" / "methods" / "sc_ppo_threshold_38_pid_random_stairs_decoupled_stairband_eval.json"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        self.assertEqual(config["runtime_env"]["SCO_HUMANOID_STAIR_DIFFICULTY_MIN"], "0.0")
        self.assertEqual(config["runtime_env"]["SCO_HUMANOID_STAIR_DIFFICULTY_MAX"], "0.35")
        protocol = config["evaluation_protocol"]
        self.assertEqual(protocol["terrain_condition"], "random_stairs_decoupled_stairband")
        self.assertEqual(protocol["stair_difficulty_band"], [0.0, 0.35])
        self.assertEqual(
            protocol["local_patch_dependency"],
            "scripts/baseline/patch_humanoid_gym_stair_difficulty_band_env.py",
        )

    def test_decoupled_stairband_sweep_reuses_selected_checkpoints(self) -> None:
        sweep_cfg = random_stairs.load_sweep_config(
            REPO_ROOT / "configs" / "sweeps" / "random_stairs_decoupled_stairband_selected_checkpoint_stress.json"
        )
        protocol = sweep_cfg["terrain_protocol"]
        self.assertEqual(protocol["stair_difficulty_band"], [0.0, 0.35])
        sc_candidate = next(item for item in sweep_cfg["candidates"] if item["id"] == "sc_ppo")
        self.assertEqual(sc_candidate["selected_checkpoints"], {"11": 300, "17": 300, "23": 400})
        self.assertIn("stairband", sc_candidate["run_name_template"])


if __name__ == "__main__":
    unittest.main()
