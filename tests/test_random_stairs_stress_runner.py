from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import _common as baseline_common  # noqa: E402
import run_random_stairs_stress_test as random_stairs  # noqa: E402


class RandomStairsStressRunnerTests(unittest.TestCase):
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

    def test_moderated_random_stairs_protocol_restores_rough_uniform_majority(self) -> None:
        config_path = REPO_ROOT / "configs" / "methods" / "sc_ppo_threshold_38_pid_random_stairs_moderated_eval.json"
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        env_overrides = config["overrides"]["env"]
        self.assertEqual(env_overrides["terrain.mesh_type"], "trimesh")
        self.assertFalse(env_overrides["terrain.measure_heights"])
        self.assertEqual(env_overrides["terrain.num_cols"], 10)
        self.assertEqual(env_overrides["terrain.terrain_proportions"], [0.0, 0.0, 0.7, 0.0, 0.0, 0.15, 0.15])

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

    def test_interpretation_flags_all_methods_collapsed(self) -> None:
        interpretation = random_stairs.build_interpretation(
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

    def test_resolve_run_dir_accepts_repo_relative_load_run_with_external_logs_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            humanoid_gym_root = tmp_root / "existing-checkout" / ".external" / "humanoid-gym"
            run_dir = humanoid_gym_root / "logs" / "demo_experiment" / "May22_demo_run"
            run_dir.mkdir(parents=True)

            resolved = baseline_common.resolve_run_dir(
                humanoid_gym_root=humanoid_gym_root,
                config={
                    "experiment_name": "demo_experiment",
                    "artifacts_root": "artifacts/demo",
                },
                load_run=".external/humanoid-gym/logs/demo_experiment/May22_demo_run",
            )

            self.assertEqual(resolved, run_dir.resolve())


if __name__ == "__main__":
    unittest.main()
