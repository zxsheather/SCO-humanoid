from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import _common as common  # noqa: E402
import run_sn_diagnostic as sn_diagnostic  # noqa: E402


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_runtime_env_prepends_current_python_bin(self) -> None:
        env = common.runtime_env()

        self.assertEqual(env["PATH"].split(os.pathsep)[0], common.current_python_bin())
        self.assertNotIn("DISPLAY", env)
        self.assertEqual(env["WANDB_MODE"], "disabled")


class SnDiagnosticRunnerTests(unittest.TestCase):
    def build_args(self, **overrides):
        defaults = {
            "preset": "smoke",
            "run_name": None,
            "train_num_envs": None,
            "max_iterations": None,
            "eval_num_envs": None,
            "episodes": None,
            "seed": None,
            "humanoid_gym_root": None,
            "rl_device": None,
            "sim_device": None,
            "analysis_root": "artifacts/analysis/sn_replacement_diagnostic",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_default_run_name_includes_preset_and_seed(self) -> None:
        config = {"run_name": "sn_ppo_rough_terrain"}
        args = self.build_args()

        self.assertEqual(
            sn_diagnostic.run_name_for(config, args),
            "sn_ppo_rough_terrain_smoke_seed123145",
        )

    def test_train_command_uses_smoke_budget(self) -> None:
        args = self.build_args()
        command = sn_diagnostic.build_train_command(
            sn_diagnostic.DEFAULT_CONFIG,
            "sn_ppo_rough_terrain_smoke_seed123145",
            args,
        )

        self.assertIn("--num-envs=16", command)
        self.assertIn("--max-iterations=1", command)
        self.assertIn("--seed=123145", command)
        self.assertIn("--config=configs/methods/sn_ppo_rough_terrain.json", command)

    def test_medium_preset_stays_single_seed_diagnostic(self) -> None:
        args = self.build_args(preset="medium")
        command = sn_diagnostic.build_train_command(
            sn_diagnostic.DEFAULT_CONFIG,
            "sn_ppo_rough_terrain_medium_seed123145",
            args,
        )

        self.assertIn("--num-envs=32", command)
        self.assertIn("--max-iterations=100", command)
        self.assertIn("--seed=123145", command)

    def test_evaluate_command_uses_run_name_when_manifest_absent(self) -> None:
        args = self.build_args()
        config = sn_diagnostic.load_config(sn_diagnostic.DEFAULT_CONFIG)
        run_name = "sn_ppo_rough_terrain_missing_manifest"
        command = sn_diagnostic.build_evaluate_command(config, sn_diagnostic.DEFAULT_CONFIG, run_name, args)

        self.assertIn("--load-run", command)
        self.assertEqual(command[command.index("--load-run") + 1], run_name)
        self.assertNotIn("--checkpoints", command)
        self.assertIn("--episodes", command)
        self.assertEqual(command[command.index("--episodes") + 1], "1")


if __name__ == "__main__":
    unittest.main()
