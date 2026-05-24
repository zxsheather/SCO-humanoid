from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import run_orthogonal_actor_diagnostic as orthogonal_actor_diagnostic  # noqa: E402


class OrthogonalActorDiagnosticRunnerTests(unittest.TestCase):
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
            "analysis_root": "artifacts/analysis/orthogonal_actor_diagnostic",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_default_run_name_includes_preset_and_seed(self) -> None:
        config = {"run_name": "orthogonal_actor_output_gain_0500_formal_probe_rough_terrain"}
        args = self.build_args()

        self.assertEqual(
            orthogonal_actor_diagnostic.run_name_for(config, args),
            "orthogonal_actor_output_gain_0500_formal_probe_rough_terrain_smoke_seed123145",
        )

    def test_train_command_uses_smoke_budget(self) -> None:
        args = self.build_args()
        command = orthogonal_actor_diagnostic.build_train_command(
            orthogonal_actor_diagnostic.DEFAULT_CONFIG,
            "orthogonal_actor_output_gain_0500_formal_probe_rough_terrain_smoke_seed123145",
            args,
        )

        self.assertIn("--num-envs=16", command)
        self.assertIn("--max-iterations=1", command)
        self.assertIn("--seed=123145", command)
        self.assertIn(
            "--config=configs/methods/orthogonal_actor_output_gain_0500_formal_probe.json",
            command,
        )

    def test_evaluate_command_uses_run_name_when_manifest_absent(self) -> None:
        args = self.build_args()
        config = orthogonal_actor_diagnostic.load_config(orthogonal_actor_diagnostic.DEFAULT_CONFIG)
        run_name = "orthogonal_actor_output_gain_0500_formal_probe_missing_manifest"
        command = orthogonal_actor_diagnostic.build_evaluate_command(
            config,
            orthogonal_actor_diagnostic.DEFAULT_CONFIG,
            run_name,
            args,
        )

        self.assertIn("--load-run", command)
        self.assertEqual(command[command.index("--load-run") + 1], run_name)
        self.assertNotIn("--checkpoints", command)
        self.assertIn("--episodes", command)
        self.assertEqual(command[command.index("--episodes") + 1], "1")


if __name__ == "__main__":
    unittest.main()
