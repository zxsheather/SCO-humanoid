from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import _common as baseline_common  # noqa: E402


class BaselineCommonTests(unittest.TestCase):
    def test_load_config_resolves_repo_relative_paths_independent_of_cwd(self) -> None:
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                config = baseline_common.load_config("configs/methods/layernorm_actor_output_gain_0750_formal_probe.json")
        finally:
            os.chdir(original_cwd)

        self.assertEqual(config["run_name"], "layernorm_actor_output_gain_0750_formal_probe_rough_terrain")

    def test_resolve_run_dir_accepts_repo_relative_load_run_with_logs_prefix(self) -> None:
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
