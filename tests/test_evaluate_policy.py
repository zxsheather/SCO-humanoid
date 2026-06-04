from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import evaluate_policy  # noqa: E402


class EvaluatePolicyTests(unittest.TestCase):
    def test_apply_evaluation_seed_updates_env_and_train_cfg(self) -> None:
        env_cfg = SimpleNamespace(seed=5)
        train_cfg = SimpleNamespace(seed=5)

        evaluate_policy.apply_evaluation_seed(env_cfg, train_cfg, 23)

        self.assertEqual(env_cfg.seed, 23)
        self.assertEqual(train_cfg.seed, 23)

    def test_apply_evaluation_seed_is_noop_when_seed_is_none(self) -> None:
        env_cfg = SimpleNamespace(seed=5)
        train_cfg = SimpleNamespace(seed=7)

        evaluate_policy.apply_evaluation_seed(env_cfg, train_cfg, None)

        self.assertEqual(env_cfg.seed, 5)
        self.assertEqual(train_cfg.seed, 7)


if __name__ == "__main__":
    unittest.main()
