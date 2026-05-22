from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import evaluate_policy  # noqa: E402


class EvaluatePolicyConstraintMetricsTests(unittest.TestCase):
    def test_action_rate_constraint_metrics_use_generic_keys(self) -> None:
        metrics = evaluate_policy.build_constraint_metrics(
            config={
                "overrides": {
                    "train": {
                        "algorithm.constraint.objective": "action_rate",
                        "algorithm.constraint.threshold": 0.2,
                    }
                }
            },
            constraint_cfg={
                "collect_local_sensitivity": False,
                "local_sensitivity_threshold": None,
            },
            sidecar_metrics={
                "constraint_objective": "action_rate",
                "constraint_cost_update": 0.31,
                "lagrange_multiplier": 0.9,
            },
            multiplier_trace_path=REPO_ROOT / "artifacts" / "missing-trace.json",
            local_sensitivity_samples=[],
            action_rate_samples=[0.1, 0.3, 0.5],
            sidecar_path=REPO_ROOT / "artifacts" / "missing-sidecar.json",
        )

        self.assertTrue(metrics["supported"])
        self.assertEqual(metrics["constraint_objective"], "action_rate")
        self.assertAlmostEqual(metrics["constraint_cost_mean"], 0.3)
        self.assertAlmostEqual(metrics["constraint_cost_std"], 0.1632993161855452)
        self.assertEqual(metrics["constraint_sample_count"], 3)
        self.assertAlmostEqual(metrics["constraint_violation_rate"], 2.0 / 3.0)
        self.assertEqual(metrics["constraint_threshold"], 0.2)
        self.assertAlmostEqual(metrics["action_rate_cost_mean"], 0.3)
        self.assertEqual(metrics["action_rate_sample_count"], 3)
        self.assertEqual(metrics["action_rate_threshold"], 0.2)
        self.assertEqual(metrics["constraint_cost_update"], 0.31)
        self.assertEqual(metrics["lagrange_multiplier"], 0.9)
        self.assertIsNone(metrics["policy_local_sensitivity_cost_mean"])

    def test_local_sensitivity_constraint_metrics_still_use_eval_samples(self) -> None:
        metrics = evaluate_policy.build_constraint_metrics(
            config={
                "overrides": {"train": {}},
                "evaluation": {
                    "constraint_logging": {
                        "collect_local_sensitivity": True,
                    }
                },
            },
            constraint_cfg={
                "collect_local_sensitivity": True,
                "local_sensitivity_threshold": 3.8,
            },
            sidecar_metrics=None,
            multiplier_trace_path=None,
            local_sensitivity_samples=[3.0, 4.0],
            action_rate_samples=[0.1, 0.2],
            sidecar_path=None,
        )

        self.assertTrue(metrics["supported"])
        self.assertEqual(metrics["constraint_objective"], "policy_local_sensitivity")
        self.assertAlmostEqual(metrics["constraint_cost_mean"], 3.5)
        self.assertAlmostEqual(metrics["policy_local_sensitivity_cost_mean"], 3.5)
        self.assertEqual(metrics["policy_local_sensitivity_sample_count"], 2)
        self.assertEqual(metrics["constraint_threshold"], 3.8)
        self.assertEqual(metrics["local_sensitivity_threshold"], 3.8)
        self.assertAlmostEqual(metrics["constraint_violation_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
