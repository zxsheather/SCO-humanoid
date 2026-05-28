from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

try:
    import _omnisafe_bridge as bridge  # noqa: E402
except ModuleNotFoundError:
    bridge = None


class LinearActor:
    def __init__(self) -> None:
        self.weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return obs @ self.weight.t()


@unittest.skipIf(bridge is None, "OmniSafe is not installed in this Python environment.")
class OmniSafeBridgeTests(unittest.TestCase):
    def test_jacobian_cost_records_violation_rate(self) -> None:
        obs = torch.ones(4, 2)

        result = bridge.compute_jacobian_cost(
            LinearActor(),
            obs,
            threshold=2.0,
            subsample_obs=0,
            cost_aggregation="mean",
        )

        self.assertAlmostEqual(result["cost_for_update"], 5.0**0.5, places=5)
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["violation_rate"], 1.0)
        self.assertEqual(len(result["per_sample_costs"]), 4)

    def test_lagrange_update_uses_threshold_error(self) -> None:
        lagrange = bridge._create_lagrange(
            cost_limit=3.8,
            lagrangian_multiplier_init=0.5,
            lambda_lr=0.01,
            lambda_optimizer="SGD",
            lagrangian_upper_bound=5.0,
        )

        result = bridge.update_omnisafe_multiplier(lagrange, cost_update=0.3, threshold=3.8)

        self.assertLess(result.multiplier, result.multiplier_before)
        self.assertAlmostEqual(result.constraint_error, -3.5, places=5)


if __name__ == "__main__":
    unittest.main()
