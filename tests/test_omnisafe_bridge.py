from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "scripts" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

try:
    import _omnisafe_bridge as bridge  # noqa: E402
    import _omnisafe_policy_loader as policy_loader  # noqa: E402
    import _omnisafe_ppolag_jacobian_hook as ppolag_hook  # noqa: E402
except ModuleNotFoundError:
    bridge = None
    policy_loader = None
    ppolag_hook = None


class LinearActor:
    def __init__(self) -> None:
        self.weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return obs @ self.weight.t()

    def parameters(self):
        return []


class LinearOmniSafeStyleActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))

    def forward(self, obs: torch.Tensor):
        return torch.distributions.Normal(self.linear(obs), torch.ones(2, device=obs.device))


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

    @unittest.skipIf(ppolag_hook is None, "OmniSafe PPO-Lag hook module is unavailable.")
    def test_ppolag_hook_cost_is_differentiable_to_actor_params(self) -> None:
        actor = LinearOmniSafeStyleActor()
        obs = torch.ones(4, 2)

        cost = ppolag_hook.compute_differentiable_jacobian_cost(
            actor,
            obs,
            threshold=2.0,
            subsample_obs=0,
            cost_aggregation="mean",
        )

        self.assertAlmostEqual(cost.cost_for_update, 5.0**0.5, places=5)
        self.assertEqual(cost.violation_rate, 1.0)
        cost.cost_tensor.backward()
        self.assertIsNotNone(actor.linear.weight.grad)
        self.assertGreater(float(actor.linear.weight.grad.abs().sum().item()), 0.0)

    @unittest.skipIf(policy_loader is None, "OmniSafe policy loader module is unavailable.")
    def test_omnisafe_policy_checkpoint_roundtrip(self) -> None:
        actor = policy_loader.create_gaussian_actor(obs_dim=3, act_dim=2, hidden_sizes=[4], device="cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model_0.pt"
            policy_loader.save_omnisafe_policy_checkpoint(
                checkpoint_path,
                actor,
                checkpoint=0,
                seed=23,
                cost_config={"threshold": 3.8},
            )
            policy, metadata = policy_loader.load_omnisafe_policy_checkpoint(checkpoint_path, device="cpu")

        action = policy(torch.zeros(1, 3))

        self.assertEqual(list(action.shape), [1, 2])
        self.assertEqual(metadata["checkpoint"], 0)
        self.assertEqual(metadata["seed"], 23)
        self.assertEqual(metadata["cost_config"]["threshold"], 3.8)


if __name__ == "__main__":
    unittest.main()
