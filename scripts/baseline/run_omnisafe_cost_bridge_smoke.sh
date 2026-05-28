#!/usr/bin/env bash
# OmniSafe cost bridge smoke test (#61)
# Verifies Jacobian cost → OmniSafe Lagrange multiplier bridge.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/methods/omnisafe_ppolag_cost_bridge_smoke.json}"

echo "=== OmniSafe cost bridge smoke (#61) ==="
echo "config: ${CONFIG}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/baseline/run_omnisafe_cost_bridge_smoke.py --config "${CONFIG}"
