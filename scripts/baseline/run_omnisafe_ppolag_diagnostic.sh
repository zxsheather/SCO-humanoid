#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# OmniSafe PPO-Lag bounded diagnostic (#63)
# Runs seeds {23,29,31} with the approved Jacobian cost bridge.
#
# Usage:
#   ./scripts/baseline/run_omnisafe_ppolag_diagnostic.sh [train|evaluate|summary|all|plan]
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python}"
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/sweeps/rough_terrain_omnisafe_ppolag_diagnostic.json}"
STAGE="${1:-all}"

if [[ ! -f "${SWEEP_CONFIG}" ]]; then
  echo "sweep config not found: ${SWEEP_CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${PYTHON_BIN}" ]]; then
  echo "python binary not found: ${PYTHON_BIN}" >&2
  exit 1
fi

export PYTHON_BIN

echo "=== OmniSafe PPO-Lag bounded diagnostic ==="
echo "sweep config : ${SWEEP_CONFIG}"
echo "python       : ${PYTHON_BIN}"
echo "stage        : ${STAGE}"
echo "seeds        : 23 / 29 / 31"
echo ""

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/baseline/run_omnisafe_ppolag_diagnostic.py \
  --sweep-config "${SWEEP_CONFIG}" \
  --stage "${STAGE}" \
  --skip-completed
