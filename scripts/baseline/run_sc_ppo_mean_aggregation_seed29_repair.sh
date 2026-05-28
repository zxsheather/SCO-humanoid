#!/usr/bin/env bash
# Bounded SC-PPO mean-aggregation repair diagnostic (#57).
#
# Usage:
#   ./scripts/baseline/run_sc_ppo_mean_aggregation_seed29_repair.sh [train|evaluate|all|plan]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python}"
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/sweeps/sc_ppo_mean_aggregation_seed29_repair.json}"
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

echo "=== SC-PPO mean-aggregation seed29 repair diagnostic ==="
echo "sweep config : ${SWEEP_CONFIG}"
echo "python       : ${PYTHON_BIN}"
echo "stage        : ${STAGE}"
echo ""

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/baseline/run_formal_comparison.py \
  --sweep-config "${SWEEP_CONFIG}" \
  --stage "${STAGE}" \
  --skip-completed
