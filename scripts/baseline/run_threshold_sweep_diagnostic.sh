#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# SC-PPO threshold sweep diagnostic (#58 / Path B)
# Sweeps thresholds 3.6 / 3.7 / 4.0 against baseline 3.8
# on diagnostic seeds {23, 29, 31}.
#
# This is a bounded diagnostic. It does NOT replace the #51
# five-seed record. A full {11,17,23,29,31} rerun is required
# before any replacement claim if a threshold passes the gates.
#
# Usage:
#   ./run_threshold_sweep_diagnostic.sh [train|evaluate|all]
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python}"
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/sweeps/rough_terrain_threshold_sweep_diagnostic.json}"
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

echo "=== SC-PPO threshold sweep diagnostic ==="
echo "sweep config : ${SWEEP_CONFIG}"
echo "python       : ${PYTHON_BIN}"
echo "stage        : ${STAGE}"
echo "thresholds   : 3.6 / 3.7 / 4.0 (baseline 3.8)"
echo "seeds        : 23 / 29 / 31"
echo ""

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/baseline/run_formal_comparison.py \
  --sweep-config "${SWEEP_CONFIG}" \
  --stage "${STAGE}" \
  --skip-completed
