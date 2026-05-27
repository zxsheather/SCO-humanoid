#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# Extended-seed robustness matrix (#51)
# Runs SC-PPO 3.8 and revised heuristic anchor with 5 seeds
# (11/17/23 canonical + 29/31 added).
#
# Usage:
#   ./run_extended_seeds_formal.sh [train|evaluate|all]
#
# The historical 3-seed record is preserved; added seeds are
# recorded under a fresh sweep name and analysis root.
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python}"
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/sweeps/rough_terrain_extended_seeds.json}"
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

echo "=== Extended-seed robustness matrix ==="
echo "sweep config : ${SWEEP_CONFIG}"
echo "python       : ${PYTHON_BIN}"
echo "stage        : ${STAGE}"
echo ""

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/baseline/run_formal_comparison.py \
  --sweep-config "${SWEEP_CONFIG}" \
  --stage "${STAGE}" \
  --skip-completed
