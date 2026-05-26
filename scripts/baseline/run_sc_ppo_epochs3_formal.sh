#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python}"
PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/sweeps/rough_terrain_sc_ppo_epochs3_probe.json}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
CXX_BIN="${CXX_BIN:-${PYTHON_BIN_DIR}/x86_64-conda-linux-gnu-c++}"
CC_BIN="${CC_BIN:-${PYTHON_BIN_DIR}/x86_64-conda-linux-gnu-cc}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python binary not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${SWEEP_CONFIG}" ]]; then
  echo "sweep config not found: ${SWEEP_CONFIG}" >&2
  exit 1
fi

export PATH="${PYTHON_BIN_DIR}:${PATH}"
export CXX="${CXX_BIN}"
export CC="${CC_BIN}"

exec "${PYTHON_BIN}" -u \
  "${REPO_ROOT}/scripts/baseline/run_formal_comparison.py" \
  --sweep-config "${SWEEP_CONFIG}" \
  --stage all \
  --skip-completed \
  --rl-device "${RL_DEVICE}" \
  --sim-device "${SIM_DEVICE}" \
  "$@"
