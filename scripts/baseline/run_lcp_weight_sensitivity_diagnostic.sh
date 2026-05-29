#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python}"
SEEDS="${SEEDS:-23 29 31}"
CHECKPOINTS="${CHECKPOINTS:-0,100,200,300,400}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-512}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-32}"
MAX_ITERATIONS="${MAX_ITERATIONS:-400}"
EPISODES="${EPISODES:-20}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/methods/lcp_soft_jacobian_penalty_weight_sensitivity}"
SUMMARY_ROOT="${SUMMARY_ROOT:-artifacts/analysis/rough_terrain_lcp_weight_sensitivity_diagnostic}"
ISSUE_ID="${ISSUE_ID:-#73}"

run_weight() {
  local weight_id="$1"
  local config="$2"
  local run_prefix="$3"
  local summary_dir="${SUMMARY_ROOT}/${weight_id}"

  echo
  echo "=== LCP weight sensitivity ${weight_id} (${ISSUE_ID}) ==="
  PYTHON_BIN="${PYTHON_BIN}" \
  CONFIG="${config}" \
  RUN_PREFIX="${run_prefix}" \
  SEEDS="${SEEDS}" \
  CHECKPOINTS="${CHECKPOINTS}" \
  TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS}" \
  EVAL_NUM_ENVS="${EVAL_NUM_ENVS}" \
  MAX_ITERATIONS="${MAX_ITERATIONS}" \
  EPISODES="${EPISODES}" \
  RL_DEVICE="${RL_DEVICE}" \
  SIM_DEVICE="${SIM_DEVICE}" \
  ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  SUMMARY_DIR="${summary_dir}" \
  ISSUE_ID="${ISSUE_ID}" \
  RUN_LABEL="LCP weight sensitivity ${weight_id}" \
    "${REPO_ROOT}/scripts/baseline/run_lcp_soft_penalty_diagnostic.sh"
}

echo "repo_root=${REPO_ROOT}"
echo "python_bin=${PYTHON_BIN}"
echo "seeds=${SEEDS}"
echo "checkpoints=${CHECKPOINTS}"
echo "train=${TRAIN_NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "eval=${EVAL_NUM_ENVS} envs x ${EPISODES} episodes per checkpoint"
echo "artifact_root=${ARTIFACT_ROOT}"
echo "summary_root=${SUMMARY_ROOT}"

PYTHON_BIN="${PYTHON_BIN}" run_weight \
  "w0001" \
  "${REPO_ROOT}/configs/methods/lcp_soft_jacobian_penalty_weight_0001_diagnostic.json" \
  "lcp_soft_jacobian_penalty_weight_0001_diagnostic"

PYTHON_BIN="${PYTHON_BIN}" run_weight \
  "w0004" \
  "${REPO_ROOT}/configs/methods/lcp_soft_jacobian_penalty_weight_0004_diagnostic.json" \
  "lcp_soft_jacobian_penalty_weight_0004_diagnostic"

echo "=== LCP weight sensitivity diagnostic complete ==="
