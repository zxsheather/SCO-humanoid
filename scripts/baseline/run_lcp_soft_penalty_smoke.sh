#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/methods/lcp_soft_jacobian_penalty_diagnostic.json}"
RUN_NAME="${RUN_NAME:-lcp_soft_jacobian_penalty_smoke_seed23}"
SEED="${SEED:-23}"
EVAL_SEED="${EVAL_SEED:-123145}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-1}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
EPISODES="${EPISODES:-1}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
ARTIFACT_DIR="artifacts/methods/lcp_soft_jacobian_penalty_diagnostic/${RUN_NAME}"

mkdir -p "${ARTIFACT_DIR}"

write_failure() {
  local stage="$1"
  local exit_code="$2"
  "${PYTHON_BIN}" - "${ARTIFACT_DIR}" "${RUN_NAME}" "${stage}" "${exit_code}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

artifact_dir = Path(sys.argv[1])
artifact_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "status": "failed",
    "run_name": sys.argv[2],
    "stage": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "note": "LCP-style soft penalty smoke failed before writing the expected completion artifact.",
}
(artifact_dir / "lcp_soft_penalty_smoke_failure.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

run_stage() {
  local stage="$1"
  shift
  local marker="$1"
  shift

  "$@"
  local exit_code="$?"
  if [ "${exit_code}" -eq 0 ]; then
    return 0
  fi
  if [ -f "${marker}" ]; then
    echo "Recovered non-zero Isaac exit (${exit_code}) after ${stage}; found ${marker}." >&2
    return 0
  fi
  write_failure "${stage}" "${exit_code}"
  return "${exit_code}"
}

echo "=== LCP-style soft penalty smoke ==="
echo "python: ${PYTHON_BIN}"
echo "config: ${CONFIG}"
echo "run_name: ${RUN_NAME}"

run_stage "train" "${ARTIFACT_DIR}/manifest.json" \
  "${PYTHON_BIN}" scripts/baseline/train_vanilla_ppo.py \
    --config="${CONFIG}" \
    --run-name="${RUN_NAME}" \
    --num-envs="${TRAIN_NUM_ENVS}" \
    --max-iterations="${MAX_ITERATIONS}" \
    --seed="${SEED}" \
    --rl-device="${RL_DEVICE}" \
    --sim-device="${SIM_DEVICE}"
train_exit="$?"
if [ "${train_exit}" -ne 0 ]; then
  exit "${train_exit}"
fi

run_stage "evaluate" "${ARTIFACT_DIR}/metrics.json" \
  "${PYTHON_BIN}" scripts/baseline/evaluate_policy.py \
    --config="${CONFIG}" \
    --run-name="${RUN_NAME}" \
    --num-envs="${EVAL_NUM_ENVS}" \
    --episodes="${EPISODES}" \
    --seed="${EVAL_SEED}" \
    --rl-device="${RL_DEVICE}" \
    --sim-device="${SIM_DEVICE}"
eval_exit="$?"
if [ "${eval_exit}" -ne 0 ]; then
  exit "${eval_exit}"
fi

echo "Wrote ${ARTIFACT_DIR}/manifest.json"
echo "Wrote ${ARTIFACT_DIR}/metrics.json"
