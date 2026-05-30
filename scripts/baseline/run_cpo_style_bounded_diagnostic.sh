#!/usr/bin/env bash
set -u

STAGE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/methods/cpo_style_bounded_diagnostic.json}"
RUN_NAME="${RUN_NAME:-cpo_style_bounded_diagnostic_seed23}"
SEED="${SEED:-23}"
CHECKPOINTS="${CHECKPOINTS:-0,1,2,3}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-4}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
EPISODES="${EPISODES:-2}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/methods/cpo_style_bounded_diagnostic}"
FORCE="${FORCE:-0}"

run_with_recovery() {
  local label="$1"
  shift
  local marker="$1"
  shift

  "$@"
  local exit_code="$?"
  if [ "${exit_code}" -eq 0 ]; then
    return 0
  fi
  if [ -f "${marker}" ]; then
    echo "Recovered non-zero Isaac exit (${exit_code}) after ${label}; found ${marker}." >&2
    return 0
  fi
  echo "${label} failed with exit code ${exit_code}; expected marker missing: ${marker}" >&2
  return "${exit_code}"
}

run_dir_from_manifest() {
  "${PYTHON_BIN}" - "${ARTIFACT_ROOT}/${RUN_NAME}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
print(manifest["run_dir"])
PY
}

train_complete() {
  "${PYTHON_BIN}" - "${ARTIFACT_ROOT}/${RUN_NAME}/manifest.json" "${CHECKPOINTS}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
checkpoints = [int(part) for part in sys.argv[2].split(",") if part]
if not manifest_path.exists():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text())
run_dir = Path(manifest.get("run_dir", ""))
if not run_dir.is_absolute():
    run_dir = Path.cwd() / run_dir
if not run_dir.exists():
    raise SystemExit(1)
missing = [ckpt for ckpt in checkpoints if not (run_dir / f"model_{ckpt}.pt").exists()]
raise SystemExit(0 if not missing else 1)
PY
}

echo "=== CPO-style bounded diagnostic (#83) ==="
echo "stage: ${STAGE}"
echo "python: ${PYTHON_BIN}"
echo "config: ${CONFIG}"
echo "run_name: ${RUN_NAME}"
echo "seed: ${SEED}"
echo "train: ${TRAIN_NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "eval: checkpoints ${CHECKPOINTS}, ${EVAL_NUM_ENVS} envs, ${EPISODES} episodes"
echo "force: ${FORCE}"

mkdir -p "${ARTIFACT_ROOT}/${RUN_NAME}"

if [ "${STAGE}" = "train" ] || [ "${STAGE}" = "all" ]; then
  if [ "${FORCE}" != "1" ] && train_complete; then
    echo "Skipping train; expected checkpoints already exist."
  else
    run_with_recovery "train" "${ARTIFACT_ROOT}/${RUN_NAME}/manifest.json" \
      "${PYTHON_BIN}" scripts/baseline/train_vanilla_ppo.py \
        --config="${CONFIG}" \
        --run-name="${RUN_NAME}" \
        --num-envs="${TRAIN_NUM_ENVS}" \
        --max-iterations="${MAX_ITERATIONS}" \
        --seed="${SEED}" \
        --rl-device="${RL_DEVICE}" \
        --sim-device="${SIM_DEVICE}" || exit "$?"
  fi
fi

if [ "${STAGE}" = "evaluate" ] || [ "${STAGE}" = "all" ]; then
  run_dir="$(run_dir_from_manifest)"
  echo "run_dir: ${run_dir}"
  if [ "${FORCE}" != "1" ] && [ -f "${ARTIFACT_ROOT}/${RUN_NAME}/checkpoint_sweep_summary.json" ]; then
    echo "Skipping checkpoint sweep; summary already exists."
  else
    run_with_recovery "checkpoint sweep" "${ARTIFACT_ROOT}/${RUN_NAME}/checkpoint_sweep_summary.json" \
      "${PYTHON_BIN}" scripts/baseline/evaluate_checkpoint_sweep.py \
        --config="${CONFIG}" \
        --run-name="${RUN_NAME}" \
        --load-run="${run_dir}" \
        --checkpoints="${CHECKPOINTS}" \
        --num-envs="${EVAL_NUM_ENVS}" \
        --episodes="${EPISODES}" \
        --rl-device="${RL_DEVICE}" \
        --sim-device="${SIM_DEVICE}" || exit "$?"
  fi
fi

if [ "${STAGE}" = "summary" ] || [ "${STAGE}" = "all" ]; then
  if [ ! -f "${ARTIFACT_ROOT}/${RUN_NAME}/checkpoint_sweep_summary.json" ]; then
    echo "Missing checkpoint sweep summary: ${ARTIFACT_ROOT}/${RUN_NAME}/checkpoint_sweep_summary.json" >&2
    exit 1
  fi
  "${PYTHON_BIN}" - "${ARTIFACT_ROOT}/${RUN_NAME}/checkpoint_sweep_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
payload = json.loads(summary_path.read_text())
print("selection_status:", payload.get("selection_status"))
print("best_checkpoint:", payload.get("best_checkpoint"))
for row in payload.get("rows", []):
    print(
        "checkpoint={checkpoint} fall={fall_rate} vel={velocity_tracking_error_mean} "
        "jacc={joint_acceleration_l2_mean} jitter={action_jitter_l2_mean} return={episode_return_mean} "
        "eval_sens={eval_policy_local_sensitivity_cost_mean}".format(**row)
    )
PY
fi

echo "=== CPO-style bounded diagnostic complete ==="
