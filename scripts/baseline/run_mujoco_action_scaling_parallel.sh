#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/methods/action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp.json}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"

TERRAIN_MODE="${TERRAIN_MODE:-isaac_mainline}"
EPISODES="${EPISODES:-20}"
SIM_DURATION="${SIM_DURATION:-20}"
JOINT_RESET_NOISE="${JOINT_RESET_NOISE:-0.1}"
BASE_XY_NOISE="${BASE_XY_NOISE:-0.0}"
COMMAND_VX="${COMMAND_VX:-0.4}"
COMMAND_VY="${COMMAND_VY:-0.0}"
COMMAND_DYAW="${COMMAND_DYAW:-0.0}"
OUTPUT_NAME="${OUTPUT_NAME:-metrics_mujoco_isaac_mainline_20ep_20s_noise01.json}"
MANIFEST_SLOT="${MANIFEST_SLOT:-mujoco_isaac_mainline_20ep_20s_noise01}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/artifacts/analysis/rough_terrain_action_scaling_probe/mujoco_parallel_logs}"

SEEDS=(11 17 23)

declare -A CHECKPOINTS=(
  [11]=400
  [17]=400
  [23]=400
)

declare -A RUN_NAMES=(
  [11]="action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed11"
  [17]="action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed17"
  [23]="action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed23"
)

declare -A LOAD_RUNS=(
  [11]="May23_01-27-22_action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed11"
  [17]="May23_02-15-58_action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed17"
  [23]="May23_03-56-01_action_scaling_threshold_38_quantile_090_pid_lower_bound_clamp_rough_terrain_seed23"
)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python binary not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export PATH="${PYTHON_BIN_DIR}:${PATH}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"

declare -A PIDS=()
declare -A LOG_PATHS=()

cleanup() {
  trap - INT TERM
  echo "received interrupt, stopping child jobs..." >&2
  for seed in "${SEEDS[@]}"; do
    local pid="${PIDS[$seed]:-}"
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
}

trap cleanup INT TERM

run_seed() {
  local seed="$1"
  local checkpoint="${CHECKPOINTS[$seed]}"
  local run_name="${RUN_NAMES[$seed]}"
  local load_run="${LOAD_RUNS[$seed]}"
  local log_path="${LOG_ROOT}/seed${seed}.log"

  echo "[seed${seed}] checkpoint=${checkpoint} load_run=${load_run} log=${log_path}"

  env -u DISPLAY CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    "${PYTHON_BIN}" -u \
    "${REPO_ROOT}/scripts/baseline/evaluate_mujoco_sim2sim.py" \
    --config "${CONFIG_PATH}" \
    --run-name "${run_name}" \
    --load-run "${load_run}" \
    --checkpoint "${checkpoint}" \
    --terrain-mode "${TERRAIN_MODE}" \
    --episodes "${EPISODES}" \
    --sim-duration "${SIM_DURATION}" \
    --joint-reset-noise "${JOINT_RESET_NOISE}" \
    --base-xy-noise "${BASE_XY_NOISE}" \
    --command-vx "${COMMAND_VX}" \
    --command-vy "${COMMAND_VY}" \
    --command-dyaw "${COMMAND_DYAW}" \
    --output-name "${OUTPUT_NAME}" \
    --manifest-slot "${MANIFEST_SLOT}" \
    2>&1 | tee "${log_path}"
}

echo "=== Action Scaling MuJoCo Replay ==="
echo "config_path=${CONFIG_PATH}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES_VALUE}"
echo "log_root=${LOG_ROOT}"

for seed in "${SEEDS[@]}"; do
  LOG_PATHS["$seed"]="${LOG_ROOT}/seed${seed}.log"
  run_seed "${seed}" &
  PIDS["$seed"]=$!
done

failures=0
for seed in "${SEEDS[@]}"; do
  pid="${PIDS[$seed]}"
  if wait "${pid}"; then
    echo "[seed${seed}] completed"
  else
    exit_code=$?
    echo "[seed${seed}] failed with exit code ${exit_code}" >&2
    echo "[seed${seed}] log: ${LOG_PATHS[$seed]}" >&2
    failures=1
  fi
done

if (( failures != 0 )); then
  echo "one or more seed jobs failed" >&2
  exit 1
fi

echo "all three seed jobs completed"
for seed in "${SEEDS[@]}"; do
  echo "[seed${seed}] log: ${LOG_PATHS[$seed]}"
done
