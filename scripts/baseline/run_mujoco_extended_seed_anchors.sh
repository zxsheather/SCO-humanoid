#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python}"
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
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/artifacts/analysis/rough_terrain_extended_seeds/mujoco_added_seed_logs}"
mkdir -p "${LOG_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python binary not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export PATH="${PYTHON_BIN_DIR}:${PATH}"

run_replay() {
  local label="$1"
  local seed="$2"
  local config_path="$3"
  local artifact_root="$4"
  local run_name="$5"
  local load_run="$6"
  local checkpoint="$7"
  local log_path="${LOG_ROOT}/${label}_seed${seed}.log"
  local output_path="${artifact_root}/${run_name}/${OUTPUT_NAME}"
  local load_run_name

  load_run_name="$(basename "${load_run}")"

  echo "[${label} seed${seed}] checkpoint=${checkpoint} load_run=${load_run_name}"
  echo "[${label} seed${seed}] log=${log_path}"

  if [[ ! -f "${config_path}" ]]; then
    echo "config not found: ${config_path}" >&2
    return 1
  fi

  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_path}" ]]; then
    echo "[${label} seed${seed}] skipping existing metrics: ${output_path}"
    return 0
  fi

  set +e
  env -u DISPLAY CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    "${PYTHON_BIN}" -u \
    "${REPO_ROOT}/scripts/baseline/evaluate_mujoco_sim2sim.py" \
    --config "${config_path}" \
    --run-name "${run_name}" \
    --load-run "${load_run_name}" \
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
  local exit_code="${PIPESTATUS[0]}"
  set -e

  if [[ "${exit_code}" -eq 0 ]]; then
    return 0
  fi
  if [[ -f "${output_path}" ]]; then
    echo "[${label} seed${seed}] recovered non-zero MuJoCo exit (${exit_code}); found ${output_path}" >&2
    return 0
  fi
  return "${exit_code}"
}

SC_CONFIG="${REPO_ROOT}/configs/methods/sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_extended_seeds.json"
SC_ROOT="${REPO_ROOT}/artifacts/methods/sc_ppo_pid_probe"
HEURISTIC_CONFIG="${REPO_ROOT}/configs/methods/heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget.json"
HEURISTIC_ROOT="${REPO_ROOT}/artifacts/methods/heuristic_smoothing_formal_protocol_revision_long_budget"

echo "repo_root=${REPO_ROOT}"
echo "python_bin=${PYTHON_BIN}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES_VALUE}"
echo "terrain_mode=${TERRAIN_MODE} episodes=${EPISODES} sim_duration=${SIM_DURATION}"
echo "joint_reset_noise=${JOINT_RESET_NOISE} skip_completed=${SKIP_COMPLETED}"
echo "log_root=${LOG_ROOT}"

run_replay \
  "scppo38" \
  "29" \
  "${SC_CONFIG}" \
  "${SC_ROOT}" \
  "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed29" \
  ".external/humanoid-gym/logs/ecolab_sc_ppo_pid_probe/May27_14-44-25_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed29" \
  "400"

run_replay \
  "scppo38" \
  "31" \
  "${SC_CONFIG}" \
  "${SC_ROOT}" \
  "sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed31" \
  ".external/humanoid-gym/logs/ecolab_sc_ppo_pid_probe/May27_15-12-58_sc_ppo_threshold_38_lambda_05_quantile_090_pid_lower_bound_clamp_rough_terrain_iter400_seed31" \
  "400"

run_replay \
  "heuristic" \
  "29" \
  "${HEURISTIC_CONFIG}" \
  "${HEURISTIC_ROOT}" \
  "heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed29" \
  ".external/humanoid-gym/logs/ecolab_heuristic_smoothing_formal_protocol_revision_long_budget/May27_15-42-13_heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed29" \
  "400"

run_replay \
  "heuristic" \
  "31" \
  "${HEURISTIC_CONFIG}" \
  "${HEURISTIC_ROOT}" \
  "heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed31" \
  ".external/humanoid-gym/logs/ecolab_heuristic_smoothing_formal_protocol_revision_long_budget/May27_16-14-33_heuristic_smoothing_action_rate_0050_formal_protocol_revision_long_budget_rough_terrain_seed31" \
  "400"

echo "matched added-seed MuJoCo anchor replay complete"
