#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/methods/lcp_soft_jacobian_penalty_diagnostic.json}"
RUN_PREFIX="${RUN_PREFIX:-lcp_soft_jacobian_penalty_diagnostic}"
SEEDS="${SEEDS:-23 29 31}"
CHECKPOINTS="${CHECKPOINTS:-0,100,200,300,400}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-512}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-32}"
MAX_ITERATIONS="${MAX_ITERATIONS:-400}"
EPISODES="${EPISODES:-20}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/methods/lcp_soft_jacobian_penalty_diagnostic}"
SUMMARY_DIR="${SUMMARY_DIR:-artifacts/analysis/rough_terrain_lcp_soft_jacobian_diagnostic}"

mkdir -p "${SUMMARY_DIR}/logs"

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

train_complete() {
  local run_name="$1"
  local checkpoint_csv="$2"
  "${PYTHON_BIN}" - "${ARTIFACT_ROOT}/${run_name}/manifest.json" "${checkpoint_csv}" <<'PY'
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

run_dir_from_manifest() {
  local run_name="$1"
  "${PYTHON_BIN}" - "${ARTIFACT_ROOT}/${run_name}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
print(manifest["run_dir"])
PY
}

echo "=== LCP-style three-seed diagnostic (#68) ==="
echo "python: ${PYTHON_BIN}"
echo "config: ${CONFIG}"
echo "seeds: ${SEEDS}"
echo "checkpoints: ${CHECKPOINTS}"
echo "train: ${TRAIN_NUM_ENVS} envs x ${MAX_ITERATIONS} iterations"
echo "eval: ${EVAL_NUM_ENVS} envs x ${EPISODES} episodes per checkpoint"

for seed in ${SEEDS}; do
  run_name="${RUN_PREFIX}_seed${seed}"
  artifact_dir="${ARTIFACT_ROOT}/${run_name}"
  mkdir -p "${artifact_dir}"

  echo
  echo "=== seed ${seed}: train ${run_name} ==="
  if train_complete "${run_name}" "${CHECKPOINTS}"; then
    echo "Skipping train for ${run_name}; expected checkpoints already exist."
  else
    run_with_recovery "train seed ${seed}" "${artifact_dir}/manifest.json" \
      "${PYTHON_BIN}" scripts/baseline/train_vanilla_ppo.py \
        --config="${CONFIG}" \
        --run-name="${run_name}" \
        --num-envs="${TRAIN_NUM_ENVS}" \
        --max-iterations="${MAX_ITERATIONS}" \
        --seed="${seed}" \
        --rl-device="${RL_DEVICE}" \
        --sim-device="${SIM_DEVICE}" || exit "$?"
  fi

  run_dir="$(run_dir_from_manifest "${run_name}")"
  echo "run_dir: ${run_dir}"

  echo "=== seed ${seed}: checkpoint sweep ==="
  if [ -f "${artifact_dir}/checkpoint_sweep_summary.json" ]; then
    echo "Skipping checkpoint sweep for ${run_name}; summary already exists."
  else
    run_with_recovery "checkpoint sweep seed ${seed}" "${artifact_dir}/checkpoint_sweep_summary.json" \
      "${PYTHON_BIN}" scripts/baseline/evaluate_checkpoint_sweep.py \
        --config="${CONFIG}" \
        --run-name="${run_name}" \
        --load-run="${run_dir}" \
        --checkpoints="${CHECKPOINTS}" \
        --num-envs="${EVAL_NUM_ENVS}" \
        --episodes="${EPISODES}" \
        --rl-device="${RL_DEVICE}" \
        --sim-device="${SIM_DEVICE}" || exit "$?"
  fi
done

"${PYTHON_BIN}" - "${ARTIFACT_ROOT}" "${SUMMARY_DIR}" "${RUN_PREFIX}" "${SEEDS}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

artifact_root = Path(sys.argv[1])
summary_dir = Path(sys.argv[2])
run_prefix = sys.argv[3]
seeds = [int(part) for part in sys.argv[4].split()]
metric_keys = [
    "fall_rate",
    "velocity_tracking_error_mean",
    "joint_acceleration_l2_mean",
    "action_jitter_l2_mean",
    "episode_return_mean",
    "eval_policy_local_sensitivity_cost_mean",
    "eval_constraint_violation_rate",
]


def aggregate(rows):
    result = {}
    for key in metric_keys:
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        result[key] = statistics.fmean(values) if values else None
    return result


per_seed = []
selected_rows = []
final_rows = []
for seed in seeds:
    run_name = f"{run_prefix}_seed{seed}"
    summary_path = artifact_root / run_name / "checkpoint_sweep_summary.json"
    payload = json.loads(summary_path.read_text())
    rows = payload["rows"]
    selected_checkpoint = int(payload["best_checkpoint"])
    selected_row = next(row for row in rows if int(row["checkpoint"]) == selected_checkpoint)
    final_row = next((row for row in rows if int(row["checkpoint"]) == 400), None)
    per_seed.append(
        {
            "seed": seed,
            "run_name": run_name,
            "selection_status": payload.get("selection_status"),
            "selected_checkpoint": selected_checkpoint,
            "selected": selected_row,
            "final": final_row,
            "summary_path": str(summary_path),
        }
    )
    selected_rows.append(selected_row)
    if final_row is not None:
        final_rows.append(final_row)

comparison_summary = {
    "method": "lcp_soft_jacobian_penalty",
    "issue": "#68",
    "seeds": seeds,
    "selected_aggregate": aggregate(selected_rows),
    "final_aggregate": aggregate(final_rows),
    "per_seed": per_seed,
    "promotion_read": {
        "all_selected_task_valid": all(row.get("fall_rate") is not None and float(row["fall_rate"]) < 1.0 for row in selected_rows),
        "all_final_task_valid": all(row.get("fall_rate") is not None and float(row["fall_rate"]) < 1.0 for row in final_rows),
        "collapsed_seed_count": sum(1 for row in selected_rows if row.get("fall_rate") is not None and float(row["fall_rate"]) >= 1.0),
    },
}
summary_dir.mkdir(parents=True, exist_ok=True)
out = summary_dir / "comparison_summary.json"
out.write_text(json.dumps(comparison_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote {out}")
PY

echo "=== LCP-style diagnostic complete ==="
