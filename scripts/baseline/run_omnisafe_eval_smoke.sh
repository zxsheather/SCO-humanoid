#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python}"
CONFIG="${CONFIG:-configs/methods/omnisafe_ppolag_eval_smoke.json}"
RUN_NAME="${RUN_NAME:-omnisafe_ppolag_eval_smoke_seed23}"
CHECKPOINTS="${CHECKPOINTS:-0,100}"
COMMON_ARGS=(
  --config="${CONFIG}"
  --run-name="${RUN_NAME}"
  --num-envs=1
  --episodes=1
  --seed=123145
  --rl-device=cuda:0
  --sim-device=cuda:0
  --create-fixture-checkpoint
)

artifact_status() {
  local run_name="$1"
  local checkpoint="$2"
  "${PYTHON_BIN}" - "${REPO_ROOT}" "${CONFIG}" "${run_name}" "${checkpoint}" <<'PY'
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
config = Path(sys.argv[2])
run_name = sys.argv[3]
checkpoint = int(sys.argv[4])
if not config.is_absolute():
    config = repo / config
payload = json.loads(config.read_text())
artifact = repo / payload["artifacts_root"] / run_name / "omnisafe_evaluation.json"
if not artifact.exists():
    print("missing")
    raise SystemExit(0)
try:
    payload = json.loads(artifact.read_text())
except json.JSONDecodeError:
    print("invalid_json")
    raise SystemExit(0)
if int(payload.get("checkpoint", -1)) != checkpoint:
    print("checkpoint_mismatch")
    raise SystemExit(0)
status = payload.get("status", "missing_status")
print(status)
PY
}

IFS=',' read -ra checkpoint_list <<< "${CHECKPOINTS}"
for checkpoint in "${checkpoint_list[@]}"; do
  checkpoint="${checkpoint// /}"
  [[ -z "${checkpoint}" ]] && continue
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_omnisafe_policy.py" "${COMMON_ARGS[@]}" --checkpoint="${checkpoint}"
  status=$?
  if [[ ${status} -ne 0 ]]; then
    complete_status="$(artifact_status "${RUN_NAME}" "${checkpoint}")"
    if [[ "${complete_status}" == "complete" ]]; then
      echo "Recovered non-zero Isaac exit (${status}) for checkpoint ${checkpoint} because evaluation artifact is complete." >&2
    else
      exit "${status}"
    fi
  fi
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_omnisafe_checkpoint_sweep.py" \
  --config="${CONFIG}" \
  --run-name="${RUN_NAME}" \
  --checkpoints="${CHECKPOINTS}"
