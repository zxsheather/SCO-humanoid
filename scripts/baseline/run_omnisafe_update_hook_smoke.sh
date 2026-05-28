#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/TinyNAS2024/zhuoxiang/sco-humanoid/bin/python}"
CONFIG="${CONFIG:-configs/methods/omnisafe_ppolag_update_hook_smoke.json}"
RUN_NAME=""
ARTIFACTS_ROOT=""
ORIGINAL_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config=*)
      CONFIG="${1#--config=}"
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config" >&2
        exit 2
      fi
      CONFIG="$2"
      shift
      ;;
    --run-name=*)
      RUN_NAME="${1#--run-name=}"
      ;;
    --run-name)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --run-name" >&2
        exit 2
      fi
      RUN_NAME="$2"
      shift
      ;;
    --artifacts-root=*)
      ARTIFACTS_ROOT="${1#--artifacts-root=}"
      ;;
    --artifacts-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --artifacts-root" >&2
        exit 2
      fi
      ARTIFACTS_ROOT="$2"
      shift
      ;;
  esac
  shift
done

read_defaults="$("${PYTHON_BIN}" - "${REPO_ROOT}" "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
config = Path(sys.argv[2])
if not config.is_absolute():
    config = repo / config
payload = json.loads(config.read_text())
print(payload["run_name"])
print(payload["artifacts_root"])
PY
)"

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="$(printf '%s\n' "${read_defaults}" | sed -n '1p')"
fi
if [[ -z "${ARTIFACTS_ROOT}" ]]; then
  ARTIFACTS_ROOT="$(printf '%s\n' "${read_defaults}" | sed -n '2p')"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_omnisafe_update_hook_smoke.py" --config="${CONFIG}" "${ORIGINAL_ARGS[@]}"
status=$?
if [[ ${status} -eq 0 ]]; then
  exit 0
fi

artifact_status="$("${PYTHON_BIN}" - "${REPO_ROOT}" "${ARTIFACTS_ROOT}" "${RUN_NAME}" <<'PY'
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
artifacts_root = Path(sys.argv[2])
run_name = sys.argv[3]
if not artifacts_root.is_absolute():
    artifacts_root = repo / artifacts_root
artifact = artifacts_root / run_name / "omnisafe_update_hook_smoke.json"
if not artifact.exists():
    print("missing")
    raise SystemExit(0)
try:
    payload = json.loads(artifact.read_text())
except json.JSONDecodeError:
    print("invalid_json")
    raise SystemExit(0)
print(payload.get("status", "missing_status"))
PY
)"

if [[ "${artifact_status}" == "complete" ]]; then
  echo "Recovered non-zero Isaac exit (${status}) because update-hook smoke artifact is complete." >&2
  exit 0
fi

exit "${status}"
