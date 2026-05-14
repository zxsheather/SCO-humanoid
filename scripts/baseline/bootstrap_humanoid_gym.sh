#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/configs/baselines/vanilla_ppo.json"

if ! command -v python >/dev/null 2>&1; then
  echo "python is required" >&2
  exit 1
fi

CHECKOUT_DIR="$(python -c 'import json,sys; from pathlib import Path; cfg=json.load(open(sys.argv[1])); print((Path(sys.argv[2]) / cfg["upstream"]["checkout_dir"]).resolve())' "${CONFIG_PATH}" "${REPO_ROOT}")"
UPSTREAM_URL="$(python -c 'import json,sys; cfg=json.load(open(sys.argv[1])); print(cfg["upstream"]["repo_url"])' "${CONFIG_PATH}")"
UPSTREAM_REF="$(python -c 'import json,sys; cfg=json.load(open(sys.argv[1])); print(cfg["upstream"]["ref"])' "${CONFIG_PATH}")"

if [[ ! -d "${CHECKOUT_DIR}" ]]; then
  git clone "${UPSTREAM_URL}" "${CHECKOUT_DIR}"
fi

git -C "${CHECKOUT_DIR}" fetch --all --tags
git -C "${CHECKOUT_DIR}" checkout "${UPSTREAM_REF}"
python -m pip install -e "${CHECKOUT_DIR}" --no-deps

echo "Humanoid-Gym ready at ${CHECKOUT_DIR}"
echo "Pinned ref: ${UPSTREAM_REF}"
