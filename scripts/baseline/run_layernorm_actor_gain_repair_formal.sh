#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export SWEEP_CONFIG="${SWEEP_CONFIG:-${REPO_ROOT}/configs/sweeps/rough_terrain_layernorm_actor_gain_repair_probe.json}"

exec "${SCRIPT_DIR}/run_layernorm_actor_formal.sh" "$@"
