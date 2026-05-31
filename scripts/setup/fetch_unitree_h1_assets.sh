#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/.local/vendor/unitree_rl_gym"
DEST_DIR="${ROOT_DIR}/.external/humanoid-gym/resources/robots/h1"
REPO_URL="https://github.com/unitreerobotics/unitree_rl_gym.git"

mkdir -p "${ROOT_DIR}/.local/vendor" "${DEST_DIR}"

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  git clone --depth 1 --filter=blob:none --no-checkout "${REPO_URL}" "${VENDOR_DIR}"
fi
git -C "${VENDOR_DIR}" sparse-checkout init --no-cone
git -C "${VENDOR_DIR}" sparse-checkout set /resources/robots/h1/ /LICENSE /README.md
git -C "${VENDOR_DIR}" fetch --depth 1 origin main
git -C "${VENDOR_DIR}" checkout --detach FETCH_HEAD

rm -rf "${DEST_DIR}/urdf" "${DEST_DIR}/meshes"
cp -a "${VENDOR_DIR}/resources/robots/h1/urdf" "${DEST_DIR}/"
cp -a "${VENDOR_DIR}/resources/robots/h1/meshes" "${DEST_DIR}/"
cp "${VENDOR_DIR}/resources/robots/h1/h1.xml" "${DEST_DIR}/h1.xml"
cp "${VENDOR_DIR}/resources/robots/h1/scene.xml" "${DEST_DIR}/scene.xml"
cp "${VENDOR_DIR}/LICENSE" "${DEST_DIR}/LICENSE.unitree_rl_gym"

cat > "${DEST_DIR}/SOURCE.txt" <<EOF
Source: ${REPO_URL}
Sparse path: resources/robots/h1
License: BSD-3-Clause, copied to LICENSE.unitree_rl_gym
Installed by: scripts/setup/fetch_unitree_h1_assets.sh
EOF

echo "Installed Unitree H1 assets to ${DEST_DIR}"
