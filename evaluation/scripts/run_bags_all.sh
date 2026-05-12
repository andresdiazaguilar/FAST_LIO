#!/usr/bin/env bash
# "Mother" runner for FAST_LIO_ws: invokes both per-dataset scripts in
# sequence, so each bag is processed with the launch file matching its LiDAR
# (Tinamu -> mid360, GEODE -> avia). Honors the same RVIZ / RATE env vars.
#
# Usage:
#   ./run_bags_all.sh
#   RATE=2.0 RVIZ=false ./run_bags_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "############################################################"
echo "# 1/2  Tinamu bags (mid360)"
echo "############################################################"
"${SCRIPT_DIR}/run_bags_tinamu_mid360.sh"

echo "############################################################"
echo "# 2/2  GEODE bags (avia)"
echo "############################################################"
"${SCRIPT_DIR}/run_bags_geode_avia.sh"

echo "All datasets done."
