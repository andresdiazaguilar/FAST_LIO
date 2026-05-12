#!/usr/bin/env bash
# Full FAST_LIO_ws pipeline: run all bags, evaluate both datasets against
# the local (in-package) ground truth, then compute the aggregate
# autoresearch-style score against the fixed FAST-LIO2 baseline.
#
# All evaluation assets (ground truth, baseline CSVs, evaluator scripts,
# scoring script) live inside the FAST_LIO ROS package so the pipeline is
# self-contained — only the bag files are external.
#
# Usage:
#   ./run_and_evaluate.sh
#   RVIZ=true ./run_and_evaluate.sh           # show rviz for every bag
#   RATE=2.0 RVIZ=false ./run_and_evaluate.sh
#   SKIP_RUN=true ./run_and_evaluate.sh       # evaluate existing trajectories only
#
# RVIZ accepts true / TRUE / on / ON / 1 / yes (and the obvious False/0/no variants);
# anything else is treated as false.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"   # FAST_LIO/evaluation/scripts
EVAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"                      # FAST_LIO/evaluation
PROJECT_DIR="/home/andres/semester_project"

if [[ -x "${PROJECT_DIR}/data/.venv/bin/python" ]]; then
    VENV_DIR="${PROJECT_DIR}/data/.venv"
else
    VENV_DIR="${PROJECT_DIR}/data/venv"
fi

PYTHON="${VENV_DIR}/bin/python"
EVO_BIN_DIR="${VENV_DIR}/bin"

# In-package evaluator + scoring scripts (copied from data/GEODE_helper and
# autoresearch_FAST_LIO_ws).
EVALUATE_GEODE="${SCRIPT_DIR}/evaluate_geode_fastlio.py"
EVALUATE_TINAMU="${SCRIPT_DIR}/evaluate_tinamu_fastlio.py"
SCORE_AUTORESEARCH="${SCRIPT_DIR}/score_autoresearch.py"

# In-package ground truth (mirrors the dataset-dir layout each evaluator expects).
GEODE_GT_DIR="${EVAL_DIR}/groundtruth/GEODE"
TINAMU_GT_DIR="${EVAL_DIR}/groundtruth/Tinamu"

# In-package fixed baselines used by the aggregate score.
GEODE_BASELINE_CSV="${EVAL_DIR}/baselines/summary_geode_base_fastlio2.csv"
TINAMU_BASELINE_CSV="${EVAL_DIR}/baselines/summary_tinamu_base_fastlio2.csv"

# Estimated trajectories produced by the run_bags_*.sh scripts (kept under
# data/ because they are generated artifacts).
GEODE_EST_DIR="${PROJECT_DIR}/data/estimated_trajectories/fastlio_geode_avia"
TINAMU_EST_DIR="${PROJECT_DIR}/data/estimated_trajectories/fastlio_tinamu_mid360"

# Evaluation outputs (also generated artifacts).
GEODE_OUT_DIR="${PROJECT_DIR}/data/geode_eval/results_fastlio"
TINAMU_OUT_DIR="${PROJECT_DIR}/data/tinamu_eval/results_fastlio"

SKIP_RUN="${SKIP_RUN:-false}"

# Normalize RVIZ here too, so the banner reflects what the bag runners will see.
case "${RVIZ:-false}" in
    true|TRUE|True|on|ON|On|1|yes|YES|Yes) RVIZ_NORM=true ;;
    *) RVIZ_NORM=false ;;
esac
export RVIZ="${RVIZ_NORM}"

echo "[config] RVIZ=${RVIZ}  RATE=${RATE:-1.0}  SKIP_RUN=${SKIP_RUN}"

# -- 1. Run all bags ----------------------------------------------------------

if [[ "${SKIP_RUN}" != "true" ]]; then
    echo "############################################################"
    echo "# STEP 1/4  Running all bags"
    echo "############################################################"
    "${SCRIPT_DIR}/run_bags_all.sh"
else
    echo "[info] SKIP_RUN=true - skipping bag playback, using existing trajectories"
fi

# -- 2. Evaluate GEODE --------------------------------------------------------

echo ""
echo "############################################################"
echo "# STEP 2/4  Evaluating GEODE trajectories"
echo "############################################################"
"${PYTHON}" "${EVALUATE_GEODE}" \
    --est-dir "${GEODE_EST_DIR}" \
    --recursive \
    --dataset-dir "${GEODE_GT_DIR}" \
    --output-dir "${GEODE_OUT_DIR}" \
    --evo-bin-dir "${EVO_BIN_DIR}"

# -- 3. Evaluate Tinamu -------------------------------------------------------

echo ""
echo "############################################################"
echo "# STEP 3/4  Evaluating Tinamu trajectories"
echo "############################################################"
"${PYTHON}" "${EVALUATE_TINAMU}" \
    --est-dir "${TINAMU_EST_DIR}" \
    --recursive \
    --dataset-dir "${TINAMU_GT_DIR}" \
    --output-dir "${TINAMU_OUT_DIR}" \
    --evo-bin-dir "${EVO_BIN_DIR}"

# -- 4. Compute aggregate autoresearch-style score ---------------------------

echo ""
echo "############################################################"
echo "# STEP 4/4  Computing aggregate score"
echo "############################################################"
"${PYTHON}" "${SCORE_AUTORESEARCH}" \
    --geode-candidate "${GEODE_OUT_DIR}/summary.csv" \
    --tinamu-candidate "${TINAMU_OUT_DIR}/summary.csv" \
    --geode-baseline "${GEODE_BASELINE_CSV}" \
    --tinamu-baseline "${TINAMU_BASELINE_CSV}"

echo ""
echo "############################################################"
echo "# Pipeline complete."
echo "#   GEODE results : ${GEODE_OUT_DIR}/summary.csv"
echo "#   Tinamu results: ${TINAMU_OUT_DIR}/summary.csv"
echo "#   Score command : ${SCORE_AUTORESEARCH}"
echo "############################################################"
