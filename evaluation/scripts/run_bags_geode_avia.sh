#!/usr/bin/env bash
# Run FAST-LIO (FAST_LIO_ws) over every .bag in data/datasets/GEODE/sensor_data,
# using the AVIA launch file, saving one TUM-format trajectory .txt per bag.
#
# Usage:
#   ./run_bags_geode_avia.sh                # run all bags, no rviz
#   RVIZ=true ./run_bags_geode_avia.sh      # run with rviz
#   RATE=2.0 ./run_bags_geode_avia.sh       # adjust rosbag playback rate
#
# Run with ./FAST_LIO_ws/scripts/run_bags_geode_avia.sh

set -euo pipefail

WS_DIR="/home/andres/semester_project/FAST_LIO_ws"
BAG_DIR="/home/andres/semester_project/data/datasets/GEODE/sensor_data"
OUT_DIR="/home/andres/semester_project/data/estimated_trajectories/fastlio_geode_avia"
RATE="${RATE:-1.0}"

# Normalize RVIZ to a strict roslaunch-bool ("true"/"false") so values like
# ON / on / 1 / yes also enable rviz.
case "${RVIZ:-false}" in
    true|TRUE|True|on|ON|On|1|yes|YES|Yes) RVIZ=true ;;
    *) RVIZ=false ;;
esac

# Fully tear down the FAST-LIO stack (roslaunch + rviz + node + roscore) so the
# next bag starts from a clean slate. Escalates SIGINT -> SIGTERM -> SIGKILL,
# then pkills orphaned rviz / fastlio_mapping if any survived.
stop_fast_lio_stack() {
    kill -INT "${launch_pid}" 2>/dev/null || true
    for _ in {1..15}; do
        kill -0 "${launch_pid}" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "${launch_pid}" 2>/dev/null; then
        echo "[shutdown] roslaunch did not exit on SIGINT; escalating" >&2
        kill -TERM "${launch_pid}" 2>/dev/null || true
        sleep 2
        kill -KILL "${launch_pid}" 2>/dev/null || true
    fi
    wait "${launch_pid}" 2>/dev/null || true

    pkill -INT rviz            2>/dev/null || true
    pkill -INT fastlio_mapping 2>/dev/null || true
    sleep 1
    pkill -KILL rviz            2>/dev/null || true
    pkill -KILL fastlio_mapping 2>/dev/null || true

    kill -INT "${roscore_pid}" 2>/dev/null || true
    for _ in {1..5}; do
        kill -0 "${roscore_pid}" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "${roscore_pid}" 2>/dev/null; then
        kill -KILL "${roscore_pid}" 2>/dev/null || true
    fi
    wait "${roscore_pid}" 2>/dev/null || true
    pkill -KILL rosmaster 2>/dev/null || true
    pkill -KILL rosout    2>/dev/null || true

    sleep 1
}

# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
# shellcheck disable=SC1091
source "${WS_DIR}/devel/setup.bash"

mkdir -p "${OUT_DIR}"

shopt -s nullglob
BAGS=("${BAG_DIR}"/*.bag)
shopt -u nullglob

if [[ ${#BAGS[@]} -eq 0 ]]; then
    echo "[error] no .bag files found in ${BAG_DIR}" >&2
    exit 1
fi

for bag in "${BAGS[@]}"; do
    bag_basename="$(basename "${bag}" .bag)"
    tag="GEODE__${bag_basename}"
    traj_path="${OUT_DIR}/${tag}.txt"
    log_path="${OUT_DIR}/${tag}.log"

    echo "============================================================"
    echo "Bag    : ${bag}"
    echo "Output : ${traj_path}"
    echo "Log    : ${log_path}"
    echo "============================================================"

    roscore >/dev/null 2>&1 &
    roscore_pid=$!
    until rostopic list >/dev/null 2>&1; do sleep 0.2; done

    rosparam set use_sim_time false

    roslaunch fast_lio mapping_avia.launch \
        rviz:="${RVIZ}" \
        trajectory_file_path:="${traj_path}" \
        > "${log_path}" 2>&1 &
    launch_pid=$!

    sleep 3

    total_dur="$(rosbag info -y -k duration "${bag}" 2>/dev/null || echo '?')"
    echo "Total bag duration: ${total_dur} s"
    rosbag play -r "${RATE}" "${bag}"

    sleep 2
    stop_fast_lio_stack

    if [[ -s "${traj_path}" ]]; then
        echo "[ok] wrote $(wc -l < "${traj_path}") poses to ${traj_path}"
    else
        echo "[warn] no trajectory written for ${tag} (see ${log_path})" >&2
    fi
done

echo "Done. Trajectories in: ${OUT_DIR}"
