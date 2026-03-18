import rosbag
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime
from bisect import bisect_left
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Cameroon dataset:
# bag_path = "data/tinamu_failure_data/cameroon_fail_short/with_IMU_initialization_fix/cameroon_0_to_400_fail_full.bag"

# Silo dataset:
# bag_path = "/home/andres/semester_project/data/tinamu_failure_data/silo_fail/silo_fail_full_case_1.bag"
# bag_path = "/home/andres/semester_project/data/tinamu_failure_data/silo_fail/silo_fail_full_case_2.bag"

# Drift_GTC dataset:
bag_path = "/home/andres/semester_project/data/tinamu_failure_data/drift_gtc/drift_gtc_fail_full.bag"

imu_topic = "/livox/imu"
deg_topic = "/fastlio/degeneracy"
post_deg_topic = "/fastlio/degeneracy_post"
eigvals_pre_translation_topic = "/fastlio/eigvals_pre_translation"
eigvals_pre_rotation_topic = "/fastlio/eigvals_pre_rotation"
eigvals_post_translation_topic = "/fastlio/eigvals_post_translation"
eigvals_post_rotation_topic = "/fastlio/eigvals_post_rotation"
gyro_bias_topic = "/fastlio/gyro_bias"
accel_bias_topic = "/fastlio/accel_bias"
inlier_ratio_topic = "/fastlio/inlier_ratio"
residual_stats_topic = "/fastlio/residual_stats"
gravity_estimate_topic = "/fastlio/gravity_estimate"
accel_gravity_components_topic = "/fastlio/accel_gravity_components"
edge_count_topic = "/struct_diag/edge_count"
plane_count_topic = "/struct_diag/plane_count"

# -------------------------
# Plot selection
# Use "all" to render everything, or a set with selected keys.
# Example:
# ENABLED_PLOTS = {"feat_pre_post", "cond_pre", "cond_post"}
# Available keys:
# "imu_omega_pre", "imu_omega_pre_running_avg", "imu_acc_pre", "lambda_min_pre", "lambda_max_pre",
# "cond_pre", "feat_pre_post", "omega_mean_pre",
# "acc_mean_pre", "lambda_scaled_min_pre", "lambda_scaled_max_pre",
# "cond_scaled_pre", "lambda_min_post", "lambda_max_post",
# "cond_post", "ratio_rt_post", "gyro_biases", "accel_biases",
# "inlier_ratio", "residual_stats", "gravity_estimate", "accel_gravity_consistency",
# "eigvals_pre_translation", "eigvals_pre_rotation",
# "eigvals_post_translation", "eigvals_post_rotation",
# "cond_eigvals_translation", "cond_eigvals_rotation",
# "min_eigvals_translation", "min_eigvals_rotation",
# "edge_feature_count", "plane_feature_count"

# combined plots: "eigenvals"
# -------------------------
# ENABLED_PLOTS = {"feat_pre_post", "cond_pre", "cond_post", "lambda_min_pre", "lambda_min_post", "gyro_biases", "accel_biases", "inlier_ratio", "residual_stats", "imu_omega_pre", "imu_omega_pre_running_avg", "imu_acc_pre", "gravity_estimate", "edge_feature_count", "plane_feature_count"}
# ENABLED_PLOTS = {"feat_pre_post", "cond_pre", "cond_post", "lambda_min_pre", "lambda_min_post", "gyro_biases", "accel_biases", "inlier_ratio", "residual_stats", "imu_omega_pre", "imu_omega_pre_running_avg", "imu_acc_pre", "gravity_estimate", "edge_feature_count"}
# ENABLED_PLOTS = {"feat_pre_post", "accel_gravity_consistency",
                    # "accel_biases", "inlier_ratio", "residual_stats", 
                    # "imu_omega_pre", "imu_omega_pre_running_avg", "imu_acc_pre", 
                    # "gravity_estimate", "edge_feature_count", "plane_feature_count", 
                    # "eigvals_post_translation", "eigvals_post_rotation", 
                    # "eigvals_pre_translation", "eigvals_pre_rotation", "gyro_biases",
                    # "cond_eigvals_translation", "cond_eigvals_rotation",
                    # "min_eigvals_translation", "min_eigvals_rotation",
                    # }

cond_numbers_and_min_eigenvalues = {"cond_eigvals_translation", "cond_eigvals_rotation",
                    "min_eigvals_translation", "min_eigvals_rotation"} 

all_eigenvalues = {"eigvals_post_translation", "eigvals_post_rotation", 
                    "eigvals_pre_translation", "eigvals_pre_rotation"}

imu = {"imu_omega_pre_running_avg", "imu_acc_pre", "gyro_biases", "accel_biases"}

feature_count = {"feat_pre_post", "edge_feature_count", "plane_feature_count", "inlier_ratio"}

gravity = {"accel_gravity_consistency", "gravity_estimate"}


# ENABLED_PLOTS = cond_numbers_and_min_eigenvalues | {"eigenvals"} | imu | feature_count | gravity | {"residual_stats"}
ENABLED_PLOTS =  {"cond_numbers_and_min_eigenvalues", "eigenvals", "imu", "feature_count", "gravity", "residual_stats"}
# ENABLED_PLOTS = {"cond_numbers_and_min_eigenvalues", "eigenvals"}

# -------------------------
# Zoomed inset selection
# -------------------------
# Master toggle for all zoomed inset subplots.
ENABLE_ZOOM_INSETS = False
# Per-plot toggle for zoomed inset subplots.
# These keys match the grouped plot keys used in ENABLED_PLOTS.
ZOOM_INSETS = {
    "cond_numbers_and_min_eigenvalues": True,
    "eigenvals": True,
    "imu": True,
    "feature_count": True,
    "gravity": True,
    "residual_stats": True,
}

OMEGA_RUNNING_AVG_WINDOW = 200

# -------------------------
# Figure saving
# -------------------------
# Toggle to save all figures generated in this run.
SAVE_PLOTS = False
# Base output directory for saved figures.
SAVE_PLOTS_DIR = "plots"
# Image format and dpi.
SAVE_PLOTS_FORMAT = "png"
SAVE_PLOTS_DPI = 300
# If True, create a timestamped subfolder for each run.
SAVE_PLOTS_TIMESTAMPED_SUBDIR = True

# Focus window and failure marker (adjust as needed)
xmin, xmax = 50, 150
# Mode: "time" (single vertical line) or "period" (shaded time interval)

FAILURE_MARKER_MODE = "period"

# Used when FAILURE_MARKER_MODE == "time"
t_marker = 37  # cameroon: 145, silo case 1: 21, silo case 2: 37, drift_gtc: 0-20

# Used when FAILURE_MARKER_MODE == "period"
# Define one or more failure periods as (start_time, end_time).
FAILURE_PERIODS = [
    (0, 20),
    (42, 50),
    (90, 95),
]

zoom_half_window = 10.0


def should_plot(key):
    return ENABLED_PLOTS == "all" or key in ENABLED_PLOTS


def should_show_zoom_inset(key):
    return ENABLE_ZOOM_INSETS and ZOOM_INSETS.get(key, True)


def _sanitize_filename(name):
    clean = re.sub(r"[^\w\s-]", "", name)
    clean = re.sub(r"[\s-]+", "_", clean.strip())
    return clean.lower() if clean else "figure"


def _figure_name(fig, idx):
    if fig._suptitle is not None and fig._suptitle.get_text():
        return f"{idx:02d}_{_sanitize_filename(fig._suptitle.get_text())}"
    if fig.axes and fig.axes[0].get_title():
        return f"{idx:02d}_{_sanitize_filename(fig.axes[0].get_title())}"
    return f"{idx:02d}_figure"


def save_all_open_figures():
    if not SAVE_PLOTS:
        return

    out_dir = SAVE_PLOTS_DIR
    if SAVE_PLOTS_TIMESTAMPED_SUBDIR:
        out_dir = os.path.join(out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    fig_numbers = plt.get_fignums()
    if not fig_numbers:
        print("SAVE_PLOTS enabled, but no figures were created.")
        return

    existing_names = set()
    for idx, fig_num in enumerate(fig_numbers, start=1):
        fig = plt.figure(fig_num)
        base_name = _figure_name(fig, idx)
        name = base_name
        suffix = 2
        while name in existing_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        existing_names.add(name)

        file_path = os.path.join(out_dir, f"{name}.{SAVE_PLOTS_FORMAT}")
        fig.savefig(file_path, dpi=SAVE_PLOTS_DPI, bbox_inches="tight")

    print(f"Saved {len(fig_numbers)} figure(s) to: {out_dir}")


def running_average(values, window):
    if window <= 1 or len(values) == 0:
        return values

    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)

    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append((prefix[i + 1] - prefix[start]) / float(i - start + 1))
    return out


# -------------------------
# Read IMU
# -------------------------
t_imu = []
wx, wy, wz = [], [], []
ax, ay, az = [], [], []

# -------------------------
# Read degeneracy (pre)
# -------------------------
t_deg = []
lambda_min = []
lambda_max = []
cond = []
omega_mean = []
acc_mean = []
effective_feature_number = []
# omega_max = []
lambda_scaled_min = []
lambda_scaled_max = []
cond_scaled = []
acc_mean_no_grav = []

# -------------------------
# Read post_degeneracy (post)
# layout you printed:
# data: [t, eff, lambda_min, lambda_max, cond, ratio_rt]
# -------------------------
t_post = []
post_eff = []
post_lambda_min = []
post_lambda_max = []
post_cond = []
post_ratio_rt = []

t_eig_pre_trans = []
eig_pre_trans_1 = []
eig_pre_trans_2 = []
eig_pre_trans_3 = []
min_eig_pre_trans = []
cond_eig_pre_trans = []

t_eig_pre_rot = []
eig_pre_rot_1 = []
eig_pre_rot_2 = []
eig_pre_rot_3 = []
min_eig_pre_rot = []
cond_eig_pre_rot = []

t_eig_post_trans = []
eig_post_trans_1 = []
eig_post_trans_2 = []
eig_post_trans_3 = []
min_eig_post_trans = []
cond_eig_post_trans = []

t_eig_post_rot = []
eig_post_rot_1 = []
eig_post_rot_2 = []
eig_post_rot_3 = []
min_eig_post_rot = []
cond_eig_post_rot = []

# -------------------------
# Read bias/inlier/residual topics
# -------------------------
t_gyro_bias = []
gyro_bx = []
gyro_by = []
gyro_bz = []
gyro_bnorm = []

t_accel_bias = []
accel_bx = []
accel_by = []
accel_bz = []
accel_bnorm = []

t_inlier_ratio = []
inlier_ratio = []

t_residual = []
residual_median = []
residual_p95 = []
residual_mean = []

t_gravity = []
gravity_x = []
gravity_y = []
gravity_z = []
gravity_norm = []

t_accel_gravity_comp = []
a_parallel = []
a_orth = []

t_edge_count = []
edge_count = []
t_plane_count = []
plane_count = []

with rosbag.Bag(bag_path, "r") as bag:
    # IMU
    for _, msg, _ in bag.read_messages(topics=[imu_topic]):
        t_imu.append(msg.header.stamp.to_sec())
        wx.append(msg.angular_velocity.x)
        wy.append(msg.angular_velocity.y)
        wz.append(msg.angular_velocity.z)
        ax.append(msg.linear_acceleration.x)
        ay.append(msg.linear_acceleration.y)
        az.append(msg.linear_acceleration.z)

    # Degeneracy (pre)
    for _, msg, _ in bag.read_messages(topics=[deg_topic]):
        # msg.data layout:
        # 0: lidar_beg_time
        # 1: effective feature num
        # 2: lambda_min
        # 3: lambda_max
        # 4: cond
        # 5: omega_mean
        # 6: acc_mean
        # 8: lambda_scaled_min (new)
        # 9: lambda_scaled_max (new)
        # 10: cond_scaled (new)
        # 11: acc_mean_no_grav (new)
        data = msg.data
        if len(data) < 8:
            continue

        t_deg.append(float(data[0]))
        effective_feature_number.append(float(data[1]))
        lambda_min.append(float(data[2]))
        lambda_max.append(float(data[3]))
        cond.append(float(data[4]))
        omega_mean.append(float(data[5]))
        # omega_max.append(float(data[6]))
        acc_mean.append(float(data[6]))
        if len(data) >= 12:
            lambda_scaled_min.append(float(data[8]))
            lambda_scaled_max.append(float(data[9]))
            cond_scaled.append(float(data[10]))
            acc_mean_no_grav.append(float(data[11]))
        else:
            lambda_scaled_min.append(float("nan"))
            lambda_scaled_max.append(float("nan"))
            cond_scaled.append(float("nan"))
            acc_mean_no_grav.append(float("nan"))

    # Degeneracy (post)
    for _, msg, _ in bag.read_messages(topics=[post_deg_topic]):
        data = msg.data
        # expected:
        # 0: lidar_beg_time
        # 1: effective feature num
        # 2: lambda_min_post
        # 3: lambda_max_post
        # 4: cond_post
        # 5: ratio_rt
        if len(data) < 6:
            continue

        t_post.append(float(data[0]))
        post_eff.append(float(data[1]))
        post_lambda_min.append(float(data[2]))
        post_lambda_max.append(float(data[3]))
        post_cond.append(float(data[4]))
        post_ratio_rt.append(float(data[5]))

    # Eigenvalue spectra (3x3 blocks)
    # expected:
    # 0: t
    # 1: effective feature num
    # 2: lambda1
    # 3: lambda2
    # 4: lambda3
    # 5: lambda_min
    # 6: lambda_max
    # 7: cond
    for _, msg, _ in bag.read_messages(topics=[eigvals_pre_translation_topic]):
        data = msg.data
        if len(data) < 8:
            continue
        t_eig_pre_trans.append(float(data[0]))
        eig_pre_trans_1.append(float(data[2]))
        eig_pre_trans_2.append(float(data[3]))
        eig_pre_trans_3.append(float(data[4]))
        min_eig_pre_trans.append(float(data[5]))
        cond_eig_pre_trans.append(float(data[7]))

    for _, msg, _ in bag.read_messages(topics=[eigvals_pre_rotation_topic]):
        data = msg.data
        if len(data) < 8:
            continue
        t_eig_pre_rot.append(float(data[0]))
        eig_pre_rot_1.append(float(data[2]))
        eig_pre_rot_2.append(float(data[3]))
        eig_pre_rot_3.append(float(data[4]))
        min_eig_pre_rot.append(float(data[5]))
        cond_eig_pre_rot.append(float(data[7]))

    for _, msg, _ in bag.read_messages(topics=[eigvals_post_translation_topic]):
        data = msg.data
        if len(data) < 8:
            continue
        t_eig_post_trans.append(float(data[0]))
        eig_post_trans_1.append(float(data[2]))
        eig_post_trans_2.append(float(data[3]))
        eig_post_trans_3.append(float(data[4]))
        min_eig_post_trans.append(float(data[5]))
        cond_eig_post_trans.append(float(data[7]))

    for _, msg, _ in bag.read_messages(topics=[eigvals_post_rotation_topic]):
        data = msg.data
        if len(data) < 8:
            continue
        t_eig_post_rot.append(float(data[0]))
        eig_post_rot_1.append(float(data[2]))
        eig_post_rot_2.append(float(data[3]))
        eig_post_rot_3.append(float(data[4]))
        min_eig_post_rot.append(float(data[5]))
        cond_eig_post_rot.append(float(data[7]))

    # Gyro bias
    for _, msg, _ in bag.read_messages(topics=[gyro_bias_topic]):
        # expected:
        # 0: lidar_end_time
        # 1: bg_x
        # 2: bg_y
        # 3: bg_z
        # 4: ||bg||
        data = msg.data
        if len(data) < 5:
            continue

        t_gyro_bias.append(float(data[0]))
        gyro_bx.append(float(data[1]))
        gyro_by.append(float(data[2]))
        gyro_bz.append(float(data[3]))
        gyro_bnorm.append(float(data[4]))

    # Accel bias
    for _, msg, _ in bag.read_messages(topics=[accel_bias_topic]):
        # expected:
        # 0: lidar_end_time
        # 1: ba_x
        # 2: ba_y
        # 3: ba_z
        # 4: ||ba||
        data = msg.data
        if len(data) < 5:
            continue

        t_accel_bias.append(float(data[0]))
        accel_bx.append(float(data[1]))
        accel_by.append(float(data[2]))
        accel_bz.append(float(data[3]))
        accel_bnorm.append(float(data[4]))

    # Inlier ratio
    for _, msg, t in bag.read_messages(topics=[inlier_ratio_topic]):
        # Support both:
        # - std_msgs/Float64 => msg.data is scalar, use bag timestamp
        # - std_msgs/Float64MultiArray => [t, inlier_ratio]
        data = msg.data
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            t_inlier_ratio.append(float(data[0]))
            inlier_ratio.append(float(data[1]))
        else:
            t_inlier_ratio.append(t.to_sec())
            inlier_ratio.append(float(data))

    # Residual stats
    for _, msg, _ in bag.read_messages(topics=[residual_stats_topic]):
        # expected:
        # 0: lidar_beg_time
        # 1: residual_median
        # 2: residual_p95
        # 3: residual_mean
        data = msg.data
        if len(data) < 4:
            continue

        t_residual.append(float(data[0]))
        residual_median.append(float(data[1]))
        residual_p95.append(float(data[2]))
        residual_mean.append(float(data[3]))

    # Gravity estimate
    for _, msg, _ in bag.read_messages(topics=[gravity_estimate_topic]):
        # expected:
        # 0: lidar_end_time
        # 1: grav_x
        # 2: grav_y
        # 3: grav_z
        # 4: ||grav||
        data = msg.data
        if len(data) < 5:
            continue

        t_gravity.append(float(data[0]))
        gravity_x.append(float(data[1]))
        gravity_y.append(float(data[2]))
        gravity_z.append(float(data[3]))
        gravity_norm.append(float(data[4]))

    # Accel components wrt gravity
    for _, msg, _ in bag.read_messages(topics=[accel_gravity_components_topic]):
        # expected:
        # 0: lidar_end_time
        # 1: a_parallel
        # 2: a_orth
        data = msg.data
        if len(data) < 3:
            continue

        t_accel_gravity_comp.append(float(data[0]))
        a_parallel.append(float(data[1]))
        a_orth.append(float(data[2]))

    # Edge / plane feature counts
    # Supports both:
    # - std_msgs/Int32 => msg.data is scalar, use bag timestamp
    # - std_msgs/Float64MultiArray => [t, count]
    for _, msg, t in bag.read_messages(topics=[edge_count_topic]):
        data = msg.data
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            t_edge_count.append(float(data[0]))
            edge_count.append(float(data[1]))
        else:
            t_edge_count.append(t.to_sec())
            edge_count.append(float(data))

    for _, msg, t in bag.read_messages(topics=[plane_count_topic]):
        data = msg.data
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            t_plane_count.append(float(data[0]))
            plane_count.append(float(data[1]))
        else:
            t_plane_count.append(t.to_sec())
            plane_count.append(float(data))

# -------------------------
# Put everything on a shared "time since start" axis
# -------------------------
t0_candidates = []
if len(t_imu) > 0:  t0_candidates.append(t_imu[0])
if len(t_deg) > 0:  t0_candidates.append(t_deg[0])
if len(t_post) > 0: t0_candidates.append(t_post[0])
if len(t_eig_pre_trans) > 0: t0_candidates.append(t_eig_pre_trans[0])
if len(t_eig_pre_rot) > 0: t0_candidates.append(t_eig_pre_rot[0])
if len(t_eig_post_trans) > 0: t0_candidates.append(t_eig_post_trans[0])
if len(t_eig_post_rot) > 0: t0_candidates.append(t_eig_post_rot[0])
if len(t_gyro_bias) > 0: t0_candidates.append(t_gyro_bias[0])
if len(t_accel_bias) > 0: t0_candidates.append(t_accel_bias[0])
if len(t_inlier_ratio) > 0: t0_candidates.append(t_inlier_ratio[0])
if len(t_residual) > 0: t0_candidates.append(t_residual[0])
if len(t_gravity) > 0: t0_candidates.append(t_gravity[0])
if len(t_accel_gravity_comp) > 0: t0_candidates.append(t_accel_gravity_comp[0])
if len(t_edge_count) > 0: t0_candidates.append(t_edge_count[0])
if len(t_plane_count) > 0: t0_candidates.append(t_plane_count[0])
t0 = min(t0_candidates) if len(t0_candidates) > 0 else 0.0

t_imu = [ti - t0 for ti in t_imu]
t_deg = [ti - t0 for ti in t_deg]
t_post = [ti - t0 for ti in t_post]
t_eig_pre_trans = [ti - t0 for ti in t_eig_pre_trans]
t_eig_pre_rot = [ti - t0 for ti in t_eig_pre_rot]
t_eig_post_trans = [ti - t0 for ti in t_eig_post_trans]
t_eig_post_rot = [ti - t0 for ti in t_eig_post_rot]
t_gyro_bias = [ti - t0 for ti in t_gyro_bias]
t_accel_bias = [ti - t0 for ti in t_accel_bias]
t_inlier_ratio = [ti - t0 for ti in t_inlier_ratio]
t_residual = [ti - t0 for ti in t_residual]
t_gravity = [ti - t0 for ti in t_gravity]
t_accel_gravity_comp = [ti - t0 for ti in t_accel_gravity_comp]
t_edge_count = [ti - t0 for ti in t_edge_count]
t_plane_count = [ti - t0 for ti in t_plane_count]


def nearest_index(sorted_times, t):
    idx = bisect_left(sorted_times, t)
    if idx <= 0:
        return 0
    if idx >= len(sorted_times):
        return len(sorted_times) - 1
    prev_i = idx - 1
    return prev_i if abs(sorted_times[prev_i] - t) <= abs(sorted_times[idx] - t) else idx

def failure_center_time():
    if FAILURE_MARKER_MODE == "period":
        periods = get_failure_periods()
        if periods:
            # Use the period whose center is closest to t_marker as zoom anchor.
            start, end = min(periods, key=lambda p: abs(0.5 * (p[0] + p[1]) - t_marker))
            return 0.5 * (start + end)
    return t_marker


def get_failure_periods():
    periods = []
    for period in FAILURE_PERIODS:
        if len(period) != 2:
            continue
        start, end = period
        if end < start:
            start, end = end, start
        periods.append((start, end))
    return periods


def add_marker_line(ax=None, linewidth=2, include_label=True):
    if ax is None:
        ax = plt.gca()
    if FAILURE_MARKER_MODE == "period":
        periods = get_failure_periods()
        if not periods:
            return
        for i, (start, end) in enumerate(periods):
            label = "failure period" if include_label and i == 0 else None
            ax.axvspan(start, end, color="r", alpha=0.2, label=label)
    else:
        label = "failure time" if include_label else None
        ax.axvline(x=t_marker, color="r", linestyle="--", linewidth=linewidth, label=label)

# =========================
# PRE plots
# =========================

# 1) Angular velocity + omega_mean (pre)
if should_plot("imu_omega_pre"):
    plt.figure()
    plt.plot(t_imu, wx, label=r"$\omega_x$")
    plt.plot(t_imu, wy, label=r"$\omega_y$")
    plt.plot(t_imu, wz, label=r"$\omega_z$")
    plt.plot(t_deg, omega_mean, label=r"$\omega_{\mathrm{mean}}$")
    # plt.plot(t_deg, omega_max, label="omega_max")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("IMU angular velocity [rad/s]")
    plt.title("IMU Gyro Angular Velocity (rad/s)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 2) IMU diagnostics group (2x2)
if (should_plot("imu_acc_pre") or
        should_plot("imu_omega_pre_running_avg") or
        should_plot("gyro_biases") or
        should_plot("accel_biases") or
        should_plot("imu")):
    wx_avg = running_average(wx, OMEGA_RUNNING_AVG_WINDOW)
    wy_avg = running_average(wy, OMEGA_RUNNING_AVG_WINDOW)
    wz_avg = running_average(wz, OMEGA_RUNNING_AVG_WINDOW)
    omega_mean_avg = running_average(omega_mean, OMEGA_RUNNING_AVG_WINDOW)
    ax_avg = running_average(ax, OMEGA_RUNNING_AVG_WINDOW)
    ay_avg = running_average(ay, OMEGA_RUNNING_AVG_WINDOW)
    az_avg = running_average(az, OMEGA_RUNNING_AVG_WINDOW)
    acc_mean_avg = running_average(acc_mean, OMEGA_RUNNING_AVG_WINDOW)
    acc_mean_no_grav_avg = running_average(acc_mean_no_grav, OMEGA_RUNNING_AVG_WINDOW)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("IMU Diagnostics")

    axp = axs[0, 0]
    axp.plot(t_imu, wx_avg, label=r"$\omega_x$")
    axp.plot(t_imu, wy_avg, label=r"$\omega_y$")
    axp.plot(t_imu, wz_avg, label=r"$\omega_z$")
    axp.plot(t_deg, omega_mean_avg, label=r"$\omega_{\mathrm{mean}}$")
    add_marker_line(axp)
    axp.set_title(f"Gyro Angular Velocity (running avg, window={OMEGA_RUNNING_AVG_WINDOW})")
    axp.set_ylabel("angular velocity [rad/s]")
    axp.grid(True)
    axp.legend()
    if should_show_zoom_inset("imu"):
        x_center = failure_center_time()
        x1 = x_center - zoom_half_window
        x2 = x_center + zoom_half_window
        axins = inset_axes(
            axp,
            width="40%",
            height="40%",
            loc="lower left",
            bbox_to_anchor=(0.08, 0.08, 1, 1),
            bbox_transform=axp.transAxes,
        )
        axins.plot(t_imu, wx_avg)
        axins.plot(t_imu, wy_avg)
        axins.plot(t_imu, wz_avg)
        axins.plot(t_deg, omega_mean_avg)
        add_marker_line(axins, linewidth=1, include_label=False)
        axins.set_xlim(x1, x2)
        y_zoom = (
            [v for t, v in zip(t_imu, wx_avg) if x1 <= t <= x2] +
            [v for t, v in zip(t_imu, wy_avg) if x1 <= t <= x2] +
            [v for t, v in zip(t_imu, wz_avg) if x1 <= t <= x2] +
            [v for t, v in zip(t_deg, omega_mean_avg) if x1 <= t <= x2]
        )
        if y_zoom:
            y_min = min(y_zoom)
            y_max = max(y_zoom)
            pad = 0.05 * max(1e-6, y_max - y_min)
            axins.set_ylim(y_min - pad, y_max + pad)
        axins.grid(True, alpha=0.3)
        axins.tick_params(labelsize=7, pad=1)
        axins.set_facecolor((1.0, 1.0, 1.0, 0.9))
        mark_inset(axp, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axp = axs[0, 1]
    axp.plot(t_imu, ax_avg, label=r"$a_x$")
    axp.plot(t_imu, ay_avg, label=r"$a_y$")
    axp.plot(t_imu, az_avg, label=r"$a_z$")
    axp.plot(t_deg, acc_mean_avg, label=r"$a_{\mathrm{mean}}$")
    axp.plot(t_deg, acc_mean_no_grav_avg, label=r"$a_{\mathrm{mean,no\ grav}}$")
    add_marker_line(axp)
    axp.set_title(f"Linear Acceleration (running avg, window={OMEGA_RUNNING_AVG_WINDOW})")
    axp.set_ylabel("acceleration")
    axp.grid(True)
    axp.legend()
    if should_show_zoom_inset("imu"):
        x_center = failure_center_time()
        x1 = x_center - zoom_half_window
        x2 = x_center + zoom_half_window
        axins = inset_axes(
            axp,
            width="40%",
            height="30%",
            loc="lower left",
            bbox_to_anchor=(0.08, 0.08, 1, 1),
            bbox_transform=axp.transAxes,
        )
        axins.plot(t_imu, ax_avg)
        axins.plot(t_imu, ay_avg)
        axins.plot(t_imu, az_avg)
        axins.plot(t_deg, acc_mean_avg)
        axins.plot(t_deg, acc_mean_no_grav_avg)
        add_marker_line(axins, linewidth=1, include_label=False)
        axins.set_xlim(x1, x2)
        y_zoom = (
            [v for t, v in zip(t_imu, ax_avg) if x1 <= t <= x2] +
            [v for t, v in zip(t_imu, ay_avg) if x1 <= t <= x2] +
            [v for t, v in zip(t_imu, az_avg) if x1 <= t <= x2] +
            [v for t, v in zip(t_deg, acc_mean_avg) if x1 <= t <= x2] +
            [v for t, v in zip(t_deg, acc_mean_no_grav_avg) if x1 <= t <= x2]
        )
        if y_zoom:
            y_min = min(y_zoom)
            y_max = max(y_zoom)
            pad = 0.05 * max(1e-6, y_max - y_min)
            axins.set_ylim(y_min - pad, y_max + pad)
        axins.grid(True, alpha=0.3)
        axins.tick_params(labelsize=7, pad=1)
        axins.set_facecolor((1.0, 1.0, 1.0, 0.9))
        mark_inset(axp, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axp = axs[1, 0]
    axp.plot(t_gyro_bias, gyro_bx, label=r"$b_{g_x}$ [rad/s]")
    axp.plot(t_gyro_bias, gyro_by, label=r"$b_{g_y}$ [rad/s]")
    axp.plot(t_gyro_bias, gyro_bz, label=r"$b_{g_z}$ [rad/s]")
    axp.plot(t_gyro_bias, gyro_bnorm, label=r"$\|b_g\|$ [rad/s]")
    add_marker_line(axp)
    axp.set_title("Gyro Biases")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("bias [rad/s]")
    axp.grid(True)
    axp.legend()

    axp = axs[1, 1]
    axp.plot(t_accel_bias, accel_bx, label=r"$b_{a_x}$ [g]")
    axp.plot(t_accel_bias, accel_by, label=r"$b_{a_y}$ [g]")
    axp.plot(t_accel_bias, accel_bz, label=r"$b_{a_z}$ [g]")
    axp.plot(t_accel_bias, accel_bnorm, label=r"$\|b_a\|$ [g]")
    add_marker_line(axp)
    axp.set_title("Accel Biases")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("bias [g]")
    axp.grid(True)
    axp.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# 3) lambda_min (pre)
if should_plot("lambda_min_pre"):
    plt.figure()
    plt.plot(t_deg, lambda_min, label="lambda_min (pre)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("lambda_min")
    plt.title("Lambda_min (pre)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 4) lambda_max (pre)
if should_plot("lambda_max_pre"):
    plt.figure()
    plt.plot(t_deg, lambda_max, label="lambda_max (pre)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("lambda_max")
    plt.title("Lambda_max (pre)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 5) condition number (pre)
if should_plot("cond_pre"):
    plt.figure()
    plt.plot(t_deg, cond, label="cond (pre)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("condition number")
    plt.title("Unscaled Condition Number (pre)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)
    plt.ylim(0, 1000)

# 6) Feature count diagnostics group (2x2)
if (should_plot("feat_pre_post") or
        should_plot("inlier_ratio") or
        should_plot("edge_feature_count") or
        should_plot("plane_feature_count") or
        should_plot("feature_count")):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Feature Count Diagnostics")

    axp = axs[0, 0]
    axp.plot(t_deg, effective_feature_number, label="# effective features")
    # axp.plot(t_post, post_eff, label="effective feature number (post)")
    add_marker_line(axp)
    axp.set_title("Effective Feature Number")
    axp.set_ylabel("count")
    axp.grid(True)
    axp.legend()

    axp = axs[0, 1]
    axp.plot(t_inlier_ratio, inlier_ratio, label="inlier ratio")
    add_marker_line(axp)
    axp.set_title("Inlier Ratio")
    axp.set_ylabel("ratio")
    axp.grid(True)
    axp.legend()

    axp = axs[1, 0]
    axp.plot(t_plane_count, plane_count, label="# planar features")
    add_marker_line(axp)
    axp.set_title("Plane Feature Count")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("count")
    axp.grid(True)
    axp.legend()

    axp = axs[1, 1]
    axp.plot(t_edge_count, edge_count, label="# edge features")
    add_marker_line(axp)
    axp.set_title("Edge Feature Count")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("count")
    axp.grid(True)
    axp.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# 7) omega_mean (pre)
if should_plot("omega_mean_pre"):
    plt.figure()
    plt.plot(t_deg, omega_mean, label="omega_mean (pre)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("omega_mean [rad/s]")
    plt.title("Mean Angular Velocity [rad/s]")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# # 8) omega_max (pre)
# if should_plot("omega_max_pre"):
#     plt.figure()
#     plt.plot(t_deg, omega_max, label="omega_max (pre)")
#     add_marker_line()
#     plt.xlabel("time [s]")
#     plt.ylabel("omega_max [rad/s]")
#     plt.title("Max Angular Velocity [rad/s]")
#     plt.legend()
#     plt.grid(True)
#     # plt.xlim(xmin, xmax)

# 9) acc_mean (pre)
if should_plot("acc_mean_pre"):
    plt.figure()
    plt.plot(t_deg, acc_mean, label="acc_mean (pre)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("acc_mean [g]")
    plt.title("Mean Linear Acceleration [g]")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 10) lambda_scaled_min (pre)
if should_plot("lambda_scaled_min_pre"):
    plt.figure()
    plt.plot(t_deg, lambda_scaled_min, label="lambda_scaled_min (pre)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("lambda_scaled_min")
    plt.title("Lambda_scaled_min (pre)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 11) lambda_scaled_max (pre)
if should_plot("lambda_scaled_max_pre"):
    plt.figure()
    plt.plot(t_deg, lambda_scaled_max, label="lambda_scaled_max (pre)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("lambda_scaled_max")
    plt.title("Lambda_scaled_max (pre)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 12) cond_scaled (pre)
if should_plot("cond_scaled_pre"):
    plt.figure()
    plt.plot(t_deg, cond_scaled, label="cond_scaled (pre)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("condition number (scaled)")
    plt.title("Scaled Condition Number (pre)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)
    plt.ylim(0, 500)

# =========================
# POST plots
# =========================

# 13) lambda_min (post)
if should_plot("lambda_min_post"):
    plt.figure()
    plt.plot(t_post, post_lambda_min, label="lambda_min (post)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("lambda_min")
    plt.title("Lambda_min (post)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 14) lambda_max (post)
if should_plot("lambda_max_post"):
    plt.figure()
    plt.plot(t_post, post_lambda_max, label="lambda_max (post)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("lambda_max")
    plt.title("Lambda_max (post)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 15) condition number (post)
if should_plot("cond_post"):
    plt.figure()
    plt.plot(t_post, post_cond, label="cond (post)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("condition number")
    plt.title("Condition Number (post)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)
    plt.ylim(0, 500)

# 17) ratio r/t (post)
if should_plot("ratio_rt_post"):
    plt.figure()
    plt.plot(t_post, post_ratio_rt, label="ratio r/t (post)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("r/t ratio")
    plt.title("Rotation/translation RMS ratio (post)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)


# 21) Residual stats
if should_plot("residual_stats"):
    plt.figure()
    plt.plot(t_residual, residual_median, label=r"$r_{\mathrm{median}}$")
    plt.plot(t_residual, residual_p95, label=r"$r_{\mathrm{p95}}$")
    plt.plot(t_residual, residual_mean, label=r"$r_{\mathrm{mean}}$")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("residual")
    plt.title("Residuals")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 22) Gravity diagnostics group (2x1)
if should_plot("gravity_estimate") or should_plot("accel_gravity_consistency") or should_plot("gravity"):
    t_accel_consistency = []
    abs_neg_a_parallel_minus_g = []
    neg_a_parallel = []
    if len(t_gravity) > 0 and len(t_accel_gravity_comp) > 0:
        for i, t_comp in enumerate(t_accel_gravity_comp):
            j = nearest_index(t_gravity, t_comp)
            t_accel_consistency.append(t_comp)
            neg_a_parallel.append(-a_parallel[i])
            abs_neg_a_parallel_minus_g.append(abs(-a_parallel[i]) - gravity_norm[j])
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Gravity Diagnostics")

    axp = axs[0]
    axp.plot(t_gravity, gravity_x, label=r"$g_x$ [m/s$^2$]")
    axp.plot(t_gravity, gravity_y, label=r"$g_y$ [m/s$^2$]")
    axp.plot(t_gravity, gravity_z, label=r"$g_z$ [m/s$^2$]")
    axp.plot(t_gravity, gravity_norm, label=r"$\|g\|$ [m/s$^2$]")
    add_marker_line(axp)
    axp.set_title("Gravity Estimate")
    axp.set_ylabel(r"gravity [m/s$^2$]")
    axp.grid(True)
    axp.legend()

    axp = axs[1]
    axp.plot(t_accel_gravity_comp, a_orth, label=r"$a_{\perp}$ [m/s$^2$]")
    axp.plot(t_accel_gravity_comp, neg_a_parallel, label=r"$-a_{\parallel}$ [m/s$^2$]")
    axp.plot(t_accel_gravity_comp, gravity_norm, label=r"$\|g\|$ [m/s$^2$]")
    # axp.plot(t_accel_consistency, abs_neg_a_parallel_minus_g, label="|-|a_parallel| - |g|| [m/s²]")
    add_marker_line(axp)
    axp.set_title("Linear Acceleration Components")
    axp.set_xlabel("time [s]")
    axp.set_ylabel(r"acceleration [m/s$^2$]")
    axp.grid(True)
    axp.legend()
    if should_show_zoom_inset("gravity"):
        x_center = failure_center_time()
        x1 = x_center - zoom_half_window
        x2 = x_center + zoom_half_window
        axins = inset_axes(
            axp,
            width="25%",
            height="50%",
            loc="lower left",
            bbox_to_anchor=(0.25, 0.08, 1, 1),
            bbox_transform=axp.transAxes,
        )
        axins.plot(t_accel_gravity_comp, a_orth)
        axins.plot(t_accel_gravity_comp, neg_a_parallel)
        axins.plot(t_accel_gravity_comp, gravity_norm)
        add_marker_line(axins, linewidth=1, include_label=False)
        axins.set_xlim(x1, x2)
        y_zoom = (
            [v for t, v in zip(t_accel_gravity_comp, a_orth) if x1 <= t <= x2] +
            [v for t, v in zip(t_accel_gravity_comp, neg_a_parallel) if x1 <= t <= x2] +
            [v for t, v in zip(t_accel_gravity_comp, gravity_norm) if x1 <= t <= x2]
        )
        if y_zoom:
            y_min = min(y_zoom)
            y_max = max(y_zoom)
            pad = 0.05 * max(1e-6, y_max - y_min)
            axins.set_ylim(y_min - pad, y_max + pad)
        axins.grid(True, alpha=0.3)
        axins.tick_params(labelsize=7, pad=1)
        axins.set_facecolor((1.0, 1.0, 1.0, 0.9))
        mark_inset(axp, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# 25) All eigenvalue spectra in one 2x2 figure
if (should_plot("eigvals_pre_translation") or
        should_plot("eigvals_pre_rotation") or
        should_plot("eigvals_post_translation") or
        should_plot("eigvals_post_rotation") or
        should_plot("eigenvals") or
        should_plot("all_eigenvalues")):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Eigenvalues (Pre/Post, Translation/Rotation)")

    ax = axs[0, 0]
    ax.plot(t_eig_pre_trans, eig_pre_trans_1, label=r"$\lambda_1$")
    ax.plot(t_eig_pre_trans, eig_pre_trans_2, label=r"$\lambda_2$")
    ax.plot(t_eig_pre_trans, eig_pre_trans_3, label=r"$\lambda_3$")
    add_marker_line(ax)
    ax.set_title("Pre Translation")
    ax.set_ylabel("eigenvalue")
    ax.grid(True)
    ax.legend()

    ax = axs[0, 1]
    ax.plot(t_eig_pre_rot, eig_pre_rot_1, label=r"$\lambda_1$")
    ax.plot(t_eig_pre_rot, eig_pre_rot_2, label=r"$\lambda_2$")
    ax.plot(t_eig_pre_rot, eig_pre_rot_3, label=r"$\lambda_3$")
    add_marker_line(ax)
    ax.set_title("Pre Rotation")
    ax.grid(True)
    ax.legend()

    ax = axs[1, 0]
    ax.plot(t_eig_post_trans, eig_post_trans_1, label=r"$\lambda_1$")
    ax.plot(t_eig_post_trans, eig_post_trans_2, label=r"$\lambda_2$")
    ax.plot(t_eig_post_trans, eig_post_trans_3, label=r"$\lambda_3$")
    add_marker_line(ax)
    ax.set_title("Post Translation")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("eigenvalue")
    ax.grid(True)
    ax.legend()

    ax = axs[1, 1]
    ax.plot(t_eig_post_rot, eig_post_rot_1, label=r"$\lambda_1$")
    ax.plot(t_eig_post_rot, eig_post_rot_2, label=r"$\lambda_2$")
    ax.plot(t_eig_post_rot, eig_post_rot_3, label=r"$\lambda_3$")
    add_marker_line(ax)
    ax.set_title("Post Rotation")
    ax.set_xlabel("time [s]")
    ax.grid(True)
    ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# 29) Condition number + minimum eigenvalue group (2x2)
if (should_plot("cond_eigvals_translation") or
        should_plot("cond_eigvals_rotation") or
        should_plot("min_eigvals_translation") or
        should_plot("min_eigvals_rotation") or
        should_plot("cond_eigvals_pre_translation") or
        should_plot("cond_eigvals_post_translation") or
        should_plot("cond_eigvals_pre_rotation") or
        should_plot("cond_eigvals_post_rotation") or
        should_plot("min_eigvals_pre_translation") or
        should_plot("min_eigvals_post_translation") or
        should_plot("min_eigvals_pre_rotation") or
        should_plot("min_eigvals_post_rotation") or
        should_plot("cond_numbers_and_min_eigenvalues")):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Condition Numbers and Minimum Eigenvalues")

    axp = axs[0, 0]
    axp.plot(t_eig_pre_trans, cond_eig_pre_trans, label=r"$\kappa$ (pre translation)")
    axp.plot(t_eig_post_trans, cond_eig_post_trans, label=r"$\kappa$ (post translation)")
    add_marker_line(axp)
    axp.set_title("Condition Number - Translation")
    axp.set_ylabel("condition number")
    axp.grid(True)
    axp.legend()
    axp.set_ylim(0, 80)
    if should_show_zoom_inset("cond_numbers_and_min_eigenvalues"):
        x_center = failure_center_time()
        x1 = x_center - zoom_half_window
        x2 = x_center + zoom_half_window
        axins = inset_axes(
            axp,
            width="40%",
            height="40%",
            loc="upper left",
            bbox_to_anchor=(0.04, -0.2, 1, 1),
            bbox_transform=axp.transAxes,
        )
        axins.plot(t_eig_pre_trans, cond_eig_pre_trans)
        axins.plot(t_eig_post_trans, cond_eig_post_trans)
        add_marker_line(axins, linewidth=1, include_label=False)
        axins.set_xlim(x1, x2)
        y_zoom = (
            [v for t, v in zip(t_eig_pre_trans, cond_eig_pre_trans) if x1 <= t <= x2] +
            [v for t, v in zip(t_eig_post_trans, cond_eig_post_trans) if x1 <= t <= x2]
        )
        if y_zoom:
            y_min = min(y_zoom)
            y_max = max(y_zoom)
            pad = 0.05 * max(1e-6, y_max - y_min)
            axins.set_ylim(y_min - pad, y_max + pad)
        axins.grid(True, alpha=0.3)
        axins.tick_params(labelsize=7, pad=1)
        axins.set_facecolor((1.0, 1.0, 1.0, 0.9))
        mark_inset(axp, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axp = axs[0, 1]
    axp.plot(t_eig_pre_rot, cond_eig_pre_rot, label=r"$\kappa$ (pre rotation)")
    axp.plot(t_eig_post_rot, cond_eig_post_rot, label=r"$\kappa$ (post rotation)")
    add_marker_line(axp)
    axp.set_title("Condition Number - Rotation")
    axp.set_ylabel("condition number")
    axp.grid(True)
    axp.legend()
    axp.set_ylim(0, 120) # Cameroon: (0, 80) 
    if should_show_zoom_inset("cond_numbers_and_min_eigenvalues"):
        x_center = failure_center_time()
        x1 = x_center - zoom_half_window
        x2 = x_center + zoom_half_window
        axins = inset_axes(
            axp,
            width="40%",
            height="40%",
            loc="upper left",
            bbox_to_anchor=(0.04, -0.2, 1, 1),
            bbox_transform=axp.transAxes,
        )
        axins.plot(t_eig_pre_rot, cond_eig_pre_rot)
        axins.plot(t_eig_post_rot, cond_eig_post_rot)
        add_marker_line(axins, linewidth=1, include_label=False)
        axins.set_xlim(x1, x2)
        y_zoom = (
            [v for t, v in zip(t_eig_pre_rot, cond_eig_pre_rot) if x1 <= t <= x2] +
            [v for t, v in zip(t_eig_post_rot, cond_eig_post_rot) if x1 <= t <= x2]
        )
        if y_zoom:
            y_min = min(y_zoom)
            y_max = max(y_zoom)
            pad = 0.05 * max(1e-6, y_max - y_min)
            axins.set_ylim(y_min - pad, y_max + pad)
        axins.grid(True, alpha=0.3)
        axins.tick_params(labelsize=7, pad=1)
        axins.set_facecolor((1.0, 1.0, 1.0, 0.9))
        mark_inset(axp, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axp = axs[1, 0]
    axp.plot(t_eig_pre_trans, min_eig_pre_trans, label=r"$\lambda_{\min}$ (pre translation)")
    axp.plot(t_eig_post_trans, min_eig_post_trans, label=r"$\lambda_{\min}$ (post translation)")
    add_marker_line(axp)
    axp.set_title("Minimum Eigenvalue - Translation")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("minimum eigenvalue")
    axp.grid(True)
    axp.legend()
    if should_show_zoom_inset("cond_numbers_and_min_eigenvalues"):
        x_center = failure_center_time()
        x1 = x_center - zoom_half_window
        x2 = x_center + zoom_half_window
        axins = inset_axes(
            axp,
            width="40%",
            height="40%",
            loc="upper right",
            bbox_to_anchor=(-0.04, -0.2, 1, 1),
            bbox_transform=axp.transAxes,
        )
        axins.plot(t_eig_pre_trans, min_eig_pre_trans)
        axins.plot(t_eig_post_trans, min_eig_post_trans)
        add_marker_line(axins, linewidth=1, include_label=False)
        axins.set_xlim(x1, x2)
        y_zoom = (
            [v for t, v in zip(t_eig_pre_trans, min_eig_pre_trans) if x1 <= t <= x2] +
            [v for t, v in zip(t_eig_post_trans, min_eig_post_trans) if x1 <= t <= x2]
        )
        if y_zoom:
            y_min = min(y_zoom)
            y_max = max(y_zoom)
            pad = 0.05 * max(1e-6, y_max - y_min)
            axins.set_ylim(y_min - pad, y_max + pad)
        axins.grid(True, alpha=0.3)
        axins.tick_params(labelsize=7, pad=1)
        axins.set_facecolor((1.0, 1.0, 1.0, 0.9))
        mark_inset(axp, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axp = axs[1, 1]
    axp.plot(t_eig_pre_rot, min_eig_pre_rot, label=r"$\lambda_{\min}$ (pre rotation)")
    axp.plot(t_eig_post_rot, min_eig_post_rot, label=r"$\lambda_{\min}$ (post rotation)")
    add_marker_line(axp)
    axp.set_title("Minimum Eigenvalue - Rotation")
    axp.set_xlabel("time [s]")
    axp.set_ylabel("minimum eigenvalue")
    axp.grid(True)
    axp.legend()
    if should_show_zoom_inset("cond_numbers_and_min_eigenvalues"):
        x_center = failure_center_time()
        x1 = x_center - zoom_half_window
        x2 = x_center + zoom_half_window
        axins = inset_axes(
            axp,
            width="40%",
            height="40%",
            loc="upper right",
            bbox_to_anchor=(-0.04, -0.2, 1, 1),
            bbox_transform=axp.transAxes,
        )
        axins.plot(t_eig_pre_rot, min_eig_pre_rot)
        axins.plot(t_eig_post_rot, min_eig_post_rot)
        add_marker_line(axins, linewidth=1, include_label=False)
        axins.set_xlim(x1, x2)
        y_zoom = (
            [v for t, v in zip(t_eig_pre_rot, min_eig_pre_rot) if x1 <= t <= x2] +
            [v for t, v in zip(t_eig_post_rot, min_eig_post_rot) if x1 <= t <= x2]
        )
        if y_zoom:
            y_min = min(y_zoom)
            y_max = max(y_zoom)
            pad = 0.05 * max(1e-6, y_max - y_min)
            axins.set_ylim(y_min - pad, y_max + pad)
        axins.grid(True, alpha=0.3)
        axins.tick_params(labelsize=7, pad=1)
        axins.set_facecolor((1.0, 1.0, 1.0, 0.9))
        mark_inset(axp, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

save_all_open_figures()
plt.show()
