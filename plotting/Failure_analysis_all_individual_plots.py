import rosbag
import matplotlib.pyplot as plt
from bisect import bisect_left

bag_path = "data/andres_data/cameroon_fail_short/with_IMU_initialization_fix/cameroon_0_to_400_fail_full.bag"

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

imu = {"imu_omega_pre", "imu_omega_pre_running_avg", "imu_acc_pre", "gyro_biases", "accel_biases"}

feature_count = {"feat_pre_post", "edge_feature_count", "plane_feature_count", "inlier_ratio"}

gravity = {"accel_gravity_consistency", "gravity_estimate"}


ENABLED_PLOTS = cond_numbers_and_min_eigenvalues | all_eigenvalues | imu | feature_count | gravity | "residual_stats"

OMEGA_RUNNING_AVG_WINDOW = 200


def should_plot(key):
    return ENABLED_PLOTS == "all" or key in ENABLED_PLOTS


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

# Focus window and failure time (adjust as needed)
xmin, xmax = 50, 150
t_marker = 145

def add_marker_line():
    plt.axvline(x=t_marker, color="r", linestyle="--", linewidth=2, label="failure time")

# =========================
# PRE plots
# =========================

# 1) Angular velocity + omega_mean (pre)
if should_plot("imu_omega_pre"):
    plt.figure()
    plt.plot(t_imu, wx, label="ωx")
    plt.plot(t_imu, wy, label="ωy")
    plt.plot(t_imu, wz, label="ωz")
    plt.plot(t_deg, omega_mean, label="omega_mean")
    # plt.plot(t_deg, omega_max, label="omega_max")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("IMU angular velocity [rad/s]")
    plt.title("IMU Gyro Angular Velocity (rad/s)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 2) Acceleration + acc_mean (pre)
if should_plot("imu_acc_pre"):
    plt.figure()
    plt.plot(t_imu, ax, label="ax")
    plt.plot(t_imu, ay, label="ay")
    plt.plot(t_imu, az, label="az")
    plt.plot(t_deg, acc_mean, label="acc_mean")
    plt.plot(t_deg, acc_mean_no_grav, label="acc_mean_no_grav")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("linear acceleration [g]")
    plt.title("IMU Linear Acceleration (g)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 2b) Angular velocity + omega_mean (pre), running average
if should_plot("imu_omega_pre_running_avg"):
    wx_avg = running_average(wx, OMEGA_RUNNING_AVG_WINDOW)
    wy_avg = running_average(wy, OMEGA_RUNNING_AVG_WINDOW)
    wz_avg = running_average(wz, OMEGA_RUNNING_AVG_WINDOW)
    omega_mean_avg = running_average(omega_mean, OMEGA_RUNNING_AVG_WINDOW)

    plt.figure()
    plt.plot(t_imu, wx_avg, label="ωx (running avg)")
    plt.plot(t_imu, wy_avg, label="ωy (running avg)")
    plt.plot(t_imu, wz_avg, label="ωz (running avg)")
    plt.plot(t_deg, omega_mean_avg, label="omega_mean (running avg)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("angular velocity [rad/s]")
    plt.title(f"IMU Gyro Angular Velocity [rad/s] (Running Average, window={OMEGA_RUNNING_AVG_WINDOW})")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

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

# 6) effective feature number (pre + post)
if should_plot("feat_pre_post"):
    plt.figure()
    plt.plot(t_deg, effective_feature_number, label="effective feature number (pre)")
    plt.plot(t_post, post_eff, label="effective feature number (post)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("effective feature number")
    plt.title("Effective Feature Number (pre + post)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

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

# 18) Gyro biases
if should_plot("gyro_biases"):
    plt.figure()
    plt.plot(t_gyro_bias, gyro_bx, label="bg_x [rad/s]")
    plt.plot(t_gyro_bias, gyro_by, label="bg_y [rad/s]")
    plt.plot(t_gyro_bias, gyro_bz, label="bg_z [rad/s]")
    plt.plot(t_gyro_bias, gyro_bnorm, label="||bg|| [rad/s]")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("gyro bias [rad/s]")
    plt.title("FAST-LIO: Gyro Biases [rad/s]")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 19) Accel biases
if should_plot("accel_biases"):
    plt.figure()
    plt.plot(t_accel_bias, accel_bx, label="ba_x [g]")
    plt.plot(t_accel_bias, accel_by, label="ba_y [g]")
    plt.plot(t_accel_bias, accel_bz, label="ba_z [g]")
    plt.plot(t_accel_bias, accel_bnorm, label="||ba|| [g]")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("accel bias [g]")
    plt.title("Accel Biases [g]")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 20) Inlier ratio
if should_plot("inlier_ratio"):
    plt.figure()
    plt.plot(t_inlier_ratio, inlier_ratio, label="inlier ratio")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("inlier ratio")
    plt.title("Inlier Ratio")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)
    # plt.ylim(0.0, 1.05)

# 21) Residual stats
if should_plot("residual_stats"):
    plt.figure()
    plt.plot(t_residual, residual_median, label="residual median")
    plt.plot(t_residual, residual_p95, label="residual p95")
    plt.plot(t_residual, residual_mean, label="residual mean")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("residual")
    plt.title("Residuals (median, p95, mean)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 22) Gravity estimate
if should_plot("gravity_estimate"):
    plt.figure()
    plt.plot(t_gravity, gravity_x, label="g_x [m/s²]")
    plt.plot(t_gravity, gravity_y, label="g_y [m/s²]")
    plt.plot(t_gravity, gravity_z, label="g_z [m/s²]")
    plt.plot(t_gravity, gravity_norm, label="||g|| [m/s²]")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("gravity estimate [m/s²]")
    plt.title("Gravity Estimate [m/s²]")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 23) Edge feature counts
if should_plot("edge_feature_count"):
    plt.figure()
    plt.plot(t_edge_count, edge_count, label="edge feature count")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("count")
    plt.title("Edge Feature Count")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 24) a_orth + |a_parallel - ||g|||
if should_plot("accel_gravity_consistency"):
    t_accel_consistency = []
    abs_a_parallel_minus_g = []
    if len(t_gravity) > 0 and len(t_accel_gravity_comp) > 0:
        for i, t_comp in enumerate(t_accel_gravity_comp):
            j = nearest_index(t_gravity, t_comp)
            t_accel_consistency.append(t_comp)
            abs_a_parallel_minus_g.append(abs(a_parallel[i]) - abs(gravity_norm[j]))

    plt.figure()
    plt.plot(t_accel_gravity_comp, a_orth, label="a_orth [m/s²]")
    # plt.plot(t_accel_gravity_comp, a_parallel, label="a_parallel [m/s²]")
    plt.plot(t_accel_consistency, abs_a_parallel_minus_g, label="|a_parallel| - |g| [m/s²]")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("acceleration [m/s²]")
    plt.title("Accel vs Gravity Consistency")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 25) Eigenvalues pre-translation (3x3)
if should_plot("eigvals_pre_translation"):
    plt.figure()
    plt.plot(t_eig_pre_trans, eig_pre_trans_1, label="lambda1")
    plt.plot(t_eig_pre_trans, eig_pre_trans_2, label="lambda2")
    plt.plot(t_eig_pre_trans, eig_pre_trans_3, label="lambda3")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("eigenvalue")
    plt.title("Pre Eigenvalues (Translation Block)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 26) Eigenvalues pre-rotation (3x3)
if should_plot("eigvals_pre_rotation"):
    plt.figure()
    plt.plot(t_eig_pre_rot, eig_pre_rot_1, label="lambda1")
    plt.plot(t_eig_pre_rot, eig_pre_rot_2, label="lambda2")
    plt.plot(t_eig_pre_rot, eig_pre_rot_3, label="lambda3")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("eigenvalue")
    plt.title("Pre Eigenvalues (Rotation Block)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 27) Eigenvalues post-translation (3x3)
if should_plot("eigvals_post_translation"):
    plt.figure()
    plt.plot(t_eig_post_trans, eig_post_trans_1, label="lambda1")
    plt.plot(t_eig_post_trans, eig_post_trans_2, label="lambda2")
    plt.plot(t_eig_post_trans, eig_post_trans_3, label="lambda3")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("eigenvalue")
    plt.title("Post Eigenvalues (Translation Block)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 28) Eigenvalues post-rotation (3x3)
if should_plot("eigvals_post_rotation"):
    plt.figure()
    plt.plot(t_eig_post_rot, eig_post_rot_1, label="lambda1")
    plt.plot(t_eig_post_rot, eig_post_rot_2, label="lambda2")
    plt.plot(t_eig_post_rot, eig_post_rot_3, label="lambda3")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("eigenvalue")
    plt.title("Post Eigenvalues (Rotation Block)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 29) Plane feature counts
if should_plot("plane_feature_count"):
    plt.figure()
    plt.plot(t_plane_count, plane_count, label="plane feature count")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("count")
    plt.title("Plane Feature Count")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 30) Condition number translation (pre + post)
if (should_plot("cond_eigvals_translation") or
        should_plot("cond_eigvals_pre_translation") or
        should_plot("cond_eigvals_post_translation")):
    plt.figure()
    plt.plot(t_eig_pre_trans, cond_eig_pre_trans, label="cond (pre translation)")
    plt.plot(t_eig_post_trans, cond_eig_post_trans, label="cond (post translation)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("condition number")
    plt.title("Condition Number (Translation Block, Pre + Post)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    # plt.xlim(xmin, xmax)

# 31) Condition number rotation (pre + post)
if (should_plot("cond_eigvals_rotation") or
        should_plot("cond_eigvals_pre_rotation") or
        should_plot("cond_eigvals_post_rotation")):
    plt.figure()
    plt.plot(t_eig_pre_rot, cond_eig_pre_rot, label="cond (pre rotation)")
    plt.plot(t_eig_post_rot, cond_eig_post_rot, label="cond (post rotation)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("condition number")
    plt.title("Condition Number (Rotation Block, Pre + Post)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    # plt.xlim(xmin, xmax)

# 32) Minimum eigenvalue translation (pre + post)
if (should_plot("min_eigvals_translation") or
        should_plot("min_eigvals_pre_translation") or
        should_plot("min_eigvals_post_translation")):
    plt.figure()
    plt.plot(t_eig_pre_trans, min_eig_pre_trans, label="lambda_min (pre translation)")
    plt.plot(t_eig_post_trans, min_eig_post_trans, label="lambda_min (post translation)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("minimum eigenvalue")
    plt.title("Minimum Eigenvalue (Translation Block, Pre + Post)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 33) Minimum eigenvalue rotation (pre + post)
if (should_plot("min_eigvals_rotation") or
        should_plot("min_eigvals_pre_rotation") or
        should_plot("min_eigvals_post_rotation")):
    plt.figure()
    plt.plot(t_eig_pre_rot, min_eig_pre_rot, label="lambda_min (pre rotation)")
    plt.plot(t_eig_post_rot, min_eig_post_rot, label="lambda_min (post rotation)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("minimum eigenvalue")
    plt.title("Minimum Eigenvalue (Rotation Block, Pre + Post)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

plt.show()
