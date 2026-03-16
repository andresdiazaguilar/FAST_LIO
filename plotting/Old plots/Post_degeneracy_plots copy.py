import rosbag
import matplotlib.pyplot as plt

# bag_path = "data/andres_data/cameroon_fail_short/cameroon_0_to_400_fail_with_post_degeneracy_data.bag"
bag_path = "data/andres_data/cameroon_fail_short/with_IMU_initialization_fix/cameroon_0_to_400_fail.bag"

imu_topic = "/livox/imu"
deg_topic = "/fastlio/degeneracy"
post_deg_topic = "/fastlio/degeneracy_post"

# -------------------------
# Plot selection
# Use "all" to render everything, or a set with selected keys.
# Example:
# ENABLED_PLOTS = {"feat_pre_post", "cond_pre", "cond_post"}
# Available keys:
# "imu_omega_pre", "imu_acc_pre", "lambda_min_pre", "lambda_max_pre",
# "cond_pre", "feat_pre_post", "omega_mean_pre", "omega_max_pre",
# "acc_mean_pre", "lambda_scaled_min_pre", "lambda_scaled_max_pre",
# "cond_scaled_pre", "lambda_min_post", "lambda_max_post",
# "cond_post", "ratio_rt_post"
# -------------------------
ENABLED_PLOTS = {"feat_pre_post", "cond_pre", "cond_scaled_pre", "cond_post", "lambda_min_pre", "lambda_min_post"}


def should_plot(key):
    return ENABLED_PLOTS == "all" or key in ENABLED_PLOTS


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
omega_max = []
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
        # 6: omega_max
        # 7: acc_mean
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
        omega_max.append(float(data[6]))
        acc_mean.append(float(data[7]))
        if len(data) >= 12:
            lambda_scaled_min.append(float(data[8]))
            lambda_scaled_max.append(float(data[9]))
            cond_scaled.append(float(data[10]))
            acc_mean_no_grav.append(float(data[11]))
        else:
            lambda_scaled_min.append(float("nan"))
            lambda_scaled_max.append(float("nan"))
            cond_scaled.append(float("nan"))

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

# -------------------------
# Put everything on a shared "time since start" axis
# -------------------------
t0_candidates = []
if len(t_imu) > 0:  t0_candidates.append(t_imu[0])
if len(t_deg) > 0:  t0_candidates.append(t_deg[0])
if len(t_post) > 0: t0_candidates.append(t_post[0])
t0 = min(t0_candidates) if len(t0_candidates) > 0 else 0.0

t_imu = [ti - t0 for ti in t_imu]
t_deg = [ti - t0 for ti in t_deg]
t_post = [ti - t0 for ti in t_post]

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
    plt.plot(t_deg, omega_mean, label="omega_mean (/fastlio/degeneracy)")
    plt.plot(t_deg, omega_max, label="omega_max (/fastlio/degeneracy)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("angular velocity [rad/s]")
    plt.title("IMU Angular Velocity + FAST-LIO omega stats (pre)")
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
    plt.ylabel("linear acceleration [m/s²]")
    plt.title("IMU Linear Acceleration + FAST-LIO acc_mean (pre)")
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
    plt.title("FAST-LIO Degeneracy: lambda_min (pre)")
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
    plt.title("FAST-LIO Degeneracy: lambda_max (pre)")
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
    plt.title("FAST-LIO Degeneracy: condition number (pre)")
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
    plt.title("FAST-LIO Degeneracy: effective feature number (pre + post)")
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
    plt.title("FAST-LIO Degeneracy: omega_mean (pre)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 8) omega_max (pre)
if should_plot("omega_max_pre"):
    plt.figure()
    plt.plot(t_deg, omega_max, label="omega_max (pre)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("omega_max [rad/s]")
    plt.title("FAST-LIO Degeneracy: omega_max (pre)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

# 9) acc_mean (pre)
if should_plot("acc_mean_pre"):
    plt.figure()
    plt.plot(t_deg, acc_mean, label="acc_mean (pre)")
    add_marker_line()
    plt.xlabel("time [s]")
    plt.ylabel("acc_mean [m/s²]")
    plt.title("FAST-LIO Degeneracy: acc_mean (pre)")
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
    plt.title("FAST-LIO Degeneracy: lambda_scaled_min (pre)")
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
    plt.title("FAST-LIO Degeneracy: lambda_scaled_max (pre)")
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
    plt.title("FAST-LIO Degeneracy: cond_scaled (pre)")
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
    plt.title("FAST-LIO Degeneracy: lambda_min (post)")
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
    plt.title("FAST-LIO Degeneracy: lambda_max (post)")
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
    plt.title("FAST-LIO Degeneracy: condition number (post)")
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
    plt.title("FAST-LIO Degeneracy: rotation/translation RMS ratio (post)")
    plt.legend()
    plt.grid(True)
    # plt.xlim(xmin, xmax)

plt.show()
