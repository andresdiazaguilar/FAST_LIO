import rosbag
import matplotlib.pyplot as plt

# Won't work with the "outdoor_Mainbuilding_10hz_2020-12-24-16-38-00.bag" because it doesn't have the degeneracy topic

# bag_path = "data/andres_data/silo_fail/silo_fail_with_degeneracy_data.bag"
# bag_path = "data/andres_data/cameroon_fail_short/cameroon_0_to_400_fail_with_degeneracy_data.bag"
bag_path = "data/andres_data/cameroon_fail_short/cameroon_100_to_400_fail_with_degeneracy_data.bag"

imu_topic = "/livox/imu"
deg_topic = "/fastlio/degeneracy"

# -------------------------
# Read IMU
# -------------------------
t_imu = []
wx, wy, wz = [], [], []
ax, ay, az = [], [], []

# -------------------------
# Read degeneracy
# -------------------------
t_deg = []
lambda_min = []
cond = []
omega_mean = []
acc_mean = []
effective_feature_number = []
omega_max = []

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

    # Degeneracy
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
        data = msg.data
        if len(data) < 8:
            continue

        t_deg.append(float(data[0]))
        effective_feature_number.append(float(data[1]))
        lambda_min.append(float(data[2]))
        cond.append(float(data[4]))
        omega_mean.append(float(data[5]))
        omega_max.append(float(data[6]))
        acc_mean.append(float(data[7]))

# -------------------------
# Put everything on a shared "time since start" axis
# -------------------------
t0 = min(t_imu[0], t_deg[0]) if len(t_deg) > 0 else t_imu[0]
t_imu = [ti - t0 for ti in t_imu]
t_deg = [ti - t0 for ti in t_deg]

# Focus window and failure time (adjust as needed)
xmin, xmax = 50, 150
t_marker = 100

# Helper to draw the marker consistently
def add_marker_line():
    plt.axvline(x=t_marker, color="r", linestyle="--", linewidth=2, label=f"failure time")

# -------------------------
# 1) Angular velocity + omega_mean
# -------------------------
plt.figure()
plt.plot(t_imu, wx, label="ωx")
plt.plot(t_imu, wy, label="ωy")
plt.plot(t_imu, wz, label="ωz")
plt.plot(t_deg, omega_mean, label="omega_mean (/fastlio/degeneracy)")
add_marker_line()
plt.xlabel("time [s]")
plt.ylabel("angular velocity [rad/s]")
plt.title("IMU Angular Velocity + FAST-LIO omega_mean")
plt.legend()
plt.grid(True)
# plt.xlim(xmin, xmax)

# -------------------------
# 2) Acceleration + acc_mean
# -------------------------
plt.figure()
plt.plot(t_imu, ax, label="ax")
plt.plot(t_imu, ay, label="ay")
plt.plot(t_imu, az, label="az")
plt.plot(t_deg, acc_mean, label="acc_mean (/fastlio/degeneracy)")
add_marker_line()
plt.xlabel("time [s]")
plt.ylabel("linear acceleration [m/s²]")
plt.title("IMU Linear Acceleration + FAST-LIO acc_mean")
plt.legend()
plt.grid(True)
# plt.xlim(xmin, xmax)

# -------------------------
# 3) lambda_min
# -------------------------
plt.figure()
plt.plot(t_deg, lambda_min, label="lambda_min")
add_marker_line()
plt.xlabel("time [s]")
plt.ylabel("lambda_min")
plt.title("FAST-LIO Degeneracy: lambda_min")
plt.legend()
plt.grid(True)
# plt.xlim(xmin, xmax)
# plt.ylim(0, 80000)

# -------------------------
# 4) condition number
# -------------------------
plt.figure()
plt.plot(t_deg, cond, label="cond")
add_marker_line()
plt.xlabel("time [s]")
plt.ylabel("condition number")
plt.title("FAST-LIO Degeneracy: condition number")
plt.legend()
plt.grid(True)
plt.xlim(xmin, xmax)
# plt.ylim(0, 4000000)
# plt.ylim(0, 400000)
plt.ylim(0, 1000)


# -------------------------
# 4) effective feature number
# -------------------------
plt.figure()
plt.plot(t_deg, effective_feature_number, label="effective feature number")
add_marker_line()
plt.xlabel("time [s]")
plt.ylabel("effective feature number")
plt.title("FAST-LIO Degeneracy: effective feature number")
plt.legend()
plt.grid(True)
# plt.xlim(xmin, xmax)
# plt.ylim(0, 800)

plt.show()