import rosbag
import matplotlib.pyplot as plt

bag_path = "data/example_data/outdoor_Mainbuilding_10hz_2020-12-24-16-38-00.bag"
topic = "/livox/imu"

t = []
wx, wy, wz = [], [], []
ax, ay, az = [], [], []

with rosbag.Bag(bag_path, "r") as bag:
    for _, msg, stamp in bag.read_messages(topics=[topic]):
        t.append(msg.header.stamp.to_sec())
        # t.append(stamp.to_sec()) # incorrect, what PlotJuggler does

        wx.append(msg.angular_velocity.x)
        wy.append(msg.angular_velocity.y)
        wz.append(msg.angular_velocity.z)

        ax.append(msg.linear_acceleration.x)
        ay.append(msg.linear_acceleration.y)
        az.append(msg.linear_acceleration.z)

t0 = t[0]
t = [ti - t0 for ti in t]

plt.figure()
plt.plot(t, wx, label="ωx")
plt.plot(t, wy, label="ωy")
plt.plot(t, wz, label="ωz")
plt.xlabel("time [s]")
plt.ylabel("angular velocity [rad/s]")
plt.legend()
plt.grid(True)
plt.xlim(10, 20)
plt.title("IMU Angular Velocity")


plt.figure()
plt.plot(t, ax, label="ax")
plt.plot(t, ay, label="ay")
plt.plot(t, az, label="az")
plt.xlabel("time [s]")
plt.ylabel("linear acceleration [m/s²]")
plt.legend()
plt.grid(True)
plt.xlim(10, 20)
plt.title("IMU Linear Acceleration")


plt.show()