import numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
rclpy.init(); n = Node('lidar_where')
frames = []
def cb(m):
    step = max(m.point_step // 4, 3)
    b = np.frombuffer(bytes(m.data), dtype=np.float32)
    frames.append(b[:(len(b)//step)*step].reshape(-1, step)[:, :3])
n.create_subscription(PointCloud2, '/livox/lidar', cb, QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5))
def grab(k):
    frames.clear()
    while len(frames) < k + 3: rclpy.spin_once(n, timeout_sec=1.0)
    p = np.vstack(frames[3:])
    return p[np.isfinite(p).all(1) & (p != 0).any(1)]
EDGES = np.arange(-180, 181, 30)
def sectors(p):
    r = np.hypot(p[:, 0], p[:, 1]); keep = (r > 0.6) & (r < 3.0)
    ang = np.degrees(np.arctan2(p[keep, 1], p[keep, 0]))
    return np.histogram(ang, bins=EDGES)[0], ang, r[keep]
print('Keep the area around the robot CLEAR. Measuring...')
base, _, _ = sectors(grab(10))
input('Now stand a box/chair about 1 m in front of the robot\'s FRONT (the LiDAR side), then press Enter: ')
new, ang, r = sectors(grab(10))
diff = new - base
print('bearing (sensor frame, 0 = sensor x axis)  -> new points')
for i, d in enumerate(diff):
    bar = '#' * int(max(d, 0) // 50)
    print(f'{EDGES[i]:+4d}..{EDGES[i+1]:+4d} deg : {d:+6d} {bar}')
k = int(np.argmax(diff))
sel = (ang >= EDGES[k]) & (ang < EDGES[k+1])
print(f'\nThe new object is at bearing {EDGES[k]:+d}..{EDGES[k+1]:+d} degrees from the sensor x axis.')
print('If that is -30..0 or 0..+30, the sensor x axis points the way the robot faces.')
