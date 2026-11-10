import numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
rclpy.init(); n = Node('lidar_hist')
frames = []
def cb(m):
    step = max(m.point_step // 4, 3)
    b = np.frombuffer(bytes(m.data), dtype=np.float32)
    b = b[:(len(b)//step)*step].reshape(-1, step)[:, :3]
    frames.append(b)
n.create_subscription(PointCloud2, '/livox/lidar', cb, QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5))
while len(frames) < 10: rclpy.spin_once(n, timeout_sec=1.0)
p = np.vstack(frames); p = p[np.isfinite(p).all(1) & (p != 0).any(1)]
print('points', len(p))
print('--- forward strip (|y|<0.4): distance ahead -> points, mean height, min/max height')
for lo in np.arange(0.0, 2.0, 0.1):
    s = p[(p[:,0]>=lo)&(p[:,0]<lo+0.1)&(abs(p[:,1])<0.4)]
    if len(s): print(f'x {lo:.1f}-{lo+0.1:.1f} m: {len(s):5d} pts  z mean {s[:,2].mean():+.2f}  [{s[:,2].min():+.2f}, {s[:,2].max():+.2f}]')
print('--- all around: nearest point in each direction')
ang = np.degrees(np.arctan2(p[:,1], p[:,0])); d = np.hypot(p[:,0], p[:,1])
for a0 in range(-180, 180, 45):
    s = d[(ang>=a0)&(ang<a0+45)]
    print(f'direction {a0:+4d}..{a0+45:+4d} deg (0 = forward): nearest {s.min():.2f} m' if len(s) else f'direction {a0}: none')
