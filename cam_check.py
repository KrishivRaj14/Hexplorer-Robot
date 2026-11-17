#!/usr/bin/env python3
"""cam_check.py (v2) - print what the depth camera sees. The robot does NOT move.

Based on the previous owner's obstacle_avoidance.py depth logic (rows 80-300,
readings 0.5-5 m, 10th percentile), with one change: his box was only the
centre 280 of 640 columns (about 35 degrees wide), and one percentile over the
whole box misses anything that fills only a small part of it. v2 uses almost
the full width (columns 20-620) cut into 8 vertical strips, takes the 10th
percentile of each strip, and the nearest strip is the obstacle.

Also prints the distance to the floor in the bottom centre of the image
(for stair/drop detection later): it should stay about the same on a flat
floor and jump much larger, or show no reading, at a drop.

Needs depth_bridge_receiver.py running (publishes /camera/depth/image_raw).

  python3 ~/cam_check.py
"""
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

ROI_X0, ROI_X1, ROI_Y0, ROI_Y1 = 20, 620, 80, 300   # his rows; almost full width
MIN_VALID, MAX_VALID = 0.5, 5.0
STRIPS = 8
FX = 337.0          # approx. focal length in pixels for an 87 deg wide 640 px image


def p10(v, need):
    return float(np.percentile(v, 10)) if len(v) > need else None


def analyze(depth_mm):
    h, w = depth_mm.shape
    roi = depth_mm[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1].astype(np.float32) / 1000.0
    ok = (roi > MIN_VALID) & (roi < MAX_VALID)
    edges = np.linspace(0, roi.shape[1], STRIPS + 1).astype(int)
    strips, angles = [], []
    for k in range(STRIPS):
        a, b = edges[k], edges[k + 1]
        strips.append(p10(roi[:, a:b][ok[:, a:b]], 100))
        u = ROI_X0 + (a + b) / 2.0                       # strip centre column
        angles.append(float(np.degrees(np.arctan((w / 2.0 - u) / FX))))  # + = left
    out = dict(strips=strips, angles=angles, front=None, angle=None, side="-")
    vals = [(v, k) for k, v in enumerate(strips) if v is not None]
    if vals:
        v, k = min(vals)
        out["front"], out["angle"] = v, angles[k]
        # side of the nearest strip, only if clearly nearer than the other half
        left = [x for x in strips[:STRIPS // 2] if x is not None]
        right = [x for x in strips[STRIPS // 2:] if x is not None]
        lm = min(left) if left else float("inf")
        rm = min(right) if right else float("inf")
        out["side"] = "LEFT" if lm < rm - 0.3 else "RIGHT" if rm < lm - 0.3 else "centre"
    fl = depth_mm[int(h * 0.85):, w // 2 - 60: w // 2 + 60].astype(np.float32) / 1000.0
    fl = fl[fl > 0.1]
    out["floor"] = float(np.median(fl)) if len(fl) > 200 else None
    return out


def fmt(v):
    return "none" if v is None else f"{v:.2f} m"


class Check(Node):
    def __init__(self):
        super().__init__("cam_check")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, "/camera/depth/image_raw", self.on_depth, qos)
        self.last, self.t, self.n = None, 0.0, 0
        self.create_timer(0.5, self.show)

    def on_depth(self, msg):
        d = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)
        self.last, self.t = analyze(d), time.time()
        self.n += 1

    def show(self):
        if self.last is None:
            print("waiting for depth images (is depth_bridge_receiver.py running?)", flush=True)
            return
        m = self.last
        bar = " ".join("  -- " if v is None else f"{v:4.2f}" for v in m["strips"])
        where = "" if m["angle"] is None else f" at about {m['angle']:+.0f} deg"
        print(f"age {time.time() - self.t:.1f}s frames {self.n} | strips L->R [{bar}]"
              f" | nearest {fmt(m['front'])}{where} side {m['side']}"
              f" | floor {fmt(m['floor'])}", flush=True)


def Check_angles():
    edges = np.linspace(0, ROI_X1 - ROI_X0, STRIPS + 1)
    return [float(np.degrees(np.arctan((320 - (ROI_X0 + (edges[k] + edges[k + 1]) / 2)) / FX)))
            for k in range(STRIPS)]


def main():
    print("CAMERA CHECK: the robot will not move. Ctrl+C to quit.")
    print("8 strips from the robot's LEFT to its RIGHT, about 85 degrees in total.")
    print(f"Strip centres at about: {', '.join(f'{a:+.0f}' for a in Check_angles())} deg (+ = left)\n")
    rclpy.init()
    node = Check()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
