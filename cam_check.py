#!/usr/bin/env python3
"""cam_check.py - print what the depth camera sees. The robot does NOT move.

Uses the previous owner's obstacle_avoidance.py depth logic unchanged:
the centre of the depth image (columns 180-460, rows 80-300), readings
between 0.5 and 5 m, nearest = 10th percentile, and left/right halves
compared to say which side the obstacle is on.

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

ROI_X0, ROI_X1, ROI_Y0, ROI_Y1 = 180, 460, 80, 300   # same as obstacle_avoidance.py
MIN_VALID, MAX_VALID = 0.5, 5.0


def p10(v, need):
    return float(np.percentile(v, 10)) if len(v) > need else None


def analyze(depth_mm):
    roi = depth_mm[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1].astype(np.float32) / 1000.0
    ok = (roi > MIN_VALID) & (roi < MAX_VALID)
    out = dict(front=p10(roi[ok], 100), left=None, right=None, side="-", floor=None)
    half = roi.shape[1] // 2
    # Note: in the image, left columns are the robot's LEFT (camera looks forward).
    lv = roi[:, :half][ok[:, :half]]
    rv = roi[:, half:][ok[:, half:]]
    out["left"], out["right"] = p10(lv, 50), p10(rv, 50)
    lm = out["left"] if out["left"] is not None else float("inf")
    rm = out["right"] if out["right"] is not None else float("inf")
    if lm < rm - 0.3:
        out["side"] = "LEFT"
    elif rm < lm - 0.3:
        out["side"] = "RIGHT"
    elif out["front"] is not None:
        out["side"] = "centre"
    h, w = depth_mm.shape
    fl = depth_mm[int(h * 0.85):, w // 2 - 60: w // 2 + 60].astype(np.float32) / 1000.0
    fl = fl[fl > 0.1]
    out["floor"] = float(np.median(fl)) if len(fl) > 200 else None
    out["floor_valid"] = len(fl)
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
        print(f"age {time.time() - self.t:.1f}s frames {self.n} | nearest ahead {fmt(m['front'])}"
              f" | left {fmt(m['left'])} right {fmt(m['right'])} -> obstacle side {m['side']}"
              f" | floor (bottom centre) {fmt(m['floor'])}", flush=True)


def main():
    print("CAMERA CHECK: the robot will not move. Ctrl+C to quit.")
    print("Stand about 1 m in front of the camera, then step to the left and right.\n")
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
