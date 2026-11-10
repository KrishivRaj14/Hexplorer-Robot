#!/usr/bin/env python3
"""walk_safe.py - walk forward a set distance measured by the LiDAR (Fast-LIO),
stop early if the LiDAR sees something in front, then go into damping mode.

Needs start_fastlio.sh running in another terminal (it provides
/lidar_odometry/pose and /livox/lidar).

  python3 ~/walk_safe.py --check     # sensors only, the robot does NOT move
  python3 ~/walk_safe.py 2.0         # stand up, walk 2.0 m, go limp (damping)
  python3 ~/walk_safe.py 2.0 --end sit   # same, but fold down gently at the end

Ctrl+C at any time: the robot stops walking and finishes the same way.

Why two processes: the sensors use cyclonedds, the robot controller uses
fastrtps, and one ROS node cannot use both. A small helper process (cyclone)
reads the sensors and sends the numbers to this process (fastrtps) over a
local UDP socket.
"""
import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time

import numpy as np

INF = float("inf")


# ---------------------------------------------------------------- geometry
def corridor_stats(pts, xmin, xmax, halfw, zmin, zmax, min_pts=5):
    """pts: Nx3+ array in the LiDAR frame (x forward, y left, z up).

    Returns (front, n, raw, rawz):
      front: distance (m, along x) to the nearest obstacle in the corridor
             in front of the robot, or None if clear. Points must lie in the
             height band zmin..zmax, and at least min_pts must be present.
      n:     number of points in the corridor and height band.
      raw:   nearest corridor point ignoring height (floor included), or None.
      rawz:  height of that point, or None. Used only to diagnose floor leaks.
    """
    p = np.asarray(pts, dtype=np.float32)[:, :3]
    ok = np.isfinite(p).all(axis=1) & np.any(p != 0, axis=1)
    p = p[ok]
    inc = (p[:, 0] > xmin) & (p[:, 0] < xmax) & (np.abs(p[:, 1]) < halfw)
    c = p[inc]
    if len(c) == 0:
        return None, 0, None, None
    i = int(np.argmin(c[:, 0]))
    raw, rawz = float(c[i, 0]), float(c[i, 2])
    f = c[(c[:, 2] > zmin) & (c[:, 2] < zmax)]
    n = int(len(f))
    if n < min_pts:
        return None, n, raw, rawz
    xs = np.sort(f[:, 0])
    return float(xs[min_pts - 1]), n, raw, rawz  # 5th-nearest: ignores stray points


# ------------------------------------------------------------ sensor helper
def run_sensor(a):
    """Runs in its own process with cyclonedds. Sends the latest numbers by UDP."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry

    rclpy.init()
    node = Node("walk_safe_sensor")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=5)
    st = dict(x=None, y=None, pose_t=0.0, front=None, n=0, raw=None, rawz=None,
              lidar_t=0.0, seq=0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def on_pose(m):
        st["x"] = m.pose.pose.position.x
        st["y"] = m.pose.pose.position.y
        st["pose_t"] = time.time()

    def on_cloud(m):
        step = max(m.point_step // 4, 3)
        buf = np.frombuffer(bytes(m.data), dtype=np.float32)
        buf = buf[: (len(buf) // step) * step].reshape(-1, step)
        front, n, raw, rawz = corridor_stats(
            buf, a.xmin, 4.0, a.halfwidth, a.zmin, a.zmax)
        st.update(front=front, n=n, raw=raw, rawz=rawz,
                  lidar_t=time.time(), seq=st["seq"] + 1)

    node.create_subscription(Odometry, "/lidar_odometry/pose", on_pose, qos)
    node.create_subscription(PointCloud2, "/livox/lidar", on_cloud, qos)

    def send():
        sock.sendto(json.dumps(st).encode(), ("127.0.0.1", a.port))

    node.create_timer(0.05, send)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------------ helpers
class Link:
    """Receives the helper's numbers."""

    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.last = None

    def get(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                return self.last
            self.last = json.loads(data)


def ages(m):
    now = time.time()
    return now - m["pose_t"], now - m["lidar_t"]


def sensors_ok(m, pose_max=0.6, lidar_max=1.0):
    if not m or m["x"] is None:
        return False
    pa, la = ages(m)
    return pa < pose_max and la < lidar_max


def fmt(v, unit="m"):
    return "none" if v is None else f"{v:.2f} {unit}"


def start_helper(a):
    env = dict(os.environ, RMW_IMPLEMENTATION=a.sensor_rmw)
    cmd = [sys.executable, os.path.abspath(__file__), "--sensor",
           "--port", str(a.port), "--xmin", str(a.xmin),
           "--halfwidth", str(a.halfwidth), "--zmin", str(a.zmin),
           "--zmax", str(a.zmax)]
    return subprocess.Popen(cmd, env=env)


# ------------------------------------------------------------------- check
def run_check(a, link):
    print("CHECK MODE: the robot will not move. Ctrl+C to quit.")
    print("Clear floor ahead should read 'obstacle: none'. Hold a box or your")
    print("leg about 0.5-1 m in front of the robot: it should show a distance.")
    print("'incl. floor' shows what the LiDAR sees ignoring height.\n")
    while True:
        time.sleep(0.5)
        m = link.get()
        if not m or m["x"] is None:
            print("waiting for sensor data (is start_fastlio.sh running?)")
            continue
        pa, la = ages(m)
        rz = "" if m["rawz"] is None else f" at height {m['rawz']:+.2f} m"
        print(f"pose x={m['x']:+.2f} y={m['y']:+.2f} (age {pa:.1f}s) | "
              f"lidar age {la:.1f}s | obstacle: {fmt(m['front'])} "
              f"({m['n']} pts) | incl. floor: {fmt(m['raw'])}{rz}")


# -------------------------------------------------------------------- walk
def run_walk(a, link):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from custom_msg.msg import RobotCommand

    # Do not stand up unless the sensors are alive.
    print("Waiting for sensors...")
    t0 = time.time()
    while time.time() - t0 < 10:
        time.sleep(0.2)
        if sensors_ok(link.get()):
            break
    else:
        print("ERROR: no fresh LiDAR/pose data. Is start_fastlio.sh running "
              "in another terminal and printing 'Relayed ... odom messages'?")
        return 1
    m = link.get()
    if m["front"] is not None and m["front"] < a.stop:
        print(f"ERROR: obstacle already {m['front']:.2f} m ahead. Not starting.")
        return 1

    rclpy.init()
    node = Node("walk_safe")
    cmd_pub = node.create_publisher(RobotCommand, "/robot_cmd", 10)
    vel_pub = node.create_publisher(Twist, "/vel_cmd", 10)

    def send(state, vx=0.0):
        c = RobotCommand()
        c.header.stamp = node.get_clock().now().to_msg()
        c.target_state = state
        v = Twist()
        v.linear.x = float(vx)
        cmd_pub.publish(c)
        vel_pub.publish(v)

    def hold(state, seconds, vx=0.0):
        for _ in range(int(seconds * 20)):
            send(state, vx)
            time.sleep(0.05)

    reason = "unknown"
    moved = lat = 0.0
    x0 = y0 = None
    standing = False
    try:
        print("Standing up (about 6 s)...")
        standing = True
        for s in (1, 2, 3):
            hold(s, 2.0)

        m = link.get()
        if not sensors_ok(m):
            raise RuntimeError("sensor data lost while standing up")
        x0, y0 = m["x"], m["y"]
        print(f"Walking {a.distance:.2f} m at {a.speed:.2f} m/s "
              f"(stop for obstacles closer than {a.stop:.2f} m)")

        timeout = a.distance / a.speed * 2.0 + 15.0
        t_start = time.time()
        last_seq, blocked = -1, 0
        last_print = 0.0
        while True:
            m = link.get()
            if not sensors_ok(m):
                reason = "STOPPED: sensor data lost"
                break
            dx, dy = m["x"] - x0, m["y"] - y0
            moved = math.hypot(dx, dy)
            if m["seq"] != last_seq:                  # count each LiDAR frame once
                last_seq = m["seq"]
                if m["front"] is not None and m["front"] < a.stop:
                    blocked += 1
                else:
                    blocked = 0
            if blocked >= 2:
                reason = f"STOPPED: obstacle {m['front']:.2f} m ahead"
                break
            if moved >= a.distance - a.lead:
                reason = "reached the target distance"
                break
            if time.time() - t_start > timeout:
                reason = "STOPPED: timeout"
                break
            remaining = a.distance - moved
            speed = a.speed if remaining > 0.3 else min(a.speed, 0.10)
            send(4, speed)
            if time.time() - last_print > 1.0:
                last_print = time.time()
                print(f"  moved {moved:.2f} m, obstacle ahead: {fmt(m['front'])}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        reason = "STOPPED: Ctrl+C"
    except Exception as e:  # noqa
        reason = f"STOPPED: error {e}"
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # finish stopping safely
        try:
            hold(4, 1.5, 0.0)                         # stop walking
            m = link.get()
            if m and m["x"] is not None and x0 is not None:
                moved = math.hypot(m["x"] - x0, m["y"] - y0)
                lat = m["y"] - y0
            print(f"\nResult: {reason}")
            print(f"LiDAR says it moved {moved:.2f} m forward "
                  f"({a.distance:.2f} m asked), sideways {lat:+.2f} m")
            if standing and a.end == "sit":
                print("Sitting down...")
                for s in (3, 1, 0):
                    hold(s, 2.0)
            elif standing:
                print("Damping mode (the robot goes limp)...")
                hold(0, 2.0)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    return 0 if reason.startswith("reached") else 2


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("distance", nargs="?", type=float, default=None,
                    help="meters to walk forward")
    ap.add_argument("--check", action="store_true",
                    help="sensors only, robot does not move")
    ap.add_argument("--sensor", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--speed", type=float, default=0.15)
    ap.add_argument("--stop", type=float, default=0.8,
                    help="stop if an obstacle is nearer than this, in m ahead of the LiDAR")
    ap.add_argument("--lead", type=float, default=0.06,
                    help="stop this far early to allow for coasting")
    ap.add_argument("--xmin", type=float, default=0.30)
    ap.add_argument("--halfwidth", type=float, default=0.40)
    ap.add_argument("--zmin", type=float, default=-0.25,
                    help="lowest obstacle height, relative to the LiDAR")
    ap.add_argument("--zmax", type=float, default=0.80)
    ap.add_argument("--end", choices=["damping", "sit"], default="damping",
                    help="how to finish: damping (go limp) or sit (fold down first)")
    ap.add_argument("--port", type=int, default=45678)
    ap.add_argument("--sensor-rmw", default="rmw_cyclonedds_cpp")
    a = ap.parse_args()

    if a.sensor:
        run_sensor(a)
        return 0
    if not a.check and a.distance is None:
        ap.error("give a distance in meters, or use --check")
    if not a.check:
        if not (0.2 <= a.distance <= 5.0):
            ap.error("distance must be 0.2 to 5.0 m for now")
        a.speed = max(0.05, min(a.speed, 0.25))

    link = Link(a.port)
    helper = start_helper(a)
    try:
        return run_check(a, link) if a.check else run_walk(a, link)
    except KeyboardInterrupt:
        return 0
    finally:
        helper.terminate()


if __name__ == "__main__":
    sys.exit(main())
