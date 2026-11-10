#!/usr/bin/env python3
"""walk_safe.py - walk forward a set distance measured by the LiDAR (Fast-LIO),
stop early if the LiDAR sees something in front, then go into damping mode.

Needs start_fastlio.sh running in another terminal (it provides
/lidar_odometry/pose and /livox/lidar).

  python3 ~/walk_safe.py --check         # sensors only, the robot does NOT move
  python3 ~/walk_safe.py 2.0             # stand up, walk 2.0 m, go limp (damping)

However it ends (target reached, obstacle, Ctrl+C, a dropped SSH connection, an
error), the robot stops walking and ends in damping mode.

The LiDAR is tilted on the robot, so the floor is not at a fixed height in the
LiDAR's view. Every frame the script finds the floor itself (the lowest surface) and
only counts points that stick up from it as obstacles. It ignores the robot's
own body (closer than --xmin) and stops if it cannot see the floor.

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


# ---------------------------------------------------------------- geometry
def fit_floor(p, rng, cell=0.25):
    """Find the floor plane z = a*x + b*y + c in LiDAR coordinates.

    The floor is the LOWEST surface the LiDAR sees. So for every 25 cm square
    of ground we keep its 2nd-lowest point (ignoring one stray point), then fit
    a plane to those points with RANSAC and refine it.
    Returns ((a, b, c) or None, text explaining the result).
    """
    if len(p) < 400:
        return None, f"only {len(p)} points"
    ix = np.floor(p[:, 0] / cell).astype(np.int64) + 200
    iy = np.floor(p[:, 1] / cell).astype(np.int64) + 200
    key = ix * 1000 + iy
    order = np.lexsort((p[:, 2], key))
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    sizes = np.diff(np.r_[starts, len(ks)])
    ok = sizes >= 6
    Q = p[order[starts[ok] + 1]].astype(np.float64)      # 2nd-lowest per cell
    nq = len(Q)
    if nq < 15:
        return None, f"only {nq} ground cells"
    idx = rng.integers(0, nq, (400, 3))
    A, B, C = Q[idx[:, 0]], Q[idx[:, 1]], Q[idx[:, 2]]
    nr = np.cross(B - A, C - A)
    ln = np.linalg.norm(nr, axis=1)
    good = ln > 1e-6
    nr = nr / np.where(good, ln, 1.0)[:, None]
    nr[nr[:, 2] < 0] *= -1
    d = -np.einsum("ij,ij->i", nr, A)
    good &= nr[:, 2] > 0.82
    good &= (d > 0.05 * nr[:, 2]) & (d < 1.2 * nr[:, 2])
    if not good.any():
        return None, f"no level plane among {nq} ground cells"
    dist = np.abs(Q @ nr.T + d)
    inl = (dist < 0.04).sum(axis=0)
    inl[~good] = 0
    k = int(np.argmax(inl))
    if inl[k] < max(20, int(0.25 * nq)):
        return None, f"best plane fits only {int(inl[k])} of {nq} ground cells"
    m = dist[:, k] < 0.04
    for _ in range(3):
        X = np.c_[Q[m, 0], Q[m, 1], np.ones(int(m.sum()))]
        coef, *_ = np.linalg.lstsq(X, Q[m, 2], rcond=None)
        res = np.abs(Q[:, 2] - (Q[:, 0] * coef[0] + Q[:, 1] * coef[1] + coef[2]))
        m = res < 0.04
        if m.sum() < 12:
            return None, "plane fit collapsed"
    G = Q[m]
    below = int((Q[:, 2] - (Q[:, 0] * coef[0] + Q[:, 1] * coef[1] + coef[2]) < -0.08).sum())
    if below > max(5, 0.1 * nq):
        return None, f"{below} ground cells lie below the best plane (not the floor)"
    near = G[np.hypot(G[:, 0], G[:, 1]) < 2.0]
    if len(near) < 8 or np.ptp(np.hypot(G[:, 0], G[:, 1])) < 0.8:
        return None, "no floor seen within 2 m"
    a, b, c = (float(v) for v in coef)
    if 1.0 / math.sqrt(1 + a * a + b * b) < 0.82 or not (-1.2 < c < -0.05):
        return None, f"implausible floor (LiDAR {-c:.2f} m up)"
    return (a, b, c), f"{int(m.sum())} of {nq} ground cells, {below} below"


def analyze(pts, a, rng, plane):
    """Look at one LiDAR frame.

    Returns a dict:
      floor_ok: the floor plane was found (or reused for up to 1 s)
      info:     short text about the floor fit (for debugging)
      tilt:     how far the floor is tilted in the LiDAR's view (degrees)
      clr:      LiDAR height above the floor (m)
      front:    distance ahead (m) of the nearest obstacle in the corridor in
                front of the robot, or None. An obstacle is at least 5 points
                between --zmin and --zmax above the floor.
      n:        number of obstacle points in the corridor
      raw:      nearest point >5 cm above the floor in the corridor (diagnostic)
    plane: dict holding the last good floor fit; updated in place.
    """
    p = np.asarray(pts, dtype=np.float32)[:, :3]
    p = p[np.isfinite(p).all(axis=1) & np.any(p != 0, axis=1)]
    now = time.time()
    r = np.hypot(p[:, 0], p[:, 1])
    coef, info = fit_floor(p[(p[:, 0] > a.xmin) & (r < 3.5)], rng)   # floor ahead only
    if coef is not None:
        plane["coef"], plane["t"] = coef, now
    elif plane["coef"] is None or now - plane["t"] > 1.0:
        return dict(floor_ok=False, info=info, tilt=None, clr=None, front=None,
                    n=0, raw=None)
    ca, cb, cc = plane["coef"]
    h = p[:, 2] - (ca * p[:, 0] + cb * p[:, 1] + cc)      # height above floor
    inc = (p[:, 0] > a.xmin) & (p[:, 0] < 4.0) & (np.abs(p[:, 1]) < a.halfwidth)
    c, hc = p[inc], h[inc]
    out = dict(floor_ok=True, info=info,
               tilt=float(np.degrees(np.arctan(np.hypot(ca, cb)))),
               clr=float(-cc), front=None, n=0, raw=None)
    low = c[hc > 0.05]
    if len(low):
        out["raw"] = float(low[:, 0].min())
    f = c[(hc > a.zmin) & (hc < a.zmax)]
    out["n"] = int(len(f))
    if len(f) >= 5:
        out["front"] = float(np.sort(f[:, 0])[4])          # 5th nearest: ignores strays
    return out


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
    rng = np.random.default_rng(1)
    plane = {"coef": None, "t": 0.0}
    st = dict(x=None, y=None, pose_t=0.0, floor_ok=False, info="", tilt=None,
              clr=None, front=None, n=0, raw=None, lidar_t=0.0, seq=0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def on_pose(m):
        st["x"] = m.pose.pose.position.x
        st["y"] = m.pose.pose.position.y
        st["pose_t"] = time.time()

    def on_cloud(m):
        step = max(m.point_step // 4, 3)
        buf = np.frombuffer(bytes(m.data), dtype=np.float32)
        buf = buf[: (len(buf) // step) * step].reshape(-1, step)
        res = analyze(buf, a, rng, plane)
        st.update(res)
        st["lidar_t"] = time.time()
        st["seq"] += 1

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


def say(*args):
    """print() that never raises (the terminal may be gone after an SSH drop)."""
    try:
        print(*args, flush=True)
    except Exception:  # noqa
        pass


def stop_signal(signum, frame):
    raise KeyboardInterrupt


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
    print("With a clear floor ahead it should say 'obstacle: none'. Hold a box or")
    print("your leg 0.5-1.5 m in front of the robot: it should show a distance.")
    print("'floor' shows the LiDAR height above the floor and its tilt.\n")
    while True:
        time.sleep(0.5)
        m = link.get()
        if not m or m["x"] is None:
            print("waiting for sensor data (is start_fastlio.sh running?)")
            continue
        pa, la = ages(m)
        if m["floor_ok"]:
            fl = f"floor ok (LiDAR {m['clr']:.2f} m up, tilt {m['tilt']:.0f} deg)"
        else:
            fl = f"FLOOR NOT FOUND ({m.get('info', '')})"
        print(f"pose x={m['x']:+.2f} y={m['y']:+.2f} (age {pa:.1f}s) | "
              f"lidar age {la:.1f}s | {fl} | obstacle: {fmt(m['front'])} "
              f"({m['n']} pts) | nearest bump: {fmt(m['raw'])}")


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
        m = link.get()
        if sensors_ok(m) and m["floor_ok"]:
            break
    else:
        print("ERROR: no fresh LiDAR/pose data, or the floor cannot be found. "
              "Is start_fastlio.sh running in another terminal and printing "
              "'Relayed ... odom messages'? Try --check.")
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
        last_seq, blocked, nofloor = -1, 0, 0
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
                nofloor = 0 if m["floor_ok"] else nofloor + 1
                if m["floor_ok"] and m["front"] is not None and m["front"] < a.stop:
                    blocked += 1
                else:
                    blocked = 0
            if nofloor >= 8:
                reason = "STOPPED: cannot see the floor"
                break
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
        # Nothing can interrupt the stop-and-damp sequence from here on.
        for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sg, signal.SIG_IGN)
        try:
            try:
                hold(4, 1.0, 0.0)                     # stop walking
                m = link.get()
                if m and m["x"] is not None and x0 is not None:
                    moved = math.hypot(m["x"] - x0, m["y"] - y0)
                    lat = m["y"] - y0
                say(f"\nResult: {reason}")
                say(f"LiDAR says it moved {moved:.2f} m forward "
                    f"({a.distance:.2f} m asked), sideways {lat:+.2f} m")
            finally:
                if standing:
                    say("Damping mode (the robot goes limp)...")
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
    ap.add_argument("--xmin", type=float, default=0.60,
                    help="ignore everything closer than this (the robot's own body)")
    ap.add_argument("--halfwidth", type=float, default=0.40)
    ap.add_argument("--zmin", type=float, default=0.10,
                    help="smallest obstacle height ABOVE THE FLOOR")
    ap.add_argument("--zmax", type=float, default=1.00,
                    help="tallest obstacle height above the floor")
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

    signal.signal(signal.SIGTERM, stop_signal)   # kill / service stop
    signal.signal(signal.SIGHUP, stop_signal)    # SSH connection dropped
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
