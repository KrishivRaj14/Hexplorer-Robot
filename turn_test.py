#!/usr/bin/env python3
"""turn_test.py - turn in place by a set angle, measured by the LiDAR (Fast-LIO),
then lie down and go into damping mode.

Needs start_fastlio.sh running in another terminal.

  python3 ~/turn_test.py 90       # turn 90 degrees LEFT  (counter-clockwise)
  python3 ~/turn_test.py -90      # turn 90 degrees RIGHT (clockwise)
  python3 ~/turn_test.py --check  # print the heading only, robot does NOT move

Based on the previous object_searcher.py: yaw from the odometry quaternion,
turning in WALK mode with angular.z (0.15 rad/s worked there), and the same
gentle shutdown: stop -> BALANCESTAND(3) -> STANDDOWN(1) -> PASSIVE/damping(0).

However it ends (target reached, Ctrl+C, a dropped SSH connection, turning the
wrong way, lost sensor data, an error), the robot stops and ends in damping.
There is NO obstacle check while turning: keep 1 m clear all around the robot
(it drifts 0.3-0.4 m while turning).

Two processes, like walk_safe.py: a helper (cyclonedds) reads the pose and
sends it over local UDP to this process (fastrtps), which drives the robot.
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


def yaw_of(q):
    """Quaternion -> yaw (radians), same formula as object_searcher.py."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------ sensor helper
def run_sensor(a):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from nav_msgs.msg import Odometry

    rclpy.init()
    node = Node("turn_test_sensor")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=5)
    st = dict(x=None, y=None, yaw=None, pose_t=0.0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def on_pose(m):
        st["x"] = m.pose.pose.position.x
        st["y"] = m.pose.pose.position.y
        st["yaw"] = yaw_of(m.pose.pose.orientation)
        st["pose_t"] = time.time()

    node.create_subscription(Odometry, "/lidar_odometry/pose", on_pose, qos)
    node.create_timer(0.05, lambda: sock.sendto(json.dumps(st).encode(),
                                                ("127.0.0.1", a.port)))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------------ helpers
class Link:
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


def fresh(m, max_age=0.6):
    return bool(m) and m["yaw"] is not None and time.time() - m["pose_t"] < max_age


def say(*args):
    try:
        print(*args, flush=True)
    except Exception:  # noqa
        pass


def stop_signal(signum, frame):
    raise KeyboardInterrupt


class Tracker:
    """Adds up how far the heading has turned (handles the +-180 wrap)."""

    def __init__(self, yaw):
        self.prev = yaw
        self.total = 0.0

    def update(self, yaw):
        self.total += wrap(yaw - self.prev)
        self.prev = yaw
        return self.total


# ------------------------------------------------------------------- check
def run_check(a, link):
    print("CHECK MODE: the robot will not move. Ctrl+C to quit.")
    print("Turn the robot by hand slowly: LEFT (counter-clockwise) should make")
    print("'turned' go UP, right should make it go DOWN.\n")
    tr = None
    while True:
        time.sleep(0.5)
        m = link.get()
        if not fresh(m):
            print("waiting for pose (is start_fastlio.sh running?)")
            continue
        if tr is None:
            tr = Tracker(m["yaw"])
        t = tr.update(m["yaw"])
        print(f"heading {math.degrees(m['yaw']):+7.1f} deg | turned {math.degrees(t):+7.1f} deg"
              f" | x={m['x']:+.2f} y={m['y']:+.2f}")


# -------------------------------------------------------------------- turn
def turn_loop(a, link, send):
    """Turn until done. Returns (reason, tracker or None). Raises on Ctrl+C."""
    clock, sleep = time.time, time.sleep
    target = math.radians(a.angle)
    sign = 1.0 if target > 0 else -1.0
    m = link.get()
    if not fresh(m):
        raise RuntimeError("pose data lost while standing up")
    tr = Tracker(m["yaw"])
    a.start = (m["x"], m["y"])
    timeout = abs(target) / a.rate * 2.0 + 10.0
    t0 = last_print = clock()
    while True:
        m = link.get()
        if not fresh(m):
            return "STOPPED: pose data lost", tr
        turned = tr.update(m["yaw"])
        remaining = abs(target) - sign * turned
        if sign * turned < -math.radians(15):
            return ("STOPPED: it turned the WRONG way (the sign of angular.z is "
                    "opposite to what I assumed)"), tr
        if remaining <= math.radians(a.lead):
            return "reached the target angle", tr
        if clock() - t0 > timeout:
            return "STOPPED: timeout (not turning fast enough?)", tr
        rate = a.rate if remaining > math.radians(15) else min(a.rate, 0.08)
        send(4, 0.0, sign * rate)
        if clock() - last_print > 1.0:
            last_print = clock()
            say(f"  turned {math.degrees(turned):+6.1f} of {a.angle:+.0f} deg")
        sleep(0.05)


def run_turn(a, link):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from custom_msg.msg import RobotCommand

    print("Waiting for pose...")
    t0 = time.time()
    while time.time() - t0 < 10:
        time.sleep(0.2)
        if fresh(link.get()):
            break
    else:
        print("ERROR: no fresh pose. Is start_fastlio.sh running and printing "
              "'Relayed ... odom messages'? Try --check.")
        return 1

    rclpy.init()
    node = Node("turn_test")
    cmd_pub = node.create_publisher(RobotCommand, "/robot_cmd", 10)
    vel_pub = node.create_publisher(Twist, "/vel_cmd", 10)

    def send(state, vx=0.0, wz=0.0):
        c = RobotCommand()
        c.header.stamp = node.get_clock().now().to_msg()
        c.target_state = state
        v = Twist()
        v.linear.x = float(vx)
        v.angular.z = float(wz)
        cmd_pub.publish(c)
        vel_pub.publish(v)

    def hold(state, seconds, vx=0.0, wz=0.0):
        for _ in range(int(seconds * 20)):
            send(state, vx, wz)
            time.sleep(0.05)

    return run_with_safe_end(a, link, send, hold, node, rclpy)


def run_with_safe_end(a, link, send, hold, node=None, rclpy=None):
    reason, tr, standing = "unknown", None, False
    try:
        print(f"Standing up (about 6 s)... then turning {a.angle:+.0f} deg "
              f"({'left' if a.angle > 0 else 'right'}) at {a.rate:.2f} rad/s")
        standing = True
        for s in (1, 2, 3):
            hold(s, 2.0)
        reason, tr = turn_loop(a, link, send)
    except KeyboardInterrupt:
        reason = "STOPPED: Ctrl+C"
    except Exception as e:  # noqa
        reason = f"STOPPED: error {e}"
    finally:
        for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sg, signal.SIG_IGN)
        try:
            try:
                hold(4, 1.0)                          # stop turning, let it settle
                m = link.get()
                say(f"\nResult: {reason}")
                if tr is not None and fresh(m, 2.0):
                    total = tr.update(m["yaw"])
                    dx, dy = m["x"] - a.start[0], m["y"] - a.start[1]
                    say(f"LiDAR says it turned {math.degrees(total):+.1f} deg "
                        f"({a.angle:+.0f} asked). LiDAR moved {math.hypot(dx, dy):.2f} m "
                        f"while turning (it sits 0.3 m in front of the body centre, "
                        f"so some movement is normal)")
            finally:
                if standing:
                    say("Lying down, then damping mode...")
                    hold(3, 1.0)                      # same order as object_searcher.py
                    hold(1, 2.0)
                    hold(0, 2.0)
        finally:
            if node is not None:
                node.destroy_node()
                rclpy.shutdown()
    return 0 if reason.startswith("reached") else 2


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("angle", nargs="?", type=float, default=None,
                    help="degrees to turn: positive = left, negative = right")
    ap.add_argument("--check", action="store_true", help="heading only, no movement")
    ap.add_argument("--sensor", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rate", type=float, default=0.15, help="turn rate in rad/s")
    ap.add_argument("--lead", type=float, default=3.0,
                    help="stop this many degrees early to allow for coasting")
    ap.add_argument("--port", type=int, default=45679)
    ap.add_argument("--sensor-rmw", default="rmw_cyclonedds_cpp")
    a = ap.parse_args()

    if a.sensor:
        run_sensor(a)
        return 0
    if not a.check:
        if a.angle is None:
            ap.error("give an angle in degrees (e.g. 90 or -90), or use --check")
        if not (10 <= abs(a.angle) <= 360):
            ap.error("angle must be 10 to 360 degrees (or -10 to -360)")
        a.rate = max(0.05, min(a.rate, 0.30))

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGHUP, stop_signal)
    link = Link(a.port)
    env = dict(os.environ, RMW_IMPLEMENTATION=a.sensor_rmw)
    helper = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--sensor",
                               "--port", str(a.port)], env=env)
    try:
        return run_check(a, link) if a.check else run_turn(a, link)
    except KeyboardInterrupt:
        return 0
    finally:
        helper.terminate()


if __name__ == "__main__":
    sys.exit(main())
