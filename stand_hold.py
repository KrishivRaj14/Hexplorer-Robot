#!/usr/bin/env python3
"""stand_hold.py - stand up and hold still, so sensors can be checked at
walking height. The robot does NOT walk. Ctrl+C to finish: it lies down
(BALANCESTAND 3 -> STANDDOWN 1) and goes into damping (0), same order as
the previous owner's obstacle_avoidance.py sit_down().

  python3 ~/stand_hold.py          # hold until Ctrl+C (max 10 minutes)
"""
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from custom_msg.msg import RobotCommand
from geometry_msgs.msg import Twist


def stop_signal(signum, frame):
    raise KeyboardInterrupt


def ros_init(rclpy):
    """Start ROS WITHOUT its own Ctrl+C handler. ROS's handler shuts the
    connection down the moment Ctrl+C is pressed, so the stop / lie-down /
    damping commands that follow could not be sent ("publisher's context is
    invalid"). With it off, Ctrl+C reaches our code first and the robot is
    always stopped and put in damping before ROS is shut down."""
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError, AttributeError):
        rclpy.init()
    signal.signal(signal.SIGINT, signal.default_int_handler)   # Ctrl+C -> KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGHUP, stop_signal)


def main():
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGHUP, stop_signal)
    ros_init(rclpy)
    node = Node("stand_hold")
    cmd_pub = node.create_publisher(RobotCommand, "/robot_cmd", 10)
    vel_pub = node.create_publisher(Twist, "/vel_cmd", 10)

    def hold(state, seconds):
        for _ in range(int(seconds * 20)):
            c = RobotCommand()
            c.header.stamp = node.get_clock().now().to_msg()
            c.target_state = state
            cmd_pub.publish(c)
            vel_pub.publish(Twist())          # zero velocity
            time.sleep(0.05)

    try:
        print("Standing up (about 6 s)...", flush=True)
        for s in (1, 2, 3):
            hold(s, 2.0)
        print("Standing still. Do your checks. Ctrl+C to lie down.", flush=True)
        hold(3, 600)                          # BALANCESTAND, max 10 minutes
        print("10 minutes up.", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sg, signal.SIG_IGN)
        try:
            print("\nLying down, then damping mode...", flush=True)
        except Exception:  # noqa
            pass
        hold(3, 1.0)
        hold(1, 2.0)
        hold(0, 2.0)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
