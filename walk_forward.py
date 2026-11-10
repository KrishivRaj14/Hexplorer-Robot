#!/usr/bin/env python3
"""Walk the Hexplorer robot forward a specified distance"""
import rclpy
from rclpy.node import Node
from custom_msg.msg import RobotCommand
from geometry_msgs.msg import Twist
import time
import sys

def walk(distance_m=0.5, speed=0.15):
    rclpy.init()
    node = Node('robot_walker')
    cmd_pub = node.create_publisher(RobotCommand, '/robot_cmd', 10)
    vel_pub = node.create_publisher(Twist, '/vel_cmd', 10)
    time.sleep(0.3)
    
    cmd = RobotCommand()
    vel = Twist()
    
    # STANDDOWN (1) -> legs fold into position
    cmd.target_state = 1
    for _ in range(40):
        cmd_pub.publish(cmd)
        time.sleep(0.05)
    
    # STANDUP (2) -> legs extend
    cmd.target_state = 2
    for _ in range(40):
        cmd_pub.publish(cmd)
        time.sleep(0.05)
    
    # BALANCE (3) -> force control standing
    cmd.target_state = 3
    for _ in range(40):
        cmd_pub.publish(cmd)
        time.sleep(0.05)
    
    # WALK (4) + velocity
    walk_time = abs(distance_m) / speed
    cmd.target_state = 4
    vel.linear.x = speed if distance_m > 0 else -speed
    
    iterations = int(walk_time / 0.05)
    for _ in range(iterations):
        cmd_pub.publish(cmd)
        vel_pub.publish(vel)
        time.sleep(0.05)
    
    # PASSIVE (0) -> damping
    cmd.target_state = 0
    for _ in range(20):
        cmd_pub.publish(cmd)
        time.sleep(0.05)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    dist = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    walk(dist)
