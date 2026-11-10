#!/bin/bash
echo "== Jetson (password 123 if asked) =="
ssh -o ConnectTimeout=5 robot@192.168.1.20 'bash -s' <<'JEOF'
echo "-- livox processes BEFORE:"; ps -eo pid,etime,cmd | grep -E "livox" | grep -v grep
echo "-- who holds the 56xxx UDP ports BEFORE:"; ss -lunp | grep -E ":56"
echo "-- killing LiDAR driver + bridge"
pkill -f livox_lidar_node; pkill -f livox_tcp_bridge.py; pkill -f "ros2 launch livox"
sleep 3
pkill -9 -f livox_lidar_node; sleep 1
echo "-- left after kill (should be empty):"; ps -eo pid,cmd | grep -E "livox" | grep -v grep
echo "-- restarting once"
bash /home/robot/jetson_services.sh
sleep 6
echo "-- fresh driver log:"; tail -15 /tmp/livox_driver.log
echo "-- UDP ports AFTER:"; ss -lunp | grep -E ":56"
JEOF
