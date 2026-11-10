#!/bin/bash
echo "== Jetson (password 123 if asked) =="
ssh -o ConnectTimeout=5 robot@192.168.1.20 'bash -s' <<'JEOF'
CFG=/home/robot/robot_controller_release/ros2_packages/livox_lidar_node/share/livox_lidar_node/config/lidar_parameters.json
echo "-- Jetson IPs:"; hostname -I
echo "-- ping LiDAR (.153 is ours, .190 is factory default):"
for ip in 192.168.1.153 192.168.1.190; do
  ping -c2 -W1 $ip >/dev/null 2>&1 && echo "$ip REPLIES" || echo "$ip no reply"
done
echo "-- config:"; grep -E 'host_ip|lidar_ip' $CFG
echo "-- systemd service:"; systemctl is-active jetson-lidar.service
echo "-- process counts (each should be 1):"
echo "driver: $(pgrep -fc livox_lidar_node)   bridge: $(pgrep -fc livox_tcp_bridge.py)"
echo "-- driver log (last 15):"; tail -15 /tmp/livox_driver.log
echo "-- bridge log (last 5):"; tail -5 /tmp/livox_bridge.log
echo "-- UDP ports open:"; ss -lun | grep -E "5610|5620|5630|5640"
JEOF
echo "== Mini PC: LiDAR receiver side =="
ls ~/hexplorer/bridges/ 2>/dev/null
