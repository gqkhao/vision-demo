#!/usr/bin/env bash
# scripts/run_pipeline.sh
#
# 一键启动视觉抓取 pipeline（教学版）。
# 用法：
#   bash scripts/run_pipeline.sh
#
# 依赖：容器内已装 ros-moveit2:humble-vision 的所有 ROS 包。
# 生产环境请改用 launch.py / launch.py 而非 shell 脚本。
#
# 三个终端的并行启动方式（shell 版，方便学习）：
#   Terminal 1: ros2 launch realsense2_camera rs435i.launch.py     # 相机
#   Terminal 2: python3 nodes/object_detector.py
#   Terminal 3: python3 nodes/depth_estimator.py
#   Terminal 4: python3 nodes/grasp_planner.py
#
# 无相机时改用合成图像：
#   python3 scripts/test_detector_synthetic.py

set -e

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

# 确保 ROS 2 环境已加载
source /opt/ros/humble/setup.bash

echo "=== 启动 4 个节点 ==="
echo "[1] object_detector  →  python3 nodes/object_detector.py"
python3 nodes/object_detector.py &
PID1=$!

sleep 2
echo "[2] depth_estimator  →  python3 nodes/depth_estimator.py"
python3 nodes/depth_estimator.py &
PID2=$!

sleep 2
echo "[3] grasp_planner    →  python3 nodes/grasp_planner.py"
python3 nodes/grasp_planner.py &
PID3=$!

sleep 2
echo "[4] synthetic_pub    →  python3 scripts/test_detector_synthetic.py"
python3 scripts/test_detector_synthetic.py &
PID4=$!

echo ""
echo "所有节点已启动："
echo "  detector   PID=$PID1"
echo "  depth      PID=$PID2"
echo "  planner    PID=$PID3"
echo "  synthetic  PID=$PID4"
echo ""
echo "按 Ctrl+C 停止所有节点。"
echo "查看话题：ros2 topic list"
echo "查看输出：ros2 topic echo /objects"
echo "          ros2 topic echo /object_3d"

trap 'kill $PID1 $PID2 $PID3 $PID4 2>/dev/null; wait' INT TERM
wait
