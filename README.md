# vision_grasp_demo

ROS 2 Humble 视觉抓取最小 Demo 骨架（档 A 教学版）。

## 结构

```
vision_grasp_demo/
├── README.md                          ← 本文件
├── nodes/
│   ├── object_detector.py             ← 图像 → 目标检测（YOLOv8 或色块）
│   ├── depth_estimator.py             ← 目标 bbox + 深度图 → 3D 位姿
│   ├── grasp_planner.py               ← 3D 位姿 → MoveIt2 抓取轨迹
│   └── vision_grasp_pipeline.py       ← 三个节点串起来的顶层管理器
├── scripts/
│   ├── test_detector_synthetic.py     ← 合成图像测试检测器（无相机）
│   └── run_pipeline.sh                ← 一键启动脚本
└── requirements.txt                   ← Python 依赖（镜像里已装，仅作文档）
```

## 快速开始

```bash
# 用 vision 镜像启动交互容器
docker run --rm -it -v $(pwd):/ws -w /ws ros-moveit2:humble-vision

# 容器内跑合成图像测试（无需相机）
python3 nodes/test_detector_synthetic.py
```

## 数据流

```
[camera/rgb]  ─┐
               ├─→ [object_detector] ──→ /objects (bbox + label)
[camera/depth]─┘                              │
                                              ↓
                          [depth_estimator] ←── /depth
                                              │
                                              ↓
                                    [grasp_planner]
                                              │
                                              ↓
                                    /move_group_action (MoveIt2)
```

## 换真实相机

修改 `nodes/object_detector.py` 顶部的 `TOPIC_RGB`、`TOPIC_DEPTH`：
- RealSense D435i 默认话题：`/camera/rgb/image_raw` + `/camera/depth/image_raw`
- 其他相机按需改

## 换真实机械臂

修改 `nodes/grasp_planner.py` 顶部的 `GROUP_NAME`（如 `manipulator`）和 `FRAME_TARGET`（如 `world`）：
- Panda: `GROUP_NAME="panda_arm"`
- UR5:   `GROUP_NAME="manipulator"`, `FRAME_TARGET="base_link"`
- Hiwonder JetArm: `GROUP_NAME="arm"`, `FRAME_TARGET="base"`
