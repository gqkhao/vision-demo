"""
nodes/object_detector.py

订阅 RGB 图像话题，运行目标检测，发布 bbox + label + 置信度。

两种检测器可切换：
  1. 色块检测器（默认）：HSV 阈值，零依赖，跑通流程用
  2. YOLOv8 检测器：真正语义检测，需要 weights（模型会自动下载）

数据流：
  /camera/rgb/image_raw  ─→  [detect]  ─→  /objects (自定义 msg)

话题输出：
  /objects:  std_msgs/String (JSON 序列化，教学简化)
    {
      "objects": [
        {"label": "red_block", "bbox": [x1, y1, x2, y2], "conf": 0.87},
        ...
      ]
    }
"""

import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String

import cv2
import numpy as np


# ============ 配置 ============
TOPIC_RGB = "/camera/rgb/image_raw"    # RealSense D435i 默认话题
TOPIC_OUT = "/objects"                 # 检测结果输出
DETECTOR = "color_block"               # "color_block" 或 "yolo"
YOLO_WEIGHTS = "yolov8n.pt"            # Ultralytics 会自动下载 ~6MB


class ObjectDetector(Node):
    def __init__(self):
        super().__init__("object_detector")
        self.bridge = CvBridge()
        self.img_sub = self.create_subscription(
            Image, TOPIC_RGB, self.image_cb, 10)
        self.out_pub = self.create_publisher(String, TOPIC_OUT, 10)
        self.detector = self._make_detector()
        self.get_logger().info(f"ObjectDetector ready, mode={DETECTOR}")

    def _make_detector(self):
        """按配置选择检测器实现。"""
        if DETECTOR == "yolo":
            from ultralytics import YOLO
            model = YOLO(YOLO_WEIGHTS)
            return {"kind": "yolo", "model": model}
        else:
            # HSV 阈值：红色（可检测 Hiwonder 红色积木/方块）
            lower_red = np.array([0, 120, 120])
            upper_red = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 120])
            upper_red2 = np.array([180, 255, 255])
            return {"kind": "color",
                    "low1": lower_red, "up1": upper_red,
                    "low2": lower_red2, "up2": upper_red2}

    def image_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"imgmsg_to_cv2 failed: {e}")
            return

        # 缩小加速检测
        small = cv2.resize(cv_img, (640, 480))
        objects = self._detect(small)

        payload = {"objects": objects}
        self.out_pub.publish(String(data=json.dumps(payload)))

        if objects:
            self.get_logger().info(f"detected {len(objects)} object(s)")

    def _detect(self, img):
        """返回 [{"label", "bbox": [x1,y1,x2,y2], "conf"}, ...]"""
        d = self.detector
        if d["kind"] == "yolo":
            return self._detect_yolo(img, d["model"])
        return self._detect_color(img, d)

    def _detect_yolo(self, img, model):
        """YOLOv8 语义检测，label 是模型自带类别名。"""
        results = model(img, verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                out.append({"label": label,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "conf": round(conf, 3)})
        return out

    def _detect_color(self, img, cfg):
        """HSV 色块检测，只找红色物体（教学用）。"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, cfg["low1"], cfg["up1"])
        m2 = cv2.inRange(hsv, cfg["low2"], cfg["up2"])
        mask = cv2.bitwise_or(m1, m2)
        # 形态学去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 300:       # 过滤噪点
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append({"label": "red_block",
                        "bbox": [x, y, x + w, y + h],
                        "conf": min(1.0, area / 2000.0)})
        return out


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
