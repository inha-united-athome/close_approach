#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ament_index_python.packages import get_package_share_directory

from inha_interfaces.srv import DetectObject
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    Pose2D,
)

from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2


class Detector(Node):
    def __init__(self, debug_enabled=False):
        super().__init__("approach_detector")
        self.bridge = CvBridge()
        self.sub_ = None
        self.targets = []
        self.debug_enabled = debug_enabled
        self.debug_output_dir = None
        self.debug_frame_index = 0

        self.qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.declare_parameter("topic_name", "/camera/camera_head/color/image_raw/compressed")
        self.declare_parameter("service_name", "/detect_object")
        self.declare_parameter("model_path", "yolov8s-world.pt")
        self.declare_parameter("confidence_threshold", 0.1)

        self.topic_name = self.get_parameter("topic_name").value
        self.service_name = self.get_parameter("service_name").value
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)

        model_path = self.get_parameter("model_path").value
        full_path = f"{get_package_share_directory('approach')}/{model_path}"
        if not model_path.startswith("/") and os.path.exists(full_path):
            model_path = full_path

        self.model = YOLO(model_path)

        if self.debug_enabled:
            self.debug_output_dir = Path.cwd() / "debug" / "detector"
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(
                f"Debug detection artifacts will be saved to {self.debug_output_dir}"
            )

        self.ser_ = self.create_service(
            DetectObject, self.service_name, self.detect_object_callback
        )
        self.array_pub = self.create_publisher(
            Detection2DArray,
            "/approach/detector/detection_array",
            self.qos_reliable,
        )
        self.detections_pub = self.create_publisher(
            CompressedImage,
            "/approach/detector/detections/compressed",
            self.qos_best_effort,
        )

        self.get_logger().info("Detector node started.")

    def _create_subscriber(self):
        if self.sub_ is None:
            self.sub_ = self.create_subscription(
                CompressedImage,
                self.topic_name,
                self.image_callback,
                self.qos_best_effort,
            )

    def _destroy_subscriber(self):
        if self.sub_ is not None:
            self.destroy_subscription(self.sub_)
            self.sub_ = None

    def detect_object_callback(self, request, response):
        self.get_logger().info("Received request to detect objects.")

        if request.start:
            self.targets = [t.strip().lower() for t in request.target]
            try:
                self.model.set_classes(self.targets)
                self.get_logger().info(f"YOLO-World classes set to: {self.targets}")
            except Exception as e:
                self.get_logger().warn(f"Model does not support 'set_classes'. Assuming standard YOLO model. (e: {e})")
                
            self._create_subscriber()
            response.success = True
            self.get_logger().info(f"Started object detection for targets: {self.targets}")
            return response

        self._destroy_subscriber()
        self.targets = []
        response.success = True
        self.get_logger().info("Stopped object detection.")
        return response

    def image_callback(self, msg):
        if not self.targets:
            return

        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model(cv_image, verbose=False, conf=self.confidence_threshold)
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        annotated = cv_image.copy()

        for result in results:
            boxes = result.boxes
            names = result.names

            for box in boxes:
                class_id = int(box.cls.item())
                class_name = names[class_id].lower()
                confidence = float(box.conf.item())

                if class_name not in self.targets:
                    continue

                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                detection = self._build_detection(msg.header, class_name, confidence, x1, y1, x2, y2)
                detection_array.detections.append(detection)
                self._draw_detection(annotated, class_name, confidence, x1, y1, x2, y2)

        detection_image_msg = self.bridge.cv2_to_compressed_imgmsg(annotated)
        detection_image_msg.header = msg.header

        self._save_debug_artifacts(annotated, detection_array)
        self.array_pub.publish(detection_array)
        self.detections_pub.publish(detection_image_msg)

    def _build_detection(self, header, class_name, confidence, x1, y1, x2, y2):
        detection = Detection2D()
        detection.header = header
        detection.id = class_name

        hypothesis = ObjectHypothesis()
        hypothesis.class_id = class_name
        hypothesis.score = confidence

        result = ObjectHypothesisWithPose()
        result.hypothesis = hypothesis
        detection.results.append(result)

        bbox = BoundingBox2D()
        center = Pose2D()
        center.position.x = (x1 + x2) / 2.0
        center.position.y = (y1 + y2) / 2.0
        center.theta = 0.0

        bbox.center = center
        bbox.size_x = x2 - x1
        bbox.size_y = y2 - y1
        detection.bbox = bbox
        return detection

    def _draw_detection(self, image, class_name, confidence, x1, y1, x2, y2):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (p1[0], max(20, p1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    def _save_debug_artifacts(self, annotated, detection_array):
        if not self.debug_enabled or self.debug_output_dir is None:
            return

        stem = f"{self.debug_frame_index:06d}"
        image_path = self.debug_output_dir / f"{stem}_detections.jpg"
        meta_path = self.debug_output_dir / f"{stem}_detections.txt"

        cv2.imwrite(str(image_path), annotated)
        with meta_path.open("w", encoding="utf-8") as handle:
            for detection in detection_array.detections:
                if not detection.results:
                    continue
                hypothesis = detection.results[0].hypothesis
                bbox = detection.bbox
                handle.write(
                    f"{hypothesis.class_id} {hypothesis.score:.4f} "
                    f"{bbox.center.position.x:.1f} {bbox.center.position.y:.1f} "
                    f"{bbox.size_x:.1f} {bbox.size_y:.1f}\n"
                )

        self.debug_frame_index += 1


def _parse_debug_arg(args):
    debug_enabled = False
    filtered_args = []
    for arg in args:
        if arg in ("debug", "--debug"):
            debug_enabled = True
            continue
        filtered_args.append(arg)
    return debug_enabled, filtered_args


def main(args=None):
    cli_args = list(sys.argv[1:] if args is None else args)
    debug_enabled, filtered_args = _parse_debug_arg(cli_args)
    rclpy.init(args=filtered_args)
    detector = Detector(debug_enabled=debug_enabled)
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
