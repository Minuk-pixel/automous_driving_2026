import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import Detection
from interfaces_pkg.msg import DetectionArray


IMAGE_TOPIC = "image_raw"
DETECTION_TOPIC = "object_detections"
WINDOW_NAME = "object_detection_visualizer"
FRAME_SKIP = 1
SYNC_QUEUE_SIZE = 5
SYNC_SLOP_SEC = 0.3
LOG_BOTTOM_Y = True
LOG_PERIOD_SEC = 0.5


class ObjectDetectionVisualizerNode(Node):
    def __init__(self):
        super().__init__("object_detection_visualizer_node")

        self.image_topic = self.declare_parameter("image_topic", IMAGE_TOPIC).value
        self.detection_topic = self.declare_parameter("detection_topic", DETECTION_TOPIC).value
        self.window_name = self.declare_parameter("window_name", WINDOW_NAME).value
        self.frame_skip = self.declare_parameter("frame_skip", FRAME_SKIP).value
        self.sync_queue_size = self.declare_parameter("sync_queue_size", SYNC_QUEUE_SIZE).value
        self.sync_slop_sec = self.declare_parameter("sync_slop_sec", SYNC_SLOP_SEC).value
        self.log_bottom_y = self.declare_parameter("log_bottom_y", LOG_BOTTOM_Y).value
        self.log_period_sec = self.declare_parameter("log_period_sec", LOG_PERIOD_SEC).value
        self.image_reliability = self.declare_parameter(
            "image_reliability",
            QoSReliabilityPolicy.BEST_EFFORT,
        ).value

        self.cv_bridge = CvBridge()
        self.frame_count = 0
        self.latest_log_time = None
        self._class_to_color = {}

        image_qos_profile = QoSProfile(
            reliability=self.image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.image_sub = message_filters.Subscriber(
            self,
            Image,
            self.image_topic,
            qos_profile=image_qos_profile,
        )
        self.detections_sub = message_filters.Subscriber(
            self,
            DetectionArray,
            self.detection_topic,
            qos_profile=10,
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.detections_sub),
            int(self.sync_queue_size),
            float(self.sync_slop_sec),
        )
        self.synchronizer.registerCallback(self.detections_cb)

        self.get_logger().info(
            f"Object detection visualizer: {self.image_topic} + {self.detection_topic}"
        )

    def detections_cb(self, img_msg: Image, detection_msg: DetectionArray):
        self.frame_count += 1
        frame_skip = max(1, int(self.frame_skip))
        if self.frame_count % frame_skip != 0:
            return

        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
        if cv_image.ndim == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        bottom_y_logs = []
        for detection in detection_msg.detections:
            bottom_y = self._draw_detection(cv_image, detection)
            bottom_y_logs.append(
                f"{detection.class_name}:{bottom_y:.1f}px"
            )

        if self.log_bottom_y and bottom_y_logs and self._should_log():
            image_h = cv_image.shape[0]
            self.get_logger().info(
                f"bbox bottom_y h={image_h}px -> " + ", ".join(bottom_y_logs)
            )

        cv2.imshow(self.window_name, cv_image)
        cv2.waitKey(1)

    def _draw_detection(self, cv_image: np.ndarray, detection: Detection):
        color = self._color_for_class(detection.class_name)
        bbox = detection.bbox

        x_center = float(bbox.center.position.x)
        y_center = float(bbox.center.position.y)
        width = float(bbox.size.x)
        height = float(bbox.size.y)

        x_min = int(round(x_center - width * 0.5))
        y_min = int(round(y_center - height * 0.5))
        x_max = int(round(x_center + width * 0.5))
        y_max = int(round(y_center + height * 0.5))

        image_h, image_w = cv_image.shape[:2]
        x_min = int(np.clip(x_min, 0, image_w - 1))
        x_max = int(np.clip(x_max, 0, image_w - 1))
        y_min = int(np.clip(y_min, 0, image_h - 1))
        y_max = int(np.clip(y_max, 0, image_h - 1))

        cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), color, 2)

        label = f"{detection.class_name} {detection.score:.2f}"
        text_origin = (x_min, max(18, y_min - 6))
        cv2.putText(
            cv_image,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        return float(y_center + height * 0.5)

    def _color_for_class(self, class_name: str):
        if class_name not in self._class_to_color:
            seed = sum(ord(ch) for ch in class_name)
            self._class_to_color[class_name] = (
                40 + (seed * 37) % 216,
                40 + (seed * 67) % 216,
                40 + (seed * 97) % 216,
            )
        return self._class_to_color[class_name]

    def _should_log(self):
        now = self.get_clock().now()
        if self.latest_log_time is None:
            self.latest_log_time = now
            return True

        elapsed = now - self.latest_log_time
        if elapsed.nanoseconds * 1e-9 < float(self.log_period_sec):
            return False

        self.latest_log_time = now
        return True


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
