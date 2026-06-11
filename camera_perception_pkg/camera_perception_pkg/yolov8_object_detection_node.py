from typing import Dict, List

import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge
from torch import cuda
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Results

from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from interfaces_pkg.msg import BoundingBox2D
from interfaces_pkg.msg import Detection
from interfaces_pkg.msg import DetectionArray


MODEL_PATH = "/home/minuk/ros2_ws/src/camera_perception_pkg/best_object_0611.pt"
DEVICE = "cuda:0"
THRESHOLD = 0.5
IMGSZ = 320
FRAME_SKIP = 1
HALF = True
IMAGE_TOPIC = "image_raw"
# IMAGE_TOPIC = "image_raw_resized"
PUB_TOPIC = "object_detections"


class Yolov8ObjectDetectionNode(LifecycleNode):
    """YOLOv8 bbox-only detector for obstacle missions."""

    def __init__(self, **kwargs) -> None:
        super().__init__("yolov8_object_detection_node", **kwargs)

        self.declare_parameter("model", MODEL_PATH)
        self.declare_parameter("device", DEVICE)
        self.declare_parameter("threshold", THRESHOLD)
        self.declare_parameter("imgsz", IMGSZ)
        self.declare_parameter("frame_skip", FRAME_SKIP)
        self.declare_parameter("half", HALF)
        self.declare_parameter("enable", True)
        self.declare_parameter("image_topic", IMAGE_TOPIC)
        self.declare_parameter("pub_topic", PUB_TOPIC)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        self.frame_count = 0
        self.get_logger().info("Yolov8ObjectDetectionNode created")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Configuring {self.get_name()}")

        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.imgsz = self.get_parameter("imgsz").get_parameter_value().integer_value
        self.frame_skip = self.get_parameter("frame_skip").get_parameter_value().integer_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.pub_topic = self.get_parameter("pub_topic").get_parameter_value().string_value
        self.reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value

        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub = self.create_lifecycle_publisher(DetectionArray, self.pub_topic, 10)
        self._srv = self.create_service(SetBool, "enable_object_detection", self.enable_cb)
        self.cv_bridge = CvBridge()

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Activating {self.get_name()}")

        try:
            self.yolo = YOLO(self.model)
            self.yolo.fuse()
        except FileNotFoundError:
            self.get_logger().error(f"Model file not found: {self.model}")
            return TransitionCallbackReturn.FAILURE
        except Exception as exc:
            self.get_logger().error(f"Error while loading model '{self.model}': {exc}")
            return TransitionCallbackReturn.FAILURE

        self._sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            self.image_qos_profile,
        )

        super().on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Deactivating {self.get_name()}")

        if hasattr(self, "yolo"):
            del self.yolo

        if "cuda" in self.device:
            cuda.empty_cache()

        if hasattr(self, "_sub") and self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None

        super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"Cleaning up {self.get_name()}")

        self.destroy_publisher(self._pub)
        del self.image_qos_profile

        return TransitionCallbackReturn.SUCCESS

    def enable_cb(self, request, response):
        self.enable = request.data
        response.success = True
        return response

    def image_cb(self, msg: Image) -> None:
        self.frame_count += 1
        frame_skip = max(1, int(self.frame_skip))
        if self.frame_count % frame_skip != 0:
            return

        if not self.enable:
            return

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        results = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            imgsz=self.imgsz,
            half=(self.half and "cuda" in self.device),
            device=self.device,
        )
        result: Results = results[0].cpu()

        detections_msg = DetectionArray()
        detections_msg.header = msg.header

        if result.boxes:
            hypothesis = self._parse_hypothesis(result)
            boxes = self._parse_boxes(result)

            for i in range(len(result.boxes)):
                detection = Detection()
                detection.class_id = hypothesis[i]["class_id"]
                detection.class_name = hypothesis[i]["class_name"]
                detection.score = hypothesis[i]["score"]
                detection.bbox = boxes[i]
                detections_msg.detections.append(detection)

        self._pub.publish(detections_msg)

        del result
        del results
        del cv_image

    def _parse_hypothesis(self, result: Results) -> List[Dict]:
        hypothesis_list = []

        box_data: Boxes
        for box_data in result.boxes:
            class_id = int(box_data.cls)
            hypothesis_list.append({
                "class_id": class_id,
                "class_name": self.yolo.names[class_id],
                "score": float(box_data.conf),
            })

        return hypothesis_list

    def _parse_boxes(self, result: Results) -> List[BoundingBox2D]:
        boxes_list = []

        box_data: Boxes
        for box_data in result.boxes:
            box = box_data.xywh[0]

            msg = BoundingBox2D()
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])
            boxes_list.append(msg)

        return boxes_list


def main():
    rclpy.init()
    node = Yolov8ObjectDetectionNode()
    node.trigger_configure()
    node.trigger_activate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
