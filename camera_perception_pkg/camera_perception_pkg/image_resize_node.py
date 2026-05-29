import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image


SUB_TOPIC_NAME = "image_raw"
PUB_TOPIC_NAME = "image_raw_resized"
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
SHOW_IMAGE = False


class ImageResizeNode(Node):
    def __init__(self):
        super().__init__("image_resize_node")

        self.sub_topic = self.declare_parameter("sub_topic", SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter("pub_topic", PUB_TOPIC_NAME).value
        self.image_width = self.declare_parameter("image_width", IMAGE_WIDTH).value
        self.image_height = self.declare_parameter("image_height", IMAGE_HEIGHT).value
        self.show_image = self.declare_parameter("show_image", SHOW_IMAGE).value
        self.image_reliability = self.declare_parameter(
            "image_reliability",
            QoSReliabilityPolicy.BEST_EFFORT,
        ).value

        self.cv_bridge = CvBridge()
        self.qos_profile = QoSProfile(
            reliability=self.image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.subscriber = self.create_subscription(
            Image,
            self.sub_topic,
            self.image_callback,
            self.qos_profile,
        )
        self.publisher = self.create_publisher(Image, self.pub_topic, self.qos_profile)

        self.get_logger().info(
            f"Resizing {self.sub_topic} -> {self.pub_topic} "
            f"({int(self.image_width)}x{int(self.image_height)})"
        )

    def image_callback(self, msg: Image):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
        resized = cv2.resize(
            cv_image,
            (int(self.image_width), int(self.image_height)),
            interpolation=cv2.INTER_AREA,
        )

        resized_msg = self.cv_bridge.cv2_to_imgmsg(resized, encoding=msg.encoding)
        resized_msg.header = msg.header
        self.publisher.publish(resized_msg)

        if self.show_image:
            cv2.imshow("image_raw_resized", resized)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ImageResizeNode()
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
