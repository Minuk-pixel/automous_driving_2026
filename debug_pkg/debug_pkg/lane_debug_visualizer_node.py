import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from interfaces_pkg.msg import DetectionArray


SUB_TOPIC_NAME = "detections"
DRIVABLE_CLASS_NAME = "lane"

IPM_SRC_POINTS = [238.0, 316.0, 402.0, 313.0, 501.0, 476.0, 155.0, 476.0]
IPM_SRC_BASE_WIDTH = 640.0
IPM_SRC_BASE_HEIGHT = 480.0
IPM_DST_LEFT_RATIO = 0.3
IPM_DST_RIGHT_RATIO = 0.7
BEV_HEIGHT_SCALE = 4.0

ROW_STRIDE = 4
MIN_ROAD_WIDTH_PX = 20
DEBUG_FRAME_SKIP = 3


class LaneDebugVisualizerNode(Node):
    def __init__(self):
        super().__init__('lane_debug_visualizer_node')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.drivable_class_name = self.declare_parameter('drivable_class_name', DRIVABLE_CLASS_NAME).value
        self.ipm_src_points = self.declare_parameter('ipm_src_points', IPM_SRC_POINTS).value
        self.ipm_src_base_width = self.declare_parameter('ipm_src_base_width', IPM_SRC_BASE_WIDTH).value
        self.ipm_src_base_height = self.declare_parameter('ipm_src_base_height', IPM_SRC_BASE_HEIGHT).value
        self.ipm_dst_left_ratio = self.declare_parameter('ipm_dst_left_ratio', IPM_DST_LEFT_RATIO).value
        self.ipm_dst_right_ratio = self.declare_parameter('ipm_dst_right_ratio', IPM_DST_RIGHT_RATIO).value
        self.bev_height_scale = self.declare_parameter('bev_height_scale', BEV_HEIGHT_SCALE).value
        self.row_stride = self.declare_parameter('row_stride', ROW_STRIDE).value
        self.min_road_width_px = self.declare_parameter('min_road_width_px', MIN_ROAD_WIDTH_PX).value
        self.debug_frame_skip = self.declare_parameter('debug_frame_skip', DEBUG_FRAME_SKIP).value

        self.frame_count = 0

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.subscriber = self.create_subscription(
            DetectionArray,
            self.sub_topic,
            self.detections_callback,
            qos_profile
        )

    def detections_callback(self, detection_msg: DetectionArray):
        self.frame_count += 1
        frame_skip = max(1, int(self.debug_frame_skip))
        if self.frame_count % frame_skip != 0:
            return

        binary_mask = self._draw_drivable_area_mask(detection_msg)
        if binary_mask is None:
            return

        bev_mask = self._bird_convert(binary_mask)
        midpoints_px = self._scan_midpoints(bev_mask)

        debug_bev = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
        if midpoints_px.shape[0] > 0:
            sample_step = max(1, midpoints_px.shape[0] // 80)
            for x_px, y_px in midpoints_px[::sample_step]:
                cv2.circle(debug_bev, (int(round(x_px)), int(round(y_px))), 2, (0, 0, 255), -1)

        cv2.imshow('lane_binary_mask', binary_mask)
        cv2.imshow('lane_bev_midpoints', debug_bev)
        cv2.waitKey(1)

    def _draw_drivable_area_mask(self, detection_msg):
        image_shape = self._get_mask_shape(detection_msg)
        if image_shape is None:
            return None

        h, w = image_shape
        cv_image = np.zeros((h, w), dtype=np.uint8)

        for detection in detection_msg.detections:
            if detection.class_name != self.drivable_class_name:
                continue

            mask_msg = detection.mask
            if not mask_msg.data:
                continue

            mask_array = np.array(
                [[int(round(ele.x)), int(round(ele.y))] for ele in mask_msg.data],
                dtype=np.int32
            )
            mask_array[:, 0] = np.clip(mask_array[:, 0], 0, w - 1)
            mask_array[:, 1] = np.clip(mask_array[:, 1], 0, h - 1)
            cv2.fillPoly(cv_image, [mask_array], color=255)

        return cv_image

    def _get_mask_shape(self, detection_msg):
        for detection in detection_msg.detections:
            if detection.mask.height > 0 and detection.mask.width > 0:
                return detection.mask.height, detection.mask.width
        return None

    def _bird_convert(self, img):
        h, w = img.shape[:2]
        bev_h = max(1, int(round(h * float(self.bev_height_scale))))
        src_mat = self._scaled_ipm_src_points(w, h)
        dst_mat = np.float32([
            [round(w * self.ipm_dst_left_ratio), 0],
            [round(w * self.ipm_dst_right_ratio), 0],
            [round(w * self.ipm_dst_right_ratio), bev_h - 1],
            [round(w * self.ipm_dst_left_ratio), bev_h - 1],
        ])
        transform_matrix = cv2.getPerspectiveTransform(src_mat, dst_mat)
        warped = cv2.warpPerspective(img, transform_matrix, (w, bev_h), flags=cv2.INTER_NEAREST)
        _, binary_warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
        return binary_warped

    def _scaled_ipm_src_points(self, image_width, image_height):
        src_mat = np.float32(self.ipm_src_points).reshape(4, 2)
        base_w = max(float(self.ipm_src_base_width), 1.0)
        base_h = max(float(self.ipm_src_base_height), 1.0)
        src_mat[:, 0] *= float(image_width) / base_w
        src_mat[:, 1] *= float(image_height) / base_h
        return src_mat

    def _scan_midpoints(self, bev_mask):
        h, _ = bev_mask.shape[:2]
        midpoints = []

        for y_px in range(h - 1, -1, -max(1, int(self.row_stride))):
            x_indices = np.flatnonzero(bev_mask[y_px, :] > 0)
            if x_indices.size < self.min_road_width_px:
                continue

            x_left = int(x_indices[0])
            x_right = int(x_indices[-1])
            if (x_right - x_left + 1) < self.min_road_width_px:
                continue

            x_mid = 0.5 * (x_left + x_right)
            midpoints.append((x_mid, float(y_px)))

        if not midpoints:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(midpoints, dtype=np.float64)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDebugVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
