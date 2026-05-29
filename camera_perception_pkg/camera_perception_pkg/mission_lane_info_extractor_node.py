import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from interfaces_pkg.msg import DetectionArray, LaneTrajectory


SUB_TOPIC_NAME = "detections"
RIGHT_TRAJECTORY_TOPIC_NAME = "mission/right_lane_trajectory"
LEFT_TRAJECTORY_TOPIC_NAME = "mission/left_lane_trajectory"

RIGHT_LANE_CLASS_NAME = "lane"
LEFT_LANE_CLASS_NAME = "lane_left"

IPM_SRC_POINTS = [238.0, 316.0, 402.0, 313.0, 501.0, 476.0, 155.0, 476.0]
IPM_SRC_BASE_WIDTH = 640.0
IPM_SRC_BASE_HEIGHT = 480.0
IPM_DST_LEFT_RATIO = 0.3
IPM_DST_RIGHT_RATIO = 0.7
BEV_HEIGHT_SCALE = 4.0

ROW_STRIDE = 4
MIN_ROAD_WIDTH_PX = 20
MIN_FIT_POINTS = 30
LATERAL_METER_PER_PIXEL = 0.005
LONGITUDINAL_METER_PER_PIXEL = 0.01
EGO_X_PX = -1.0


class MissionLaneInfoExtractorNode(Node):
    def __init__(self):
        super().__init__("mission_lane_info_extractor_node")

        self.sub_topic = self.declare_parameter("sub_detection_topic", SUB_TOPIC_NAME).value
        self.right_trajectory_topic = self.declare_parameter(
            "right_trajectory_topic",
            RIGHT_TRAJECTORY_TOPIC_NAME,
        ).value
        self.left_trajectory_topic = self.declare_parameter(
            "left_trajectory_topic",
            LEFT_TRAJECTORY_TOPIC_NAME,
        ).value

        self.right_lane_class_name = self.declare_parameter(
            "right_lane_class_name",
            RIGHT_LANE_CLASS_NAME,
        ).value
        self.left_lane_class_name = self.declare_parameter(
            "left_lane_class_name",
            LEFT_LANE_CLASS_NAME,
        ).value

        self.ipm_src_points = self.declare_parameter("ipm_src_points", IPM_SRC_POINTS).value
        self.ipm_src_base_width = self.declare_parameter("ipm_src_base_width", IPM_SRC_BASE_WIDTH).value
        self.ipm_src_base_height = self.declare_parameter("ipm_src_base_height", IPM_SRC_BASE_HEIGHT).value
        self.ipm_dst_left_ratio = self.declare_parameter("ipm_dst_left_ratio", IPM_DST_LEFT_RATIO).value
        self.ipm_dst_right_ratio = self.declare_parameter("ipm_dst_right_ratio", IPM_DST_RIGHT_RATIO).value
        self.bev_height_scale = self.declare_parameter("bev_height_scale", BEV_HEIGHT_SCALE).value

        self.row_stride = self.declare_parameter("row_stride", ROW_STRIDE).value
        self.min_road_width_px = self.declare_parameter("min_road_width_px", MIN_ROAD_WIDTH_PX).value
        self.min_fit_points = self.declare_parameter("min_fit_points", MIN_FIT_POINTS).value
        self.lateral_meter_per_pixel = self.declare_parameter(
            "lateral_meter_per_pixel",
            LATERAL_METER_PER_PIXEL,
        ).value
        self.longitudinal_meter_per_pixel = self.declare_parameter(
            "longitudinal_meter_per_pixel",
            LONGITUDINAL_METER_PER_PIXEL,
        ).value
        self.ego_x_px = self.declare_parameter("ego_x_px", EGO_X_PX).value

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.subscriber = self.create_subscription(
            DetectionArray,
            self.sub_topic,
            self.detections_callback,
            self.qos_profile,
        )
        self.right_trajectory_publisher = self.create_publisher(
            LaneTrajectory,
            self.right_trajectory_topic,
            self.qos_profile,
        )
        self.left_trajectory_publisher = self.create_publisher(
            LaneTrajectory,
            self.left_trajectory_topic,
            self.qos_profile,
        )

    def detections_callback(self, detection_msg: DetectionArray):
        self._process_lane(
            detection_msg,
            self.right_lane_class_name,
            self.right_trajectory_publisher,
        )
        self._process_lane(
            detection_msg,
            self.left_lane_class_name,
            self.left_trajectory_publisher,
        )

    def _process_lane(self, detection_msg, class_name, publisher):
        binary_mask = self._draw_lane_mask(detection_msg, class_name)
        if binary_mask is None or cv2.countNonZero(binary_mask) == 0:
            self._publish_invalid_trajectory(detection_msg, publisher)
            return

        bev_mask = self._bird_convert(binary_mask)
        midpoints_px = self._scan_midpoints(bev_mask)
        if midpoints_px.shape[0] < self.min_fit_points:
            self._publish_invalid_trajectory(detection_msg, publisher)
            return

        x_forward_m, y_lateral_m = self._pixel_to_meter(midpoints_px, bev_mask.shape)
        coeffs, fit_error_m = self._fit_polynomial(x_forward_m, y_lateral_m)
        self._publish_trajectory(
            detection_msg,
            publisher,
            coeffs,
            x_forward_m,
            fit_error_m,
            midpoints_px.shape[0],
        )

    def _draw_lane_mask(self, detection_msg: DetectionArray, class_name: str):
        image_shape = self._get_mask_shape(detection_msg)
        if image_shape is None:
            return None

        h, w = image_shape
        cv_image = np.zeros((h, w), dtype=np.uint8)

        for detection in detection_msg.detections:
            if detection.class_name != class_name:
                continue

            mask_msg = detection.mask
            if not mask_msg.data:
                continue

            mask_array = np.array(
                [[int(round(ele.x)), int(round(ele.y))] for ele in mask_msg.data],
                dtype=np.int32,
            )
            mask_array[:, 0] = np.clip(mask_array[:, 0], 0, w - 1)
            mask_array[:, 1] = np.clip(mask_array[:, 1], 0, h - 1)
            cv2.fillPoly(cv_image, [mask_array], color=255)

        return cv_image

    def _get_mask_shape(self, detection_msg: DetectionArray):
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

    def _pixel_to_meter(self, midpoints_px, bev_shape):
        h, w = bev_shape[:2]
        ego_x_px = self.ego_x_px if self.ego_x_px >= 0.0 else (w - 1) * 0.5

        x_px = midpoints_px[:, 0]
        y_px = midpoints_px[:, 1]

        x_forward_m = (h - 1.0 - y_px) * self.longitudinal_meter_per_pixel
        y_lateral_m = (ego_x_px - x_px) * self.lateral_meter_per_pixel

        order = np.argsort(x_forward_m)
        return x_forward_m[order], y_lateral_m[order]

    def _fit_polynomial(self, x_forward_m, y_lateral_m):
        unique_x, unique_indices = np.unique(x_forward_m, return_index=True)
        unique_y = y_lateral_m[unique_indices]

        coeffs = np.polyfit(unique_x, unique_y, deg=2)
        fitted_y = np.polyval(coeffs, unique_x)
        fit_error_m = float(np.sqrt(np.mean((fitted_y - unique_y) ** 2)))
        return coeffs, fit_error_m

    def _publish_trajectory(self, detection_msg, publisher, coeffs, x_forward_m, fit_error_m, num_points):
        lane_trajectory = LaneTrajectory()
        lane_trajectory.header = detection_msg.header
        lane_trajectory.valid = True
        lane_trajectory.a = float(coeffs[0])
        lane_trajectory.b = float(coeffs[1])
        lane_trajectory.c = float(coeffs[2])
        lane_trajectory.x_min_m = float(np.min(x_forward_m))
        lane_trajectory.x_max_m = float(np.max(x_forward_m))
        lane_trajectory.fit_error_m = float(fit_error_m)
        lane_trajectory.num_points = int(num_points)
        publisher.publish(lane_trajectory)

    def _publish_invalid_trajectory(self, detection_msg, publisher):
        lane_trajectory = LaneTrajectory()
        lane_trajectory.header = detection_msg.header
        lane_trajectory.valid = False
        publisher.publish(lane_trajectory)


def main(args=None):
    rclpy.init(args=args)
    node = MissionLaneInfoExtractorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
