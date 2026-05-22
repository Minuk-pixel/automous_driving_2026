import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from interfaces_pkg.msg import TargetPoint, LaneInfo, LaneTrajectory, DetectionArray

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"
PUB_TRAJECTORY_TOPIC_NAME = "lane_trajectory"

# YOLO segmentation class for the drivable road area
DRIVABLE_CLASS_NAME = "lane"

# IPM source points in the original camera image.
IPM_SRC_POINTS = [238.0, 316.0, 402.0, 313.0, 501.0, 476.0, 155.0, 476.0]
IPM_DST_LEFT_RATIO = 0.3
IPM_DST_RIGHT_RATIO = 0.7

# Row scan and meter calibration parameters.
ROW_STRIDE = 4
MIN_ROAD_WIDTH_PX = 20
MIN_FIT_POINTS = 30
LATERAL_METER_PER_PIXEL = 0.005
LONGITUDINAL_METER_PER_PIXEL = 0.01
EGO_X_PX = -1.0  # negative value means image center
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.pub_trajectory_topic = self.declare_parameter('pub_trajectory_topic', PUB_TRAJECTORY_TOPIC_NAME).value
        self.drivable_class_name = self.declare_parameter('drivable_class_name', DRIVABLE_CLASS_NAME).value
        self.ipm_src_points = self.declare_parameter('ipm_src_points', IPM_SRC_POINTS).value
        self.ipm_dst_left_ratio = self.declare_parameter('ipm_dst_left_ratio', IPM_DST_LEFT_RATIO).value
        self.ipm_dst_right_ratio = self.declare_parameter('ipm_dst_right_ratio', IPM_DST_RIGHT_RATIO).value
        self.row_stride = self.declare_parameter('row_stride', ROW_STRIDE).value
        self.min_road_width_px = self.declare_parameter('min_road_width_px', MIN_ROAD_WIDTH_PX).value
        self.min_fit_points = self.declare_parameter('min_fit_points', MIN_FIT_POINTS).value
        self.lateral_meter_per_pixel = self.declare_parameter('lateral_meter_per_pixel', LATERAL_METER_PER_PIXEL).value
        self.longitudinal_meter_per_pixel = self.declare_parameter('longitudinal_meter_per_pixel', LONGITUDINAL_METER_PER_PIXEL).value
        self.ego_x_px = self.declare_parameter('ego_x_px', EGO_X_PX).value

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)
        self.trajectory_publisher = self.create_publisher(LaneTrajectory, self.pub_trajectory_topic, self.qos_profile)

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            self._publish_invalid_trajectory(detection_msg)
            self._publish_legacy_lane_info(None, None)
            return

        binary_mask = self._draw_drivable_area_mask(detection_msg)
        if binary_mask is None or cv2.countNonZero(binary_mask) == 0:
            self._publish_invalid_trajectory(detection_msg)
            self._publish_legacy_lane_info(None, None)
            return

        bev_mask = self._bird_convert(binary_mask)
        midpoints_px = self._scan_midpoints(bev_mask)

        if midpoints_px.shape[0] < self.min_fit_points:
            self.get_logger().warn(
                f"Not enough midpoint rows for lane fitting: {midpoints_px.shape[0]}"
            )
            self._publish_invalid_trajectory(detection_msg)
            self._publish_legacy_lane_info(None, None)
            return

        x_forward_m, y_lateral_m = self._pixel_to_meter(midpoints_px, bev_mask.shape)
        coeffs, fit_error_m = self._fit_polynomial(x_forward_m, y_lateral_m)

        self._publish_trajectory(detection_msg, coeffs, x_forward_m, fit_error_m, midpoints_px.shape[0])
        self._publish_legacy_lane_info(midpoints_px, coeffs)

    def _draw_drivable_area_mask(self, detection_msg: DetectionArray):
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

    def _get_mask_shape(self, detection_msg: DetectionArray):
        for detection in detection_msg.detections:
            if detection.mask.height > 0 and detection.mask.width > 0:
                return detection.mask.height, detection.mask.width
        return None

    def _bird_convert(self, img):
        h, w = img.shape[:2]
        src_mat = np.float32(self.ipm_src_points).reshape(4, 2)
        dst_mat = np.float32([
            [round(w * self.ipm_dst_left_ratio), 0],
            [round(w * self.ipm_dst_right_ratio), 0],
            [round(w * self.ipm_dst_right_ratio), h - 1],
            [round(w * self.ipm_dst_left_ratio), h - 1],
        ])
        transform_matrix = cv2.getPerspectiveTransform(src_mat, dst_mat)
        warped = cv2.warpPerspective(img, transform_matrix, (w, h), flags=cv2.INTER_NEAREST)
        _, binary_warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
        return binary_warped

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

    def _publish_trajectory(self, detection_msg, coeffs, x_forward_m, fit_error_m, num_points):
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
        self.trajectory_publisher.publish(lane_trajectory)

    def _publish_invalid_trajectory(self, detection_msg):
        lane_trajectory = LaneTrajectory()
        lane_trajectory.header = detection_msg.header
        lane_trajectory.valid = False
        self.trajectory_publisher.publish(lane_trajectory)

    def _publish_legacy_lane_info(self, midpoints_px, coeffs):
        target_points = []

        if midpoints_px is not None and midpoints_px.shape[0] > 0:
            sample_indices = np.linspace(0, midpoints_px.shape[0] - 1, min(3, midpoints_px.shape[0]), dtype=np.int64)
            for idx in sample_indices:
                target_point = TargetPoint()
                target_point.target_x = int(round(midpoints_px[idx, 0]))
                target_point.target_y = int(round(midpoints_px[idx, 1]))
                target_points.append(target_point)

        lane = LaneInfo()
        lane.slope = 0.0 if coeffs is None else float(np.rad2deg(np.arctan(coeffs[1])))
        lane.target_points = target_points

        self.publisher.publish(lane)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()
  
if __name__ == '__main__':
    main()
