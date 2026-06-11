import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from sensor_msgs.msg import Image
from interfaces_pkg.msg import DetectionArray, LaneTrajectory, MotionCommand
from .lib.lane_keeping_controller import LaneKeepingController


RIGHT_LANE_TOPIC = "mission/right_lane_trajectory"
LEFT_LANE_TOPIC = "mission/left_lane_trajectory"
OBJECT_DETECTION_TOPIC = "object_detections"
IMAGE_TOPIC = "image_raw"
PUB_TOPIC_NAME = "topic_control_signal"

TIMER = 0.05
VEHICLE_SPEED_MPS = 0.5
LEFT_SPEED_COMMAND = 150
RIGHT_SPEED_COMMAND = 150
LANE_UNAVAILABLE_SPEED = 50

STANLEY_GAIN = 0.8
HEADING_GAIN = 0.1 #0.3
CURVATURE_GAIN = 0.0
SOFTENING_SPEED_MPS = 0.15
FRONT_AXLE_OFFSET_M = 0.0

MAX_STEER_RAD = np.deg2rad(20.0)
MAX_STEERING_STEP = 7
STEERING_SIGN = -1.0
LANE_TIMEOUT_SEC = 1.0
LOG_PERIOD_SEC = 0.5

LEFT_SHIFT_DURATION_SEC = 3.0
RIGHT_SHIFT_DURATION_SEC = 1.0
LEFT_SHIFT_STEERING = -7
RIGHT_SHIFT_STEERING = 7
ENABLE_STAGE2_TO_STAGE3 = True
STAGE2_TO_STAGE3_LOCKOUT_SEC = 3.0

DEFAULT_IMAGE_HEIGHT = 240
STAGE1_OBSTACLE_CLOSE_RATIO_FROM_TOP = 0.65 # 0.7
STAGE2_OBSTACLE_CLOSE_RATIO_FROM_TOP = 0.75
TRAFFIC_LIGHT_CLOSE_BOTTOM_Y_PX = 75.2 # 73.0

STAGE1_OBSTACLE_CLASS_NAMES = ["car_2"]
STAGE2_OBSTACLE_CLASS_NAMES = ["car_3"]
RED_LIGHT_CLASS_NAMES = ["traffic_red", "red", "red_light", "traffic_light_red", "red_traffic_light"]
GREEN_LIGHT_CLASS_NAMES = ["traffic_green", "green", "green_light", "traffic_light_green", "green_traffic_light"]

STAGE_1_RIGHT_LANE = 1
STAGE_2_LEFT_SHIFT = 2
STAGE_2_LEFT_LANE = 20
STAGE_3_RIGHT_SHIFT = 3
STAGE_3_RIGHT_LANE = 30
STAGE_3_STOP_RED = 31
STAGE_4_RIGHT_LANE = 4

STAGE_NAMES = {
    STAGE_1_RIGHT_LANE: "stage1_right_lane",
    STAGE_2_LEFT_SHIFT: "stage2_left_shift",
    STAGE_2_LEFT_LANE: "stage2_left_lane",
    STAGE_3_RIGHT_SHIFT: "stage3_right_shift",
    STAGE_3_RIGHT_LANE: "stage3_right_lane",
    STAGE_3_STOP_RED: "stage3_stop_red",
    STAGE_4_RIGHT_LANE: "stage4_right_lane",
}


class ObstacleMissionControllerNode(Node):
    def __init__(self):
        super().__init__("obstacle_mission_controller_node")

        self.right_lane_topic = self.declare_parameter("right_lane_topic", RIGHT_LANE_TOPIC).value
        self.left_lane_topic = self.declare_parameter("left_lane_topic", LEFT_LANE_TOPIC).value
        self.object_detection_topic = self.declare_parameter(
            "object_detection_topic",
            OBJECT_DETECTION_TOPIC,
        ).value
        self.image_topic = self.declare_parameter("image_topic", IMAGE_TOPIC).value
        self.pub_topic = self.declare_parameter("pub_topic", PUB_TOPIC_NAME).value
        self.timer_period = self.declare_parameter("timer", TIMER).value

        self.vehicle_speed_mps = self.declare_parameter("vehicle_speed_mps", VEHICLE_SPEED_MPS).value
        self.left_speed_command = self.declare_parameter("left_speed_command", LEFT_SPEED_COMMAND).value
        self.right_speed_command = self.declare_parameter("right_speed_command", RIGHT_SPEED_COMMAND).value
        self.lane_unavailable_speed = self.declare_parameter(
            "lane_unavailable_speed",
            LANE_UNAVAILABLE_SPEED,
        ).value

        self.stanley_gain = self.declare_parameter("stanley_gain", STANLEY_GAIN).value
        self.heading_gain = self.declare_parameter("heading_gain", HEADING_GAIN).value
        self.curvature_gain = self.declare_parameter("curvature_gain", CURVATURE_GAIN).value
        self.softening_speed_mps = self.declare_parameter(
            "softening_speed_mps",
            SOFTENING_SPEED_MPS,
        ).value
        self.front_axle_offset_m = self.declare_parameter(
            "front_axle_offset_m",
            FRONT_AXLE_OFFSET_M,
        ).value

        self.max_steer_rad = self.declare_parameter("max_steer_rad", float(MAX_STEER_RAD)).value
        self.max_steering_step = self.declare_parameter("max_steering_step", MAX_STEERING_STEP).value
        self.steering_sign = self.declare_parameter("steering_sign", STEERING_SIGN).value
        self.lane_timeout_sec = self.declare_parameter("lane_timeout_sec", LANE_TIMEOUT_SEC).value
        self.log_period_sec = self.declare_parameter("log_period_sec", LOG_PERIOD_SEC).value

        self.left_shift_duration_sec = self.declare_parameter(
            "left_shift_duration_sec",
            LEFT_SHIFT_DURATION_SEC,
        ).value
        self.right_shift_duration_sec = self.declare_parameter(
            "right_shift_duration_sec",
            RIGHT_SHIFT_DURATION_SEC,
        ).value
        self.left_shift_steering = self.declare_parameter("left_shift_steering", LEFT_SHIFT_STEERING).value
        self.right_shift_steering = self.declare_parameter("right_shift_steering", RIGHT_SHIFT_STEERING).value
        self.enable_stage2_to_stage3 = self.declare_parameter(
            "enable_stage2_to_stage3",
            ENABLE_STAGE2_TO_STAGE3,
        ).value
        self.stage2_to_stage3_lockout_sec = self.declare_parameter(
            "stage2_to_stage3_lockout_sec",
            STAGE2_TO_STAGE3_LOCKOUT_SEC,
        ).value

        self.default_image_height = self.declare_parameter(
            "default_image_height",
            DEFAULT_IMAGE_HEIGHT,
        ).value
        self.stage1_obstacle_close_ratio_from_top = self.declare_parameter(
            "stage1_obstacle_close_ratio_from_top",
            STAGE1_OBSTACLE_CLOSE_RATIO_FROM_TOP,
        ).value
        self.stage2_obstacle_close_ratio_from_top = self.declare_parameter(
            "stage2_obstacle_close_ratio_from_top",
            STAGE2_OBSTACLE_CLOSE_RATIO_FROM_TOP,
        ).value
        self.traffic_light_close_bottom_y_px = self.declare_parameter(
            "traffic_light_close_bottom_y_px",
            TRAFFIC_LIGHT_CLOSE_BOTTOM_Y_PX,
        ).value
        self.stage1_obstacle_class_names = self.declare_parameter(
            "stage1_obstacle_class_names",
            STAGE1_OBSTACLE_CLASS_NAMES,
        ).value
        self.stage2_obstacle_class_names = self.declare_parameter(
            "stage2_obstacle_class_names",
            STAGE2_OBSTACLE_CLASS_NAMES,
        ).value
        self.red_light_class_names = self.declare_parameter(
            "red_light_class_names",
            RED_LIGHT_CLASS_NAMES,
        ).value
        self.green_light_class_names = self.declare_parameter(
            "green_light_class_names",
            GREEN_LIGHT_CLASS_NAMES,
        ).value

        self.controller = LaneKeepingController(
            stanley_gain=self.stanley_gain,
            curvature_gain=self.curvature_gain,
            softening_speed_mps=self.softening_speed_mps,
            front_axle_offset_m=self.front_axle_offset_m,
            heading_gain=self.heading_gain,
        )

        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.right_lane = None
        self.right_lane_time = None
        self.left_lane = None
        self.left_lane_time = None
        self.latest_detections = DetectionArray()
        self.image_height = int(self.default_image_height)

        self.stage = STAGE_1_RIGHT_LANE
        self.stage_start_time = self.get_clock().now()
        self.latest_log_time = None

        self.right_lane_sub = self.create_subscription(
            LaneTrajectory,
            self.right_lane_topic,
            self.right_lane_callback,
            reliable_qos,
        )
        self.left_lane_sub = self.create_subscription(
            LaneTrajectory,
            self.left_lane_topic,
            self.left_lane_callback,
            reliable_qos,
        )
        self.object_sub = self.create_subscription(
            DetectionArray,
            self.object_detection_topic,
            self.object_detections_callback,
            reliable_qos,
        )
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            image_qos,
        )
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, reliable_qos)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info("Obstacle mission controller started")

    def right_lane_callback(self, msg: LaneTrajectory):
        if msg.valid:
            self.right_lane = msg
            self.right_lane_time = self.get_clock().now()

    def left_lane_callback(self, msg: LaneTrajectory):
        if msg.valid:
            self.left_lane = msg
            self.left_lane_time = self.get_clock().now()

    def object_detections_callback(self, msg: DetectionArray):
        self.latest_detections = msg

    def image_callback(self, msg: Image):
        if msg.height > 0:
            self.image_height = int(msg.height)

    def timer_callback(self):
        self._update_stage()

        if self.stage == STAGE_2_LEFT_SHIFT:
            self._publish_drive(self.left_shift_steering)
            self._log_status("forced left shift")
            return

        if self.stage == STAGE_3_RIGHT_SHIFT:
            self._publish_drive(self.right_shift_steering)
            self._log_status("forced right shift")
            return

        if self.stage == STAGE_3_STOP_RED:
            self._publish_motion_command(0, 0, 0)
            self._log_status("stopped at red light")
            return

        if (
            self.stage == STAGE_1_RIGHT_LANE
            or self.stage == STAGE_3_RIGHT_LANE
            or self.stage == STAGE_4_RIGHT_LANE
        ):
            self._follow_lane(self.right_lane, self.right_lane_time, "right")
            return

        if self.stage == STAGE_2_LEFT_LANE:
            self._follow_lane(self.left_lane, self.left_lane_time, "left")
            return

        self._publish_motion_command(0, 0, 0)

    def _update_stage(self):
        if self.stage == STAGE_1_RIGHT_LANE and self._obstacle_is_close(
            self.stage1_obstacle_class_names,
            self.stage1_obstacle_close_ratio_from_top,
        ):
            self._set_stage(STAGE_2_LEFT_SHIFT)
            return

        if self.stage == STAGE_2_LEFT_SHIFT and self._stage_elapsed_sec() >= self.left_shift_duration_sec:
            self._set_stage(STAGE_2_LEFT_LANE)
            return

        if (
            self.enable_stage2_to_stage3
            and self.stage == STAGE_2_LEFT_LANE
            and self._stage_elapsed_sec() >= self.stage2_to_stage3_lockout_sec
            and self._obstacle_is_close(
                self.stage2_obstacle_class_names,
                self.stage2_obstacle_close_ratio_from_top,
            )
        ):
            self._set_stage(STAGE_3_RIGHT_SHIFT)
            return

        if self.stage == STAGE_3_RIGHT_SHIFT and self._stage_elapsed_sec() >= self.right_shift_duration_sec:
            self._set_stage(STAGE_3_RIGHT_LANE)
            return

        if self.stage == STAGE_3_RIGHT_LANE and self._red_light_is_close():
            self._set_stage(STAGE_3_STOP_RED)
            return

        if self.stage == STAGE_3_STOP_RED and self._green_light_is_close():
            self._set_stage(STAGE_4_RIGHT_LANE)

    def _set_stage(self, stage):
        if self.stage == stage:
            return

        old_stage_name = STAGE_NAMES.get(self.stage, str(self.stage))
        new_stage_name = STAGE_NAMES.get(stage, str(stage))
        self.stage = stage
        self.stage_start_time = self.get_clock().now()
        self.get_logger().warn(f"Mission stage changed: {old_stage_name} -> {new_stage_name}")

    def _stage_elapsed_sec(self):
        elapsed = self.get_clock().now() - self.stage_start_time
        return elapsed.nanoseconds * 1e-9

    def _follow_lane(self, lane_msg, lane_time, lane_name):
        if self._lane_is_unavailable(lane_msg, lane_time):
            speed = int(self.lane_unavailable_speed)
            self._publish_motion_command(0, speed, speed)
            self._log_status(f"{lane_name} lane unavailable")
            return

        coeffs = (lane_msg.a, lane_msg.b, lane_msg.c)
        delta_rad, debug = self.controller.compute_delta(coeffs, self.vehicle_speed_mps)
        steering_command = self._delta_to_steering_step(delta_rad)
        self._publish_drive(steering_command)

        if self._should_log():
            self.get_logger().info(
                f"{STAGE_NAMES.get(self.stage)} {lane_name}: "
                f"delta={np.rad2deg(delta_rad):.2f}deg, "
                f"steering={steering_command}, "
                f"cte={debug.cte_m:.3f}m, "
                f"theta_e={np.rad2deg(debug.theta_e_rad):.2f}deg"
            )

    def _lane_is_unavailable(self, lane_msg, lane_time):
        if lane_msg is None or lane_time is None:
            return True

        elapsed = self.get_clock().now() - lane_time
        return elapsed.nanoseconds * 1e-9 > self.lane_timeout_sec

    def _obstacle_is_close(self, class_names, close_ratio_from_top):
        threshold_y = float(self.image_height) * float(close_ratio_from_top)
        for detection in self.latest_detections.detections:
            if self._class_matches(detection.class_name, class_names):
                if self._bbox_bottom_y(detection) >= threshold_y:
                    return True
        return False

    def _red_light_is_close(self):
        return self._traffic_light_is_close(self.red_light_class_names)

    def _green_light_is_close(self):
        return self._traffic_light_is_close(self.green_light_class_names)

    def _traffic_light_is_close(self, class_names):
        threshold_y = float(self.traffic_light_close_bottom_y_px)
        for detection in self.latest_detections.detections:
            if self._class_matches(detection.class_name, class_names):
                if self._bbox_bottom_y(detection) <= threshold_y:
                    return True
        return False

    def _bbox_bottom_y(self, detection):
        bbox = detection.bbox
        return float(bbox.center.position.y) + 0.5 * float(bbox.size.y)

    def _class_matches(self, class_name, allowed_names):
        normalized = str(class_name).strip().lower()
        for allowed in allowed_names:
            allowed_normalized = str(allowed).strip().lower()
            if normalized == allowed_normalized:
                return True
        return False

    def _delta_to_steering_step(self, delta_rad):
        max_steer_rad = max(abs(float(self.max_steer_rad)), 1e-6)
        normalized = np.clip(delta_rad / max_steer_rad, -1.0, 1.0)
        steering = self.steering_sign * normalized * self.max_steering_step
        return int(np.clip(round(steering), -self.max_steering_step, self.max_steering_step))

    def _publish_drive(self, steering):
        self._publish_motion_command(
            int(steering),
            int(self.left_speed_command),
            int(self.right_speed_command),
        )

    def _publish_motion_command(self, steering, left_speed, right_speed):
        motion_command_msg = MotionCommand()
        motion_command_msg.steering = int(steering)
        motion_command_msg.left_speed = int(left_speed)
        motion_command_msg.right_speed = int(right_speed)
        self.publisher.publish(motion_command_msg)

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

    def _log_status(self, detail):
        if self._should_log():
            self.get_logger().info(f"{STAGE_NAMES.get(self.stage)}: {detail}")


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleMissionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node._publish_motion_command(0, 0, 0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
