import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from interfaces_pkg.msg import LaneTrajectory, MotionCommand
from .lib.lane_keeping_controller import LaneKeepingController


#---------------Variable Setting---------------
SUB_TRAJECTORY_TOPIC_NAME = "lane_trajectory"
PUB_TOPIC_NAME = "topic_control_signal"

TIMER = 0.05
VEHICLE_SPEED_MPS = 0.5
LEFT_SPEED_COMMAND = 255
RIGHT_SPEED_COMMAND = 255
# vehicle_speed_mps 크게 설정
# → 같은 CTE에도 조향 반응 약해짐

# vehicle_speed_mps 작게 설정
# → 같은 CTE에도 조향 반응 강해짐

# softening_speed_mps 크게 설정
# → 저속에서 조향 튐 완화

STANLEY_GAIN = 1.0 # 1.4
HEADING_GAIN = 0.1
CURVATURE_GAIN = 0.0
SOFTENING_SPEED_MPS = 0.15
FRONT_AXLE_OFFSET_M = 0.0

MAX_STEER_RAD = np.deg2rad(20.0)
MAX_STEERING_STEP = 7
STEERING_SIGN = -1.0
LANE_TIMEOUT_SEC = 0.8
LOG_PERIOD_SEC = 0.5
#----------------------------------------------


class StanleyControllerNode(Node):
    def __init__(self):
        super().__init__('stanley_controller_node')

        self.sub_trajectory_topic = self.declare_parameter(
            'sub_trajectory_topic', SUB_TRAJECTORY_TOPIC_NAME
        ).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.timer_period = self.declare_parameter('timer', TIMER).value

        self.vehicle_speed_mps = self.declare_parameter('vehicle_speed_mps', VEHICLE_SPEED_MPS).value
        self.left_speed_command = self.declare_parameter('left_speed_command', LEFT_SPEED_COMMAND).value
        self.right_speed_command = self.declare_parameter('right_speed_command', RIGHT_SPEED_COMMAND).value

        self.stanley_gain = self.declare_parameter('stanley_gain', STANLEY_GAIN).value
        self.heading_gain = self.declare_parameter('heading_gain', HEADING_GAIN).value
        self.curvature_gain = self.declare_parameter('curvature_gain', CURVATURE_GAIN).value
        self.softening_speed_mps = self.declare_parameter('softening_speed_mps', SOFTENING_SPEED_MPS).value
        self.front_axle_offset_m = self.declare_parameter('front_axle_offset_m', FRONT_AXLE_OFFSET_M).value

        self.max_steer_rad = self.declare_parameter('max_steer_rad', float(MAX_STEER_RAD)).value
        self.max_steering_step = self.declare_parameter('max_steering_step', MAX_STEERING_STEP).value
        self.steering_sign = self.declare_parameter('steering_sign', STEERING_SIGN).value
        self.lane_timeout_sec = self.declare_parameter('lane_timeout_sec', LANE_TIMEOUT_SEC).value
        self.log_period_sec = self.declare_parameter('log_period_sec', LOG_PERIOD_SEC).value

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.controller = LaneKeepingController(
            stanley_gain=self.stanley_gain,
            curvature_gain=self.curvature_gain,
            softening_speed_mps=self.softening_speed_mps,
            front_axle_offset_m=self.front_axle_offset_m,
            heading_gain=self.heading_gain,
        )

        self.latest_lane_trajectory = None
        self.latest_lane_time = None
        self.latest_log_time = None

        self.trajectory_sub = self.create_subscription(
            LaneTrajectory,
            self.sub_trajectory_topic,
            self.trajectory_callback,
            self.qos_profile
        )
        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def trajectory_callback(self, msg: LaneTrajectory):
        if not msg.valid:
            self.latest_lane_trajectory = None
            self.latest_lane_time = None
            return

        self.latest_lane_trajectory = msg
        self.latest_lane_time = self.get_clock().now()

    def timer_callback(self):
        if self._lane_is_unavailable():
            self._publish_motion_command(0, 50, 50)
            return

        coeffs = (
            self.latest_lane_trajectory.a,
            self.latest_lane_trajectory.b,
            self.latest_lane_trajectory.c,
        )
        delta_rad, debug = self.controller.compute_delta(coeffs, self.vehicle_speed_mps)
        steering_command = self._delta_to_steering_step(delta_rad)

        if self._should_log():
            self.get_logger().info(
                f"delta={np.rad2deg(delta_rad):.2f}deg, "
                f"steering={steering_command}, "
                f"cte={debug.cte_m:.3f}m, "
                f"theta_e={np.rad2deg(debug.theta_e_rad):.2f}deg, "
                f"kappa={debug.curvature:.3f}",
            )

        self._publish_motion_command(
            steering_command,
            int(self.left_speed_command),
            int(self.right_speed_command)
        )

    def _lane_is_unavailable(self):
        if self.latest_lane_trajectory is None or self.latest_lane_time is None:
            return True

        elapsed = self.get_clock().now() - self.latest_lane_time
        return elapsed.nanoseconds * 1e-9 > self.lane_timeout_sec

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

    def _delta_to_steering_step(self, delta_rad):
        max_steer_rad = max(abs(float(self.max_steer_rad)), 1e-6)
        normalized = np.clip(delta_rad / max_steer_rad, -1.0, 1.0)
        steering = self.steering_sign * normalized * self.max_steering_step
        return int(np.clip(round(steering), -self.max_steering_step, self.max_steering_step))

    def _publish_motion_command(self, steering, left_speed, right_speed):
        motion_command_msg = MotionCommand()
        motion_command_msg.steering = int(steering)
        motion_command_msg.left_speed = int(left_speed)
        motion_command_msg.right_speed = int(right_speed)
        self.publisher.publish(motion_command_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StanleyControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node._publish_motion_command(0, 0, 0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
