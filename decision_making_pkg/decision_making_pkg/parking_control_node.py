import numpy as np
import rclpy
from rclpy.node import Node

from interfaces_pkg.msg import MotionCommand
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32

"""
실행 명령어

ls -l /dev/ttyUSB* /dev/ttyACM*

sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyACM0

ros2 launch vehicle_bringup_pkg real_vehicle_parking.launch.py \
  use_visualization:=true \
  use_control:=true \
  use_serial:=true

"""


class MotionNode(Node):
    def __init__(self):
        super().__init__('parking_motion_node')

        self.perc_sub = self.create_subscription(
            Float32MultiArray,
            '/perception_data',
            self.perc_callback,
            10,
        )
        self.control_pub = self.create_publisher(
            MotionCommand,
            '/topic_control_signal',
            10,
        )
        self.stage_pub = self.create_publisher(Int32, '/current_stage', 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.stage = 1
        self.stage_start_time = self.now_sec()
        self.speed = 0
        self.steer = 0
        self.perc_data = [0.0] * 23
        self.stage5_lost_count = 0
        self.stage5_align_stable_count = 0
        self.stage5_alignment_locked = False
        self.parking_complete = False

        self.ROI_POINT_THRESHOLD = 5
        self.STAGE1_ROI_ENABLE_DELAY = 7.0
        self.STAGE1_SPEED = 80
        self.STAGE2_SPEED = 50
        self.STAGE2_STEER = -7
        self.STAGE2_TARGET_DEG = 55.0
        self.STAGE2_DEG_TOL = 10.0
        self.STAGE3_STOP_TIME = 1.5
        self.STAGE3_STEER = 7
        self.STAGE4_SPEED = -60
        self.STAGE4_STEER = 7
        self.STAGE4_DEG_TOL = 2.0
        self.STAGE5_SPEED = -60
        self.STAGE5_STEER = 0
        self.STAGE5_FINE_STEER = 1
        self.STAGE5_ALIGN_TOL = 1.0
        self.STAGE5_ALIGN_STABLE_FRAMES = 8
        self.STAGE5_MIN_REVERSE_TIME = 4
        self.STAGE5_LOST_FRAMES_TO_STOP = 5
        self.STAGE6_STOP_TIME = 5.0
        self.STAGE7_STRAIGHT_TIME = 2.0
        self.STAGE7_RIGHT_TURN_TIME = 10.0
        self.STAGE7_STRAIGHT_SPEED = 50
        self.STAGE7_TURN_SPEED = 100
        self.STAGE7_EXIT_SPEED = 100
        self.STAGE7_RIGHT_STEER = 7
        self.STAGE7_STRAIGHT_STEER = 0
        self.REAR_SIDE_CLEAR_FLAG_INDEX = 22

        self.get_logger().info('Motion Node Started with Stage 1-7 Logic.')

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def set_stage(self, stage):
        self.stage = stage
        self.stage_start_time = self.now_sec()
        if stage == 5:
            self.stage5_lost_count = 0
            self.stage5_align_stable_count = 0
            self.stage5_alignment_locked = False

    def elapsed(self):
        return self.now_sec() - self.stage_start_time

    def change_stage(self, next_stage, log_msg="", stop_car=True):
        if stop_car:
            self.speed, self.steer = 0, 0
        self.set_stage(next_stage)
        if log_msg:
            self.get_logger().info(log_msg)

    def perc_callback(self, msg):
        data_len = min(len(msg.data), len(self.perc_data))
        self.perc_data[:data_len] = msg.data[:data_len]

    def publish_cmd(self):
        stage_msg = Int32()
        stage_msg.data = self.stage
        self.stage_pub.publish(stage_msg)

        cmd_msg = MotionCommand()
        cmd_msg.steering = int(self.steer)
        cmd_msg.left_speed = int(self.speed)
        cmd_msg.right_speed = int(self.speed)
        self.control_pub.publish(cmd_msg)

    def gap_center(self):
        gap_center_x = self.perc_data[16]
        gap_center_y = self.perc_data[17]

        if abs(gap_center_x) > 1e-6 or abs(gap_center_y) > 1e-6:
            return gap_center_x, gap_center_y

        c1_x, c1_y = self.perc_data[8], self.perc_data[9]
        c2_x, c2_y = self.perc_data[10], self.perc_data[11]
        if (
            (abs(c1_x) > 1e-6 or abs(c1_y) > 1e-6)
            and (abs(c2_x) > 1e-6 or abs(c2_y) > 1e-6)
        ):
            return (c1_x + c2_x) / 2.0, (c1_y + c2_y) / 2.0

        return None

    def gap_line_angle_deg(self):
        center = self.gap_center()
        if center is None:
            return None

        gap_center_x, gap_center_y = center
        screen_x = -gap_center_x
        screen_y = -gap_center_y
        if abs(screen_x) < 1e-6 and abs(screen_y) < 1e-6:
            return None

        return float(np.degrees(np.arctan2(screen_y, screen_x)))

    def stage2_angle_ready(self):
        gap_line_deg = self.gap_line_angle_deg()
        if gap_line_deg is None:
            return False

        angle_error = abs(gap_line_deg - self.STAGE2_TARGET_DEG)
        self.get_logger().debug(
            f'Stage 2 gap angle: {gap_line_deg:.1f}deg '
            f'(target {self.STAGE2_TARGET_DEG:.1f}deg)',
            throttle_duration_sec=0.2,
        )
        return angle_error <= self.STAGE2_DEG_TOL

    def update_stage5_alignment(self, is_svm_valid, deg_diff):
        if self.stage5_alignment_locked:
            self.steer = self.STAGE5_STEER
            return

        if not is_svm_valid:
            self.steer = self.STAGE5_STEER
            self.stage5_align_stable_count = 0
            return

        if abs(deg_diff) <= self.STAGE5_ALIGN_TOL:
            self.steer = self.STAGE5_STEER
            self.stage5_align_stable_count += 1
            if (
                self.stage5_align_stable_count
                >= self.STAGE5_ALIGN_STABLE_FRAMES
            ):
                self.stage5_alignment_locked = True
                self.get_logger().info(
                    'deg_diff stabilized within +-1deg. '
                    'Stage 5: lock steering to 0.'
                )
            return

        self.stage5_align_stable_count = 0
        if deg_diff > 0:
            self.steer = self.STAGE5_FINE_STEER
        else:
            self.steer = -self.STAGE5_FINE_STEER

    def control_loop(self):
        point_count = self.perc_data[0]
        is_svm_valid = bool(self.perc_data[1])
        deg_diff = self.perc_data[3]
        two_cars_detected = bool(self.perc_data[15])
        rear_side_clear = bool(self.perc_data[self.REAR_SIDE_CLEAR_FLAG_INDEX])

        if self.stage == 1:
            self.speed, self.steer = self.STAGE1_SPEED, -1
            if (
                self.elapsed() >= self.STAGE1_ROI_ENABLE_DELAY
                and point_count >= self.ROI_POINT_THRESHOLD
            ):
                self.change_stage(
                    2,
                    'ROI detected. Stage 2: max left turn.',
                    stop_car=False,
                )

        elif self.stage == 2:
            self.speed, self.steer = self.STAGE2_SPEED, self.STAGE2_STEER
            if self.stage2_angle_ready():
                self.change_stage(
                    3,
                    'Gap line reached 48deg. Stage 3: pause and set +7 steer.',
                    stop_car=False,
                )

        elif self.stage == 3:
            self.speed, self.steer = 0, self.STAGE3_STEER
            if self.elapsed() >= self.STAGE3_STOP_TIME:
                self.change_stage(
                    4,
                    'Stage 3 pause done. Stage 4: reverse with +7 steer.',
                    stop_car=False,
                )

        elif self.stage == 4:
            self.speed, self.steer = self.STAGE4_SPEED, self.STAGE4_STEER
            if is_svm_valid and abs(deg_diff) <= self.STAGE4_DEG_TOL:
                self.change_stage(
                    5,
                    f'deg_diff={deg_diff:.1f}deg. Stage 5: fine align.',
                    stop_car=False,
                )

        elif self.stage == 5:
            if self.parking_complete:
                self.change_stage(
                    6,
                    'Stage 5 parking complete. Stage 6: stop for 5 sec.',
                    stop_car=True,
                )
            else:
                self.speed = self.STAGE5_SPEED
                self.update_stage5_alignment(is_svm_valid, deg_diff)
                if self.elapsed() >= self.STAGE5_MIN_REVERSE_TIME:
                    if rear_side_clear:
                        self.speed, self.steer = 0, self.STAGE5_STEER
                        self.parking_complete = True
                        self.get_logger().info(
                            'One side near 90/270deg is clear. '
                            'Stage 5: parking complete.'
                        )
                    elif two_cars_detected:
                        self.stage5_lost_count = 0
                    else:
                        self.stage5_lost_count += 1

                    if (
                        not self.parking_complete
                        and self.stage5_lost_count
                        >= self.STAGE5_LOST_FRAMES_TO_STOP
                    ):
                        self.speed = 0
                        self.parking_complete = True
                        self.get_logger().info(
                            'Side cars disappeared. Stage 5: parking complete.'
                        )

        elif self.stage == 6:
            self.speed, self.steer = 0, 0
            if self.elapsed() >= self.STAGE6_STOP_TIME:
                self.change_stage(
                    7,
                    'Stage 6 stop done. Stage 7: exit sequence.',
                    stop_car=False,
                )

        elif self.stage == 7:
            elapsed = self.elapsed()
            if elapsed < self.STAGE7_STRAIGHT_TIME:
                self.speed = self.STAGE7_STRAIGHT_SPEED
                self.steer = self.STAGE7_STRAIGHT_STEER
            elif elapsed < self.STAGE7_STRAIGHT_TIME + self.STAGE7_RIGHT_TURN_TIME:
                self.speed = self.STAGE7_TURN_SPEED
                self.steer = self.STAGE7_RIGHT_STEER
            else:
                self.speed = self.STAGE7_EXIT_SPEED
                self.steer = self.STAGE7_STRAIGHT_STEER

        else:
            self.speed, self.steer = 0, 0
            self.stage = 7

        self.publish_cmd()


def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
