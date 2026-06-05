import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int32
from interfaces_pkg.msg import MotionCommand

LIDAR_MIN_RANGE_M = 0.3
LIDAR_MAX_RANGE_M = 3.0

class MotionNode(Node):
    def __init__(self):
        super().__init__('parking_motion_node')

        # Subscriber & Publisher
        self.perc_sub = self.create_subscription(Float32MultiArray, '/perception_data', self.perc_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/lidar_raw', self.scan_callback, 10)
        self.control_pub = self.create_publisher(MotionCommand, '/topic_control_signal', 10)
        self.stage_pub = self.create_publisher(Int32, '/current_stage', 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        # 제어 변수
        self.stage = 1
        self.stage_start_time = self.now_sec()
        self.speed = 0
        self.steer = 0
        self.perc_data = [0.0] * 12

        self.upper_count = 0
        self.upper_max_x = 9999.0

        # 탈출(Exit) 목표 각도 변수
        self.saved_svm_deg = 0.0
        self.target_exit_deg = 0.0

        # === [튜닝 파라미터 모음] ===
        self.STOP_X_MARGIN = -100.0
        self.EXIT_X_MARGIN = 600.0
        self.UPPER_Y_MIN = 300.0
        self.UPPER_Y_MAX = 3000.0
        self.X_RANGE_LIMIT = 2500.0
        self.MIN_STOP_POINTS = 3

        self.EXIT_SPEED = 30
        self.EXIT_MAX_STEER = 6.0 
        self.EXIT_KP = 0.003 # 동적 우회전 P제어 상수
        # ============================

        self.get_logger().info('Motion Node Started with Optimized Logic.')

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def set_stage(self, stage):
        self.stage = stage
        self.stage_start_time = self.now_sec()

    def elapsed(self):
        return self.now_sec() - self.stage_start_time

    # [최적화 2] 반복되는 "정지 + 상태 변경 + 로그 출력"을 하나로 묶은 헬퍼 함수
    def change_stage(self, next_stage, log_msg="", stop_car=True):
        if stop_car:
            self.speed, self.steer = 0, 0
        self.set_stage(next_stage)
        if log_msg:
            self.get_logger().info(log_msg)

    def perc_callback(self, msg):
        # [최적화 3] 파이썬 슬라이싱을 이용해 for문 없이 한 번에 고속 복사
        data_len = min(len(msg.data), 12)
        self.perc_data[:data_len] = msg.data[:data_len]

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        valid = np.isfinite(ranges) & (ranges > LIDAR_MIN_RANGE_M) & (ranges <= LIDAR_MAX_RANGE_M)

        if not np.any(valid):
            self.upper_count = 0
            self.upper_max_x = 9999.0
            return

        valid_ranges = ranges[valid] * 1000.0
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid_angles = angles[valid] - (np.pi / 2)

        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)

        mask = (
            (y > self.UPPER_Y_MIN) &
            (y < self.UPPER_Y_MAX) &
            (np.abs(x) < self.X_RANGE_LIMIT)
        )

        upper_x = x[mask]
        self.upper_count = len(upper_x)
        self.upper_max_x = float(np.max(upper_x)) if self.upper_count > 0 else 9999.0

    def publish_cmd(self):
        stage_msg = Int32()
        stage_msg.data = self.stage
        self.stage_pub.publish(stage_msg)

        cmd_msg = MotionCommand()
        cmd_msg.steering = int(self.steer)
        cmd_msg.left_speed = int(self.speed)
        cmd_msg.right_speed = int(self.speed)
        self.control_pub.publish(cmd_msg)
        self.get_logger().info(
            f'Published control: stage={self.stage}, '
            f'steer={cmd_msg.steering}, speed={cmd_msg.left_speed}, '
            f'roi_points={self.perc_data[0]:.0f}, svm={int(bool(self.perc_data[1]))}',
            throttle_duration_sec=1.0,
        )

    def control_loop(self):
        # 로컬 변수 할당
        point_count = self.perc_data[0]
        is_svm_valid = bool(self.perc_data[1])
        x_diff = self.perc_data[2]
        deg_diff = self.perc_data[3]
        min_rear_dist = self.perc_data[4]
        w0 = self.perc_data[5]
        w1 = self.perc_data[6]
        b  = self.perc_data[7]
        
        elapsed = self.elapsed()

        # [최적화 4] change_stage()를 활용하여 제어 루프를 훨씬 깔끔하게 정리
        if self.stage == 1:
            self.speed, self.steer = 100, 0
            if point_count >= 5:
                self.change_stage(2, 'Spot Found. Stage 2.')

        # elif self.stage == 2:
        #     if elapsed < 1.0:
        #         self.speed, self.steer = 0, 0
        #     elif elapsed < 6.4:
        #         self.speed, self.steer = 50, -7
        #     else:
        #         self.change_stage(3, 'Stage 3.')

        # elif self.stage == 3:
        #     self.speed, self.steer = 0, 0
        #     if elapsed > 2.0:
        #         self.change_stage(4, 'Stage 4: Align.')

        elif self.stage == 4:
            if not is_svm_valid:
                self.speed, self.steer = 0, 0
            else:
                if 89.9 <= abs(deg_diff) <= 90.1:
                    self.change_stage(7, 'Alignment done. Stage 7.')
                else:
                    self.steer = np.clip(-((0.1 * x_diff) + (0.2 * deg_diff)), -7, 7)
                    self.speed = -30
                    if min_rear_dist < 800.0:
                        if abs(x_diff) < 150.0 and abs(deg_diff) < 7.0:
                            self.change_stage(8, 'Parking done. Stage 8.')
                        else:
                            self.change_stage(5, 'Correction. Stage 5.')

        elif self.stage == 5:
            if elapsed < 1.5:
                self.speed = 50
                if is_svm_valid: 
                    self.steer = np.clip(0.02 * x_diff, -7, 7)
            else:
                # [버그 수정] 튜플 할당 에러 방지
                self.change_stage(6, stop_car=True) 

        elif self.stage == 6:
            if elapsed < 1.0:
                self.speed, self.steer = -50, -self.steer
            else:
                self.change_stage(4, stop_car=True)

        elif self.stage == 7:
            self.speed, self.steer = -30, 0
            if self.upper_count >= self.MIN_STOP_POINTS and self.upper_max_x < self.STOP_X_MARGIN:
                self.change_stage(8, 'Parking complete. Wait 5 sec.')

        elif self.stage == 8:
            self.speed, self.steer = 0, 0
            if elapsed > 5.0:
                self.saved_svm_deg = deg_diff if is_svm_valid else 90.0
                
                # 목표 탈출 각도 계산
                offset = -90.0 if self.saved_svm_deg > 0 else 90.0
                self.target_exit_deg = self.saved_svm_deg + offset
                    
                self.change_stage(9, f'Stage 9: Forward exit. (직선 각도: {self.saved_svm_deg:.1f} / 목표 각도: {self.target_exit_deg:.1f})', stop_car=False)

        elif self.stage == 9:
            self.speed, self.steer = 30, 0
            if self.upper_count >= self.MIN_STOP_POINTS and self.upper_max_x > self.EXIT_X_MARGIN:
                self.change_stage(10, 'Stage 10: 목표 각도를 향해 동적 우회전 시작!')

        elif self.stage == 10:
            self.speed = self.EXIT_SPEED
            
            if is_svm_valid:
                # 1. P제어 동적 조향
                norm = np.sqrt(w0**2 + w1**2)
                if norm > 1e-6:
                    line_error_mm = (b / norm) * 1000.0
                    dynamic_steer = self.EXIT_MAX_STEER + (self.EXIT_KP * line_error_mm)
                    self.steer = np.clip(dynamic_steer, 3.0, 11.0)
                else:
                    self.steer = self.EXIT_MAX_STEER
                
                # 2. 직교 판단 (오차 1도 이내)
                angle_error = abs(deg_diff - self.target_exit_deg)
                if angle_error > 90.0:
                    angle_error = 180.0 - angle_error
                
                self.get_logger().info(
                    f'[Stage 10] 조향: {self.steer:.1f} / 현재 각도: {deg_diff:.1f}도 (목표: {self.target_exit_deg:.1f}도, 오차: {angle_error:.1f}도)',
                    throttle_duration_sec=0.2
                )

                if angle_error <= 1.5: 
                    self.change_stage(11, '✅ 1차선과 수직(90도) 달성! 목표 각도 명중. Stage 11 진입.')
            else:
                self.steer = self.EXIT_MAX_STEER
                self.get_logger().info('[Stage 10] SVM 대기 중... 기본 우회전', throttle_duration_sec=0.2)

        elif self.stage == 11:
            self.speed, self.steer = 100, 0
            self.get_logger().info('주차장 탈출 완료. 직진 중.', throttle_duration_sec=2.0)

        self.publish_cmd()

def main(args=None):
    rclpy.init(args=args)
    node = MotionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__': main()
