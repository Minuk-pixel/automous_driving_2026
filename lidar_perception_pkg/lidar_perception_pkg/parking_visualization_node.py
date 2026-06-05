import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int32

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('parking_vis_node')
        
        # Subscriber 연결
        self.scan_sub = self.create_subscription(LaserScan, 'lidar_raw', self.lidar_callback, 10)
        self.perc_sub = self.create_subscription(Float32MultiArray, '/perception_data', self.perc_callback, 10)
        self.stage_sub = self.create_subscription(Int32, '/current_stage', self.stage_callback, 10)
        
        self.timer = self.create_timer(0.1, self.draw_loop)
        
        self.lidar_data = np.array([])
        self.perc_data = [0.0] * 23
        self.current_stage = 1
        self.last_stage_msg_time = None
        self.roi_switch_threshold = 8
        self.roi_detect_count = 0
        self.show_stage2_monitor = False
        self.stage2_gap_angle_count = 0
        self.force_svm_monitor = False
        
        self.frame_size = 800
        self.max_distance = 4000
        self.rotate_view_180 = True
        self.mirror_svm_view = True
        self.roi_side_x_min = -2000.0
        self.roi_side_x_max = -800.0
        self.roi_rear_y_min = -500.0
        self.roi_rear_y_max = 0.0
        self.stage2_x_min = 0.0
        self.stage2_x_max = 500.0
        self.stage2_y_min = 0.0
        self.stage2_y_max = 2000.0
        self.object_x_min = -2200.0
        self.object_x_max = 2200.0
        self.object_y_min = -2500.0
        self.object_y_max = 2000.0
        self.mirror_stage2_view = False
        self.mirror_stage2_y_view = True
        self.mirror_roi_view = True
        self.mirror_roi_y_view = True
        self.mirror_svm_y_view = True
        self.stage2_gap_line_target_deg = 50.0
        self.stage2_gap_line_deg_tol = 5.0
        self.stage2_gap_angle_frames = 3
        self.get_logger().info('Visualization Node Started! (Front-view display enabled)')

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        valid_indices = np.isfinite(ranges) & (ranges > 0.2) & (ranges <= 10.0)
        valid_ranges = ranges[valid_indices] * 1000.0 
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid_angles = angles[valid_indices] - (np.pi / 2) 
        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)
        self.lidar_data = np.vstack((x_coords, y_coords)).T

    def perc_callback(self, msg): 
        if len(msg.data) >= 12:
            self.perc_data = msg.data
            if self.perc_data[0] >= self.roi_switch_threshold:
                self.roi_detect_count += 1
                self.show_stage2_monitor = True
            else:
                self.roi_detect_count = 0
            
    def stage_callback(self, msg): 
        self.current_stage = msg.data
        self.last_stage_msg_time = self.get_clock().now().nanoseconds / 1e9
        if self.current_stage >= 3:
            self.force_svm_monitor = True
        else:
            self.force_svm_monitor = False

    def has_stage_topic_signal(self):
        if self.last_stage_msg_time is None:
            return False

        now = self.get_clock().now().nanoseconds / 1e9
        return (now - self.last_stage_msg_time) <= 1.0

    def display_stage_text(self):
        if self.has_stage_topic_signal():
            return f"Stage {self.current_stage}"
        if self.is_visual_straight_reverse_ready():
            return "Stage 5 visual"
        if self.force_svm_monitor:
            return "Stage 3 visual"
        if self.show_stage2_monitor:
            return "Stage 2 visual"
        return "Stage 1 visual"

    def is_visual_straight_reverse_ready(self):
        return (
            len(self.perc_data) >= 19 and
            bool(self.perc_data[1]) and
            abs(self.perc_data[3]) <= 1.0
        )

    def draw_stage_overlay(self, frame, origin=(10, 28), color=(0, 255, 0)):
        cv2.putText(
            frame,
            self.display_stage_text(),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    def gap_line_angle_deg(self):
        if len(self.perc_data) < 19:
            return None

        gap_center_x = self.perc_data[16]
        gap_center_y = self.perc_data[17]
        screen_x = -gap_center_x
        screen_y = -gap_center_y
        if abs(screen_x) < 1e-6 and abs(screen_y) < 1e-6:
            return None

        return float(np.degrees(np.arctan2(screen_y, screen_x)))

    def update_stage2_svm_switch(self):
        if len(self.perc_data) < 19 or self.force_svm_monitor:
            return

        two_cars_detected = bool(self.perc_data[15])
        gap_center_x = self.perc_data[16]
        gap_center_y = self.perc_data[17]
        gap_line_deg = self.gap_line_angle_deg()
        angle_ready = (
            two_cars_detected and
            gap_line_deg is not None and
            (-gap_center_x) > 0.0 and
            (-gap_center_y) > 0.0 and
            abs(gap_line_deg - self.stage2_gap_line_target_deg) <= self.stage2_gap_line_deg_tol
        )

        if angle_ready:
            self.stage2_gap_angle_count += 1
            if self.stage2_gap_angle_count >= self.stage2_gap_angle_frames:
                self.force_svm_monitor = True
        else:
            self.stage2_gap_angle_count = 0

    def world_to_pixel(self, x, y, center, mirror_x=False, mirror_y=False):
        if self.rotate_view_180:
            x, y = -x, -y
        if mirror_x:
            x = -x
        if mirror_y:
            y = -y

        ix = int(center + x / self.max_distance * center)
        iy = int(center - y / self.max_distance * center)
        return ix, iy

    def is_in_frame(self, ix, iy, frame_size=None):
        size = frame_size or self.frame_size
        return 0 <= ix < size and 0 <= iy < size

    def draw_axes(self, frame, center, color=(50, 50, 50)):
        cv2.line(frame, (center, 0), (center, frame.shape[0]), color, 1)
        cv2.line(frame, (0, center), (frame.shape[1], center), color, 1)

    def draw_world_point(self, frame, x, y, center, color, radius=2, thickness=-1, mirror_x=False, mirror_y=False):
        ix, iy = self.world_to_pixel(x, y, center, mirror_x=mirror_x, mirror_y=mirror_y)
        if self.is_in_frame(ix, iy, frame.shape[0]):
            cv2.circle(frame, (ix, iy), radius, color, thickness)
            return ix, iy
        return None

    def draw_world_points(self, frame, points, center, color, radius=2, mirror_x=False, mirror_y=False):
        for x, y in points:
            self.draw_world_point(
                frame,
                x,
                y,
                center,
                color,
                radius=radius,
                mirror_x=mirror_x,
                mirror_y=mirror_y,
            )

    def stage2_pixel(self, x, y, center):
        return self.world_to_pixel(
            x,
            y,
            center,
            mirror_x=self.mirror_stage2_view,
            mirror_y=self.mirror_stage2_y_view,
        )

    def svm_pixel(self, x, y, center):
        return self.world_to_pixel(
            x,
            y,
            center,
            mirror_x=self.mirror_svm_view,
            mirror_y=self.mirror_svm_y_view,
        )

    # ==========================================
    # 화면 렌더링 함수
    # ==========================================
    def draw_stage2_object_map(self):
        """Stage 2: 좌회전 중 +X 방향에서 나타나는 물체를 동심원 거리로 확인"""
        frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
        center = self.frame_size // 2
        self.draw_axes(frame, center)

        # 1m 단위 동심원
        for distance in range(1000, self.max_distance + 1, 1000):
            radius = int(distance / self.max_distance * center)
            cv2.circle(frame, (center, center), radius, (80, 80, 80), 1)
            cv2.putText(
                frame,
                f"{distance // 1000}m",
                (center + radius + 4, center - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (120, 120, 120),
                1,
            )

        self.draw_world_points(
            frame,
            self.lidar_data,
            center,
            (0, 220, 80),
            radius=2,
            mirror_x=self.mirror_stage2_view,
            mirror_y=self.mirror_stage2_y_view,
        )

        if len(self.perc_data) >= 19:
            centroid_pixels = []
            for cx, cy, color in (
                (self.perc_data[8], self.perc_data[9], (0, 255, 255)),
                (self.perc_data[10], self.perc_data[11], (255, 255, 0)),
            ):
                if cx != 0.0 or cy != 0.0:
                    pixel = self.draw_world_point(
                        frame,
                        cx,
                        cy,
                        center,
                        color,
                        radius=8,
                        thickness=2,
                        mirror_x=self.mirror_stage2_view,
                        mirror_y=self.mirror_stage2_y_view,
                    )
                    if pixel is not None:
                        ix, iy = pixel
                        centroid_pixels.append((ix, iy))

            if len(centroid_pixels) == 2:
                cv2.line(frame, centroid_pixels[0], centroid_pixels[1], (255, 255, 255), 2)
                gap_ix, gap_iy = self.stage2_pixel(
                    self.perc_data[16],
                    self.perc_data[17],
                    center,
                )
                if 0 <= gap_ix < self.frame_size and 0 <= gap_iy < self.frame_size:
                    cv2.line(frame, (center, center), (gap_ix, gap_iy), (255, 255, 255), 2)
                    cv2.circle(frame, (gap_ix, gap_iy), 5, (255, 255, 255), -1)
                    gap_deg = self.gap_line_angle_deg()
                    if gap_deg is not None:
                        cv2.putText(frame, f"Gap angle: {gap_deg:.1f}deg", (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                c1_x, c1_y = self.perc_data[8], self.perc_data[9]
                c2_x, c2_y = self.perc_data[10], self.perc_data[11]
                dx = -(c2_x - c1_x)
                dy = c2_y - c1_y
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    line_deg = float(np.degrees(np.arctan2(dy, dx)))
                    while line_deg >= 90.0:
                        line_deg -= 180.0
                    while line_deg < -90.0:
                        line_deg += 180.0
                    cv2.putText(frame, f"Centerline: {line_deg:.1f}deg", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.circle(frame, (center, center), 5, (0, 0, 255), -1)
        self.draw_stage_overlay(frame, (10, 30), (0, 255, 0))
        cv2.putText(frame, "Object Monitor | 4m range", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Two cars: {int(self.perc_data[15]) if len(self.perc_data) >= 19 else 0}", (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow('Stage 2 Object Monitor', frame)

    def draw_roi_frame(self):
        """Stage 1: ROI 영역 확대 서브 모니터"""
        roi_frame = np.zeros((500, 500, 3), dtype=np.uint8)
        width, height = 500, 500
        margin = 50
        x_min, x_max = -4000.0, 4000.0
        y_min, y_max = -1500.0, 1500.0

        def to_pixel(x, y):
            draw_x = -x if self.mirror_roi_view else x
            draw_y = -y if self.mirror_roi_y_view else y
            px = int(margin + (draw_x - x_min) / (x_max - x_min) * (width - 2 * margin))
            py = int(height - margin - (draw_y - y_min) / (y_max - y_min) * (height - 2 * margin))
            return px, py

        cv2.line(roi_frame, to_pixel(0, y_min), to_pixel(0, y_max), (70, 70, 70), 1)
        cv2.line(roi_frame, to_pixel(x_min, 0), to_pixel(x_max, 0), (70, 70, 70), 1)

        for x, y in self.lidar_data:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                px, py = to_pixel(x, y)
                cv2.circle(roi_frame, (px, py), 3, (0, 220, 80), -1)

        roi_boxes = (
            (self.roi_side_x_min, self.roi_rear_y_max, self.roi_side_x_max, self.roi_rear_y_min),
        )

        for left_x, top_y, right_x, bottom_y in roi_boxes:
            cv2.rectangle(
                roi_frame,
                to_pixel(left_x, top_y),
                to_pixel(right_x, bottom_y),
                (0, 255, 255),
                2,
            )

        self.draw_stage_overlay(roi_frame, (10, 25), (0, 255, 255))
        cv2.putText(roi_frame, "Rear Side ROI Monitor", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(roi_frame, "object range: screen right X 0.8-1.5m", (10, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        cv2.putText(roi_frame, f"Points: {int(self.perc_data[0])}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(self.perc_data) >= 19:
            cv2.putText(roi_frame, f"Right car: {int(self.perc_data[12])}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.putText(roi_frame, f"Two cars: {int(self.perc_data[15])}", (10, 157), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.putText(roi_frame, f"Gap: {self.perc_data[18]:.0f}mm", (10, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.putText(roi_frame, "front +Y", (width // 2 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(roi_frame, "right +X", (width - 105, height // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(roi_frame, "LIDAR", to_pixel(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.circle(roi_frame, to_pixel(0, 0), 5, (0, 0, 255), -1)
        cv2.imshow('ROI Monitor (Zoomed)', roi_frame)

    def draw_dbscan_svm_map(self):
        """Stage 3~5: SVM 주차 전용 모니터"""
        frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
        center = self.frame_size // 2
        self.draw_axes(frame, center)

        mask = (
            (self.lidar_data[:, 0] >= self.object_x_min) &
            (self.lidar_data[:, 0] <= self.object_x_max) &
            (self.lidar_data[:, 1] >= self.object_y_min) &
            (self.lidar_data[:, 1] <= self.object_y_max)
        )
        rear_points = self.lidar_data[mask]
        other_points = self.lidar_data[~mask]

        # 1. 관심 영역 밖 점들은 회색 처리
        self.draw_world_points(
            frame,
            other_points,
            center,
            (100, 100, 100),
            radius=2,
            mirror_x=self.mirror_svm_view,
            mirror_y=self.mirror_svm_y_view,
        )

        is_svm = bool(self.perc_data[1])
        c1 = np.array([self.perc_data[8], self.perc_data[9]])
        c2 = np.array([self.perc_data[10], self.perc_data[11]])
        gap_center = np.array([self.perc_data[16], self.perc_data[17]])

        # 2. DBSCAN 중심점 기준으로 차량 색상 칠하기
        for x, y in rear_points:
            ix, iy = self.svm_pixel(x, y, center)
            
            if 0 <= ix < self.frame_size and 0 <= iy < self.frame_size:
                if is_svm:
                    pt = np.array([x, y])
                    dist1 = np.linalg.norm(pt - c1)
                    dist2 = np.linalg.norm(pt - c2)
                    
                    if dist1 < 1000 and dist1 < dist2:
                        cv2.circle(frame, (ix, iy), 4, (0, 255, 255), -1)
                    elif dist2 < 1000 and dist2 < dist1:
                        cv2.circle(frame, (ix, iy), 4, (255, 255, 0), -1)
                    else:
                        cv2.circle(frame, (ix, iy), 2, (0, 255, 0), -1)   # 소속 안됨: 초록색
                else:
                    cv2.circle(frame, (ix, iy), 2, (0, 255, 0), -1)

        # 3. SVM 기준선 그리기
        if is_svm:
            x_diff, deg_diff = self.perc_data[2], self.perc_data[3]
            w0, w1, b = self.perc_data[5], self.perc_data[6], self.perc_data[7]

            pts_m = []
            if abs(w1) > abs(w0):
                for x_m in [-12.0, 12.0]: pts_m.append((x_m, -(w0 * x_m + b) / w1))
            else:
                for y_m in [-12.0, 12.0]: pts_m.append((-(w1 * y_m + b) / w0, y_m))
            
            ix1, iy1 = self.svm_pixel(
                pts_m[0][0] * 1000,
                pts_m[0][1] * 1000,
                center,
            )
            ix2, iy2 = self.svm_pixel(
                pts_m[1][0] * 1000,
                pts_m[1][1] * 1000,
                center,
            )
            
            cv2.line(frame, (ix1, iy1), (ix2, iy2), (255, 255, 255), 2)
            for centroid, color, label in (
                (c1, (0, 255, 255), "car1"),
                (c2, (255, 255, 0), "car2"),
                (gap_center, (255, 255, 255), "gap"),
            ):
                if centroid[0] != 0.0 or centroid[1] != 0.0:
                    cx, cy = self.svm_pixel(
                        centroid[0],
                        centroid[1],
                        center,
                    )
                    if 0 <= cx < self.frame_size and 0 <= cy < self.frame_size:
                        cv2.circle(frame, (cx, cy), 8, color, 2)
                        cv2.circle(frame, (cx, cy), 3, color, -1)
                        cv2.putText(frame, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            gap_ix, gap_iy = self.svm_pixel(
                gap_center[0],
                gap_center[1],
                center,
            )
            if 0 <= gap_ix < self.frame_size and 0 <= gap_iy < self.frame_size:
                cv2.line(frame, (center, center), (gap_ix, gap_iy), (180, 180, 180), 1)
            cv2.putText(frame, f"x_diff: {x_diff:.1f}mm, deg_diff: {deg_diff:.1f}deg", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, "Straight trigger: -1.0 <= deg_diff <= 1.0", (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            if len(self.perc_data) >= 20:
                cv2.putText(frame, f"side pts: {self.perc_data[19]:.0f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            if len(self.perc_data) >= 23:
                cv2.putText(frame, f"90/270 pts: {self.perc_data[20]:.0f}/{self.perc_data[21]:.0f}", (10, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        # 내 차 위치
        cv2.circle(frame, (center, center), 5, (0, 0, 255), -1)
        if self.has_stage_topic_signal():
            mode_text = {
                3: "SVM Parking Monitor | reverse ready",
                4: "SVM Parking Monitor | curved reverse",
                5: "SVM Parking Monitor | straight reverse",
            }.get(self.current_stage, "SVM Parking Monitor")
        elif self.is_visual_straight_reverse_ready():
            mode_text = "SVM Parking Monitor | straight reverse visual"
        else:
            mode_text = "SVM Parking Monitor | reverse ready visual"
        self.draw_stage_overlay(frame, (10, 30), (0, 255, 0))
        cv2.putText(frame, mode_text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow('SVM Parking Monitor', frame)

    # ==========================================
    # 메인 루프: ROI, Stage 2 물체 확인, Stage 3~5 SVM 주차 확인을 분리
    # ==========================================
    def draw_loop(self):
        if len(self.lidar_data) == 0: return

        self.draw_roi_frame()
        self.update_stage2_svm_switch()

        try: cv2.destroyWindow('Main Lidar Map (12m)')
        except: pass

        if self.current_stage >= 3 or self.force_svm_monitor:
            try: cv2.destroyWindow('Stage 2 Object Monitor')
            except: pass
            self.draw_dbscan_svm_map()
        elif self.show_stage2_monitor or self.current_stage == 2:
            try: cv2.destroyWindow('SVM Parking Monitor')
            except: pass
            self.draw_stage2_object_map()
        else:
            try: cv2.destroyWindow('SVM Parking Monitor')
            except: pass
            try: cv2.destroyWindow('Stage 2 Object Monitor')
            except: pass

        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()