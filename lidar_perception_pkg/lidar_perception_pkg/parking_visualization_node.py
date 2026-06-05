import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int32

LIDAR_MIN_RANGE_M = 0.3
LIDAR_MAX_RANGE_M = 3.0
PARKING_ROI_X_MIN_MM = 800.0
PARKING_ROI_X_MAX_MM = 2000.0
PARKING_ROI_Y_MIN_MM = 400.0
PARKING_ROI_Y_MAX_MM = 1100.0

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('parking_vis_node')
        
        # Subscriber 연결
        self.scan_sub = self.create_subscription(LaserScan, '/lidar_raw', self.lidar_callback, 10)
        self.perc_sub = self.create_subscription(Float32MultiArray, '/perception_data', self.perc_callback, 10)
        self.stage_sub = self.create_subscription(Int32, '/current_stage', self.stage_callback, 10)
        
        self.timer = self.create_timer(0.1, self.draw_loop)
        
        self.lidar_data = np.array([])
        self.perc_data = [0.0] * 12 
        self.current_stage = 1
        
        self.frame_size = 800
        self.max_distance = 3000
        self.main_map_min_distance = int(LIDAR_MIN_RANGE_M * 1000)
        self.main_map_max_distance = int(LIDAR_MAX_RANGE_M * 1000)
        self.get_logger().info('Visualization Node Started! (Restored Dual Monitor UI)')

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        valid_indices = np.isfinite(ranges) & (ranges > LIDAR_MIN_RANGE_M) & (ranges <= LIDAR_MAX_RANGE_M)
        valid_ranges = ranges[valid_indices] * 1000.0 
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid_angles = angles[valid_indices] - (np.pi / 2) 
        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)
        self.lidar_data = np.vstack((x_coords, y_coords)).T

    def perc_callback(self, msg): 
        if len(msg.data) >= 12: self.perc_data = msg.data
            
    def stage_callback(self, msg): 
        self.current_stage = msg.data

    # ==========================================
    # 화면 렌더링 함수들을 옛날처럼 분리했습니다!
    # ==========================================
    def world_to_main_map_pixel(self, x_mm, y_mm):
        center = self.frame_size // 2
        scale = center / self.main_map_max_distance
        return int(center + x_mm * scale), int(center + y_mm * scale)

    def draw_main_lidar_map(self):
        """Stage 1, 2: 3m radius main map for close-range parking checks."""
        frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
        center = self.frame_size // 2
        
        cv2.line(frame, (center, 0), (center, self.frame_size), (50, 50, 50), 1)
        cv2.line(frame, (0, center), (self.frame_size, center), (50, 50, 50), 1)

        # 0.5m unit circles inside the 3m map.
        for distance in range(500, self.main_map_max_distance + 1, 500):
            radius = int(distance / self.main_map_max_distance * center)
            cv2.circle(frame, (center, center), radius, (80, 80, 80), 1)

        distances = np.linalg.norm(self.lidar_data, axis=1)
        visible_mask = (
            (distances >= self.main_map_min_distance) &
            (distances <= self.main_map_max_distance)
        )

        for x, y in self.lidar_data[visible_mask]:
            ix, iy = self.world_to_main_map_pixel(x, y)
            if 0 <= ix < self.frame_size and 0 <= iy < self.frame_size:
                cv2.circle(frame, (ix, iy), 3, (0, 255, 0), -1)

        cv2.circle(frame, (center, center), 5, (0, 0, 255), -1)
        cv2.imshow('Main Lidar Map (3m)', frame)

    def draw_roi_frame(self):
        """Stage 1: ROI 영역 확대 서브 모니터"""
        roi_frame = np.zeros((400, 400, 3), dtype=np.uint8)
        margin = 35
        usable = 400 - (2 * margin)
        roi_padding_mm = 200.0
        roi_center_x = (PARKING_ROI_X_MIN_MM + PARKING_ROI_X_MAX_MM) * 0.5
        roi_center_y = (PARKING_ROI_Y_MIN_MM + PARKING_ROI_Y_MAX_MM) * 0.5
        roi_view_span = max(
            PARKING_ROI_X_MAX_MM - PARKING_ROI_X_MIN_MM,
            PARKING_ROI_Y_MAX_MM - PARKING_ROI_Y_MIN_MM,
        ) + (2.0 * roi_padding_mm)
        view_x_min = roi_center_x - (roi_view_span * 0.5)
        view_y_min = roi_center_y - (roi_view_span * 0.5)

        def roi_to_pixel(x_mm, y_mm):
            ix = int(margin + ((x_mm - view_x_min) / roi_view_span) * usable)
            iy = int(margin + ((y_mm - view_y_min) / roi_view_span) * usable)
            return ix, iy

        roi_mask = (
            (self.lidar_data[:, 0] >= PARKING_ROI_X_MIN_MM) &
            (self.lidar_data[:, 0] <= PARKING_ROI_X_MAX_MM) &
            (self.lidar_data[:, 1] >= PARKING_ROI_Y_MIN_MM) &
            (self.lidar_data[:, 1] <= PARKING_ROI_Y_MAX_MM)
        )

        view_mask = (
            (self.lidar_data[:, 0] >= view_x_min) &
            (self.lidar_data[:, 0] <= view_x_min + roi_view_span) &
            (self.lidar_data[:, 1] >= view_y_min) &
            (self.lidar_data[:, 1] <= view_y_min + roi_view_span)
        )

        for x, y in self.lidar_data[view_mask]:
            ix, iy = roi_to_pixel(x, y)
            if 0 <= ix < 400 and 0 <= iy < 400:
                cv2.circle(roi_frame, (ix, iy), 3, (70, 70, 70), -1)

        roi_left_top = roi_to_pixel(PARKING_ROI_X_MIN_MM, PARKING_ROI_Y_MIN_MM)
        roi_right_bottom = roi_to_pixel(PARKING_ROI_X_MAX_MM, PARKING_ROI_Y_MAX_MM)
        cv2.rectangle(roi_frame, roi_left_top, roi_right_bottom, (0, 255, 255), 2)

        for x, y in self.lidar_data[roi_mask]:
            ix, iy = roi_to_pixel(x, y)
            if 0 <= ix < 400 and 0 <= iy < 400:
                cv2.circle(roi_frame, (ix, iy), 4, (0, 255, 0), -1)

        roi_distance = np.hypot(roi_center_x, roi_center_y)
        roi_angle = np.degrees(np.arctan2(roi_center_y, roi_center_x))
        cv2.putText(roi_frame, "ROI Monitor", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(roi_frame, f"x:{PARKING_ROI_X_MIN_MM:.0f}-{PARKING_ROI_X_MAX_MM:.0f} y:{PARKING_ROI_Y_MIN_MM:.0f}-{PARKING_ROI_Y_MAX_MM:.0f}mm", (10, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(roi_frame, f"Points:{int(self.perc_data[0])} center:{roi_distance:.0f}mm {roi_angle:.0f}deg", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imshow('ROI Monitor (Zoomed)', roi_frame)

    def draw_dbscan_svm_map(self):
        """Stage 3~7: 3m parking monitor with the same orientation as the main map."""
        frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
        center = self.frame_size // 2
        
        cv2.line(frame, (center, 0), (center, self.frame_size), (50, 50, 50), 1)
        cv2.line(frame, (0, center), (self.frame_size, center), (50, 50, 50), 1)

        distances = np.linalg.norm(self.lidar_data, axis=1)
        mask = (
            (distances >= self.main_map_min_distance) &
            (distances <= self.main_map_max_distance)
        )
        rear_points = self.lidar_data[mask]
        other_points = self.lidar_data[~mask]

        # 1. 관심 영역 밖 점들은 회색 처리
        for x, y in other_points:
            ix, iy = self.world_to_main_map_pixel(x, y)
            if 0 <= ix < self.frame_size and 0 <= iy < self.frame_size:
                cv2.circle(frame, (ix, iy), 2, (100, 100, 100), -1)

        is_svm = bool(self.perc_data[1])
        c1 = np.array([self.perc_data[8], self.perc_data[9]])
        c2 = np.array([self.perc_data[10], self.perc_data[11]])

        # 2. DBSCAN 중심점 기준으로 차량 색상 칠하기
        for x, y in rear_points:
            ix, iy = self.world_to_main_map_pixel(x, y)
            
            if 0 <= ix < self.frame_size and 0 <= iy < self.frame_size:
                if is_svm:
                    pt = np.array([x, y])
                    dist1 = np.linalg.norm(pt - c1)
                    dist2 = np.linalg.norm(pt - c2)
                    
                    if dist1 < 1000 and dist1 < dist2:
                        cv2.circle(frame, (ix, iy), 3, (255, 255, 0), -1) # 1번차: BGR(255,255,0) 옥색/하늘색
                    elif dist2 < 1000 and dist2 < dist1:
                        cv2.circle(frame, (ix, iy), 3, (0, 255, 255), -1) # 2번차: BGR(0,255,255) 노란색
                    else:
                        cv2.circle(frame, (ix, iy), 2, (0, 255, 0), -1)   # 소속 안됨: 초록색
                else:
                    cv2.circle(frame, (ix, iy), 2, (0, 255, 0), -1)

        # 3. SVM 기준선 그리기
        if is_svm:
            x_diff, deg_diff = self.perc_data[2], self.perc_data[3]
            w0, w1, b = self.perc_data[5], self.perc_data[6], self.perc_data[7]

            pts_m = []
            map_range_m = self.main_map_max_distance / 1000.0
            if abs(w1) > abs(w0):
                for x_m in [-map_range_m, map_range_m]: pts_m.append((x_m, -(w0 * x_m + b) / w1))
            else:
                for y_m in [-map_range_m, map_range_m]: pts_m.append((-(w1 * y_m + b) / w0, y_m))
            
            ix1, iy1 = self.world_to_main_map_pixel(pts_m[0][0] * 1000, pts_m[0][1] * 1000)
            ix2, iy2 = self.world_to_main_map_pixel(pts_m[1][0] * 1000, pts_m[1][1] * 1000)
            
            cv2.line(frame, (ix1, iy1), (ix2, iy2), (255, 255, 255), 2)
            cv2.putText(frame, f"x_diff: {x_diff:.1f}mm, deg_diff: {deg_diff:.1f}deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 내 차 위치
        cv2.circle(frame, (center, center), 5, (0, 0, 255), -1)
        cv2.imshow('SVM Parking Monitor', frame)

    # ==========================================
    # 메인 루프: 스테이지에 맞춰 창 열고 닫기
    # ==========================================
    def draw_loop(self):
        if len(self.lidar_data) == 0: return

        if self.current_stage == 1:
            self.draw_main_lidar_map()
            self.draw_roi_frame()
            
        elif self.current_stage == 2:
            try: cv2.destroyWindow('ROI Monitor (Zoomed)')
            except: pass
            self.draw_main_lidar_map()
            
        elif self.current_stage >= 3:
            try: cv2.destroyWindow('Main Lidar Map (3m)')
            except: pass
            self.draw_dbscan_svm_map()

        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__': main()
