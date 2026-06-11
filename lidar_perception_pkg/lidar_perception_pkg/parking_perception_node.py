import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
from sklearn.cluster import DBSCAN # [변경됨] K-Means -> DBSCAN
from sklearn.svm import SVC


CLUSTER_HOLD_SEC = 1.0
CLUSTER_HOLD_INDICES = (
    1, 2, 3,       # SVM valid, x_diff, deg_diff
    5, 6, 7,       # SVM coefficients
    8, 9, 10, 11,  # car centroids
    12, 13, 14,    # right car flag + centroid
    15, 16, 17, 18, 19,  # two-car flag, gap, side-pair count
)


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('parking_perception_node')
        
        self.scan_sub = self.create_subscription(LaserScan, 'lidar_raw', self.lidar_callback, 10)
        self.stage_sub = self.create_subscription(Int32, '/current_stage', self.stage_callback, 10)
        self.perception_pub = self.create_publisher(Float32MultiArray, '/perception_data', 10)
        
        # [변경됨] DBSCAN 설정: 반경 550mm 내에 점이 3개 이상 있으면 하나의 군집으로 인정
        self.dbscan = DBSCAN(eps=800, min_samples=2)
        self.svm = SVC(kernel='linear')
        
        # Stage 2 tracking: lock the first right-side car, keep following it
        # as it moves down, then lock a newly appearing right-side car.
        self.primary_car_centroid = None
        self.secondary_car_centroid = None
        self.current_stage = 1
        self.last_cluster_data = None
        self.last_cluster_time = None

        # 실차 튜닝용 ROI/판단 파라미터
        # ROI 진입 판단 범위: 현재 시각화 기준 화면 오른쪽/전방 0.5m 박스
        self.right_x_min = -500.0
        self.right_x_max = 0.0
        self.right_y_min = -500.0
        self.right_y_max = 0.0
        self.car_candidate_max_dist = 4000.0
        self.tracking_match_max_dist = 900.0
        self.min_gap_width = 600.0
        self.wall_y_max = -450.0
        self.wall_min_x_span = 550.0
        self.wall_min_count = 8
        self.first_car_x_min = -2000.0
        self.first_car_x_max = -500.0
        self.first_car_y_min = -700.0
        self.first_car_y_max = 1000.0
        self.first_car_target_x = -1000.0
        self.first_car_target_y = 0.0
        self.primary_y_for_secondary_search = -300.0
        self.second_car_screen_right_min = -2000.0
        self.second_car_screen_right_max = -500.0
        self.second_car_y_min = -700.0
        self.second_car_y_max = 1200.0
        self.second_car_y_offset = 250.0
        self.side_pair_min_abs_x = 150.0
        self.side_pair_max_abs_x = 1800.0
        self.side_pair_y_min = -2500.0
        self.side_pair_y_max = 500.0
        self.side_pair_max_y_gap = 1200.0
        self.side_pair_target_y = -900.0
        self.rear_side_angle_window_deg = 5.0
        self.rear_side_stop_max_range = 2.5
        self.rear_side_stop_min_range = 0.2

        # ROI: stage 1에서 stage 2로 넘어가기 위한 짧은 감지 박스
        self.roi_side_x_min = -2000.0
        self.roi_side_x_max = -800.0
        self.roi_rear_y_min = -500.0
        self.roi_rear_y_max = 0.0
        
        self.get_logger().info('Perception Node Started: Running DBSCAN & Tracking...')

    def stage_callback(self, msg):
        if msg.data != self.current_stage:
            self.clear_cluster_hold()
        self.current_stage = msg.data

    def clear_cluster_hold(self):
        self.last_cluster_data = None
        self.last_cluster_time = None

    def cluster_output_is_valid(self, data):
        return (
            bool(data[1])
            # or bool(data[12])
            or bool(data[15])
            or data[19] > 0.0
        )

    def cluster_hold_is_available(self):
        if self.last_cluster_data is None or self.last_cluster_time is None:
            return False

        elapsed = self.get_clock().now() - self.last_cluster_time
        return elapsed.nanoseconds * 1e-9 <= CLUSTER_HOLD_SEC

    def update_or_apply_cluster_hold(self, data):
        if self.cluster_output_is_valid(data):
            self.last_cluster_data = list(data)
            self.last_cluster_time = self.get_clock().now()
            return data

        if not self.cluster_hold_is_available():
            return data

        held_data = list(data)
        for idx in CLUSTER_HOLD_INDICES:
            held_data[idx] = self.last_cluster_data[idx]
        return held_data

    def make_clusters(self, points, labels):
        clusters = []
        unique_labels = set(labels) - {-1}

        for lbl in unique_labels:
            pts = points[labels == lbl]
            if len(pts) == 0:
                continue

            centroid = np.mean(pts, axis=0)
            min_xy = np.min(pts, axis=0)
            max_xy = np.max(pts, axis=0)
            clusters.append({
                'centroid': centroid,
                'points': pts,
                'bbox_min': min_xy,
                'bbox_max': max_xy,
                'count': len(pts),
            })

        return clusters

    def find_right_car_cluster(self, clusters):
        right_candidates = []

        for cluster in clusters:
            cx, cy = cluster['centroid']
            if (
                self.right_x_min <= cx <= self.right_x_max and
                self.right_y_min <= cy <= self.right_y_max
            ):
                right_candidates.append(cluster)

        if not right_candidates:
            return None

        return max(right_candidates, key=lambda c: c['count'])

    def is_wall_like_cluster(self, cluster):
        cx, cy = cluster['centroid']
        x_span = cluster['bbox_max'][0] - cluster['bbox_min'][0]

        return (
            cy <= self.wall_y_max and
            x_span >= self.wall_min_x_span and
            cluster['count'] >= self.wall_min_count
        )

    def make_car_candidates(self, clusters):
        candidates = []
        for cluster in clusters:
            cx, cy = cluster['centroid']
            dist = np.hypot(cx, cy)
            if dist <= self.car_candidate_max_dist and not self.is_wall_like_cluster(cluster):
                candidates.append(cluster)

        return candidates

    def match_tracked_cluster(self, clusters, target, exclude_cluster=None):
        if target is None:
            return None

        best_cluster = None
        best_dist = float('inf')
        for cluster in clusters:
            if cluster is exclude_cluster:
                continue

            dist = np.linalg.norm(cluster['centroid'] - target)
            if dist < best_dist:
                best_cluster = cluster
                best_dist = dist

        if best_cluster is None or best_dist > self.tracking_match_max_dist:
            return None

        return best_cluster

    def select_primary_car_cluster(self, clusters, candidates):
        tracked = self.match_tracked_cluster(clusters, self.primary_car_centroid)
        if tracked is not None:
            self.primary_car_centroid = tracked['centroid']
            return tracked

        first_car_candidates = []
        for cluster in candidates:
            cx, cy = cluster['centroid']
            if (
                self.first_car_x_min <= cx <= self.first_car_x_max and
                self.first_car_y_min <= cy <= self.first_car_y_max
            ):
                first_car_candidates.append(cluster)

        if not first_car_candidates:
            self.primary_car_centroid = None
            self.secondary_car_centroid = None
            return None

        # The first parked car is the first compact object on the screen-right
        # side. In raw coordinates, more negative x appears farther right.
        selected = min(
            first_car_candidates,
            key=lambda c: (
                c['centroid'][0],
                np.linalg.norm(c['centroid'] - np.array([self.first_car_target_x, self.first_car_target_y])),
                -c['count'],
            ),
        )
        self.primary_car_centroid = selected['centroid']
        self.secondary_car_centroid = None
        return selected

    def select_secondary_car_cluster(self, clusters, candidates, primary_cluster):
        primary = primary_cluster['centroid']
        if primary[1] > self.primary_y_for_secondary_search:
            self.secondary_car_centroid = None
            return None

        secondary_candidates = []

        for cluster in candidates:
            if cluster is primary_cluster:
                continue

            centroid = cluster['centroid']
            gap = np.linalg.norm(centroid - primary)
            appears_above_primary = centroid[1] >= primary[1] + self.second_car_y_offset
            appears_on_screen_right = (
                self.second_car_screen_right_min <= centroid[0] <= self.second_car_screen_right_max
            )
            appears_in_search_band = self.second_car_y_min <= centroid[1] <= self.second_car_y_max
            if (
                appears_above_primary and
                appears_on_screen_right and
                appears_in_search_band and
                gap >= self.min_gap_width
            ):
                secondary_candidates.append(cluster)

        tracked = self.match_tracked_cluster(clusters, self.secondary_car_centroid, exclude_cluster=primary_cluster)
        if tracked is not None:
            self.secondary_car_centroid = tracked['centroid']
            return tracked

        if not secondary_candidates:
            self.secondary_car_centroid = None
            return None

        selected = min(
            secondary_candidates,
            key=lambda c: (
                c['centroid'][0],
                np.linalg.norm(c['centroid'] - np.array([self.first_car_target_x, self.first_car_target_y])),
                -c['count'],
            ),
        )
        self.secondary_car_centroid = selected['centroid']
        return selected

    def select_side_parking_pair(self, candidates):
        left_candidates = []
        right_candidates = []

        for cluster in candidates:
            cx, cy = cluster['centroid']
            in_rear_band = self.side_pair_y_min <= cy <= self.side_pair_y_max
            in_side_band = self.side_pair_min_abs_x <= abs(cx) <= self.side_pair_max_abs_x
            if not in_rear_band or not in_side_band:
                continue

            if cx < 0:
                left_candidates.append(cluster)
            else:
                right_candidates.append(cluster)

        best_pair = None
        best_score = float('inf')
        for left in left_candidates:
            for right in right_candidates:
                left_centroid = left['centroid']
                right_centroid = right['centroid']
                x_gap = abs(right_centroid[0] - left_centroid[0])
                y_gap = abs(right_centroid[1] - left_centroid[1])
                if x_gap < self.min_gap_width or y_gap > self.side_pair_max_y_gap:
                    continue

                symmetry_error = abs(abs(left_centroid[0]) - abs(right_centroid[0]))
                mean_y = (left_centroid[1] + right_centroid[1]) / 2.0
                y_error = abs(mean_y - self.side_pair_target_y)
                count_bonus = left['count'] + right['count']
                score = symmetry_error + y_gap + (0.25 * y_error) - (8.0 * count_bonus)
                if score < best_score:
                    best_score = score
                    best_pair = (left, right)

        return best_pair

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        valid_indices = np.isfinite(ranges) & (ranges > 0.2) & (ranges <= 10.0)
        valid_ranges = ranges[valid_indices] * 1000.0 
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        angle_degs = np.degrees(angles) % 360.0
        stop_range_mask = (
            np.isfinite(ranges) &
            (ranges > self.rear_side_stop_min_range) &
            (ranges <= self.rear_side_stop_max_range)
        )
        angle_90_diff = np.abs(((angle_degs - 90.0 + 180.0) % 360.0) - 180.0)
        angle_270_diff = np.abs(((angle_degs - 270.0 + 180.0) % 360.0) - 180.0)
        rear_side_90_count = float(np.sum(
            stop_range_mask & (angle_90_diff <= self.rear_side_angle_window_deg)
        ))
        rear_side_270_count = float(np.sum(
            stop_range_mask & (angle_270_diff <= self.rear_side_angle_window_deg)
        ))
        rear_side_clear = float(
            rear_side_90_count == 0.0 or rear_side_270_count == 0.0
        )
        valid_angles = angles[valid_indices] - (np.pi / 2) 
        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)
        lidar_data = np.vstack((x_coords, y_coords)).T

        point_count = 0.0
        is_svm_valid = 0.0
        x_diff, deg_diff, min_rear_dist = 0.0, 0.0, 9999.0
        w0, w1, b = 0.0, 0.0, 0.0
        c1_x, c1_y, c2_x, c2_y = 0.0, 0.0, 0.0, 0.0 # [추가] 시각화 노드에 보낼 중심점
        right_car_detected = 0.0
        right_cx, right_cy = 0.0, 0.0
        two_cars_detected = 0.0
        gap_center_x, gap_center_y, gap_width = 0.0, 0.0, 0.0
        side_pair_point_count = 0.0

        if len(lidar_data) == 0: return

        # ROI & 충돌 방지 계산
        roi_mask = (
            (lidar_data[:, 0] >= self.roi_side_x_min) &
            (lidar_data[:, 0] <= self.roi_side_x_max) &
            (lidar_data[:, 1] >= self.roi_rear_y_min) &
            (lidar_data[:, 1] <= self.roi_rear_y_max)
        )
        point_count = float(np.sum(roi_mask))

        close_mask = (lidar_data[:, 1] > 0) & (lidar_data[:, 1] < 1500) & (np.abs(lidar_data[:, 0]) < 800)
        close_pts = lidar_data[close_mask]
        if len(close_pts) > 0:
            dists = np.sqrt(close_pts[:, 0]**2 + close_pts[:, 1]**2)
            min_rear_dist = float(np.min(dists))

        # [핵심 변경] DBSCAN 및 Tracking 로직
        dists_from_lidar = np.hypot(lidar_data[:, 0], lidar_data[:, 1])
        mask = dists_from_lidar <= self.car_candidate_max_dist
        rear_points = lidar_data[mask]

        if len(rear_points) > 10:
            labels = self.dbscan.fit_predict(rear_points)
            clusters = self.make_clusters(rear_points, labels)

            candidates = self.make_car_candidates(clusters)

            right_cluster = self.find_right_car_cluster(candidates)
            if right_cluster is not None:
                right_car_detected = 1.0
                right_cx, right_cy = right_cluster['centroid']

            side_pair = None
            if self.current_stage >= 4:
                side_pair = self.select_side_parking_pair(candidates)

            if side_pair is not None:
                primary_cluster, secondary_cluster = side_pair
                self.primary_car_centroid = primary_cluster['centroid']
                self.secondary_car_centroid = secondary_cluster['centroid']
                c1_x, c1_y = primary_cluster['centroid']
            elif self.current_stage >= 5:
                primary_cluster = None
                secondary_cluster = None
                self.primary_car_centroid = None
                self.secondary_car_centroid = None
            else:
                primary_cluster = self.select_primary_car_cluster(clusters, candidates)
                if primary_cluster is not None:
                    c1_x, c1_y = primary_cluster['centroid']

                secondary_cluster = None
                if primary_cluster is not None:
                    secondary_cluster = self.select_secondary_car_cluster(clusters, candidates, primary_cluster)

            if primary_cluster is not None and secondary_cluster is not None:
                car_pts = [primary_cluster['points'], secondary_cluster['points']]
                side_pair_point_count = float(
                    primary_cluster['count'] + secondary_cluster['count']
                )
                c2_x, c2_y = secondary_cluster['centroid']
                gap_center = (np.array(primary_cluster['centroid']) + np.array(secondary_cluster['centroid'])) / 2.0
                gap_center_x, gap_center_y = float(gap_center[0]), float(gap_center[1])
                gap_width = float(np.linalg.norm(np.array(primary_cluster['centroid']) - np.array(secondary_cluster['centroid'])))
                if gap_width >= self.min_gap_width:
                    two_cars_detected = 1.0

                # SVM용 라벨 만들기 (1번차: 0, 2번차: 1)
                svm_pts = np.vstack((car_pts[0], car_pts[1]))
                svm_labels = np.array([0]*len(car_pts[0]) + [1]*len(car_pts[1]))

                scaled_points = svm_pts / 1000.0
                self.svm.fit(scaled_points, svm_labels)
                w = self.svm.coef_[0]
                b_val = self.svm.intercept_[0]
                w0, w1, b = float(w[0]), float(w[1]), float(b_val)

                if w[0] != 0:
                    x_diff = float((-(w[1] * 2.5 + b_val) / w[0]) * 1000.0)
                    deg_diff = float(np.degrees(np.arctan(-w[1] / w[0])))
                    is_svm_valid = 1.0

        # 기존 12개 값은 유지하고, 뒤에 주차 stage 판단용 값을 추가합니다.
        out_msg = Float32MultiArray()
        out_data = [
            point_count, is_svm_valid, x_diff, deg_diff, min_rear_dist,
            w0, w1, b, c1_x, c1_y, c2_x, c2_y,
            right_car_detected, right_cx, right_cy,
            two_cars_detected, gap_center_x, gap_center_y, gap_width,
            side_pair_point_count,
            rear_side_90_count, rear_side_270_count, rear_side_clear,
        ]
        out_msg.data = self.update_or_apply_cluster_hold(out_data)
        self.perception_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()
