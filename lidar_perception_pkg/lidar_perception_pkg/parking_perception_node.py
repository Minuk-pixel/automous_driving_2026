import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from sklearn.cluster import DBSCAN # [변경됨] K-Means -> DBSCAN
from sklearn.svm import SVC

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('parking_perception_node')
        
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.perception_pub = self.create_publisher(Float32MultiArray, '/perception_data', 10)
        
        # [변경됨] DBSCAN 설정: 반경 400mm 내에 점이 5개 이상 있으면 하나의 군집으로 인정
        self.dbscan = DBSCAN(eps=400, min_samples=5)
        self.svm = SVC(kernel='linear')
        
        # [추가됨] Tracking 변수: [1번차 중심점, 2번차 중심점]
        self.tracked_centroids = []
        
        self.get_logger().info('Perception Node Started: Running DBSCAN & Tracking...')

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        valid_indices = np.isfinite(ranges) & (ranges > 1.2) & (ranges <= 12.0)
        valid_ranges = ranges[valid_indices] * 1000.0 
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid_angles = angles[valid_indices] - (np.pi / 2) 
        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)
        lidar_data = np.vstack((x_coords, y_coords)).T

        point_count = 0.0
        is_svm_valid = 0.0
        x_diff, deg_diff, min_rear_dist = 0.0, 0.0, 9999.0
        w0, w1, b = 0.0, 0.0, 0.0
        c1_x, c1_y, c2_x, c2_y = 0.0, 0.0, 0.0, 0.0 # [추가] 시각화 노드에 보낼 중심점

        if len(lidar_data) == 0: return

        # ROI & 충돌 방지 계산 (기존과 동일)
        roi_mask = (lidar_data[:, 0] >= -500) & (lidar_data[:, 0] <= 500) & (lidar_data[:, 1] >= 2000) & (lidar_data[:, 1] <= 3000)
        point_count = float(np.sum(roi_mask))

        close_mask = (lidar_data[:, 1] > 0) & (lidar_data[:, 1] < 1500) & (np.abs(lidar_data[:, 0]) < 800)
        close_pts = lidar_data[close_mask]
        if len(close_pts) > 0:
            dists = np.sqrt(close_pts[:, 0]**2 + close_pts[:, 1]**2)
            min_rear_dist = float(np.min(dists))

        # [핵심 변경] DBSCAN 및 Tracking 로직
        mask = (np.abs(lidar_data[:, 0]) < 6000) & (lidar_data[:, 1] > -6000) & (lidar_data[:, 1] < 8000)
        rear_points = lidar_data[mask]

        if len(rear_points) > 10:
            labels = self.dbscan.fit_predict(rear_points)
            unique_labels = set(labels) - {-1} # -1은 노이즈이므로 제외
            
            clusters = []
            for lbl in unique_labels:
                pts = rear_points[labels == lbl]
                centroid = np.mean(pts, axis=0)
                clusters.append({'centroid': centroid, 'points': pts})

            if len(clusters) >= 2:
                # 1. 처음 두 대의 차를 인식할 때 (가장 점이 많은 두 덩어리 선택)
                if len(self.tracked_centroids) == 0:
                    clusters.sort(key=lambda x: len(x['points']), reverse=True)
                    self.tracked_centroids = [clusters[0]['centroid'], clusters[1]['centroid']]

                # 2. Tracking: 이전 중심점과 가장 가까운 현재 군집을 매칭
                car_pts = [[], []]
                new_centroids = [[0.0, 0.0], [0.0, 0.0]]
                
                for i, target_c in enumerate(self.tracked_centroids):
                    best_idx, min_dist = -1, float('inf')
                    for j, cluster in enumerate(clusters):
                        dist = np.linalg.norm(cluster['centroid'] - target_c)
                        if dist < min_dist:
                            min_dist, best_idx = dist, j
                            
                    # 매칭된 군집이 1.5m(1500mm) 이내로 이동했을 때만 동일 차량으로 인정
                    if best_idx != -1 and min_dist < 1500:
                        car_pts[i] = clusters[best_idx]['points']
                        new_centroids[i] = clusters[best_idx]['centroid']
                        clusters.pop(best_idx) # 중복 매칭 방지
                
                # 업데이트 및 SVM 진행
                if len(car_pts[0]) > 0 and len(car_pts[1]) > 0:
                    self.tracked_centroids = new_centroids
                    c1_x, c1_y = new_centroids[0]
                    c2_x, c2_y = new_centroids[1]

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

        # [변경됨] 배열 크기가 8개에서 12개로 늘어납니다 (중심점 4개 추가)
        out_msg = Float32MultiArray()
        out_msg.data = [point_count, is_svm_valid, x_diff, deg_diff, min_rear_dist, w0, w1, b, c1_x, c1_y, c2_x, c2_y]
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
