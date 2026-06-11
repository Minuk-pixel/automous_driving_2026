# 자율주행 캡스톤 디자인 최종보고서 초안

> 작성 메모: 본 문서는 선배 팀 최종보고서의 구성과 서술 흐름을 벤치마크하여 작성한 초안이다. 실제 제출 전 팀명, 학번, 이름, 실제 주행 결과, 사진 번호, 표 번호는 수정해야 한다.

## 1. 실험 목적

본 보고서에는 자율주행 캡스톤 디자인 수업의 주행 평가 과제를 해결하기 위해 설계 및 구현한 자율주행 소프트웨어의 구조와 실험 결과를 정리하여 서술한다. 본 수업에서 사용하는 차량은 실제 차량의 약 1/5 규모로 축소된 전동차 플랫폼이며, 제공된 하드웨어와 ROS2 기반 소프트웨어 구조를 활용하여 제한된 트랙 환경에서 자율주행 미션을 수행하는 것을 목표로 한다.

최종 평가는 시간측정 경기, 장애물 회피 및 신호등 미션, 수직 주차 미션으로 나뉜다. 시간측정 경기에서는 바깥쪽 차선을 따라 반시계 방향으로 2바퀴를 주행해야 하며, 미션수행 경기에서는 정적 장애물 회피, 신호등 정차 및 출발, 두 차량 사이의 수직 주차를 수행해야 한다. 본 프로젝트에서는 전방 카메라 1대와 차량 후방에 장착된 LiDAR 1대만을 핵심 인지 센서로 사용하였다. 전방 카메라는 차선, 주행 가능 영역, 장애물, 신호등 인식에 사용하였고, 후방 LiDAR는 주차 공간 탐색과 주차 기준선 추출에 사용하였다.

[그림 1 삽입: 최종 주행 평가 트랙 또는 룰북의 트랙 구성 이미지]

## 2. 실험 과정

### 2.1 실험 환경

#### [2.1.1 HW Specification]

본 팀은 딥러닝 기반 차선 및 객체 인식을 실시간으로 수행하기 위해 NVIDIA RTX 4060 Laptop GPU가 탑재된 ASUS TUF Gaming A14 노트북을 사용하였다. 실제 개발 장비에서 확인된 CPU는 AMD Ryzen 7 8845HS이며, ASUS 공식 사양상 FA401UV 계열은 Ryzen 7 8845HS, RTX 4060 Laptop GPU, LPDDR5X 메모리 구성을 제공한다. 아래 표는 개발 및 주행 테스트에 사용한 노트북의 주요 사양이다.

**TABLE 1. SYSTEM SPECIFICATIONS**

| Category | Details |
| --- | --- |
| Model | ASUS TUF Gaming A14 FA401UV |
| OS | Ubuntu 22.04.5 LTS |
| CPU | AMD Ryzen 7 8845HS |
| Cores / Threads | 8 Cores / 16 Threads |
| Clock Speed | Base 3.8 GHz, Max Boost 5.1 GHz |
| Memory | LPDDR5X, 32 GB class |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8 GB GDDR6 |
| Storage | 1 TB M.2 NVMe PCIe 4.0 SSD class |
| Development Framework | ROS2 Humble, Python 3.10.12 |
| Main Libraries | PyTorch 2.8.0, Ultralytics 8.4.47, OpenCV 4.11.0, NumPy 1.26.4, scikit-learn 1.7.2 |

[그림 2 삽입: 차량 전체 사진. 전방 카메라와 후방 LiDAR 위치를 색상 박스로 표시]

본 프로젝트에서 사용한 인지 센서는 전방 카메라 1대와 후방 LiDAR 1대이다. 전방 카메라는 차량 진행 방향의 이미지를 수집하며, YOLO 기반 segmentation 모델을 이용해 주행 가능 영역 및 차선을 인식하고, object detection 모델을 이용해 장애물 차량과 신호등을 검출한다. 시간측정 주행과 장애물 회피 및 신호등 미션은 전방 카메라에 크게 의존한다.

LiDAR는 차량 후방에 장착하였다. 이는 수직 주차 미션에서 차량이 후진하면서 주차 공간과 주변 차량을 인식해야 하기 때문이다. LiDAR는 각도별 거리 정보를 LaserScan 형태로 제공하며, 이를 기반으로 ROI 내부의 포인트 개수, DBSCAN 기반 군집, SVM 기반 기준선을 계산하였다. 초음파 센서 등 다른 센서는 사용하지 않았으며, 전체 인지 시스템을 카메라와 LiDAR 두 센서로 단순화하였다.

#### [2.1.2 SW Architecture]

본 프로젝트는 조교진이 제공한 ROS2 Humble 기반 architecture를 바탕으로 구현되었다. 초기 워크스페이스에는 `camera_perception_pkg`, `lidar_perception_pkg`, `decision_making_pkg`, `serial_communication_pkg`, `debug_pkg`, `interfaces_pkg` 등이 포함되어 있었으며, 각 패키지는 인지, 판단, 제어, 디버깅, 메시지 정의의 역할을 담당한다.

[그림 3 삽입: 조교님 제공 ROS2 Architecture 원본 구조도]

본 팀은 제공된 노드를 그대로 사용하는 대신, 미션 요구사항에 맞게 일부 노드를 수정하고 새로운 노드를 추가하였다. 예를 들어 기존 `lane_info_extractor_node.py`는 YOLO segmentation 결과로부터 차선의 기울기와 목표점을 계산하는 구조였으나, 본 프로젝트에서는 주행 가능 영역을 BEV로 변환하고, row-by-row scan을 통해 중심선을 추출한 뒤, meter 단위로 2차 곡선을 fitting하여 `LaneTrajectory` 메시지를 publish하도록 수정하였다. 또한 장애물 미션을 위해 `mission_lane_info_extractor_node.py`, `yolov8_object_detection_node.py`, `obstacle_mission_controller_node.py`를 추가하였다. 주차 미션에서는 `parking_perception_node.py`, `parking_control_node.py`, `parking_visualization_node.py`를 추가하였다.

YOLO 모델은 두 종류로 나누어 사용하였다. 첫 번째는 주행 가능 영역과 차선을 검출하기 위한 segmentation 모델이며, `lane`, `lane_left` 등 주행 영역 class를 mask 형태로 추론한다. 두 번째는 장애물 및 신호등을 검출하기 위한 object detection 모델이며, `car_2`, `car_3`, `traffic_red`, `traffic_green` 등의 class를 bounding box 형태로 출력한다. 카메라 입력 해상도는 처리 지연을 줄이기 위해 320x240으로 낮추었고, YOLO inference 크기 또한 320으로 설정하여 실시간성을 확보하고자 하였다.

본 프로젝트의 소프트웨어는 미션별 launch 파일로 분리되어 실행된다.

- `lane_keeping.launch.py`: 시간측정 주행용 launch
- `obstacle_mission.launch.py`: 장애물 회피 및 신호등 미션용 launch
- `parking.launch.py`: 수직 주차 미션용 launch

미션별 launch 파일은 각 미션에 필요한 노드만 실행하도록 구성하였다. 이를 통해 사용하지 않는 perception 및 debug 노드를 줄이고, 테스트 시 각 미션의 문제를 독립적으로 확인할 수 있도록 하였다.

## 2.2 자율주행 SW 구현

### [2.2.1 시간측정 주행 - lane_keeping.launch.py]

시간측정 주행은 전방 카메라 한 대만을 사용하여 수행하였다. 룰북에 따르면 차량은 바깥쪽 차선을 따라 반시계 방향으로 2바퀴 주행해야 하며, 차선 침범 및 이탈은 페널티의 원인이 된다. 따라서 시간측정 주행의 핵심은 주행 가능 영역을 안정적으로 추출하고, 이를 기준으로 차량 중심이 주행 영역 중앙을 따라가도록 조향 명령을 생성하는 것이다.

[그림 4 삽입: lane_keeping.launch.py 노드 구조도]


시간측정 주행의 perception 및 trajectory pipeline은 다음과 같다.

1. **YOLO Segmentation**
   전방 카메라 이미지에서 custom YOLO segmentation 모델을 이용하여 주행 가능 영역 class인 `lane`을 검출한다. YOLO 출력 mask는 binary image로 변환되며, 주행 가능 영역은 255, 배경은 0으로 표현된다.

2. **Bird's Eye View 변환**
   전방 카메라 영상은 원근 효과로 인해 가까운 차선과 먼 차선의 폭이 다르게 나타난다. 이를 보정하기 위해 `cv2.warpPerspective`를 사용하여 binary mask를 bird's eye view로 변환하였다. 실험 과정에서 BEV 결과가 실제 차선처럼 충분히 길게 보이지 않아, BEV height scale 값을 조정하여 종방향 정보를 더 길게 표현하였다.

[그림 5 삽입: YOLO segmentation 결과와 binary mask]
[그림 6 삽입: BEV 변환 전후 비교 이미지]

3. **Row-by-Row Scan**
   BEV binary mask를 아래쪽 행부터 위쪽 행까지 scan하며, 각 row에서 주행 가능 영역의 leftmost pixel과 rightmost pixel을 찾는다. 두 점의 평균을 해당 row의 midpoint로 사용한다. 이 midpoint 배열은 차량이 따라야 할 reference path의 원시 데이터가 된다.

4. **Pixel-to-Meter 좌표 변환**
   2차 곡선 fitting을 pixel coordinate에서 바로 수행하지 않고, 사전에 정의한 pixel-to-meter scale을 사용하여 차량 좌표계의 meter 단위로 변환하였다. x축은 차량 전방 방향, y축은 차량 좌우 방향으로 정의하였다. 이를 통해 제어기에서 사용하는 CTE와 곡률 값이 실제 차량 스케일과 대응되도록 하였다.

5. **2차 곡선 Fitting**
   meter 단위 midpoint에 대해 `np.polyfit`을 사용하여 다음과 같은 2차 다항식을 fitting하였다.

   \[
   y = ax^2 + bx + c
   \]

   이 곡선은 차량이 따라야 할 target reference path로 사용된다. fitting 결과는 `LaneTrajectory` 메시지에 포함되어 publish된다.

6. **Stanley Controller + Curvature Feedforward**
   `stanley_controller_node.py`는 `LaneTrajectory`를 subscribe하여 Stanley control 기반 조향각을 계산한다. 본 프로젝트에서 사용한 제어식은 다음과 같다.

   \[
   \delta = K_h \theta_e + \tan^{-1}\left(\frac{k e}{v + v_s}\right) + K_c \kappa
   \]

   여기서 \(\theta_e\)는 heading error, \(e\)는 cross-track error, \(v\)는 차량 속도, \(v_s\)는 zero division을 방지하기 위한 softening speed, \(\kappa\)는 곡률, \(K_h\)는 heading gain, \(K_c\)는 curvature gain이다. 초기 테스트에서 차선 기울기가 조향에 과하게 반영되어 차량이 너무 빠르게 꺾이는 현상이 나타났기 때문에 heading gain을 별도 파라미터로 추가하여 조향 반응을 완화하였다.

[그림 7 삽입: BEV midpoint와 fitting된 2차 곡선 시각화]
[그림 8 삽입: Stanley control에서 CTE, heading error, curvature를 설명하는 도식]

### [2.2.2 장애물 회피 및 신호등 미션 - obstacle_mission.launch.py]

장애물 회피 및 신호등 미션은 룰북상 장애물 회피 구간과 신호등 구간이 연속적으로 구성되어 있다. 본 프로젝트에서는 동적 장애물이 아닌 정적 장애물 회피를 대상으로 하였으며, 전방 카메라 기반 segmentation과 object detection을 동시에 사용하였다.

[그림 9 삽입: obstacle_mission.launch.py 노드 구조도]

장애물 미션에서는 오른쪽 차선 class인 `lane`과 왼쪽 차선 class인 `lane_left`를 모두 사용하였다. `mission_lane_info_extractor_node.py`는 두 class에 대해 각각 주행 가능 영역 mask를 생성하고, 시간측정 주행과 동일한 BEV, midpoint scan, meter 변환, 2차 곡선 fitting 과정을 수행한다. 그 결과 오른쪽 차선 기준 trajectory와 왼쪽 차선 기준 trajectory를 각각 publish한다.

미션 제어는 stage 기반 finite state machine으로 구현하였다.

- **Stage 1: Right Lane Following**
  차량은 오른쪽 차선인 `lane`을 기준으로 Stanley control을 수행한다. 이때 object detection 모델이 `car_2`를 검출하고, bounding box의 하단 y좌표가 이미지 높이 기준 일정 임계값에 도달하면 장애물이 가까운 것으로 판단하고 Stage 2로 전환한다.

- **Stage 2: Left Avoid and Left Lane Following**
  Stage 2 진입 직후에는 일정 시간 동안 왼쪽 최대 조향을 수행하여 정적 장애물을 회피한다. 이후에는 `lane_left` trajectory를 기준으로 Stanley control을 수행한다. 테스트 중 `car_2`와 `car_3` class가 혼동되어 Stage 2 진입 직후 곧바로 Stage 3으로 넘어가는 문제가 발생하였다. 이를 방지하기 위해 Stage 2 진입 후 일정 시간 동안 Stage 3 전이를 금지하는 lockout logic을 추가하였다.

- **Stage 3: Right Avoid and Right Lane Following**
  `car_3`가 가까워진 것으로 판단되면 오른쪽 최대 조향을 수행하고, 이후 다시 오른쪽 차선 trajectory를 따라 주행한다.

- **Stage 3 Stop Red**
  신호등 class `traffic_red`가 검출되고, bounding box 위치가 정지 기준에 도달하면 차량을 정지시킨다. 신호등은 화면 상단에 위치하기 때문에 실제 접근 시 bounding box 하단 y좌표가 일반 장애물과 반대로 감소하는 경향을 보였고, 실험적으로 정지 기준을 설정하였다.

- **Stage 4: Green Light Start**
  `traffic_green`이 검출되면 정지 상태를 해제하고 다시 차선 주행을 수행한다. 초기에는 단순 직진으로 구성하였으나, 이후 차선 유지 안정성을 위해 Stanley control을 계속 적용하는 구조로 수정하였다.

[그림 10 삽입: object detection 결과. car_2, car_3, traffic_red/green bbox 표시]
[그림 11 삽입: Stage 전이 FSM 도식]

### [2.2.3 수직 주차 미션 - parking.launch.py]

수직 주차 미션은 차량 후방에 장착된 LiDAR만을 이용하여 수행하였다. 룰북에 따르면 차량은 주차된 두 차량 사이에 후진으로 수직 주차해야 하며, 주차 후 3~5초 정차한 뒤 반대쪽 OUT 라인을 통과해야 한다. 또한 차동 회전을 이용한 회전 주차는 허용되지 않으므로, 본 프로젝트에서는 조향 기반의 후진 주차를 수행하도록 제어 로직을 구성하였다.

[그림 12 삽입: parking.launch.py 노드 구조도]

주차 미션은 크게 세 부분으로 구성하였다.

1. **주차 가능 공간 탐색**
   `parking_perception_node.py`는 `/lidar_raw`를 subscribe하여 0.3m에서 3.0m 사이의 유효 LiDAR point만 사용한다. 초기 주차 공간 탐색을 위해 특정 ROI를 설정하고, ROI 내부 point 개수를 계산한다. point 개수가 일정 기준 이상이면 차량이 주차 시작 위치에 도달한 것으로 판단하여 stage를 전환한다. ROI 위치와 크기는 실제 LiDAR map을 보며 조정하였다.

[그림 13 삽입: Main Lidar Map 3m 시각화와 ROI Monitor 화면]

2. **DBSCAN 및 SVM 기반 기준선 추출**
   후진 주차를 수행하기 위해서는 차량이 따라야 할 기준선이 필요하다. 본 프로젝트에서는 LiDAR point cloud를 DBSCAN으로 clustering하여 주차된 차량에 해당하는 point cluster를 분리하고, 두 cluster를 구분하는 SVM decision boundary를 계산하였다. 이 decision boundary는 두 주차 차량 사이의 중앙 기준선에 가까운 역할을 한다.

[그림 14 삽입: DBSCAN cluster와 SVM decision boundary 시각화]

3. **Stage 기반 후진 주차 제어**
   `parking_control_node.py`는 `/perception_data`와 `/lidar_raw`를 함께 사용하여 stage 기반 주차 동작을 수행한다. 초기에는 ROI 내부 point 수를 기준으로 주차 가능 위치를 판단하고, 이후 SVM decision boundary에 대한 `x_diff`와 `deg_diff`를 이용하여 조향각을 계산한다.

   기본 조향 제어는 다음과 같은 선형 가중합 구조를 따른다.

   \[
   \theta = w_1 x_{diff} + w_2 deg_{diff}
   \]

   차량이 기준선과 평행하게 정렬되도록 후진하며, 충돌 위험이 있거나 정렬 상태가 충분하지 않은 경우에는 전진 보정 stage를 수행한다. 보정 후 다시 후진 주차를 수행하는 반복 구조를 통해 주차 성공률을 높이고자 하였다.

현재 주차 미션은 LiDAR ROI 및 시각화 튜닝을 지속적으로 진행 중이며, Main Lidar Map을 3m 반경으로 축소하고 상하반전 좌표계를 적용하여 실제 차량 주변 point를 더 직관적으로 확인할 수 있도록 수정하였다.

## 3. 실험 결과

### 3.1 시간측정 주행 결과

시간측정 주행에서는 YOLO segmentation 모델의 인식 성능과 BEV 변환 품질이 주행 안정성에 큰 영향을 미쳤다. 초기 모델은 실내 트랙의 조명 변화와 바닥 반사에 의해 주행 가능 영역 mask가 불안정하게 출력되는 경우가 있었다. 이를 개선하기 위해 트랙 환경에서 수집한 데이터를 추가로 학습하고, segmentation 모델의 인식 결과를 지속적으로 확인하였다.

또한 원본 카메라 이미지의 해상도가 높을 경우 YOLO inference 및 후속 BEV 변환 과정에서 delay가 크게 증가하였다. 실제 카메라를 사용할 때 bag file 테스트보다 delay가 더 심하게 느껴졌으며, 이는 카메라 프레임 입력과 YOLO 추론이 동시에 부하를 발생시키기 때문으로 판단하였다. 이를 해결하기 위해 camera publisher에서 이미지를 320x240으로 resize하고, YOLO inference size를 320으로 낮추었다. 이 조정을 통해 프레임 처리량을 확보하고 차량이 인식 지연으로 자주 정지하는 문제를 완화하였다.

BEV 변환에서도 시행착오가 있었다. 초기 BEV 결과는 차선 또는 주행 가능 영역이 실제보다 종방향으로 짧게 표현되어, 차선의 기울기가 조향에 과하게 선반영되는 느낌이 있었다. 이를 보완하기 위해 BEV height scale을 증가시켜 top-down view에서 도로가 더 길게 나타나도록 조정하였다. 또한 Stanley controller에서 heading error 항이 조향에 과하게 반영되어 차량이 빠르게 꺾이는 문제가 있어 `HEADING_GAIN`을 추가하고 값을 낮추어 조향 반응을 완화하였다.

[그림 15 삽입: segmentation 학습 전후 또는 인식 결과 비교]
[그림 16 삽입: BEV height scale 조정 전후 비교]
[그림 17 삽입: 조향 로그. delta, steering, cte, theta_e 표시]

### 3.2 장애물 회피 및 신호등 미션 결과

장애물 회피 및 신호등 미션에서는 segmentation 모델과 object detection 모델이 동시에 사용되었다. 차선 제어는 오른쪽 차선과 왼쪽 차선을 각각 추출하는 방식으로 구현하였고, 장애물 회피는 `car_2`, `car_3` class의 bounding box 위치를 기준으로 stage를 전환하는 구조로 구성하였다.

실험 과정에서 가장 크게 문제가 되었던 부분은 object detection 모델의 class 혼동이었다. 특히 Stage 2로 진입한 직후 왼쪽 최대 조향을 수행하는 구간에서, 기존에 `car_2`로 인식되던 장애물이 순간적으로 `car_3`로 인식되는 경우가 있었다. 이때 stage transition 조건이 즉시 만족되어 차량이 Stage 2를 충분히 수행하지 못하고 Stage 3으로 넘어가는 문제가 발생하였다. 이를 해결하기 위해 Stage 2 진입 후 일정 시간 동안 Stage 3 전이를 막는 lockout logic을 추가하였다. 이 방법은 모델 인식이 순간적으로 흔들리더라도 제어 state가 급격하게 변하는 것을 방지하는 데 효과적이었다.

신호등 미션에서는 `traffic_red`와 `traffic_green` class를 object detection 모델이 직접 검출하도록 구성하였다. 빨간불이 가까워졌다고 판단되는 bounding box 위치에서 차량을 정지시키고, 초록불 검출 시 다시 출발하도록 구성하였다. 신호등은 카메라 화면 상단에 위치하므로 일반적인 전방 장애물과 달리 접근 시 bbox 하단 y좌표가 감소하는 경향을 보였고, 이에 따라 실험적으로 정지 기준 픽셀값을 설정하였다.

[그림 18 삽입: car_2/car_3 인식 결과와 stage 전이 로그]
[그림 19 삽입: traffic_red 정지 시점 bbox와 traffic_green 출발 시점 bbox]

### 3.3 수직 주차 미션 결과

주차 미션은 아직 완성 단계에 도달하지 못했으며, 현재는 LiDAR map 시각화와 ROI 기반 주차 시작 위치 판단을 중심으로 튜닝을 진행하였다. 초기에는 LiDAR map의 표시 범위가 12m로 설정되어 있어 실제 주차에서 중요한 0.3m에서 3.0m 사이의 point가 화면 중앙에 작게 모여 보였다. 이로 인해 ROI 위치를 직관적으로 조정하기 어려웠다. 이후 Main Lidar Map을 3m 반경으로 재구성하고, 상하반전 좌표계를 적용하여 실제 차량 기준의 주변 point 분포를 더 쉽게 확인할 수 있도록 수정하였다.

ROI 튜닝 과정에서는 메인 라이다맵에서 주차 차량에 해당하는 point cluster가 우측 하단에 모이는 것을 확인하였고, 해당 영역을 기준으로 ROI를 여러 차례 평행 이동 및 확장하였다. `parking_perception_node.py`와 `parking_visualization_node.py`가 동일한 ROI 범위를 사용하도록 맞추어, 시각화에서 보이는 ROI와 실제 stage 판단에 사용되는 ROI가 일치하도록 하였다.

DBSCAN 및 SVM 기반 주차 기준선 추출 구조는 구현되어 있으나, 실제 주차 완료까지 안정적으로 수행하기 위해서는 ROI 조건, SVM 기준선 안정성, 후진 및 전진 보정 stage의 조향 파라미터를 추가로 튜닝해야 한다. 현재까지의 실험에서는 LiDAR 기반 point cloud 시각화와 주차 시작 위치 판단 구조를 확보하였고, 이후 반복 테스트를 통해 주차 수렴 로직을 개선할 예정이다.

[그림 20 삽입: 3m Main Lidar Map과 ROI Monitor]
[그림 21 삽입: DBSCAN/SVM Parking Monitor]
[그림 22 삽입: 주차 stage 로그 또는 parking_control_node 제어 로그]

## 4. 결론 및 논의

본 프로젝트는 1/5 스케일 전동차 플랫폼에서 ROS2 기반 자율주행 소프트웨어를 구현하고, 시간측정 주행, 정적 장애물 회피 및 신호등 미션, 수직 주차 미션을 수행하는 것을 목표로 하였다. 전체 시스템은 조교진이 제공한 ROS2 architecture를 기반으로 하되, 각 미션의 요구사항에 맞춰 perception, decision making, visualization 노드를 수정 및 추가하는 방식으로 확장하였다.

시간측정 주행에서는 YOLO segmentation 기반 주행 가능 영역 추출과 BEV 변환, meter 단위 2차 곡선 fitting, Stanley controller를 결합하여 차선 유지 주행을 구현하였다. 특히 pixel 좌표계에서 직접 제어하지 않고 meter 좌표계에서 reference path를 생성함으로써 CTE와 곡률 기반 제어가 실제 차량 스케일과 대응되도록 하였다. 또한 해상도와 inference size를 낮추고, BEV height scale과 heading gain을 조정하여 인식 지연과 과도한 조향 문제를 완화하였다.

장애물 및 신호등 미션에서는 segmentation과 object detection을 동시에 사용하고, stage 기반 제어 구조를 도입하였다. `car_2`, `car_3`, `traffic_red`, `traffic_green` class를 이용해 상황을 판단하고, stage별로 오른쪽 차선 추종, 왼쪽 회피, 오른쪽 회피, 빨간불 정지, 초록불 출발을 수행하였다. YOLO 모델의 class 혼동으로 stage가 급격히 변하는 문제가 있었으나, lockout logic을 추가하여 제어 안정성을 높였다.

주차 미션에서는 후방 LiDAR만을 이용해 주차 공간을 인식하고, ROI, DBSCAN, SVM 기반으로 주차 기준선을 추출하는 구조를 구현하였다. 아직 완성도는 개선이 필요하지만, LiDAR map 시각화와 ROI 기반 stage 전환 구조를 확보하였으며, 향후 주차 제어 파라미터와 보정 stage를 추가적으로 튜닝하면 수직 주차 미션을 완성할 수 있을 것으로 판단된다.

본 프로젝트의 한계는 다음과 같다. 첫째, YOLO inference가 카메라 입력 해상도와 GPU/CPU 부하에 민감하여 실제 카메라 주행 시 delay가 발생할 수 있다. 둘째, object detection 모델의 class 혼동이 stage 기반 제어에 직접적인 영향을 줄 수 있다. 셋째, 주차 미션에서 LiDAR ROI와 SVM 기준선은 환경과 차량 위치에 민감하기 때문에 더 많은 반복 실험과 robust한 예외 처리가 필요하다.

향후에는 YOLO 모델 경량화, TensorRT 등 추론 최적화, mission state machine의 안정화, LiDAR 기반 주차 제어의 파라미터 자동 튜닝을 통해 전체 시스템의 안정성과 재현성을 높일 수 있을 것이다.

## 그림 삽입 체크리스트

| 위치 | 추천 그림 |
| --- | --- |
| 그림 1 | 룰북 트랙 구성 또는 평가 미션 전체 도식 | 
| 그림 2 | 차량 전체 사진, 전방 카메라와 후방 LiDAR 위치 표시 | 
| 그림 3 | 조교님 제공 ROS2 Architecture 원본 구조도 | 
| 그림 4 | lane_keeping.launch.py 노드 구조도 | 
| 그림 5 | YOLO segmentation mask 결과 | 
| 그림 6 | BEV 변환 전후 비교 | 생략
| 그림 7 | midpoint scan 및 2차 곡선 fitting 결과 |(이거 위아래로 기니까 위아래로 비율 줄이거나 그냥 잘라서 써도 됌)
| 그림 8 | Stanley control 변수 도식 | 
| 그림 9 | obstacle_mission.launch.py 노드 구조도 | 
| 그림 10 | car_2/car_3/traffic light object detection 결과 | 
| 그림 11 | 장애물 미션 stage FSM |
| 그림 12 | parking.launch.py 노드 구조도 | 
| 그림 13 | Main Lidar Map 3m 및 ROI Monitor |
| 그림 14 | DBSCAN/SVM Parking Monitor |
| 그림 15~22 | 실험 결과 캡처, 로그, 튜닝 전후 비교 | 생략

## References

1. 성균관대학교 자동화연구실, 「2026-1 주행평가 룰미팅 ver3.2」, 2026.
2. 성균관대학교 자동화연구실, 「5주차 오프라인 ROS2 Architecture」, 2026.
3. 성균관대학교 자동화연구실, 「6주차 오프라인 하드웨어 조립」, 2026.
4. ASUS, “ASUS TUF Gaming A14 (2024) Tech Specs,” https://www.asus.com/es/laptops/for-gaming/tuf-gaming/asus-tuf-gaming-a14-2024/techspec/
5. Ultralytics, “YOLO Documentation,” https://docs.ultralytics.com/
6. ROS 2 Documentation, “Humble Hawksbill,” https://docs.ros.org/en/humble/
