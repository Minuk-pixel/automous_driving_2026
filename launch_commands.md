# ROS2 Launch Commands

자율주행 캡스톤 프로젝트 실행 명령어 정리.

## 0. Common Setup

```bash
cd ~/ros2_ws
source install/setup.bash
```

장치 권한 설정:

```bash
sudo chmod a+rw /dev/ttyACM0  # Arduino
sudo chmod 666 /dev/ttyUSB0   # LiDAR
```

빌드 후 다시 실행할 때:

```bash
cd ~/ros2_ws
colcon build --packages-select camera_perception_pkg decision_making_pkg lidar_perception_pkg debug_pkg serial_communication_pkg interfaces_pkg --symlink-install
source install/setup.bash
```

## 1. Lane Keeping / Time Trial

기본 주행:

```bash
ros2 launch decision_making_pkg lane_keeping.launch.py \
  show_image:=false \
  show_lane_debug:=false \
  use_image_publisher:=true \
  use_serial:=true
```

디버그 화면 포함:

```bash
ros2 launch decision_making_pkg lane_keeping.launch.py \
  show_image:=true \
  show_lane_debug:=true \
  use_image_publisher:=true \
  use_serial:=true
```

bag file로 `/image_raw`가 이미 publish되는 경우:

```bash
ros2 launch decision_making_pkg lane_keeping.launch.py \
  show_image:=false \
  show_lane_debug:=true \
  use_image_publisher:=false \
  use_serial:=true
```

바퀴를 움직이지 않고 perception/debug만 확인:

```bash
ros2 launch decision_making_pkg lane_keeping.launch.py \
  show_image:=false \
  show_lane_debug:=true \
  use_image_publisher:=true \
  use_serial:=false
```

## 2. Obstacle / Traffic Light Mission

기본 주행:

```bash
ros2 launch decision_making_pkg obstacle_mission.launch.py \
  show_image:=false \
  show_object_debug:=false \
  use_image_publisher:=true \
  use_serial:=true
```

디버그 화면 포함:

```bash
ros2 launch decision_making_pkg obstacle_mission.launch.py \
  show_image:=true \
  show_object_debug:=true \
  use_image_publisher:=true \
  use_serial:=true
```

bag file 테스트:

```bash
ros2 launch decision_making_pkg obstacle_mission_bag.launch.py \
  show_resized_image:=false \
  show_object_debug:=true \
  use_serial:=true
```

바퀴를 움직이지 않고 perception/debug만 확인:

```bash
ros2 launch decision_making_pkg obstacle_mission.launch.py \
  show_image:=false \
  show_object_debug:=true \
  use_image_publisher:=true \
  use_serial:=false
```

## 3. Parking Mission

기본 주행:

```bash
ros2 launch decision_making_pkg parking.launch.py \
  use_serial:=true
```

바퀴를 움직이지 않고 LiDAR map/perception만 확인:

```bash
ros2 launch decision_making_pkg parking.launch.py \
  use_serial:=false
```

노드별 단독 확인:

```bash
ros2 run lidar_perception_pkg lidar_publisher_node
```

```bash
ros2 run lidar_perception_pkg parking_perception_node
```

```bash
ros2 run lidar_perception_pkg parking_visualization_node
```

```bash
ros2 run decision_making_pkg parking_control_node
```

```bash
ros2 run serial_communication_pkg serial_sender_node
```

## 4. Useful Checks

토픽 목록 확인:

```bash
ros2 topic list
```

토픽 publish 주기 확인:

```bash
ros2 topic hz /image_raw
ros2 topic hz /detections
ros2 topic hz /object_detections
ros2 topic hz /lane_trajectory
ros2 topic hz /topic_control_signal
ros2 topic hz /lidar_raw
ros2 topic hz /perception_data
```

제어 명령 확인:

```bash
ros2 topic echo /topic_control_signal
```

라이다 raw 확인:

```bash
ros2 topic echo /lidar_raw
```

주차 perception 결과 확인:

```bash
ros2 topic echo /perception_data
```

## 5. Notes

- 실제 차량을 움직일 때는 `use_serial:=true`.
- bag file이나 외부 카메라 노드가 `/image_raw`를 이미 publish하면 `use_image_publisher:=false`.
- 화면 디버그가 느리면 `show_image:=false`, `show_lane_debug:=false`, `show_object_debug:=false`.
- Arduino는 보통 `/dev/ttyACM0`, LiDAR는 보통 `/dev/ttyUSB0`.
- 장치명이 다르면 `ls /dev/ttyA*` 또는 `ls /dev/ttyUSB*`로 확인한다.


