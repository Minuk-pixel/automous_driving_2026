from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    data_source = LaunchConfiguration('data_source')
    cam_num = LaunchConfiguration('cam_num')
    camera_timer = LaunchConfiguration('camera_timer')
    video_path = LaunchConfiguration('video_path')
    show_image = LaunchConfiguration('show_image')
    show_lane_debug = LaunchConfiguration('show_lane_debug')
    yolo_model = LaunchConfiguration('yolo_model')
    yolo_device = LaunchConfiguration('yolo_device')
    yolo_threshold = LaunchConfiguration('yolo_threshold')
    yolo_imgsz = LaunchConfiguration('yolo_imgsz')
    yolo_frame_skip = LaunchConfiguration('yolo_frame_skip')
    yolo_half = LaunchConfiguration('yolo_half')
    yolo_image_reliability = LaunchConfiguration('yolo_image_reliability')
    bev_height_scale = LaunchConfiguration('bev_height_scale')
    vehicle_speed_mps = LaunchConfiguration('vehicle_speed_mps')
    heading_gain = LaunchConfiguration('heading_gain')
    lane_timeout_sec = LaunchConfiguration('lane_timeout_sec')
    log_period_sec = LaunchConfiguration('log_period_sec')
    use_image_publisher = LaunchConfiguration('use_image_publisher')
    use_serial = LaunchConfiguration('use_serial')
    serial_port = LaunchConfiguration('serial_port')
    serial_baud_rate = LaunchConfiguration('serial_baud_rate')

    return LaunchDescription([
        DeclareLaunchArgument('data_source', default_value='camera'),
        DeclareLaunchArgument('cam_num', default_value='4'),
        DeclareLaunchArgument('camera_timer', default_value='0.1'),
        DeclareLaunchArgument(
            'video_path',
            default_value='/home/minuk/ros2_ws/src/camera_perception_pkg/camera_perception_pkg/lib/Collected_Datasets/driving_simulation.mp4'
        ),
        DeclareLaunchArgument('show_image', default_value='true'),
        DeclareLaunchArgument('show_lane_debug', default_value='false'),
        DeclareLaunchArgument('yolo_model', default_value='/home/minuk/ros2_ws/src/camera_perception_pkg/best_0524.pt'),
        DeclareLaunchArgument('yolo_device', default_value='cuda:0'),
        DeclareLaunchArgument('yolo_threshold', default_value='0.5'),
        DeclareLaunchArgument('yolo_imgsz', default_value='320'),
        DeclareLaunchArgument('yolo_frame_skip', default_value='2'),
        DeclareLaunchArgument('yolo_half', default_value='true'),
        DeclareLaunchArgument('yolo_image_reliability', default_value='2'),
        DeclareLaunchArgument('bev_height_scale', default_value='1.25'),
        DeclareLaunchArgument('vehicle_speed_mps', default_value='0.5'),
        DeclareLaunchArgument('heading_gain', default_value='0.7'),
        DeclareLaunchArgument('lane_timeout_sec', default_value='0.8'),
        DeclareLaunchArgument('log_period_sec', default_value='0.5'),
        DeclareLaunchArgument('use_image_publisher', default_value='true'),
        DeclareLaunchArgument('use_serial', default_value='true'),
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('serial_baud_rate', default_value='115200'),

        Node(
            package='camera_perception_pkg',
            executable='image_publisher_node',
            name='image_publisher_node',
            output='screen',
            condition=IfCondition(use_image_publisher),
            parameters=[{
                'data_source': data_source,
                'cam_num': cam_num,
                'video_path': video_path,
                'logger': show_image,
                'timer': camera_timer,
                'pub_topic': 'image_raw',
            }],
        ),

        Node(
            package='camera_perception_pkg',
            executable='yolov8_node',
            name='yolov8_node',
            output='screen',
            parameters=[{
                'model': yolo_model,
                'device': yolo_device,
                'threshold': yolo_threshold,
                'imgsz': ParameterValue(yolo_imgsz, value_type=int),
                'frame_skip': ParameterValue(yolo_frame_skip, value_type=int),
                'half': ParameterValue(yolo_half, value_type=bool),
                'enable': True,
                'image_reliability': ParameterValue(yolo_image_reliability, value_type=int),
            }],
        ),

        Node(
            package='camera_perception_pkg',
            executable='lane_info_extractor_node',
            name='lane_info_extractor_node',
            output='screen',
            parameters=[{
                'sub_detection_topic': 'detections',
                'pub_trajectory_topic': 'lane_trajectory',
                'drivable_class_name': 'lane',
                'bev_height_scale': bev_height_scale,
                'row_stride': 4,
            }],
        ),

        Node(
            package='debug_pkg',
            executable='lane_debug_visualizer_node',
            name='lane_debug_visualizer_node',
            output='screen',
            condition=IfCondition(show_lane_debug),
            parameters=[{
                'sub_detection_topic': 'detections',
                'drivable_class_name': 'lane',
                'bev_height_scale': bev_height_scale,
                'row_stride': 4,
                'debug_frame_skip': 3,
            }],
        ),

        Node(
            package='decision_making_pkg',
            executable='stanley_controller_node',
            name='stanley_controller_node',
            output='screen',
            parameters=[{
                'sub_trajectory_topic': 'lane_trajectory',
                'pub_topic': 'topic_control_signal',
                'vehicle_speed_mps': vehicle_speed_mps,
                'heading_gain': heading_gain,
                'lane_timeout_sec': lane_timeout_sec,
                'log_period_sec': log_period_sec,
            }],
        ),

        Node(
            package='serial_communication_pkg',
            executable='serial_sender_node',
            name='serial_sender_node',
            output='screen',
            condition=IfCondition(use_serial),
            parameters=[{
                'sub_topic': 'topic_control_signal',
                'port': serial_port,
                'baud_rate': ParameterValue(serial_baud_rate, value_type=int),
            }],
        ),
    ])
