from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    data_source = LaunchConfiguration('data_source')
    cam_num = LaunchConfiguration('cam_num')
    video_path = LaunchConfiguration('video_path')
    show_image = LaunchConfiguration('show_image')
    yolo_model = LaunchConfiguration('yolo_model')
    yolo_device = LaunchConfiguration('yolo_device')
    yolo_threshold = LaunchConfiguration('yolo_threshold')
    vehicle_speed_mps = LaunchConfiguration('vehicle_speed_mps')
    use_serial = LaunchConfiguration('use_serial')

    return LaunchDescription([
        DeclareLaunchArgument('data_source', default_value='camera'),
        DeclareLaunchArgument('cam_num', default_value='4'),
        DeclareLaunchArgument(
            'video_path',
            default_value='/home/minuk/ros2_ws/src/camera_perception_pkg/camera_perception_pkg/lib/Collected_Datasets/driving_simulation.mp4'
        ),
        DeclareLaunchArgument('show_image', default_value='true'),
        DeclareLaunchArgument('yolo_model', default_value='/home/minuk/ros2_ws/src/camera_perception_pkg/best_0521.pt'),
        DeclareLaunchArgument('yolo_device', default_value='cuda:0'),
        DeclareLaunchArgument('yolo_threshold', default_value='0.5'),
        DeclareLaunchArgument('vehicle_speed_mps', default_value='0.5'),
        DeclareLaunchArgument('use_serial', default_value='true'),

        Node(
            package='camera_perception_pkg',
            executable='image_publisher_node',
            name='image_publisher_node',
            output='screen',
            parameters=[{
                'data_source': data_source,
                'cam_num': cam_num,
                'video_path': video_path,
                'logger': show_image,
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
                'enable': True,
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
                'show_image': show_image,
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
            }],
        ),
    ])
