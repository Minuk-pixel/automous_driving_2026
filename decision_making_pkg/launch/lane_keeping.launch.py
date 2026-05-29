from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    show_image = LaunchConfiguration('show_image')
    show_lane_debug = LaunchConfiguration('show_lane_debug')
    use_image_publisher = LaunchConfiguration('use_image_publisher')
    use_serial = LaunchConfiguration('use_serial')

    return LaunchDescription([
        DeclareLaunchArgument('show_image', default_value='true'),
        DeclareLaunchArgument('show_lane_debug', default_value='false'),
        DeclareLaunchArgument('use_image_publisher', default_value='true'),
        DeclareLaunchArgument('use_serial', default_value='true'),

        Node(
            package='camera_perception_pkg',
            executable='image_publisher_node',
            name='image_publisher_node',
            output='screen',
            condition=IfCondition(use_image_publisher),
            parameters=[{
                'logger': show_image,
            }],
        ),

        Node(
            package='camera_perception_pkg',
            executable='yolov8_node',
            name='yolov8_node',
            output='screen',
        ),

        Node(
            package='camera_perception_pkg',
            executable='lane_info_extractor_node',
            name='lane_info_extractor_node',
            output='screen',
        ),

        Node(
            package='debug_pkg',
            executable='lane_debug_visualizer_node',
            name='lane_debug_visualizer_node',
            output='screen',
            condition=IfCondition(show_lane_debug),
        ),

        Node(
            package='decision_making_pkg',
            executable='stanley_controller_node',
            name='stanley_controller_node',
            output='screen',
        ),

        Node(
            package='serial_communication_pkg',
            executable='serial_sender_node',
            name='serial_sender_node',
            output='screen',
            condition=IfCondition(use_serial),
        ),
    ])
