from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_serial = LaunchConfiguration('use_serial')

    return LaunchDescription([
        DeclareLaunchArgument('use_serial', default_value='true'),

        Node(
            package='lidar_perception_pkg',
            executable='lidar_publisher_node',
            name='lidar_publisher_node',
            output='screen',
        ),

        Node(
            package='lidar_perception_pkg',
            executable='parking_perception_node',
            name='parking_perception_node',
            output='screen',
        ),

        Node(
            package='decision_making_pkg',
            executable='parking_control_node',
            name='parking_control_node',
            output='screen',
        ),

        Node(
            package='lidar_perception_pkg',
            executable='parking_visualization_node',
            name='parking_visualization_node',
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
