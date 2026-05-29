from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    show_resized_image = LaunchConfiguration("show_resized_image")
    show_object_debug = LaunchConfiguration("show_object_debug")
    use_serial = LaunchConfiguration("use_serial")

    return LaunchDescription([
        DeclareLaunchArgument("show_resized_image", default_value="false"),
        DeclareLaunchArgument("show_object_debug", default_value="false"),
        DeclareLaunchArgument("use_serial", default_value="true"),

        Node(
            package="camera_perception_pkg",
            executable="image_resize_node",
            name="image_resize_node",
            output="screen",
            parameters=[{
                "sub_topic": "image_raw",
                "pub_topic": "image_raw_resized",
                "show_image": show_resized_image,
            }],
        ),

        Node(
            package="camera_perception_pkg",
            executable="yolov8_node",
            name="yolov8_node",
            output="screen",
            parameters=[{
                "image_topic": "image_raw_resized",
            }],
        ),

        Node(
            package="camera_perception_pkg",
            executable="yolov8_object_detection_node",
            name="yolov8_object_detection_node",
            output="screen",
            parameters=[{
                "image_topic": "image_raw_resized",
            }],
        ),

        Node(
            package="camera_perception_pkg",
            executable="mission_lane_info_extractor_node",
            name="mission_lane_info_extractor_node",
            output="screen",
        ),

        Node(
            package="debug_pkg",
            executable="object_detection_visualizer_node",
            name="object_detection_visualizer_node",
            output="screen",
            condition=IfCondition(show_object_debug),
            parameters=[{
                "image_topic": "image_raw_resized",
                "image_reliability": 2,
            }],
        ),

        Node(
            package="debug_pkg",
            executable="yolov8_visualizer_node",
            name="yolov8_visualizer_node",
            output="screen",
            condition=IfCondition(show_object_debug),
            parameters=[{
                "image_topic": "image_raw_resized",
            }],
        ),

        Node(
            package="decision_making_pkg",
            executable="obstacle_mission_controller_node",
            name="obstacle_mission_controller_node",
            output="screen",
            parameters=[{
                "image_topic": "image_raw_resized",
            }],
        ),

        Node(
            package="serial_communication_pkg",
            executable="serial_sender_node",
            name="serial_sender_node",
            output="screen",
            condition=IfCondition(use_serial),
        ),
    ])
