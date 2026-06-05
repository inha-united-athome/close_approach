from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("close_approach"), "config", "close_approach.yaml"]
    )

    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to close_approach parameter YAML.",
            ),
            Node(
                package="close_approach",
                executable="approach_node",
                name="approach_node",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
