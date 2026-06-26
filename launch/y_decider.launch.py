from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("close_approach")
    cfg = PathJoinSubstitution([pkg, "config", "y_decider.yaml"])

    return LaunchDescription(
        [
            Node(
                package="close_approach",
                executable="y_decider_node",
                name="y_decider",
                parameters=[cfg],
                output="screen",
            )
        ]
    )
