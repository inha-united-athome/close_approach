from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("close_approach")
    cfg = PathJoinSubstitution([pkg, "config", "dead_reckoning.yaml"])

    return LaunchDescription([
        Node(
            package="close_approach",
            executable="dead_reckoning_approach_node",
            name="dead_reckoning_approach_node",
            parameters=[cfg],
            output="screen",
        ),
    ])
