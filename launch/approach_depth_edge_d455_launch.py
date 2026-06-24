from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("close_approach")
    depth_engine = LaunchConfiguration("depth_engine")

    cfg_pc = PathJoinSubstitution([pkg, "config", "pc_detector.yaml"])
    cfg_edge = PathJoinSubstitution([pkg, "config", "depth_edge_detector_d455.yaml"])
    cfg_manager = PathJoinSubstitution([pkg, "config", "approach_manager.yaml"])
    cfg_ctrl = PathJoinSubstitution([pkg, "config", "approach_controller.yaml"])
    cfg_logger = PathJoinSubstitution(
        [pkg, "config", "approach_debug_logger.yaml"]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "depth_engine",
            default_value=PathJoinSubstitution(
                [pkg, "models", "depth_anything_v2_vits.engine"]
            ),
            description="Absolute path to the robot-built Depth Anything engine",
        ),
        Node(
            package="close_approach",
            executable="pc_detector_node",
            name="pc_detector",
            parameters=[cfg_pc],
            output="screen",
        ),
        Node(
            package="close_approach",
            executable="depth_edge_detector_node",
            name="depth_edge_detector",
            parameters=[cfg_edge, {"depth_engine": depth_engine}],
            output="screen",
        ),
        Node(
            package="close_approach",
            executable="approach_manager_node",
            name="approach_manager",
            parameters=[cfg_manager],
            output="screen",
        ),
        Node(
            package="close_approach",
            executable="approach_controller_node",
            name="approach_controller",
            parameters=[cfg_ctrl],
            output="screen",
        ),
        Node(
            package="close_approach",
            executable="approach_debug_logger_node",
            name="approach_debug_logger",
            parameters=[cfg_logger],
            output="screen",
        ),
    ])
