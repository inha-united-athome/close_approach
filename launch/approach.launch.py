from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("close_approach")

    cfg_pc      = PathJoinSubstitution([pkg, "config", "pc_detector.yaml"])
    cfg_lidar   = PathJoinSubstitution([pkg, "config", "lidar_x_detector.yaml"])
    cfg_edge    = PathJoinSubstitution([pkg, "config", "edge_detector.yaml"])
    cfg_manager = PathJoinSubstitution([pkg, "config", "approach_manager.yaml"])
    cfg_ctrl    = PathJoinSubstitution([pkg, "config", "approach_controller.yaml"])
    cfg_logger  = PathJoinSubstitution([pkg, "config", "approach_debug_logger.yaml"])

    return LaunchDescription([
        # ── 포인트클라우드 기반 종방향(x) 오차 검출 ──────────────────────
        Node(
            package="close_approach",
            executable="pc_detector_node",
            name="pc_detector",
            parameters=[cfg_pc],
            output="screen",
        ),

        # ── LiDAR 기반 x fallback/safety 검출 ──────────────────────────
        Node(
            package="close_approach",
            executable="lidar_x_detector_node",
            name="lidar_x_detector",
            parameters=[cfg_lidar],
            output="screen",
        ),

        # ── 이미지 엣지 기반 yaw 오차 검출 ──────────────────────────────
        Node(
            package="close_approach",
            executable="edge_detector_node",
            name="edge_detector",
            parameters=[cfg_edge],
            output="screen",
        ),

        # ── 상태 관리 + 액션 서버 ───────────────────────────────────────
        Node(
            package="close_approach",
            executable="approach_manager_node",
            name="approach_manager",
            parameters=[cfg_manager],
            output="screen",
        ),

        # ── PID 속도 제어 ────────────────────────────────────────────────
        Node(
            package="close_approach",
            executable="approach_controller_node",
            name="approach_controller",
            parameters=[cfg_ctrl],
            output="screen",
        ),

        # ── 접근 1회 단위 CSV + 이미지/PCD 디버그 저장 ─────────────────
        Node(
            package="close_approach",
            executable="approach_debug_logger_node",
            name="approach_debug_logger",
            parameters=[cfg_logger],
            output="screen",
        ),
    ])
