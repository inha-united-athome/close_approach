from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("close_approach")
    cfg = PathJoinSubstitution([pkg, "config", "align_decider.yaml"])

    return LaunchDescription([
        # 두 물체 중점 → 45도 dog-leg (방향 + 주행거리) 판단 액션 노드
        Node(
            package="close_approach",
            executable="align_decider_node",
            name="align_decider",
            parameters=[cfg],
            output="screen",
        ),
    ])
