from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("close_approach")
    cfg = PathJoinSubstitution([pkg, "config", "visual_servo.yaml"])

    return LaunchDescription([
        # bbox 두 클래스 중점 → yaw 비주얼 서보잉 + x 전진 제어기
        Node(
            package="close_approach",
            executable="visual_servo_node",
            name="visual_servo",
            parameters=[cfg],
            output="screen",
        ),
    ])
