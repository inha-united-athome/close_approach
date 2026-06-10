from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration('config_file')

    default_config = PathJoinSubstitution([
        FindPackageShare('close_approach'),
        'config',
        'submap_close_approach.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to submap close approach parameter YAML',
        ),
        Node(
            package='close_approach',
            executable='submap_close_approach_perception_node',
            name='submap_close_approach_perception_node',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package='close_approach',
            executable='submap_close_approach_controller_node',
            name='submap_close_approach_controller_node',
            output='screen',
            parameters=[config_file],
        ),
    ])
