from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='planning_module',
            executable='planning_node',
            name='planning_module',
            output='screen',
            parameters=['config/planning_params.yaml']
        )
    ])
