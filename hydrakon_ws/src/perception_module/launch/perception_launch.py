from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perception_module',
            executable='peception_node',
            name='perception_module',
            output='screen',
            parameters=['config/perception_params.yaml']
        )
    ])
