from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lidar_handler',
            executable='lidar_node',
            name='lidar_handler',
            output='screen',
            parameters=['config/lidar_params.yaml']
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', 'config/lidar_rviz_config.rviz'],
            output='screen'
        )
    ])
