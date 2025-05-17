from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='zed2i_camera_sim',
            executable='zed_node',
            name='zed2i_camera_sim',
            output='screen',
            parameters=['config/zed_params.yaml']
        )
    ])
