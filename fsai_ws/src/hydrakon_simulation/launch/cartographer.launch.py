from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hydrakon_simulation',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ],
            arguments=[
                '-configuration_directory', '/home/abdul/Documents/Formula-Student-AI/fsai_ws/src/hydrakon_simulation/hydrakon_simulation/config',
                '-configuration_basename', 'lidar_camera_fusion.lua'
            ],
            remappings=[
                ('/points2', '/carla/lidar_points')  # Remap your LiDAR topic
            ]
        ),
        Node(
            package='cartographer_ros',
            executable='occupancy_grid_node',
            name='occupancy_grid_node',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ]
        )
    ])