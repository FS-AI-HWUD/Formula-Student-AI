from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Simplified launch file without conditions or complex substitutions
    
    # Add the fusion node
    fusion_node = Node(
        package='my_carla_driver',
        executable='lidar_camera_fusion',
        name='lidar_camera_fusion',
        output='screen',
        parameters=[{
            'model_path': '/runs/best.pt',
            'use_carla': True,
            'output_dir': '/fsai_ws/src/my_carla_driver/my_carla_driver/dataset',
            'show_opencv_windows': False,  # Set to False to only use RViz
            'lidar_point_size': 0.4,  # Increased size of LiDAR points in meters
            'pointnet_model_path': 'fsai_ws/src/my_carla_driver/my_carla_driver/pointnet_detector.pth',  # Path to PointNet model
            'accumulate_lidar_frames': 3,  # Number of frames to accumulate for denser visualization
        }]
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
    )
    
    # Return the launch description
    return LaunchDescription([
        fusion_node,
        rviz_node,
    ])
