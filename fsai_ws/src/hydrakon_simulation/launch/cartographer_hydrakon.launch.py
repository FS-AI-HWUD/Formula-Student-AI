import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Get directory to config file
    config_dir = os.path.join(
        os.path.expanduser('~'),
        'Documents/Formula-Student-AI/fsai_ws/src/hydrakon_simulation/config'
    )
    
    # Verify the config file exists
    config_file = os.path.join(config_dir, 'lidar_camera_fusion.lua')
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Path to rviz config
    cartographer_ros_pkg_dir = get_package_share_directory('cartographer_ros')
    rviz_config_file = os.path.join(cartographer_ros_pkg_dir, 'configuration_files', 'demo_3d.rviz')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'),
            
        # Since CARLA's point clouds are in the "map" frame already, we just need
        # to connect other frames properly
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher_map_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher_map_to_ego',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link']
        ),
        
        # Cartographer node - IMPORTANT: We use the existing frame_id of the point cloud
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-configuration_directory', config_dir,
                      '-configuration_basename', 'lidar_camera_fusion.lua', '--v=3'], # Enable verbose logging
            remappings=[
                ('points2', '/carla/lidar_points'),
            ]),
            
        # Occupancy grid node for visualization
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-resolution', '0.05', '-publish_period_sec', '1.0']),
            
        # RViz for visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'),
    ])