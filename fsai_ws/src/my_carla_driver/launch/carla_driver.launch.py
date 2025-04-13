from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the launch directory
    pkg_dir = get_package_share_directory('my_carla_driver')
    config_dir = os.path.join(pkg_dir, 'config')
    
    # Create a launch configuration for parameters
    config_file = os.path.join(config_dir, 'params.yaml')
    
    # Create the launch description and populate
    ld = LaunchDescription()
    
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/dalek/Desktop/runs/detect/train8/weights/best.pt',
        description='Path to the YOLO model weights'
    )
    
    use_carla_arg = DeclareLaunchArgument(
        'use_carla',
        default_value='true',
        description='Whether to connect to Carla simulation'
    )
    
    # Add the arguments to the launch description
    ld.add_action(model_path_arg)
    ld.add_action(use_carla_arg)
    
    # Add the main driver node
    driver_node = Node(
        package='my_carla_driver',
        executable='carla_driver',
        name='carla_driver',
        output='screen',
        parameters=[
            {'model_path': LaunchConfiguration('model_path')},
            {'use_carla': LaunchConfiguration('use_carla')},
        ]
    )
    
    ld.add_action(driver_node)
    
    return ld
