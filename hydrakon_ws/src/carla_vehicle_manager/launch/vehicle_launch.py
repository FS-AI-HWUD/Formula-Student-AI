from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='carla_vehicle_manager',
            executable='vehicle_node',
            name='carla_vehicle_manager',
            output='screen',
            parameters=['config/vehicle_params.yaml']
        ),

        Node(
            package='carla_vehicle_manager',
            executable='ins_node',
            name='ins_node',
            output='screen',
            parameters=['config/ins_params.yaml']
        )
    ])
