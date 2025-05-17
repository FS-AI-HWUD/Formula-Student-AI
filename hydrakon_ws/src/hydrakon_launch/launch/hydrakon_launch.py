from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        ExecuteProcess(
            cmd=['/usr/bin/python3', '/home/aditya/hydrakon_ws/scripts/carla_cleanup.py'],
            shell=False,
            output='screen'
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_tf_pub',
            arguments=['0', '0', '3.5', '0', '0', '0', 'map', 'lidar_link'],
            output='screen'
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_tf_pub',
            arguments=['0', '0', '0.5', '0', '0', '0', 'base_link', 'imu_link'],
            output='screen'
        ),

        Node(
            package='carla_vehicle_manager',
            executable='vehicle_node',
            name='carla_vehicle_manager',
            output='screen',
            parameters=['config/vehicle_params.yaml']
        ),

        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='carla_vehicle_manager',
                    executable='imu_node',
                    name='carla_imu_node',
                    output='screen',
                    parameters=['config/imu_params.yaml']
                )
            ]
        ),

        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='lidar_handler',
                    executable='lidar_node',
                    name='lidar_handler',
                    output='screen',
                    parameters=['config/lidar_params.yaml']
                )
            ]
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='zed2i_camera_sim',
                    executable='zed_node',
                    name='zed_camera_sim_node',
                    output='screen',
                    parameters=['config/zed_camera_params.yaml']
                )
            ]
        ),

        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     arguments=['-d', 'config/lidar_rviz_config.rviz'],
        #     output='screen'
        # )

        Node(
            package='lidar_cluster',
            executable='lidar_cluster_node',
            name='lidar_cluster_node',
            output='screen',
            parameters=['config/lidar_cluster.yaml']
        )
    ])
