from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    node = Node(
        package="hydrakon_can",
        executable="hydrakon_can_node",
        name="hydrakon_can",
        parameters=[
            {"use_sim_time": True},
            {"can_debug": 1},
            {"simulate_can": 1},
            {"can_interface": "vcan0"},
            {"loop_rate": 100},         # keep as int
            {"rpm_limit": 100.0},       # must be float!
            {"max_dec": 5.0},           # maximum deceleration (m/s^2)
            {"engine_threshold": -5.0}, # engine braking threshold (m/s^2)
            {"cmd_timeout": 0.5},      # command timeout (seconds)
            {"debug_logging": False}    # enable debug logging
        ],
        # arguments=['--ros-args', '--log-level', 'debug'],
        output='screen'
    )

    return LaunchDescription([node])