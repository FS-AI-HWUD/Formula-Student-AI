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
            {"loop_rate": 100}, # keep as int
            {"rpm_limit": 100.0}, # must be float!
            {"max_acc": 5.0},
            {"max_braking": 5.0},
            {"cmd_timeout": 0.5}
        ],
        # arguments=['--ros-args', '--log-level', 'debug'],
    )
    
    ld = LaunchDescription()
    ld.add_action(node)
    return ld