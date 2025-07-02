# hydrakon_can

This module provides a bridge between **ROS 2 vehicle control commands** and a **CAN bus (`can0`)**, simulating the process of sending and receiving data with the ADS-DV.

## Overview

The system connects a ROS2 control node to a simulated CAN interface using a `hydrakon_can_node` bridge. This setup enables testing of vehicle control logic without the need for physical CAN hardware.

## Requirements

- ROS 2 (Humble, Foxy, or compatible)
- `socketCAN` and `vcan`  modules
- `FS_AI CAN` interface llibrary for encoding/decoding CAN frames
- `hydrakon_can` node bridge implementation

## Node: `/hydrakon_can`

### Publishers

| Topic Name                      | Message Type                                    | Description                    |
|--------------------------------|--------------------------------------------------|--------------------------------|
| `/hydrakon_can/state`          | `hydrakon_can/msg/CanState`                      | AS state and AMI state feedback|
| `/hydrakon_can/state_str`      | `std_msgs/msg/String`                            | CAN state as a string          |
| `/hydrakon_can/twist`          | `geometry_msgs/msg/TwistWithCovarianceStamped`   | Vehicle linear and angular velocity|
| `/hydrakon_can/vehicle_command`| `hydrakon_can/msg/VehicleCommand`                | Command to vehicle controller  |
| `/hydrakon_can/wheel_speed`    | `hydrakon_can/msg/WheelSpeed`                    | Feedback about wheel and steering from car.|

### Subscribers

| Topic Name                           | Message Type                                 | Description                         |
|--------------------------------------|----------------------------------------------|-------------------------------------|
| `/hydrakon_can/command`              | `ackermann_msgs/msg/AckermannDriveStamped`   | Motion command input                |
| `/hydrakon_can/driving_flag`         | `std_msgs/msg/Bool`                          | Flag to enable/disable driving      |
| `/hydrakon_can/is_mission_completed` | `std_msgs/msg/Bool`                          | Mission status input                |

### Services

| Service Name                            | Type                                             | Description                      |
|----------------------------------------|--------------------------------------------------|----------------------------------|
| `/hydrakon_can/ebs_request`            | `std_srvs/srv/Trigger`                            | Emergency braking system trigger |

### Parameters

| Parameter                                | Type    | Default     | Description                                                                 |
|------------------------------------------|---------|-------------|-----------------------------------------------------------------------------|
| `can_debug`                              | int     | 0           | Enables debug mode in the CAN node.                                         |
| `can_interface`                          | string  | "can0"      | Specifies the CAN interface to use - FSAI library.                          |
| `cmd_timeout`                            | double  | 0.5         | Timeout for control cmd in seconds, triggers EBS if expired.                |
| `debug_logging`                          | bool    | false       | Enables/disables debug logging.                                             |
| `engine_threshold`                       | float   | -5.0        | Maximum deceleration allowed for engine braking.                            |
| `loop_rate`                              | int     | 100         | Frequency at which the node runs (in Hz).                                   |
| `max_dec`                                | float   | 10.0        | Assumed maximum deceleration from mechanical brakes.                        |
| `rpm_limit`                              | float   | 4000.0      | Sets the maximum RPM limit for the vehicle.                                 |
| `simulate_can`                           | int     | 0           | Enables simulated CAN data instead of real hardware.                        |
| `use_sim_time`                           | bool    | false       | If true, uses simulation time (`/clock`) instead of system time.            |


### Launch File

| Launch File Name                       | Description                     |
|----------------------------------------|---------------------------------|
| `hydrakon_can_launch.py`               | Launches hydrakon_can node      |