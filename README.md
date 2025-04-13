# Access the Carla simulation
```shell

cd /Carla-0.10.0-Linux-Shipping/Linux$

./CarlaUnreal.sh
```

# Setup the ROS Workspace
1. First cd into the directory fsai_ws and then source ros2 using `source /opt/ros/humble/setup.bash`

2. Then build the workspace using `colcon build --symlink-install`

3. Then source the workspace itself using `source install/setup.bash`

4. Run the simulation using `ros2 launch hydrakon_simulation lidar_camera_fusion.launch.py`
