# To access the carla simulation
"Note the carla simulation will require high gpu power for the simulation".

this is the repository for the carla map and also how to start it up
```shell

cd /Carla-0.10.0-Linux-Shipping/Linux$

./CarlaUnreal.sh
```

#Run the carla code
```shell
cd fsai_ws/src/joseph_code
```
- There will be four files of python code that are currently there
    - Main.py
    - Path_planner.py
    - lidar_integration
    - lidar_3d_visualizer.py
- To run all these codes together you only run in the terminal workspace
```shell
python3 main.py
```
