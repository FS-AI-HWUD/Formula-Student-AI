import carla
import numpy as np
import open3d as o3d
import asyncio
import os

# Setup Carla client
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# LiDAR settings
LIDAR_CONFIG = {
    "channels": 16,
    "range": 150.0,
    "rotation_frequency": 20,
    "points_per_second": 576000,
    "upper_fov": 15,
    "lower_fov": -25,
}

base_file_path = 'PCD_Train_Data'
os.makedirs(base_file_path, exist_ok=True)

# LiDAR Callback
def lidar_callback(data, point_list):
    points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (len(points) // 4, 4))[:, :3]
    point_list.extend(points)
    print(f"Received {len(points)} points from LiDAR.")

async def capture_lidar_data(lidar):
    point_list = []
    lidar.listen(lambda data: lidar_callback(data, point_list))
    await asyncio.sleep(1.0) # Wait for data to be collected
    lidar.stop()
    return np.array(point_list)

async def process_cones_with_instances(lidar):
    cones = [actor for actor in world.get_actors() if 'cone' in actor.type_id]

    initial_pc = await capture_lidar_data(lidar)
    if initial_pc.size == 0:
        print("No points captured in the initial scene.")
        return
    
    labels = np.zeros(len(initial_pc), dtype=int)
    instance_id = 1

    for cone in cones:
        cone_location = cone.get_location()
        distances = np.linalg.norm(initial_pc - np.array([cone_location.x, cone_location.y, cone_location.z]), axis = 1)
        instance_id += 1

    save_labeled_point_cloud(initial_pc, labels, "labeled_initial_scene.pcd")

def save_labeled_point_cloud(pc_data, labels, file_name):
    file_path = os.path.join(base_file_path, file_name)
    with open(file_path, 'w') as file:
        file.write("# .PCD v0.7 - Point Cloud Data file format\n")
        file.write("VERSION 0.7\n")
        file.write("FIELDS x y z instance\n")
        file.write("SIZE 4 4 4 4\n")
        file.write("TYPE F F F I\n")
        file.write("COUNT 1 1 1 1\n")
        file.write("WIDTH {}\n".format(len(pc_data)))
        file.write("HEIGHT 1\n")
        file.write("POINTS {}\n".format(len(pc_data)))
        file.write("DATA ascii\n")
        for point, label in zip(pc_data, labels):
            file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {label}\n")
        print(f"Labeled Point Cloud saved to {file_path}")

async def main():
    blueprint_library = world.get_blueprint_library()
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    for key, value in LIDAR_CONFIG.items():
        lidar_bp.set_attribute(str(key), str(value))

        spawn_point = carla.Transform(carla.Location(x=-50, y=0, z=5))
        lidar = world.spawn_actor(lidar_bp, spawn_point)

        await process_cones_with_instances(lidar)
        lidar.destroy()

asyncio.run(main())