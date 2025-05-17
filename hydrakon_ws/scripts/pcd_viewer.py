import open3d as o3d
import os
import time
import glob


def visualize_latest_pcd(pcd_dir="/tmp/lidar_pcd"):
    if not os.path.exists(pcd_dir):
        print(f"PCD directory not found: {pcd_dir}")
        return

    print(f"Watching {pcd_dir} for new PCD files... Press Ctrl+C to stop.")
    shown = set()

    try:
        while True:
            files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
            new_files = [f for f in files if f not in shown]

            for pcd_file in new_files:
                print(f"Displaying {pcd_file}")
                pcd = o3d.io.read_point_cloud(pcd_file)
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(pcd)
                vis.run()
                vis.destroy_window()
                shown.add(pcd_file)

            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopped PCD viewer.")


if __name__ == "__main__":
    visualize_latest_pcd()
