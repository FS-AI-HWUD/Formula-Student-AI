# lidar_3d_visualizer.py
import open3d as o3d
import numpy as np
import threading
import time

class LidarVisualizer:
    def __init__(self):
        self.point_cloud = o3d.geometry.PointCloud()
        self.bounding_boxes = []
        self.running = True
        self.lock = threading.Lock()
        self.vis = None
        self.vis_thread = None
        
    def start(self):
        """Start visualization in a separate thread"""
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.daemon = True
        self.vis_thread.start()
        
    def _visualization_thread(self):
        """Thread for Open3D visualization"""
        # Create a visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name='Formula Student LiDAR',
            width=960,
            height=540,
            left=480,
            top=270)
        
        # Set visualizer options
        self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        self.vis.get_render_option().point_size = 2
        self.vis.get_render_option().show_coordinate_frame = True
        
        # Add point cloud geometry
        self.vis.add_geometry(self.point_cloud)
        
        # Main visualization loop
        while self.running:
            with self.lock:
                # Update point cloud
                self.vis.update_geometry(self.point_cloud)
                
                # Clear old bounding boxes
                for bbox in self.bounding_boxes:
                    self.vis.remove_geometry(bbox, False)
                self.bounding_boxes.clear()
                
                # Add new bounding boxes
                for bbox in self.bounding_boxes:
                    self.vis.add_geometry(bbox, False)
            
            # Update view
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)
        
        # Clean up
        self.vis.destroy_window()
        
    def update_point_cloud(self, points, colors=None):
        """Update the point cloud data"""
        if points is None or len(points) == 0:
            return
            
        with self.lock:
            self.point_cloud.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
            else:
                # Color points by height (Z-value)
                colors = np.zeros((len(points), 3))
                # Normalize Z to [0, 1] range
                z_min = np.min(points[:, 2])
                z_max = np.max(points[:, 2])
                if z_max > z_min:
                    normalized_z = (points[:, 2] - z_min) / (z_max - z_min)
                    # Blue to red color map
                    colors[:, 0] = normalized_z  # Red
                    colors[:, 2] = 1 - normalized_z  # Blue
                else:
                    colors[:, 2] = 1.0  # All blue if no height variation
                self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    def update_bounding_boxes(self, boxes, colors=None):
        """Update the bounding boxes for detected cones"""
        with self.lock:
            self.bounding_boxes.clear()
            
            for i, box in enumerate(boxes):
                # Extract box parameters
                min_bound = box[:3]
                max_bound = box[3:6] if len(box) > 5 else box[3:]
                
                # Create open3d box
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                
                # Assign color based on cone type if provided
                if colors is not None and i < len(colors):
                    bbox.color = colors[i]
                else:
                    # Default colors: blue, yellow, orange
                    if i % 3 == 0:
                        bbox.color = [1, 0, 0]  # Blue
                    elif i % 3 == 1:
                        bbox.color = [0, 1, 1]  # Yellow
                    else:
                        bbox.color = [0, 0.5, 1]  # Orange
                
                self.bounding_boxes.append(bbox)
    
    def stop(self):
        """Stop the visualization thread"""
        self.running = False
        if self.vis_thread is not None:
            self.vis_thread.join(timeout=1.0)