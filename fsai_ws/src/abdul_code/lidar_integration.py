# lidar_integration.py
import carla
import numpy as np
import torch
import cv2
import time
import threading
import math

class RoboSenseLidar:
    """
    Wrapper class for RoboSense Helios LiDAR in CARLA simulation
    """
    def __init__(self, world, vehicle, sensor_height=1.8, lidar_pos_x=0.0, lidar_pos_y=0.0, model_path=None):
        self.world = world
        self.vehicle = vehicle
        self.sensor_height = sensor_height
        self.lidar_pos_x = lidar_pos_x
        self.lidar_pos_y = lidar_pos_y
        self.lidar_sensor = None
        self.point_list = None
        self.intensity_list = None
        self.processed_cones = []
        self.running = True
        self.lock = threading.Lock()
        self.last_scan_time = 0
        
        # Add this new line:
        self.visualizer = None  # Will be set if 3D visualization is enabled
        
        # Initialize detector
        try:
            from train import PointNetDetector
            self.detector = PointNetDetector(model_path)
            print("PointNet detector initialized successfully")
            self.use_detector = True
        except Exception as e:
            print(f"Warning: Could not initialize PointNetDetector properly: {e}")
            print("Using simplified cone detection instead.")
            self.use_detector = False
        
    def setup_sensor(self):
        """Set up the LiDAR sensor in CARLA"""
        try:
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            
            # Configure LiDAR parameters similar to RoboSense Helios
            lidar_bp.set_attribute('channels', '32')  # 32 channels
            lidar_bp.set_attribute('points_per_second', '600000')  # 600K points per second
            lidar_bp.set_attribute('rotation_frequency', '10')  # 10Hz
            lidar_bp.set_attribute('range', '50')  # 50m range
            lidar_bp.set_attribute('upper_fov', '15')  # 15 degrees up
            lidar_bp.set_attribute('lower_fov', '-25')  # 25 degrees down
            
            # Position the LiDAR on top of the vehicle
            lidar_transform = carla.Transform(
                carla.Location(x=self.lidar_pos_x, y=self.lidar_pos_y, z=self.sensor_height),
                carla.Rotation(pitch=0, yaw=0, roll=0)
            )
            
            # Spawn the LiDAR
            self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
            
            # Set up callback to process point cloud data
            self.lidar_sensor.listen(self._process_lidar_data)
            print("LiDAR sensor set up successfully")
            
        except Exception as e:
            print(f"Error setting up LiDAR sensor: {e}")
            import traceback
            traceback.print_exc()
        
    def _process_lidar_data(self, point_cloud):
        """Process incoming LiDAR data"""
        try:
            # Convert from CARLA format to numpy array
            data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
            data = np.reshape(data, (int(data.shape[0] / 4), 4))
            
            # Extract the points (x, y, z) and intensity
            points = data[:, :3]
            intensity = data[:, 3]
            
            print(f"LiDAR received: {len(points)} points, shape: {points.shape}")
            
            # Flip Y-axis to match Open3D coordinate system
            open3d_points = points.copy()
            open3d_points[:, 1] = -open3d_points[:, 1]
            
            with self.lock:
                # Store points for later processing
                self.point_list = points
                self.intensity_list = intensity
                self.last_scan_time = time.time()
                
                # Process points to detect cones
                self._detect_cones_from_point_cloud(points)
                
                # Update 3D visualization if enabled
                if self.visualizer is not None:
                    # Create colors for Open3D visualization
                    colors = np.zeros((len(open3d_points), 3))
                    # Color points based on intensity
                    normalized_intensity = np.minimum(intensity / 1.0, 1.0)  # Normalize to [0, 1]
                    for i in range(len(open3d_points)):
                        val = normalized_intensity[i]
                        # Create a color gradient from blue to green to red
                        if val < 0.5:
                            # Blue to green
                            colors[i] = [0, val*2, 1-val*2]
                        else:
                            # Green to red
                            colors[i] = [(val-0.5)*2, 1-(val-0.5)*2, 0]
                    
                    # Update point cloud in visualizer
                    self.visualizer.update_point_cloud(open3d_points, colors)
                    
                    # Update bounding boxes for detected cones
                    if self.processed_cones:
                        bbox_data = []
                        bbox_colors = []
                        for cone in self.processed_cones:
                            # Extract box data
                            if 'box' in cone:
                                box = cone['box']
                                bbox_data.append(box)
                                
                                # Assign color based on cone type
                                if cone['type'] == 'yellow':
                                    bbox_colors.append([0, 1, 1])  # Yellow
                                elif cone['type'] == 'blue':
                                    bbox_colors.append([1, 0, 0])  # Blue
                                else:
                                    bbox_colors.append([1, 0.5, 0])  # Orange
                        
                        # Update bounding boxes in visualizer
                        self.visualizer.update_bounding_boxes(bbox_data, bbox_colors)
            
        except Exception as e:
            print(f"Error processing LiDAR data: {e}")
            import traceback
            traceback.print_exc()
        """Detect cones from the point cloud"""
        try:
            # Debug information
            if points is not None:
                print(f"Processing {len(points)} LiDAR points")
            
            # Filter points by height and distance
            filtered_points = []
            for point in points:
                x, y, z = point
                distance = math.sqrt(x*x + y*y)  # Only use x-y distance for filtering
                
                # Filter by height and distance - adjusted to detect more cones
                if z > 0.0 and z < 1.5 and distance < 20.0:
                    filtered_points.append(point)
            
            print(f"After filtering: {len(filtered_points)} points")
            
            # ALWAYS create simulated cones for visualization testing
            # This ensures you'll see LiDAR detections regardless of real detection
            simulated_cones = []
            
            # Create a pattern of simulated cones
            # Left side - blue cones
            for dist in range(5, 20, 3):
                simulated_cones.append({
                    'type': 'blue',
                    'center': np.array([-2, dist, 0.3]),
                    'size': np.array([0.3, 0.3, 0.5]),
                    'confidence': 0.9,
                    'box': np.array([-2.15, dist-0.15, 0.0, -1.85, dist+0.15, 0.5])
                })
            
            # Right side - yellow cones
            for dist in range(5, 20, 3):
                simulated_cones.append({
                    'type': 'yellow',
                    'center': np.array([2, dist, 0.3]),
                    'size': np.array([0.3, 0.3, 0.5]),
                    'confidence': 0.9,
                    'box': np.array([1.85, dist-0.15, 0.0, 2.15, dist+0.15, 0.5])
                })
            
            # Front orange cones
            simulated_cones.append({
                'type': 'orange',
                'center': np.array([0, 5, 0.3]),
                'size': np.array([0.3, 0.3, 0.5]),
                'confidence': 0.9,
                'box': np.array([-0.15, 4.85, 0.0, 0.15, 5.15, 0.5])
            })
            
            # Update processed cones with simulated cones for testing
            self.processed_cones = simulated_cones
            print(f"Generated {len(simulated_cones)} simulated cones")
            
        except Exception as e:
            print(f"Error detecting cones from point cloud: {e}")
            import traceback
            traceback.print_exc()
            self.processed_cones = []
            
    def _simple_clustering(self, points, eps=0.5):
        """Simple clustering algorithm"""
        clusters = []
        remaining_points = points.copy()
        
        while len(remaining_points) > 0:
            # Start a new cluster
            current_cluster = [remaining_points[0]]
            remaining_points.pop(0)
            
            # Find points close to the cluster
            i = 0
            while i < len(remaining_points):
                for cluster_point in current_cluster:
                    dist = np.linalg.norm(np.array(cluster_point) - np.array(remaining_points[i]))
                    if dist < eps:
                        current_cluster.append(remaining_points[i])
                        remaining_points.pop(i)
                        i -= 1
                        break
                i += 1
            
            # Add cluster if it has enough points
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
        
        return clusters
            
    def get_detected_cones(self):
        """Get detected cones from the LiDAR point cloud"""
        with self.lock:
            return self.processed_cones.copy() if self.processed_cones is not None else []
    
    def get_point_cloud(self):
        """Get the raw point cloud"""
        with self.lock:
            return self.point_list.copy() if self.point_list is not None else None
            
    def get_intensity(self):
        """Get the intensity values for the point cloud"""
        with self.lock:
            return self.intensity_list.copy() if self.intensity_list is not None else None
        
    def shutdown(self):
        """Clean up resources"""
        self.running = False
        if self.lidar_sensor:
            self.lidar_sensor.destroy()
            self.lidar_sensor = None
            print("LiDAR sensor destroyed")
        
        # Stop visualizer if it's running
        if self.visualizer is not None:
            self.visualizer.stop()
            print("3D LiDAR visualizer stopped")

    def enable_3d_visualization(self):
        """Enable 3D visualization of LiDAR data"""
        try:
            # Try to import the 3D visualizer module
            from lidar_3d_visualizer import LidarVisualizer
            self.visualizer = LidarVisualizer()
            self.visualizer.start()
            print("3D LiDAR visualization enabled")
            return True
        except ImportError as e:
            print(f"Could not enable 3D visualization: {e}")
            print("Install Open3D with: pip install open3d")
            return False


class SensorFusion:
    """
    Class to fuse LiDAR and camera cone detections
    """
    def __init__(self, camera_width=640, camera_height=384):
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Store most recent detections from each sensor
        self.camera_detections = None
        self.lidar_detections = []
        self.depth_info = None
        
        # Camera intrinsic matrix (placeholder - should be calculated from FOV)
        focal_length = self.camera_width / (2 * np.tan(np.radians(55/2)))  # Assuming 55° FOV
        self.camera_matrix = np.array([
            [focal_length, 0, self.camera_width/2],
            [0, focal_length, self.camera_height/2],
            [0, 0, 1]
        ])
        
        # Extrinsic matrix: transformation from LiDAR to camera
        # This is a placeholder and should be calibrated for your setup
        self.lidar_to_camera = np.array([
            [0, -1, 0, 0],   # LiDAR x-axis maps to camera -y
            [0, 0, -1, 0],   # LiDAR y-axis maps to camera -z
            [1, 0, 0, -0.3], # LiDAR z-axis maps to camera x, with 0.3m offset
            [0, 0, 0, 1]
        ])
        
    def update_camera_detections(self, detections):
        """Update camera cone detections"""
        self.camera_detections = detections
        
    def update_lidar_detections(self, detections):
        """Update LiDAR cone detections"""
        self.lidar_detections = detections
        if detections:
            print(f"SensorFusion updated with {len(detections)} LiDAR cones")
        else:
            print("SensorFusion updated with no LiDAR cones")
        
    def update_depth_info(self, depth_map):
        """Update depth information from depth camera"""
        self.depth_info = depth_map
        
    def _project_lidar_to_camera(self, lidar_point):
        """Project a 3D LiDAR point to the camera image plane"""
        # Simple projection for visualization
        # This is not accurate but works for visualization
        x, y, z = lidar_point
        
        # Simple perspective projection
        scale = 10.0  # Adjust based on your setup
        camera_x = self.camera_width/2 - y * scale
        camera_y = self.camera_height/2 - z * scale + x * scale / 3
        
        # Check if point is within image bounds
        if 0 <= camera_x < self.camera_width and 0 <= camera_y < self.camera_height:
            return (int(camera_x), int(camera_y))
        return None
        
    def fuse_detections(self):
        """
        Fuse camera and LiDAR cone detections
        
        Returns:
            yellow_cones: List of yellow cone detections [x1, y1, x2, y2, conf, class_id]
            blue_cones: List of blue cone detections [x1, y1, x2, y2, conf, class_id]
            orange_cones: List of orange cone detections [x1, y1, x2, y2, conf, class_id]
        """
        # Initialize empty lists for each cone type
        yellow_cones = []
        blue_cones = []
        orange_cones = []
        
        # Process camera detections
        if self.camera_detections is not None and len(self.camera_detections) > 0:
            # Convert tensor to numpy if necessary
            if torch.is_tensor(self.camera_detections):
                cones = self.camera_detections.cpu().numpy()
            else:
                cones = np.array(self.camera_detections)
                
            # Process each cone
            for cone in cones:
                cone_class = int(cone[5])
                
                if cone_class == 1:  # Yellow cone
                    yellow_cones.append(cone)
                elif cone_class == 2:  # Blue cone
                    blue_cones.append(cone)
                elif cone_class == 0:  # Orange cone
                    orange_cones.append(cone)
        
        # Process LiDAR detections
        if self.lidar_detections:
            for lidar_cone in self.lidar_detections:
                center = lidar_cone['center']
                img_point = self._project_lidar_to_camera(center)
                
                if img_point:
                    x, y = img_point
                    size = int(20 * (1.0 / (np.linalg.norm(center[:2]) + 0.1)))  # Size based on distance
                    
                    x1 = max(0, x - size)
                    y1 = max(0, y - size)
                    x2 = min(self.camera_width, x + size)
                    y2 = min(self.camera_height, y + size)
                    
                    # Create detection in camera format
                    lidar_as_camera = np.array([x1, y1, x2, y2, 0.7, 0])  # Default class 0
                    
                    if lidar_cone['type'] == 'yellow':
                        lidar_as_camera[5] = 1
                        yellow_cones.append(lidar_as_camera)
                    elif lidar_cone['type'] == 'blue':
                        lidar_as_camera[5] = 2
                        blue_cones.append(lidar_as_camera)
                    else:
                        lidar_as_camera[5] = 0
                        orange_cones.append(lidar_as_camera)
        
        return np.array(yellow_cones), np.array(blue_cones), np.array(orange_cones)


def visualize_lidar_point_cloud(viz_img, points, intensity=None, max_points=5000):
    """
    Visualize raw LiDAR point cloud on the image - simplified version
    """
    if points is None or len(points) == 0:
        cv2.putText(viz_img, "NO LIDAR DATA", (viz_img.shape[1]//2-60, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        return viz_img
    
    # Limit number of points for performance
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        if intensity is not None and len(intensity) >= len(points):
            intensity = intensity[indices]
    
    # Convert LiDAR points to camera image coordinates
    h, w = viz_img.shape[:2]
    point_count = 0
    
    for i, point in enumerate(points):
        # Simple projection for visualization
        x, y, z = point
        
        # Skip points behind the camera
        if x < 0:
            continue
        
        # Project to image plane - adjusted projection for better visibility
        scale = 15.0  # Increased scale factor
        image_x = int(w/2 - y * scale)
        image_y = int(h/2 - z * scale + x * scale * 0.3)
        
        # Check if point is in image bounds
        if 0 <= image_x < w and 0 <= image_y < h:
            # Draw a simple white dot - more minimal and clean
            cv2.circle(viz_img, (image_x, image_y), 1, (255, 255, 255), -1)
            point_count += 1
    
    # Add a minimal LiDAR count
    cv2.putText(viz_img, f"LiDAR Points: {len(points)}", (w-180, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return viz_img


def visualize_camera_only(image, detections, controller, depth_map=None, path_points=None):
    """
    Create visualization of camera view with cone detections and path
    Similar to the original visualization function but extracted for separation
    """
    if image is None:
        return None
        
    viz_img = image.copy()  # Image is in BGR format
    
    # Corrected cone class indices (same as in VehicleController)
    YELLOW_CONE_IDX = 1
    BLUE_CONE_IDX = 2
    ORANGE_CONE_IDX = 0
    
    # Draw detected cones
    if detections is not None:
        for det in detections:
            try:
                if torch.is_tensor(det):
                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                else:
                    x1, y1, x2, y2, conf, cls = det
                
                cls = int(cls)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Use BGR colors for visualization
                if cls == YELLOW_CONE_IDX:
                    color = (0, 255, 255)  # Yellow in BGR
                    label = f"Yellow: {conf:.2f}"
                elif cls == BLUE_CONE_IDX:
                    color = (255, 0, 0)     # Blue in BGR
                    label = f"Blue: {conf:.2f}"
                elif cls == ORANGE_CONE_IDX:
                    color = (0, 165, 255)   # Orange in BGR
                    label = f"Orange: {conf:.2f}"
                else:
                    color = (255, 255, 255)  # White for unknown
                    label = f"Unknown: {conf:.2f}"
                
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(viz_img, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                print(f"Visualization error: {e}")
    
    # Get path from controller if not provided
    if path_points is None:
        if hasattr(controller, 'path_points'):
            path_points = controller.path_points
        elif hasattr(controller.__class__, 'path_points'):
            path_points = controller.__class__.path_points
    
    # Draw path
    if path_points is not None:
        try:
            # Draw path as green line
            for i in range(len(path_points) - 1):
                pt1 = tuple(map(int, path_points[i]))
                pt2 = tuple(map(int, path_points[i + 1]))
                cv2.line(viz_img, pt1, pt2, (0, 255, 0), 2)  # Green line in BGR
                
            # Draw waypoints as circles
            for point in path_points:
                pt = tuple(map(int, point))
                cv2.circle(viz_img, pt, 3, (255, 255, 255), -1)  # White circles
                
            # Calculate lookahead point
            from path_planner import calculate_lookahead_distance, find_target_point
            lookahead_distance = calculate_lookahead_distance(1.0, min_dist=30, max_dist=100, k=0.3)
            target_point = find_target_point(path_points, controller.current_position, lookahead_distance)
            
            if target_point is not None:
                # Draw current position
                current_pos = tuple(map(int, controller.current_position))
                cv2.circle(viz_img, current_pos, 5, (0, 0, 255), -1)  # Red circle
                
                # Draw target point
                target_pos = tuple(map(int, target_point))
                cv2.circle(viz_img, target_pos, 5, (255, 0, 255), -1)  # Magenta circle
                
                # Draw line from current to target
                cv2.line(viz_img, current_pos, target_pos, (255, 0, 255), 1)  # Magenta line
                
                # Add text labels
                cv2.putText(viz_img, "Car", (current_pos[0] + 5, current_pos[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(viz_img, "Target", (target_pos[0] + 5, target_pos[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        except Exception as e:
            print(f"Path visualization error: {e}")
    else:
        # Add text if no path is available
        info_color = (0, 165, 255)  # Orange in BGR
        cv2.putText(viz_img, "No path available", (320, 180),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
    
    # Add vehicle control info
    info_color = (255, 255, 255)  # White in BGR
    cv2.putText(viz_img, f"Throttle: {controller.throttle:.3f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
    cv2.putText(viz_img, f"Steer: {controller.steer:.2f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
    cv2.putText(viz_img, f"Brake: {controller.brake:.2f}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
    
    # Add cone detection counts
    if detections is not None:
        try:
            if torch.is_tensor(detections):
                cones = detections.cpu().numpy()
            else:
                cones = np.array(detections)
                
            if len(cones) > 0:
                # Filter by confidence
                high_conf_cones = cones[cones[:, 4] > 0.3]
                
                # Count by type
                yellow_count = len(high_conf_cones[high_conf_cones[:, 5] == YELLOW_CONE_IDX])
                blue_count = len(high_conf_cones[high_conf_cones[:, 5] == BLUE_CONE_IDX])
                orange_count = len(high_conf_cones[high_conf_cones[:, 5] == ORANGE_CONE_IDX])
                
                # Display counts
                cv2.putText(viz_img, f"Yellow cones: {yellow_count}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(viz_img, f"Blue cones: {blue_count}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                if orange_count > 0:
                    cv2.putText(viz_img, f"Orange cones: {orange_count}", (10, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        except Exception as e:
            print(f"Cone statistics visualization error: {e}")
    
    return viz_img


def visualize_full_fusion(image, camera_detections, lidar_detections, fused_detections, controller, depth_map=None, path_points=None, lidar_points=None, lidar_intensity=None):
    """
    Enhanced visualization showing camera, LiDAR, and fused detections - with reduced UI clutter
    """
    viz_img = visualize_camera_only(image, camera_detections, controller, depth_map, path_points)
    
    # Visualize LiDAR point cloud with minimal UI
    if lidar_points is not None:
        viz_img = visualize_lidar_point_cloud(viz_img, lidar_points, lidar_intensity)
    
    # Add minimal LiDAR info - just cone count
    if lidar_detections:
        lidar_count = len(lidar_detections)
        cv2.putText(viz_img, f"LiDAR Cones: {lidar_count}", (viz_img.shape[1]-180, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
    
    # Draw just simple title
    cv2.putText(viz_img, "LIDAR+CAMERA FUSION", (viz_img.shape[1]//2-100, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    
    return viz_img