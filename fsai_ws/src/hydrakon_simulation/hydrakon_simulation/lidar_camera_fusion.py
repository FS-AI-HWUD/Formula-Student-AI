import rclpy
from rclpy.node import Node
import numpy as np
import time
import cv2
import os
import open3d as o3d
from threading import Lock
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
import carla
import sys
import threading
import torch
from sklearn.cluster import DBSCAN

# Import your existing modules
from .zed_2i import Zed2iCamera
# Import the new path planner
from .path_planning import PathPlanner, WorldCone, ConePair


def transform_points(points, transform):
    """
    Transform an array of points (N x 3) using a 4x4 transformation matrix.
    """
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points, ones))
    points_world = (transform @ points_hom.T).T[:, :3]
    return points_world


class LidarCameraFusionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_fusion_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '/home/aditya/Documents/Hydrakon/src/my_carla_driver/model/best.pt')
        self.declare_parameter('use_carla', True)
        self.declare_parameter('carla.host', 'localhost')
        self.declare_parameter('carla.port', 2000)
        self.declare_parameter('carla.timeout', 10.0)
        self.declare_parameter('output_dir', '/home/aditya/Documents/Hydrakon/src/my_carla_driver/model/dataset')
        self.declare_parameter('show_opencv_windows', True)
        self.declare_parameter('lidar_point_size', 0.4)  # Increased default point size
        self.declare_parameter('pointnet_model_path', '/home/aditya/Documents/Hydrakon/src/my_carla_driver/model/pointnet_detector.pth')
        self.declare_parameter('accumulate_lidar_frames', 3)  # Number of frames to accumulate
        
        # Parameters for Pure Pursuit
        self.declare_parameter('pure_pursuit.lookahead_distance', 4.0)
        self.declare_parameter('pure_pursuit.max_lookahead_pairs', 3)
        self.declare_parameter('pure_pursuit.depth_min', 1.0)
        self.declare_parameter('pure_pursuit.depth_max', 20.0)
        self.declare_parameter('pure_pursuit.default_track_width', 3.5)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.use_carla = self.get_parameter('use_carla').value
        self.carla_host = self.get_parameter('carla.host').value
        self.carla_port = self.get_parameter('carla.port').value
        self.carla_timeout = self.get_parameter('carla.timeout').value
        self.output_dir = self.get_parameter('output_dir').value
        self.show_opencv_windows = self.get_parameter('show_opencv_windows').value
        self.lidar_point_size = self.get_parameter('lidar_point_size').value
        self.pointnet_model_path = self.get_parameter('pointnet_model_path').value
        self.accumulate_frames = self.get_parameter('accumulate_lidar_frames').value
        
        # Pure Pursuit parameters
        self.lookahead_distance = self.get_parameter('pure_pursuit.lookahead_distance').value
        self.max_lookahead_pairs = self.get_parameter('pure_pursuit.max_lookahead_pairs').value
        self.pp_depth_min = self.get_parameter('pure_pursuit.depth_min').value
        self.pp_depth_max = self.get_parameter('pure_pursuit.depth_max').value
        self.default_track_width = self.get_parameter('pure_pursuit.default_track_width').value
        
        # Bridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()
        
        # Create publishers
        self.rgb_pub = self.create_publisher(Image, '/carla/rgb_image', 10)
        self.depth_pub = self.create_publisher(Image, '/carla/depth_image', 10)
        self.path_pub = self.create_publisher(Image, '/carla/path_image', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/carla/lidar_points', 10)
        self.fused_pub = self.create_publisher(PointCloud2, '/carla/fused_points', 10)
        self.cone_marker_pub = self.create_publisher(MarkerArray, '/carla/lidar_cones', 10)
        self.path_vis_pub = self.create_publisher(Path, '/carla/path', 10)
        
        # Transform broadcaster for tf tree
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Initialize state variables
        self.world = None
        self.vehicle = None
        self.zed_camera = None
        self.path_planner = None
        self.lidar = None
        self.lidar_data = None
        self.lidar_lock = Lock()
        self.vis_thread = None
        
        # Initialize LiDAR history for point accumulation
        self.lidar_history = []
        self.lidar_history_lock = Lock()
        
        # Vehicle tracking
        self.vehicle_poses = []
        self.cone_map = []
        
        # Initialize speed and steering control variables
        self.prev_target_speed = 0.0
        self.prev_steering = 0.0
        
        # Turn detection state
        self.lidar_right_turn_detected = False
        self.lidar_left_turn_detected = False
        self.lidar_turn_distance = float('inf')
        self.lidar_turn_confidence = 0.0
        self.uturn_state = {'detected': False, 'distance': float('inf'), 'score': 0.0, 'direction': 'right'}
        
        # Timer for main processing
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz
        
        # Setup Carla and sensors
        self.setup()
    
    def setup(self):
        """Initialize Carla, vehicle, camera, and LiDAR."""
        if not self.use_carla:
            self.get_logger().info("Carla integration disabled")
            return False
            
        try:
            client = carla.Client(self.carla_host, self.carla_port)
            client.set_timeout(self.carla_timeout)
            self.get_logger().info("Connecting to CARLA server...")
            
            self.world = client.get_world()
            self.get_logger().info("Connected to CARLA world successfully")
            
            # Set fixed time step for better stability
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.05  # 20 Hz - match our timer callback
            settings.synchronous_mode = True  # Enable synchronous mode
            self.world.apply_settings(settings)
            self.get_logger().info("Applied synchronous mode settings")
            
            self.vehicle = self.spawn_vehicle()
            if not self.vehicle:
                self.get_logger().error("Failed to spawn vehicle")
                return False
                
            # Initialize ZED camera
            self.zed_camera = Zed2iCamera(self.world, self.vehicle, 
                                         resolution=(1280, 720), 
                                         fps=30, 
                                         model_path=self.model_path)
            
            if not self.zed_camera.setup():
                self.get_logger().error("Failed to setup ZED camera")
                return False
            
            # Initialize path planner using the new pure pursuit planner
            self.path_planner = PathPlanner(self.zed_camera, 
                                           depth_min=self.pp_depth_min, 
                                           depth_max=self.pp_depth_max, 
                                           cone_spacing=1.5, 
                                           visualize=True)
            
            # Configure path planner
            self.path_planner.lookahead_distance = self.lookahead_distance
            self.path_planner.max_lookahead_pairs = self.max_lookahead_pairs
            self.path_planner.default_track_width = self.default_track_width
            self.get_logger().info(f"Pure pursuit path planner configured with lookahead={self.lookahead_distance}m")
            
            # Setup LiDAR
            if not self.setup_lidar():
                self.get_logger().error("Failed to setup LiDAR")
                return False
                
            # Enable OpenCV windows if configured
            if self.show_opencv_windows:
                self.vis_thread = threading.Thread(target=self.visualization_thread)
                self.vis_thread.daemon = True
                self.vis_thread.start()
                self.get_logger().info("OpenCV visualization enabled")
            else:
                self.get_logger().info("OpenCV visualization disabled (use RViz)")
            
            # Initial car movement - push forward to ensure physics engagement
            self.set_initial_movement()
                
            self.get_logger().info("Carla setup completed successfully")
            return True
                
        except Exception as e:
            self.get_logger().error(f"Error setting up Carla: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
        
    def _enhance_path_with_lidar_boundaries(self, lidar_boundaries):
        """
        Enhance the current path using LiDAR boundary information.
        
        Args:
            lidar_boundaries: Dictionary with 'left_boundary' and 'right_boundary' points
        """
        try:
            left_boundary = lidar_boundaries.get('left_boundary', [])
            right_boundary = lidar_boundaries.get('right_boundary', [])
            
            if not left_boundary and not right_boundary:
                return
                
            self.get_logger().info(f"Enhancing path with LiDAR boundaries: {len(left_boundary)} left, "
                                   f"{len(right_boundary)} right points")
            
            # If we don't have a path yet, create one from LiDAR boundaries
            if not self.path_planner.path or len(self.path_planner.path) < 2:
                path_points = []
                
                # Get all depth points from boundaries
                all_depths = []
                for x, depth in left_boundary:
                    all_depths.append(depth)
                for x, depth in right_boundary:
                    all_depths.append(depth)
                
                # Sort and remove duplicates
                all_depths = sorted(set(all_depths))
                
                # For each depth, find the midpoint between left and right boundaries
                for depth in all_depths:
                    left_x = None
                    right_x = None
                    
                    # Find closest left boundary point
                    best_left_dist = float('inf')
                    for x, d in left_boundary:
                        dist = abs(d - depth)
                        if dist < best_left_dist:
                            best_left_dist = dist
                            left_x = x
                    
                    # Find closest right boundary point
                    best_right_dist = float('inf')
                    for x, d in right_boundary:
                        dist = abs(d - depth)
                        if dist < best_right_dist:
                            best_right_dist = dist
                            right_x = x
                    
                    # If we have both boundaries, calculate midpoint
                    if left_x is not None and right_x is not None and best_left_dist < 2.0 and best_right_dist < 2.0:
                        mid_x = (left_x + right_x) / 2
                        path_points.append((mid_x, depth))
                    # If we only have left boundary, estimate midpoint
                    elif left_x is not None and best_left_dist < 2.0:
                        mid_x = left_x + self.path_planner.default_track_width / 2
                        path_points.append((mid_x, depth))
                    # If we only have right boundary, estimate midpoint
                    elif right_x is not None and best_right_dist < 2.0:
                        mid_x = right_x - self.path_planner.default_track_width / 2
                        path_points.append((mid_x, depth))
                
                if path_points:
                    # Sort by depth
                    path_points.sort(key=lambda p: p[1])
                    
                    # Add a point at vehicle position
                    path_points.insert(0, (0.0, 0.5))
                    
                    # Set as path
                    self.path_planner.path = path_points
                    self.get_logger().info(f"Created path from LiDAR boundaries with {len(path_points)} points")
            
            # If we already have a path, try to extend it
            elif self.path_planner.path:
                # Get maximum depth of existing path
                max_depth = max([p[1] for p in self.path_planner.path])
                
                # Find boundary points beyond current path
                extension_points = []
                
                for depth in sorted(set([d for _, d in left_boundary + right_boundary])):
                    if depth > max_depth + 1.0:  # Beyond current path
                        left_x = None
                        right_x = None
                        
                        # Find closest left boundary point
                        best_left_dist = float('inf')
                        for x, d in left_boundary:
                            if abs(d - depth) < best_left_dist:
                                best_left_dist = abs(d - depth)
                                left_x = x
                        
                        # Find closest right boundary point
                        best_right_dist = float('inf')
                        for x, d in right_boundary:
                            if abs(d - depth) < best_right_dist:
                                best_right_dist = abs(d - depth)
                                right_x = x
                        
                        # Calculate midpoint if possible
                        if left_x is not None and right_x is not None and best_left_dist < 2.0 and best_right_dist < 2.0:
                            mid_x = (left_x + right_x) / 2
                            extension_points.append((mid_x, depth))
                        elif left_x is not None and best_left_dist < 2.0:
                            # Estimate from left boundary
                            last_width = self.path_planner.default_track_width
                            if len(self.path_planner.path) >= 2:
                                # Try to maintain similar track width
                                for i in range(len(self.path_planner.path) - 1, 0, -1):
                                    p = self.path_planner.path[i]
                                    # Find a nearby left boundary point to estimate width
                                    for x, d in left_boundary:
                                        if abs(d - p[1]) < 1.0:
                                            last_width = abs(p[0] - x) * 2
                                            break
                            mid_x = left_x + last_width / 2
                            extension_points.append((mid_x, depth))
                        elif right_x is not None and best_right_dist < 2.0:
                            # Estimate from right boundary
                            last_width = self.path_planner.default_track_width
                            if len(self.path_planner.path) >= 2:
                                # Try to maintain similar track width
                                for i in range(len(self.path_planner.path) - 1, 0, -1):
                                    p = self.path_planner.path[i]
                                    # Find a nearby right boundary point to estimate width
                                    for x, d in right_boundary:
                                        if abs(d - p[1]) < 1.0:
                                            last_width = abs(p[0] - x) * 2
                                            break
                            mid_x = right_x - last_width / 2
                            extension_points.append((mid_x, depth))
                
                if extension_points:
                    # Sort by depth
                    extension_points.sort(key=lambda p: p[1])
                    
                    # Add to existing path
                    self.path_planner.path.extend(extension_points)
                    
                    # Limit path length if needed
                    max_path_points = 20
                    if len(self.path_planner.path) > max_path_points:
                        self.path_planner.path = self.path_planner.path[:max_path_points]
                    
                    self.get_logger().info(f"Extended path with {len(extension_points)} points from LiDAR boundaries")
                    
                    # Update target point for pure pursuit
                    if hasattr(self.path_planner, '_find_target_point'):
                        self.path_planner._find_target_point()
        
        except Exception as e:
            self.get_logger().error(f"Error enhancing path with LiDAR boundaries: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())#!/usr/bin/env python3
            
    def set_initial_movement(self):
        """Apply initial throttle to ensure the vehicle is properly engaged in physics."""
        if not self.vehicle:
            return
            
        try:
            # Create control command with aggressive forward motion
            control = carla.VehicleControl()
            control.throttle = 0.8  # Increased from 0.5 to 0.8 - stronger initial throttle
            control.steer = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.reverse = False
            
            # Apply control to vehicle
            self.vehicle.apply_control(control)
            
            # Tick the world multiple times to ensure physics engagement
            for _ in range(10):  # Increased from 5 to 10 ticks for better initial movement
                self.world.tick()
                
            self.get_logger().info("Applied strong initial movement to engage vehicle physics")
        except Exception as e:
            self.get_logger().error(f"Error setting initial movement: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def spawn_vehicle(self):
        """Spawn a vehicle in the CARLA world."""
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.*')[0]
            if not vehicle_bp:
                self.get_logger().error("No vehicle blueprints found")
                return None
                
            self.get_logger().info(f"Using vehicle blueprint: {vehicle_bp.id}")
            
            spawn_transform = carla.Transform(
                carla.Location(x=-35.0, y=0.0, z=5.0),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
            self.get_logger().info(f"Vehicle spawned at {spawn_transform.location}")
            
            time.sleep(2.0)
            if not vehicle.is_alive:
                self.get_logger().error("Vehicle failed to spawn or is not alive")
                return None
            return vehicle
        except Exception as e:
            self.get_logger().error(f"Error spawning vehicle: {str(e)}")
            return None
    
    def setup_lidar(self):
        """Setup LiDAR sensor."""
        try:
            # Setup LiDAR sensor
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            if not lidar_bp:
                self.get_logger().error("LiDAR blueprint not found")
                return False
                
            # Configure LiDAR attributes
            lidar_bp.set_attribute('channels', '128')
            lidar_bp.set_attribute('points_per_second', '1000000')
            lidar_bp.set_attribute('rotation_frequency', '20')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('upper_fov', '30')
            lidar_bp.set_attribute('lower_fov', '-30')
            
            # Position LiDAR above the vehicle
            lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.2))
            self.get_logger().info(f"Spawning LiDAR at {lidar_transform.location}")
            
            self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
            
            # Listen to LiDAR data
            self.lidar.listen(lambda data: self._lidar_callback(data))
            self.get_logger().info("LiDAR sensor spawned successfully")
            
            return True
        except Exception as e:
            self.get_logger().error(f"Error setting up LiDAR sensor: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def _lidar_callback(self, data):
        """Callback to process LiDAR data (runs in sensor thread)."""
        try:
            # Process raw LiDAR data into an (N, 4) array and keep only x, y, z
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (-1, 4))
            
            with self.lidar_lock:
                self.lidar_data = points[:, :3]
                self.lidar_frame = data.frame
                self.lidar_timestamp = data.timestamp
            
            self.get_logger().debug(f"Captured {len(self.lidar_data)} LiDAR points")
            
            # Save PCD file every 10 frames
            if hasattr(self, 'output_dir') and data.frame % 10 == 0:
                pcd_temp = o3d.geometry.PointCloud()
                pcd_temp.points = o3d.utility.Vector3dVector(points[:, :3])
                pcd_path = os.path.join(self.output_dir, f"lidar_{data.frame:04d}.pcd")
                o3d.io.write_point_cloud(pcd_path, pcd_temp)
                self.get_logger().info(f"Saved PCD to {pcd_path}")
        except Exception as e:
            self.get_logger().error(f"Error in LiDAR callback: {str(e)}")
    
    def timer_callback(self):
        """Main control loop callback."""
        if not self.use_carla or not self.vehicle or not self.zed_camera:
            return
            
        try:
            # Tick the world in synchronous mode
            if hasattr(self, 'world') and self.world:
                settings = self.world.get_settings()
                if settings.synchronous_mode:
                    self.world.tick()
                    
            # Process camera frame
            self.zed_camera.process_frame()
            
            # Process LiDAR data
            self.process_lidar_data()
            
            # Fuse camera and LiDAR data - extract boundaries or cones
            lidar_boundaries = self.extract_lidar_boundaries()
            
            # Plan path using Pure Pursuit planner with fusion data
            if self.path_planner:
                # Process detections regularly since plan_path_with_fusion doesn't exist
                self.path_planner.plan_path()
                
                # Add custom code to enhance with LiDAR boundaries
                if lidar_boundaries and ('left_boundary' in lidar_boundaries or 'right_boundary' in lidar_boundaries):
                    self._enhance_path_with_lidar_boundaries(lidar_boundaries)
            
            # Control the car and measure latency
            start_time = time.time()
            self.control_car()
            control_latency = (time.time() - start_time) * 1000  # in milliseconds
            
            # Get current speed
            current_speed = 0
            if hasattr(self, 'vehicle') and self.vehicle:
                velocity = self.vehicle.get_velocity()
                current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
                # Log current speed for debugging
                self.get_logger().info(f"Current vehicle speed: {current_speed:.2f} m/s ({current_speed * 2.237:.2f} mph)")
            
            # Publish path visualization for RViz
            self.publish_path_for_rviz()
            
            # Publish data for visualization
            self.publish_data()
            
            # Broadcast transforms
            self.broadcast_tf()
            
            # Visualize cone detections
            self.visualize_detected_cones()
            
            # Create system performance metrics
            self.update_and_publish_metrics(control_latency, current_speed, self.prev_target_speed)
            
        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def extract_lidar_boundaries(self):
        """
        Extract track boundaries from LiDAR points to enhance path planning.
        
        Returns:
            Dictionary with 'left_boundary' and 'right_boundary' points
        """
        if not hasattr(self, 'latest_lidar_points') or self.latest_lidar_points is None:
            return None
            
        try:
            boundaries = {'left_boundary': [], 'right_boundary': []}
            
            # Get a copy of the points
            points = self.latest_lidar_points.copy()
            
            # Filter points by height (typical cone height range)
            ground_level = np.min(points[:, 2]) if len(points) > 0 else 0
            height_min = ground_level + 0.05  # 5cm above ground
            height_max = ground_level + 0.5   # Up to 50cm tall (cone height)
            
            height_mask = (points[:, 2] >= height_min) & (points[:, 2] <= height_max)
            filtered_points = points[height_mask]
            
            if len(filtered_points) < 10:
                return boundaries
            
            # Convert to 2D points (top-down view)
            points_2d = filtered_points[:, :2]  # Only X and Y
            
            # Use clustering to find potential cones or boundary points
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(points_2d)
            labels = clustering.labels_
            
            # Process clusters
            for label in set(labels):
                if label == -1:  # Skip noise
                    continue
                    
                # Get points in this cluster
                cluster_points = points_2d[labels == label]
                center = np.mean(cluster_points, axis=0)
                
                # Determine if this is a boundary point
                # Classify based on X position (left/right of vehicle)
                # In vehicle coordinates: +X is right, +Y is forward
                if center[0] > 0.5:  # Right side
                    boundaries['right_boundary'].append((float(center[0]), float(center[1])))
                elif center[0] < -0.5:  # Left side
                    boundaries['left_boundary'].append((float(center[0]), float(center[1])))
            
            # Sort by distance (Y coordinate)
            if boundaries['left_boundary']:
                boundaries['left_boundary'].sort(key=lambda p: p[1])
            if boundaries['right_boundary']:
                boundaries['right_boundary'].sort(key=lambda p: p[1])
            
            self.get_logger().info(f"Extracted boundaries: {len(boundaries['left_boundary'])} left, "
                                  f"{len(boundaries['right_boundary'])} right points")
            
            return boundaries
            
        except Exception as e:
            self.get_logger().error(f"Error extracting LiDAR boundaries: {str(e)}")
            return {'left_boundary': [], 'right_boundary': []}
    
    def detect_cones_from_lidar(self, lidar_points):
        """
        Detect cones from LiDAR point cloud
        
        Args:
            lidar_points: Numpy array of shape (N, 3) containing LiDAR points
            
        Returns:
            list of detected cones with position and confidence
        """
        try:
            if len(lidar_points) < 10:  # Need minimum number of points
                return []
                
            # Simple clustering to find cone-like objects
            detected_cones = []
            
            # Find clusters of points close to ground
            ground_height = np.min(lidar_points[:, 2]) + 0.1  # Slightly above ground
            
            # Filter points close to ground
            near_ground_mask = (lidar_points[:, 2] < ground_height + 0.5) & (lidar_points[:, 2] > ground_height)
            near_ground_points = lidar_points[near_ground_mask]
            
            if len(near_ground_points) > 5:
                # Cluster points
                clustering = DBSCAN(eps=0.5, min_samples=5).fit(near_ground_points[:, :3])
                labels = clustering.labels_
                
                # Get cluster centers
                unique_labels = set(labels)
                for label in unique_labels:
                    if label != -1:  # Ignore noise
                        cluster_points = near_ground_points[labels == label]
                        center = np.mean(cluster_points, axis=0)
                        
                        # Simple size check - cones are small
                        max_distance = np.max(np.sqrt(np.sum((cluster_points - center)**2, axis=1)))
                        if max_distance < 0.5:  # Cones are typically small
                            # This is a potential cone
                            confidence = 0.7  # Simulated confidence
                            detected_cones.append({
                                'position': center,
                                'confidence': confidence,
                                'size': [0.3, 0.3, 0.4]  # Approximate cone size
                            })
            
            self.get_logger().info(f"Detected {len(detected_cones)} cones from LiDAR")
            return detected_cones
            
        except Exception as e:
            self.get_logger().error(f"Error in cone detection: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return []

    def process_lidar_data(self):
        """Process and analyze LiDAR data."""
        if self.lidar is None or not hasattr(self, 'lidar_data') or self.lidar_data is None:
            self.get_logger().warn("LiDAR data not available")
            return
            
        try:
            # Get sensor transform
            sensor_transform = np.array(self.lidar.get_transform().get_matrix())
            
            # Transform LiDAR points from sensor to world coordinates
            with self.lidar_lock:
                lidar_data = self.lidar_data.copy()
                point_count = len(lidar_data)
            
            self.get_logger().info(f"Processing {point_count} LiDAR points")
            
            if lidar_data.size == 0:
                self.get_logger().warn("No LiDAR points to process")
                return
                    
            # Filter out points that are too far away
            distances = np.sqrt(np.sum(lidar_data**2, axis=1))
            
            # Log distance information
            min_dist = np.min(distances) if distances.size > 0 else float('inf')
            max_dist = np.max(distances) if distances.size > 0 else 0
            self.get_logger().info(f"LiDAR distance range: {min_dist:.2f}m to {max_dist:.2f}m")
            
            close_points_mask = distances < 50.0  # Filter to 50 meters
            lidar_data = lidar_data[close_points_mask]
            
            # Transform points to world coordinates
            points_world = transform_points(lidar_data, sensor_transform)
            
            # Accumulate points across multiple frames to increase density
            with self.lidar_history_lock:
                self.lidar_history.append(points_world)
                if len(self.lidar_history) > self.accumulate_frames:
                    self.lidar_history.pop(0)  # Remove oldest frame
                
                # Combine points from all frames in history
                all_points = np.vstack(self.lidar_history)
                
                self.get_logger().info(f"Accumulated {len(all_points)} LiDAR points from {len(self.lidar_history)} frames")
            
            # Store for external use
            self.latest_lidar_points = all_points
            
            # Log detection results
            detected_cones = self.detect_cones_from_lidar(all_points)
            self.get_logger().info(f"LiDAR detected {len(detected_cones)} cones")
            
            # Visualize detected cones
            self.visualize_3d_cones(detected_cones)
            
            # Create PointCloud2 header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "map"
            
            # Create colored point cloud
            fields = [
                pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='intensity', offset=12, datatype=pc2.PointField.FLOAT32, count=1)
            ]
            
            # Create structured array for colored points
            structured_points = np.zeros(len(all_points), 
                                        dtype=[
                                            ('x', np.float32),
                                            ('y', np.float32),
                                            ('z', np.float32),
                                            ('intensity', np.float32)
                                        ])
            
            # Fill structured array
            structured_points['x'] = all_points[:, 0]
            structured_points['y'] = all_points[:, 1]
            structured_points['z'] = all_points[:, 2]
            
            # Color points based on height (z value)
            min_z = np.min(all_points[:, 2])
            max_z = np.max(all_points[:, 2])
            z_range = max_z - min_z
            if z_range > 0:
                intensity = (all_points[:, 2] - min_z) / z_range
            else:
                intensity = np.ones(len(all_points))
            
            structured_points['intensity'] = intensity
            
            # Create and publish the point cloud
            pc_msg = pc2.create_cloud(header, fields, structured_points)
            self.lidar_pub.publish(pc_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def visualize_3d_cones(self, detected_cones):
        """
        Visualize detected cones as 3D markers in RViz
        
        Args:
            detected_cones: List of dictionaries containing cone position and confidence
        """
        try:
            marker_array = MarkerArray()
            
            for i, cone in enumerate(detected_cones):
                # Create a cone marker
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "lidar_cones"
                marker.id = i
                marker.type = Marker.CYLINDER  # Use cylinder for cone
                marker.action = Marker.ADD
                
                # Set position
                marker.pose.position.x = cone['position'][0]
                marker.pose.position.y = cone['position'][1]
                marker.pose.position.z = cone['position'][2]
                
                # Set orientation (upright)
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # Set scale
                marker.scale.x = cone['size'][0]  # Diameter
                marker.scale.y = cone['size'][1]  # Diameter
                marker.scale.z = cone['size'][2]  # Height
                
                # Set color based on confidence
                confidence = cone['confidence']
                marker.color.r = 1.0
                marker.color.g = confidence  # Higher confidence = more yellow
                marker.color.b = 0.0
                marker.color.a = 0.8  # Slightly transparent
                
                # Make it persistent for a while
                marker.lifetime.sec = 1
                
                marker_array.markers.append(marker)
            
            # Publish the marker array
            self.cone_marker_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Error visualizing cones: {str(e)}")

    def publish_data(self):
        """Publish camera and path planning data."""
        if (self.zed_camera is None or 
            not hasattr(self.zed_camera, 'rgb_image') or 
            not hasattr(self.zed_camera, 'depth_image') or
            self.zed_camera.rgb_image is None or 
            self.zed_camera.depth_image is None):
            return
            
        try:
            # RGB image
            rgb_img = self.zed_camera.rgb_image.copy()
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
            rgb_msg.header.stamp = self.get_clock().now().to_msg()
            rgb_msg.header.frame_id = "camera_link"
            self.rgb_pub.publish(rgb_msg)
            
            # Depth image
            _, depth_img = self.zed_camera.depth_image
            if depth_img is not None:
                depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="bgr8")
                depth_msg.header.stamp = self.get_clock().now().to_msg()
                depth_msg.header.frame_id = "camera_link"
                self.depth_pub.publish(depth_msg)
            
            # Path visualization
            if self.path_planner and rgb_img is not None:
                path_img = self.path_planner.draw_path(rgb_img.copy())
                path_msg = self.bridge.cv2_to_imgmsg(path_img, encoding="bgr8")
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.header.frame_id = "camera_link"
                self.path_pub.publish(path_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error publishing data: {str(e)}")
    
    def broadcast_tf(self):
        """Broadcast coordinate transforms for the vehicle and sensors."""
        try:
            if not hasattr(self, 'vehicle') or not self.vehicle:
                return
                
            # Get vehicle transform
            vehicle_transform = self.vehicle.get_transform()
            
            # Broadcast vehicle to map transform
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"
            
            # Set translation
            t.transform.translation.x = vehicle_transform.location.x
            t.transform.translation.y = vehicle_transform.location.y
            t.transform.translation.z = vehicle_transform.location.z
            
            # Convert rotation to quaternion
            roll = np.radians(vehicle_transform.rotation.roll)
            pitch = np.radians(vehicle_transform.rotation.pitch)
            yaw = np.radians(vehicle_transform.rotation.yaw)
            
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            
            t.transform.rotation.w = cr * cp * cy + sr * sp * sy
            t.transform.rotation.x = sr * cp * cy - cr * sp * sy
            t.transform.rotation.y = cr * sp * cy + sr * cp * sy
            t.transform.rotation.z = cr * cp * sy - sr * sp * cy
            
            # Broadcast the transform
            self.tf_broadcaster.sendTransform(t)
            
            # Store vehicle pose for mapping
            self.vehicle_poses.append((
                vehicle_transform.location.x,
                vehicle_transform.location.y,
                vehicle_transform.rotation.yaw
            ))
            
            # Limit stored poses to last 1000
            if len(self.vehicle_poses) > 1000:
                self.vehicle_poses = self.vehicle_poses[-1000:]
            
        except Exception as e:
            self.get_logger().error(f"Error broadcasting transforms: {str(e)}")
    
    def visualization_thread(self):
        """Thread for OpenCV visualization."""
        while self.show_opencv_windows:
            try:
                if (self.zed_camera is not None and 
                    hasattr(self.zed_camera, 'rgb_image') and 
                    self.zed_camera.rgb_image is not None):
                    
                    # Show RGB image with detections
                    cv2.imshow('RGB Image with Detections', self.zed_camera.rgb_image)
                    
                    # Show path visualization if available
                    if self.path_planner and hasattr(self.path_planner, 'path') and self.path_planner.path:
                        path_img = self.path_planner.draw_path(self.zed_camera.rgb_image.copy())
                        cv2.imshow('Pure Pursuit Path', path_img)
                    
                    # Show depth image if available
                    if (hasattr(self.zed_camera, 'depth_image') and 
                        self.zed_camera.depth_image is not None):
                        _, depth_img = self.zed_camera.depth_image
                        if depth_img is not None:
                            cv2.imshow('Depth Image', depth_img)
                    
                    # Break loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.show_opencv_windows = False
                        break
                    
                    time.sleep(0.05)  # 20 Hz refresh rate
                else:
                    time.sleep(0.1)  # Slower refresh rate when no data
                    
            except Exception as e:
                self.get_logger().error(f"Error in visualization thread: {str(e)}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
    
    def destroy_node(self):
        """Clean up resources on node shutdown."""
        self.get_logger().info("Shutting down fusion node...")
        
        # Disable OpenCV windows
        self.show_opencv_windows = False
        if self.vis_thread and self.vis_thread.is_alive():
            self.vis_thread.join(timeout=1.0)
        
        # Clean up Carla resources
        if self.lidar:
            self.lidar.stop()
            self.lidar.destroy()
            self.get_logger().info("LiDAR sensor destroyed")
        
        if self.zed_camera:
            self.zed_camera.shutdown()
            self.get_logger().info("ZED camera shut down")
        
        if self.vehicle:
            self.vehicle.destroy()
            self.get_logger().info("Vehicle destroyed")
        
        super().destroy_node()

    def publish_path_for_rviz(self):
        """Publish the planned path for visualization in RViz."""
        try:
            if not hasattr(self, 'path_planner') or not self.path_planner or not self.vehicle:
                return
                
            # Get the planned path
            path = self.path_planner.get_path()
            if path is None or len(path) == 0:
                return
                
            # Get vehicle transform
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            # Create Path message
            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Convert vehicle's yaw to rotation matrix
            yaw = np.radians(vehicle_rotation.yaw)
            rotation_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            
            # Convert path points to PoseStamped messages
            for point in path:
                # In pure pursuit, path is in (lateral, forward) coordinates
                # Convert to (forward, lateral) for vehicle frame
                local_point = np.array([point[1], point[0]])  # Swap coordinates
                world_point = rotation_matrix @ local_point
                
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = vehicle_location.x + world_point[0]
                pose.pose.position.y = vehicle_location.y + world_point[1]
                pose.pose.position.z = vehicle_location.z  # Keep same height as vehicle
                
                # Set orientation tangent to the path
                if len(path_msg.poses) > 0:
                    # Get direction to next point
                    dx = world_point[0]
                    dy = world_point[1]
                    heading = np.arctan2(dy, dx)
                    
                    # Convert to quaternion
                    qw = np.cos(heading / 2)
                    qz = np.sin(heading / 2)
                    pose.pose.orientation.w = qw
                    pose.pose.orientation.z = qz
                    pose.pose.orientation.x = 0.0
                    pose.pose.orientation.y = 0.0
                else:
                    # First point uses vehicle's orientation
                    pose.pose.orientation.w = 1.0
                    pose.pose.orientation.x = 0.0
                    pose.pose.orientation.y = 0.0
                    pose.pose.orientation.z = 0.0
                
                path_msg.poses.append(pose)
            
            # Publish the path
            self.path_vis_pub.publish(path_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing path for RViz: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def control_car(self):
        """Control the car using pure pursuit steering and adaptive speed with improved cone avoidance."""
        if not hasattr(self, 'path_planner') or self.path_planner is None:
            # Fail-safe: if no path planner, move forward with small steering
            self.get_logger().warn("No path planner available! Using fail-safe control!")
            self.set_car_controls(0.0, 3.0)  # Move forward with no steering at higher speed
            return
        
        try:
            # Get current speed
            current_speed = 0.0
            if hasattr(self, 'vehicle') and self.vehicle:
                velocity = self.vehicle.get_velocity()
                current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
            # HIGHER SPEED LIMIT - Increased from 8.0 to 12.0 m/s (about 27 mph)
            if current_speed > 12.0:
                # Apply brakes to slow down
                import carla
                self.get_logger().warn(f"OVER SPEED LIMIT! Current: {current_speed:.2f} m/s - applying brakes")
                
                # Create control command with braking
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 0.4  # Strong braking
                control.steer = self.prev_steering if hasattr(self, 'prev_steering') else 0.0
                control.reverse = False
                
                # Apply control
                self.vehicle.apply_control(control)
                # Force physics update
                self.world.tick()
                
                return
                
            # Check if we have a valid path
            if not self.path_planner.path or len(self.path_planner.path) < 2:
                # No valid path - move forward with minimal steering
                self.get_logger().warn("No valid path! Moving forward...")
                self.set_car_controls(0.0, 3.0)  # Increased from 1.5 to 3.0
                return
                
            # Get steering angle from pure pursuit planner
            lookahead = max(3.0, min(7.0, current_speed * 0.8))  # Increased adaptive lookahead for stability at higher speeds
            steering = self.path_planner.calculate_steering(lookahead_distance=lookahead)
            
            # Limit steering angle based on speed to prevent tipping 
            max_steering = 1.0
            if current_speed > 5.0:  # Adjusted threshold
                max_steering = 0.8
            elif current_speed > 8.0:  # Adjusted threshold
                max_steering = 0.6
                
            steering = np.clip(steering, -max_steering, max_steering)
            
            # FASTER SPEED CONTROL - Increased all speeds
            base_speed = 6.0   # Increased from 3.0 to 6.0 m/s (about 13.4 mph)
            turn_speed = 3.5   # Increased from 1.5 to 3.5 m/s (about 7.8 mph)
            sharp_turn_speed = 2.0  # Increased from 0.8 to 2.0 m/s (about 4.5 mph)
            
            # Default to base speed
            target_speed = base_speed
            
            # IMPROVED TURN HANDLING: Determine target speed based on steering magnitude
            steering_magnitude = abs(steering)
            if steering_magnitude > 0.7:  # Very sharp turn
                target_speed = sharp_turn_speed
                self.get_logger().warn(f"SHARP TURN detected! Slowing to {target_speed:.2f}m/s")
            elif steering_magnitude > 0.3:  # Moderate turn
                target_speed = turn_speed
                self.get_logger().warn(f"Turn detected! Slowing to {target_speed:.2f}m/s")
            
            # ENHANCED CONE AVOIDANCE - detect cones earlier and slow down more aggressively
            cone_in_path = False
            closest_cone_dist = float('inf')
            
            if hasattr(self.zed_camera, 'cone_detections'):
                for cone in self.zed_camera.cone_detections:
                    if 'depth' not in cone:
                        continue
                        
                    depth = cone['depth']
                    # Keep track of closest cone regardless of whether it's in path
                    if depth < closest_cone_dist:
                        closest_cone_dist = depth
                    
                    # Check if cone is in path - wider detection corridor for earlier avoidance
                    if 'box' in cone:
                        x1, y1, x2, y2 = cone['box']
                        center_x = (x1 + x2) // 2
                        image_center_x = self.zed_camera.resolution[0] // 2
                        
                        # WIDER detection corridor based on distance - more conservative
                        center_threshold = 350 - (200 * min(1.0, depth / 15.0))
                        
                        # Check if cone is in path
                        if abs(center_x - image_center_x) < center_threshold:
                            cone_in_path = True
                            self.get_logger().warn(f"Cone detected in path at {depth:.2f}m (offset: {abs(center_x - image_center_x)}px)")
                
                # IMPROVED CONE AVOIDANCE BEHAVIOR
                if cone_in_path:
                    # Calculate safe speed based on distance to cone
                    # Start slowing down earlier (10m instead of 5m)
                    if closest_cone_dist < 10.0:
                        # More aggressive slowdown curve with distance
                        # At 10m -> 75% of base speed
                        # At 5m -> 40% of base speed 
                        # At 3m -> 25% of base speed
                        # At 1m -> 10% of base speed
                        cone_factor = min(1.0, max(0.1, closest_cone_dist / 10.0))
                        
                        # Exponential slowdown (more aggressive than linear)
                        cone_speed = base_speed * (cone_factor * cone_factor)
                        
                        if cone_speed < target_speed:
                            target_speed = cone_speed
                            self.get_logger().warn(f"CONE AVOIDANCE: Slowing to {target_speed:.2f}m/s at {closest_cone_dist:.2f}m distance")
                
                # Additional safety: emergency stop if very close to any cone (regardless of path)
                if closest_cone_dist < 1.0:
                    target_speed = 0.0  # Stop completely
                    self.get_logger().error(f"EMERGENCY STOP! Cone extremely close: {closest_cone_dist:.2f}m")
            
            # Apply gradual acceleration/deceleration
            if hasattr(self, 'prev_target_speed'):
                # More responsive speed adaptation
                max_accel = 0.4  # Increased from 0.2 to 0.4 m/s² - faster acceleration
                max_decel = 0.7  # Increased from 0.5 to 0.7 m/s² - stronger deceleration for better safety
                
                if target_speed > self.prev_target_speed:
                    # Accelerating
                    target_speed = min(target_speed, self.prev_target_speed + max_accel)
                else:
                    # Decelerating - respond quickly to obstacles and turns
                    target_speed = max(target_speed, self.prev_target_speed - max_decel)
            
            # Update for next iteration
            self.prev_target_speed = target_speed
            self.prev_steering = steering
            
            # Apply control to vehicle
            self.set_car_controls(steering, target_speed)
            
            # Log control decision
            self.get_logger().info(f"Pure pursuit control: steering={steering:.2f}, speed={target_speed:.2f}m/s")
            
        except Exception as e:
            self.get_logger().error(f"Error in car control: {str(e)}")
            # Safety fallback
            self.set_car_controls(0.0, 2.0)  # Increased from 1.0 to 2.0 m/s (about 4.5 mph)
            import traceback
    
    def set_car_controls(self, steering, speed):
        """
        Set the car's steering and speed with improved braking for obstacles.
        
        Args:
            steering: Steering angle in range [-1, 1]
            speed: Target speed in m/s (positive for forward, negative for reverse)
        """
        if self.vehicle is None:
            self.get_logger().error("Cannot set controls: No vehicle available")
            return
        
        try:
            import carla
            
            # Get current speed for better control
            velocity = self.vehicle.get_velocity()
            current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Create control command
            control = carla.VehicleControl()
            
            # Set steering (-1 to 1)
            control.steer = float(steering)
            
            # Determine if we need to accelerate, maintain speed, or brake
            if speed > 0:  # Forward motion
                # Check if we need emergency braking (going much faster than target)
                if speed < 0.5 and current_speed > 2.0:  # Hard stop requested while moving fast
                    # Emergency braking
                    control.throttle = 0.0
                    control.brake = 0.8  # Strong braking
                    control.reverse = False
                    self.get_logger().error(f"EMERGENCY BRAKING: Current speed {current_speed:.2f} with target {speed:.2f}")
                # Check if we need to slow down
                elif current_speed > speed + 0.5:  # If going more than 0.5 m/s too fast
                    # Apply brakes - stronger braking for higher overspeeds
                    speed_diff = current_speed - speed
                    brake_force = min(0.7, max(0.2, speed_diff * 0.3))  # Scale brake force with speed difference
                    
                    control.throttle = 0.0
                    control.brake = brake_force
                    control.reverse = False
                    
                    self.get_logger().warn(f"BRAKING: Current {current_speed:.2f} > target {speed:.2f}, brake={brake_force:.2f}")
                else:
                    # Normal forward throttle - proportional to desired speed
                    min_throttle = 0.4   # Increased from 0.3 to 0.4 - higher minimum throttle to ensure movement
                    max_throttle = 1.0   # Increased from 0.9 to 1.0 - maximum throttle for better acceleration
                    
                    # Progressive throttle control
                    if current_speed < speed - 0.5:  # Need substantial acceleration
                        # Stronger throttle when far from target speed
                        speed_diff = min(5.0, speed - current_speed)  # Cap to avoid excessive values
                        throttle = min(max_throttle, min_throttle + speed_diff * 0.2)  # Increased multiplier from 0.15 to 0.2
                    elif current_speed < speed - 0.1:  # Need minor acceleration
                        throttle = min_throttle + 0.15  # Increased from 0.1 to 0.15 - slight boost
                    else:  # At or near target speed
                        throttle = min_throttle
                    
                    control.throttle = throttle
                    control.brake = 0.0
                    control.reverse = False
                    
                    self.get_logger().info(f"Throttle: {control.throttle:.2f} for speed={speed:.2f}m/s (current: {current_speed:.2f})")
            else:  # Zero or negative speed - stop
                control.throttle = 0.0
                # Apply brakes harder if moving faster
                control.brake = min(0.9, max(0.4, current_speed * 0.2))  # Scale with current speed
                control.reverse = False
                
                self.get_logger().warn(f"Stopping vehicle with brake={control.brake:.2f}")
            
            # Apply control to vehicle
            self.vehicle.apply_control(control)
            
            # FORCE PHYSICS UPDATE FOR IMMEDIATE MOVEMENT
            self.world.tick()
            
            # Log that control was applied
            self.get_logger().info(f"Applied control: throttle={control.throttle:.2f}, " 
                                f"brake={control.brake:.2f}, " 
                                f"steer={control.steer:.2f}, " 
                                f"reverse={control.reverse}")
        except Exception as e:
            self.get_logger().error(f"Error setting car controls: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def update_and_publish_metrics(self, control_latency, current_speed, target_speed):
        """Update and publish performance metrics for RViz visualization."""
        try:
            # Initialize metrics if needed
            if not hasattr(self, 'control_metrics'):
                self.control_metrics = {
                    'latency': [],
                    'fps_counter': 0,
                    'fps_timer': time.time(),
                    'fps': 0,
                    'speeds': []
                }
            
            # Update metrics
            self.control_metrics['latency'].append(control_latency)
            if len(self.control_metrics['latency']) > 30:
                self.control_metrics['latency'] = self.control_metrics['latency'][-30:]
            
            self.control_metrics['speeds'].append(current_speed)
            if len(self.control_metrics['speeds']) > 100:
                self.control_metrics['speeds'] = self.control_metrics['speeds'][-100:]
            
            # Update FPS calculation
            self.control_metrics['fps_counter'] += 1
            current_time = time.time()
            if current_time - self.control_metrics['fps_timer'] >= 1.0:
                self.control_metrics['fps'] = self.control_metrics['fps_counter'] / (current_time - self.control_metrics['fps_timer'])
                self.control_metrics['fps_counter'] = 0
                self.control_metrics['fps_timer'] = current_time
            
            # Create visualization markers for RViz
            marker_array = MarkerArray()
            
            # Get current vehicle position
            if hasattr(self, 'vehicle') and self.vehicle:
                vehicle_loc = self.vehicle.get_location()
                base_x = vehicle_loc.x
                base_y = vehicle_loc.y
                base_z = vehicle_loc.z + 2.5  # Above vehicle
            else:
                base_x = 0.0
                base_y = 0.0
                base_z = 2.5
            
            # Create text display for metrics
            metrics_marker = Marker()
            metrics_marker.header.frame_id = "map"
            metrics_marker.header.stamp = self.get_clock().now().to_msg()
            metrics_marker.ns = "performance_metrics"
            metrics_marker.id = 0
            metrics_marker.type = Marker.TEXT_VIEW_FACING
            metrics_marker.action = Marker.ADD
            metrics_marker.pose.position.x = base_x
            metrics_marker.pose.position.y = base_y
            metrics_marker.pose.position.z = base_z
            metrics_marker.pose.orientation.w = 1.0
            metrics_marker.scale.z = 0.5  # Text size
            
            # Color based on performance (green = good, red = bad)
            avg_latency = sum(self.control_metrics['latency']) / len(self.control_metrics['latency']) if self.control_metrics['latency'] else 0
            if avg_latency < 100:
                metrics_marker.color.r = 0.0
                metrics_marker.color.g = 1.0
                metrics_marker.color.b = 0.0
            else:
                metrics_marker.color.r = 1.0
                metrics_marker.color.g = 0.0
                metrics_marker.color.b = 0.0
            metrics_marker.color.a = 1.0
            
            # Create formatted text
            mph = current_speed * 2.237  # Convert m/s to mph
            target_mph = target_speed * 2.237
            
            metrics_marker.text = (
                f"FPS: {self.control_metrics['fps']:.1f}\n" +
                f"Latency: {avg_latency:.1f}ms\n" +
                f"Speed: {current_speed:.1f}m/s ({mph:.1f}mph)\n" +
                f"Target: {target_speed:.1f}m/s ({target_mph:.1f}mph)"
            )
            
            marker_array.markers.append(metrics_marker)
            
            # Create and publish performance metrics
            if not hasattr(self, 'metrics_pub'):
                self.metrics_pub = self.create_publisher(MarkerArray, '/carla/performance_metrics', 10)
            
            self.metrics_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing metrics: {str(e)}")

    def visualize_detected_cones(self):
        """Visualize cone detections in RViz."""
        try:
            marker_array = MarkerArray()
            marker_id = 0
            
            # Visualize camera-detected cones
            if hasattr(self.zed_camera, 'cone_detections'):
                # Define a default field of view if not present in camera object
                camera_fov_h = 90.0  # Default horizontal field of view in degrees
                if hasattr(self.zed_camera, 'fov_h'):
                    camera_fov_h = self.zed_camera.fov_h
                
                for cone in self.zed_camera.cone_detections:
                    # Skip if no depth info
                    if 'depth' not in cone:
                        continue
                        
                    # Get cone data
                    depth = cone['depth']
                    cls = cone.get('cls', 2)  # Default to unknown class (2)
                    
                    # Get cone position in vehicle frame
                    if 'box' in cone:
                        x1, y1, x2, y2 = cone['box']
                        center_x = (x1 + x2) // 2
                        
                        # Calculate lateral position from image center
                        image_center_x = self.zed_camera.resolution[0] // 2
                        angle_rad = np.radians(((center_x - image_center_x) / image_center_x) * (camera_fov_h / 2))
                        lateral = depth * np.tan(angle_rad)
                        
                        # Get vehicle position and orientation
                        if hasattr(self, 'vehicle') and self.vehicle:
                            vehicle_transform = self.vehicle.get_transform()
                            yaw_rad = np.radians(vehicle_transform.rotation.yaw)
                            
                            # Transform to world coordinates
                            world_x = vehicle_transform.location.x + depth * np.cos(yaw_rad) - lateral * np.sin(yaw_rad)
                            world_y = vehicle_transform.location.y + depth * np.sin(yaw_rad) + lateral * np.cos(yaw_rad)
                            world_z = vehicle_transform.location.z + 0.2  # Slightly above ground
                            
                            # Create marker
                            marker = Marker()
                            marker.header.frame_id = "map"
                            marker.header.stamp = self.get_clock().now().to_msg()
                            marker.ns = "cone_detections"
                            marker.id = marker_id
                            marker.type = Marker.CYLINDER
                            marker.action = Marker.ADD
                            
                            # Set position
                            marker.pose.position.x = world_x
                            marker.pose.position.y = world_y
                            marker.pose.position.z = world_z
                            
                            # Set orientation (upright)
                            marker.pose.orientation.w = 1.0
                            
                            # Set scale
                            marker.scale.x = 0.3  # Diameter
                            marker.scale.y = 0.3  # Diameter
                            marker.scale.z = 0.5  # Height
                            
                            # Set color based on class (0=yellow, 1=blue, 2=unknown)
                            if cls == 0:  # Yellow
                                marker.color.r = 1.0
                                marker.color.g = 1.0
                                marker.color.b = 0.0
                            elif cls == 1:  # Blue
                                marker.color.r = 0.0
                                marker.color.g = 0.0
                                marker.color.b = 1.0
                            else:  # Unknown
                                marker.color.r = 1.0
                                marker.color.g = 1.0
                                marker.color.b = 1.0
                                
                            marker.color.a = 0.8
                            
                            # Set lifetime
                            marker.lifetime.sec = 1
                            
                            # Add to array
                            marker_array.markers.append(marker)
                            marker_id += 1
            
            # If we have markers, publish them
            if marker_array.markers:
                if not hasattr(self, 'cone_vis_pub'):
                    self.cone_vis_pub = self.create_publisher(MarkerArray, '/cone_detections', 10)
                self.cone_vis_pub.publish(marker_array)
                
        except Exception as e:
            self.get_logger().error(f"Error visualizing cones: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LidarCameraFusionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
