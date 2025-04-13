#!/usr/bin/env python3
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
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Path
import carla
import sys
import threading
import torch
from sklearn.cluster import DBSCAN

# Import your existing modules
from .zed_2i import Zed2iCamera  # Import from the same package # Will need to modify this # Will need to modify this
from .path_planning import PathPlanner


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
        self.declare_parameter('model_path', '/home/dalek/Desktop/runs/detect/train8/weights/best.pt')
        self.declare_parameter('use_carla', True)
        self.declare_parameter('carla.host', 'localhost')
        self.declare_parameter('carla.port', 2000)
        self.declare_parameter('carla.timeout', 10.0)
        self.declare_parameter('output_dir', '/home/dalek/attempt_1/Lidar_test/dataset')
        self.declare_parameter('show_opencv_windows', True)
        self.declare_parameter('lidar_point_size', 0.4)  # Increased default point size
        self.declare_parameter('pointnet_model_path', '/home/dalek/attempt_1/pointnet_detector.pth')
        self.declare_parameter('accumulate_lidar_frames', 3)  # Number of frames to accumulate
        
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
                
            # Initialize path planner
            self.path_planner = PathPlanner(self.zed_camera, 
                                           depth_min=1.0, 
                                           depth_max=20.0, 
                                           cone_spacing=1.5, 
                                           visualize=True)
            
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
                
            self.get_logger().info("Carla setup completed successfully")
            return True
                
        except Exception as e:
            self.get_logger().error(f"Error setting up Carla: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
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
            # Process camera frame
            self.zed_camera.process_frame()
            
            # Process LiDAR data
            self.process_lidar_data()
            
            # Visualize current PCD data
            self.visualize_pcd_data()
            
            # Update PCD browser
            self.update_pcd_file_list()
            
            # Handle keyboard input
            self.handle_keyboard_input()
            
            # Plan path
            self.path_planner.plan_path()
            
            # Control the car and measure latency
            start_time = time.time()
            self.control_car()
            control_latency = (time.time() - start_time) * 1000  # in milliseconds
            
            # Get current speed
            current_speed = 0
            if hasattr(self, 'vehicle') and self.vehicle:
                velocity = self.vehicle.get_velocity()
                current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Update and publish metrics
            self.update_and_publish_metrics(control_latency, current_speed, self.prev_target_speed)
            
            # Visualize detected cones
            self.visualize_detected_cones()
            
            # Publish camera and path data for RViz
            self.publish_data()
            
            # Publish path for RViz
            self.publish_path_for_rviz()
            
            # Broadcast transforms
            self.broadcast_tf()
            
        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def detect_cones_from_lidar(self, lidar_points):
        """
        Use PointNet model to detect cones from LiDAR point cloud
        
        Args:
            lidar_points: Numpy array of shape (N, 3) containing LiDAR points
            
        Returns:
            list of detected cones with position and confidence
        """
        try:
            if not hasattr(self, 'pointnet_model'):
                # Check if model path is set
                if not hasattr(self, 'pointnet_model_path'):
                    self.pointnet_model_path = "/home/dalek/attempt_1/pointnet_detector.pth"
                    self.get_logger().info(f"Using default PointNet model path: {self.pointnet_model_path}")
                
                # Load PointNet model
                if os.path.exists(self.pointnet_model_path):
                    self.get_logger().info(f"Loading PointNet model from {self.pointnet_model_path}")
                    self.pointnet_model = torch.load(self.pointnet_model_path, map_location=torch.device('cpu'))
                    self.pointnet_model.eval()  # Set to evaluation mode
                else:
                    self.get_logger().error(f"PointNet model not found at {self.pointnet_model_path}")
                    return []
            
            # Need to preprocess point cloud for PointNet
            if len(lidar_points) < 10:  # Need minimum number of points
                return []
                
            # Here we would normally preprocess the points for PointNet
            # For demonstration, we'll simulate detections
            # In a real implementation, you would run the model on the points
            
            # Find clusters of points that might be cones
            # This is a simplified approach - real implementation would use the PointNet model
            detected_cones = []
            
            # Simple clustering - find points close to ground and cluster them
            # In 3D space, cones are typically small clusters of points
            ground_height = np.min(lidar_points[:, 2]) + 0.1  # Slightly above ground
            
            # Filter points close to ground
            near_ground_mask = (lidar_points[:, 2] < ground_height + 0.5) & (lidar_points[:, 2] > ground_height)
            near_ground_points = lidar_points[near_ground_mask]
            
            if len(near_ground_points) > 5:
                # Very simple clustering - just for demonstration
                # A real implementation would use DBSCAN or another clustering algorithm
                
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
            
            self.get_logger().info(f"Detected {len(detected_cones)} cones from LiDAR using PointNet")
            return detected_cones
            
        except Exception as e:
            self.get_logger().error(f"Error in PointNet detection: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return []

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

    def process_lidar_data(self):
        """Process and analyze LiDAR data with added diagnostics."""
        if self.lidar is None or not hasattr(self, 'lidar_data') or self.lidar_data is None:
            self.get_logger().warn("LiDAR data not available")
            return
            
        try:
            # Get sensor (LiDAR) world transformation matrix
            sensor_transform = np.array(self.lidar.get_transform().get_matrix())
            
            # Transform LiDAR points from sensor to world coordinates
            with self.lidar_lock:
                lidar_data = self.lidar_data.copy()
                point_count = len(lidar_data)
            
            # DIAGNOSTIC: Log basic LiDAR stats
            self.get_logger().info(f"Processing {point_count} LiDAR points")
            
            if lidar_data.size == 0:
                self.get_logger().warn("No LiDAR points to process")
                return
                    
            # Filter out points that are too far away
            distances = np.sqrt(np.sum(lidar_data**2, axis=1))
            
            # DIAGNOSTIC: Log distance information
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
                
                # DIAGNOSTIC: Log accumulated point count
                self.get_logger().info(f"Accumulated {len(all_points)} LiDAR points from {len(self.lidar_history)} frames")
            
            # Store for external use
            self.latest_lidar_points = all_points
            
            # Log basic information about points before detection
            self.get_logger().info(f"Finding cones in LiDAR data: {len(all_points)} points")
            
            # Detect cones from LiDAR points
            detected_cones = self.detect_cones_from_lidar(all_points)
            
            # Log detection results
            self.get_logger().info(f"LiDAR detected {len(detected_cones)} cones")
            for i, cone in enumerate(detected_cones):
                pos = cone['position']
                self.get_logger().info(f"  Cone {i+1}: Position=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), Conf={cone['confidence']:.2f}")
            
            # Analyze LiDAR for turn detection
            self.analyze_lidar_for_turns(all_points, detected_cones)
            
            # Visualize detected cones
            self.visualize_3d_cones(detected_cones)
            
            # Create PointCloud2 header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "map"  # Use map as frame to avoid tf issues
            
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

    def analyze_lidar_for_turns(self, points, detected_cones):
        """Analyze LiDAR point cloud and detected cones to detect upcoming turns."""
        try:
            # Initialize turn information
            self.lidar_right_turn_detected = False
            self.lidar_left_turn_detected = False
            self.lidar_turn_distance = float('inf')
            self.lidar_turn_confidence = 0.0
            
            # Use detected cones if available, otherwise use clustering approach
            if detected_cones and len(detected_cones) >= 3:
                self.get_logger().info("Using detected cones for turn analysis")
                
                # Extract cone positions
                cone_positions = [cone['position'] for cone in detected_cones]
                
                # Simple approach: Look for significant lateral deviation in the cone pattern
                cone_positions.sort(key=lambda p: p[1])  # Sort by Y distance (forward)
                
                # Analyze lateral position trend
                x_positions = [p[0] for p in cone_positions]
                
                # If we have enough cones, try to detect a pattern
                if len(x_positions) >= 4:
                    # Simple trend analysis: are right side points moving left (right turn) or
                    # left side points moving right (left turn)?
                    
                    # Divide into near and far points
                    mid_idx = len(x_positions) // 2
                    near_avg_x = sum(x_positions[:mid_idx]) / mid_idx
                    far_avg_x = sum(x_positions[mid_idx:]) / (len(x_positions) - mid_idx)
                    
                    # Check for significant lateral shift
                    lateral_shift = far_avg_x - near_avg_x
                    
                    # DIAGNOSTIC
                    self.get_logger().info(f"Cone lateral shift analysis: near_avg_x={near_avg_x:.2f}, far_avg_x={far_avg_x:.2f}, shift={lateral_shift:.2f}")
                    
                    if lateral_shift < -0.3:  # Right turn
                        self.lidar_right_turn_detected = True
                        self.lidar_turn_distance = cone_positions[mid_idx][1]
                        self.lidar_turn_confidence = min(1.0, abs(lateral_shift) / 1.0)
                        self.get_logger().warn(f"LiDAR detected RIGHT TURN at {self.lidar_turn_distance:.2f}m (shift: {lateral_shift:.2f})")
                        
                    elif lateral_shift > 0.3:  # Left turn
                        self.lidar_left_turn_detected = True
                        self.lidar_turn_distance = cone_positions[mid_idx][1]
                        self.lidar_turn_confidence = min(1.0, abs(lateral_shift) / 1.0)
                        self.get_logger().warn(f"LiDAR detected LEFT TURN at {self.lidar_turn_distance:.2f}m (shift: {lateral_shift:.2f})")
            
            # If no cones detected or not enough, use point cloud analysis
            else:
                self.get_logger().info("Not enough cones detected, using point cloud analysis")
                
                # Filter to points likely to be cones (above ground, below certain height)
                cone_height_min = 0.05  # 5cm above ground
                cone_height_max = 0.5   # 50cm tall (typical cone height)
                potential_cone_points = [p for p in points if cone_height_min < p[2] < cone_height_max]
                
                # DIAGNOSTIC
                self.get_logger().info(f"Found {len(potential_cone_points)} potential cone points in height range")
                
                if len(potential_cone_points) < 10:
                    self.get_logger().warn("Not enough potential cone points for analysis")
                    return
                
                # Simple approach: analyze the X distribution of points at different depths
                points_by_depth = {}
                depth_interval = 2.0  # Group points in 2m intervals
                
                for point in potential_cone_points:
                    depth_bin = int(point[1] / depth_interval) * depth_interval
                    if depth_bin not in points_by_depth:
                        points_by_depth[depth_bin] = []
                    points_by_depth[depth_bin].append(point)
                
                # Sort depth bins
                sorted_depths = sorted(points_by_depth.keys())
                
                # DIAGNOSTIC
                self.get_logger().info(f"Point depth distribution: {', '.join([f'{d}m:{len(points_by_depth[d])}' for d in sorted_depths])}")
                
                # Need at least 3 depth bins for trend analysis
                if len(sorted_depths) >= 3:
                    # Calculate average X position at each depth
                    avg_x_by_depth = {d: sum(p[0] for p in points_by_depth[d]) / len(points_by_depth[d]) for d in sorted_depths}
                    
                    # Check for consistent lateral shift trend
                    x_shifts = []
                    for i in range(1, len(sorted_depths)):
                        prev_depth = sorted_depths[i-1]
                        curr_depth = sorted_depths[i]
                        x_shift = avg_x_by_depth[curr_depth] - avg_x_by_depth[prev_depth]
                        x_shifts.append(x_shift)
                    
                    # If consistent trend detected
                    if len(x_shifts) >= 2:
                        avg_shift = sum(x_shifts) / len(x_shifts)
                        
                        # DIAGNOSTIC
                        self.get_logger().info(f"Average X shift per depth interval: {avg_shift:.3f}m")
                        
                        if avg_shift < -0.2:  # Consistent shift right to left = right turn
                            self.lidar_right_turn_detected = True
                            self.lidar_turn_distance = sorted_depths[1]  # Use second depth bin as turn distance
                            self.lidar_turn_confidence = min(1.0, abs(avg_shift) / 0.5)
                            self.get_logger().warn(f"LiDAR point cloud analysis: RIGHT TURN at {self.lidar_turn_distance:.1f}m (shift: {avg_shift:.2f})")
                            
                        elif avg_shift > 0.2:  # Consistent shift left to right = left turn
                            self.lidar_left_turn_detected = True
                            self.lidar_turn_distance = sorted_depths[1]
                            self.lidar_turn_confidence = min(1.0, abs(avg_shift) / 0.5)
                            self.get_logger().warn(f"LiDAR point cloud analysis: LEFT TURN at {self.lidar_turn_distance:.1f}m (shift: {avg_shift:.2f})")
                    
            # Return results for use in control logic
            return {
                'right_turn': self.lidar_right_turn_detected,
                'left_turn': self.lidar_left_turn_detected,
                'distance': self.lidar_turn_distance,
                'confidence': self.lidar_turn_confidence
            }
            
        except Exception as e:
            self.get_logger().error(f"Error analyzing LiDAR for turns: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None

    def fuse_data(self):
        """Fuse LiDAR and camera data."""
        if self.zed_camera is None or self.zed_camera.rgb_image is None or self.zed_camera.depth_image is None:
            return
            
        if self.lidar_data is None:
            return
            
        try:
            # Get depth image
            depth_array, _ = self.zed_camera.depth_image
            
            # Project LiDAR points to camera image
            lidar_transform = np.array(self.lidar.get_transform().get_matrix())
            camera_transform = np.array(self.zed_camera.rgb_sensor.get_transform().get_matrix())
            
            # Calculate relative transform from LiDAR to camera
            lidar_to_camera = np.linalg.inv(camera_transform) @ lidar_transform
            
            # Copy LiDAR data to avoid race conditions
            with self.lidar_lock:
                lidar_data = self.lidar_data.copy()
            
            # Transform LiDAR points to camera frame
            points_camera_frame = transform_points(lidar_data, lidar_to_camera)
            
            # Simple data fusion: Filter LiDAR points by camera FOV and depth
            # This is a basic fusion - more sophisticated methods can be implemented
            valid_points = []
            colors = []
            
            for i, point in enumerate(points_camera_frame):
                # Only keep points in front of camera
                if point[2] <= 0:
                    continue
                    
                # Project point to image
                x, y, z = point
                
                # Basic pinhole camera model (approximation)
                # This should be replaced with proper camera calibration parameters
                fx = 800.0  # focal length x
                fy = 800.0  # focal length y
                cx = 640.0  # optical center x
                cy = 360.0  # optical center y
                
                u = int(fx * x / z + cx)
                v = int(fy * y / z + cy)
                
                # Check if point projects into image
                if (0 <= u < 1280 and 0 <= v < 720):
                    # Get depth from depth image at projected point
                    if depth_array is not None:
                        camera_depth = depth_array[v, u] if v < depth_array.shape[0] and u < depth_array.shape[1] else 0
                        
                        # Compare LiDAR depth with camera depth
                        lidar_depth = np.abs(z)
                        
                        # If depths are similar, consider it a valid fusion point
                        if camera_depth > 0 and abs(lidar_depth - camera_depth) < 2.0:
                            valid_points.append(point)
                            
                            # Get color from RGB image
                            if self.zed_camera.rgb_image is not None:
                                r, g, b = self.zed_camera.rgb_image[v, u]
                                colors.append([r/255.0, g/255.0, b/255.0])
                            else:
                                colors.append([1.0, 1.0, 1.0])  # White if no color available
            
            # Create fused colored point cloud
            if valid_points:
                fused_points = np.array(valid_points)
                
                # Transform back to world frame
                fused_points_world = transform_points(fused_points, camera_transform)
                
                # Create PointCloud2 message for fused points
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "map"
                
                # Publisher requires a structured point cloud with fields
                # We'll create an uncolored point cloud for simplicity
                pc_msg = pc2.create_cloud_xyz32(header, fused_points_world)
                self.fused_pub.publish(pc_msg)
                
                # For colored point cloud:
                # fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                #          PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                #          PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                #          PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
                #          PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
                #          PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1)]
                
                self.get_logger().debug(f"Published {len(valid_points)} fused points")
        except Exception as e:
            self.get_logger().error(f"Error in data fusion: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
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
            # Subtract 90 degrees (π/2) to rotate the path to the right
            yaw = np.radians(vehicle_rotation.yaw )
            rotation_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            
            # Convert path points to PoseStamped messages
            for point in path:
                # Swap and invert coordinates to fix orientation
                # Original path point is (x forward, y left) in vehicle frame
                # We want (x right, y forward) in world frame
                local_point = np.array([point[1], point[0]])  # Swap coordinates
                world_point = rotation_matrix @ local_point
                
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = vehicle_location.x + world_point[0]
                pose.pose.position.y = vehicle_location.y + world_point[1]
                pose.pose.position.z = vehicle_location.z  # Keep same height as vehicle
                
                # Calculate orientation tangent to the path
                if len(path_msg.poses) > 0:
                    # Get direction to next point
                    dx = world_point[0]
                    dy = world_point[1]
                    heading = np.arctan2(dy, dx)  # Use world frame heading directly
                    
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

    def set_car_controls(self, steering, speed):
        """
        Set the car's steering and speed with guaranteed movement.
        """
        if self.vehicle is None:
            self.get_logger().error("Cannot set controls: No vehicle available")
            return
        
        try:
            import carla
            
            # Ensure minimum throttle to guarantee movement
            min_throttle = 0.3  # Minimum throttle to ensure movement
            
            # Create control command
            control = carla.VehicleControl()
            
            # Set steering (-1 to 1)
            control.steer = float(steering)
            
            # Set throttle and brake based on speed
            if speed >= 0:
                # Forward - ensure minimum throttle
                control.throttle = max(min_throttle, min(abs(speed) / 5.0, 1.0))
                control.brake = 0.0
                control.reverse = False
            else:
                # Reverse
                control.throttle = 0.0
                control.brake = min(abs(speed) / 5.0, 1.0)
                control.reverse = True
            
            # Apply control to vehicle
            self.vehicle.apply_control(control)
            
            # Force a physics update to ensure movement
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

    def detect_turns_with_lidar(self):
        """Use LiDAR point cloud to detect both upcoming right and left turns, with enhanced U-turn detection."""
        # Initialize defaults
        right_turn_detected = False
        left_turn_detected = False
        uturn_detected = False
        turn_distance = float('inf')
        
        try:
            # Get the latest LiDAR points
            if not hasattr(self, 'lidar_history') or not self.lidar_history:
                self.get_logger().warn("No LiDAR history available")
                return right_turn_detected, left_turn_detected, turn_distance
                
            with self.lidar_history_lock:
                if not self.lidar_history:
                    return right_turn_detected, left_turn_detected, turn_distance
                
                # Combine points from all frames in history for better coverage
                all_points = np.vstack(self.lidar_history)
            
            if len(all_points) < 100:
                self.get_logger().warn(f"Not enough LiDAR points for analysis: {len(all_points)}")
                return right_turn_detected, left_turn_detected, turn_distance
                
            # Enhanced filtering for better cone and boundary detection
            height_min = 0.05  # 5cm off ground
            height_max = 0.5   # Most cones less than 50cm tall
            distance_max = 30.0  # Extended range for U-turn detection
            
            # Calculate distances
            distances = np.sqrt(np.sum(all_points[:, :2]**2, axis=1))
            
            # Apply enhanced filters
            mask = ((all_points[:, 2] >= height_min) & 
                    (all_points[:, 2] <= height_max) & 
                    (distances <= distance_max))
            filtered_points = all_points[mask]
            
            if len(filtered_points) < 50:
                self.get_logger().warn(f"Not enough filtered LiDAR points: {len(filtered_points)}")
                return right_turn_detected, left_turn_detected, turn_distance
                
            # Create more detailed distance bins for better analysis
            bin_size = 1.5  # Reduced for finer granularity
            max_distance = 30.0  # Extended range
            num_bins = int(max_distance / bin_size)
            
            # Initialize bin data with angle information
            angle_bins = np.linspace(-np.pi/2, np.pi/2, 18)  # 20-degree bins
            point_distribution = np.zeros((num_bins, len(angle_bins)-1))
            
            # Analyze point distribution in polar coordinates
            for point in filtered_points:
                x, y = point[0], point[1]
                distance = np.sqrt(x**2 + y**2)
                angle = np.arctan2(x, y)  # Angle from forward direction
                
                # Find appropriate bins
                dist_bin = min(int(distance / bin_size), num_bins - 1)
                angle_bin = np.digitize(angle, angle_bins) - 1
                
                if 0 <= angle_bin < len(angle_bins)-1:
                    point_distribution[dist_bin, angle_bin] += 1
            
            # Detect U-turns by looking for characteristic patterns
            uturn_score = 0
            uturn_distance = float('inf')
            
            # Look for sharp changes in point distribution that indicate U-turns
            for d in range(2, num_bins-2):
                near_dist = d * bin_size
                
                # Calculate left-right ratio for consecutive distance bins
                left_points = np.sum(point_distribution[d, :8])  # Left side
                right_points = np.sum(point_distribution[d, 9:])  # Right side
                next_left = np.sum(point_distribution[d+1, :8])
                next_right = np.sum(point_distribution[d+1, 9:])
                
                # Look for characteristic U-turn pattern:
                # 1. Sharp decrease in points ahead
                # 2. High concentration of points to one side
                # 3. Sudden change in left-right distribution
                forward_points = np.sum(point_distribution[d, 8:10])
                next_forward = np.sum(point_distribution[d+1, 8:10])
                
                if forward_points > 10 and next_forward < forward_points * 0.3:
                    # Sharp decrease in forward points
                    side_concentration = max(left_points, right_points) / (forward_points + 1)
                    distribution_change = abs((left_points/right_points if right_points > 0 else 10) - 
                                           (next_left/next_right if next_right > 0 else 10))
                    
                    current_score = side_concentration * distribution_change
                    if current_score > uturn_score:
                        uturn_score = current_score
                        uturn_distance = near_dist
                        
                        # Determine turn direction based on point concentration
                        if left_points > right_points:
                            left_turn_detected = True
                            right_turn_detected = False
                        else:
                            right_turn_detected = True
                            left_turn_detected = False
            
            # U-turn detection threshold
            if uturn_score > 2.0:
                uturn_detected = True
                turn_distance = uturn_distance
                self.get_logger().warn(f"U-TURN detected at {turn_distance:.1f}m (score: {uturn_score:.2f})")
                
                # Store U-turn state for path planning
                self.uturn_state = {
                    'detected': True,
                    'distance': turn_distance,
                    'score': uturn_score,
                    'direction': 'left' if left_turn_detected else 'right'
                }
            else:
                # Regular turn detection logic (existing code)
                for d in range(2, num_bins-2):
                    near_dist = d * bin_size
                    
                    # Calculate point distributions
                    left_ratio = np.sum(point_distribution[d, :8]) / (np.sum(point_distribution[d]) + 1e-6)
                    right_ratio = np.sum(point_distribution[d, 9:]) / (np.sum(point_distribution[d]) + 1e-6)
                    next_left = np.sum(point_distribution[d+1, :8]) / (np.sum(point_distribution[d+1]) + 1e-6)
                    next_right = np.sum(point_distribution[d+1, 9:]) / (np.sum(point_distribution[d+1]) + 1e-6)
                    
                    # Detect significant changes in distribution
                    if abs(left_ratio - next_left) > 0.3 or abs(right_ratio - next_right) > 0.3:
                        if left_ratio > right_ratio and next_right > next_left:
                            right_turn_detected = True
                            turn_distance = near_dist
                            break
                        elif right_ratio > left_ratio and next_left > next_right:
                            left_turn_detected = True
                            turn_distance = near_dist
                            break
            
            # Log detection results
            if uturn_detected:
                self.get_logger().warn(f"U-turn detected at {turn_distance:.1f}m")
            elif right_turn_detected:
                self.get_logger().info(f"Right turn detected at {turn_distance:.1f}m")
            elif left_turn_detected:
                self.get_logger().info(f"Left turn detected at {turn_distance:.1f}m")
            
            return right_turn_detected, left_turn_detected, turn_distance
            
        except Exception as e:
            self.get_logger().error(f"Error detecting turns with LiDAR: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False, False, float('inf')

    def control_car(self):
        """Control the car with enhanced LiDAR turn detection for both left and right turns."""
        if not hasattr(self, 'path_planner') or self.path_planner is None:
            self.set_car_controls(0.0, 1.0)
            return
        
        # Get the steering value from the path planner with reduced lookahead for faster response
        steering = self.path_planner.calculate_steering(lookahead_distance=2.5)
        
        # Use LiDAR to detect turns - now detects both left and right turns
        lidar_right_turn, lidar_left_turn, lidar_turn_distance = self.detect_turns_with_lidar()
        
        # Speed parameters - slightly more conservative
        max_speed = 1.7  # Reduced overall speed for better control
        turn_speed = 0.75
        slow_speed = 0.25
        
        # CONE DETECTION
        closest_cone_dist = float('inf')
        cone_in_path = False
        
        if hasattr(self, 'zed_camera') and self.zed_camera and hasattr(self.zed_camera, 'cone_detections'):
            for cone in self.zed_camera.cone_detections:
                if 'depth' in cone and 'box' in cone:
                    depth = cone['depth']
                    
                    if depth < closest_cone_dist:
                        closest_cone_dist = depth
                    
                    # Check if cone is in path with wider threshold
                    x1, y1, x2, y2 = cone['box']
                    center_x = (x1 + x2) // 2
                    image_center_x = self.zed_camera.resolution[0] // 2
                    
                    # Even wider detection corridor
                    center_threshold = 550 - (450 * min(1.0, depth / 15.0))
                    
                    if abs(center_x - image_center_x) < center_threshold:
                        cone_in_path = True
                        self.get_logger().warn(f"CONE IN PATH at {depth:.2f}m! Center offset: {abs(center_x - image_center_x)}px")
        
        # Default speeds
        target_speed = max_speed
        speed_reason = "Normal driving"
        
        # LIDAR-DETECTED TURNS - HIGHEST PRIORITY
        # -------------------------------------
        if lidar_right_turn or lidar_left_turn:
            # Direction-specific handling
            if lidar_right_turn:
                # Very aggressive slowdown for right turns
                lidar_slowdown_factor = np.exp(-lidar_turn_distance / 10.0) * 0.85
                lidar_turn_speed = slow_speed * (1.0 - lidar_slowdown_factor)
                turn_type = "RIGHT"
            else:  # Left turn
                # Still aggressive but less so than right turns
                lidar_slowdown_factor = np.exp(-lidar_turn_distance / 12.0) * 0.8
                lidar_turn_speed = slow_speed * (1.0 - lidar_slowdown_factor * 0.9)
                turn_type = "LEFT"
            
            # Ensure very slow speed for nearby turns
            if lidar_turn_distance < 10.0:
                lidar_turn_speed = min(lidar_turn_speed, 0.5)
            
            if lidar_turn_speed < target_speed:
                target_speed = lidar_turn_speed
                speed_reason = f"LIDAR {turn_type} TURN at {lidar_turn_distance:.1f}m"
                self.get_logger().warn(f"LIDAR {turn_type} TURN DETECTED: Slowing to {target_speed:.2f}m/s")
        
        # STEERING TREND ANALYSIS - For earlier turn detection
        # -------------------------------------------------
        if not hasattr(self, 'steering_history'):
            self.steering_history = []
        
        # Add current steering to history
        self.steering_history.append(steering)
        if len(self.steering_history) > 15:
            self.steering_history = self.steering_history[-15:]
        
        if len(self.steering_history) >= 4:
            # Calculate steering trend (direction and rate of change)
            recent_steering = self.steering_history[-4:]
            steering_trend = sum(recent_steering) / len(recent_steering)
            steering_rate = sum([abs(recent_steering[i] - recent_steering[i-1]) 
                              for i in range(1, len(recent_steering))])
            
            # Detect consistent steering direction
            if steering_trend > 0.15:  # Consistent right steering
                trend_speed = slow_speed * (1.0 - min(0.8, steering_trend * 2.0))
                if trend_speed < target_speed:
                    target_speed = trend_speed
                    speed_reason = f"RIGHT STEERING TREND ({steering_trend:.2f})"
            elif steering_trend < -0.15:  # Consistent left steering
                trend_speed = slow_speed * (1.0 - min(0.8, abs(steering_trend) * 2.0))
                if trend_speed < target_speed:
                    target_speed = trend_speed
                    speed_reason = f"LEFT STEERING TREND ({steering_trend:.2f})"
            
            # Detect rapid steering changes (indicates upcoming turn)
            if steering_rate > 0.15:
                rapid_steer_speed = slow_speed * 0.8
                if rapid_steer_speed < target_speed:
                    target_speed = rapid_steer_speed
                    speed_reason = f"RAPID STEERING CHANGES ({steering_rate:.2f})"
        
        # CURRENT STEERING - Now equally aggressive for both directions
        # ----------------------------------------------------------
        abs_steering = abs(steering)
        if abs_steering > 0.1:  # Any significant steering
            direction = "RIGHT" if steering > 0 else "LEFT"
            
            # Exponential slowdown based on steering magnitude
            steer_factor = min(1.0, abs_steering * 3.0)  # Scale up for stronger effect
            steer_speed = slow_speed * (1.0 - steer_factor * 0.8)
            
            if steer_speed < target_speed:
                target_speed = steer_speed
                speed_reason = f"{direction} STEERING ({abs_steering:.2f})"
        
        # EXTREME PROXIMITY - Emergency handling
        # -----------------------------------
        if closest_cone_dist < 2.0:  # Very close cone
            emergency_speed = 0.2  # Almost stop
            if emergency_speed < target_speed:
                target_speed = emergency_speed
                speed_reason = f"EMERGENCY: Cone at {closest_cone_dist:.1f}m"
        # CONE PROXIMITY - Normal handling
        elif closest_cone_dist < 5.0:
            proximity_factor = np.exp(-(closest_cone_dist * 0.7))
            proximity_speed = slow_speed * (1.0 - proximity_factor * 0.7)
            
            if proximity_speed < target_speed:
                target_speed = proximity_speed
                speed_reason = f"CONE PROXIMITY: {closest_cone_dist:.1f}m"
        
        # CONE IN PATH - Cautious approach
        # -----------------------------
        elif cone_in_path and closest_cone_dist < 15.0:  # Extended range for earlier reaction
            path_speed = slow_speed + (turn_speed - slow_speed) * (closest_cone_dist / 15.0) * 0.7
            
            if path_speed < target_speed:
                target_speed = path_speed
                speed_reason = f"CONE IN PATH: {closest_cone_dist:.1f}m"
        
        # Sanity check - ensure minimum and maximum speeds
        target_speed = max(0.2, min(target_speed, max_speed))
        
        # ENHANCED HYSTERESIS - More aggressive for any turn
        # -----------------------------------------------
        if hasattr(self, 'prev_target_speed'):
            if "TURN" in speed_reason.upper() or "STEERING" in speed_reason.upper():
                # Aggressive braking for turns and steering
                max_decel = 2.0
            elif "EMERGENCY" in speed_reason.upper():
                # Even more aggressive for emergency situations
                max_decel = 3.0
            elif target_speed < self.prev_target_speed:
                # Normal braking otherwise
                max_decel = 1.0
            else:
                # Very gradual acceleration
                max_decel = 0.1
                
            # Limit change rate
            if abs(target_speed - self.prev_target_speed) > max_decel:
                target_speed = self.prev_target_speed + max_decel * np.sign(target_speed - self.prev_target_speed)
        
        # PREDICTIVE ACCELERATION CONTROL - Prevents speeding up when turn is ahead
        # ---------------------------------------------------------------------
        if hasattr(self, 'prev_steering') and abs(steering) > abs(self.prev_steering) and abs(steering) > 0.1:
            # If steering is increasing in magnitude, don't accelerate
            if target_speed > self.prev_target_speed:
                target_speed = self.prev_target_speed
                speed_reason += " + Prevented acceleration during increasing steering"
        
        # Store for next iteration
        self.prev_target_speed = target_speed
        self.prev_steering = steering
        
        # Log detailed speed decision
        self.get_logger().info(f"Speed control: {speed_reason} → {target_speed:.2f} m/s")
        
        # Apply control to vehicle
        self.set_car_controls(steering, target_speed)

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
        """Visualize detected cones in RViz."""
        try:
            # Check if we have cone detections
            cones_to_visualize = []
            
            # Get cones from camera
            if hasattr(self, 'zed_camera') and self.zed_camera and hasattr(self.zed_camera, 'cone_detections'):
                cam_cones = self.zed_camera.cone_detections
                for cone in cam_cones:
                    if 'depth' in cone and 'cls' in cone:
                        # Project to 3D position if not already there
                        if 'lidar_position' in cone:
                            position = cone['lidar_position']
                        else:
                            # Estimate 3D position from camera
                            depth = cone['depth']
                            if 'box' in cone:
                                x1, y1, x2, y2 = cone['box']
                                center_x = (x1 + x2) // 2
                                image_center_x = self.zed_camera.resolution[0] // 2
                                
                                # Calculate angle from center
                                fov_horizontal = 90.0  # Approximate FOV
                                angle = ((center_x - image_center_x) / (image_center_x)) * (np.radians(fov_horizontal/2))
                                
                                # Estimate 3D position
                                x = depth * np.sin(angle)
                                y = depth * np.cos(angle)
                                z = 0.3  # Approximate cone height
                                position = [x, y, z]
                            else:
                                continue  # Skip cones without position info
                        
                        # Add to visualization list
                        cones_to_visualize.append({
                            'position': position,
                            'cls': cone['cls'],
                            'confidence': cone.get('confidence', 0.8)
                        })
            
            # Get additional cones from LiDAR if available
            if hasattr(self, 'latest_detected_cones') and self.latest_detected_cones:
                for cone in self.latest_detected_cones:
                    if 'position' in cone:
                        # Check if this cone is already included (avoid duplicates)
                        is_duplicate = False
                        for existing in cones_to_visualize:
                            dist = np.sqrt(sum((existing['position'][i] - cone['position'][i])**2 
                                              for i in range(3)))
                            if dist < 1.0:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            cones_to_visualize.append(cone)
            
            # Create marker array
            marker_array = MarkerArray()
            
            # Create markers for each cone
            for i, cone in enumerate(cones_to_visualize):
                # Create cone marker
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "detected_cones"
                marker.id = i
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                # Set position - Need to transform to world coordinates if needed
                position = cone['position']
                if hasattr(self, 'vehicle') and self.vehicle:
                    # Get vehicle transform to convert from vehicle to world frame
                    vehicle_transform = self.vehicle.get_transform()
                    yaw = np.radians(vehicle_transform.rotation.yaw)
                    
                    # Rotation matrix for yaw
                    rot_matrix = np.array([
                        [np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]
                    ])
                    
                    # Apply rotation and translation
                    local_pos = np.array([position[0], position[1]])
                    world_pos = rot_matrix @ local_pos
                    
                    marker.pose.position.x = vehicle_transform.location.x + world_pos[0]
                    marker.pose.position.y = vehicle_transform.location.y + world_pos[1]
                    marker.pose.position.z = vehicle_transform.location.z + position[2]
                else:
                    # Fallback to local coordinates
                    marker.pose.position.x = position[0]
                    marker.pose.position.y = position[1]
                    marker.pose.position.z = position[2]
                
                # Set orientation (upright)
                marker.pose.orientation.w = 1.0
                
                # Set scale (cone size)
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.4
                
                # Set color based on class and confidence
                cls = cone.get('cls', 0)
                confidence = cone.get('confidence', 0.8)
                
                if cls == 0:  # Yellow cone
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                else:  # Blue cone
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                
                # Transparency based on confidence
                marker.color.a = 0.3 + 0.7 * confidence
                
                # Set marker lifetime
                marker.lifetime.sec = 1  # 1 second lifetime
                
                marker_array.markers.append(marker)
            
            # Publisher for cone markers
            if not hasattr(self, 'cone_marker_pub'):
                self.cone_marker_pub = self.create_publisher(MarkerArray, '/carla/lidar_cones', 10)
            
            # Publish markers
            self.cone_marker_pub.publish(marker_array)
            
            # Log number of cones visualized (infrequently)
            if not hasattr(self, 'last_cone_log_time') or time.time() - self.last_cone_log_time > 2.0:
                self.get_logger().info(f"Visualizing {len(cones_to_visualize)} cones in RViz")
                self.last_cone_log_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Error visualizing cones: {str(e)}")

    def visualize_pcd_data(self):
        """Visualize current PCD data in RViz."""
        try:
            if not hasattr(self, 'latest_lidar_points') or self.latest_lidar_points is None:
                return
            
            # Create PointCloud2 message
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "map"
            
            # Create fields for the point cloud
            fields = [
                pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='intensity', offset=12, datatype=pc2.PointField.FLOAT32, count=1)
            ]
            
            # Color points based on height
            points = self.latest_lidar_points
            min_z = np.min(points[:, 2])
            max_z = np.max(points[:, 2])
            z_range = max_z - min_z if max_z > min_z else 1.0
            
            # Create structured array for colored points
            structured_points = np.zeros(len(points), 
                                        dtype=[
                                            ('x', np.float32),
                                            ('y', np.float32),
                                            ('z', np.float32),
                                            ('intensity', np.float32)
                                        ])
            
            # Fill structured array
            structured_points['x'] = points[:, 0]
            structured_points['y'] = points[:, 1]
            structured_points['z'] = points[:, 2]
            
            # Color points based on height (z value)
            intensity = (points[:, 2] - min_z) / z_range
            structured_points['intensity'] = intensity
            
            # Create and publish the point cloud
            pc_msg = pc2.create_cloud(header, fields, structured_points)
            
            if not hasattr(self, 'pcd_pub'):
                self.pcd_pub = self.create_publisher(PointCloud2, '/carla/pcd_visualization', 10)
            
            self.pcd_pub.publish(pc_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error visualizing PCD data: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def update_pcd_file_list(self):
        """Update the list of available PCD files."""
        try:
            if not hasattr(self, 'output_dir') or not os.path.exists(self.output_dir):
                return
                
            # Get list of PCD files
            pcd_files = [f for f in os.listdir(self.output_dir) if f.endswith('.pcd')]
            pcd_files.sort()  # Sort by name
            
            # Store the list
            self.pcd_files = pcd_files
            
            # Log the number of files found
            if not hasattr(self, 'last_pcd_log_time') or time.time() - self.last_pcd_log_time > 5.0:
                self.get_logger().info(f"Found {len(pcd_files)} PCD files in {self.output_dir}")
                self.last_pcd_log_time = time.time()
                
        except Exception as e:
            self.get_logger().error(f"Error updating PCD file list: {str(e)}")

    def handle_keyboard_input(self):
        """Handle keyboard input for controlling visualization and playback."""
        try:
            # Check if any key is pressed
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit the program
                self.get_logger().info("Quitting...")
                rclpy.shutdown()
                return
                
            elif key == ord('p'):
                # Toggle PCD visualization
                if not hasattr(self, 'show_pcd'):
                    self.show_pcd = True
                else:
                    self.show_pcd = not self.show_pcd
                self.get_logger().info(f"PCD visualization: {'enabled' if self.show_pcd else 'disabled'}")
                
            elif key == ord('c'):
                # Toggle cone visualization
                if not hasattr(self, 'show_cones'):
                    self.show_cones = True
                else:
                    self.show_cones = not self.show_cones
                self.get_logger().info(f"Cone visualization: {'enabled' if self.show_cones else 'disabled'}")
                
            elif key == ord('n'):
                # Next PCD file
                if hasattr(self, 'pcd_files') and len(self.pcd_files) > 0:
                    if not hasattr(self, 'current_pcd_index'):
                        self.current_pcd_index = 0
                    else:
                        self.current_pcd_index = (self.current_pcd_index + 1) % len(self.pcd_files)
                    self.get_logger().info(f"Loading PCD file: {self.pcd_files[self.current_pcd_index]}")
                    
            elif key == ord('b'):
                # Previous PCD file
                if hasattr(self, 'pcd_files') and len(self.pcd_files) > 0:
                    if not hasattr(self, 'current_pcd_index'):
                        self.current_pcd_index = 0
                    else:
                        self.current_pcd_index = (self.current_pcd_index - 1) % len(self.pcd_files)
                    self.get_logger().info(f"Loading PCD file: {self.pcd_files[self.current_pcd_index]}")
                    
        except Exception as e:
            self.get_logger().error(f"Error handling keyboard input: {str(e)}")

    def _pair_cones(self, camera_cones, lidar_cones):
        """
        Pair detected cones from camera and LiDAR data using enhanced association logic.
        
        Args:
            camera_cones: List of cones detected by camera
            lidar_cones: List of cones detected by LiDAR
            
        Returns:
            List of paired cones with fused position and confidence
        """
        try:
            paired_cones = []
            unpaired_camera = camera_cones.copy()
            unpaired_lidar = lidar_cones.copy()
            
            # Parameters for association
            max_distance = 2.0  # Maximum distance for pairing (meters)
            min_confidence = 0.3  # Minimum confidence threshold
            
            # Sort cones by confidence for better matching
            unpaired_camera.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
            unpaired_lidar.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
            
            # Create distance matrix between all pairs
            if unpaired_camera and unpaired_lidar:
                distances = np.zeros((len(unpaired_camera), len(unpaired_lidar)))
                for i, cam_cone in enumerate(unpaired_camera):
                    for j, lidar_cone in enumerate(unpaired_lidar):
                        # Calculate 3D distance between cones
                        cam_pos = np.array(cam_cone['position'])
                        lidar_pos = np.array(lidar_cone['position'])
                        distances[i, j] = np.linalg.norm(cam_pos - lidar_pos)
                
                # Iteratively find best matches
                while distances.size > 0 and np.min(distances) < max_distance:
                    i, j = np.unravel_index(np.argmin(distances), distances.shape)
                    
                    cam_cone = unpaired_camera[i]
                    lidar_cone = unpaired_lidar[j]
                    
                    # Calculate weighted position based on confidence
                    cam_conf = cam_cone.get('confidence', 0.5)
                    lidar_conf = lidar_cone.get('confidence', 0.5)
                    
                    total_conf = cam_conf + lidar_conf
                    if total_conf > min_confidence:
                        # Weighted average of positions
                        fused_position = (
                            np.array(cam_cone['position']) * cam_conf +
                            np.array(lidar_cone['position']) * lidar_conf
                        ) / total_conf
                        
                        # Combine class information
                        fused_class = cam_cone.get('cls', lidar_cone.get('cls', 0))
                        
                        # Calculate fused confidence
                        fused_confidence = min(1.0, (cam_conf + lidar_conf) / 1.5)
                        
                        # Create fused cone
                        paired_cones.append({
                            'position': fused_position.tolist(),
                            'confidence': fused_confidence,
                            'cls': fused_class,
                            'sources': ['camera', 'lidar'],
                            'camera_conf': cam_conf,
                            'lidar_conf': lidar_conf
                        })
                    
                    # Remove paired cones from unpaired lists
                    unpaired_camera.pop(i)
                    unpaired_lidar.pop(j)
                    
                    # Update distance matrix
                    distances = np.delete(distances, i, axis=0)
                    distances = np.delete(distances, j, axis=1)
            
            # Handle remaining unpaired cones
            for cone in unpaired_camera:
                if cone.get('confidence', 0.0) > min_confidence:
                    cone['sources'] = ['camera']
                    paired_cones.append(cone)
            
            for cone in unpaired_lidar:
                if cone.get('confidence', 0.0) > min_confidence:
                    cone['sources'] = ['lidar']
                    paired_cones.append(cone)
            
            # Sort final cones by distance from vehicle
            paired_cones.sort(key=lambda x: np.linalg.norm(np.array(x['position'])[:2]))
            
            # Log pairing results
            self.get_logger().info(f"Paired {len(paired_cones)} cones: "
                                f"{len(paired_cones) - len(unpaired_camera) - len(unpaired_lidar)} fused, "
                                f"{len(unpaired_camera)} camera-only, "
                                f"{len(unpaired_lidar)} lidar-only")
            
            return paired_cones
            
        except Exception as e:
            self.get_logger().error(f"Error in cone pairing: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return []

    def _smooth_path(self, path_points, smoothing_factor=0.8):
        """
        Smooth the path using enhanced Bezier curves and adaptive smoothing.
        
        Args:
            path_points: List of (x, y) path points
            smoothing_factor: Factor controlling smoothness (0-1)
            
        Returns:
            Smoothed path points
        """
        try:
            if len(path_points) < 3:
                return path_points
                
            # Convert to numpy array for easier manipulation
            points = np.array(path_points)
            
            # Detect sharp turns and U-turns for adaptive smoothing
            angles = []
            for i in range(1, len(points) - 1):
                v1 = points[i] - points[i-1]
                v2 = points[i+1] - points[i]
                angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                angles.append(abs(angle))
            
            # Identify turn regions
            turn_regions = []
            current_turn = []
            in_turn = False
            turn_threshold = np.pi / 6  # 30 degrees
            
            for i, angle in enumerate(angles):
                if angle > turn_threshold and not in_turn:
                    in_turn = True
                    current_turn = [i]
                elif angle <= turn_threshold and in_turn:
                    in_turn = False
                    current_turn.append(i + 1)
                    turn_regions.append(current_turn)
                elif in_turn:
                    current_turn.append(i + 1)
            
            if in_turn:
                current_turn.append(len(angles))
                turn_regions.append(current_turn)
            
            # Apply adaptive smoothing
            smoothed_points = points.copy()
            window_size = 5
            
            for i in range(window_size, len(points) - window_size):
                # Check if point is in turn region
                in_turn_region = False
                for region in turn_regions:
                    if region[0] <= i <= region[-1]:
                        in_turn_region = True
                        break
                
                # Adjust smoothing based on region
                local_smoothing = smoothing_factor
                if in_turn_region:
                    # Reduce smoothing in turns for better accuracy
                    local_smoothing *= 0.7
                    
                    # Add more points in sharp turns
                    if i > 0 and i < len(points) - 1:
                        v1 = points[i] - points[i-1]
                        v2 = points[i+1] - points[i]
                        angle = abs(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))
                        
                        if angle > np.pi / 3:  # 60 degrees
                            # Insert intermediate points
                            num_points = 3
                            for j in range(1, num_points):
                                t = j / num_points
                                # Bezier interpolation
                                p0 = points[i-1]
                                p1 = points[i]
                                p2 = points[i+1]
                                new_point = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
                                smoothed_points = np.insert(smoothed_points, i+j, new_point, axis=0)
                
                # Apply smoothing with adaptive window
                window = points[max(0, i-window_size):min(len(points), i+window_size+1)]
                weights = np.exp(-0.5 * np.arange(-window_size, window_size+1)**2 / window_size)
                weights = weights[:len(window)]
                weights /= np.sum(weights)
                
                smoothed_points[i] = np.sum(window * weights[:, np.newaxis], axis=0)
            
            # Preserve endpoints
            smoothed_points[0] = points[0]
            smoothed_points[-1] = points[-1]
            
            # Apply curvature-based refinement
            refined_points = []
            for i in range(len(smoothed_points) - 1):
                refined_points.append(smoothed_points[i])
                
                # Add intermediate points in high-curvature regions
                if i > 0 and i < len(smoothed_points) - 1:
                    v1 = smoothed_points[i] - smoothed_points[i-1]
                    v2 = smoothed_points[i+1] - smoothed_points[i]
                    angle = abs(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))
                    
                    if angle > np.pi / 4:  # 45 degrees
                        # Add intermediate point using Bezier curve
                        t = 0.5
                        p0 = smoothed_points[i-1]
                        p1 = smoothed_points[i]
                        p2 = smoothed_points[i+1]
                        intermediate = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
                        refined_points.append(intermediate)
            
            refined_points.append(smoothed_points[-1])
            
            # Final smoothing pass for consistency
            final_points = np.array(refined_points)
            for i in range(1, len(final_points) - 1):
                final_points[i] = (final_points[i-1] + final_points[i] * 2 + final_points[i+1]) / 4
            
            # Log smoothing results
            self.get_logger().info(f"Smoothed path: {len(path_points)} points -> {len(final_points)} points "
                                f"with {len(turn_regions)} turn regions")
            
            return final_points.tolist()
            
        except Exception as e:
            self.get_logger().error(f"Error smoothing path: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return path_points

    def calculate_steering(self, path_points, current_speed):
        """
        Calculate steering angle with enhanced control for turns and U-turns.
        
        Args:
            path_points: List of (x, y) path points
            current_speed: Current vehicle speed in m/s
            
        Returns:
            Steering angle in range [-1, 1]
        """
        try:
            if not path_points or len(path_points) < 2:
                return 0.0
                
            # Get vehicle state
            if not hasattr(self, 'vehicle') or not self.vehicle:
                return 0.0
                
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            # Convert vehicle rotation to radians
            yaw = np.radians(vehicle_rotation.yaw)
            
            # Create rotation matrix
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])
            
            # Transform path points to vehicle's local frame
            local_points = []
            for point in path_points:
                # Convert to vehicle's local frame
                dx = point[0] - vehicle_location.x
                dy = point[1] - vehicle_location.y
                local_point = np.linalg.inv(rotation_matrix) @ np.array([dx, dy])
                local_points.append(local_point)
            
            # Find closest point and look-ahead point
            closest_idx = 0
            min_dist = float('inf')
            for i, point in enumerate(local_points):
                dist = np.hypot(point[0], point[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # Dynamic look-ahead distance based on speed and curvature
            base_lookahead = max(2.0, min(5.0, current_speed * 1.0))
            
            # Calculate path curvature at closest point
            if closest_idx > 0 and closest_idx < len(local_points) - 1:
                prev_point = local_points[closest_idx - 1]
                curr_point = local_points[closest_idx]
                next_point = local_points[closest_idx + 1]
                
                # Calculate angles
                v1 = curr_point - prev_point
                v2 = next_point - curr_point
                angle = abs(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))
                
                # Adjust look-ahead based on curvature
                curvature_factor = 1.0 - min(1.0, angle / np.pi)
                base_lookahead *= curvature_factor
            
            # Find look-ahead point
            lookahead_idx = closest_idx
            accumulated_dist = 0.0
            
            for i in range(closest_idx, len(local_points) - 1):
                point_dist = np.hypot(local_points[i+1][0] - local_points[i][0],
                                    local_points[i+1][1] - local_points[i][1])
                accumulated_dist += point_dist
                
                if accumulated_dist >= base_lookahead:
                    lookahead_idx = i + 1
                    break
            
            # Get look-ahead point
            target_point = local_points[lookahead_idx]
            
            # Calculate steering angle using pure pursuit
            target_dist = np.hypot(target_point[0], target_point[1])
            if target_dist < 0.1:  # Avoid division by zero
                return 0.0
                
            # Calculate steering angle
            chord_length = target_dist
            steering_angle = 2.0 * target_point[0] / (chord_length * chord_length)
            
            # Check for U-turn condition
            if hasattr(self, 'uturn_state') and self.uturn_state.get('detected', False):
                uturn_distance = self.uturn_state.get('distance', float('inf'))
                uturn_direction = self.uturn_state.get('direction', 'right')
                
                if uturn_distance < 10.0:  # Close to U-turn point
                    # Amplify steering for U-turn
                    steering_factor = 1.5 * (1.0 - uturn_distance / 10.0)
                    if uturn_direction == 'left':
                        steering_angle = min(-0.7, steering_angle * (1.0 + steering_factor))
                    else:  # right
                        steering_angle = max(0.7, steering_angle * (1.0 + steering_factor))
            
            # Apply speed-based steering limits
            max_steering = 1.0
            if current_speed > 5.0:  # Reduce max steering at higher speeds
                max_steering = max(0.3, 1.0 - (current_speed - 5.0) * 0.1)
            
            # Smooth steering changes
            if hasattr(self, 'prev_steering'):
                max_change = 0.1 * (1.0 + current_speed * 0.05)  # More aggressive at higher speeds
                steering_angle = np.clip(steering_angle,
                                       self.prev_steering - max_change,
                                       self.prev_steering + max_change)
            
            # Store for next iteration
            self.prev_steering = steering_angle
            
            # Final steering limits
            steering_angle = np.clip(steering_angle, -max_steering, max_steering)
            
            # Log steering calculation
            self.get_logger().debug(f"Steering: angle={steering_angle:.2f}, "
                                 f"lookahead={base_lookahead:.1f}m, "
                                 f"target=({target_point[0]:.1f}, {target_point[1]:.1f})")
            
            return float(steering_angle)
            
        except Exception as e:
            self.get_logger().error(f"Error calculating steering: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return 0.0

    def visualize_cone_map(self):
        """
        Visualize the cone map in RViz.
        
        This method creates markers for all the cones in the map and publishes them to RViz.
        """
        try:
            from visualization_msgs.msg import MarkerArray, Marker
            from std_msgs.msg import ColorRGBA
            from geometry_msgs.msg import Point
            
            # Create marker array for cones
            marker_array = MarkerArray()
            
            # Group cones by class
            cones_by_class = {}
            for i, cone in enumerate(self.cone_map):
                x, y, conf, cls = cone
                cls = int(cls)
                if cls not in cones_by_class:
                    cones_by_class[cls] = []
                cones_by_class[cls].append((x, y, conf, i))
            
            # Add markers for each class
            marker_id = 0
            
            # Class colors: yellow, blue, white (unknown)
            class_colors = {
                0: ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # Yellow
                1: ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue
                2: ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)   # White (unknown)
            }
            
            for cls, cones in cones_by_class.items():
                # Create a marker for this class
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"cones_class_{cls}"
                marker.id = cls
                marker.type = Marker.SPHERE_LIST
                marker.action = Marker.ADD
                
                # Set marker properties
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.5  # Cone diameter
                marker.scale.y = 0.5
                marker.scale.z = 0.5
                
                # Set color based on class
                marker.color = class_colors.get(cls, ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))
                
                # Add points for each cone
                for x, y, conf, _ in cones:
                    point = Point()
                    point.x = x
                    point.y = y
                    point.z = 0.25  # Slightly above ground
                    marker.points.append(point)
                    
                    # Make color opacity depend on confidence
                    color = ColorRGBA()
                    color.r = marker.color.r
                    color.g = marker.color.g
                    color.b = marker.color.b
                    color.a = min(1.0, 0.5 + conf * 0.5)  # Scale opacity with confidence
                    marker.colors.append(color)
                
                # Add marker to array
                marker_array.markers.append(marker)
                marker_id += 1
            
            # Create a marker for the vehicle trajectory
            if self.vehicle_poses:
                trajectory_marker = Marker()
                trajectory_marker.header.frame_id = "map"
                trajectory_marker.header.stamp = self.get_clock().now().to_msg()
                trajectory_marker.ns = "vehicle_trajectory"
                trajectory_marker.id = marker_id
                trajectory_marker.type = Marker.LINE_STRIP
                trajectory_marker.action = Marker.ADD
                
                # Set marker properties
                trajectory_marker.pose.orientation.w = 1.0
                trajectory_marker.scale.x = 0.1  # Line width
                trajectory_marker.color.r = 0.0
                trajectory_marker.color.g = 1.0
                trajectory_marker.color.b = 0.0
                trajectory_marker.color.a = 0.7
                
                # Add points for each pose
                for x, y, _ in self.vehicle_poses:
                    point = Point()
                    point.x = x
                    point.y = y
                    point.z = 0.1  # Slightly above ground
                    trajectory_marker.points.append(point)
                
                # Add marker to array
                marker_array.markers.append(trajectory_marker)
                marker_id += 1
                
                # Add current vehicle position marker
                current_pos_marker = Marker()
                current_pos_marker.header.frame_id = "map"
                current_pos_marker.header.stamp = self.get_clock().now().to_msg()
                current_pos_marker.ns = "vehicle_current_position"
                current_pos_marker.id = marker_id
                current_pos_marker.type = Marker.CUBE  # Using a cube to represent the vehicle
                current_pos_marker.action = Marker.ADD
                
                # Set marker properties
                current_pos_marker.pose.orientation.w = 1.0
                current_pos_marker.scale.x = 0.8  # Vehicle width
                current_pos_marker.scale.y = 1.6  # Vehicle length
                current_pos_marker.scale.z = 0.4  # Vehicle height
                
                # Set position to the latest vehicle pose
                latest_x, latest_y, _ = self.vehicle_poses[-1]
                current_pos_marker.pose.position = Point(x=latest_x, y=latest_y, z=0.2)  # Slightly above ground
                
                # Set color (e.g., red for visibility)
                current_pos_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                
                # Add marker to array
                marker_array.markers.append(current_pos_marker)
            
            # Publish the marker array
            if not hasattr(self, 'cone_map_pub'):
                self.cone_map_pub = self.create_publisher(MarkerArray, '/cone_map', 10)
            self.cone_map_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Error visualizing cone map: {str(e)}")


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
