import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import os
from datetime import datetime
import cv2

class ConeMapper:
    """
    SLAM-based cone mapping system to track cone positions and build a global map.
    """
    def __init__(self, max_distance=50.0, fusion_threshold=1.0, output_dir="./mapping_results"):
        """
        Initialize the cone mapper.
        
        Args:
            max_distance: Maximum distance in meters to track cones
            fusion_threshold: Distance threshold in meters to fuse cone detections
            output_dir: Directory for visualization outputs
        """
        # Global map of cones, stored as [[x, y, confidence, class], ...]
        # Class: 0 = yellow, 1 = blue, 2 = unknown
        self.cone_map = []
        
        # Configuration
        self.max_distance = max_distance
        self.fusion_threshold = fusion_threshold
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # History of vehicle poses for path optimization
        self.vehicle_poses = []  # List of (x, y, yaw) in world frame
        
        # Track statistics for mapping quality
        self.total_updates = 0
        self.total_new_cones = 0
        self.total_updated_cones = 0
        
        # KDTree for efficient cone lookups (rebuilt on updates)
        self.cone_tree = None
        
        # Store timestamp of last map update
        self.last_update_time = 0
        
        # Store history of planned paths for visualization
        self.planned_paths = []
        
        # Confidence grid for heatmap visualization
        self.confidence_grid = {}  # (x, y) -> confidence
        
        # Enhanced clustering parameters
        self.cluster_eps = 0.8  # Slightly tighter clustering
        self.cluster_min_samples = 2  # More sensitive clustering
        
        # Track sensor-specific cone detections
        self.camera_cones = []
        self.lidar_cones = []
        
        # Visualization parameters
        self.enable_visualization = True
        self.visualization_interval = 10  # Update visualization every N updates
        
    def update_vehicle_pose(self, x, y, yaw):
        """
        Update the current vehicle pose.
        
        Args:
            x, y: Position in world coordinates
            yaw: Orientation in radians
        """
        self.vehicle_poses.append((x, y, yaw))
        
        # Keep only the last 100 poses to limit memory usage
        if len(self.vehicle_poses) > 100:
            self.vehicle_poses = self.vehicle_poses[-100:]
    
    def update_map(self, detections, vehicle_pose, lidar_points=None, current_time=None):
        """Update the map with new cone detections and LiDAR data."""
        if current_time is None:
            current_time = self.last_update_time + 1
        self.last_update_time = current_time
        self.update_vehicle_pose(*vehicle_pose)

        veh_x, veh_y, veh_yaw = vehicle_pose
        cos_yaw = np.cos(veh_yaw)
        sin_yaw = np.sin(veh_yaw)
        
        # Process LiDAR-based cones if available
        lidar_cones = []
        if lidar_points is not None and len(lidar_points) > 0:
            # Cluster LiDAR points to find potential cones
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(lidar_points[:, :3])
            labels = clustering.labels_
            for label in set(labels) - {-1}:  # Exclude noise
                cluster = lidar_points[labels == label]
                center = np.mean(cluster, axis=0)
                dist = np.sqrt(center[0]**2 + center[1]**2)
                if dist < self.max_distance:
                    # Transform to world frame
                    world_x = center[1] * cos_yaw - center[0] * sin_yaw + veh_x  # y is forward
                    world_y = center[1] * sin_yaw + center[0] * cos_yaw + veh_y  # x is lateral
                    lidar_cones.append([world_x, world_y, 0.8, 2])  # Unknown class initially

        # Process camera detections
        camera_detections = []
        for x, y, conf, cls in detections or []:
            world_x = x * cos_yaw - y * sin_yaw + veh_x
            world_y = x * sin_yaw + y * cos_yaw + veh_y
            dist = np.sqrt((world_x - veh_x)**2 + (world_y - veh_y)**2)
            if dist < self.max_distance:
                camera_detections.append([world_x, world_y, conf, cls])

        # Fuse LiDAR and camera data
        all_detections = lidar_cones + camera_detections
        new_cones = 0

        if not self.cone_tree and self.cone_map:
            self.cone_tree = KDTree(np.array([[c[0], c[1]] for c in self.cone_map]))

        for detection in all_detections:
            x, y, conf, cls = detection
            if self.cone_tree and len(self.cone_map) > 0:
                dist, idx = self.cone_tree.query([x, y])
                if dist < self.fusion_threshold:
                    existing = self.cone_map[idx]
                    total_conf = existing[2] + conf
                    w_new = conf / total_conf
                    w_old = existing[2] / total_conf
                    updated_x = w_old * existing[0] + w_new * x
                    updated_y = w_old * existing[1] + w_new * y
                    updated_cls = cls if conf > existing[2] else existing[3]
                    self.cone_map[idx] = [updated_x, updated_y, total_conf, updated_cls]
                    self.total_updated_cones += 1
                    continue
            self.cone_map.append([x, y, conf, cls])
            new_cones += 1
            self.total_new_cones += 1

        if self.cone_map:
            self.cone_tree = KDTree(np.array([[c[0], c[1]] for c in self.cone_map]))
        
        self.total_updates += 1
        return new_cones
    
    def cluster_map(self, eps=None, min_samples=None):
        """
        Cluster the map to remove duplicate cones and reduce noise.
        
        Args:
            eps: Maximum distance between cones to be considered the same
            min_samples: Minimum number of samples in a cluster
            
        Returns:
            Number of clusters (unique cones)
        """
        if len(self.cone_map) < 2:
            return len(self.cone_map)
        
        # Use instance parameters if not specified
        if eps is None:
            eps = self.cluster_eps
        if min_samples is None:
            min_samples = self.cluster_min_samples
        
        # Extract cone positions
        positions = np.array([[c[0], c[1]] for c in self.cone_map])
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
        labels = clustering.labels_
        
        # Group cones by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # Create a new map with one cone per cluster
        new_map = []
        
        for label, indices in clusters.items():
            if label == -1:  # Noise points
                for idx in indices:
                    new_map.append(self.cone_map[idx])
                continue
            
            # Merge cones in the same cluster
            cluster_cones = [self.cone_map[idx] for idx in indices]
            
            # Find most confident class
            cls_counts = {}
            for cone in cluster_cones:
                cls = cone[3]
                conf = cone[2]
                if cls not in cls_counts:
                    cls_counts[cls] = 0
                cls_counts[cls] += conf
            
            most_common_cls = max(cls_counts.items(), key=lambda x: x[1])[0] if cls_counts else 2
            
            # Calculate weighted average position
            total_conf = sum(cone[2] for cone in cluster_cones)
            avg_x = sum(cone[0] * cone[2] for cone in cluster_cones) / total_conf
            avg_y = sum(cone[1] * cone[2] for cone in cluster_cones) / total_conf
            
            new_map.append([avg_x, avg_y, total_conf, most_common_cls])
        
        # Update the map
        self.cone_map = new_map
        
        # Rebuild KDTree
        if self.cone_map:
            cone_positions = np.array([[c[0], c[1]] for c in self.cone_map])
            self.cone_tree = KDTree(cone_positions)
        
        return len(new_map)
    
    def get_cones_by_class(self):
        """
        Get cones grouped by class.
        
        Returns:
            Dictionary with keys 0, 1, 2 (yellow, blue, unknown) and values as lists of [x, y] positions
        """
        result = {0: [], 1: [], 2: []}
        
        for cone in self.cone_map:
            x, y, conf, cls = cone
            cls = int(cls)
            result[cls].append([x, y])
        
        return result
    
    def get_cone_map_with_confidence(self):
        """
        Get the full cone map with confidence values.
        
        Returns:
            Dictionary with keys 0, 1, 2 (yellow, blue, unknown) and values as lists of [x, y, confidence] positions
        """
        result = {0: [], 1: [], 2: []}
        
        for cone in self.cone_map:
            x, y, conf, cls = cone
            cls = int(cls)
            result[cls].append([x, y, conf])
        
        return result
    
    def get_cones_in_view(self, vehicle_pose, max_distance=20.0, fov_degrees=90):
        """
        Get cones that are currently in the vehicle's field of view.
        
        Args:
            vehicle_pose: (x, y, yaw) in world frame
            max_distance: Maximum distance to consider
            fov_degrees: Field of view in degrees
            
        Returns:
            Dictionary with keys 0, 1, 2 (yellow, blue, unknown) and values as lists of [x, y] positions
            in vehicle frame
        """
        if not self.cone_map:
            return {0: [], 1: [], 2: []}
        
        veh_x, veh_y, veh_yaw = vehicle_pose
        
        # Half FOV in radians
        half_fov = np.radians(fov_degrees / 2)
        
        # Rotation matrix for world to vehicle transform
        cos_yaw = np.cos(-veh_yaw)
        sin_yaw = np.sin(-veh_yaw)
        
        result = {0: [], 1: [], 2: []}
        
        for cone in self.cone_map:
            # Calculate cone position relative to vehicle
            rel_x = cone[0] - veh_x
            rel_y = cone[1] - veh_y
            
            # Calculate distance and angle
            distance = np.sqrt(rel_x**2 + rel_y**2)
            angle = np.arctan2(rel_y, rel_x) - veh_yaw
            
            # Normalize angle to [-pi, pi]
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            
            # Check if cone is in view
            if distance <= max_distance and abs(angle) <= half_fov:
                # Transform cone to vehicle frame
                veh_frame_x = rel_x * cos_yaw - rel_y * sin_yaw
                veh_frame_y = rel_x * sin_yaw + rel_y * cos_yaw
                
                cls = int(cone[3])
                result[cls].append([veh_frame_x, veh_frame_y])
        
        return result
    
    def store_planned_path(self, path_points):
        """
        Store a planned path for visualization.
        
        Args:
            path_points: List of (x, y) waypoints in vehicle frame
            
        Returns:
            None
        """
        if path_points:
            # Convert to world frame if we have a vehicle pose
            if self.vehicle_poses:
                veh_x, veh_y, veh_yaw = self.vehicle_poses[-1]
                
                # Rotation matrix for vehicle to world transform
                cos_yaw = np.cos(veh_yaw)
                sin_yaw = np.sin(veh_yaw)
                
                world_path = []
                for x, y in path_points:
                    # Apply rotation and translation
                    world_x = x * cos_yaw - y * sin_yaw + veh_x
                    world_y = x * sin_yaw + y * cos_yaw + veh_y
                    world_path.append((world_x, world_y))
                
                self.planned_paths.append(world_path)
                
                # Keep only the last 10 paths
                if len(self.planned_paths) > 10:
                    self.planned_paths = self.planned_paths[-10:]
    
    def plan_path_using_map(self, vehicle_pose, lookahead_distance=20.0, point_spacing=0.5):
        """
        Plan a path through the mapped cones.
        
        Args:
            vehicle_pose: (x, y, yaw) in world frame
            lookahead_distance: How far ahead to plan
            point_spacing: Spacing between path points
            
        Returns:
            List of (x, y) waypoints in vehicle frame
        """
        if not self.cone_map:
            return []
        
        # Get cones by class
        cones_by_class = self.get_cones_by_class()
        yellow_cones = cones_by_class[0]
        blue_cones = cones_by_class[1]
        
        if not yellow_cones or not blue_cones:
            return []
        
        veh_x, veh_y, veh_yaw = vehicle_pose
        
        # Determine track segment we're currently on by finding closest cones
        closest_yellow = None
        closest_blue = None
        min_yellow_dist = float('inf')
        min_blue_dist = float('inf')
        
        for cone in yellow_cones:
            dist = np.sqrt((cone[0] - veh_x)**2 + (cone[1] - veh_y)**2)
            if dist < min_yellow_dist:
                min_yellow_dist = dist
                closest_yellow = cone
        
        for cone in blue_cones:
            dist = np.sqrt((cone[0] - veh_x)**2 + (cone[1] - veh_y)**2)
            if dist < min_blue_dist:
                min_blue_dist = dist
                closest_blue = cone
        
        # Calculate forward direction vector
        forward_x = np.cos(veh_yaw)
        forward_y = np.sin(veh_yaw)
        
        # Find cones ahead of vehicle
        ahead_yellow = []
        ahead_blue = []
        
        for cone in yellow_cones:
            # Calculate vector from vehicle to cone
            vec_x = cone[0] - veh_x
            vec_y = cone[1] - veh_y
            
            # Project onto forward direction
            projection = vec_x * forward_x + vec_y * forward_y
            
            # Keep cones ahead of vehicle and within lookahead distance
            if projection > 0 and projection < lookahead_distance:
                ahead_yellow.append(cone + [projection])  # Add projection distance
        
        for cone in blue_cones:
            # Calculate vector from vehicle to cone
            vec_x = cone[0] - veh_x
            vec_y = cone[1] - veh_y
            
            # Project onto forward direction
            projection = vec_x * forward_x + vec_y * forward_y
            
            # Keep cones ahead of vehicle and within lookahead distance
            if projection > 0 and projection < lookahead_distance:
                ahead_blue.append(cone + [projection])  # Add projection distance
        
        # Sort by projection distance
        ahead_yellow.sort(key=lambda c: c[2])
        ahead_blue.sort(key=lambda c: c[2])
        
        if not ahead_yellow or not ahead_blue:
            return []
        
        # Generate waypoints by finding midpoints between cone pairs
        waypoints = []
        
        # First, find cone pairs at similar distances
        max_dist_diff = 3.0  # Maximum distance difference for pairing
        track_width = 3.5  # Default track width
        
        # Determine actual track width from nearby cones
        if closest_yellow and closest_blue:
            track_width = np.sqrt((closest_yellow[0] - closest_blue[0])**2 + 
                                  (closest_yellow[1] - closest_blue[1])**2)
        
        # Generate sample points along the track
        current_dist = 0
        while current_dist < lookahead_distance:
            # Find closest yellow and blue cones
            closest_yellow_idx = None
            closest_blue_idx = None
            min_yellow_diff = float('inf')
            min_blue_diff = float('inf')
            
            for i, cone in enumerate(ahead_yellow):
                diff = abs(cone[2] - current_dist)
                if diff < min_yellow_diff:
                    min_yellow_diff = diff
                    closest_yellow_idx = i
            
            for i, cone in enumerate(ahead_blue):
                diff = abs(cone[2] - current_dist)
                if diff < min_blue_diff:
                    min_blue_diff = diff
                    closest_blue_idx = i
            
            # Generate waypoint if we have both cones
            if closest_yellow_idx is not None and closest_blue_idx is not None:
                yellow_cone = ahead_yellow[closest_yellow_idx]
                blue_cone = ahead_blue[closest_blue_idx]
                
                # Only use pair if they're at similar distances
                if abs(yellow_cone[2] - blue_cone[2]) < max_dist_diff:
                    # Calculate midpoint
                    mid_x = (yellow_cone[0] + blue_cone[0]) / 2
                    mid_y = (yellow_cone[1] + blue_cone[1]) / 2
                    
                    # Transform to vehicle frame
                    rel_x = mid_x - veh_x
                    rel_y = mid_y - veh_y
                    
                    veh_frame_x = rel_x * np.cos(-veh_yaw) - rel_y * np.sin(-veh_yaw)
                    veh_frame_y = rel_x * np.sin(-veh_yaw) + rel_y * np.cos(-veh_yaw)
                    
                    waypoints.append((veh_frame_x, veh_frame_y))
            elif closest_yellow_idx is not None:
                # Only yellow cone, estimate blue position
                yellow_cone = ahead_yellow[closest_yellow_idx]
                
                # Calculate normal vector to track direction
                if len(waypoints) >= 2:
                    # Use previous waypoints to determine track direction
                    prev_x, prev_y = waypoints[-2]
                    last_x, last_y = waypoints[-1]
                    
                    track_dir_x = last_x - prev_x
                    track_dir_y = last_y - prev_y
                    norm = np.sqrt(track_dir_x**2 + track_dir_y**2)
                    
                    if norm > 0:
                        # Normalize and rotate 90 degrees to the right
                        normal_x = -track_dir_y / norm
                        normal_y = track_dir_x / norm
                        
                        # Calculate estimated blue cone position
                        # Yellow cones are traditionally on the left, blue on the right
                        blue_x = yellow_cone[0] - normal_x * track_width
                        blue_y = yellow_cone[1] - normal_y * track_width
                        
                        # Calculate midpoint
                        mid_x = (yellow_cone[0] + blue_x) / 2
                        mid_y = (yellow_cone[1] + blue_y) / 2
                        
                        # Transform to vehicle frame
                        rel_x = mid_x - veh_x
                        rel_y = mid_y - veh_y
                        
                        veh_frame_x = rel_x * np.cos(-veh_yaw) - rel_y * np.sin(-veh_yaw)
                        veh_frame_y = rel_x * np.sin(-veh_yaw) + rel_y * np.cos(-veh_yaw)
                        
                        waypoints.append((veh_frame_x, veh_frame_y))
            elif closest_blue_idx is not None:
                # Only blue cone, estimate yellow position
                blue_cone = ahead_blue[closest_blue_idx]
                
                # Calculate normal vector to track direction
                if len(waypoints) >= 2:
                    # Use previous waypoints to determine track direction
                    prev_x, prev_y = waypoints[-2]
                    last_x, last_y = waypoints[-1]
                    
                    track_dir_x = last_x - prev_x
                    track_dir_y = last_y - prev_y
                    norm = np.sqrt(track_dir_x**2 + track_dir_y**2)
                    
                    if norm > 0:
                        # Normalize and rotate 90 degrees to the left
                        normal_x = track_dir_y / norm
                        normal_y = -track_dir_x / norm
                        
                        # Calculate estimated yellow cone position
                        yellow_x = blue_cone[0] + normal_x * track_width
                        yellow_y = blue_cone[1] + normal_y * track_width
                        
                        # Calculate midpoint
                        mid_x = (yellow_x + blue_cone[0]) / 2
                        mid_y = (yellow_y + blue_cone[1]) / 2
                        
                        # Transform to vehicle frame
                        rel_x = mid_x - veh_x
                        rel_y = mid_y - veh_y
                        
                        veh_frame_x = rel_x * np.cos(-veh_yaw) - rel_y * np.sin(-veh_yaw)
                        veh_frame_y = rel_x * np.sin(-veh_yaw) + rel_y * np.cos(-veh_yaw)
                        
                        waypoints.append((veh_frame_x, veh_frame_y))
            
            # Move to next distance
            current_dist += point_spacing
        
        # Store the planned path for visualization
        self.store_planned_path(waypoints)
        
        return waypoints
    
    def generate_visualization(self):
        """
        Generate a visualization of the cone map and vehicle path.
        
        This is useful for debugging and thesis presentations.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.title('SLAM-Based Cone Mapping')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        
        # Plot cone map
        yellow_cones = []
        blue_cones = []
        unknown_cones = []
        
        for cone in self.cone_map:
            x, y, conf, cls = cone
            if cls == 0:  # Yellow
                yellow_cones.append((x, y, conf))
            elif cls == 1:  # Blue
                blue_cones.append((x, y, conf))
            else:  # Unknown
                unknown_cones.append((x, y, conf))
        
        # Plot cones
        if yellow_cones:
            x, y, conf = zip(*yellow_cones)
            plt.scatter(x, y, c='yellow', marker='^', s=50, alpha=0.7, edgecolors='black', label='Yellow Cones')
        
        if blue_cones:
            x, y, conf = zip(*blue_cones)
            plt.scatter(x, y, c='blue', marker='^', s=50, alpha=0.7, edgecolors='black', label='Blue Cones')
        
        if unknown_cones:
            x, y, conf = zip(*unknown_cones)
            plt.scatter(x, y, c='gray', marker='^', s=50, alpha=0.7, edgecolors='black', label='Unknown Cones')
        
        # Plot vehicle path
        if self.vehicle_poses:
            x = [pose[0] for pose in self.vehicle_poses]
            y = [pose[1] for pose in self.vehicle_poses]
            plt.plot(x, y, 'g-', linewidth=2, alpha=0.7, label='Vehicle Path')
            
            # Highlight current position
            if len(self.vehicle_poses) > 0:
                current = self.vehicle_poses[-1]
                plt.scatter(current[0], current[1], c='red', s=100, marker='o', label='Current Position')
                
                # Draw vehicle orientation
                arrow_length = 2.0
                plt.arrow(current[0], current[1], 
                         arrow_length * np.cos(current[2]), 
                         arrow_length * np.sin(current[2]),
                         head_width=0.5, head_length=0.7, fc='red', ec='red')
        
        # Plot the latest planned path
        if self.planned_paths:
            latest_path = self.planned_paths[-1]
            path_x = [p[0] for p in latest_path]
            path_y = [p[1] for p in latest_path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')
        
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cone_map_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate additional visualizations
        self._generate_heatmap()
        
        return filepath
    
    def _generate_heatmap(self):
        """Generate a heatmap of the cone confidence values."""
        if not self.confidence_grid:
            return
        
        # Create grid for heatmap
        min_x = min(key[0] for key in self.confidence_grid.keys())
        max_x = max(key[0] for key in self.confidence_grid.keys())
        min_y = min(key[1] for key in self.confidence_grid.keys())
        max_y = max(key[1] for key in self.confidence_grid.keys())
        
        grid_size_x = max_x - min_x + 1
        grid_size_y = max_y - min_y + 1
        
        # Create heatmap grid
        heatmap = np.zeros((grid_size_y, grid_size_x))
        
        # Fill the grid with confidence values
        for (x, y), conf in self.confidence_grid.items():
            grid_x = x - min_x
            grid_y = y - min_y
            
            if 0 <= grid_x < grid_size_x and 0 <= grid_y < grid_size_y:
                heatmap[grid_y, grid_x] = conf
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.title('Cone Detection Confidence Heatmap')
        plt.imshow(heatmap, origin='lower', cmap='viridis', 
                  extent=[min_x, max_x, min_y, max_y])
        plt.colorbar(label='Confidence')
        
        # Overlay vehicle path
        if self.vehicle_poses:
            x = [pose[0] for pose in self.vehicle_poses]
            y = [pose[1] for pose in self.vehicle_poses]
            plt.plot(x, y, 'w-', linewidth=2, label='Vehicle Path')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.grid(True, color='w', alpha=0.3)
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"confidence_heatmap_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report for thesis purposes."""
        if not self.cone_map:
            return "No cone data available for analysis"
        
        # Create figures directory if it doesn't exist
        analysis_dir = os.path.join(self.output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Stats dictionary
        stats = {
            "total_cones": len(self.cone_map),
            "yellow_cones": len([c for c in self.cone_map if c[3] == 0]),
            "blue_cones": len([c for c in self.cone_map if c[3] == 1]),
            "unknown_cones": len([c for c in self.cone_map if c[3] == 2]),
            "total_updates": self.total_updates,
            "new_cones_added": self.total_new_cones,
            "cones_updated": self.total_updated_cones,
        }
        
        # Create multi-figure report
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle('SLAM-Based Cone Mapping Analysis', fontsize=16)
        
        # 2x2 grid for different visualizations
        gs = fig.add_gridspec(2, 2)
        
        # Map visualization
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Cone Map')
        
        # Plot cone map
        for cone in self.cone_map:
            x, y, conf, cls = cone
            if cls == 0:  # Yellow
                ax1.scatter(x, y, c='yellow', marker='^', s=30, alpha=0.7, edgecolors='black')
            elif cls == 1:  # Blue
                ax1.scatter(x, y, c='blue', marker='^', s=30, alpha=0.7, edgecolors='black')
            else:  # Unknown
                ax1.scatter(x, y, c='gray', marker='^', s=30, alpha=0.7, edgecolors='black')
        
        # Plot vehicle path
        if self.vehicle_poses:
            x = [pose[0] for pose in self.vehicle_poses]
            y = [pose[1] for pose in self.vehicle_poses]
            ax1.plot(x, y, 'g-', linewidth=1, alpha=0.7)
        
        ax1.grid(True)
        ax1.set_aspect('equal')
        
        # Confidence distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('Cone Confidence Distribution')
        
        # Get confidences by class
        yellow_conf = [c[2] for c in self.cone_map if c[3] == 0]
        blue_conf = [c[2] for c in self.cone_map if c[3] == 1]
        unknown_conf = [c[2] for c in self.cone_map if c[3] == 2]
        
        # Plot histograms
        if yellow_conf:
            ax2.hist(yellow_conf, bins=10, alpha=0.5, color='yellow', edgecolor='black', label='Yellow')
        if blue_conf:
            ax2.hist(blue_conf, bins=10, alpha=0.5, color='blue', edgecolor='black', label='Blue')
        if unknown_conf:
            ax2.hist(unknown_conf, bins=10, alpha=0.5, color='gray', edgecolor='black', label='Unknown')
        
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True)
        
        # Distance between cone pairs
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_title('Track Width Analysis')
        
        # Get cones by class
        yellow_cones = [(c[0], c[1]) for c in self.cone_map if c[3] == 0]
        blue_cones = [(c[0], c[1]) for c in self.cone_map if c[3] == 1]
        
        # Calculate track width at different points
        track_widths = []
        
        if yellow_cones and blue_cones:
            # Create KDTrees for efficient nearest neighbor search
            yellow_tree = KDTree(yellow_cones)
            
            for blue_cone in blue_cones:
                dist, idx = yellow_tree.query(blue_cone)
                if dist < 10.0:  # Only consider reasonable distances
                    yellow_cone = yellow_cones[idx]
                    track_widths.append(dist)
        
        if track_widths:
            ax3.hist(track_widths, bins=15, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(x=np.mean(track_widths), color='red', linestyle='-', label=f'Mean: {np.mean(track_widths):.2f}m')
            ax3.set_xlabel('Track Width (m)')
            ax3.set_ylabel('Count')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for track width analysis', 
                    horizontalalignment='center', verticalalignment='center')
        
        ax3.grid(True)
        
        # Stats table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title('Mapping Statistics')
        ax4.axis('off')
        
        # Create table content
        table_data = []
        for key, value in stats.items():
            table_data.append([key.replace('_', ' ').title(), value])
        
        # Create table
        table = ax4.table(cellText=table_data, loc='center', colWidths=[0.6, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Add timestamp and save
        plt.figtext(0.02, 0.02, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                   fontsize=8)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mapping_analysis_{timestamp}.png"
        filepath = os.path.join(analysis_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text report
        report_text = f"""
        SLAM-Based Cone Mapping Analysis Report
        ======================================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Map Statistics:
        - Total mapped cones: {stats['total_cones']}
        - Yellow cones: {stats['yellow_cones']}
        - Blue cones: {stats['blue_cones']}
        - Unknown cones: {stats['unknown_cones']}
        
        Mapping Performance:
        - Total updates: {stats['total_updates']}
        - New cones added: {stats['new_cones_added']}
        - Cones updated: {stats['cones_updated']}
        
        Track Analysis:
        - Average track width: {np.mean(track_widths) if track_widths else 'N/A'} meters
        - Track width std dev: {np.std(track_widths) if track_widths else 'N/A'} meters
        
        Vehicle Trajectory:
        - Path length: {self._calculate_path_length()} meters
        - Average speed: {self._calculate_average_speed()} m/s
        
        Analysis figures saved to: {analysis_dir}
        """
        
        # Save report
        report_file = os.path.join(analysis_dir, f"mapping_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        return filepath
    
    def _calculate_path_length(self):
        """Calculate the total path length from vehicle poses."""
        if len(self.vehicle_poses) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(self.vehicle_poses)):
            prev_x, prev_y, _ = self.vehicle_poses[i-1]
            curr_x, curr_y, _ = self.vehicle_poses[i]
            
            segment_length = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            total_length += segment_length
        
        return total_length
    
    def _calculate_average_speed(self):
        """Estimate average speed from trajectory if timestamps available."""
        if not hasattr(self, 'vehicle_timestamps') or len(self.vehicle_timestamps) < 2:
            return 0.0
        
        path_length = self._calculate_path_length()
        time_delta = self.vehicle_timestamps[-1] - self.vehicle_timestamps[0]
        
        if time_delta > 0:
            return path_length / time_delta
        return 0.0