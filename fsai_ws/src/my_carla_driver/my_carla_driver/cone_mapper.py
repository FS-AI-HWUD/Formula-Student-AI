import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

class ConeMapper:
    """
    SLAM-based cone mapping system to track cone positions and build a global map.
    """
    def __init__(self, max_distance=50.0, fusion_threshold=1.0):
        """
        Initialize the cone mapper.
        
        Args:
            max_distance: Maximum distance in meters to track cones
            fusion_threshold: Distance threshold in meters to fuse cone detections
        """
        # Global map of cones, stored as [[x, y, confidence, class], ...]
        # Class: 0 = yellow, 1 = blue, 2 = unknown
        self.cone_map = []
        
        # Configuration
        self.max_distance = max_distance
        self.fusion_threshold = fusion_threshold
        
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
    
    def update_map(self, detections, vehicle_pose, current_time=None):
        """
        Update the map with new cone detections.
        
        Args:
            detections: List of detected cones [[x, y, confidence, class], ...] in vehicle frame
            vehicle_pose: (x, y, yaw) in world frame
            current_time: Current timestamp (if None, increments by 1)
            
        Returns:
            Number of new cones added to the map
        """
        if current_time is None:
            current_time = self.last_update_time + 1
        
        self.last_update_time = current_time
        self.update_vehicle_pose(*vehicle_pose)
        
        if not detections:
            return 0
        
        veh_x, veh_y, veh_yaw = vehicle_pose
        
        # Transform detections from vehicle to world frame
        cos_yaw = np.cos(veh_yaw)
        sin_yaw = np.sin(veh_yaw)
        
        world_detections = []
        for x, y, conf, cls in detections:
            # Apply rotation and translation
            world_x = x * cos_yaw - y * sin_yaw + veh_x
            world_y = x * sin_yaw + y * cos_yaw + veh_y
            world_detections.append([world_x, world_y, conf, cls])
        
        # Count new cones added
        new_cones = 0
        
        # Build KDTree if needed
        if not self.cone_tree and self.cone_map:
            cone_positions = np.array([[c[0], c[1]] for c in self.cone_map])
            self.cone_tree = KDTree(cone_positions)
        
        # Process each detection
        for detection in world_detections:
            x, y, conf, cls = detection
            
            # Skip if cone is too far away
            dist_from_vehicle = np.sqrt((x - veh_x)**2 + (y - veh_y)**2)
            if dist_from_vehicle > self.max_distance:
                continue
            
            # Check if this cone already exists in the map
            if self.cone_tree:
                dist, idx = self.cone_tree.query([x, y])
                
                if dist < self.fusion_threshold:
                    # Update existing cone
                    existing_cone = self.cone_map[idx]
                    
                    # Update position with weighted average
                    existing_conf = existing_cone[2]
                    total_conf = existing_conf + conf
                    weight_new = conf / total_conf
                    weight_old = existing_conf / total_conf
                    
                    updated_x = weight_old * existing_cone[0] + weight_new * x
                    updated_y = weight_old * existing_cone[1] + weight_new * y
                    
                    # Update class if more confident
                    updated_cls = existing_cone[3]
                    if conf > existing_conf:
                        updated_cls = cls
                    
                    # Update cone in map
                    self.cone_map[idx] = [updated_x, updated_y, total_conf, updated_cls]
                    self.total_updated_cones += 1
                    continue
            
            # Add new cone to map
            self.cone_map.append([x, y, conf, cls])
            new_cones += 1
            self.total_new_cones += 1
        
        # Rebuild KDTree after updates
        if self.cone_map:
            cone_positions = np.array([[c[0], c[1]] for c in self.cone_map])
            self.cone_tree = KDTree(cone_positions)
        
        self.total_updates += 1
        return new_cones
    
    def cluster_map(self, eps=1.0, min_samples=1):
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
            cls = int(cone[3])
            result[cls].append([cone[0], cone[1]])
        
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
        
        return waypoints