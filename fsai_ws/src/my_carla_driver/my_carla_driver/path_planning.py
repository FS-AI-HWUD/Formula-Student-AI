import numpy as np
from scipy.interpolate import CubicSpline
import cv2

class PathPlanner:
    def __init__(self, zed_camera, depth_min=0.5, depth_max=12.5, cone_spacing=1.5, visualize=True):
        self.zed_camera = zed_camera
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.cone_spacing = cone_spacing
        self.visualize = visualize
        self.path = None
        self.previous_path = None  # Store the previous path for temporal smoothing
        self.image_width = zed_camera.resolution[0]
        self.image_height = zed_camera.resolution[1]
        self.fov_horizontal = 90.0
        self.wheelbase = 2.7
    
    def _filter_cones(self, cones, depths):
        filtered_cones = []
        filtered_depths = []
        
        print("Skipping depth corridor filter as requested...")
        for (x1, y1, x2, y2, cls), depth in zip(cones, depths):
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            filtered_cones.append((center_x, center_y, cls))
            filtered_depths.append(depth)
            print(f"Kept cone: Class = {cls}, Depth = {depth:.2f}m, Center = ({center_x}, {center_y})")
        
        print(f"Total cones after removing filter: {len(filtered_cones)}")
        return filtered_cones, filtered_depths
    
    def detect_uturn(self, yellow_world, blue_world, min_cones=2, shift_threshold=0.3, depth_tolerance=2.0):
        """Detect if the cone arrangement indicates a U-turn with improved recognition.
        
        Args:
            yellow_world (list): List of (x, depth) coordinates for yellow cones
            blue_world (list): List of (x, depth) coordinates for blue cones
            min_cones (int): Minimum number of cones required for each color
            shift_threshold (float): Minimum lateral shift threshold
            depth_tolerance (float): Maximum depth difference for cone pairing
            
        Returns:
            tuple: (is_uturn, uturn_info) where uturn_info is a dict with 'depth' and 'direction'
        """
        if len(yellow_world) < min_cones or len(blue_world) < min_cones:
            return False, None
        
        # Sort cones by depth
        yellow_sorted = sorted(yellow_world, key=lambda p: p[1])
        blue_sorted = sorted(blue_world, key=lambda p: p[1])
        
        # Check for characteristic U-turn patterns:
        # 1. Significant lateral shift in cones
        # 2. Cones get closer to each other and then further apart
        # 3. Both boundaries curve in the same direction
        
        # Calculate lateral shifts for both boundaries
        if len(yellow_sorted) >= 3:
            yellow_shifts = [(yellow_sorted[i][0] - yellow_sorted[i-1][0]) / 
                            max(0.1, yellow_sorted[i][1] - yellow_sorted[i-1][1]) 
                            for i in range(1, len(yellow_sorted))]
        else:
            yellow_shifts = []
        
        if len(blue_sorted) >= 3:
            blue_shifts = [(blue_sorted[i][0] - blue_sorted[i-1][0]) / 
                          max(0.1, blue_sorted[i][1] - blue_sorted[i-1][1]) 
                          for i in range(1, len(blue_sorted))]
        else:
            blue_shifts = []
        
        # Check for U-turn pattern - consistent shifts in both boundaries
        is_uturn = False
        uturn_depth = None
        uturn_direction = None
        
        if yellow_shifts and blue_shifts:
            # Get average shifts
            avg_yellow_shift = sum(yellow_shifts) / len(yellow_shifts)
            avg_blue_shift = sum(blue_shifts) / len(blue_shifts)
            
            # Calculate consistency of shifts (standard deviation)
            yellow_consistency = np.std(yellow_shifts) if len(yellow_shifts) > 1 else 999
            blue_consistency = np.std(blue_shifts) if len(blue_shifts) > 1 else 999
            
            # Check for consistent direction in shifts
            yellow_direction = all(s > shift_threshold for s in yellow_shifts) or all(s < -shift_threshold for s in yellow_shifts)
            blue_direction = all(s > shift_threshold for s in blue_shifts) or all(s < -shift_threshold for s in blue_shifts)
            
            # 1. Check for significant and consistent shifts
            significant_shift = abs(avg_yellow_shift) > shift_threshold and abs(avg_blue_shift) > shift_threshold
            consistent_shift = yellow_consistency < 0.4 and blue_consistency < 0.4
            
            # 2. Check for shifts in opposite directions (one boundary moves left, other right)
            opposite_shifts = avg_yellow_shift * avg_blue_shift < 0
            
            # 3. Check track width pattern - narrows then widens (U-turn apex)
            track_widths = []
            for y_pos in yellow_sorted:
                for b_pos in blue_sorted:
                    if abs(y_pos[1] - b_pos[1]) < depth_tolerance:  # Similar depths
                        width = abs(y_pos[0] - b_pos[0])
                        track_widths.append((width, (y_pos[1] + b_pos[1]) / 2))  # Store width and average depth
            
            # Look for track width changes
            if len(track_widths) >= 3:
                track_widths.sort(key=lambda x: x[1])  # Sort by depth
                
                # Check for narrowing then widening pattern
                width_changes = [track_widths[i][0] - track_widths[i-1][0] for i in range(1, len(track_widths))]
                
                # Look for sign change in width_changes (narrowing to widening)
                sign_changes = []
                for i in range(1, len(width_changes)):
                    if width_changes[i-1] * width_changes[i] < 0:  # Sign change
                        sign_changes.append(i)
                
                if sign_changes:
                    # We have width pattern changes - look for narrowing then widening
                    for idx in sign_changes:
                        if idx > 0 and idx < len(width_changes) - 1:
                            if width_changes[idx-1] < -0.1 and width_changes[idx+1] > 0.1:
                                # We have a narrowing then widening pattern
                                uturn_idx = idx + 1  # Position in track_widths
                                uturn_depth = track_widths[uturn_idx][1]
                                
                                # Determine U-turn direction from shift patterns
                                if avg_yellow_shift > 0 and avg_blue_shift < 0:
                                    uturn_direction = "right"
                                elif avg_yellow_shift < 0 and avg_blue_shift > 0:
                                    uturn_direction = "left"
                                else:
                                    # If shift patterns are unclear, use track boundary positions
                                    # Find depths near the U-turn
                                    near_uturn = [(y, b) for y in yellow_sorted for b in blue_sorted 
                                                 if abs(y[1] - uturn_depth) < depth_tolerance and abs(b[1] - uturn_depth) < depth_tolerance]
                                    
                                    if near_uturn:
                                        # Average positions of boundaries near U-turn
                                        avg_y = np.mean([y[0] for y, _ in near_uturn])
                                        avg_b = np.mean([b[0] for _, b in near_uturn])
                                        
                                        # Usually yellow is positive X, blue is negative X
                                        # If yellow is much larger than blue, likely turning right
                                        if avg_y > avg_b:
                                            uturn_direction = "right"
                                        else:
                                            uturn_direction = "left"
                                
                                is_uturn = True
                                print(f"U-turn detected at depth {uturn_depth:.1f}m, direction: {uturn_direction}")
                                break
            
            # Combine all evidence
            if significant_shift and consistent_shift and opposite_shifts:
                # Likely a U-turn - estimate the depth
                if not uturn_depth:  # If not determined from width analysis
                    # Find depth where track is narrowest
                    if track_widths:
                        min_width_idx = min(range(len(track_widths)), key=lambda i: track_widths[i][0])
                        uturn_depth = track_widths[min_width_idx][1]
                    else:
                        # Estimate depth based on midpoint of detected cones
                        yellow_mid_idx = len(yellow_sorted) // 2
                        blue_mid_idx = len(blue_sorted) // 2
                        uturn_depth = (yellow_sorted[yellow_mid_idx][1] + blue_sorted[blue_mid_idx][1]) / 2
                
                # Determine direction if not already set
                if not uturn_direction:
                    if avg_yellow_shift > 0 and avg_blue_shift < 0:
                        uturn_direction = "right"
                    elif avg_yellow_shift < 0 and avg_blue_shift > 0:
                        uturn_direction = "left"
                    else:
                        uturn_direction = "unknown"
                
                is_uturn = True
                print(f"U-turn detected at approximately {uturn_depth:.1f}m, direction: {uturn_direction}")
        
        return is_uturn, {"depth": uturn_depth, "direction": uturn_direction} if is_uturn else None
    
    def calculate_steering(self, lookahead_distance=4.0, steering_gain=1.0, max_steering_angle=30.0):
        """Calculate steering based on the planned path.
        
        Args:
            lookahead_distance (float): Distance ahead to look for target point (in meters)
            steering_gain (float): Multiplier for steering sensitivity
            max_steering_angle (float): Maximum steering angle in degrees
        """
        if self.path is None or len(self.path) < 2:
            return 0.0
        
        # Find point at lookahead distance
        cumulative_dist = 0.0
        target_idx = 0
        
        for i in range(1, len(self.path)):
            x1, y1 = self.path[i-1]
            x2, y2 = self.path[i]
            segment_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            cumulative_dist += segment_length
            
            if cumulative_dist >= lookahead_distance:
                target_idx = i
                break
        
        if target_idx > 0:
            # Get target point
            target_x, target_y = self.path[target_idx]
            
            # Calculate angle to target point (from car's perspective)
            angle = np.arctan2(target_x, target_y)
            
            # Apply steering gain
            angle *= steering_gain
            
            # Normalize steering angle to range [-1, 1]
            max_steering_rad = np.radians(max_steering_angle)
            normalized_steering = np.clip(angle / max_steering_rad, -1.0, 1.0)
            
            return normalized_steering
        
        return 0.0  # Default: go straight
    
    def _pair_cones(self, cones, depths, min_track_width=2.0, max_track_width=6.0, depth_step=0.3):
        """Pair cones to create a path through the center of the track with improved gap handling.
        
        Args:
            cones (list): List of cone coordinates and classes
            depths (list): List of cone depths
            min_track_width (float): Minimum valid track width in meters
            max_track_width (float): Maximum valid track width in meters
            depth_step (float): Distance between waypoints in meters
        """
        # Separate yellow and blue cones
        yellow_cones = [(c, d) for c, d in zip(cones, depths) if c[2] == 0]
        blue_cones = [(c, d) for c, d in zip(cones, depths) if c[2] == 1]
        
        print(f"Yellow cones: {len(yellow_cones)}, Blue cones: {len(blue_cones)}")
        
        # Convert to world coordinates using depth
        yellow_world = []
        for cone, depth in yellow_cones:
            center_x = cone[0]
            # Calculate angle from image center
            angle = ((center_x - self.image_width / 2) / (self.image_width / 2)) * (self.fov_horizontal / 2)
            # Calculate world X coordinate
            world_x = depth * np.tan(np.radians(angle))
            # Store as (x, depth) coordinates in meters
            yellow_world.append((world_x, depth))
        
        blue_world = []
        for cone, depth in blue_cones:
            center_x = cone[0]
            angle = ((center_x - self.image_width / 2) / (self.image_width / 2)) * (self.fov_horizontal / 2)
            world_x = depth * np.tan(np.radians(angle))
            blue_world.append((world_x, depth))
        
        # Sort by depth
        yellow_world.sort(key=lambda p: p[1])
        blue_world.sort(key=lambda p: p[1])
        
        # Initialize waypoints at car position
        waypoints = [(0.0, 0.5)]  # Start at car position
        
        # Calculate track width from actual cone positions
        track_widths = []
        for y_pos in yellow_world:
            for b_pos in blue_world:
                # Find cones at similar depths
                if abs(y_pos[1] - b_pos[1]) < 2.0:
                    width = abs(y_pos[0] - b_pos[0])
                    # Reasonable width check
                    if min_track_width < width < max_track_width:
                        track_widths.append(width)
        
        # Use median track width if available, otherwise use default
        track_width = np.median(track_widths) if track_widths else 3.5
        print(f"Calculated track width: {track_width:.2f}m")
        
        # Store the last valid boundary positions to handle gaps
        last_valid_yellow_pos = None
        last_valid_blue_pos = None
        
        # Track the last observed trend for each boundary
        yellow_trend = 0.0  # Direction of yellow boundary movement
        blue_trend = 0.0    # Direction of blue boundary movement
        
        # Find maximum detected cone depth
        max_depth = 15.0  # Default maximum depth
        if yellow_world:
            max_depth = max(max_depth, max(y[1] for y in yellow_world))
        if blue_world:
            max_depth = max(max_depth, max(b[1] for b in blue_world))
        
        # Sample depths at higher resolution for better path continuity
        depths = np.arange(0.5, min(max_depth + 3.0, 20.0), depth_step)
        
        # Maximum allowed gap in meters before extrapolation is needed
        max_gap_threshold = 6.0
        
        # Generate waypoints at each depth
        for depth in depths:
            # Find closest yellow and blue cones at this depth
            closest_yellow = None
            closest_blue = None
            best_y_dist = float('inf')
            best_b_dist = float('inf')
            
            for y_pos in yellow_world:
                dist = abs(y_pos[1] - depth)
                if dist < best_y_dist and dist < 3.0:  # Within 3m
                    best_y_dist = dist
                    closest_yellow = y_pos
            
            for b_pos in blue_world:
                dist = abs(b_pos[1] - depth)
                if dist < best_b_dist and dist < 3.0:  # Within 3m
                    best_b_dist = dist
                    closest_blue = b_pos
            
            # Handle yellow boundary gaps
            if closest_yellow is None and last_valid_yellow_pos is not None:
                # Check if gap is too large
                if depth - last_valid_yellow_pos[1] <= max_gap_threshold:
                    # Extrapolate yellow position based on previous trend
                    # Calculate trend if we have at least two previous yellow points
                    if len(waypoints) >= 2 and yellow_world:
                        # Find previous yellow points to determine trend
                        prev_yellow_points = [y for y in yellow_world if y[1] < depth]
                        if len(prev_yellow_points) >= 2:
                            # Sort by depth to get the most recent points
                            prev_yellow_points.sort(key=lambda p: p[1], reverse=True)
                            if len(prev_yellow_points) >= 2:
                                p1, p2 = prev_yellow_points[:2]
                                # Calculate lateral change per meter of depth
                                yellow_trend = (p1[0] - p2[0]) / max(0.1, p1[1] - p2[1])
                    
                    # Extrapolate position
                    gap_distance = depth - last_valid_yellow_pos[1]
                    extrapolated_x = last_valid_yellow_pos[0] + yellow_trend * gap_distance
                    closest_yellow = (extrapolated_x, depth)
                    print(f"Extrapolated yellow cone at depth {depth:.1f}m: x={extrapolated_x:.2f}m (trend: {yellow_trend:.2f})")
            
            # Update last valid yellow position
            if closest_yellow is not None:
                last_valid_yellow_pos = closest_yellow
            
            # Handle blue boundary gaps
            if closest_blue is None and last_valid_blue_pos is not None:
                # Check if gap is too large
                if depth - last_valid_blue_pos[1] <= max_gap_threshold:
                    # Extrapolate blue position based on previous trend
                    # Calculate trend if we have at least two previous blue points
                    if len(waypoints) >= 2 and blue_world:
                        # Find previous blue points to determine trend
                        prev_blue_points = [b for b in blue_world if b[1] < depth]
                        if len(prev_blue_points) >= 2:
                            # Sort by depth to get the most recent points
                            prev_blue_points.sort(key=lambda p: p[1], reverse=True)
                            if len(prev_blue_points) >= 2:
                                p1, p2 = prev_blue_points[:2]
                                # Calculate lateral change per meter of depth
                                blue_trend = (p1[0] - p2[0]) / max(0.1, p1[1] - p2[1])
                    
                    # Extrapolate position
                    gap_distance = depth - last_valid_blue_pos[1]
                    extrapolated_x = last_valid_blue_pos[0] + blue_trend * gap_distance
                    closest_blue = (extrapolated_x, depth)
                    print(f"Extrapolated blue cone at depth {depth:.1f}m: x={extrapolated_x:.2f}m (trend: {blue_trend:.2f})")
            
            # Update last valid blue position
            if closest_blue is not None:
                last_valid_blue_pos = closest_blue
            
            # Generate waypoint based on available cones
            if closest_yellow is not None and closest_blue is not None:
                # Both colors available - find midpoint
                midpoint_x = (closest_yellow[0] + closest_blue[0]) / 2
                waypoints.append((midpoint_x, depth))
            elif closest_yellow is not None:
                # Only yellow - estimate blue position
                estimated_blue_x = closest_yellow[0] - track_width
                midpoint_x = (closest_yellow[0] + estimated_blue_x) / 2
                waypoints.append((midpoint_x, depth))
            elif closest_blue is not None:
                # Only blue - estimate yellow position
                estimated_yellow_x = closest_blue[0] + track_width
                midpoint_x = (estimated_yellow_x + closest_blue[0]) / 2
                waypoints.append((midpoint_x, depth))
            else:
                # No cones at this depth and no extrapolation was possible
                # Continue in the same direction as recent waypoints trend
                if len(waypoints) >= 2:
                    # Calculate the trend from the last two waypoints
                    last_x, last_y = waypoints[-1]
                    prev_x, prev_y = waypoints[-2]
                    
                    # Calculate lateral change per meter
                    if last_y > prev_y:  # Avoid division by zero
                        lateral_trend = (last_x - prev_x) / (last_y - prev_y)
                        # Add new waypoint continuing the trend
                        extrapolated_x = last_x + lateral_trend * (depth - last_y)
                        waypoints.append((extrapolated_x, depth))
                    else:
                        # Just continue straight if we can't determine trend
                        waypoints.append((last_x, depth))
                else:
                    # With no history, continue straight
                    waypoints.append((0.0, depth))
        
        # Apply enhanced smoothing for better continuity
        if len(waypoints) > 2:
            # Initial smoothing to reduce noise
            smoothed_waypoints = [waypoints[0]]  # Keep first point
            for i in range(1, len(waypoints) - 1):
                x, depth = waypoints[i]
                prev_x = waypoints[i-1][0]
                next_x = waypoints[i+1][0]
                # Apply smoothing with adaptive weights
                smoothed_x = 0.25 * prev_x + 0.5 * x + 0.25 * next_x
                smoothed_waypoints.append((smoothed_x, depth))
            smoothed_waypoints.append(waypoints[-1])  # Keep last point
            
            # Secondary pass with larger window for better continuity
            final_waypoints = [smoothed_waypoints[0]]
            for i in range(1, len(smoothed_waypoints) - 1):
                x, depth = smoothed_waypoints[i]
                
                # Get wider context if available
                window_size = 2
                points_before = min(window_size, i)
                points_after = min(window_size, len(smoothed_waypoints) - i - 1)
                
                window_x = [smoothed_waypoints[i-j][0] for j in range(1, points_before+1)]
                window_x.append(x)
                window_x.extend([smoothed_waypoints[i+j][0] for j in range(1, points_after+1)])
                
                # Apply Gaussian-like weighting
                weights = [0.5 ** j for j in range(points_before, -1, -1)]
                weights.extend([0.5 ** j for j in range(1, points_after+1)])
                weights = [w / sum(weights) for w in weights]
                
                # Weighted average
                weighted_x = sum(wx * w for wx, w in zip(window_x, weights))
                final_waypoints.append((weighted_x, depth))
            
            final_waypoints.append(smoothed_waypoints[-1])
            return final_waypoints
        
        return waypoints
    
    def _smooth_path(self, x_coords, y_coords, smoothing_factor=0.1, num_points=100):
        """Smooth path using cubic spline interpolation.
        
        Args:
            x_coords (list): List of x coordinates
            y_coords (list): List of y coordinates
            smoothing_factor (float): Smoothing factor for spline (0 = no smoothing)
            num_points (int): Number of points in smoothed path
            
        Returns:
            tuple: (smoothed_x, smoothed_y) coordinates
        """
        if len(x_coords) < 3:
            return x_coords, y_coords
        
        # Convert to numpy arrays
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        
        # Sort by y-coordinate (depth)
        indices = np.argsort(y_coords)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]
        
        # Create cubic spline
        try:
            spline = CubicSpline(y_coords, x_coords, bc_type='natural')
            
            # Generate evenly spaced y-coordinates
            y_smooth = np.linspace(y_coords[0], y_coords[-1], num_points)
            
            # Evaluate spline at new points
            x_smooth = spline(y_smooth)
            
            # Apply additional smoothing if needed
            if smoothing_factor > 0:
                window_size = int(len(x_smooth) * smoothing_factor)
                if window_size > 1:
                    x_smooth = np.convolve(x_smooth, np.ones(window_size)/window_size, mode='same')
            
            return x_smooth, y_smooth
            
        except Exception as e:
            print(f"Error in path smoothing: {e}")
            return x_coords, y_coords
    
    def _smooth_path_temporally(self, new_path):
        """Blend the new path with the previous path to reduce sudden jumps."""
        if self.previous_path is None or len(self.previous_path) != len(new_path):
            self.previous_path = new_path
            return new_path
        
        alpha = 0.7  # Smoothing factor (0 = use previous path, 1 = use new path)
        smoothed_path = []
        for (new_x, new_y), (prev_x, prev_y) in zip(new_path, self.previous_path):
            smoothed_x = alpha * new_x + (1 - alpha) * prev_x
            smoothed_y = alpha * new_y + (1 - alpha) * prev_y
            smoothed_path.append((smoothed_x, smoothed_y))
        
        self.previous_path = smoothed_path
        return smoothed_path
    
    def plan_path(self):
        try:
            if not self.zed_camera.cone_detections:
                print("No cone detections available")
                self.path = None
                return
            
            cones = []
            depths = []
            for detection in self.zed_camera.cone_detections:
                x1, y1, x2, y2 = detection['box']
                cls = detection['cls']
                depth = detection['depth']
                print(f"Using cone: Class = {cls}, Depth = {depth:.2f}m, Box = ({x1}, {y1}, {x2}, {y2})")
                cones.append((x1, y1, x2, y2, cls))
                depths.append(depth)
            
            print(f"Using {len(cones)} detected cones from zed_2i.py")
            
            filtered_cones, filtered_depths = self._filter_cones(cones, depths)
            if not filtered_cones:
                print("No cones after filtering")
                self.path = None
                return
            
            yellow_cones = [(c, d) for c, d in zip(filtered_cones, filtered_depths) if c[2] == 0]
            blue_cones = [(c, d) for c, d in zip(filtered_cones, filtered_depths) if c[2] == 1]
            print(f"Yellow cones: {len(yellow_cones)}, Blue cones: {len(blue_cones)}")
            
            waypoints = self._pair_cones(filtered_cones, filtered_depths)
            
            if waypoints:
                x_coords, y_coords = zip(*waypoints)
                smoothed_x, smoothed_y = self._smooth_path(x_coords, y_coords)
                new_path = list(zip(smoothed_x, smoothed_y))
                
                # Apply temporal smoothing
                new_path = self._smooth_path_temporally(new_path)
                
                # Sort the path by depth to ensure smooth drawing
                new_path.sort(key=lambda p: p[1])
                self.path = new_path
                
                print(f"Path successfully created with {len(self.path)} points!")
            else:
                print("No waypoints found")
                self.path = None
                
        except Exception as e:
            print(f"Error in path planning: {e}")
            self.path = None
    
    def draw_path(self, image):
        if not self.visualize or self.path is None or len(self.path) < 2:
            return image
        
        car_x = self.image_width // 2
        car_y = self.image_height - 20
        
        # Draw car indicator
        car_points = np.array([
            [car_x, car_y - 10],
            [car_x - 10, car_y + 10],
            [car_x + 10, car_y + 10]
        ], dtype=np.int32)
        cv2.fillPoly(image, [car_points], (0, 255, 255))
        
        # Improved depth-to-pixel mapping function
        def depth_to_pixel_y(depth):
            # Non-linear mapping for better perspective
            if depth <= 1.0:
                # Close points (linear mapping for close range)
                y = int(car_y - (depth / 1.0) * 60)
            else:
                # Far points (non-linear mapping for better perspective)
                y = int(car_y - 60 - ((depth - 1.0) ** 0.85) * 40)
            
            return max(0, min(y, self.image_height - 1))
        
        # Convert path points to image coordinates
        path_pixels = [(car_x, car_y)]  # Start at car position
        
        for world_x, world_y in self.path:
            # Convert lateral position to image x-coordinate
            angle = np.arctan2(world_x, world_y)
            pixel_x = int(car_x + (angle / np.radians(self.fov_horizontal / 2)) * 
                         (self.image_width / 2))
            
            # Convert depth to image y-coordinate using our improved mapping
            pixel_y = depth_to_pixel_y(world_y)
            
            path_pixels.append((pixel_x, pixel_y))
        
        # Draw path with visual enhancements
        if len(path_pixels) >= 2:
            # Draw outline first for better visibility
            cv2.polylines(image, [np.array(path_pixels)], False, (0, 0, 0), 7)
            
            # Draw color gradient path
            for i in range(len(path_pixels) - 1):
                ratio = i / (len(path_pixels) - 1)
                # Color gradient: red (near) -> yellow -> green (far)
                if ratio < 0.5:
                    r, g, b = 255, int(255 * ratio * 2), 0
                else:
                    r, g, b = int(255 * (1 - (ratio - 0.5) * 2)), 255, 0
                
                # Draw line segments
                cv2.line(image, path_pixels[i], path_pixels[i + 1], (b, g, r), 4)
                
                # Add distance markers at intervals
                if i % 10 == 0 and i > 0:
                    x, y = path_pixels[i]
                    depth = self.path[i][1]
                    label = f"{depth:.1f}m"
                    cv2.putText(image, label, (x + 5, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(image, (x, y), 3, (255, 255, 255), -1)
            
            # Highlight steering target point
            lookahead_distance = 5.0
            target_idx = 0
            for i, (_, depth) in enumerate(self.path):
                if depth >= lookahead_distance:
                    target_idx = i
                    break
            
            if target_idx < len(path_pixels):
                tx, ty = path_pixels[target_idx]
                cv2.circle(image, (tx, ty), 8, (0, 0, 255), -1)
                cv2.line(image, (car_x, car_y), (tx, ty), (0, 0, 255), 2)
            
            # Add path info text
            path_info = f"Path: {len(self.path)} points, {self.path[-1][1]:.1f}m"
            cv2.putText(image, path_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Add steering info
            steering = self.calculate_steering()
            steer_text = f"Steering: {steering:.2f}"
            cv2.putText(image, steer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        return image
    
    def get_path(self):
        return self.path

    def plan_path_with_cones(self, cones, depths):
        """
        Plan a path using provided cone detections
        
        Args:
            cones: List of (center_x, center_y, cls) tuples
            depths: List of depth values for each cone
        """
        try:
            if not cones:
                print("No cone detections provided")
                self.path = None
                return
            
            print(f"Planning path with {len(cones)} provided cones")
            
            # Filter cones (if needed)
            filtered_cones, filtered_depths = self._filter_cones(cones, depths)
            if not filtered_cones:
                print("No cones after filtering")
                self.path = None
                return
            
            # Generate waypoints from cones
            waypoints = self._pair_cones(filtered_cones, filtered_depths)
            
            if waypoints:
                x_coords, y_coords = zip(*waypoints)
                smoothed_x, smoothed_y = self._smooth_path(x_coords, y_coords)
                new_path = list(zip(smoothed_x, smoothed_y))
                
                # Apply temporal smoothing
                new_path = self._smooth_path_temporally(new_path)
                
                # Sort the path by depth to ensure smooth drawing
                new_path.sort(key=lambda p: p[1])
                self.path = new_path
                
                print(f"Path successfully created with {len(self.path)} points!")
            else:
                print("No waypoints found")
                self.path = None
        
        except Exception as e:
            print(f"Error in path planning with cones: {e}")
            import traceback
            traceback.print_exc()
            self.path = None

    def _limit_path_length(self, waypoints, max_distance=15.0, min_points=10):
        """
        Limit the path length to a reasonable distance to prevent overextension.
        
        Args:
            waypoints: List of (x, depth) waypoints
            max_distance: Maximum forward distance in meters
            min_points: Minimum number of points to keep
            
        Returns:
            Limited waypoints
        """
        if not waypoints or len(waypoints) < min_points:
            return waypoints
        
        # Sort by depth
        sorted_waypoints = sorted(waypoints, key=lambda p: p[1])
        
        # Keep points up to max_distance
        limited_waypoints = [p for p in sorted_waypoints if p[1] <= max_distance]
        
        # Ensure we have at least min_points
        if len(limited_waypoints) < min_points and len(sorted_waypoints) >= min_points:
            limited_waypoints = sorted_waypoints[:min_points]
        
        # Add additional check for extrapolation reliability
        # If the furthest points are based on extrapolation without actual cone detections,
        # they might be less reliable, so adjust the curve to prevent extreme deviations
        
        if len(limited_waypoints) >= 3:
            # Check the last few points for extreme lateral deviation
            last_points = limited_waypoints[-3:]
            lateral_changes = [abs(p2[0] - p1[0]) for p1, p2 in zip(last_points[:-1], last_points[1:])]
            
            # If there's significant lateral change in the last segments
            if any(change > 1.0 for change in lateral_changes):
                # Calculate average x-position from stable portion of the path
                stable_portion = limited_waypoints[:-3]
                if stable_portion:
                    avg_x = sum(p[0] for p in stable_portion) / len(stable_portion)
                    
                    # Dampen extreme deviations in the last points
                    damping_factor = 0.7  # Adjust based on testing
                    for i in range(-3, 0):
                        # Blend with average x to prevent extreme extrapolation
                        current_x = limited_waypoints[i][0]
                        limited_waypoints[i] = (
                            damping_factor * current_x + (1 - damping_factor) * avg_x,
                            limited_waypoints[i][1]
                        )
        
        return limited_waypoints

    def _add_cone_safety_margins(self, waypoints, yellow_world, blue_world, safety_margin=0.3):
        """
        Add safety margins around cones to avoid getting too close or hitting them.
        
        Args:
            waypoints: List of (x, depth) waypoints
            yellow_world: List of (x, depth) for yellow cones
            blue_world: List of (x, depth) for blue cones
            safety_margin: Safety margin in meters
            
        Returns:
            Adjusted waypoints
        """
        if not waypoints or (not yellow_world and not blue_world):
            return waypoints
        
        adjusted_waypoints = []
        
        for wp_x, wp_depth in waypoints:
            # Find closest cones at this depth
            closest_yellow = None
            closest_blue = None
            best_y_dist = float('inf')
            best_b_dist = float('inf')
            
            # Find closest yellow cone
            for y_x, y_depth in yellow_world:
                dist = abs(y_depth - wp_depth)
                if dist < best_y_dist:
                    best_y_dist = dist
                    closest_yellow = (y_x, y_depth)
            
            # Find closest blue cone
            for b_x, b_depth in blue_world:
                dist = abs(b_depth - wp_depth)
                if dist < best_b_dist:
                    best_b_dist = dist
                    closest_blue = (b_x, b_depth)
            
            # Apply safety adjustments if cones are found
            adjusted_x = wp_x
            
            # Only adjust if we have close cones (within 3 meters depth-wise)
            if closest_yellow and best_y_dist < 3.0 and closest_blue and best_b_dist < 3.0:
                # We have both cones - check if too close to either
                y_x, _ = closest_yellow
                b_x, _ = closest_blue
                
                # Calculate distances to the cones
                dist_to_yellow = abs(wp_x - y_x)
                dist_to_blue = abs(wp_x - b_x)
                
                # Assuming yellow is on left (positive X) and blue on right (negative X)
                if dist_to_yellow < safety_margin:
                    # Too close to yellow cone, adjust rightward
                    adjusted_x = y_x - safety_margin
                elif dist_to_blue < safety_margin:
                    # Too close to blue cone, adjust leftward
                    adjusted_x = b_x + safety_margin
                    
                # Track width sanity check
                track_width = abs(y_x - b_x)
                if track_width > 2.0:
                    # Additional check to ensure we're still within the track
                    if adjusted_x > y_x - safety_margin:
                        adjusted_x = y_x - safety_margin
                    elif adjusted_x < b_x + safety_margin:
                        adjusted_x = b_x + safety_margin
                
            elif closest_yellow and best_y_dist < 3.0:
                # Only yellow cone - ensure safety margin
                y_x, _ = closest_yellow
                if wp_x > y_x - safety_margin:
                    adjusted_x = y_x - safety_margin
                    
            elif closest_blue and best_b_dist < 3.0:
                # Only blue cone - ensure safety margin
                b_x, _ = closest_blue
                if wp_x < b_x + safety_margin:
                    adjusted_x = b_x + safety_margin
            
            adjusted_waypoints.append((adjusted_x, wp_depth))
        
        return adjusted_waypoints

    def _detect_cones_early(self, lidar_points, min_confidence=0.6):
        """
        Enhanced early cone detection using LiDAR to detect cones before camera can see them.
        
        Args:
            lidar_points: Numpy array of shape (N, 3) containing LiDAR points
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of additional cone points (x, depth, cls)
        """
        if len(lidar_points) < 50:
            return []
        
        try:
            # Filter points to likely be cones
            ground_z = np.min(lidar_points[:, 2]) + 0.1  # Slightly above ground
            cone_mask = (lidar_points[:, 2] < ground_z + 0.5) & (lidar_points[:, 2] > ground_z)
            potential_cone_points = lidar_points[cone_mask]
            
            if len(potential_cone_points) < 10:
                return []
            
            # Cluster points to find cones
            from sklearn.cluster import DBSCAN
            
            # Consider only points in front of the vehicle and at reasonable distance
            front_mask = potential_cone_points[:, 1] > 0  # y > 0 (forward)
            distance_mask = np.sqrt(np.sum(potential_cone_points[:, :2]**2, axis=1)) < 25.0  # within 25m
            valid_points = potential_cone_points[front_mask & distance_mask]
            
            if len(valid_points) < 10:
                return []
            
            # Find clusters
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(valid_points[:, :3])
            labels = clustering.labels_
            
            # Process clusters to find cone-like objects
            detected_cones = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                    
                cluster = valid_points[labels == label]
                
                # Check if cluster has properties of a cone
                # Cones are small, relatively isolated objects
                if 5 <= len(cluster) <= 50:  # Not too few, not too many points
                    # Calculate cluster properties
                    center = np.mean(cluster, axis=0)
                    x, y, z = center
                    
                    # Calculate cluster dimensions
                    x_range = np.max(cluster[:, 0]) - np.min(cluster[:, 0])
                    y_range = np.max(cluster[:, 1]) - np.min(cluster[:, 1])
                    z_range = np.max(cluster[:, 2]) - np.min(cluster[:, 2])
                    
                    # Cone-like properties: small footprint, reasonable height
                    if max(x_range, y_range) < 0.7 and 0.1 < z_range < 0.5:
                        # This looks like a cone
                        
                        # Try to determine color by position (usually yellow on left, blue on right)
                        # This is a very rough heuristic - in a real system you'd use camera data or reflectivity
                        cone_type = 0 if x > 0 else 1  # 0 = yellow, 1 = blue
                        
                        # Calculate confidence based on point density and shape
                        point_density = len(cluster) / (x_range * y_range * z_range + 0.001)
                        shape_factor = min(1.0, max(0.1, 0.3 / max(x_range, y_range)))
                        confidence = min(0.9, 0.5 + point_density/500 + shape_factor)
                        
                        if confidence >= min_confidence:
                            # Convert to expected format
                            detected_cones.append((x, y, cone_type, confidence))
            
            return detected_cones
        
        except Exception as e:
            print(f"Error in early cone detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def plan_path_with_fusion(self, lidar_boundaries=None, lidar_points=None):
        """
        Plan path using camera-detected cones, LiDAR-detected boundaries,
        and enhanced early cone detection for improved performance.
        
        Args:
            lidar_boundaries: Dictionary with 'left_boundary' and 'right_boundary' points from LiDAR
            lidar_points: Raw LiDAR points for early cone detection
            
        Returns:
            True if path planning succeeded, False otherwise
        """
        try:
            # Process camera detections
            camera_cones = []
            camera_depths = []
            
            if hasattr(self.zed_camera, 'cone_detections') and self.zed_camera.cone_detections:
                for detection in self.zed_camera.cone_detections:
                    x1, y1, x2, y2 = detection['box']
                    cls = detection['cls']
                    depth = detection['depth']
                    print(f"Camera cone: Class = {cls}, Depth = {depth:.2f}m, Box = ({x1}, {y1}, {x2}, {y2})")
                    camera_cones.append((x1, y1, x2, y2, cls))
                    camera_depths.append(depth)
                
                print(f"Using {len(camera_cones)} detected cones from camera")
            
            # Detect early cones with LiDAR if available
            early_detected_cones = []
            if lidar_points is not None and len(lidar_points) > 0:
                early_detected_cones = self._detect_cones_early(lidar_points)
                print(f"Detected {len(early_detected_cones)} additional cones with LiDAR")
                
                # Convert LiDAR-detected cones to camera format for processing
                for x, y, cls, conf in early_detected_cones:
                    # Only add if far enough away (beyond reliable camera detection)
                    if y > 10.0:
                        # Calculate approximate image coordinates based on FOV
                        angle = np.arctan2(x, y)
                        image_center_x = self.image_width // 2
                        image_x = int(image_center_x + (angle / np.radians(self.fov_horizontal / 2)) * image_center_x)
                        
                        # Create estimated bounding box
                        box_size = int(100 / (y/10))  # Size decreases with distance
                        half_size = box_size // 2
                        x1 = max(0, image_x - half_size)
                        x2 = min(self.image_width - 1, image_x + half_size)
                        y1 = self.image_height // 2  # Approximate position
                        y2 = y1 + box_size
                        
                        # Add to camera detections list
                        camera_cones.append((x1, y1, x2, y2, cls))
                        camera_depths.append(y)
                        print(f"Added early-detected cone: Class={cls}, Depth={y:.2f}m")
            
            # Process combined cone list
            filtered_cones = []
            filtered_depths = []
            
            if camera_cones:
                filtered_cones, filtered_depths = self._filter_cones(camera_cones, camera_depths)
                print(f"After filtering: {len(filtered_cones)} cones")
            
            # Generate camera-based waypoints
            camera_waypoints = []
            yellow_world = []
            blue_world = []
            
            if filtered_cones:
                # Convert to world coordinates
                for cone, depth in zip(filtered_cones, filtered_depths):
                    center_x = cone[0]
                    # Calculate angle from image center
                    angle = ((center_x - self.image_width / 2) / (self.image_width / 2)) * (self.fov_horizontal / 2)
                    # Calculate world X coordinate
                    world_x = depth * np.tan(np.radians(angle))
                    
                    # Add to world coordinates list based on class
                    if cone[2] == 0:  # Yellow
                        yellow_world.append((world_x, depth))
                    else:  # Blue
                        blue_world.append((world_x, depth))
                
                # Update cone lists
                print(f"World coordinates: {len(yellow_world)} yellow cones, {len(blue_world)} blue cones")
                
                # Generate camera-based waypoints
                camera_waypoints = self._pair_cones(filtered_cones, filtered_depths)
            
            # Process LiDAR boundaries if available
            lidar_waypoints = []
            if lidar_boundaries and ('left_boundary' in lidar_boundaries) and ('right_boundary' in lidar_boundaries):
                left_boundary = lidar_boundaries['left_boundary']
                right_boundary = lidar_boundaries['right_boundary']
                
                if left_boundary and right_boundary:
                    print(f"Using LiDAR boundaries: {len(left_boundary)} left points, {len(right_boundary)} right points")
                    
                    # Enhance yellow/blue world lists with LiDAR boundary data
                    for point in left_boundary:
                        # Check if point is far enough to be useful
                        if point[1] > 10.0 and not any(abs(y[1] - point[1]) < 2.0 for y in yellow_world):
                            # This is a new boundary point not covered by camera
                            yellow_world.append(point)
                    
                    for point in right_boundary:
                        # Check if point is far enough to be useful
                        if point[1] > 10.0 and not any(abs(b[1] - point[1]) < 2.0 for b in blue_world):
                            # This is a new boundary point not covered by camera
                            blue_world.append(point)
                    
                    # Generate LiDAR-based waypoints from boundaries
                    all_depths = sorted(set([p[1] for p in left_boundary + right_boundary]))
                    for depth in all_depths:
                        # Find closest boundary points
                        closest_left = None
                        closest_right = None
                        best_left_dist = float('inf')
                        best_right_dist = float('inf')
                        
                        for point in left_boundary:
                            dist = abs(point[1] - depth)
                            if dist < best_left_dist:
                                best_left_dist = dist
                                closest_left = point
                        
                        for point in right_boundary:
                            dist = abs(point[1] - depth)
                            if dist < best_right_dist:
                                best_right_dist = dist
                                closest_right = point
                        
                        # If we have both boundaries at similar depths
                        if closest_left and closest_right and best_left_dist < 2.0 and best_right_dist < 2.0:
                            # Calculate midpoint
                            midpoint_x = (closest_left[0] + closest_right[0]) / 2
                            lidar_waypoints.append((midpoint_x, depth))
            
            # Fuse waypoints if we have data from both sources
            final_waypoints = []
            if camera_waypoints and lidar_waypoints:
                # Use confidence-weighted fusion
                final_waypoints = self._fuse_waypoints(camera_waypoints, lidar_waypoints)
                print(f"Fused {len(camera_waypoints)} camera waypoints with {len(lidar_waypoints)} LiDAR waypoints")
            elif camera_waypoints:
                final_waypoints = camera_waypoints
                print(f"Using {len(camera_waypoints)} camera waypoints")
            elif lidar_waypoints:
                final_waypoints = lidar_waypoints
                print(f"Using {len(lidar_waypoints)} LiDAR waypoints")
            else:
                print("No waypoints generated from any source")
                self.path = None
                return False
            
            # Add safety margins to avoid hitting cones
            if yellow_world or blue_world:
                final_waypoints = self._add_cone_safety_margins(final_waypoints, yellow_world, blue_world)
                print("Added safety margins to waypoints")
            
            # Limit path length to prevent extreme extrapolation
            final_waypoints = self._limit_path_length(final_waypoints)
            print(f"Limited path length: final path has {len(final_waypoints)} points")
            
            # Apply smoothing
            if final_waypoints:
                x_coords, y_coords = zip(*final_waypoints)
                smoothed_x, smoothed_y = self._smooth_path(x_coords, y_coords)
                new_path = list(zip(smoothed_x, smoothed_y))
                
                # Apply temporal smoothing
                new_path = self._smooth_path_temporally(new_path)
                
                # Sort the path by depth to ensure smooth drawing
                new_path.sort(key=lambda p: p[1])
                self.path = new_path
                
                print(f"Path successfully created with {len(self.path)} points")
                return True
            
            print("No valid path could be created")
            self.path = None
            return False
                
        except Exception as e:
            print(f"Error in path planning: {e}")
            import traceback
            traceback.print_exc()
            self.path = None
            return False

