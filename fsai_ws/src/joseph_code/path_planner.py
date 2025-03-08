import numpy as np
import time
import threading
import math
from collections import deque

def calculate_lookahead_distance(current_speed, min_dist=50, max_dist=150, k=0.5):
    """
    Calculate a dynamic lookahead distance based on current speed.
    
    Args:
        current_speed: Vehicle speed in m/s
        min_dist: Minimum lookahead distance in pixels
        max_dist: Maximum lookahead distance in pixels
        k: Scaling factor
    
    Returns:
        Lookahead distance in pixels
    """
    # Ensure speed is positive
    speed = max(0.1, current_speed)
    
    # Scale lookahead with speed (higher speed = further lookahead)
    lookahead = min_dist + k * speed * 50
    
    # Clamp to reasonable bounds
    return min(max_dist, max(min_dist, lookahead))

def find_target_point(path_points, current_position, lookahead_distance):
    """
    Find a target point on the path that is approximately lookahead_distance away from current position.
    
    Args:
        path_points: List of [x, y] points defining the path
        current_position: Current vehicle position [x, y]
        lookahead_distance: Desired lookahead distance
    
    Returns:
        Target point [x, y] or None if no suitable point is found
    """
    # Check if path_points is None or empty
    if path_points is None or isinstance(path_points, list) and len(path_points) == 0:
        return None

    # Check if it's a numpy array with data
    if hasattr(path_points, 'size') and path_points.size == 0:
        return None
        
    # Make sure we have at least 2 points
    if hasattr(path_points, 'shape') and len(path_points.shape) > 0 and path_points.shape[0] < 2:
        return None
    
    # Convert to numpy arrays for easier math
    if not isinstance(path_points, np.ndarray):
        path_points = np.array(path_points)
    
    if not isinstance(current_position, np.ndarray):
        current_position = np.array(current_position)
    
    # Find closest point on path to current position
    dists = np.sum((path_points - current_position) ** 2, axis=1)
    closest_idx = np.argmin(dists)
    
    # Look ahead from closest point
    for i in range(closest_idx, len(path_points)):
        # Calculate distance from current position to this path point
        dist = np.linalg.norm(path_points[i] - current_position)
        
        # If we've reached our desired lookahead distance
        if dist >= lookahead_distance:
            return path_points[i]
    
    # If we couldn't find a point at exactly the lookahead distance,
    # return the last point on the path
    if path_points.shape[0] > 0:
        return path_points[-1]
    else:
        return None

class PathGenerator:
    """Class to generate a path between detected cones"""
    
    def __init__(self, image_width=640, image_height=384):
        self.image_width = image_width
        self.image_height = image_height
        self.center_line_smooth_factor = 0.5
        self.path_smooth_factor = 0.7
        self.min_cone_confidence = 0.3
        self.min_cone_distance = 20  # pixels
        self.racing_line_bias = -0.1  # Negative values favor blue side
    
    def _filter_cones(self, yellow_cones, blue_cones):
        """Filter cones by confidence and remove duplicates"""
        # Filter by confidence
        if yellow_cones is not None and len(yellow_cones) > 0:
            yellow_cones = yellow_cones[yellow_cones[:, 4] > self.min_cone_confidence]
        else:
            yellow_cones = np.array([])
            
        if blue_cones is not None and len(blue_cones) > 0:
            blue_cones = blue_cones[blue_cones[:, 4] > self.min_cone_confidence]
        else:
            blue_cones = np.array([])
        
        # Extract centers of cones
        yellow_centers = []
        if len(yellow_cones) > 0:
            for cone in yellow_cones:
                x1, y1, x2, y2 = cone[:4]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                yellow_centers.append([center_x, center_y])
        
        blue_centers = []
        if len(blue_cones) > 0:
            for cone in blue_cones:
                x1, y1, x2, y2 = cone[:4]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                blue_centers.append([center_x, center_y])
        
        # Convert to numpy arrays
        yellow_centers = np.array(yellow_centers)
        blue_centers = np.array(blue_centers)
        
        return yellow_centers, blue_centers
    
    def _calculate_center_line(self, yellow_centers, blue_centers):
        """Calculate center line between yellow and blue cones"""
        if len(yellow_centers) == 0 or len(blue_centers) == 0:
            return np.array([])
        
        # Sort cones by y-coordinate (distance from car)
        if len(yellow_centers) > 0:
            yellow_centers = yellow_centers[yellow_centers[:, 1].argsort()]
        
        if len(blue_centers) > 0:
            blue_centers = blue_centers[blue_centers[:, 1].argsort()]
        
        # Match yellow and blue cones by proximity in y-coordinate
        center_points = []
        
        # Use the longer array as the reference
        if len(yellow_centers) >= len(blue_centers):
            reference = yellow_centers
            comparison = blue_centers
            left_side = False
        else:
            reference = blue_centers
            comparison = yellow_centers
            left_side = True
        
        # Track previous track width to detect parallel tracks
        # Only for left turns (when blue cones are on left and yellow on right)
        prev_width = None
        
        for ref_cone in reference:
            # Find closest cone from other color by y-coordinate
            if len(comparison) == 0:
                continue
                    
            dists = np.abs(comparison[:, 1] - ref_cone[1])
            closest_idx = np.argmin(dists)
            closest_cone = comparison[closest_idx]
            
            # Only pair cones if they're reasonably close in y-coordinate
            y_diff = abs(ref_cone[1] - closest_cone[1])
            if y_diff < 50:  # Threshold for vertical proximity
                
                # NEW: Only apply parallel track filtering for left-side scenarios
                # This ensures right turns are completely unaffected
                if not left_side:  # Blue cones on left, yellow on right (normal orientation)
                    current_width = abs(ref_cone[0] - closest_cone[0])
                    
                    # Check if we have at least 3 blue cones and the rightmost blue cone (comparison) is 
                    # significantly right of where it should be (parallel track detection)
                    if prev_width is not None and len(comparison) >= 3:
                        # Calculate x-difference between consecutive blue cones
                        # This can indicate if there's a sudden shift that might be a parallel track
                        blue_x_positions = [cone[0] for cone in comparison]
                        
                        # Check for blue cones that are unusually far to the right
                        if closest_cone[0] > 320:  # Right of center
                            # Look for another blue cone that might be more appropriate
                            for idx, dist in enumerate(dists):
                                if dist < 80:  # Reasonable vertical alignment
                                    alt_cone = comparison[idx]
                                    # If this cone is more to the left where blue cones should be
                                    if alt_cone[0] < closest_cone[0] - 50:  # Significantly more left
                                        closest_cone = alt_cone
                                        print("Filtered out potential parallel track blue cone")
                                        break
                
                # Calculate center point
                if left_side:
                    # yellow on right, blue on left
                    center_x = (closest_cone[0] + ref_cone[0]) / 2
                else:
                    # yellow on left, blue on right
                    center_x = (ref_cone[0] + closest_cone[0]) / 2
                    
                center_y = (ref_cone[1] + closest_cone[1]) / 2
                center_points.append([center_x, center_y])
                
                # Update track width
                prev_width = abs(ref_cone[0] - closest_cone[0])
        
        # Handle case where we couldn't pair any cones
        if not center_points:
            # Create a simple center line assuming yellow cones on right, blue on left
            all_x = np.mean([np.mean(yellow_centers[:, 0]) if len(yellow_centers) > 0 else self.image_width,
                            np.mean(blue_centers[:, 0]) if len(blue_centers) > 0 else 0])
            
            # Use bottom of image for y if we have no other reference
            center_points = [[all_x, self.image_height - 50]]
        
        return np.array(center_points)
    
    def _smooth_path(self, points, smooth_factor=0.5):
        """Apply a simple moving average smoothing to the path"""
        if len(points) <= 2:
            return points
            
        smoothed = np.copy(points)
        for i in range(1, len(points) - 1):
            smoothed[i] = points[i] * (1 - smooth_factor) + (points[i-1] + points[i+1]) * smooth_factor / 2
            
        return smoothed
    
    def _optimize_racing_line(self, center_line, yellow_centers, blue_centers, use_racing_line=True):
        """Optimize the racing line to take corners efficiently"""
        if not use_racing_line or len(center_line) < 3:
            return center_line
            
        racing_line = np.copy(center_line)
        
        # Apply global racing line bias (negative = favor blue side)
        if self.racing_line_bias != 0:
            # Shift the entire racing line
            for i in range(len(center_line)):
                # Create a vector perpendicular to the path
                # For simplicity, we'll just use a horizontal shift
                racing_line[i][0] += self.racing_line_bias * 30  # Scale the bias
        
        # Simple optimization: shift racing line toward inside of turns
        for i in range(1, len(center_line) - 1):
            prev_pt = center_line[i-1]
            curr_pt = center_line[i]
            next_pt = center_line[i+1]
            
            # Calculate direction vectors
            dir1 = curr_pt - prev_pt
            dir2 = next_pt - curr_pt
            
            # Normalize
            dir1 = dir1 / max(np.linalg.norm(dir1), 1e-6)
            dir2 = dir2 / max(np.linalg.norm(dir2), 1e-6)
            
            # Check if turning
            turn_indicator = np.cross(dir1, dir2)
            
            # Positive cross product means turning left (shift toward right/yellow cones)
            # Negative cross product means turning right (shift toward left/blue cones)
            shift = 0
            if abs(turn_indicator) > 0.2:  # Only apply if actually turning
                shift_factor = min(0.3, abs(turn_indicator))
                if turn_indicator > 0:
                    # Turning left, shift right
                    if len(yellow_centers) > 0:
                        # Calculate safe shift distance based on distance to yellow cones
                        dists = np.sum((yellow_centers - curr_pt) ** 2, axis=1)
                        min_dist = np.sqrt(np.min(dists)) if len(dists) > 0 else 100
                        shift = min(min_dist * 0.3, 20) * shift_factor
                else:
                    # Turning right, shift left
                    if len(blue_centers) > 0:
                        # Calculate safe shift distance based on distance to blue cones
                        dists = np.sum((blue_centers - curr_pt) ** 2, axis=1)
                        min_dist = np.sqrt(np.min(dists)) if len(dists) > 0 else 100
                        shift = -min(min_dist * 0.3, 20) * shift_factor
            
            # Apply shift perpendicular to path
            perp = np.array([-dir1[1], dir1[0]])  # Perpendicular vector
            racing_line[i] = curr_pt + perp * shift
        
        return racing_line
    
    def generate_path(self, yellow_cones, blue_cones, use_racing_line=True):
        """
        Generate a smooth path based on detected yellow and blue cones.
        
        Args:
            yellow_cones: Numpy array of yellow cone detections [x1, y1, x2, y2, conf, class_id]
            blue_cones: Numpy array of blue cone detections [x1, y1, x2, y2, conf, class_id]
            use_racing_line: Whether to optimize for racing line or just stay centered
            
        Returns:
            path_points: Numpy array of path points as [x, y]
            curvature: Estimated curvature of the path
        """
        # Filter cones
        yellow_centers, blue_centers = self._filter_cones(yellow_cones, blue_cones)
        
        # If we don't have enough cones of both colors, we can't make a path
        if len(yellow_centers) == 0 and len(blue_centers) == 0:
            return np.array([]), 0
        
        # If we don't have any cones of one color, use a fallback
        if len(yellow_centers) == 0:
            # Only blue cones visible - assume yellow is to the right
            yellow_centers = np.array([[blue_center[0] + 300, blue_center[1]] for blue_center in blue_centers])
        
        if len(blue_centers) == 0:
            # Only yellow cones visible - assume blue is to the left
            blue_centers = np.array([[yellow_center[0] - 300, yellow_center[1]] for yellow_center in yellow_centers])
        
        # Calculate center line between cones
        center_line = self._calculate_center_line(yellow_centers, blue_centers)
        
        if len(center_line) == 0:
            return np.array([]), 0
        
        # Smooth the center line
        center_line = self._smooth_path(center_line, self.center_line_smooth_factor)
        
        # Optimize for racing line if requested
        if use_racing_line:
            path_points = self._optimize_racing_line(center_line, yellow_centers, blue_centers, use_racing_line)
        else:
            path_points = center_line
        
        # Apply final smoothing to the path
        path_points = self._smooth_path(path_points, self.path_smooth_factor)
        
        # Calculate average curvature (a simple approximation)
        curvature = 0
        if len(path_points) >= 3:
            angles = []
            for i in range(1, len(path_points) - 1):
                v1 = path_points[i] - path_points[i-1]
                v2 = path_points[i+1] - path_points[i]
                
                # Normalize vectors
                v1 = v1 / max(np.linalg.norm(v1), 1e-6)
                v2 = v2 / max(np.linalg.norm(v2), 1e-6)
                
                # Calculate angle
                dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot_product)
                angles.append(angle)
            
            if angles:
                curvature = np.mean(angles)
        
        return path_points, curvature

class AsyncPathPlanner:
    """
    Asynchronous path planner that runs in a separate thread to compute paths.
    """
    
    def __init__(self, image_width=640, image_height=384, use_racing_line=True):
        self.image_width = image_width
        self.image_height = image_height
        self.path_generator = PathGenerator(image_width, image_height)
        self.use_racing_line = use_racing_line
        
        # Thread control
        self.running = True
        self.needs_update = False
        self.lock = threading.Lock()
        
        # Data
        self.yellow_cones = None
        self.blue_cones = None
        self.path_points = None
        self.curvature = 0
        
        # Start thread
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def update_cones(self, yellow_cones, blue_cones):
        """
        Update the cone detections.
        
        Args:
            yellow_cones: Numpy array of yellow cone detections
            blue_cones: Numpy array of blue cone detections
        """
        with self.lock:
            self.yellow_cones = yellow_cones.copy() if yellow_cones is not None and len(yellow_cones) > 0 else None
            self.blue_cones = blue_cones.copy() if blue_cones is not None and len(blue_cones) > 0 else None
            self.needs_update = True
    
    def get_latest_path(self):
        """
        Get the latest computed path.
        
        Returns:
            path_points: Numpy array of path points as [x, y]
            curvature: Estimated curvature of the path
        """
        with self.lock:
            if self.path_points is not None:
                return self.path_points.copy(), self.curvature
            else:
                return None, 0
    
    def _update_loop(self):
        """Thread loop for path computation"""
        while self.running:
            local_yellow_cones = None
            local_blue_cones = None
            
            # Check if we need to update the path
            with self.lock:
                if self.needs_update:
                    # Make local copies of the cone data
                    if self.yellow_cones is not None:
                        local_yellow_cones = self.yellow_cones.copy()
                    if self.blue_cones is not None:
                        local_blue_cones = self.blue_cones.copy()
                    
                    self.needs_update = False
            
            # Generate path if we have cone data
            if local_yellow_cones is not None or local_blue_cones is not None:
                try:
                    path_points, curvature = self.path_generator.generate_path(
                        local_yellow_cones, 
                        local_blue_cones,
                        self.use_racing_line
                    )
                    
                    # Update the path
                    with self.lock:
                        self.path_points = path_points
                        self.curvature = curvature
                except Exception as e:
                    print(f"Path generation error: {e}")
            
            # Sleep a bit to avoid consuming too much CPU
            time.sleep(0.01)
    
    def shutdown(self):
        """Stop the background thread"""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            print("Path planner thread stopped")