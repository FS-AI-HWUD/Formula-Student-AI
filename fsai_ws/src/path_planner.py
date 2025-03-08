import numpy as np
import scipy.interpolate as interpolate
import threading
import queue
import time

class PathPlanner:
    def __init__(self):
        self.look_ahead_points = 5
        self.path_resolution = 20
        self.min_cones_for_path = 2
        self.max_path_width = 400
        self.last_valid_path = None
        self.last_valid_curvature = None
        
    def generate_path(self, yellow_cones, blue_cones):
        """
        Generate a smooth path through the center of yellow and blue cones.
        
        Args:
            yellow_cones: Numpy array of yellow cone detections [x1, y1, x2, y2, conf, cls]
            blue_cones: Numpy array of blue cone detections [x1, y1, x2, y2, conf, cls]
            
        Returns:
            tuple: (path_points, curvature) or (last_valid_path, None) if path generation fails
        """
        try:
            if yellow_cones is None or blue_cones is None:
                return self.last_valid_path, self.last_valid_curvature
                
            if len(yellow_cones) < self.min_cones_for_path or len(blue_cones) < self.min_cones_for_path:
                return self.last_valid_path, self.last_valid_curvature
                
            # Calculate cone centers
            yellow_centers = np.array([
                [(cone[0] + cone[2])/2, (cone[1] + cone[3])/2]
                for cone in yellow_cones
            ])
            
            blue_centers = np.array([
                [(cone[0] + cone[2])/2, (cone[1] + cone[3])/2]
                for cone in blue_cones
            ])
            
            # Sort cones by y-coordinate (distance from car)
            yellow_sorted = yellow_centers[yellow_centers[:, 1].argsort()]
            blue_sorted = blue_centers[blue_centers[:, 1].argsort()]
            
            # Calculate track direction vector from previous path if available
            track_direction = np.array([0, 1])  # Default to straight ahead
            if self.last_valid_path is not None and len(self.last_valid_path) > 3:
                # Use the last segment of the previous path to estimate direction
                last_segment = self.last_valid_path[-3:]
                track_direction = last_segment[-1] - last_segment[0]
                if np.linalg.norm(track_direction) > 0:
                    track_direction = track_direction / np.linalg.norm(track_direction)
            
            # Smarter cone pairing based on track direction and position
            centerline_points = []
            used_yellow = set()
            used_blue = set()
            
            # Process each yellow cone to find the best matching blue cone
            for i, yellow in enumerate(yellow_sorted):
                if i in used_yellow:
                    continue
                    
                best_match = None
                best_distance = float('inf')
                best_idx = -1
                
                # Find the closest blue cone that forms a reasonable track width
                for j, blue in enumerate(blue_sorted):
                    if j in used_blue:
                        continue
                        
                    # Skip if blue cone is too far ahead or behind yellow cone
                    y_diff = abs(yellow[1] - blue[1])
                    if y_diff > 100:  # Adjusted based on your image size
                        continue
                        
                    # Calculate distance and check if it's a reasonable track width
                    width_vector = yellow - blue
                    distance = np.linalg.norm(width_vector)
                    
                    if distance > self.max_path_width:
                        continue
                    
                    # Check if orientation is reasonable (perpendicular to track direction)
                    if distance > 0:
                        norm_width = width_vector / distance
                        # Width should be perpendicular to track direction
                        alignment = abs(np.dot(norm_width, track_direction))
                        if alignment > 0.7:  # If width vector is too aligned with track direction
                            continue
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = blue
                        best_idx = j
                
                # If we found a matching blue cone, create a midpoint
                if best_match is not None:
                    mid_point = [(yellow[0] + best_match[0])/2, (yellow[1] + best_match[1])/2]
                    centerline_points.append(mid_point)
                    used_yellow.add(i)
                    used_blue.add(best_idx)
            
            # Check if we have enough points for a path
            if len(centerline_points) < self.min_cones_for_path:
                return self.last_valid_path, self.last_valid_curvature
                
            # Convert to numpy array for processing
            centerline_points = np.array(centerline_points)
            
            # Sort points by y-coordinate to ensure proper ordering
            centerline_points = centerline_points[centerline_points[:, 1].argsort()]
            
            # Generate spline - simple path if only few points
            if len(centerline_points) < 3:
                # Simple linear path
                t = np.linspace(0, 1, self.path_resolution)
                path_points = np.zeros((self.path_resolution, 2))
                
                # Linear interpolation
                for i in range(self.path_resolution):
                    path_points[i] = centerline_points[0] * (1-t[i]) + centerline_points[-1] * t[i]
                
                self.last_valid_path = path_points
                self.last_valid_curvature = None
                return path_points, None
            else:
                # Spline for more complex paths
                t = np.linspace(0, 1, len(centerline_points))
                t_new = np.linspace(0, 1, self.path_resolution)
                
                # Create cubic spline interpolation
                cs_x = interpolate.CubicSpline(t, centerline_points[:, 0])
                cs_y = interpolate.CubicSpline(t, centerline_points[:, 1])
                
                path_points = np.column_stack((cs_x(t_new), cs_y(t_new)))
                
                # Calculate curvature for speed control
                dx = np.gradient(cs_x(t_new))
                dy = np.gradient(cs_y(t_new))
                d2x = np.gradient(dx)
                d2y = np.gradient(dy)
                curvature = np.abs(dx * d2y - dy * d2x) / (dx * dx + dy * dy)**1.5
                
                self.last_valid_path = path_points
                self.last_valid_curvature = curvature
                return path_points, curvature
            
        except Exception as e:
            print(f"Path generation error: {e}")
            return self.last_valid_path, self.last_valid_curvature


class AsyncPathPlanner:
    """
    Asynchronous wrapper for the PathPlanner class that runs path planning
    in a separate thread to avoid blocking the main perception loop.
    """
    def __init__(self):
        self.path_planner = PathPlanner()
        self.path_queue = queue.Queue(maxsize=1)  # Only keep latest path
        self.running = True
        self.yellow_cones = None
        self.blue_cones = None
        self.new_data = threading.Event()
        self.lock = threading.Lock()
        self.last_planning_time = 0
        self.min_planning_interval = 0.05  # Minimum 50ms between path planning updates
        
        # Start the planning thread
        self.thread = threading.Thread(target=self._planning_worker)
        self.thread.daemon = True
        self.thread.start()
    
    def update_cones(self, yellow_cones, blue_cones):
        """
        Update the cone detections for path planning.
        
        Args:
            yellow_cones: Numpy array of yellow cone detections
            blue_cones: Numpy array of blue cone detections
        """
        with self.lock:
            self.yellow_cones = yellow_cones.copy() if yellow_cones is not None else None
            self.blue_cones = blue_cones.copy() if blue_cones is not None else None
        self.new_data.set()  # Signal that new data is available
    
    def get_latest_path(self):
        """
        Get the most recently computed path.
        
        Returns:
            tuple: (path_points, curvature) or (None, None) if no path is available
        """
        try:
            return self.path_queue.get_nowait()
        except queue.Empty:
            # Return the last valid path if available
            return self.path_planner.last_valid_path, self.path_planner.last_valid_curvature
    
    def _planning_worker(self):
        """
        Worker function that runs in a separate thread to compute paths.
        """
        while self.running:
            # Wait for new data
            self.new_data.wait(timeout=0.1)
            
            # Skip if minimum interval hasn't passed
            current_time = time.time()
            if current_time - self.last_planning_time < self.min_planning_interval:
                time.sleep(0.01)  # Short sleep to prevent CPU spinning
                continue
                
            if not self.new_data.is_set():
                continue
            
            # Clear the event
            self.new_data.clear()
            
            # Generate path
            yellow_cones = None
            blue_cones = None
            
            with self.lock:
                if self.yellow_cones is not None:
                    yellow_cones = self.yellow_cones.copy()
                if self.blue_cones is not None:
                    blue_cones = self.blue_cones.copy()
            
            if yellow_cones is not None and blue_cones is not None:
                # Record planning start time
                start_time = time.time()
                
                # Generate the path
                path_points, curvature = self.path_planner.generate_path(
                    yellow_cones, blue_cones)
                
                # Update queue (overwrite old path)
                try:
                    self.path_queue.get_nowait()  # Remove old path if exists
                except queue.Empty:
                    pass
                    
                if path_points is not None:
                    self.path_queue.put((path_points, curvature))
                
                # Record planning end time
                self.last_planning_time = time.time()
                planning_duration = self.last_planning_time - start_time
                
                # Occasionally log planning performance
                if np.random.random() < 0.05:  # Log roughly 5% of the time
                    print(f"Path planning took {planning_duration*1000:.1f}ms")
    
    def shutdown(self):
        """
        Cleanly shut down the planning thread.
        """
        self.running = False
        self.new_data.set()  # Wake up the thread if it's waiting
        self.thread.join(timeout=1.0)
        print("AsyncPathPlanner shut down.")


# Additional utility functions for path planning

def calculate_lookahead_distance(speed, min_dist=50, max_dist=200, k=0.5):
    """
    Calculate an appropriate lookahead distance based on current speed.
    
    Args:
        speed: Current speed in m/s
        min_dist: Minimum lookahead distance in pixels
        max_dist: Maximum lookahead distance in pixels
        k: Speed scaling factor
        
    Returns:
        float: Lookahead distance in pixels
    """
    # Linear scaling with speed, capped at min/max
    distance = min_dist + k * speed * 100  # Convert speed to pixel space
    return np.clip(distance, min_dist, max_dist)


def find_target_point(path_points, current_pos, lookahead_distance):
    """
    Find a target point on the path that is approximately lookahead_distance away.
    
    Args:
        path_points: Array of path points [[x1,y1], [x2,y2], ...]
        current_pos: Current position [x, y]
        lookahead_distance: Desired lookahead distance
        
    Returns:
        array: Target point [x, y]
    """
    if path_points is None or len(path_points) < 2:
        return None
    
    # Find closest point on path to current position
    distances = np.linalg.norm(path_points - np.array(current_pos), axis=1)
    closest_idx = np.argmin(distances)
    
    # Look ahead from the closest point
    for i in range(closest_idx, len(path_points)):
        distance = np.linalg.norm(path_points[i] - np.array(current_pos))
        if distance >= lookahead_distance:
            return path_points[i]
    
    # If we can't find a point far enough ahead, return the furthest point
    return path_points[-1]
