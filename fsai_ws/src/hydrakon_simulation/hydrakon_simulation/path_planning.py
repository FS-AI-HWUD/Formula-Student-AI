import numpy as np
import cv2
import math
from typing import List, Tuple, Dict, Optional, Union

"""
Pure Pursuit Path Planner for Formula Student

This module implements a path planning system using the pure pursuit algorithm
specifically designed for cone-based Formula Student autonomous racing.

It focuses on looking ahead for 3 cone pairs to determine the optimal path
and steering angle based on the pure pursuit algorithm.

Author: Atlas Racing
"""

class WorldCone:
    """Class to store cone in world coordinates."""
    def __init__(self, x, depth, cls, confidence=1.0):
        self.x = x          # Lateral position in meters (positive = left, negative = right)
        self.depth = depth  # Forward distance in meters
        self.cls = cls      # 0 for yellow, 1 for blue
        self.confidence = confidence  # Detection confidence


class ConePair:
    """Class to store a pair of cones (one yellow, one blue)."""
    def __init__(self, yellow, blue, midpoint, width, valid=True):
        self.yellow = yellow          # Yellow cone (WorldCone or None)
        self.blue = blue              # Blue cone (WorldCone or None)
        self.midpoint = midpoint      # (x, depth) of midpoint
        self.width = width            # Track width at this pair
        self.valid = valid            # Whether this pair is valid
    
    @classmethod
    def from_cones(cls, yellow, blue, default_width=3.5):
        """Create a cone pair from yellow and blue cones."""
        if yellow and blue:
            midpoint_x = (yellow.x + blue.x) / 2
            midpoint_depth = (yellow.depth + blue.depth) / 2
            width = abs(yellow.x - blue.x)
            return cls(yellow, blue, (midpoint_x, midpoint_depth), width)
        elif yellow:
            # Only yellow cone available, estimate blue position
            midpoint_x = yellow.x - (default_width / 2)
            return cls(yellow, None, (midpoint_x, yellow.depth), default_width)
        elif blue:
            # Only blue cone available, estimate yellow position
            midpoint_x = blue.x + (default_width / 2)
            return cls(None, blue, (midpoint_x, blue.depth), default_width)
        else:
            # Should not happen, but for safety
            return cls(None, None, (0, 0), default_width, valid=False)


class ConePair:
    """Class to store a pair of cones (one yellow, one blue)."""
    def __init__(self, yellow, blue, midpoint, width, valid=True):
        self.yellow = yellow          # Yellow cone (WorldCone or None)
        self.blue = blue              # Blue cone (WorldCone or None)
        self.midpoint = midpoint      # (x, depth) of midpoint
        self.width = width            # Track width at this pair
        self.valid = valid            # Whether this pair is valid
    
    @classmethod
    def from_cones(cls, yellow, blue, default_width=3.5):
        """Create a cone pair from yellow and blue cones."""
        if yellow and blue:
            midpoint_x = (yellow.x + blue.x) / 2
            midpoint_depth = (yellow.depth + blue.depth) / 2
            width = abs(yellow.x - blue.x)
            return cls(yellow, blue, (midpoint_x, midpoint_depth), width)
        elif yellow:
            # Only yellow cone available, estimate blue position
            midpoint_x = yellow.x - (default_width / 2)
            return cls(yellow, None, (midpoint_x, yellow.depth), default_width)
        elif blue:
            # Only blue cone available, estimate yellow position
            midpoint_x = blue.x + (default_width / 2)
            return cls(None, blue, (midpoint_x, blue.depth), default_width)
        else:
            # Should not happen, but for safety
            return cls(None, None, (0, 0), default_width, valid=False)

class PathPlanner:
    """
    Path planner using pure pursuit algorithm for Formula Student vehicle.
    Focuses on looking ahead for 3 cone pairs to determine optimal path.
    Maintains safe margins from cones and treats cones as gates.
    """
    
    def __init__(self, 
             zed_camera, 
             depth_min=0.5, 
             depth_max=12.5, 
             cone_spacing=1.5,
             visualize=True):
        """
        Initialize the pure pursuit path planner.
        
        Args:
            zed_camera: Camera object that provides cone detections
            depth_min: Minimum depth for cone detection in meters
            depth_max: Maximum depth for cone detection in meters
            cone_spacing: Expected spacing between cones in meters
            visualize: Whether to visualize the path
        """
        self.zed_camera = zed_camera
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.cone_spacing = cone_spacing
        self.visualize = visualize
        
        # Constants
        self.image_width = zed_camera.resolution[0]
        self.image_height = zed_camera.resolution[1]
        self.fov_horizontal = 90.0  # Camera horizontal field of view in degrees
        self.wheelbase = 2.7        # Vehicle wheelbase in meters
        
        # Pure pursuit parameters
        self.lookahead_distance = 4.0  # Base lookahead distance in meters
        self.max_lookahead_pairs = 3   # Maximum number of cone pairs to look ahead
        self.default_track_width = 3.5 # Default track width in meters
        
        # Safety margin from cones (in meters)
        self.safety_margin = 0.6       # Keep this distance from cones
        
        # State variables
        self.cone_pairs = []           # List of ConePair objects
        self.path = None               # Current path (for compatibility)
        self.target_point = None       # Target point for pure pursuit (x, depth)
        self.steering_angle = 0.0      # Current steering angle
        self.current_track_width = self.default_track_width
        
        # Track visualization - store blue and yellow cone connections
        self.blue_boundary = []        # List of blue cone positions (x, depth)
        self.yellow_boundary = []      # List of yellow cone positions (x, depth)
        
        # U-turn detection
        self.in_uturn = False          # Whether we're in a U-turn
        self.uturn_side = "none"       # Which side (yellow or blue) the U-turn is on
        self.turn_radius = self.default_track_width * 1.5  # Default turn radius
        
        # Previous state for smoothing
        self.previous_path = None      # For compatibility with old code
        self.prev_steering_angle = 0.0
        self.prev_target_point = None
        
        # Debug image for visualization
        self.debug_image = None

    def update(self) -> float:
        """
        Update the planner with new detections and calculate steering angle.
        
        Returns:
            float: Normalized steering angle in range [-1, 1]
        """
        try:
            # Process detections
            self._process_detections()
            
            # Connect the cones to form track boundaries
            self._connect_boundary_cones()
            
            # Find target point
            self._find_target_point()
            
            # Calculate steering
            self._calculate_steering()
            
            # Smooth steering
            self._smooth_steering()
            
            # Always visualize when called
            if self.visualize and hasattr(self.zed_camera, 'rgb_image') and self.zed_camera.rgb_image is not None:
                try:
                    self.debug_image = self.draw_path(self.zed_camera.rgb_image.copy())
                    cv2.imshow("Pure Pursuit Path", self.debug_image)
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"Warning: Visualization failed: {str(e)}")
            
            return self.steering_angle
        except Exception as e:
            print(f"Error in pure pursuit update: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def plan_path(self):
        """
        Process cone detections and plan a path using pure pursuit algorithm.
        This method is the main entry point and matches the original interface.
        """
        try:
            # Process detections to create cone pairs
            self._process_detections()
            
            # Connect the cones to form track boundaries
            self._connect_boundary_cones()
            
            # Find target point for pure pursuit
            self._find_target_point()
            
            # Calculate steering angle
            self._calculate_steering()
            
            # Smooth steering
            self._smooth_steering()
            
            # Create path from cone pairs for compatibility with old code
            self._create_compatible_path()
            
            # Always visualize when called
            if self.visualize and hasattr(self.zed_camera, 'rgb_image') and self.zed_camera.rgb_image is not None:
                try:
                    self.debug_image = self.draw_path(self.zed_camera.rgb_image.copy())
                    cv2.imshow("Pure Pursuit Path", self.debug_image)
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"Warning: Visualization failed: {str(e)}")
            
            print(f"Pure pursuit path planned with {len(self.cone_pairs)} cone pairs")
            
        except Exception as e:
            print(f"Error in pure pursuit path planning: {str(e)}")
            import traceback
            traceback.print_exc()
            self.path = None
    
    def _process_detections(self):
        """Process cone detections from ZED camera."""
        # Get cone detections
        if not hasattr(self.zed_camera, 'cone_detections') or not self.zed_camera.cone_detections:
            print("No cone detections available")
            self.cone_pairs = []
            return
        
        # Extract cone data
        cones = []
        depths = []
        for detection in self.zed_camera.cone_detections:
            if 'box' not in detection or 'cls' not in detection or 'depth' not in detection:
                continue
                
            x1, y1, x2, y2 = detection['box']
            cls = detection['cls']
            depth = detection['depth']
            print(f"Using cone: Class = {cls}, Depth = {depth:.2f}m, Box = ({x1}, {y1}, {x2}, {y2})")
            cones.append((x1, y1, x2, y2, cls))
            depths.append(depth)
        
        # Filter cones using existing method for compatibility
        filtered_cones, filtered_depths = self._filter_cones(cones, depths)
        
        if not filtered_cones:
            print("No cones after filtering")
            self.cone_pairs = []
            return
        
        # Convert to world coordinates
        world_cones = self._to_world_coordinates(filtered_cones, filtered_depths)
        
        # Pair cones
        self._pair_cones(world_cones)
    
    def _filter_cones(self, cones, depths):
        """
        Filter cones based on depth and other criteria.
        Reuses the method from the original code for compatibility.
        """
        filtered_cones = []
        filtered_depths = []
        
        print("Filtering cones...")
        for (x1, y1, x2, y2, cls), depth in zip(cones, depths):
            # Apply depth filter
            if self.depth_min <= depth <= self.depth_max:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                filtered_cones.append((center_x, center_y, cls))
                filtered_depths.append(depth)
                print(f"Kept cone: Class = {cls}, Depth = {depth:.2f}m, Center = ({center_x}, {center_y})")
            else:
                print(f"Filtered out cone: Depth = {depth:.2f}m out of range [{self.depth_min}, {self.depth_max}]")
        
        print(f"Total cones after filtering: {len(filtered_cones)}")
        return filtered_cones, filtered_depths
    
    def _to_world_coordinates(self, filtered_cones, filtered_depths):
        """
        Convert image-space cone detections to world coordinates.
        
        Args:
            filtered_cones: List of (center_x, center_y, cls) tuples
            filtered_depths: List of depth values for each cone
            
        Returns:
            List of WorldCone objects
        """
        world_cones = []
        
        for (center_x, center_y, cls), depth in zip(filtered_cones, filtered_depths):
            # Calculate angle from center of image
            angle = ((center_x - self.image_width / 2) / (self.image_width / 2)) * (self.fov_horizontal / 2)
            
            # Convert to world coordinates
            world_x = depth * np.tan(np.radians(angle))
            
            # Add to world cones list
            world_cones.append(WorldCone(world_x, depth, cls))
        
        return world_cones
    
    def _pair_cones(self, world_cones):
        """
        Pair yellow and blue cones to create track boundaries.
        
        Args:
            world_cones: List of WorldCone objects
        """
        # Separate yellow and blue cones
        yellow_cones = [cone for cone in world_cones if cone.cls == 0]
        blue_cones = [cone for cone in world_cones if cone.cls == 1]
        
        # Sort by depth
        yellow_cones.sort(key=lambda c: c.depth)
        blue_cones.sort(key=lambda c: c.depth)
        
        # Store yellow and blue cones for boundary visualization
        self.yellow_boundary = [(cone.x, cone.depth) for cone in yellow_cones]
        self.blue_boundary = [(cone.x, cone.depth) for cone in blue_cones]
        
        print(f"Yellow cones: {len(yellow_cones)}, Blue cones: {len(blue_cones)}")
        
        # Clear previous pairs
        self.cone_pairs = []
        
        # If no cones detected, return
        if not yellow_cones and not blue_cones:
            return
        
        # Calculate reasonable track width from data if possible
        track_widths = []
        for y in yellow_cones:
            for b in blue_cones:
                # Match cones at similar depths
                if abs(y.depth - b.depth) < 2.0:
                    width = abs(y.x - b.x)
                    if 2.0 < width < 6.0:  # Reasonable track width range
                        track_widths.append(width)
        
        # Update track width estimate
        if track_widths:
            self.current_track_width = np.median(track_widths)
            print(f"Calculated track width: {self.current_track_width:.2f}m")
        
        # Create initial pairs using direct matching
        matched_yellows = set()
        matched_blues = set()
        
        # First pass: direct matching of cones at similar depths
        for i, y in enumerate(yellow_cones):
            best_blue = None
            best_dist = float('inf')
            
            for j, b in enumerate(blue_cones):
                depth_diff = abs(y.depth - b.depth)
                if depth_diff < 2.0 and depth_diff < best_dist:
                    best_dist = depth_diff
                    best_blue = j
            
            if best_blue is not None:
                # Create a pair
                b = blue_cones[best_blue]
                pair = ConePair.from_cones(y, b, self.current_track_width)
                self.cone_pairs.append(pair)
                
                # Mark as matched
                matched_yellows.add(i)
                matched_blues.add(best_blue)
        
        # Second pass: handle unpaired cones
        # Add remaining yellow cones
        for i, y in enumerate(yellow_cones):
            if i not in matched_yellows:
                pair = ConePair.from_cones(y, None, self.current_track_width)
                self.cone_pairs.append(pair)
        
        # Add remaining blue cones
        for i, b in enumerate(blue_cones):
            if i not in matched_blues:
                pair = ConePair.from_cones(None, b, self.current_track_width)
                self.cone_pairs.append(pair)
        
        # Sort by depth
        self.cone_pairs.sort(key=lambda p: p.midpoint[1])
        
        # Limit to max_lookahead_pairs
        if len(self.cone_pairs) > self.max_lookahead_pairs:
            self.cone_pairs = self.cone_pairs[:self.max_lookahead_pairs]
            print(f"Limited to {self.max_lookahead_pairs} cone pairs")
        
        print(f"Created {len(self.cone_pairs)} cone pairs")
    
    def _connect_boundary_cones(self):
        """Connect cones to form continuous track boundaries with improved U-turn handling"""
        if not self.cone_pairs:
            return
                
        # Ensure we have the blue and yellow boundaries
        if not hasattr(self, 'blue_boundary') or not hasattr(self, 'yellow_boundary'):
            self.blue_boundary = []
            self.yellow_boundary = []
                
        # Reset boundaries for fresh calculation
        self.blue_boundary = []
        self.yellow_boundary = []
        
        # Extract from cone pairs
        for pair in self.cone_pairs:
            if pair.blue is not None:  # Explicit None check
                self.blue_boundary.append((pair.blue.x, pair.blue.depth))
            if pair.yellow is not None:  # Explicit None check 
                self.yellow_boundary.append((pair.yellow.x, pair.yellow.depth))
        
        # Sort by depth
        self.blue_boundary.sort(key=lambda p: p[1])
        self.yellow_boundary.sort(key=lambda p: p[1])
        
        # Count how many cones of each color we have
        yellow_count = len(self.yellow_boundary)
        blue_count = len(self.blue_boundary)
        
        # Check if we potentially have a U-turn (significant imbalance of cone colors)
        potential_uturn = False
        uturn_yellow_side = False
        uturn_blue_side = False
        
        if yellow_count > 1 and yellow_count > 2 * blue_count:
            potential_uturn = True
            uturn_yellow_side = True
            print(f"U-turn detection: Likely yellow-side U-turn ({yellow_count} yellow vs {blue_count} blue)")
        elif blue_count > 1 and blue_count > 2 * yellow_count:
            potential_uturn = True
            uturn_blue_side = True
            print(f"U-turn detection: Likely blue-side U-turn ({blue_count} blue vs {yellow_count} yellow)")
        
        # If we have a potential U-turn, examine the cone pattern
        if potential_uturn:
            if uturn_yellow_side and yellow_count >= 3:
                # Analyze yellow cone pattern for U-turn shape
                yellow_xs = [x for x, _ in self.yellow_boundary]
                yellow_depth_sorted = sorted(self.yellow_boundary, key=lambda p: p[1])  # Sort by increasing depth
                
                # Check if yellows form an arc (x-coordinates change consistently)
                if len(yellow_xs) >= 3:
                    # Check if the x-values are decreasing (moving right to left as depth increases)
                    x_diffs = [yellow_xs[i] - yellow_xs[i-1] for i in range(1, len(yellow_xs))]
                    consistent_direction = all(diff < 0 for diff in x_diffs) or all(diff > 0 for diff in x_diffs)
                    
                    if consistent_direction:
                        print("U-turn confirmed: Yellow cones form an arc with consistent direction")
                        # Store that we're in a U-turn for reference
                        self.in_uturn = True
                        self.uturn_side = "yellow"
                        
                        # Calculate typical turn radius for the path planning
                        if len(self.yellow_boundary) >= 2:
                            # Estimate radius from two farthest cones
                            first_cone = self.yellow_boundary[0]
                            last_cone = self.yellow_boundary[-1]
                            dx = last_cone[0] - first_cone[0]
                            dy = last_cone[1] - first_cone[1]
                            self.turn_radius = np.sqrt(dx**2 + dy**2) / 2
                            print(f"Estimated turn radius: {self.turn_radius:.2f}m")
                        else:
                            # Fallback to a reasonable radius
                            self.turn_radius = self.default_track_width * 1.5
            
            elif uturn_blue_side and blue_count >= 3:
                # Analyze blue cone pattern for U-turn shape
                blue_xs = [x for x, _ in self.blue_boundary]
                blue_depth_sorted = sorted(self.blue_boundary, key=lambda p: p[1])  # Sort by increasing depth
                
                # Check if blues form an arc (x-coordinates change consistently)
                if len(blue_xs) >= 3:
                    # Check if the x-values are decreasing (moving right to left as depth increases)
                    x_diffs = [blue_xs[i] - blue_xs[i-1] for i in range(1, len(blue_xs))]
                    consistent_direction = all(diff < 0 for diff in x_diffs) or all(diff > 0 for diff in x_diffs)
                    
                    if consistent_direction:
                        print("U-turn confirmed: Blue cones form an arc with consistent direction")
                        # Store that we're in a U-turn for reference
                        self.in_uturn = True
                        self.uturn_side = "blue"
                        
                        # Calculate typical turn radius for the path planning
                        if len(self.blue_boundary) >= 2:
                            # Estimate radius from two farthest cones
                            first_cone = self.blue_boundary[0]
                            last_cone = self.blue_boundary[-1]
                            dx = last_cone[0] - first_cone[0]
                            dy = last_cone[1] - first_cone[1]
                            self.turn_radius = np.sqrt(dx**2 + dy**2) / 2
                            print(f"Estimated turn radius: {self.turn_radius:.2f}m")
                        else:
                            # Fallback to a reasonable radius
                            self.turn_radius = self.default_track_width * 1.5
            else:
                # Not enough cones to confirm U-turn
                self.in_uturn = False
                self.uturn_side = "none"
        else:
            # Not in a U-turn
            self.in_uturn = False
            self.uturn_side = "none"
    
    def _find_target_point(self):
        """Find the target point for pure pursuit algorithm with improved cone hugging in U-turns."""
        if not self.cone_pairs:
            self.target_point = None
            return
        
        # Count how many pairs have only one cone
        yellow_only_count = sum(1 for pair in self.cone_pairs if pair.yellow is not None and pair.blue is None)
        blue_only_count = sum(1 for pair in self.cone_pairs if pair.blue is not None and pair.yellow is None)
        
        # Determine if we're in a potential U-turn (mostly seeing one color)
        potential_uturn = False
        follow_yellow = False
        follow_blue = False
        
        if yellow_only_count > 1 and yellow_only_count > blue_only_count:
            # Multiple yellow cones but few or no blue cones - might be a yellow-side U-turn
            potential_uturn = True
            follow_yellow = True
            print(f"Potential U-turn detected for targeting! Following yellow cones ({yellow_only_count} yellow, {blue_only_count} blue)")
        elif blue_only_count > 1 and blue_only_count > yellow_only_count:
            # Multiple blue cones but few or no blue cones - might be a blue-side U-turn
            potential_uturn = True
            follow_blue = True
            print(f"Potential U-turn detected for targeting! Following blue cones ({blue_only_count} blue, {yellow_only_count} yellow)")
        
        # If we're in a U-turn, find a target point that hugs the visible cones
        if potential_uturn:
            # Set shorter lookahead for U-turns
            u_turn_lookahead = min(self.lookahead_distance, 2.0)  # Even shorter lookahead to hug curve
            
            # Find the target point directly from the cone boundary
            if follow_yellow and self.yellow_boundary:
                # Find the point on the yellow boundary closest to our lookahead distance
                best_yellow_point = None
                best_dist_diff = float('inf')
                
                for x, depth in self.yellow_boundary:
                    dist_diff = abs(depth - u_turn_lookahead)
                    if dist_diff < best_dist_diff:
                        best_dist_diff = dist_diff
                        best_yellow_point = (x, depth)
                
                if best_yellow_point:
                    # Place target point right next to the yellow cone (slightly to the right)
                    target_x = best_yellow_point[0] - self.safety_margin
                    target_depth = best_yellow_point[1]
                    self.target_point = (target_x, target_depth)
                    print(f"U-turn: Hugging yellow cone at ({best_yellow_point[0]:.2f}, {best_yellow_point[1]:.2f})")
                    return
                    
            elif follow_blue and self.blue_boundary:
                # Find the point on the blue boundary closest to our lookahead distance
                best_blue_point = None
                best_dist_diff = float('inf')
                
                for x, depth in self.blue_boundary:
                    dist_diff = abs(depth - u_turn_lookahead)
                    if dist_diff < best_dist_diff:
                        best_dist_diff = dist_diff
                        best_blue_point = (x, depth)
                
                if best_blue_point:
                    # Place target point right next to the blue cone (slightly to the left)
                    target_x = best_blue_point[0] + self.safety_margin
                    target_depth = best_blue_point[1]
                    self.target_point = (target_x, target_depth)
                    print(f"U-turn: Hugging blue cone at ({best_blue_point[0]:.2f}, {best_blue_point[1]:.2f})")
                    return
        
        # If we're not in a U-turn or couldn't find a good cone to hug, use standard approach
        # Get lookahead distance (adjusted based on speed if available)
        lookahead = self.lookahead_distance
        
        # Find the pair closest to lookahead distance
        best_pair = None
        best_dist_diff = float('inf')
        
        for pair in self.cone_pairs:
            dist_diff = abs(pair.midpoint[1] - lookahead)
            if dist_diff < best_dist_diff:
                best_dist_diff = dist_diff
                best_pair = pair
        
        if best_pair:
            # Handle based on whether we're in a potential U-turn
            if potential_uturn:
                if follow_yellow and best_pair.yellow is not None:
                    # Following yellow cones in a U-turn - stay closer to yellow cones
                    # Just use safety margin to follow cone more tightly
                    adjusted_x = best_pair.yellow.x - self.safety_margin
                    self.target_point = (adjusted_x, best_pair.yellow.depth)
                    print(f"U-turn target: Hugging yellow cone at ({best_pair.yellow.x:.2f}, {best_pair.yellow.depth:.2f})")
                
                elif follow_blue and best_pair.blue is not None:
                    # Following blue cones in a U-turn - stay closer to blue cones
                    # Just use safety margin to follow cone more tightly
                    adjusted_x = best_pair.blue.x + self.safety_margin
                    self.target_point = (adjusted_x, best_pair.blue.depth)
                    print(f"U-turn target: Hugging blue cone at ({best_pair.blue.x:.2f}, {best_pair.blue.depth:.2f})")
                
                # If the best pair has both cones even in U-turn mode, use standard approach
                elif best_pair.yellow is not None and best_pair.blue is not None:
                    # Standard case with both cones - use midpoint with safety margins
                    yellow_x, blue_x = best_pair.yellow.x, best_pair.blue.x
                    
                    # Ensure we're not too close to either cone
                    if yellow_x > blue_x:  # Normal case: yellow on left, blue on right
                        adjusted_yellow_x = yellow_x - self.safety_margin
                        adjusted_blue_x = blue_x + self.safety_margin
                    else:  # Reversed case: blue on left, yellow on right
                        adjusted_yellow_x = yellow_x + self.safety_margin
                        adjusted_blue_x = blue_x - self.safety_margin
                    
                    # Check if there's still space between cones after applying margins
                    if (yellow_x > blue_x and adjusted_yellow_x > adjusted_blue_x) or \
                    (yellow_x < blue_x and adjusted_yellow_x < adjusted_blue_x):
                        # Calculate new midpoint
                        adjusted_midpoint_x = (adjusted_yellow_x + adjusted_blue_x) / 2
                        
                        # Update target point with adjusted midpoint
                        self.target_point = (adjusted_midpoint_x, best_pair.midpoint[1])
                    else:
                        # Not enough space - use original midpoint
                        self.target_point = best_pair.midpoint
                        print(f"Warning: Not enough space for safety margin at depth {best_pair.midpoint[1]:.2f}m")
                
                # Fallback if no matching cones in the best pair
                else:
                    self.target_point = best_pair.midpoint
                    print(f"Warning: U-turn detected but best pair doesn't have the needed cone type")
            else:
                # Standard approach (not a U-turn)
                # Apply safety margin to keep away from cones
                if best_pair.yellow is not None and best_pair.blue is not None:
                    # We have both cones - adjust the midpoint to maintain safety margin
                    yellow_x, blue_x = best_pair.yellow.x, best_pair.blue.x
                    
                    # Ensure we're not too close to either cone
                    if yellow_x > blue_x:  # Normal case: yellow on left, blue on right
                        adjusted_yellow_x = yellow_x - self.safety_margin
                        adjusted_blue_x = blue_x + self.safety_margin
                    else:  # Reversed case: blue on left, yellow on right
                        adjusted_yellow_x = yellow_x + self.safety_margin
                        adjusted_blue_x = blue_x - self.safety_margin
                    
                    # Check if there's still space between cones after applying margins
                    if (yellow_x > blue_x and adjusted_yellow_x > adjusted_blue_x) or \
                    (yellow_x < blue_x and adjusted_yellow_x < adjusted_blue_x):
                        # Calculate new midpoint
                        adjusted_midpoint_x = (adjusted_yellow_x + adjusted_blue_x) / 2
                        
                        # Update target point with adjusted midpoint
                        self.target_point = (adjusted_midpoint_x, best_pair.midpoint[1])
                    else:
                        # Not enough space - use original midpoint
                        self.target_point = best_pair.midpoint
                        print(f"Warning: Not enough space for safety margin at depth {best_pair.midpoint[1]:.2f}m")
                else:
                    # Just one cone, use the existing midpoint but with safety margin
                    if best_pair.yellow is not None:
                        # Yellow cone (left) - move right by safety margin
                        adjusted_x = best_pair.midpoint[0] - self.safety_margin 
                        self.target_point = (adjusted_x, best_pair.midpoint[1])
                    elif best_pair.blue is not None:
                        # Blue cone (right) - move left by safety margin
                        adjusted_x = best_pair.midpoint[0] + self.safety_margin
                        self.target_point = (adjusted_x, best_pair.midpoint[1])
                    else:
                        # Fallback to original midpoint
                        self.target_point = best_pair.midpoint
            
            print(f"Target point selected at ({self.target_point[0]:.2f}, {self.target_point[1]:.2f})")
        else:
            # Fallback: use the farthest pair
            self.target_point = self.cone_pairs[-1].midpoint if self.cone_pairs else None
    
    def _calculate_steering(self):
        """Calculate steering angle using pure pursuit algorithm."""
        if not self.target_point:
            self.steering_angle = 0.0
            return
        
        # Get target point
        target_x, target_depth = self.target_point
        
        # Calculate angle to target point (from car's perspective)
        alpha = np.arctan2(target_x, target_depth)
        
        # Pure pursuit steering angle calculation
        steering_angle = np.arctan2(2 * self.wheelbase * np.sin(alpha), target_depth)
        
        # Convert to normalized steering angle [-1, 1]
        max_steering_rad = np.radians(30.0)  # Maximum steering angle
        self.steering_angle = np.clip(steering_angle / max_steering_rad, -1.0, 1.0)
        
        print(f"Pure pursuit steering angle: {self.steering_angle:.2f}")
    
    def _smooth_steering(self):
        """Apply smoothing to steering angle for stability."""
        if self.prev_steering_angle is not None:
            # Exponential smoothing
            alpha = 0.7  # Smoothing factor (higher = less smoothing)
            old_steering = self.steering_angle
            self.steering_angle = alpha * self.steering_angle + (1 - alpha) * self.prev_steering_angle
            print(f"Smoothed steering: {old_steering:.2f} -> {self.steering_angle:.2f}")
        
        # Store current steering angle for next iteration
        self.prev_steering_angle = self.steering_angle
    
    def _create_compatible_path(self):
        """Create a path compatible with the original code's format that hugs visible cones in U-turns."""
        if not self.cone_pairs:
            self.path = None
            return
                    
        # Create path that follows the midpoint of track with safety margins
        waypoints = []
        
        # Count how many pairs have only one cone
        yellow_only_count = sum(1 for pair in self.cone_pairs if pair.yellow is not None and pair.blue is None)
        blue_only_count = sum(1 for pair in self.cone_pairs if pair.blue is not None and pair.yellow is None)
        
        # Determine if we're in a potential U-turn (mostly seeing one color)
        potential_uturn = False
        follow_yellow = False
        follow_blue = False
        
        if yellow_only_count > 1 and yellow_only_count > blue_only_count:
            # Multiple yellow cones but few or no blue cones - might be a yellow-side U-turn
            potential_uturn = True
            follow_yellow = True
            print(f"Potential U-turn detected! Following yellow cones ({yellow_only_count} yellow, {blue_only_count} blue)")
        elif blue_only_count > 1 and blue_only_count > yellow_only_count:
            # Multiple blue cones but few or no yellow cones - might be a blue-side U-turn
            potential_uturn = True
            follow_blue = True
            print(f"Potential U-turn detected! Following blue cones ({blue_only_count} blue, {yellow_only_count} yellow)")
        
        # If following one color in a U-turn, create a path that hugs the curve of the visible cones
        if potential_uturn:
            if follow_yellow and self.yellow_boundary:
                # Create a path that follows the yellow boundary closely
                for x, depth in self.yellow_boundary:
                    # Stay a fixed distance to the right of yellow cones
                    # Use just the safety margin (closer than before)
                    adjusted_x = x - self.safety_margin
                    waypoints.append((adjusted_x, depth))
                    print(f"U-turn: Hugging yellow cone at ({x:.2f}, {depth:.2f}) with offset {self.safety_margin:.2f}m")
                
                # Make sure waypoints are sorted by depth
                waypoints.sort(key=lambda p: p[1])
                
                # Calculate and smooth the path curvature
                self._smooth_boundary_path(waypoints, "yellow")
                
            elif follow_blue and self.blue_boundary:
                # Create a path that follows the blue boundary closely
                for x, depth in self.blue_boundary:
                    # Stay a fixed distance to the left of blue cones
                    # Use just the safety margin (closer than before)
                    adjusted_x = x + self.safety_margin
                    waypoints.append((adjusted_x, depth))
                    print(f"U-turn: Hugging blue cone at ({x:.2f}, {depth:.2f}) with offset {self.safety_margin:.2f}m")
                
                # Make sure waypoints are sorted by depth
                waypoints.sort(key=lambda p: p[1])
                
                # Calculate and smooth the path curvature
                self._smooth_boundary_path(waypoints, "blue")
            
            else:
                # Process the cone pairs in the standard way as fallback
                self._process_cone_pairs_for_path(waypoints, potential_uturn, follow_yellow, follow_blue)
        else:
            # Not in a U-turn, process cone pairs normally
            self._process_cone_pairs_for_path(waypoints, potential_uturn, follow_yellow, follow_blue)
        
        # Add point at car position for continuity
        waypoints.insert(0, (0.0, 0.5))
        
        # Ensure waypoints are sorted by depth
        waypoints.sort(key=lambda p: p[1])
        
        # Store as path
        self.path = waypoints
        
        print(f"Created compatible path with {len(self.path)} points")

    def _smooth_boundary_path(self, waypoints, side):
        """
        Smooth the path that follows a boundary to create a natural curve.
        
        Args:
            waypoints: List of waypoints to smooth
            side: Which side the path is following ("yellow" or "blue")
        """
        if len(waypoints) <= 2:
            return  # Not enough points to smooth
        
        # Copy original waypoints for reference
        original_waypoints = waypoints.copy()
        
        # Sort by depth to ensure order
        original_waypoints.sort(key=lambda p: p[1])
        
        # Clear the current waypoints
        waypoints.clear()
        
        # Add first point
        waypoints.append(original_waypoints[0])
        
        # If we have enough points, calculate a smooth curve
        if len(original_waypoints) >= 3:
            # Add intermediate points with smoothing
            for i in range(1, len(original_waypoints)-1):
                prev_pt = original_waypoints[i-1]
                curr_pt = original_waypoints[i]
                next_pt = original_waypoints[i+1]
                
                # Calculate the average position for smoothing
                smooth_x = (prev_pt[0] + curr_pt[0] + next_pt[0]) / 3.0
                
                # Only smooth x-coordinate (lateral position), keep original depth
                waypoints.append((smooth_x, curr_pt[1]))
                
                # Add extra interpolated points for smoother curves
                # This helps the car follow the curve more naturally
                mid_depth1 = (prev_pt[1] + curr_pt[1]) / 2
                mid_depth2 = (curr_pt[1] + next_pt[1]) / 2
                
                # Calculate interpolated x values
                k1 = (curr_pt[1] - prev_pt[1]) / (curr_pt[1] - prev_pt[1] + 1e-6)  # Avoid division by zero
                k2 = (next_pt[1] - curr_pt[1]) / (next_pt[1] - curr_pt[1] + 1e-6)  # Avoid division by zero
                
                mid_x1 = prev_pt[0] + k1 * (curr_pt[0] - prev_pt[0])
                mid_x2 = curr_pt[0] + k2 * (next_pt[0] - curr_pt[0])
                
                # Add interpolated points
                if i == 1:  # Only add before point for first segment
                    waypoints.append((mid_x1, mid_depth1))
                
                waypoints.append((mid_x2, mid_depth2))
        
        # Add last point
        waypoints.append(original_waypoints[-1])
        
        # Sort again by depth to ensure order after smoothing
        waypoints.sort(key=lambda p: p[1])

    def _process_cone_pairs_for_path(self, waypoints, potential_uturn, follow_yellow, follow_blue):
        """
        Process cone pairs to generate path waypoints.
        
        Args:
            waypoints: List to store the generated waypoints
            potential_uturn: Whether a potential U-turn is detected
            follow_yellow: Whether to follow yellow cones in a U-turn
            follow_blue: Whether to follow blue cones in a U-turn
        """
        # Process each cone pair
        for pair in self.cone_pairs:
            # Both cones visible - use standard approach
            if pair.yellow is not None and pair.blue is not None:
                # Standard case: adjust midpoint with safety margins
                yellow_x, blue_x = pair.yellow.x, pair.blue.x
                
                # Ensure we're not too close to either cone
                if yellow_x > blue_x:  # Normal case: yellow on left, blue on right
                    adjusted_yellow_x = yellow_x - self.safety_margin
                    adjusted_blue_x = blue_x + self.safety_margin
                else:  # Reversed case: blue on left, yellow on right
                    adjusted_yellow_x = yellow_x + self.safety_margin
                    adjusted_blue_x = blue_x - self.safety_margin
                
                # Check if there's still space between cones after applying margins
                if (yellow_x > blue_x and adjusted_yellow_x > adjusted_blue_x) or \
                (yellow_x < blue_x and adjusted_yellow_x < adjusted_blue_x):
                    # Calculate safe midpoint
                    adjusted_midpoint_x = (adjusted_yellow_x + adjusted_blue_x) / 2
                    waypoints.append((adjusted_midpoint_x, pair.midpoint[1]))
                else:
                    # Not enough space - use original midpoint
                    waypoints.append((pair.midpoint[0], pair.midpoint[1]))
                    print(f"Warning: Not enough space for safety margin at depth {pair.midpoint[1]:.2f}m")
            else:
                # Only one cone visible - handle based on U-turn detection
                if potential_uturn:
                    if follow_yellow and pair.yellow is not None:
                        # Following yellow cones in a U-turn - stay closer to yellow cones
                        # Just use safety margin to hug the cone closely
                        adjusted_x = pair.yellow.x - self.safety_margin
                        waypoints.append((adjusted_x, pair.yellow.depth))
                        print(f"U-turn: Hugging yellow cone at ({pair.yellow.x:.2f}, {pair.yellow.depth:.2f})")
                    
                    elif follow_blue and pair.blue is not None:
                        # Following blue cones in a U-turn - stay closer to blue cones
                        # Just use safety margin to hug the cone closely
                        adjusted_x = pair.blue.x + self.safety_margin
                        waypoints.append((adjusted_x, pair.blue.depth))
                        print(f"U-turn: Hugging blue cone at ({pair.blue.x:.2f}, {pair.blue.depth:.2f})")
                    
                    # If we're following one color but have the other, include it with standard safety margin
                    elif follow_yellow and pair.blue is not None:
                        adjusted_x = pair.blue.x + self.safety_margin
                        waypoints.append((adjusted_x, pair.blue.depth))
                    elif follow_blue and pair.yellow is not None:
                        adjusted_x = pair.yellow.x - self.safety_margin
                        waypoints.append((adjusted_x, pair.yellow.depth))
                else:
                    # Standard single-cone handling (no U-turn)
                    if pair.yellow is not None:
                        # Yellow cone (left) - move right by safety margin
                        adjusted_x = pair.midpoint[0] - self.safety_margin 
                        waypoints.append((adjusted_x, pair.midpoint[1]))
                    elif pair.blue is not None:
                        # Blue cone (right) - move left by safety margin
                        adjusted_x = pair.midpoint[0] + self.safety_margin
                        waypoints.append((adjusted_x, pair.midpoint[1]))
    
    def calculate_steering(self, lookahead_distance=4.0, steering_gain=1.0, max_steering_angle=30.0):
        """
        Calculate steering based on the planned path with improved U-turn handling.
        
        Args:
            lookahead_distance (float): Distance ahead to look for target point (in meters)
            steering_gain (float): Multiplier for steering sensitivity
            max_steering_angle (float): Maximum steering angle in degrees
            
        Returns:
            float: Normalized steering angle in range [-1, 1]
        """
        try:
            # Adjust lookahead distance for U-turns
            if hasattr(self, 'in_uturn') and self.in_uturn:
                # Use a shorter lookahead in U-turns for tighter turning
                u_turn_lookahead = min(lookahead_distance, 2.5)  # Shorter lookahead in U-turns
                print(f"U-turn: Reducing lookahead from {lookahead_distance}m to {u_turn_lookahead}m")
                lookahead_distance = u_turn_lookahead
            
            # Update lookahead distance if different
            if lookahead_distance != self.lookahead_distance:
                self.lookahead_distance = lookahead_distance
                self._find_target_point()
                self._calculate_steering()
                self._smooth_steering()
            
            # Adjust steering for tighter U-turns
            if hasattr(self, 'in_uturn') and self.in_uturn:
                # Increase steering amplitude in U-turns
                u_turn_gain = 1.3  # Higher gain for more aggressive steering in U-turns
                old_steering = self.steering_angle
                self.steering_angle = np.clip(self.steering_angle * u_turn_gain, -1.0, 1.0)
                print(f"U-turn: Increasing steering from {old_steering:.2f} to {self.steering_angle:.2f}")
            
            # Always visualize when called
            if self.visualize and hasattr(self.zed_camera, 'rgb_image') and self.zed_camera.rgb_image is not None:
                try:
                    self.debug_image = self.draw_path(self.zed_camera.rgb_image.copy())
                    cv2.imshow("Pure Pursuit Path", self.debug_image)
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"Warning: Visualization failed: {str(e)}")
            
            return self.steering_angle
        except Exception as e:
            print(f"Error calculating steering: {str(e)}")
            return 0.0  # Safe default
    
    def get_path(self):
        """Get the current path (for compatibility with original code)."""
        return self.path
    
    def draw_path(self, image):
        """
        Draw the planned path with U-turn detection visualization.
        
        Args:
            image: Input image for visualization
            
        Returns:
            Image with visualization overlays
        """
        if image is None:
            # Create a blank image if none provided
            image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # Create a copy of the image
        viz_image = image.copy()
        
        # Detect potential U-turn for visualization
        yellow_only_count = sum(1 for pair in self.cone_pairs if pair.yellow is not None and pair.blue is None)
        blue_only_count = sum(1 for pair in self.cone_pairs if pair.blue is not None and pair.yellow is None)
        
        # Determine if we're in a potential U-turn (mostly seeing one color)
        in_uturn = False
        uturn_side = "none"
        if yellow_only_count > 1 and yellow_only_count > blue_only_count:
            in_uturn = True
            uturn_side = "yellow"
        elif blue_only_count > 1 and blue_only_count > yellow_only_count:
            in_uturn = True
            uturn_side = "blue"
        
        # Draw car indicator at bottom center of image
        car_x = self.image_width // 2
        car_y = self.image_height - 20
        
        # Draw car triangle
        car_points = np.array([
            [car_x, car_y - 10],
            [car_x - 10, car_y + 10],
            [car_x + 10, car_y + 10]
        ], dtype=np.int32)
        cv2.fillPoly(viz_image, [car_points], (0, 255, 255))
        
        # Helper function to convert world coords to image coords
        def world_to_image(x, depth):
            # Convert lateral position to image x-coordinate
            angle = np.arctan2(x, depth)
            px = int(car_x + (angle / np.radians(self.fov_horizontal / 2)) * (self.image_width / 2))
            
            # Convert depth to image y-coordinate (non-linear mapping for perspective)
            if depth <= 1.0:
                py = int(car_y - (depth / 1.0) * 60)
            else:
                py = int(car_y - 60 - ((depth - 1.0) ** 0.85) * 40)
            
            return max(0, min(px, self.image_width - 1)), max(0, min(py, self.image_height - 1))
        
        # Draw cone pairs
        for i, pair in enumerate(self.cone_pairs):
            # Draw pair midpoint
            mid_x, mid_depth = pair.midpoint
            mid_px, mid_py = world_to_image(mid_x, mid_depth)
            
            # Color gradient based on distance
            ratio = i / max(1, len(self.cone_pairs) - 1)
            if ratio < 0.5:
                color = (0, int(255 * ratio * 2), 255)  # Blue -> Cyan
            else:
                color = (0, 255, int(255 * (1 - (ratio - 0.5) * 2)))  # Cyan -> Green
                    
            cv2.circle(viz_image, (mid_px, mid_py), 5, color, -1)
            
            # Draw line connecting cone pair
            if pair.yellow is not None and pair.blue is not None:
                y_px, y_py = world_to_image(pair.yellow.x, pair.yellow.depth)
                b_px, b_py = world_to_image(pair.blue.x, pair.blue.depth)
                
                # Draw yellow cone
                cv2.circle(viz_image, (y_px, y_py), 4, (0, 255, 255), -1)
                
                # Draw blue cone
                cv2.circle(viz_image, (b_px, b_py), 4, (255, 0, 0), -1)
                
                # Draw track boundary between the gate
                cv2.line(viz_image, (y_px, y_py), (b_px, b_py), (255, 255, 255), 1)
            else:
                # Only one cone in the pair - draw larger with highlighted border if in U-turn
                if pair.yellow is not None:
                    y_px, y_py = world_to_image(pair.yellow.x, pair.yellow.depth)
                    # Special highlight for cones being followed in U-turn
                    if in_uturn and uturn_side == "yellow":
                        # Draw larger highlighted cone
                        cv2.circle(viz_image, (y_px, y_py), 7, (0, 0, 0), -1)  # Black background
                        cv2.circle(viz_image, (y_px, y_py), 6, (0, 255, 255), -1)  # Yellow cone
                        cv2.circle(viz_image, (y_px, y_py), 3, (255, 255, 255), -1)  # White center
                    else:
                        # Regular yellow cone
                        cv2.circle(viz_image, (y_px, y_py), 4, (0, 255, 255), -1)
                
                if pair.blue is not None:
                    b_px, b_py = world_to_image(pair.blue.x, pair.blue.depth)
                    # Special highlight for cones being followed in U-turn
                    if in_uturn and uturn_side == "blue":
                        # Draw larger highlighted cone
                        cv2.circle(viz_image, (b_px, b_py), 7, (0, 0, 0), -1)  # Black background
                        cv2.circle(viz_image, (b_px, b_py), 6, (255, 0, 0), -1)  # Blue cone
                        cv2.circle(viz_image, (b_px, b_py), 3, (255, 255, 255), -1)  # White center
                    else:
                        # Regular blue cone
                        cv2.circle(viz_image, (b_px, b_py), 4, (255, 0, 0), -1)
        
        # Draw connected boundary cones (yellow side)
        yellow_pixels = []
        for x, depth in self.yellow_boundary:
            px, py = world_to_image(x, depth)
            yellow_pixels.append((px, py))
            # Yellow cones are drawn by the cone pair loop above
        
        # Draw connected boundary cones (blue side)
        blue_pixels = []
        for x, depth in self.blue_boundary:
            px, py = world_to_image(x, depth)
            blue_pixels.append((px, py))
            # Blue cones are drawn by the cone pair loop above
        
        # Draw connected yellow boundary line
        if len(yellow_pixels) >= 2:
            for i in range(len(yellow_pixels) - 1):
                # Highlight yellow boundary in U-turn case
                if in_uturn and uturn_side == "yellow":
                    cv2.line(viz_image, yellow_pixels[i], yellow_pixels[i + 1], (0, 255, 255), 3)  # Thicker line
                else:
                    cv2.line(viz_image, yellow_pixels[i], yellow_pixels[i + 1], (0, 255, 255), 2)
        
        # Draw connected blue boundary line
        if len(blue_pixels) >= 2:
            for i in range(len(blue_pixels) - 1):
                # Highlight blue boundary in U-turn case
                if in_uturn and uturn_side == "blue":
                    cv2.line(viz_image, blue_pixels[i], blue_pixels[i + 1], (255, 0, 0), 3)  # Thicker line
                else:
                    cv2.line(viz_image, blue_pixels[i], blue_pixels[i + 1], (255, 0, 0), 2)
        
        # Add safety margin visualization
        if hasattr(self, 'safety_margin') and self.safety_margin > 0:
            # Draw yellow boundary with safety margin
            yellow_safe = []
            for x, depth in self.yellow_boundary:
                # Move right by safety margin
                safe_x = x - self.safety_margin
                px, py = world_to_image(safe_x, depth)
                yellow_safe.append((px, py))
            
            # Draw blue boundary with safety margin
            blue_safe = []
            for x, depth in self.blue_boundary:
                # Move left by safety margin
                safe_x = x + self.safety_margin
                px, py = world_to_image(safe_x, depth)
                blue_safe.append((px, py))
            
            # Draw safety margin boundaries with dotted lines (compatible with older OpenCV versions)
            if len(yellow_safe) >= 2:
                for i in range(len(yellow_safe) - 1):
                    # Draw dotted line by creating small segments
                    pt1 = yellow_safe[i]
                    pt2 = yellow_safe[i + 1]
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
                    
                    # Create dots every 5 pixels
                    for j in range(0, dist, 10):
                        x = int(pt1[0] + j * dx / dist)
                        y = int(pt1[1] + j * dy / dist)
                        cv2.circle(viz_image, (x, y), 1, (0, 160, 160), -1)
            
            if len(blue_safe) >= 2:
                for i in range(len(blue_safe) - 1):
                    # Draw dotted line by creating small segments
                    pt1 = blue_safe[i]
                    pt2 = blue_safe[i + 1]
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
                    
                    # Create dots every 5 pixels
                    for j in range(0, dist, 10):
                        x = int(pt1[0] + j * dx / dist)
                        y = int(pt1[1] + j * dy / dist)
                        cv2.circle(viz_image, (x, y), 1, (160, 0, 0), -1)
            
            # If in U-turn, draw additional turning radius visualization
            if in_uturn:
                # Calculate an approximate turning circle
                turn_radius = self.default_track_width * 1.5  # Approximate turning radius
                if uturn_side == "yellow" and self.yellow_boundary:
                    # For yellow-side U-turn, get the innermost yellow cone
                    innermost_yellow = min(self.yellow_boundary, key=lambda p: p[1])
                    center_x = innermost_yellow[0] - turn_radius
                    center_depth = innermost_yellow[1]
                    center_px, center_py = world_to_image(center_x, center_depth)
                    
                    # Draw turning circle
                    for angle in range(0, 181, 10):  # 0-180 degrees in 10 degree steps
                        rad = np.radians(angle)
                        x = center_x + turn_radius * np.cos(rad)
                        y = center_depth + turn_radius * np.sin(rad)
                        px, py = world_to_image(x, y)
                        cv2.circle(viz_image, (px, py), 1, (0, 180, 180), -1)
                
                elif uturn_side == "blue" and self.blue_boundary:
                    # For blue-side U-turn, get the innermost blue cone
                    innermost_blue = min(self.blue_boundary, key=lambda p: p[1])
                    center_x = innermost_blue[0] + turn_radius
                    center_depth = innermost_blue[1]
                    center_px, center_py = world_to_image(center_x, center_depth)
                    
                    # Draw turning circle
                    for angle in range(0, 181, 10):  # 0-180 degrees in 10 degree steps
                        rad = np.radians(angle)
                        x = center_x - turn_radius * np.cos(rad)
                        y = center_depth + turn_radius * np.sin(rad)
                        px, py = world_to_image(x, y)
                        cv2.circle(viz_image, (px, py), 1, (180, 0, 180), -1)
        
        # Draw path with visual enhancements (using original code's path display format)
        path_pixels = []
        if self.path:
            for world_x, world_y in self.path:
                px, py = world_to_image(world_x, world_y)
                path_pixels.append((px, py))
        
        if len(path_pixels) >= 2:
            # Draw outline first for better visibility
            cv2.polylines(viz_image, [np.array(path_pixels)], False, (0, 0, 0), 7)
            
            # Draw color gradient path
            for i in range(len(path_pixels) - 1):
                ratio = i / (len(path_pixels) - 1)
                # Color gradient: red (near) -> yellow -> green (far)
                if ratio < 0.5:
                    r, g, b = 255, int(255 * ratio * 2), 0
                else:
                    r, g, b = int(255 * (1 - (ratio - 0.5) * 2)), 255, 0
                
                # Draw line segments
                cv2.line(viz_image, path_pixels[i], path_pixels[i + 1], (b, g, r), 4)
        
        # Draw target point with special marker
        if self.target_point:
            tx, td = self.target_point
            tpx, tpy = world_to_image(tx, td)
            
            # Draw target circle 
            cv2.circle(viz_image, (tpx, tpy), 8, (0, 0, 255), -1)
            cv2.circle(viz_image, (tpx, tpy), 10, (255, 255, 255), 2)
            
            # Draw line from car to target
            cv2.line(viz_image, (car_x, car_y), (tpx, tpy), (0, 0, 255), 2)
            
            # Label the target point
            target_text = f"{td:.1f}m"
            cv2.putText(viz_image, target_text, (tpx + 5, tpy - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add path info text
        path_info = f"Path: {len(self.path) if self.path else 0} points" + (f", {self.path[-1][1]:.1f}m" if self.path and len(self.path) > 0 else "")
        cv2.putText(viz_image, path_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add steering info
        steer_text = f"Steering: {self.steering_angle:.2f}"
        cv2.putText(viz_image, steer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Add U-turn status
        if in_uturn:
            uturn_text = f"U-TURN DETECTED: following {uturn_side} cones"
            cv2.putText(viz_image, uturn_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Add pure pursuit info
            pp_text = f"Safety margin: {self.safety_margin:.1f}m, Lookahead: {self.lookahead_distance:.1f}m"
            cv2.putText(viz_image, pp_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        return viz_image


def plan_path_with_cones(self, cones, depths):
    """
    Plan a path using provided cone detections.
    This maintains compatibility with the original interface.
    
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
        
        # Filter cones
        filtered_cones, filtered_depths = self._filter_cones(cones, depths)
        if not filtered_cones:
            print("No cones after filtering")
            self.path = None
            return
        
        # Convert to world coordinates and process
        world_cones = self._to_world_coordinates(filtered_cones, filtered_depths)
        
        # Pair cones
        self._pair_cones(world_cones)
        
        # Connect boundary cones
        self._connect_boundary_cones()
        
        # Find target point for pure pursuit
        self._find_target_point()
        
        # Calculate steering angle
        self._calculate_steering()
        
        # Smooth steering
        self._smooth_steering()
        
        # Create path for compatibility
        self._create_compatible_path()
        
        # Always visualize when called
        if self.visualize and hasattr(self.zed_camera, 'rgb_image') and self.zed_camera.rgb_image is not None:
            self.debug_image = self.draw_path(self.zed_camera.rgb_image.copy())
            cv2.imshow("Pure Pursuit Path", self.debug_image)
            cv2.waitKey(1)
        
        print(f"Path successfully created with {len(self.path) if self.path else 0} points!")
    
    except Exception as e:
        print(f"Error in path planning with cones: {e}")
        import traceback
        traceback.print_exc()
        self.path = None

def plan_path_with_fusion(self, lidar_boundaries=None, lidar_points=None):
    """
    Plan path using camera-detected cones and LiDAR data if available.
    Maintains compatibility with the original interface.
    
    Args:
        lidar_boundaries: Dictionary with 'left_boundary' and 'right_boundary' points from LiDAR
        lidar_points: Raw LiDAR points for early cone detection
        
    Returns:
        True if path planning succeeded, False otherwise
    """
    try:
        # First process camera detections
        self._process_detections()
        
        # If LiDAR boundaries are available, enhance cone pairs
        if lidar_boundaries and ('left_boundary' in lidar_boundaries or 'right_boundary' in lidar_boundaries):
            self._enhance_with_lidar_boundaries(lidar_boundaries)
        
        # Connect boundary cones
        self._connect_boundary_cones()
        
        # Find target point for pure pursuit
        self._find_target_point()
        
        # Calculate steering angle
        self._calculate_steering()
        
        # Smooth steering
        self._smooth_steering()
        
        # Create path for compatibility
        self._create_compatible_path()
        
        # Always visualize when called
        if self.visualize and hasattr(self.zed_camera, 'rgb_image') and self.zed_camera.rgb_image is not None:
            self.debug_image = self.draw_path(self.zed_camera.rgb_image.copy())
            cv2.imshow("Pure Pursuit Path", self.debug_image)
            cv2.waitKey(1)
        
        return len(self.cone_pairs) > 0
        
    except Exception as e:
        print(f"Error in fusion path planning: {e}")
        import traceback
        traceback.print_exc()
        self.path = None
        return False
