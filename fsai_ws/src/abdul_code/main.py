import carla
import time
import numpy as np
import cv2
import pygame
from ultralytics import YOLO
import torch
import os
from path_planner import AsyncPathPlanner, find_target_point, calculate_lookahead_distance
from lidar_integration import RoboSenseLidar, SensorFusion, visualize_camera_only, visualize_full_fusion
import argparse
import pickle

class VehicleController:
    def __init__(self):
        self.throttle = 2.2
        self.steer = 0.4
        self.brake = 0.1
        self.last_steer = 0.9
        self.reverse = False
        
        # Ultra-low speed settings
        self.MAX_THROTTLE = 2.25  # Very low throttle
        self.MIN_THROTTLE = 0.05  # Minimal throttle
        self.BRAKE_FORCE = 0.8    # Strong brake
        
        # Initialize speed control
        self.target_speed = 2.2  # 1 m/s (very slow)
        self.last_speed_error = 0
        self.speed_error_sum = 0
        self.kp = 0.05  # Proportional gain
        self.ki = 0.001  # Integral gain (very small)
        self.kd = 0.02  # Derivative gain
        
        # Path planner - now using the async version
        self.path_planner = AsyncPathPlanner()
        
        # Current position (for simulation)
        self.current_position = [320, 350]  # Center bottom of image
        
        # Corrected cone class indices
        self.YELLOW_CONE_IDX = 1  # Remapped from 0
        self.BLUE_CONE_IDX = 2    # Remapped from 1
        self.ORANGE_CONE_IDX = 0  # Remapped from 2
        
    def compute_control(self, vehicle, detected_cones, depth_map=None):
        try:
            # Default settings - never let throttle drop below this
            self.throttle = 1.0
            self.steer = 0.0
            self.brake = 0.0

            # Get current speed
            # Get current speed
            velocity = vehicle.get_velocity()
            current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Target speed - slightly higher to ensure movement
            self.target_speed = 0.9
            
            # PID speed control
            speed_error = self.target_speed - current_speed
            p_term = self.kp * speed_error
            
            # Integral term with anti-windup
            if abs(speed_error) < 1.0:
                self.speed_error_sum = np.clip(self.speed_error_sum + speed_error, -10, 10)
            i_term = self.ki * self.speed_error_sum
            
            # Derivative term
            d_term = self.kd * (speed_error - self.last_speed_error)
            self.last_speed_error = speed_error
            
            # Calculate throttle adjustment
            throttle_adjustment = p_term + i_term + d_term
            
            # Strong braking if overspeeding
            if current_speed > self.target_speed + 0.5:
                self.throttle = 0
                self.brake = self.BRAKE_FORCE
            else:
                # Higher minimum throttle to ensure car never stops
                self.throttle = np.clip(0.18 + throttle_adjustment, 0.15, 0.3)
                self.brake = 0

            # Variables to track cone visibility and path status
            yellow_visible = False
            blue_visible = False
            has_path = False
            emergency_turn = False
            turn_direction = 0  # 0=straight, 1=right, -1=left
            u_turn_detected = False
            
            # Track previous path points to detect path going outside circuit
            if not hasattr(self, 'prev_path_points'):
                self.prev_path_points = None
                self.erratic_path_counter = 0
            
            # Initialize turn tracking
            if not hasattr(self, 'pending_turn'):
                self.pending_turn = False
                self.turn_direction_pending = 0
                self.turn_countdown = 0
            
            # Initialize for early turning simulation
            if not hasattr(self, 'simulated_turn_progress'):
                self.simulated_turn_progress = 0
                self.simulated_turn_direction = 0

            if detected_cones is not None:
                cones = detected_cones.cpu().numpy()
                
                # Filter cones by confidence
                high_conf_cones = cones[cones[:, 4] > 0.3]
                
                # Use corrected class indices
                yellow_cones = high_conf_cones[high_conf_cones[:, 5] == self.YELLOW_CONE_IDX]
                blue_cones = high_conf_cones[high_conf_cones[:, 5] == self.BLUE_CONE_IDX]
                
                # Check cone visibility
                yellow_visible = len(yellow_cones) > 0
                blue_visible = len(blue_cones) > 0
                
                # Pre-path detection of turns based on cone distribution
                if yellow_visible and blue_visible:
                    # Get cone positions
                    yellow_xs = [(cone[0] + cone[2]) / 2 for cone in yellow_cones]
                    yellow_ys = [(cone[1] + cone[3]) / 2 for cone in yellow_cones]
                    blue_xs = [(cone[0] + cone[2]) / 2 for cone in blue_cones]
                    blue_ys = [(cone[1] + cone[3]) / 2 for cone in blue_cones]
                    
                    # Find closest cones to the car (highest y-values)
                    if yellow_ys:
                        closest_yellow_idx = np.argmax(yellow_ys)
                        closest_yellow_x = yellow_xs[closest_yellow_idx]
                        closest_yellow_y = yellow_ys[closest_yellow_idx]
                    else:
                        closest_yellow_x = 0
                        closest_yellow_y = 0
                        
                    if blue_ys:
                        closest_blue_idx = np.argmax(blue_ys)
                        closest_blue_x = blue_xs[closest_blue_idx]
                        closest_blue_y = blue_ys[closest_blue_idx]
                    else:
                        closest_blue_x = 0
                        closest_blue_y = 0
                    
                    # Check for turns by looking at furthest visible cones
                    if len(yellow_ys) >= 2 and len(blue_ys) >= 2:
                        # Sort by distance (lowest y-value = furthest away)
                        yellow_indices = np.argsort(yellow_ys)
                        blue_indices = np.argsort(blue_ys)
                        
                        # Get furthest cones
                        furthest_yellow_x = yellow_xs[yellow_indices[0]]
                        furthest_blue_x = blue_xs[blue_indices[0]]
                        
                        # Detect potential turns
                        if furthest_yellow_x - furthest_blue_x > 100:
                            if not self.pending_turn or self.turn_direction_pending != 1:
                                self.pending_turn = True
                                self.turn_direction_pending = 1
                                self.turn_countdown = 10  # Frames to start turning
                        elif furthest_blue_x - furthest_yellow_x > 100:
                            if not self.pending_turn or self.turn_direction_pending != -1:
                                self.pending_turn = True
                                self.turn_direction_pending = -1
                                self.turn_countdown = 10  # Frames to start turning
                    
                    # Extreme early detection for U-turns - check for cone alignment patterns
                    if len(yellow_cones) >= 3 and len(blue_cones) >= 3:
                        # Check yellow cone alignment to detect right U-turn
                        sorted_yellow = sorted(yellow_cones, key=lambda x: (x[1] + x[3]) / 2)
                        y_diffs = []
                        x_diffs = []
                        for i in range(len(sorted_yellow) - 1):
                            y1 = (sorted_yellow[i][1] + sorted_yellow[i][3]) / 2
                            y2 = (sorted_yellow[i+1][1] + sorted_yellow[i+1][3]) / 2
                            x1 = (sorted_yellow[i][0] + sorted_yellow[i][2]) / 2
                            x2 = (sorted_yellow[i+1][0] + sorted_yellow[i+1][2]) / 2
                            y_diffs.append(abs(y1 - y2))
                            x_diffs.append(abs(x1 - x2))
                        
                        if len(y_diffs) >= 2 and len(x_diffs) >= 2:
                            # If cones have similar y-positions but large x-differences, likely a U-turn
                            if max(y_diffs) < 60 and max(x_diffs) > 100:
                                self.pending_turn = True
                                self.turn_direction_pending = 1  # Right U-turn
                                self.turn_countdown = 5  # Immediate turn
                        
                        # Check blue cone alignment to detect left U-turn
                        sorted_blue = sorted(blue_cones, key=lambda x: (x[1] + x[3]) / 2)
                        y_diffs = []
                        x_diffs = []
                        for i in range(len(sorted_blue) - 1):
                            y1 = (sorted_blue[i][1] + sorted_blue[i][3]) / 2
                            y2 = (sorted_blue[i+1][1] + sorted_blue[i+1][3]) / 2
                            x1 = (sorted_blue[i][0] + sorted_blue[i][2]) / 2
                            x2 = (sorted_blue[i+1][0] + sorted_blue[i+1][2]) / 2
                            y_diffs.append(abs(y1 - y2))
                            x_diffs.append(abs(x1 - x2))
                        
                        if len(y_diffs) >= 2 and len(x_diffs) >= 2:
                            if max(y_diffs) < 60 and max(x_diffs) > 100:
                                self.pending_turn = True
                                self.turn_direction_pending = -1  # Left U-turn
                                self.turn_countdown = 5  # Immediate turn
                
                # Handle pending turns from early detection
                if self.pending_turn:
                    self.turn_countdown -= 1
                    
                    if self.turn_countdown <= 0:
                        # Start simulated turn
                        self.simulated_turn_direction = self.turn_direction_pending
                        self.simulated_turn_progress = 10  # Number of frames to simulate turn
                        self.pending_turn = False
                
                # Apply simulated turn if active
                if self.simulated_turn_progress > 0:
                    # Add strong bias in the anticipated direction
                    bias_strength = 0.3 * (self.simulated_turn_progress / 10)  # Fade out as we progress
                    self.steer += self.simulated_turn_direction * bias_strength
                    self.simulated_turn_progress -= 1
                
                # Update path planner with new cone detections
                self.path_planner.update_cones(yellow_cones, blue_cones)
                
                # Get the latest path
                path_points, curvature = self.path_planner.get_latest_path()
                
                # Check for erratic path (going outside circuit)
                if path_points is not None and len(path_points) > 3:
                    # Check if we've had a previous path
                    if self.prev_path_points is not None and len(self.prev_path_points) > 3:
                        # Calculate path change
                        path_change = np.mean(np.abs(path_points[:min(len(path_points), len(self.prev_path_points))] - 
                                                self.prev_path_points[:min(len(path_points), len(self.prev_path_points))]))
                        
                        # If path changed dramatically
                        if path_change > 50:
                            self.erratic_path_counter += 1
                        else:
                            self.erratic_path_counter = max(0, self.erratic_path_counter - 1)
                    
                    # Store current path for next comparison
                    self.prev_path_points = path_points.copy()
                    
                    # If path has been erratic, add centering bias
                    if self.erratic_path_counter > 3:
                        # Assume center of image is safe, bias toward center
                        center_bias = -0.2 if self.steer > 0 else 0.2
                        self.steer += center_bias
                
                # Store path for visualization
                if path_points is not None and len(path_points) > 0:
                    VehicleController.path_points = path_points
                    self.path_points = path_points
                    has_path = True
                    
                    # Use extremely far lookahead for ultra-early turn detection
                    close_lookahead = 5     # Very close for immediate response
                    mid_lookahead = 50      # Increased for earlier detection
                    far_lookahead = 150     # Extremely far for super-early detection
                    
                    # Find target points
                    close_point = find_target_point(path_points, self.current_position, close_lookahead)
                    mid_point = find_target_point(path_points, self.current_position, mid_lookahead)
                    far_point = find_target_point(path_points, self.current_position, far_lookahead)
                    
                    # Check for U-turn based on curvature
                    if abs(curvature) > 0.15:
                        u_turn_detected = True
                    
                    # Detect significant turns
                    significant_turn = abs(curvature) > 0.1
                    
                    if close_point is not None and mid_point is not None and far_point is not None:
                        # Calculate steering based on blended target points
                        image_center = 320
                        close_error = close_point[0] - image_center
                        mid_error = mid_point[0] - image_center
                        far_error = far_point[0] - image_center
                        
                        # Extreme weighting toward far lookahead for very early turning
                        if u_turn_detected:
                            blended_error = 0.15 * close_error + 0.35 * mid_error + 0.5 * far_error
                        elif significant_turn:
                            blended_error = 0.15 * close_error + 0.35 * mid_error + 0.5 * far_error
                        else:
                            blended_error = 0.2 * close_error + 0.3 * mid_error + 0.5 * far_error
                        
                        # Extra responsive steering for tight turns
                        steering_divisor = 0.6 if u_turn_detected else (0.8 if significant_turn else 0.9)
                        desired_steer = np.clip(blended_error / (image_center * steering_divisor), -1.0, 1.0)
                        
                        # Apply enhanced curvature compensation with stronger effect
                        curvature_multiplier = 20.0 if u_turn_detected else (15.0 if significant_turn else 10.0)
                        curvature_factor = min(abs(curvature) * curvature_multiplier, 0.9)
                        
                        # Detect turning direction based on path curvature
                        if curvature > 0.01:  # Left turn (positive curvature)
                            turn_direction = -1
                            # Make left turns more aggressive when yellow cones are detected ahead
                            # Check for yellow cones in front region
                            yellow_ahead = any(cone for cone in yellow_cones if 
                                              (cone[1] + cone[3])/2 > 280 and  # High Y (close)
                                              abs((cone[0] + cone[2])/2 - 320) < 150)  # Center-ish X

                            # Increase left turn aggression if yellow cones are directly ahead
                            if yellow_ahead:
                                # More aggressive left turn to avoid yellow cones ahead
                                left_boost = 0.25 if u_turn_detected else 0.18
                                desired_steer = desired_steer * (1.0 + curvature_factor)  # More amplification
                                desired_steer -= left_boost * curvature_factor * 1.2  # Extra boost
                                print("Yellow cone ahead during left turn - increasing turn aggression")
                            else:
                                # Normal left turn
                                left_boost = 0.18 if u_turn_detected else 0.1
                                desired_steer = desired_steer * (1.0 + curvature_factor * 0.8)
                                desired_steer -= left_boost * curvature_factor
                        elif curvature < -0.01:  # Right turn (negative curvature)
                            # Keep right turns as they were
                            turn_direction = 1
                            right_boost = 0.2 if u_turn_detected else 0.1
                            desired_steer = desired_steer * (1.0 + curvature_factor)
                            desired_steer += right_boost * curvature_factor
                        
                        # Extremely aggressive steering changes in U-turns
                        steer_diff = desired_steer - self.last_steer
                        max_steer_change = 0.5 if u_turn_detected else (0.4 if significant_turn else 0.3)
                        steer_diff = np.clip(steer_diff, -max_steer_change, max_steer_change)
                        self.steer = self.last_steer + steer_diff
                        self.last_steer = self.steer
                        
                        # Check if there's a very sharp turn ahead
                        far_threshold = image_center * (0.1 if u_turn_detected else 0.2)
                        close_threshold = image_center * (0.05 if u_turn_detected else 0.1)
                        
                        if (abs(far_error) > far_threshold) and (abs(close_error) < close_threshold):
                            # Very sharp turn ahead - add strong bias immediately
                            turn_bias_value = 0.6 if u_turn_detected else 0.5
                            turn_bias = np.sign(far_error) * turn_bias_value
                            self.steer += turn_bias
                        
                        # Add centering bias for safety
                        if yellow_visible and blue_visible:
                            # Calculate average x-positions
                            avg_yellow_x = np.mean([(cone[0] + cone[2]) / 2 for cone in yellow_cones])
                            avg_blue_x = np.mean([(cone[0] + cone[2]) / 2 for cone in blue_cones])
                            
                            # Calculate center between cones
                            cone_center_x = (avg_yellow_x + avg_blue_x) / 2
                            
                            if abs(cone_center_x - image_center) > 50:  # If off-center
                                center_bias = (cone_center_x - image_center) / 500  # Small correction
                                self.steer += center_bias
                
                # Add this after the path planning logic but before emergency handling

                # Check for blue cones that are getting close and preemptively reduce left turning
                if blue_visible:
                    # Find how many blue cones are in the "danger zone" (close and to the left)
                    danger_blue_cones = [cone for cone in blue_cones if 
                                         cone[3] > 250 and  # Y position is high (close to car)
                                         (cone[0] + cone[2])/2 < 250]  # X position is left of center
                    
                    # If there are blue cones in danger zone, limit left turning, but consider if we're in a planned left turn
                    if len(danger_blue_cones) > 0:
                        closest_danger_y = max([cone[3] for cone in danger_blue_cones])
                        danger_factor = min(1.0, (closest_danger_y - 250) / 100)
                        
                        if turn_direction == -1:  # We're in a planned left turn
                            # Check if there are yellow cones ahead that need to be avoided
                            yellow_ahead = any(cone for cone in yellow_cones if 
                                              cone[3] > 280 and abs((cone[0] + cone[2])/2 - 320) < 150)
                            
                            if yellow_ahead:
                                # Less restrictive limit if yellow cones are directly ahead (need to turn left more)
                                max_left_steer = -0.3 - (0.3 * (1.0 - danger_factor))  # Between -0.3 and -0.6
                                print(f"Yellow ahead during left turn - allowing sharper left turn: {max_left_steer:.2f}")
                            else:
                                # Normal blue cone avoidance
                                max_left_steer = -0.1 - (0.3 * (1.0 - danger_factor))  # Between -0.1 and -0.4
                            
                            # Apply the limit, but less aggressively during planned turns
                            if self.steer < max_left_steer:
                                self.steer = max_left_steer
                                print(f"Limiting left turn due to blue cone proximity: {max_left_steer:.2f}")
                        else:
                            # Not in a planned left turn, apply normal restrictions
                            max_left_steer = -0.1 - (0.3 * (1.0 - danger_factor))
                            if self.steer < 0:
                                self.steer = max(self.steer, max_left_steer)
                                print(f"Preemptively limiting left turn due to blue cone proximity: {max_left_steer:.2f}")

                # EMERGENCY HANDLING FOR ONE-SIDED CONE VISIBILITY
                if not has_path and (yellow_visible or blue_visible) and not (yellow_visible and blue_visible):
                    emergency_turn = True
                    
                    if yellow_visible and not blue_visible:
                        # Only yellow cones visible (should be on right) - turn left aggressively
                        emergency_steer = -0.8  # Very aggressive left turn
                        turn_direction = -1
                        
                        # Calculate average x-position of yellow cones to adjust steering
                        yellow_xs = []
                        for cone in yellow_cones:
                            x1, y1, x2, y2 = cone[:4]
                            center_x = (x1 + x2) / 2
                            yellow_xs.append(center_x)
                        
                        if yellow_xs:
                            avg_yellow_x = np.mean(yellow_xs)
                            if avg_yellow_x > 500:  # Far right
                                emergency_steer = -0.5  # Moderate left turn
                            elif avg_yellow_x < 400:  # Closer to center
                                emergency_steer = -1.0  # Maximum left turn
                                
                    elif blue_visible and not yellow_visible:
                        # Only blue cones visible (should be on left) - turn right
                        emergency_steer = 0.5  # Turn right moderately
                        turn_direction = 1
                        
                        # Calculate average x-position of blue cones to adjust steering
                        blue_xs = []
                        for cone in blue_cones:
                            x1, y1, x2, y2 = cone[:4]
                            center_x = (x1 + x2) / 2
                            blue_xs.append(center_x)
                        
                        if blue_xs:
                            avg_blue_x = np.mean(blue_xs)
                            if avg_blue_x < 140:  # Far left
                                emergency_steer = 0.3  # Gentle right turn
                            elif avg_blue_x > 240:  # Closer to center
                                emergency_steer = 0.7  # Sharper right turn
                    
                    # Apply emergency steering with immediate response
                    steer_diff = emergency_steer - self.last_steer
                    max_steer_change = 0.4  # Very fast emergency response
                    steer_diff = np.clip(steer_diff, -max_steer_change, max_steer_change)
                    self.steer = self.last_steer + steer_diff
                    self.last_steer = self.steer
            
            # Special handling for U-turns - always maintain higher throttle
            if u_turn_detected:
                # Keep moving through U-turns
                self.throttle = max(self.throttle, 0.3)
                self.brake = 0
            else:
                # Adjust throttle based on turning - but never too low
                if abs(self.steer) > 0.7:
                    # Very sharp turn
                    self.throttle = max(0.15, self.throttle * 0.7)  # Higher minimum in turns
                elif abs(self.steer) > 0.4:
                    # Moderate turn
                    self.throttle = max(0.18, self.throttle * 0.8)  # Less reduction
                    
                # Speed up slightly on straightaways to maintain momentum
                if abs(self.steer) < 0.2 and not emergency_turn:
                    self.throttle = max(self.throttle, 0.25)  # Minimum throttle on straights
                
            # Never let throttle drop too low - keep the car moving at all times
            self.throttle = max(0.15, self.throttle)

            # Safety check to avoid hitting cones that are too close
            if yellow_visible and blue_visible:
                # Get the closest blue and yellow cones
                closest_blue_y = max([cone[3] for cone in blue_cones]) if len(blue_cones) > 0 else 0
                closest_yellow_y = max([cone[3] for cone in yellow_cones]) if len(yellow_cones) > 0 else 0
                
                # More aggressive limiting for left turns near blue cones
                if closest_blue_y > 280 and self.steer < -0.25:  # Reduced threshold (was 300/-0.3)
                    self.steer = max(self.steer, -0.3)  # Less aggressive left steering (was -0.4)
                    print("Too close to blue cone - strictly limiting left turn")
                
                # Keep the yellow cone check as is
                if closest_yellow_y > 300 and self.steer > 0.3:  # Close to yellow cone and turning right
                    self.steer = min(self.steer, 0.4)  # Limit right turn
                    print("Too close to yellow cone - limiting right turn")

            return carla.VehicleControl(
                throttle=self.throttle,
                steer=self.steer,
                brake=self.brake,
                reverse=self.reverse
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return carla.VehicleControl(throttle=0.01, brake=0.1)


class SensorData:
    def __init__(self):
        self.rgb_img = None          # BGR image for visualization
        self.depth_img = None        # Colorized depth for visualization
        self.depth_raw = None        # Raw depth data
        self.detections = None       # Camera cone detections
        self.lidar_points = None     # LiDAR point cloud
        self.path_points = None      # Path planner output
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update = time.time()


def process_image(image):
    """Process RGB camera image"""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    # Return the BGR channels (first three channels)
    bgr_array = array[:, :, :3]
    return bgr_array


def process_depth(image):
    """Process depth camera image"""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    bgr_array = array[:, :, :3]
    normalized = cv2.normalize(bgr_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    # Convert to RGB for display in Pygame
    depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
    return depth_colormap, bgr_array


def main():
    try:
        # Initialize Pygame for visualization
        pygame.init()
        display = pygame.display.set_mode((1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Formula Student AI - Camera View")
        
        # Add controller attribute to store path
        VehicleController.path_points = None

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        best_pt_path = os.path.join(base_dir, 'runs', 'best.pt')

        # Load YOLO model for camera-based detection
        model_path = best_pt_path
        model = YOLO(model_path)
        dummy_img = np.zeros((384, 640, 3), dtype=np.uint8)
        model(dummy_img, verbose=False)
        
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()

        # Spawn vehicle
        spawn_transform = carla.Transform(
            carla.Location(x=-35.0, y=0.0, z=5.0),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_transform)

        # Set up RGB camera
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', '640')
        rgb_bp.set_attribute('image_size_y', '384')
        rgb_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(
            carla.Location(x=2.5, z=2.0),
            carla.Rotation(pitch=-15)
        )
        rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)

        # Set up depth camera
        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '640')
        depth_bp.set_attribute('image_size_y', '384')
        depth_bp.set_attribute('fov', '110')
        depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
        
        # Set up LiDAR
        # After creating the LiDAR object
        lidar = RoboSenseLidar(world, vehicle, sensor_height=1.8, lidar_pos_x=0.0, lidar_pos_y=0.0)
        lidar.setup_sensor()
        lidar.enable_3d_visualization()  # Add this line to enable 3D visualization
        print("LiDAR system initialized")

        # Set up sensor fusion
        sensor_fusion = SensorFusion(camera_width=640, camera_height=384)
        print("Sensor fusion initialized")
        
        # Initialize vehicle controller and sensor data
        controller = VehicleController()
        sensor_data = SensorData()

        # Add these arguments to your parser
        parser = argparse.ArgumentParser(description='Formula Student AI')
        parser.add_argument('--collect-data', action='store_true', help='Enable data collection for PointNet training')
        parser.add_argument('--data-dir', type=str, default='pointnet_training_data', help='Directory to save training data')
        args = parser.parse_args()

        # In your main function, after initializing sensors:
        if args.collect_data:
            os.makedirs(args.data_dir, exist_ok=True)
            print(f"Data collection enabled. Saving to {args.data_dir}")
            data_collection_counter = 0

        # RGB camera callback
        def rgb_callback(image):
            try:
                # Process RGB image
                bgr_image = process_image(image)
                rgb_for_model = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                
                # Run YOLO model for cone detection
                results = model(rgb_for_model, 
                                conf=0.15,
                                iou=0.4,
                                max_det=50,
                                verbose=False)
                
                if len(results) > 0:
                    sensor_data.detections = results[0].boxes.data
                    
                    # Update sensor fusion with camera detections
                    sensor_fusion.update_camera_detections(sensor_data.detections)
                    
                    # Update path planner if detections are available
                    if sensor_data.detections is not None and len(sensor_data.detections) > 0:
                        cones = sensor_data.detections.cpu().numpy()
                        high_conf_cones = cones[cones[:, 4] > 0.3]
                        
                        # Using corrected class indices
                        yellow_cones = high_conf_cones[high_conf_cones[:, 5] == controller.YELLOW_CONE_IDX]
                        blue_cones = high_conf_cones[high_conf_cones[:, 5] == controller.BLUE_CONE_IDX]
                        
                        controller.path_planner.update_cones(yellow_cones, blue_cones)
                        path_points, _ = controller.path_planner.get_latest_path()
                        
                        sensor_data.path_points = path_points
                
                # Store the image for visualization
                sensor_data.rgb_img = bgr_image
                sensor_data.frame_count += 1
                
            except Exception as e:
                import traceback
                traceback.print_exc()

        # Depth camera callback
        def depth_callback(image):
            try:
                depth_colormap, depth_raw = process_depth(image)
                sensor_data.depth_img = depth_colormap
                sensor_data.depth_raw = depth_raw
                
                # Update sensor fusion with depth information
                sensor_fusion.update_depth_info(depth_raw)
                
            except Exception as e:
                import traceback
                traceback.print_exc()

        # Set up camera callbacks
        rgb_camera.listen(rgb_callback)
        depth_camera.listen(depth_callback)

        # Main game loop
        running = True
        clock = pygame.time.Clock()
        frame_count = 0
        last_time = time.time()

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        controller.reverse = not controller.reverse
                    elif event.key == pygame.K_SPACE:
                        vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0))

            # Get LiDAR data
            lidar_cones = lidar.get_detected_cones()
            
            # Update sensor fusion with LiDAR detections
            sensor_fusion.update_lidar_detections(lidar_cones)
            
            # Get fused detections
            yellow_cones, blue_cones, orange_cones = sensor_fusion.fuse_detections()
            
            # Compute vehicle control - can use original detections or fused detections
            if sensor_data.rgb_img is not None:
                # Use original method with camera-only detections for compatibility
                control = controller.compute_control(
                    vehicle, 
                    sensor_data.detections,  # Original camera detections
                    sensor_data.depth_raw
                )
                vehicle.apply_control(control)

            # In the main loop, after handling events:

            # Get LiDAR data
            if hasattr(lidar, 'get_detected_cones'):
                lidar_cones = lidar.get_detected_cones()
                lidar_points = lidar.get_point_cloud() if hasattr(lidar, 'get_point_cloud') else None
                
                # Store point cloud in sensor_data
                sensor_data.lidar_points = lidar_points
                
                # Update sensor fusion with LiDAR detections
                sensor_fusion.update_lidar_detections(lidar_cones)

            # Get fused detections
            yellow_cones, blue_cones, orange_cones = sensor_fusion.fuse_detections()

            # Visualization
            if sensor_data.rgb_img is not None:
                # Switch between visualization modes
                use_fusion_viz = True  # Set to False to use camera-only visualization

                # In the main loop, update the visualize_full_fusion call
                if use_fusion_viz:
                    viz_img = visualize_full_fusion(
                        sensor_data.rgb_img,
                        sensor_data.detections,
                        lidar_cones if 'lidar_cones' in locals() else None, 
                        (yellow_cones, blue_cones, orange_cones),
                        controller,
                        sensor_data.depth_raw,
                        sensor_data.path_points,
                        lidar.get_point_cloud() if hasattr(lidar, 'get_point_cloud') else None,   # Add this line
                        lidar.intensity_list if hasattr(lidar, 'intensity_list') else None  # Add this line
                    )
                else:
                    viz_img = visualize_camera_only(
                        sensor_data.rgb_img,
                        sensor_data.detections,
                        controller,
                        sensor_data.depth_raw,
                        sensor_data.path_points
                    )
                
                # Convert from BGR to RGB for Pygame display
                viz_img_rgb = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
                viz_surface = pygame.surfarray.make_surface(np.transpose(viz_img_rgb, (1, 0, 2)))
                display.blit(viz_surface, (0, 0))
                pygame.display.flip()

                # Calculate FPS
                frame_count += 1
                if frame_count >= 10:
                    now = time.time()
                    fps = frame_count / (now - last_time)
                    pygame.display.set_caption(f"Formula Student AI - Camera View - FPS: {fps:.1f}")
                    frame_count = 0
                    last_time = now

            # Limit frame rate
            clock.tick(30)

    finally:
        # Clean up resources
        pygame.quit()
        
        # Destroy camera sensors
        if 'rgb_camera' in locals():
            rgb_camera.destroy()
        if 'depth_camera' in locals():
            depth_camera.destroy()
            
        # Shut down LiDAR
        if 'lidar' in locals() and lidar is not None:
            lidar.shutdown()
            
        # Shut down path planner
        if 'controller' in locals() and hasattr(controller, 'path_planner'):
            try:
                if hasattr(controller.path_planner, 'shutdown'):
                    controller.path_planner.shutdown()
            except Exception as e:
                pass
                
        # Destroy vehicle
        if 'vehicle' in locals():
            vehicle.destroy()


if __name__ == '__main__':
    main()

