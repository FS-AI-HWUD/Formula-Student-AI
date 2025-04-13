import carla
import numpy as np
import cv2
from ultralytics import YOLO
import queue
import threading
import time
from collections import defaultdict

class Zed2iCamera:
    def __init__(self, world, vehicle, resolution=(1280, 720), fps=30, model_path=None, use_ros2=False):
        self.world = world
        self.vehicle = vehicle
        self.resolution = resolution
        self.fps = fps
        self.model_path = model_path
        self.use_ros2 = use_ros2
        
        self.rgb_sensor = None
        self.depth_sensor = None
        self.rgb_image = None
        self.depth_image = None
        self.cone_distances = []
        self.cone_detections = []
        
        self.rgb_queue = queue.Queue(maxsize=1)
        self.depth_queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        
        # External data from ROS2 (if enabled)
        self.ext_rgb_image = None
        self.ext_depth_array = None
        self.ext_depth_image = None
        
        # Store history of cone detections for smoothing
        self.cone_history = defaultdict(list)  # Key: (cls, center_x), Value: list of (depth, timestamp)
        
        try:
            self.yolo_model = YOLO(model_path) if model_path else None
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.yolo_model = None
    
    def setup(self):
        blueprint_library = self.world.get_blueprint_library()
        
        try:
            print("Available sensor blueprints:")
            for bp in blueprint_library.filter('sensor.*'):
                print(bp.id)
            
            rgb_bp = blueprint_library.find('sensor.camera.rgb')
            if not rgb_bp:
                raise ValueError("RGB camera blueprint not found")
            
            rgb_bp.set_attribute('image_size_x', str(self.resolution[0]))
            rgb_bp.set_attribute('image_size_y', str(self.resolution[1]))
            
            rgb_transform = carla.Transform(carla.Location(x=1.5, z=1.8))
            self.rgb_sensor = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.vehicle)
            if not self.rgb_sensor.is_alive:
                raise RuntimeError("RGB camera failed to spawn or is not alive")
            self.rgb_sensor.listen(lambda image: self._process_rgb(image))
            print("RGB camera spawned successfully")
            
            depth_bp = blueprint_library.find('sensor.camera.depth')
            if not depth_bp:
                raise ValueError("Depth camera blueprint not found")
            
            depth_bp.set_attribute('image_size_x', str(self.resolution[0]))
            depth_bp.set_attribute('image_size_y', str(self.resolution[1]))
            
            self.depth_sensor = self.world.spawn_actor(depth_bp, rgb_transform, attach_to=self.vehicle)
            if not self.depth_sensor.is_alive:
                raise RuntimeError("Depth camera failed to spawn or is not alive")
            self.depth_sensor.listen(lambda image: self._process_depth(image))
            print("Depth camera spawned successfully")
            
            time.sleep(2.0)
            return True
        except Exception as e:
            print(f"Error setting up cameras: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_rgb(self, image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            rgb_image = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
            
            try:
                self.rgb_queue.put_nowait(rgb_image)
            except queue.Full:
                pass
        except Exception as e:
            print(f"Error processing RGB image: {e}")
    
    def _process_depth(self, image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            
            # Extract R, G, B channels (0 to 255)
            r = array[:, :, 0].astype(np.float32)
            g = array[:, :, 1].astype(np.float32)
            b = array[:, :, 2].astype(np.float32)
            
            # Combine R, G, B into a normalized depth value (0 to 1)
            normalized_depth = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
            print(f"Normalized depth range: min = {normalized_depth.min():.4f}, max = {normalized_depth.max():.4f}")
            
            # CARLA's depth encoding: logarithmic depth
            # Modify depth range for better accuracy
            min_depth = 0.1  # Minimum depth in meters
            max_depth = 20.0  # Reduced from 25.0 for better precision
            
            # Linear depth instead of logarithmic (more accurate for CARLA)
            depth_array = min_depth + normalized_depth * (max_depth - min_depth)
            
            # Apply depth correction factor - adjust this based on your observations
            # If closest cone is showing as 9m but should be ~3m, use a factor of 0.33
            correction_factor = 0.5  # Adjust based on testing
            depth_array = depth_array * correction_factor
            
            print(f"Depth range after scaling: min = {depth_array.min():.2f}m, max = {depth_array.max():.2f}m")
            
            # Create depth visualization
            depth_image = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            
            try:
                self.depth_queue.put_nowait((depth_array, depth_image))
            except queue.Full:
                pass
        except Exception as e:
            print(f"Error processing depth image: {e}")
    
    def set_external_images(self, rgb_image=None, depth_array=None, depth_image=None):
        with self.lock:
            if rgb_image is not None:
                self.ext_rgb_image = rgb_image
            
            if depth_array is not None:
                self.ext_depth_array = depth_array
                
                if depth_image is None and depth_array is not None:
                    normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                    self.ext_depth_image = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
                else:
                    self.ext_depth_image = depth_image
    
    def estimate_depth_from_perspective(self, y_position, image_height, camera_height=1.8, camera_angle=15):
        horizon_y = image_height * 0.4
        if y_position <= horizon_y:
            return 100.0
        
        normalized_y = (y_position - horizon_y) / (image_height - horizon_y)
        
        camera_angle_rad = np.radians(camera_angle)
        ground_angle = np.pi/2 - camera_angle_rad - np.pi/2 * (1 - normalized_y)
        
        if ground_angle > 0:
            estimated_depth = camera_height / np.tan(ground_angle)
            return min(50.0, max(1.0, estimated_depth))
        else:
            return 50.0

    def _draw_detections(self, results):
        with self.lock:
            self.cone_distances = []
            self.cone_detections = []
            if self.rgb_image is None or self.depth_image is None:
                return
            
            depth_array, _ = self.depth_image
            image_height = self.rgb_image.shape[0]
            image_width = self.rgb_image.shape[1]
            
            horizon_line = int(image_height * 0.45)
            
            detected_cones = []
            current_time = time.time()
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    
                    print(f"Detected object: Class ID = {cls}, Confidence = {conf:.2f}")
                    
                    if (cls == 0 or cls == 1) and conf > 0.3:
                        center_x = (x1 + x2) // 2
                        bottom_y = min(y2, depth_array.shape[0] - 1)
                        
                        if 0 <= bottom_y < depth_array.shape[0] and 0 <= center_x < depth_array.shape[1]:
                            y_start = max(0, bottom_y - 5)
                            x_start = max(0, center_x - 3)
                            y_end = min(depth_array.shape[0], bottom_y + 1)
                            x_end = min(depth_array.shape[1], center_x + 4)
                            
                            depth_region = depth_array[y_start:y_end, x_start:x_end]
                            if depth_region.size > 0:
                                raw_depth = float(np.median(depth_region))
                                
                                # Smooth depth using history
                                cone_key = (cls, center_x)
                                self.cone_history[cone_key].append((raw_depth, current_time))
                                
                                # Keep only recent detections (last 0.5 seconds)
                                self.cone_history[cone_key] = [(d, t) for d, t in self.cone_history[cone_key] if current_time - t < 0.5]
                                
                                if len(self.cone_history[cone_key]) > 1:
                                    depths = [d for d, t in self.cone_history[cone_key]]
                                    smoothed_depth = np.mean(depths)
                                else:
                                    smoothed_depth = raw_depth
                                
                                detected_cones.append({
                                    'box': (x1, y1, x2, y2),
                                    'cls': cls,
                                    'raw_depth': raw_depth,
                                    'smoothed_depth': smoothed_depth,
                                    'center_x': center_x,
                                    'bottom_y': bottom_y,
                                    'conf': conf,
                                    'box_size': (x2 - x1) * (y2 - y1)
                                })
            
            detected_cones.sort(key=lambda c: c['bottom_y'], reverse=True)
            
            min_depth = 2.0
            max_depth = 20.0
            
            if detected_cones:
                min_y = min(c['bottom_y'] for c in detected_cones)
                max_y = max(c['bottom_y'] for c in detected_cones)
                y_range = max_y - min_y
                
                max_box_size = max(c['box_size'] for c in detected_cones)
                
                if y_range > 0:
                    for i, cone in enumerate(detected_cones):
                        y_norm = (cone['bottom_y'] - min_y) / y_range
                        depth_factor = np.exp(3.0 * (1.0 - y_norm)) - 0.8
                        normalized_depth = depth_factor / (np.exp(3.0) - 0.8)
                        
                        position_depth = min_depth + (max_depth - min_depth) * normalized_depth
                        
                        if y_norm > 0.85:
                            position_depth *= 0.8
                        
                        box_size_norm = min(1.0, cone['box_size'] / 5000.0)
                        adjustment = 1.0 - box_size_norm * 0.3
                        position_depth *= adjustment
                        
                        if i > 0 and detected_cones[i-1]['bottom_y'] - cone['bottom_y'] < 30:
                            prev_depth = detected_cones[i-1].get('depth', None)
                            if prev_depth is not None:
                                max_jump = 4.0
                                if position_depth - prev_depth > max_jump:
                                    position_depth = prev_depth + max_jump
                        
                        # Combine smoothed depth with position-based depth
                        final_depth = 0.5 * cone['smoothed_depth'] + 0.5 * position_depth
                        cone['depth'] = final_depth
                        
                        print(f"Raw depth at ({cone['center_x']}, {cone['bottom_y']}): {cone['raw_depth']:.2f} meters, " +
                              f"Smoothed: {cone['smoothed_depth']:.2f}m, Position-based: {position_depth:.2f}m, Final: {final_depth:.2f}m")
                        
                        self.cone_distances.append(final_depth)
                        detection = {
                            'box': cone['box'],
                            'cls': cone['cls'],
                            'depth': final_depth,
                            'y_pos': cone['bottom_y']
                        }
                        self.cone_detections.append(detection)
                        
                        x1, y1, x2, y2 = cone['box']
                        color = (0, 255, 0) if cone['cls'] == 0 else (255, 0, 0)
                        cv2.rectangle(self.rgb_image, (x1, y1), (x2, y2), color, 2)
                        label = f"Cone: {final_depth:.2f}m"
                        cv2.putText(self.rgb_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            self._highlight_track_boundaries()

    def _highlight_track_boundaries(self):
        if not self.rgb_image or not self.cone_detections:
            return
        
        yellow_cones = [(c['box'], c['depth']) for c in self.cone_detections if c['cls'] == 0]
        blue_cones = [(c['box'], c['depth']) for c in self.cone_detections if c['cls'] == 1]
        
        if not yellow_cones or not blue_cones:
            return
        
        yellow_cones.sort(key=lambda x: x[1])
        blue_cones.sort(key=lambda x: x[1])
        
        if len(yellow_cones) >= 2:
            for i in range(len(yellow_cones) - 1):
                (x1, y1, x2, y2), _ = yellow_cones[i]
                (x1_next, y1_next, x2_next, y2_next), _ = yellow_cones[i + 1]
                center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
                center2 = ((x1_next + x2_next) // 2, (y1_next + y2_next) // 2)
                cv2.line(self.rgb_image, center1, center2, (0, 255, 255), 2)
        
        if len(blue_cones) >= 2:
            for i in range(len(blue_cones) - 1):
                (x1, y1, x2, y2), _ = blue_cones[i]
                (x1_next, y1_next, x2_next, y2_next), _ = blue_cones[i + 1]
                center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
                center2 = ((x1_next + x2_next) // 2, (y1_next + y2_next) // 2)
                cv2.line(self.rgb_image, center1, center2, (255, 0, 0), 2)

    def process_frame(self):
        try:
            rgb_available = False
            depth_available = False
            
            if not self.rgb_queue.empty():
                with self.lock:
                    self.rgb_image = self.rgb_queue.get()
                rgb_available = True
            
            if not self.depth_queue.empty():
                with self.lock:
                    self.depth_image = self.depth_queue.get()
                depth_available = True
            
            if rgb_available and depth_available and self.yolo_model:
                # Lower confidence threshold to detect more cones
                # The default was 0.25, let's try an even lower threshold
                results = self.yolo_model(self.rgb_image, conf=0.2)
                self._draw_detections(results)
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def shutdown(self):
        if self.rgb_sensor:
            self.rgb_sensor.stop()
            self.rgb_sensor.destroy()
        if self.depth_sensor:
            self.depth_sensor.stop()
            self.depth_sensor.destroy()
        print("ZED 2i camera shut down")