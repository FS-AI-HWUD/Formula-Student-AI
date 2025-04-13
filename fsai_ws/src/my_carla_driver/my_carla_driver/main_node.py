#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
import threading
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys
import os

# Import our modules
from zed_2i import Zed2iCamera
from path_planning import PathPlanner

class CarlaDriverNode(Node):
    def __init__(self):
        super().__init__('carla_driver')
        
        # Declare parameters
        self.declare_parameter('model_path', '/home/dalek/Desktop/runs/detect/train8/weights/best.pt')
        self.declare_parameter('use_carla', True)
        
        # Camera parameters
        self.declare_parameter('camera.resolution_width', 1280)
        self.declare_parameter('camera.resolution_height', 720)
        self.declare_parameter('camera.fps', 30)
        
        # Path planning parameters
        self.declare_parameter('path_planning.depth_min', 1.0)
        self.declare_parameter('path_planning.depth_max', 20.0)
        self.declare_parameter('path_planning.cone_spacing', 1.5)
        self.declare_parameter('path_planning.visualize', True)
        
        # Carla parameters
        self.declare_parameter('carla.host', 'localhost')
        self.declare_parameter('carla.port', 2000)
        self.declare_parameter('carla.timeout', 10.0)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.use_carla = self.get_parameter('use_carla').value
        
        self.camera_width = self.get_parameter('camera.resolution_width').value
        self.camera_height = self.get_parameter('camera.resolution_height').value
        self.camera_fps = self.get_parameter('camera.fps').value
        
        self.depth_min = self.get_parameter('path_planning.depth_min').value
        self.depth_max = self.get_parameter('path_planning.depth_max').value
        self.cone_spacing = self.get_parameter('path_planning.cone_spacing').value
        self.visualize = self.get_parameter('path_planning.visualize').value
        
        self.carla_host = self.get_parameter('carla.host').value
        self.carla_port = self.get_parameter('carla.port').value
        self.carla_timeout = self.get_parameter('carla.timeout').value
        
        # Bridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()
        
        # Create publishers for visualization
        self.rgb_pub = self.create_publisher(Image, 'visualization/rgb', 10)
        self.depth_pub = self.create_publisher(Image, 'visualization/depth', 10)
        self.path_pub = self.create_publisher(Image, 'visualization/path', 10)
        
        # Initialize Carla and camera if needed
        self.world = None
        self.vehicle = None
        self.zed_camera = None
        self.path_planner = None
        
        # Timer for the main processing loop
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)
        
        # Thread for visualization
        self.vis_thread = None
        self.running = True
        
        # Initialize everything
        self.setup()
    
    def setup(self):
        """Initialize Carla, camera, and path planner"""
        if self.use_carla:
            try:
                self.get_logger().info("Connecting to CARLA server...")
                
                # Import Carla here to avoid issues if not installed
                import carla
                
                client = carla.Client(self.carla_host, self.carla_port)
                client.set_timeout(self.carla_timeout)
                self.world = client.get_world()
                self.get_logger().info("Connected to CARLA world successfully")
                
                # Spawn a vehicle
                self.vehicle = self.spawn_vehicle(self.world)
                
                # Initialize the ZED camera
                resolution = (self.camera_width, self.camera_height)
                self.zed_camera = Zed2iCamera(self.world, self.vehicle, 
                                            resolution=resolution, 
                                            fps=self.camera_fps, 
                                            model_path=self.model_path)
                
                if not self.zed_camera.setup():
                    self.get_logger().error("Failed to setup ZED camera")
                    return
                
                # Initialize the path planner
                self.path_planner = PathPlanner(self.zed_camera, 
                                                depth_min=self.depth_min, 
                                                depth_max=self.depth_max, 
                                                cone_spacing=self.cone_spacing, 
                                                visualize=self.visualize)
                
                self.get_logger().info("Carla setup completed successfully")
                
                # Start visualization in a separate thread if requested
                if self.visualize:
                    self.vis_thread = threading.Thread(target=self.visualization_thread)
                    self.vis_thread.daemon = True
                    self.vis_thread.start()
                    
            except ImportError:
                self.get_logger().error("Carla Python API not found. Install or add to PYTHONPATH.")
                self.use_carla = False
            except Exception as e:
                self.get_logger().error(f"Error setting up Carla: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
                self.use_carla = False
    
    def spawn_vehicle(self, world):
        """Spawn a vehicle in the CARLA world."""
        try:
            import carla
            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.*')[0]
            
            if not vehicle_bp:
                self.get_logger().error("No vehicle blueprints found")
                available_bps = [bp.id for bp in blueprint_library.filter('vehicle.*')]
                self.get_logger().info(f"Available blueprints: {available_bps}")
                raise ValueError("No vehicle blueprint available")
                
            self.get_logger().info(f"Using vehicle blueprint: {vehicle_bp.id}")
            
            spawn_transform = carla.Transform(
                carla.Location(x=-35.0, y=0.0, z=5.0),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )
            vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
            self.get_logger().info(f"Vehicle spawned at {spawn_transform.location}")
            
            time.sleep(2.0)
            if not vehicle.is_alive:
                raise RuntimeError("Vehicle failed to spawn or is not alive")
            return vehicle
        except Exception as e:
            self.get_logger().error(f"Error spawning vehicle: {e}")
            raise
    
    def set_car_controls(self, steering, speed):
        """
        Set the car's steering and speed.
        
        Args:
            steering (float): Steering value from -1.0 (full left) to 1.0 (full right).
            speed (float): Speed in meters per second.
        """
        self.get_logger().info(f"Setting car controls: Steering = {steering:.2f}, Speed = {speed:.2f} m/s")
        
        if self.use_carla and self.vehicle:
            try:
                import carla
                # Convert to Carla control
                control = carla.VehicleControl()
                control.steer = steering
                
                # Simple speed control - adapt as needed
                if speed > 0:
                    control.throttle = min(speed / 10.0, 1.0)  # Simple mapping to 0-1
                    control.brake = 0.0
                else:
                    control.throttle = 0.0
                    control.brake = min(abs(speed) / 10.0, 1.0)
                
                self.vehicle.apply_control(control)
            except Exception as e:
                self.get_logger().error(f"Error setting car controls: {e}")
    
    def control_car(self):
        """Control the car based on the path planner output"""
        if not self.path_planner:
            return
            
        # Get the steering value from the path planner
        steering = self.path_planner.calculate_steering(lookahead_distance=5.0)
        
        # Calculate speed based on path curvature
        base_speed = 5.0  # Base speed for straight sections (m/s)
        turn_speed = 2.0  # Reduced speed for turns (m/s)
        speed = base_speed  # Default speed
        
        # If no path is available, stop the car
        if self.path_planner.path is None or len(self.path_planner.path) < 2:
            speed = 0.0
            steering = 0.0
            self.get_logger().info("No path available. Stopping the car.")
        else:
            # Estimate path curvature using the steering angle
            abs_steering = abs(steering)
            if abs_steering > 0.3:  # If steering angle is large, reduce speed
                speed = turn_speed
                self.get_logger().info(f"Sharp turn detected (steering = {steering:.2f}). Reducing speed to {speed:.2f} m/s.")
            else:
                self.get_logger().info(f"Straight section (steering = {steering:.2f}). Setting speed to {speed:.2f} m/s.")
        
        # Set the car's steering and speed
        self.set_car_controls(steering, speed)
    
    def timer_callback(self):
        """Main processing callback for ROS2 timer"""
        if not self.use_carla or not self.zed_camera or not self.path_planner:
            return
            
        try:
            # Process camera frame
            self.zed_camera.process_frame()
            
            # Plan path
            self.path_planner.plan_path()
            
            # Control the car
            self.control_car()
            
            # Publish visualization images
            self.publish_images()
            
        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def publish_images(self):
        """Publish visualization images to ROS2 topics"""
        if not self.zed_camera or not self.zed_camera.rgb_image or not self.zed_camera.depth_image:
            return
            
        try:
            # RGB image
            rgb_img = self.zed_camera.rgb_image.copy()
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
            rgb_msg.header.stamp = self.get_clock().now().to_msg()
            rgb_msg.header.frame_id = "camera"
            self.rgb_pub.publish(rgb_msg)
            
            # Depth image
            _, depth_img = self.zed_camera.depth_image
            if depth_img is not None:
                depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="bgr8")
                depth_msg.header.stamp = self.get_clock().now().to_msg()
                depth_msg.header.frame_id = "camera"
                self.depth_pub.publish(depth_msg)
            
            # Path visualization
            if self.path_planner and rgb_img is not None:
                path_img = self.path_planner.draw_path(rgb_img.copy())
                path_msg = self.bridge.cv2_to_imgmsg(path_img, encoding="bgr8")
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.header.frame_id = "camera"
                self.path_pub.publish(path_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error publishing images: {e}")
    
    def visualization_thread(self):
        """Thread for showing visualization windows"""
        while self.running and rclpy.ok():
            try:
                if self.zed_camera and self.zed_camera.rgb_image is not None:
                    # Display RGB image with path
                    if self.path_planner:
                        rgb_with_path = self.path_planner.draw_path(self.zed_camera.rgb_image.copy())
                        cv2.imshow("RGB Camera with Path", rgb_with_path)
                    
                    # Display depth image
                    if self.zed_camera.depth_image is not None:
                        _, depth_img = self.zed_camera.depth_image
                        cv2.imshow("Depth Map", depth_img)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                self.get_logger().error(f"Error in visualization thread: {e}")
            
            time.sleep(0.033)  # ~30fps
    
    def destroy_node(self):
        """Clean up resources on node shutdown"""
        self.running = False
        
        # Stop visualization thread
        if self.vis_thread and self.vis_thread.is_alive():
            self.vis_thread.join(timeout=1.0)
        
        # Clean up Carla resources
        if self.zed_camera:
            self.zed_camera.shutdown()
        
        if self.vehicle:
            self.vehicle.destroy()
        
        cv2.destroyAllWindows()
        self.get_logger().info("Cleanup complete")
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CarlaDriverNode()
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
