"""
CARLA ZED2i camera publisher for ROS2
Connects to an existing CARLA instance and publishes camera feeds using ZED2i naming conventions
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import carla
import cv2
import time
import traceback

class ZED2iCameraPublisher(Node):
    def __init__(self):
        super().__init__('zed2i_camera_publisher')
        
        self.declare_parameter('carla.host', 'localhost')
        self.declare_parameter('carla.port', 2000)
        self.declare_parameter('camera.width', 1280)
        self.declare_parameter('camera.height', 720)
        self.declare_parameter('camera.fov', 90.0)
        self.declare_parameter('camera.x', 1.5)
        self.declare_parameter('camera.y', 0.0)
        self.declare_parameter('camera.z', 1.8)
        self.declare_parameter('left_topic', '/zed2i/rgb/image')
        self.declare_parameter('depth_topic', '/zed2i/depth/image')
        self.declare_parameter('camera_info_topic', '/zed2i/camera_info')
        self.declare_parameter('debug_level', 1)
        
        self.host = self.get_parameter('carla.host').value
        self.port = self.get_parameter('carla.port').value
        self.width = self.get_parameter('camera.width').value
        self.height = self.get_parameter('camera.height').value
        self.fov = self.get_parameter('camera.fov').value
        self.camera_x = self.get_parameter('camera.x').value
        self.camera_y = self.get_parameter('camera.y').value
        self.camera_z = self.get_parameter('camera.z').value
        self.left_topic = self.get_parameter('left_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.debug_level = self.get_parameter('debug_level').value
        
        self.rgb_pub = self.create_publisher(Image, self.left_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)
        
        self.bridge = CvBridge()
        
        self.client = None
        self.world = None
        self.vehicle = None
        self.rgb_camera = None
        self.depth_camera = None
        self.rgb_count = 0
        self.depth_count = 0
        self.connect_to_carla()
        
        status_period = 1.0 if self.debug_level > 1 else 5.0
        self.timer = self.create_timer(status_period, self.status_callback)
        
    def connect_to_carla(self):
        """Connect to an existing CARLA instance"""
        try:
            # Connect to CARLA
            self.get_logger().info(f"Connecting to CARLA at {self.host}:{self.port}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.get_logger().info(f"Connected to CARLA world")
            self.find_vehicle()
            
            if self.vehicle:
                self.setup_cameras()
            else:
                self.get_logger().error("No vehicle found to attach cameras to")
                
        except Exception as e:
            self.get_logger().error(f"Error connecting to CARLA: {e}")
            self.get_logger().error(traceback.format_exc())
    
    def find_vehicle(self):
        """Find an existing vehicle in the world"""
        try:
            all_actors = self.world.get_actors()
            if self.debug_level > 0:
                self.get_logger().info(f"Found {len(all_actors)} actors in the world")
            
            vehicles = all_actors.filter('vehicle.*')
            if self.debug_level > 0:
                self.get_logger().info(f"Found {len(vehicles)} vehicles in the world")
            
            if vehicles:
                self.vehicle = vehicles[0]
                self.get_logger().info(f"Using vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id})")
            else:
                self.get_logger().warn("No vehicles found in the world")
                
        except Exception as e:
            self.get_logger().error(f"Error finding vehicle: {e}")
            self.get_logger().error(traceback.format_exc())
    
    def setup_cameras(self):
        """Setup RGB and depth cameras on the vehicle"""
        if not self.vehicle:
            self.get_logger().error("No vehicle available for cameras")
            return
            
        try:
            blueprint_library = self.world.get_blueprint_library()
            rgb_bp = blueprint_library.find('sensor.camera.rgb')
            rgb_bp.set_attribute('image_size_x', str(self.width))
            rgb_bp.set_attribute('image_size_y', str(self.height))
            rgb_bp.set_attribute('fov', str(self.fov))
            camera_transform = carla.Transform(
                carla.Location(x=self.camera_x, y=self.camera_y, z=self.camera_z)
            )
            
            self.rgb_camera = self.world.spawn_actor(
                rgb_bp, 
                camera_transform, 
                attach_to=self.vehicle
            )
            
            self.rgb_camera.listen(self.rgb_callback)
            self.get_logger().info(f"RGB camera created and attached to vehicle {self.vehicle.id}")
            
            depth_bp = blueprint_library.find('sensor.camera.depth')
            depth_bp.set_attribute('image_size_x', str(self.width))
            depth_bp.set_attribute('image_size_y', str(self.height))
            depth_bp.set_attribute('fov', str(self.fov))
            
            self.depth_camera = self.world.spawn_actor(
                depth_bp, 
                camera_transform, 
                attach_to=self.vehicle
            )
            self.depth_camera.listen(self.depth_callback)
            self.get_logger().info("Depth camera created")
            
        except Exception as e:
            self.get_logger().error(f"Error setting up cameras: {e}")
            self.get_logger().error(traceback.format_exc())
    
    def rgb_callback(self, image):
        """Process and publish RGB image"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            rgb_image = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
            
            msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            
            self.rgb_pub.publish(msg)
            self.rgb_count += 1
            self.publish_camera_info(image.width, image.height)
            if self.rgb_count == 1 or (self.debug_level > 1 and self.rgb_count % 100 == 0):
                self.get_logger().info(f"RGB image published. Size: {image.height}x{image.width}, Count: {self.rgb_count}")
            
        except Exception as e:
            self.get_logger().error(f"Error in RGB callback: {e}")
    
    def depth_callback(self, image):
        """Process and publish depth image"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))

            normalized = array.astype(np.float32) / 255.0
            depth_array = (normalized[:, :, 2] + normalized[:, :, 1] * 256.0 + normalized[:, :, 0] * 256.0 * 256.0) / 1000.0
            msg = self.bridge.cv2_to_imgmsg(depth_array, encoding='32FC1')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            self.depth_pub.publish(msg)
            self.depth_count += 1

            if self.depth_count == 1 or (self.debug_level > 1 and self.depth_count % 100 == 0):
                self.get_logger().info(f"Depth image published. Size: {image.height}x{image.width}, Count: {self.depth_count}")
            if self.debug_level > 1 and self.depth_count % 100 == 0:
                min_depth = np.min(depth_array)
                max_depth = np.max(depth_array)
                mean_depth = np.mean(depth_array)
                self.get_logger().info(f"Depth range: {min_depth:.2f}m - {max_depth:.2f}m, Mean: {mean_depth:.2f}m")
            
        except Exception as e:
            self.get_logger().error(f"Error in depth callback: {e}")
    
    def publish_camera_info(self, width, height):
        """Publish camera calibration information"""
        try:
            msg = CameraInfo()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            msg.width = width
            msg.height = height
            focal_length = width / (2.0 * np.tan(np.radians(self.fov) / 2.0))
            cx = width / 2.0
            cy = height / 2.0
            
            msg.k = [
                focal_length, 0.0, cx,
                0.0, focal_length, cy,
                0.0, 0.0, 1.0
            ]
            msg.distortion_model = "plumb_bob"
            msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            msg.p = [
                focal_length, 0.0, cx, 0.0,
                0.0, focal_length, cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
            self.camera_info_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing camera info: {e}")
    
    def status_callback(self):
        """Print status information periodically"""
        if self.debug_level > 0:
            self.get_logger().info(f"Status: RGB frames={self.rgb_count}, Depth frames={self.depth_count}")
        if self.vehicle:
            try:
                if self.vehicle.is_alive:
                    if self.debug_level > 0:
                        self.get_logger().info(f"Vehicle {self.vehicle.id} is active")
                else:
                    self.get_logger().warn(f"Vehicle {self.vehicle.id} is no longer alive")
                    self.vehicle = None
                    self.find_vehicle()
            except Exception:
                self.get_logger().warn("Could not check vehicle status, attempting to find a new one")
                self.vehicle = None
                self.find_vehicle()
    
    def destroy_node(self):
        """Clean up CARLA resources"""
        self.get_logger().info("Shutting down ZED2i camera publisher...")
        
        # Stop and destroy cameras
        if hasattr(self, 'rgb_camera') and self.rgb_camera:
            try:
                self.rgb_camera.stop()
                self.rgb_camera.destroy()
                self.get_logger().info("RGB camera destroyed")
            except Exception as e:
                self.get_logger().error(f"Error destroying RGB camera: {e}")
        
        if hasattr(self, 'depth_camera') and self.depth_camera:
            try:
                self.depth_camera.stop()
                self.depth_camera.destroy()
                self.get_logger().info("Depth camera destroyed")
            except Exception as e:
                self.get_logger().error(f"Error destroying depth camera: {e}")
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    node = ZED2iCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
