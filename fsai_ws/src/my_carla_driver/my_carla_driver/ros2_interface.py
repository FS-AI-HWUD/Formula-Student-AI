# ros2_interface.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import threading
import time
import cv2

class CarlaROS2Listener(Node):
    """
    A class that listens to ROS2 topics published by Carla and makes the data available
    to the rest of your application without changing your existing code structure.
    """
    def __init__(self):
        super().__init__('carla_listener')
        
        self.bridge = CvBridge()
        
        # Store latest images
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_depth_array = None
        
        # Setup subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/carla/vehicle4/rgb2/image',
            self.rgb_callback,
            10)
        
        self.depth_sub = self.create_subscription(
            Image,
            '/carla/vehicle4/depth3/image',
            self.depth_callback,
            10)
        
        self.get_logger().info('Carla ROS2 Listener initialized')
        
        # ROS2 spin in a separate thread
        self.ros_thread = threading.Thread(target=self._ros_spin)
        self.ros_thread.daemon = True
        self.running = True
    
    def start(self):
        """Start the ROS2 processing thread"""
        self.running = True
        self.ros_thread.start()
        self.get_logger().info('ROS2 listener thread started')
        return True
    
    def stop(self):
        """Stop the ROS2 processing thread"""
        self.running = False
        if self.ros_thread.is_alive():
            self.ros_thread.join(timeout=1.0)
        self.get_logger().info('ROS2 listener thread stopped')
    
    def _ros_spin(self):
        """Thread function for ROS2 spinning"""
        while self.running and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.01)  # Small sleep to prevent high CPU usage
    
    def rgb_callback(self, msg):
        """Process RGB image from ROS2"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting RGB image: {e}')
    
    def depth_callback(self, msg):
        """Process depth image from ROS2"""
        try:
            # Try different encodings as Carla might use various formats
            try:
                depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            except:
                try:
                    # If 32FC1 fails, try 16UC1 (common for depth images)
                    depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                    # Convert from millimeters to meters if necessary
                    if depth_array.dtype == np.uint16:
                        depth_array = depth_array.astype(np.float32) / 1000.0
                except:
                    # Fallback to trying passthrough
                    depth_array = self.bridge.imgmsg_to_cv2(msg)
            
            # Create colored depth image for visualization
            try:
                normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_image = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
                
                self.latest_depth_array = depth_array
                self.latest_depth_image = depth_image
            except Exception as e:
                self.get_logger().error(f'Error processing depth visualization: {e}')
                
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
    
    def get_latest_rgb(self):
        """Get the latest RGB image"""
        return self.latest_rgb_image
    
    def get_latest_depth(self):
        """Get the latest depth image and array"""
        return self.latest_depth_array, self.latest_depth_image


# Example integration with your main code (to be imported, not run directly)
def init_ros2():
    """Initialize ROS2 and return the listener node"""
    rclpy.init()
    listener = CarlaROS2Listener()
    listener.start()
    return listener

def shutdown_ros2(listener):
    """Shutdown ROS2"""
    if listener:
        listener.stop()
    rclpy.shutdown()
