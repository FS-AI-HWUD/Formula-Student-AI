"""
CARLA IMU Sensor Node
This node publishes IMU data from a CARLA vehicle with configurable noise parameters.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
import carla
import math
import numpy as np
import transforms3d
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import threading


class CarlaIMUNode(Node):
    def __init__(self):
        super().__init__('carla_imu_sensor_node')
        
        self.declare_parameter('carla.host', 'localhost')
        self.declare_parameter('carla.port', 2000)
        self.declare_parameter('carla.timeout', 10.0)
        self.declare_parameter('update_frequency', 100.0)
        
        self.declare_parameter('imu.noise_accel_stddev_x', 0.01)  
        self.declare_parameter('imu.noise_accel_stddev_y', 0.01)  
        self.declare_parameter('imu.noise_accel_stddev_z', 0.01)  
        self.declare_parameter('imu.noise_gyro_stddev_x', 0.001)  
        self.declare_parameter('imu.noise_gyro_stddev_y', 0.001)  
        self.declare_parameter('imu.noise_gyro_stddev_z', 0.001)  
        self.declare_parameter('imu.noise_gyro_bias_x', 0.0)  
        self.declare_parameter('imu.noise_gyro_bias_y', 0.0)  
        self.declare_parameter('imu.noise_gyro_bias_z', 0.0)  
        self.declare_parameter('imu.sensor_tick', 0.01)
        self.declare_parameter('imu.noise_seed', 0)
        
        self.declare_parameter('frame_id.imu', 'imu_link')
        self.declare_parameter('frame_id.vehicle', 'base_link')
        
        self.host = self.get_parameter('carla.host').value
        self.port = self.get_parameter('carla.port').value
        self.timeout = self.get_parameter('carla.timeout').value
        self.update_freq = self.get_parameter('update_frequency').value
        
        self.accel_stddev_x = self.get_parameter('imu.noise_accel_stddev_x').value
        self.accel_stddev_y = self.get_parameter('imu.noise_accel_stddev_y').value
        self.accel_stddev_z = self.get_parameter('imu.noise_accel_stddev_z').value
        self.gyro_stddev_x = self.get_parameter('imu.noise_gyro_stddev_x').value
        self.gyro_stddev_y = self.get_parameter('imu.noise_gyro_stddev_y').value
        self.gyro_stddev_z = self.get_parameter('imu.noise_gyro_stddev_z').value
        self.gyro_bias_x = self.get_parameter('imu.noise_gyro_bias_x').value
        self.gyro_bias_y = self.get_parameter('imu.noise_gyro_bias_y').value
        self.gyro_bias_z = self.get_parameter('imu.noise_gyro_bias_z').value
        self.sensor_tick = self.get_parameter('imu.sensor_tick').value
        self.noise_seed = self.get_parameter('imu.noise_seed').value
        
        self.imu_frame = self.get_parameter('frame_id.imu').value
        self.vehicle_frame = self.get_parameter('frame_id.vehicle').value
        
        if self.noise_seed > 0:
            np.random.seed(self.noise_seed)
        
        self.imu_pub = self.create_publisher(Imu, '/carla/imu_sensor', 10)
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.client = None
        self.world = None
        self.vehicle = None
        self.imu_sensor = None
        
        self.latest_imu_data = None
        self.data_lock = threading.Lock()
        
        self.imu_count = 0
        
        self.setup_carla()
        
        self.imu_timer = self.create_timer(1.0/self.update_freq, self.imu_publish_callback)
        self.status_timer = self.create_timer(5.0, self.status_callback)
    
    def setup_carla(self):
        """Connect to CARLA and setup vehicle and sensors."""
        try:
            self.get_logger().info(f"Connecting to CARLA at {self.host}:{self.port}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            
            self.world = self.client.get_world()
            self.get_logger().info("Connected to CARLA world")
            
            # Find existing vehicle
            self.find_vehicle()
            
            if self.vehicle:
                self.setup_imu_sensor()
            else:
                self.get_logger().error("No vehicle found to attach IMU sensor to")
                
        except Exception as e:
            self.get_logger().error(f"Error connecting to CARLA: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def find_vehicle(self):
        """Find an existing vehicle in the world."""
        try:
            all_actors = self.world.get_actors()
            vehicles = all_actors.filter('vehicle.*')
            
            if vehicles:
                self.vehicle = vehicles[0]
                self.get_logger().info(f"Found vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id})")
            else:
                self.get_logger().warn("No vehicles found in the world")
                
        except Exception as e:
            self.get_logger().error(f"Error finding vehicle: {str(e)}")
    
    def setup_imu_sensor(self):
        """Setup IMU sensor on the vehicle."""
        if not self.vehicle:
            return
            
        try:
            blueprint_library = self.world.get_blueprint_library()
            imu_bp = blueprint_library.find('sensor.other.imu')
            if not imu_bp:
                self.get_logger().error("IMU blueprint not found")
                return
            imu_bp.set_attribute('sensor_tick', str(self.sensor_tick))
            imu_bp.set_attribute('noise_accel_stddev_x', str(self.accel_stddev_x))
            imu_bp.set_attribute('noise_accel_stddev_y', str(self.accel_stddev_y))
            imu_bp.set_attribute('noise_accel_stddev_z', str(self.accel_stddev_z))
            imu_bp.set_attribute('noise_gyro_stddev_x', str(self.gyro_stddev_x))
            imu_bp.set_attribute('noise_gyro_stddev_y', str(self.gyro_stddev_y))
            imu_bp.set_attribute('noise_gyro_stddev_z', str(self.gyro_stddev_z))
            imu_bp.set_attribute('noise_gyro_bias_x', str(self.gyro_bias_x))
            imu_bp.set_attribute('noise_gyro_bias_y', str(self.gyro_bias_y))
            imu_bp.set_attribute('noise_gyro_bias_z', str(self.gyro_bias_z))
            
            if self.noise_seed > 0:
                imu_bp.set_attribute('noise_seed', str(self.noise_seed))
            
            imu_transform = carla.Transform(carla.Location(z=0.5))
            self.imu_sensor = self.world.spawn_actor(
                imu_bp,
                imu_transform,
                attach_to=self.vehicle
            )
            self.imu_sensor.listen(self.imu_callback)
            self.get_logger().info("IMU sensor created and attached to vehicle")
            
        except Exception as e:
            self.get_logger().error(f"Error setting up IMU sensor: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def imu_callback(self, data):
        """Process IMU data from CARLA."""
        try:
            with self.data_lock:
                self.latest_imu_data = {
                    'timestamp': data.timestamp,
                    'frame': data.frame,
                    'accelerometer': data.accelerometer,
                    'gyroscope': data.gyroscope,
                    'compass': data.compass,
                    'transform': data.transform
                }
            self.imu_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error in IMU callback: {str(e)}")
    
    def imu_publish_callback(self):
        """Publish IMU data at the specified rate."""
        if not self.latest_imu_data:
            return
            
        try:
            with self.data_lock:
                imu_data = self.latest_imu_data.copy() if self.latest_imu_data else None
            
            if not imu_data:
                return
            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.imu_frame
            
            msg.linear_acceleration.x = imu_data['accelerometer'].x
            msg.linear_acceleration.y = imu_data['accelerometer'].y
            msg.linear_acceleration.z = imu_data['accelerometer'].z
            
            msg.angular_velocity.x = imu_data['gyroscope'].x
            msg.angular_velocity.y = imu_data['gyroscope'].y
            msg.angular_velocity.z = imu_data['gyroscope'].z
            
            transform = imu_data['transform']
            
            roll = math.radians(transform.rotation.roll)
            pitch = math.radians(transform.rotation.pitch)
            yaw = math.radians(transform.rotation.yaw)
            
            quat = transforms3d.euler.euler2quat(roll, pitch, yaw, 'sxyz')
            
            msg.orientation.w = float(quat[0])
            msg.orientation.x = float(quat[1])
            msg.orientation.y = float(quat[2])
            msg.orientation.z = float(quat[3])
            
            orientation_cov = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.02]
            msg.orientation_covariance = orientation_cov

            ang_vel_cov = [
                float(self.gyro_stddev_x**2), 0.0, 0.0,
                0.0, float(self.gyro_stddev_y**2), 0.0,
                0.0, 0.0, float(self.gyro_stddev_z**2)
            ]
            msg.angular_velocity_covariance = ang_vel_cov

            accel_cov = [
                float(self.accel_stddev_x**2), 0.0, 0.0,
                0.0, float(self.accel_stddev_y**2), 0.0,
                0.0, 0.0, float(self.accel_stddev_z**2)
            ]
            msg.linear_acceleration_covariance = accel_cov
            self.imu_pub.publish(msg)
            self.broadcast_tf(imu_data['transform'])
            
        except Exception as e:
            self.get_logger().error(f"Error publishing IMU data: {str(e)}")
    
    def broadcast_tf(self, imu_transform=None):
        """Broadcast the transform from vehicle to IMU sensor."""
        if not self.vehicle:
            return
            
        try:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.vehicle_frame
            t.child_frame_id = self.imu_frame
            
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.5
            
            t.transform.rotation.w = 1.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            
            self.tf_broadcaster.sendTransform(t)
            
        except Exception as e:
            self.get_logger().error(f"Error broadcasting transform: {str(e)}")
    
    def status_callback(self):
        """Report status information."""
        self.get_logger().info(f"Status: IMU frames={self.imu_count}")
        if self.vehicle:
            try:
                if self.vehicle.is_alive:
                    self.get_logger().info(f"Vehicle {self.vehicle.id} is active")
                    
                    loc = self.vehicle.get_location()
                    self.get_logger().info(f"Vehicle location: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")
                    
                    vel = self.vehicle.get_velocity()
                    speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    self.get_logger().info(f"Vehicle speed: {speed:.2f} km/h")
                    
                    if self.latest_imu_data:
                        accel = self.latest_imu_data['accelerometer']
                        gyro = self.latest_imu_data['gyroscope']
                        self.get_logger().info(f"IMU accel: ({accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f}) m/s², "
                                              f"gyro: ({gyro.x:.2f}, {gyro.y:.2f}, {gyro.z:.2f}) rad/s")
                else:
                    self.get_logger().warn(f"Vehicle {self.vehicle.id} is no longer alive")
                    self.vehicle = None
                    self.find_vehicle()
            except Exception:
                self.get_logger().warn("Could not check vehicle status, attempting to find a new one")
                self.vehicle = None
                self.find_vehicle()
    
    def destroy_node(self):
        """Clean up resources before shutting down."""
        self.get_logger().info("Shutting down IMU node")
        if self.imu_sensor:
            try:
                self.imu_sensor.stop()
                self.imu_sensor.destroy()
                self.get_logger().info("IMU sensor destroyed")
            except Exception as e:
                self.get_logger().error(f"Error destroying IMU sensor: {str(e)}")
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CarlaIMUNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()