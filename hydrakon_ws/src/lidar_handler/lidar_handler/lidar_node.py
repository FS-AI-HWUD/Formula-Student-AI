import rclpy
from rclpy.node import Node
import carla
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sys

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_handler')

        self.declare_parameter("carla_host", "localhost")
        self.declare_parameter("carla_port", 2000)

        self.host = self.get_parameter("carla_host").get_parameter_value().string_value
        self.port = self.get_parameter("carla_port").get_parameter_value().integer_value

        self.publisher_ = self.create_publisher(PointCloud2, '/carla/lidar', 10)

        self.vehicle = None
        self.world = None
        self.lidar = None

        self.get_logger().info(f"Using Python interpreter: {sys.executable}")

        self.setup()

    def setup(self):
        client = carla.Client(self.host, self.port)
        client.set_timeout(5.0)
        self.world = client.get_world()

        actors = self.world.get_actors().filter('vehicle.*')
        if not actors:
            self.get_logger().error("No vehicle found.")
            return

        self.vehicle = actors[0]
        self.setup_lidar()

    def setup_lidar(self):
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '500000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('range', '100')

        lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.2))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)

        self.lidar.listen(lambda data: self._lidar_callback(data))
        self.get_logger().info("LiDAR sensor attached and streaming.")

    def _lidar_callback(self, data):
        # Convert LiDAR raw data to PointCloud2
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        cloud_msg = self.create_pointcloud2(points[:, :3])
        self.publisher_.publish(cloud_msg)

    def create_pointcloud2(self, points):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "lidar_link"
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = False
        msg.is_bigendian = False

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.point_step = 12  # 3 * 4 bytes
        msg.row_step = msg.point_step * len(points)
        msg.data = np.asarray(points, dtype=np.float32).tobytes()

        return msg


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    rclpy.spin(node)
    rclpy.shutdown()
