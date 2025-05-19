#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import os
import traceback
from ament_index_python.packages import get_package_share_directory

class LidarClusterNode(Node):
    def __init__(self):
        super().__init__('lidar_cluster_node')
        
        # Declare parameters
        self.declare_parameter("raw_lidar_topic_name", "/carla/lidar")
        self.declare_parameter("lidar_cluster_topic_name", "/perception/lidar_cluster")
        self.declare_parameter("cone_markers_topic", "/perception/cone_markers")
        self.declare_parameter("eps", 0.7)
        self.declare_parameter("min_points", 2)
        self.declare_parameter("debug_mode", True)
        
        # Cone filter parameters
        self.declare_parameter("min_distance", 1.0)
        self.declare_parameter("max_distance", 30.0)
        self.declare_parameter("fov_angle", 360.0)
        self.declare_parameter("min_height", 0.1)
        self.declare_parameter("max_height", 0.5)
        self.declare_parameter("max_width", 0.5)
        self.declare_parameter("ground_z_threshold", -2.2)
        
        # Get parameter values
        self.raw_topic = self.get_parameter("raw_lidar_topic_name").get_parameter_value().string_value
        self.cluster_topic = self.get_parameter("lidar_cluster_topic_name").get_parameter_value().string_value
        self.cone_markers_topic = self.get_parameter("cone_markers_topic").get_parameter_value().string_value
        self.eps = self.get_parameter("eps").get_parameter_value().double_value
        self.min_points = self.get_parameter("min_points").get_parameter_value().integer_value
        self.debug_mode = self.get_parameter("debug_mode").get_parameter_value().bool_value
        
        # Get cone filter parameters
        self.min_distance = self.get_parameter("min_distance").get_parameter_value().double_value
        self.max_distance = self.get_parameter("max_distance").get_parameter_value().double_value
        self.fov_angle = self.get_parameter("fov_angle").get_parameter_value().double_value
        self.min_height = self.get_parameter("min_height").get_parameter_value().double_value
        self.max_height = self.get_parameter("max_height").get_parameter_value().double_value
        self.max_width = self.get_parameter("max_width").get_parameter_value().double_value
        self.ground_z_threshold = self.get_parameter("ground_z_threshold").get_parameter_value().double_value
        
        # Create subscription and publishers
        self.sub = self.create_subscription(PointCloud2, self.raw_topic, self.callback, 10)
        self.pub = self.create_publisher(PointCloud2, self.cluster_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.cone_markers_topic, 10)
        # self.test_marker_timer = self.create_timer(1.0, self.timer_callback_test_marker)

        # Some debig info
        self.get_logger().info(f"LidarClusterNode initialized. Subscribing to: {self.raw_topic}")
        self.get_logger().info(f"DBSCAN parameters: eps={self.eps}, min_points={self.min_points}")
        self.get_logger().info(f"Field of view: {self.fov_angle} degrees, Distance range: {self.min_distance}-{self.max_distance}m")
        
        # Message counter
        self.msg_count = 0
    
    def extract_points(self, msg):
        """Extract XYZ points from PointCloud2 message"""
        try:
            points_list = []
            for p in pc2.read_points(msg, skip_nans=True):
                if hasattr(p, "__len__") and len(p) >= 3:
                    points_list.append([float(p[0]), float(p[1]), float(p[2])])
            
            if not points_list:
                return np.empty((0, 3), dtype=np.float32)
                
            return np.array(points_list, dtype=np.float32)
            
        except Exception as e:
            self.get_logger().error(f"Error extracting points: {e}")
            self.get_logger().error(traceback.format_exc())
            return np.empty((0, 3), dtype=np.float32)
    
    def filter_points_for_cones(self, points):
        """Even more relaxed filtering for cone detection"""
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
        
        front_fov_angle = 90.0
        
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        angles = np.arctan2(points[:, 1], points[:, 0]) * 180.0 / np.pi
        distance_mask = (distances >= 0.5) & (distances <= 50.0)
        angle_mask = (angles >= -front_fov_angle) & (angles <= front_fov_angle)
        height_mask = (points[:, 2] > -2.5) & (points[:, 2] < -0.5)
        combined_mask = distance_mask & angle_mask & height_mask
        filtered_points = points[combined_mask]
        ground_mask = points[:, 2] <= -2.3
        ground_points = points[distance_mask & angle_mask & ground_mask]
        
        if self.debug_mode:
            self.get_logger().info(f"Filtered {len(filtered_points)} points for cone detection (from {len(points)} total)")
            if len(distances) > 0:
                self.get_logger().info(f"Distance range: {np.min(distances):.2f} to {np.max(distances):.2f}m")
            if len(angles) > 0:
                self.get_logger().info(f"Angle range: {np.min(angles):.2f} to {np.max(angles):.2f} degrees")
            if len(points) > 0:
                self.get_logger().info(f"Z range: {np.min(points[:, 2]):.2f} to {np.max(points[:, 2]):.2f}m")
        
        return filtered_points, ground_points
    
    def dbscan_clustering(self, points, eps=0.5, min_points=3):
        """
        DBSCAN clustering for point clouds.
        
        Args:
            points: Nx3 array of points
            eps: Maximum distance between points in a cluster
            min_points: Minimum number of points to form a cluster
            
        Returns:
            labels: Cluster ID for each point (-1 for noise points)
        """
        if len(points) == 0:
            return np.array([], dtype=np.int32)
        labels = np.full(len(points), -1, dtype=np.int32)
        cluster_id = 0
        
        for i in range(len(points)):
            if labels[i] != -1:
                continue
            neighbors = []
            for j in range(len(points)):
                if np.linalg.norm(points[i] - points[j]) < eps:
                    neighbors.append(j)
            if len(neighbors) < min_points:
                continue
            cluster_id += 1
            labels[i] = cluster_id
            j = 0
            while j < len(neighbors):
                point_id = neighbors[j]
                if labels[point_id] == -1:
                    labels[point_id] = cluster_id
                    
                    new_neighbors = []
                    for k in range(len(points)):
                        if np.linalg.norm(points[point_id] - points[k]) < eps:
                            new_neighbors.append(k)
                    
                    if len(new_neighbors) >= min_points:
                        for neighbor in new_neighbors:
                            if neighbor not in neighbors:
                                neighbors.append(neighbor)
                
                j += 1
        
        num_clusters = cluster_id
        num_noise = np.sum(labels == -1)
        
        if num_clusters > 0:
            self.get_logger().info(f"DBSCAN found {num_clusters} clusters and {num_noise} noise points")
        
        return labels
    
    def deduplicate_positions(self, positions, min_distance=0.5):
        """
        Remove duplicate positions that are too close to each other
        
        Args:
            positions: List of position arrays [x, y, z]
            min_distance: Minimum distance between unique positions
            
        Returns:
            Deduplicated list of positions
        """
        if len(positions) <= 1:
            return positions
        positions_array = np.array(positions)
        keep_mask = np.ones(len(positions), dtype=bool)
        for i in range(len(positions)):
            if not keep_mask[i]:
                continue
            distances = np.linalg.norm(positions_array - positions_array[i], axis=1)
            close_indices = np.where((distances > 0) & (distances < min_distance))[0]
            for j in close_indices:
                if j > i:
                    keep_mask[j] = False
        return [pos for i, pos in enumerate(positions) if keep_mask[i]]
    
    def filter_cone_clusters(self, points, labels):
        """
        Filter clusters to keep only those that resemble traffic cones
        """
        if len(points) == 0 or len(np.unique(labels)) <= 1:
            return [], labels
        
        unique_clusters = np.unique(labels)
        if -1 in unique_clusters:
            unique_clusters = unique_clusters[unique_clusters != -1]
        
        if len(unique_clusters) == 0:
            return [], labels
        
        filtered_labels = labels.copy()
        cone_centroids = []
        for cluster_id in unique_clusters:
            cluster_mask = (labels == cluster_id)
            cluster_points = points[cluster_mask]
            centroid = np.mean(cluster_points, axis=0)
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)
            width_x = max_bounds[0] - min_bounds[0]
            width_y = max_bounds[1] - min_bounds[1]
            height = abs(max_bounds[2] - min_bounds[2])
            width = max(width_x, width_y)
            self.get_logger().info(f"Cluster {cluster_id}: {len(cluster_points)} points, width={width:.2f}m, height={height:.2f}m")
            is_cone = (
                width <= 0.5 and
                height <= 0.7 and
                height >= 0.15 and
                len(cluster_points) >= 3
            )
            
            if not is_cone:
                if width > 0.5:
                    self.get_logger().info(f"Rejected: width {width:.2f}m > 0.5m")
                elif height > 0.7:
                    self.get_logger().info(f"Rejected: height {height:.2f}m > 0.7m")
                elif height < 0.15:
                    self.get_logger().info(f"Rejected: height {height:.2f}m < 0.15m")
                elif len(cluster_points) < 3:
                    self.get_logger().info(f"Rejected: only {len(cluster_points)} points")
                filtered_labels[cluster_mask] = -1
            else:
                cone_centroids.append(centroid)
        
        if len(cone_centroids) > 0:
            self.get_logger().info(f"Found {len(cone_centroids)} cones out of {len(unique_clusters)} clusters")
        
        return cone_centroids, filtered_labels
    
    def callback(self, msg):
        try:
            self.msg_count += 1
            points = self.extract_points(msg)
            
            if len(points) == 0:
                self.get_logger().warn("No points extracted from message")
                return
            cone_points, ground_points = self.filter_points_for_cones(points)
            
            if len(cone_points) == 0:
                if self.debug_mode and self.msg_count % 10 == 0:
                    self.get_logger().info("No points passed cone filtering")
                return
            labels = self.dbscan_clustering(cone_points, self.eps, self.min_points)
            cone_centroids, filtered_labels = self.filter_cone_clusters(cone_points, labels)
            if len(cone_centroids) > 1:
                original_count = len(cone_centroids)
                cone_centroids = self.deduplicate_positions(cone_centroids, min_distance=0.5)
                if len(cone_centroids) < original_count:
                    self.get_logger().info(f"Removed {original_count - len(cone_centroids)} duplicate cone positions")
            
            self.publish_clusters(cone_points, filtered_labels, msg.header)
            
            if cone_centroids:
                self.publish_cone_markers(cone_centroids, msg.header.stamp)
            
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")
            self.get_logger().error(traceback.format_exc())
    
    def publish_clusters(self, points, labels, header):
        """Publish clustered point cloud with cluster IDs"""
        try:
            if len(points) == 0:
                return
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='cluster_id', offset=12, datatype=PointField.INT32, count=1)
            ]
            cloud_data = np.zeros(len(points), 
                                 dtype=[
                                     ('x', np.float32),
                                     ('y', np.float32),
                                     ('z', np.float32),
                                     ('cluster_id', np.int32)
                                 ])
            cloud_data['x'] = points[:, 0]
            cloud_data['y'] = points[:, 1]
            cloud_data['z'] = points[:, 2]
            cloud_data['cluster_id'] = labels
            clustered_cloud = pc2.create_cloud(header, fields, cloud_data)
            self.pub.publish(clustered_cloud)
            if len(labels) > 0:
                unique_ids = np.unique(labels)
                num_clusters = len(unique_ids) - (1 if -1 in unique_ids else 0)
                if num_clusters > 0:
                    self.get_logger().info(f"Published {num_clusters} clusters ({len(points)} points)")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing clusters: {e}")
            self.get_logger().error(traceback.format_exc())
    
    def publish_cone_markers(self, cone_centroids, stamp):
        """Publish cone markers using direct file path for mesh"""
        try:
            marker_array = MarkerArray()
            test_marker = Marker()
            test_marker.header.frame_id = "map"
            test_marker.header.stamp = stamp
            test_marker.ns = "test_markers"
            test_marker.id = 999
            test_marker.type = Marker.SPHERE
            test_marker.action = Marker.ADD
            
            test_marker.pose.position.x = 0.0
            test_marker.pose.position.y = 0.0
            test_marker.pose.position.z = 0.0
            
            test_marker.pose.orientation.w = 1.0
            
            test_marker.scale.x = 1.0
            test_marker.scale.y = 1.0
            test_marker.scale.z = 1.0
            
            test_marker.color.r = 1.0
            test_marker.color.g = 1.0
            test_marker.color.b = 0.0
            test_marker.color.a = 0.7
            
            marker_array.markers.append(test_marker)
            try:
                package_path = get_package_share_directory('lidar_cluster')
                mesh_file = os.path.join(package_path, 'meshes', 'Traffic_Cone.d')
                if os.path.exists(mesh_file):
                    self.get_logger().info(f"Mesh file found at: {mesh_file}")
                    
                    for i, centroid in enumerate(cone_centroids):
                        mesh_marker = Marker()
                        mesh_marker.header.frame_id = "map"
                        mesh_marker.header.stamp = stamp
                        mesh_marker.ns = "cone_mesh_markers"
                        mesh_marker.id = i
                        mesh_marker.type = Marker.MESH_RESOURCE
                        mesh_marker.mesh_resource = f"file://{mesh_file}"
                        mesh_marker.mesh_use_embedded_materials = True
                        mesh_marker.action = Marker.ADD
                        mesh_marker.pose.position.x = float(centroid[0])
                        mesh_marker.pose.position.y = float(centroid[1])
                        mesh_marker.pose.position.z = float(centroid[2]) + 2.0
                        mesh_marker.pose.orientation.w = 1.0
                        mesh_marker.scale.x = 0.01
                        mesh_marker.scale.y = 0.01
                        mesh_marker.scale.z = 0.01
                        mesh_marker.color.r = 1.0
                        mesh_marker.color.g = 0.5
                        mesh_marker.color.b = 0.0
                        mesh_marker.color.a = 1.0
                        marker_array.markers.append(mesh_marker)
                else:
                    self.get_logger().error(f"Mesh file not found at: {mesh_file}")
                    raise FileNotFoundError(f"Mesh file not found: {mesh_file}")      
            except Exception as e:
                self.get_logger().warn(f"Error with mesh markers, using cylinders instead: {e}")
                for i, centroid in enumerate(cone_centroids):
                    cylinder = Marker()
                    cylinder.header.frame_id = "map"
                    cylinder.header.stamp = stamp
                    cylinder.ns = "cone_cylinder_markers"
                    cylinder.id = i
                    cylinder.type = Marker.CYLINDER
                    cylinder.action = Marker.ADD
                    cylinder.pose.position.x = float(centroid[0])
                    cylinder.pose.position.y = float(centroid[1])
                    cylinder.pose.position.z = float(centroid[2]) + 2.5
                    cylinder.pose.orientation.w = 1.0
                    cylinder.scale.x = 0.5  # diameter
                    cylinder.scale.y = 0.5  # diameter
                    cylinder.scale.z = 1.0  # height
                    # Orange color
                    cylinder.color.r = 1.0
                    cylinder.color.g = 0.5
                    cylinder.color.b = 0.0
                    cylinder.color.a = 1.0
                    marker_array.markers.append(cylinder)
            self.marker_pub.publish(marker_array)
            self.get_logger().info(f"Published {len(cone_centroids)} cone markers")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing cone markers: {e}")
            self.get_logger().error(traceback.format_exc())

    # def timer_callback_test_marker(self):
    #     """Publish a test marker every second"""
    #     marker_array = MarkerArray()
        
    #     # Create a simple sphere marker
    #     marker = Marker()
    #     marker.header.frame_id = "map"  # Try with "map" first
    #     marker.header.stamp = self.get_clock().now().to_msg()
    #     marker.ns = "test_markers"
    #     marker.id = 0
    #     marker.type = Marker.SPHERE
    #     marker.action = Marker.ADD
        
    #     # Position at origin
    #     marker.pose.position.x = 0.0
    #     marker.pose.position.y = 0.0
    #     marker.pose.position.z = 0.0
        
    #     marker.pose.orientation.w = 1.0
        
    #     # Big red sphere
    #     marker.scale.x = 2.0
    #     marker.scale.y = 2.0
    #     marker.scale.z = 2.0
        
    #     marker.color.r = 1.0
    #     marker.color.g = 0.0
    #     marker.color.b = 0.0
    #     marker.color.a = 1.0
        
    #     marker_array.markers.append(marker)
        
    #     # Publish the marker array
    #     self.marker_pub.publish(marker_array)
    #     self.get_logger().info("Published test marker")

def main(args=None):
    rclpy.init(args=args)
    node = LidarClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
        node.get_logger().error(traceback.format_exc())
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()