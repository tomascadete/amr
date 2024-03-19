import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from sklearn.cluster import DBSCAN
from std_msgs.msg import String
import json
from nav_msgs.msg import Odometry
import transforms3d as t3d

class Perception(Node):
    def __init__(self):
        super().__init__('perception')
        self.subscription = self.create_subscription(LaserScan, 'scan', self.laser_scan_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(String, '/detected_objects_with_id', 10)
        self.occupancy_grid = None
        self.resolution = 0.1  # meters per pixel
        self.map_width = 20  # meters
        self.map_height = 20  # meters
        self.grid_width = int(self.map_width / self.resolution)
        self.grid_height = int(self.map_height / self.resolution)
        self.known_clusters = []
        self.cluster_id_counter = 0
        self.robot_pose = None


    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose


    def transform_to_global(self, local_x, local_y):
        if not self.robot_pose:
            return local_x, local_y  # Fallback if no pose is available

        # Extract robot position and orientation
        px = self.robot_pose.position.x
        py = self.robot_pose.position.y
        quaternion = (
            self.robot_pose.orientation.x,
            self.robot_pose.orientation.y,
            self.robot_pose.orientation.z,
            self.robot_pose.orientation.w
        )
        _, _, yaw = t3d.euler.quat2euler(quaternion, axes='sxyz')

        # Perform the transformation
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        global_x = px + cos_yaw * local_x - sin_yaw * local_y
        global_y = py + sin_yaw * local_x + cos_yaw * local_y

        return global_x, global_y


    def laser_scan_callback(self, msg):
        # Convert polar coordinates to Cartesian coordinates
        angles = np.arange(msg.angle_min, msg.angle_max + msg.angle_increment, msg.angle_increment)
        ranges = np.array(msg.ranges)
        valid_ranges = np.where(np.logical_and(ranges > msg.range_min, ranges < msg.range_max))
        x = ranges[valid_ranges] * np.cos(angles[valid_ranges])
        y = ranges[valid_ranges] * np.sin(angles[valid_ranges])
        x_grid = np.floor((x + self.map_width / 2) / self.resolution).astype(int)
        y_grid = np.floor((y + self.map_height / 2) / self.resolution).astype(int)

        # Initialize occupancy grid if not already done
        if self.occupancy_grid is None:
            self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

        # Update occupancy grid but beware of the grid limits
        x_grid = np.clip(x_grid, 0, self.grid_width - 1)
        y_grid = np.clip(y_grid, 0, self.grid_height - 1)
        self.occupancy_grid[y_grid, x_grid] = 255

        # Apply DBSCAN clustering on the updated occupancy grid
        occupied_coords = np.column_stack(np.nonzero(self.occupancy_grid))
        # Higher eps: more points will be included in the clusters
        # Higher min_samples: more points will be considered as core points
        # Common values for ep
        clustering = DBSCAN(eps=3, min_samples=4).fit(occupied_coords)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)


        cluster_details = []
        # Convert all centroids to global coordinates and calculate cluster sizes
        for i in range(n_clusters):
            cluster_points = occupied_coords[labels == i]
            centroid_local = np.mean(cluster_points, axis=0)
            max_distance = np.max(np.linalg.norm(cluster_points - centroid_local, axis=1))
            # Convert grid coordinates back to local coordinates (x, y)
            x_local, y_local = (centroid_local[0] * self.resolution) - (self.map_width / 2), (centroid_local[1] * self.resolution) - (self.map_height / 2)
            x_global, y_global = self.transform_to_global(x_local, y_local)
            cluster_details.append({'centroid': [x_global, y_global], 'size': max_distance * self.resolution})

        # Match and update known clusters with global centroids and sizes
        new_clusters = []
        for cluster in cluster_details:
            centroid_global, size = cluster['centroid'], cluster['size']
            matched = False
            for known_cluster in self.known_clusters:
                known_centroid_global, known_size = known_cluster['centroid'], known_cluster.get('size', 0)
                # Consider both position and size for matching
                distance = np.linalg.norm(np.array(centroid_global) - np.array(known_centroid_global))
                size_difference = abs(size - known_size)
                if distance < 0.1 and size_difference < (self.resolution * 5):  # Example thresholds
                    known_cluster['centroid'] = centroid_global
                    known_cluster['size'] = size  # Update size
                    new_clusters.append(known_cluster)
                    matched = True
                    break
            if not matched:
                new_cluster = {'id': self.cluster_id_counter, 'centroid': centroid_global, 'size': size}
                self.cluster_id_counter += 1
                new_clusters.append(new_cluster)

        # Update the known clusters
        self.known_clusters = new_clusters

        # Publish the detected clusters
        self.publish_clusters()

        # Print the detected clusters' id and global coordinates in the terminal
        for cluster in self.known_clusters:
            self.get_logger().info(f'Cluster {cluster["id"]} at {cluster["centroid"]}')
        

    # Publish the detected clusters
    def publish_clusters(self):
        msg = String()
        msg.data = json.dumps([{'id': cluster['id'], 'centroid': cluster['centroid']} for cluster in self.known_clusters])
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    perception = Perception()
    rclpy.spin(perception)
    perception.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
