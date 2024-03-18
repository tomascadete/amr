import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from sklearn.cluster import DBSCAN
from std_msgs.msg import String
import json

class Perception(Node):
    def __init__(self):
        super().__init__('perception')
        self.subscription = self.create_subscription(LaserScan, 'scan', self.laser_scan_callback, 10)
        self.publisher = self.create_publisher(String, '/detected_objects_with_id', 10)
        self.occupancy_grid = None
        self.resolution = 0.1  # meters per pixel
        self.map_width = 20  # meters
        self.map_height = 20  # meters
        self.grid_width = int(self.map_width / self.resolution)
        self.grid_height = int(self.map_height / self.resolution)
        self.known_clusters = []
        self.cluster_id_counter = 0

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
        clustering = DBSCAN(eps=8, min_samples=3).fit(occupied_coords)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Update the known clusters
        new_clusters = []
        for i in range(n_clusters):
            centroid = np.mean(occupied_coords[labels == i], axis=0)
            size = np.max(np.linalg.norm(occupied_coords[labels == i] - centroid, axis=1)) * 2
            matched = False
            for known_cluster in self.known_clusters:
                if np.linalg.norm(centroid - known_cluster['centroid']) < size / 2:
                    known_cluster['centroid'] = centroid
                    new_clusters.append(known_cluster)
                    matched = True
                    break
            if not matched:
                new_cluster = {'id': self.cluster_id_counter, 'centroid': centroid}
                self.cluster_id_counter += 1
                new_clusters.append(new_cluster)

        self.known_clusters = new_clusters
        self.publish_clusters()

    # Publish the detected clusters
    def publish_clusters(self):
        msg = String()
        msg.data = json.dumps([{'id': cluster['id'], 'centroid': cluster['centroid'].tolist()} for cluster in self.known_clusters])
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    perception = Perception()
    rclpy.spin(perception)
    perception.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
