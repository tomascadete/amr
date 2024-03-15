import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class Perception(Node):
    def __init__(self):
        super().__init__('perception')
        self.subscription = self.create_subscription(LaserScan, 'scan', self.laser_scan_callback, 10)
        self.occupancy_grid = None
        self.resolution = 0.1  # meters per pixel
        self.map_width = 20  # in meters
        self.map_height = 20  # in meters
        self.grid_width = int(self.map_width / self.resolution)
        self.grid_height = int(self.map_height / self.resolution)

    def laser_scan_callback(self, msg):
        angles = np.arange(msg.angle_min, msg.angle_max + msg.angle_increment, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # Filter out invalid ranges
        valid_ranges = np.where(np.logical_and(ranges > msg.range_min, ranges < msg.range_max))

        # Convert polar coordinates to Cartesian coordinates
        x = ranges[valid_ranges] * np.cos(angles[valid_ranges])
        y = ranges[valid_ranges] * np.sin(angles[valid_ranges])

        # Convert Cartesian coordinates to grid coordinates
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

        # Count the number of clusters found (excluding noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f'Number of clusters found: {n_clusters}')

        # Log information about each cluster
        for i in range(n_clusters):
            centroid = np.mean(occupied_coords[labels == i], axis=0)
            max_distance = np.max(np.linalg.norm(occupied_coords[labels == i] - centroid, axis=1))
            size = max_distance * 2
            self.get_logger().info(f'Cluster {i+1}: centroid=({centroid[0]:.2f}, {centroid[1]:.2f}), size={size:.2f} m')

        # Visualize the occupancy grid and the clusters in the same plot
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(self.occupancy_grid, cmap='gray', origin='lower')
        plt.title('Occupancy Grid')
        plt.subplot(122)
        plt.imshow(self.occupancy_grid, cmap='gray', origin='lower')
        for i in range(n_clusters):
            plt.plot(occupied_coords[labels == i][:, 1], occupied_coords[labels == i][:, 0], 'o')
        plt.title('Clusters')
        plt.show()
            


            

def main(args=None):
    rclpy.init(args=args)
    perception = Perception()
    rclpy.spin(perception)
    perception.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
