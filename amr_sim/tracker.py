import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
import transforms3d

class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        self.lidar_subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)
        self.robot_odom = None
        self.initialize_occupancy_grid()

    def odom_callback(self, msg):
        self.robot_odom = msg

    def lidar_callback(self, msg):
        if self.robot_odom is None:
            self.get_logger().info('No odometry data available.')
            return

        # Get robot's current position and orientation
        robot_x = self.robot_odom.pose.pose.position.x
        robot_y = self.robot_odom.pose.pose.position.y
        orientation_q = self.robot_odom.pose.pose.orientation
        _, _, yaw = transforms3d.euler.quat2euler([orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z], axes='sxyz')

        self.get_logger().info(f'Robot position: ({robot_x}, {robot_y})')
        self.get_logger().info(f'Robot orientation: {yaw} radians')

        # Compute global coordinates for each LIDAR point
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        adjusted_angles = angles + yaw

        valid_indices = np.where((ranges > msg.range_min) & (ranges < msg.range_max))
        valid_ranges = ranges[valid_indices]
        valid_angles = adjusted_angles[valid_indices]

        global_x = valid_ranges * np.cos(valid_angles) + robot_x
        global_y = valid_ranges * np.sin(valid_angles) + robot_y

        # Clear the occupancy grid
        self.occupancy_grid.data = [0] * (self.occupancy_grid.info.width * self.occupancy_grid.info.height)

        # Update the occupancy grid
        for x, y in zip(global_x, global_y):
            grid_x = int((x - self.occupancy_grid.info.origin.position.x) / self.occupancy_grid.info.resolution)
            grid_y = int((y - self.occupancy_grid.info.origin.position.y) / self.occupancy_grid.info.resolution)
            if 0 <= grid_x < self.occupancy_grid.info.width and 0 <= grid_y < self.occupancy_grid.info.height:
                self.occupancy_grid.data[grid_y * self.occupancy_grid.info.width + grid_x] = 100

        self.occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.occupancy_grid)

    def initialize_occupancy_grid(self):
        self.occupancy_grid = OccupancyGrid()
        self.occupancy_grid.header.frame_id = 'map'
        self.occupancy_grid.info.resolution = 0.5  # meters per grid cell
        self.occupancy_grid.info.width = 200  # cells
        self.occupancy_grid.info.height = 200  # cells
        self.occupancy_grid.info.origin.position.x = -50.0  # meters
        self.occupancy_grid.info.origin.position.y = -50.0  # meters
        self.occupancy_grid.data = [0] * (self.occupancy_grid.info.width * self.occupancy_grid.info.height)

def main(args=None):
    rclpy.init(args=args)
    tracker = ObjectTracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
