import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid
import matplotlib.pyplot as plt
import numpy as np

class Planner(Node):
    def __init__(self):
        super().__init__('planner')
        self.create_subscription(OccupancyGrid, '/occupancy_grid', self.occupancy_grid_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher_ = self.create_publisher(Path, '/path', 10)
        self.goal = [20.0, 0.0]
        self.robot_pose = [0.0, 0.0]
        self.obstacles = []
        self.occ_grid = None
        self.init_plot()

    def init_plot(self):
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots()
        self.im = None

    def odom_callback(self, msg):
        self.robot_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]

    def occupancy_grid_callback(self, msg):
        # Convert the OccupancyGrid data to a numpy array and flip it along the x-axis
        data = np.flipud(np.array(msg.data).reshape((msg.info.height, msg.info.width)))
        if self.im is None:
            # Set the extent to map cell indices to world coordinates
            extent = [-msg.info.width * msg.info.resolution / 2,
                      msg.info.width * msg.info.resolution / 2,
                      -msg.info.height * msg.info.resolution / 2,
                      msg.info.height * msg.info.resolution / 2]
            self.im = self.ax.imshow(data, cmap='gray', vmin=-1, vmax=100, extent=extent)
            self.ax.set_title("Occupancy Grid")
            self.ax.set_xlabel('X (meters)')
            self.ax.set_ylabel('Y (meters)')
        else:
            self.im.set_data(data)
        plt.pause(0.05)  # Short pause to update the plot

def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
