import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from amr_interfaces.msg import ObstacleArray
import numpy as np
import matplotlib.pyplot as plt

class Planner(Node):
    def __init__(self):
        super().__init__('planner')
        self.create_subscription(ObstacleArray, '/tracked_objects', self.tracked_objects_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal = [10.0, 0.0]
        self.robot_pose = [0.0, 0.0]
        self.obstacles = []
        self.grid_resolution = 0.1
        self.grid_size = 500  # 50 / 0.1 = 500 cells on each side
        self.grid = np.zeros((self.grid_size, self.grid_size))

    def odom_callback(self, msg):
        self.robot_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]

    def tracked_objects_callback(self, msg):
        self.obstacles = []
        for obstacle in msg.obstacles:
            self.obstacles.append([obstacle.x, obstacle.y])
        self.generate_path()

    def generate_path(self):
        # Reset grid
        self.grid.fill(0)
        
        # Update grid for obstacles
        for obstacle in self.obstacles:
            ox, oy = int(obstacle[0] / self.grid_resolution + self.grid_size / 2), int(obstacle[1] / self.grid_resolution + self.grid_size / 2)
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                self.grid[ox-10:ox+11, oy-10:oy+11] = 1  # Circle radius on grid

        # TODO: Implement A* algorithm to find path from robot_pose to goal while avoiding obstacles




def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
