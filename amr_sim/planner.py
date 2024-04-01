import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from amr_interfaces.msg import Object, ObstacleArray
import numpy as np

class Planner(Node):
    def __init__(self):
        super().__init__('planner')
        self.subscription = self.create_subscription(
            ObstacleArray,
            '/tracked_objects',
            self.tracked_objects_callback,
            10)
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.goal = [10.0, 0.0]  # Goal position
        self.robot_pose = [0.0, 0.0]
        self.obstacles = None

    def odom_callback(self, msg):
        self.robot_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]


    def tracked_objects_callback(self, msg):
        self.obstacles = msg
        self.plan_path()

    # Here we implement Artificial Potential Fields to plan a path
    def plan_path(self):
        if self.obstacles is not None:
            # Constants for attractive and repulsive forces
            zeta = 1.0      # Scaling factor for the attractive force
            eta = 100.0     # Scaling factor for repulsive forces
            Q_max = 10.0     # Influence range of the obstacles, in meters

            # Convert goal and robot positions to numpy arrays for vectorized operations
            goal_pos = np.array(self.goal)
            robot_pos = np.array(self.robot_pose)

            # Compute the attractive force
            F_att = zeta * (goal_pos - robot_pos)

            # Initialize the repulsive force
            F_rep = np.zeros(2)

            # Compute the repulsive force from each obstacle
            for obstacle in self.obstacles.obstacles:
                obstacle_pos = np.array([obstacle.x, obstacle.y])
                d = np.linalg.norm(robot_pos - obstacle_pos)  # Distance between robot and obstacle

                if d < Q_max:
                    F_rep += eta * ((1.0 / d) - (1.0 / Q_max)) * (1.0 / d**2) * (robot_pos - obstacle_pos)

            # Compute the total force
            F_total = F_att + F_rep

            # Log the total force vector
            self.get_logger().info(f'Total force vector: {F_total}')

                


def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()