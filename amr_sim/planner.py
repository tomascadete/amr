import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from amr_interfaces.msg import Object, ObstacleArray
import numpy as np
import heapq

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
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_publisher = self.create_publisher(Path, '/path', 10)
        self.goal = [20.0, 20.0]  # Goal position
        self.robot_pose = [0.0, 0.0]
        self.obstacles = None

    def odom_callback(self, msg):
        self.robot_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]


    def tracked_objects_callback(self, msg):
        self.obstacles = [(obstacle.x, obstacle.y) for obstacle in msg.obstacles]
        self.generate_path()


    def generate_path(self):
        # If there are no obstacles, we can move directly to the goal
        if self.obstacles is None or len(self.obstacles) == 0:
            path = Path()
            pose = PoseStamped()
            pose.pose.position.x = self.goal[0]
            pose.pose.position.y = self.goal[1]
            path.poses.append(pose)
            self.path_publisher.publish(path)
            self.get_logger().info('No obstacles detected, moving directly to the goal.')
            return
        
        # Constants for potential field calculation
        attractive_scale = 1.0
        repulsive_scale = 100.0
        repulsive_threshold = 5.0

        def attractive_potential(x, y):
            return 0.5 * attractive_scale * np.linalg.norm(np.array([x, y]) - np.array(self.goal))**2
        
        def repulsive_potential(x, y):
            potential = 0.0
            for obstacle in self.obstacles:
                distance = np.linalg.norm(np.array([x, y]) - np.array(obstacle))
                if distance < repulsive_threshold:
                    potential += 0.5 * repulsive_scale * (1.0 / distance - 1.0 / repulsive_threshold)**2
            return potential
            
        def total_potential(x, y):
            return attractive_potential(x, y) + repulsive_potential(x, y)
        
        # Generate path
        path = Path()
        current_position = np.array(self.robot_pose)
        step_size = 0.1
        max_steps = 1000

        for _ in range(max_steps):
            # Calculate gradient of potential field
            grad_x = (total_potential(current_position[0] + step_size, current_position[1]) - total_potential(current_position[0] - step_size, current_position[1])) / (2 * step_size)
            grad_y = (total_potential(current_position[0], current_position[1] + step_size) - total_potential(current_position[0], current_position[1] - step_size)) / (2 * step_size)
            grad = np.array([grad_x, grad_y])

            # Move in the direction of negative gradient
            next_position = current_position - step_size * grad
            if np.linalg.norm(grad) < 1e-5:
                break  # Stop if gradient is very small

            # Add the new position to the path
            pose = PoseStamped()
            pose.pose.position.x = next_position[0]
            pose.pose.position.y = next_position[1]
            path.poses.append(pose)

            current_position = next_position

            # Stop if goal is reached
            if np.linalg.norm(current_position - np.array(self.goal)) < step_size:
                break

        self.path_publisher.publish(path)
        self.get_logger().info('Path generated avoiding obstacles.')
          


def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()