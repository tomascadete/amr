import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
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
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal = [20.0, 0.0]  # Goal position
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
            eta = 50.0    # Scaling factor for repulsive forces
            Q_max = 10.0     # Influence range of the obstacles, in meters

            # Convert goal and robot positions to numpy arrays for vectorized operations
            goal_pos = np.array(self.goal)
            robot_pos = np.array(self.robot_pose)

            # Compute the attractive force
            F_att = zeta * (goal_pos - robot_pos)

            # Initialize the repulsive force
            F_rep = np.zeros(2)
            

            # Compute the repulsive force from each obstacle
            # If the first obstacle has x and y set to np.inf, it means perception didn't detect anything
            # In this case, F_rep will be np.zeros(2) and the loop will be skipped
            for obstacle in self.obstacles.obstacles:
                if np.isinf([obstacle.x, obstacle.y]).any():
                    break

                # Compute the distance between the robot and the obstacle
                dist = np.linalg.norm(robot_pos - np.array([obstacle.x, obstacle.y]))

                # Compute the repulsive force
                if dist < Q_max:
                    F_rep += (eta * (1.0 / dist - 1.0 / Q_max) * (1.0 / dist**2) *
                              ((robot_pos - np.array([obstacle.x, obstacle.y])) / dist))

            # Compute the total force
            F_total = F_att + F_rep

            # Log the total force vector
            self.get_logger().info(f'Total force vector: {F_total}')

            linear_speed_limit = 0.5 # Limit the linear speed to 0.5 m/s
            angular_speed_limit = 0.5 # Limit the angular speed to 0.5 rad/s

            # Convert F_total to linear and angular velocities
            linear_speed = np.linalg.norm(F_total)
            angular_speed = np.arctan2(F_total[1], F_total[0])

            # Scale velocity to ensure they are within the limits
            linear_velocity = min(linear_speed, linear_speed_limit)
            
            # Create and publish the Twist message
            twist = Twist()
            twist.linear.x = linear_velocity
            twist.angular.z = angular_speed
            self.cmd_vel_publisher.publish(twist)

            # Log the linear and angular velocities
            self.get_logger().info(f'Publishing velocities: linear={twist.linear.x}, angular={twist.angular.z}')

                


def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()