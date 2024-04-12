import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
import numpy as np

class Controller(Node):
    def __init__(self):
        super().__init__('controller')
        self.path_subscription = self.create_subscription(
            Path,
            '/path',
            self.path_callback,
            10)
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.current_path = None
        self.current_target_index = 0
        self.robot_pose = None

    def path_callback(self, msg):
        # self.get_logger().info('Received new path')
        self.current_path = msg
        self.current_target_index = 2  # Reset target index whenever a new path is received

    def odom_callback(self, msg):
        self.robot_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]

    def control_loop(self):
        if self.current_path is None:
            return
        
        if self.robot_pose is None:
            return

        if self.current_target_index >= len(self.current_path.poses):
            return
    

        target_pose = self.current_path.poses[self.current_target_index].pose
        target_position = [target_pose.position.x, target_pose.position.y]

        # Calculate the distance to the target
        distance = np.linalg.norm(np.array(target_position) - np.array(self.robot_pose))

        if distance < 0.1:
            self.current_target_index += 1
            if self.current_target_index >= len(self.current_path.poses):
                # self.get_logger().info('Reached the goal')
                return
            target_pose = self.current_path.poses[self.current_target_index].pose
            target_position = [target_pose.position.x, target_pose.position.y]

        # Calculate the angle to the target
        angle = np.arctan2(target_position[1] - self.robot_pose[1], target_position[0] - self.robot_pose[0])

        # Define the linear and angular velocities
        linear_velocity = 0.5
        angular_velocity = 1.0 * (angle - np.arctan2(self.robot_pose[1], self.robot_pose[0]))


        # Publish the velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_velocity
        cmd_vel.angular.z = angular_velocity
        self.cmd_vel_publisher.publish(cmd_vel)
        # self.get_logger().info(f'Linear Velocity: {linear_velocity}, Angular Velocity: {angular_velocity}')


def main(args=None):
    rclpy.init(args=args)
    controller = Controller()

    try:
        while rclpy.ok():
            rclpy.spin_once(controller, timeout_sec=0.1)
            controller.control_loop()
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
