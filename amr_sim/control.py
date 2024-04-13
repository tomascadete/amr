import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
import numpy as np
import math
import transforms3d

class Control(Node):
    def __init__(self):
        super().__init__('control_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Path, '/path', self.path_update, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.current_path = []
        self.current_index = 0
        self.odometry = None
        self.timer = self.create_timer(0.1, self.publish_twist)

    def odom_callback(self, msg):
        self.odometry = msg

    def path_update(self, msg):
        self.current_path = msg.poses
        self.current_index = 0
        self.get_logger().info(f'Path updated with {len(self.current_path)} waypoints')

    def publish_twist(self):
        if self.current_index < len(self.current_path) and self.odometry:
            current_odom_pose = self.odometry.pose.pose
            next_pose = self.current_path[self.current_index].pose

            x, y, theta = current_odom_pose.position.x, current_odom_pose.position.y, self.get_yaw_from_quaternion(current_odom_pose.orientation)
            x_next, y_next = next_pose.position.x, next_pose.position.y

            dx = x_next - x
            dy = y_next - y

            # Target angle in global frame
            target_angle = math.atan2(dy, dx)
            # Current orientation error
            orientation_error = self.normalize_angle(target_angle - theta)

            omega = 0.1 * orientation_error  # Proportional gain
            v = 0.1 * np.sqrt(dx**2 + dy**2) if abs(orientation_error) < np.pi / 4 else 0.0  # Move only when facing target

            twist = Twist()
            twist.linear.x = v
            twist.angular.z = omega
            self.publisher_.publish(twist)

            if np.sqrt(dx**2 + dy**2) < 0.5:
                self.current_index += 1
                self.get_logger().info(f'Reached waypoint {self.current_index}')
        else:
            self.get_logger().info('No waypoints to follow or odometry data missing')

    def get_yaw_from_quaternion(self, quaternion):
        q = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        euler = transforms3d.euler.quat2euler(q, axes='sxyz')
        return euler[2]

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    control_node = Control()
    rclpy.spin(control_node)
    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
