import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
import numpy as np
import transforms3d as t3d
from amr_interfaces.msg import Emergency


class Controller(Node):
    def __init__(self):
        super().__init__('controller')
        self.subscription_path = self.create_subscription(Path, '/path', self.path_callback, 10)
        self.subscription_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.subscription_emergency = self.create_subscription(Emergency, '/emergency', self.emergency_callback, 10)
        self.publisher_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.current_path = []
        self.current_pose = None
        self.align_to_zero = False
        self.waypoint_threshold = 0.5  # meters
        self.angular_threshold = np.pi / 2  # radians
        self.linear_speed = 3.0  # Maximum linear speed
        self.angular_speed = np.pi * 2  # Maximum angular speed
        self.emergency_state = False

    # Emergency state is when the robot is in a position from where it must escape as fast as possible
    def emergency_callback(self, msg):
        if msg.emergency_state == 1:
            self.emergency_state = True
            self.linear_speed = 5.0
            self.angular_threshold = np.pi
        elif msg.emergency_state == 0:
            self.emergency_state = False
            self.linear_speed = 3.0
            self.angular_threshold = np.pi / 2



    def path_callback(self, msg):
        incoming_path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        current_path = [(pose.pose.position.x, pose.pose.position.y) for pose in self.current_path]
        if not incoming_path or incoming_path == current_path[:len(incoming_path)]:
            return
        self.current_path = msg.poses
        # self.get_logger().info('New path received')

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        if self.align_to_zero:
            self.align_robot_to_zero()
        else:
            self.navigate()


    def navigate(self):
        if not self.current_pose or not self.current_path:
            return

        current_waypoint = self.current_path[0].pose.position
        dx = current_waypoint.x - self.current_pose.position.x
        dy = current_waypoint.y - self.current_pose.position.y
        distance = np.sqrt(dx**2 + dy**2)

        _, _, yaw = t3d.euler.quat2euler([self.current_pose.orientation.w,
                                          self.current_pose.orientation.x,
                                          self.current_pose.orientation.y,
                                          self.current_pose.orientation.z], axes='sxyz')
        angle_to_target = np.arctan2(dy, dx)
        angle_diff = (angle_to_target - yaw + np.pi) % (2 * np.pi) - np.pi

        if distance < self.waypoint_threshold:
            self.get_logger().info(f'Waypoint reached: ({current_waypoint.x}, {current_waypoint.y})')
            self.current_path.pop(0)
            if not self.current_path:
                self.align_to_zero = True
                self.align_robot_to_zero()
                return

        twist = Twist()
        angle_error = abs(angle_diff)
        if angle_error < self.angular_threshold:
            scale = 1 - (angle_error / self.angular_threshold)  # Scale speed based on angle error
            twist.linear.x = self.linear_speed * scale
        twist.angular.z = np.clip(angle_diff, -self.angular_speed, self.angular_speed)

        self.publisher_vel.publish(twist)
        # self.get_logger().info(f'Linear speed: {twist.linear.x}, Angular speed: {twist.angular.z}')

    def align_robot_to_zero(self):
        _, _, yaw = t3d.euler.quat2euler([self.current_pose.orientation.w,
                                          self.current_pose.orientation.x,
                                          self.current_pose.orientation.y,
                                          self.current_pose.orientation.z], axes='sxyz')
        
        # Target orientation is 0 radians
        angle_diff = -yaw
        if abs(angle_diff) <= 0.1:
            self.align_to_zero = False
            self.stop_robot()
            return
        
        twist = Twist()
        twist.angular.z = np.clip(angle_diff, -self.angular_speed, self.angular_speed)
        self.publisher_vel.publish(twist)

    def stop_robot(self):
        self.publisher_vel.publish(Twist())
        self.get_logger().info('Destination reached, stopping robot.')

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
