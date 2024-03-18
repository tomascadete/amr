import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import json
from math import pi
import time

class RobotControl(Node):
    def __init__(self):
        super().__init__('robot_control')
        # Publisher for the robot's velocity command
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Publisher for the robot's state
        self.state_publisher = self.create_publisher(String, '/robot_state', 10)

        # Use a timer to periodically execute the publish_state_and_move function
        self.timer = self.create_timer(1, self.publish_state_and_move)

        # Initialize the robot's speed and orientation
        self.speed = 0.0  # m/s
        self.orientation = 0.0  # radians

    def publish_state_and_move(self):
        # Publish the robot's state
        state_msg = String()
        state_msg.data = json.dumps({'speed': self.speed, 'orientation': self.orientation})
        self.state_publisher.publish(state_msg)
        self.get_logger().info(f'Publishing robot state: {state_msg.data}')

        # Move the robot slowly forward
        vel_msg = Twist()
        vel_msg.linear.x = self.speed
        vel_msg.angular.z = 0.0
        self.velocity_publisher.publish(vel_msg)
        self.get_logger().info('Moving the robot')

def main(args=None):
    rclpy.init(args=args)
    robot_control = RobotControl()
    rclpy.spin(robot_control)
    robot_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
