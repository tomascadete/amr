import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
from ultralytics import YOLO
from amr_interfaces.msg import LightColour

model = YOLO("traffic_light.pt")

class TrafficLightDetection(Node):
    def __init__(self):
        super().__init__('traffic_light_detection')
        self.image_subscription = self.create_subscription(
            Image,
            '/kinect_camera/image_raw',
            self.listener_callback,
            10)
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        # Create a publisher to publish red green or yellow detected traffic light
        self.publisher = self.create_publisher(LightColour, '/light_status', 10)
        self.light_status = None
        self.bridge = CvBridge()
        self.robot_pose = None
        self.crossing = False
        self.crosswalk_entry = np.array([12.0, 37.0])
        self.crosswalk_exit = np.array([29.0, 37.0])


    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose
        # If the robot is within 1m of the crosswalk entry, set crossing to True
        if self.robot_pose and np.linalg.norm([self.robot_pose.position.x - self.crosswalk_entry[0], self.robot_pose.position.y - self.crosswalk_entry[1]]) < 1:
            self.crossing = True
        # If the robot is within 1m of the crosswalk exit, set crossing to False
        if self.robot_pose and np.linalg.norm([self.robot_pose.position.x - self.crosswalk_exit[0], self.robot_pose.position.y - self.crosswalk_exit[1]]) < 1:
            self.crossing = False


    def listener_callback(self, msg):
        if not self.crossing:
            return
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            results = model(cv_image)

            class_names = model.names

            if results:
                for detection in results:
                    if detection.boxes.xyxy.shape[0] > 0:  # Check if there are bounding boxes
                        cls = detection.boxes.cls[0].numpy()
                        class_name = class_names[int(cls)]
                        if class_name == 'Traffic_Blue':
                            self.light_status = 'green'
                            self.get_logger().info('Traffic light is green')
                        elif class_name == 'Traffic Red':
                            self.light_status = 'red'
                            self.get_logger().info('Traffic light is red')
                        elif class_name == 'Traffic_Yellow':
                            self.light_status = 'yellow'
                            self.get_logger().info('Traffic light is yellow')

            # Publish the detected traffic light status only if there is a change in status
            if self.light_status:
                msg = LightColour()
                msg.colour = self.light_status
                self.publisher.publish(msg)
                self.light_status = None


        except CvBridgeError as e:
            self.get_logger().error('Could not convert from ROS Image message to OpenCV Image: %s' % str(e))


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetection()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()

