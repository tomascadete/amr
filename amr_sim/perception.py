import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
import transforms3d as t3d
from amr_interfaces.msg import Object
from ultralytics import YOLO





# Load the YOLOv8m PyTorch model
model = YOLO("yolov8m.pt")


# Camera intrinsics
fx = 528.433756558705  # Focal length in x
fy = 528.433756558705  # Focal length in y
cx = 320.5             # Optical center x
cy = 240.5             # Optical center y



class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/kinect_camera/image_raw',
            self.listener_callback,
            10)
        self.subscription = self.create_subscription(
            Image,
            '/kinect_camera/depth/image_raw',
            self.depth_callback,
            10)
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        self.publisher = self.create_publisher(Object, '/detections', 10)

        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        default_pose = Odometry()
        default_pose.pose.pose.position.x = 0.0
        default_pose.pose.pose.position.y = 0.0
        default_pose.pose.pose.position.z = 0.0
        default_pose.pose.pose.orientation.x = 0.0
        default_pose.pose.pose.orientation.y = 0.0
        default_pose.pose.pose.orientation.z = 0.0
        default_pose.pose.pose.orientation.w = 1.0
        self.robot_pose = default_pose.pose.pose
        self.depth_map = None


    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def depth_callback(self, msg):
        try:
            self.depth_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        except CvBridgeError as e:
            self.get_logger().error('Could not convert from ROS Image message to OpenCV Image: %s' % str(e))


    def listener_callback(self, msg):
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Check if the depth map has been received
            if self.depth_map is None:
                return

            # Perform object detection using the YOLOv8 model
            results = model(cv_image)

            # Get the robot's position and orientation
            x_robot = self.robot_pose.position.x
            y_robot = self.robot_pose.position.y
            z_robot = self.robot_pose.position.z
            qx = self.robot_pose.orientation.x
            qy = self.robot_pose.orientation.y
            qz = self.robot_pose.orientation.z
            qw = self.robot_pose.orientation.w
            # Convert the quaternion to Euler angles 
            yaw, pitch, roll = t3d.euler.quat2euler([qx, qy, qz, qw], axes='sxyz')

            class_names = model.names

            # Check if results are not empty
            if results:
                # Process each detected object
                for detection in results:
                    if detection.boxes.xyxy.shape[0] > 0:  # Check if there are bounding boxes
                        xyxy = detection.boxes.xyxy[0].numpy()
                        conf = detection.boxes.conf[0].numpy()
                        cls = detection.boxes.cls[0].numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        class_name = class_names[int(cls)]

                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        depth = self.depth_map[int(center_y), int(center_x)]

                        x_camera = (center_x - cx) * depth / fx
                        y_camera = (center_y - cy) * depth / fy
                        z_camera = depth

                        x_robot_base = z_camera
                        y_robot_base = -x_camera
                        z_robot_base = -y_camera

                        x_world = x_robot + np.cos(yaw) * x_robot_base - np.sin(yaw) * y_robot_base
                        y_world = y_robot + np.sin(yaw) * x_robot_base + np.cos(yaw) * y_robot_base
                        z_world = z_robot + z_robot_base

                        # If the object is higher than 1.5 meters, ignore it
                        if z_world > 1.5:
                            continue

                        object_msg = Object()
                        object_msg.type = class_name
                        object_msg.x = x_world
                        object_msg.y = y_world
                        object_msg.z = z_world
                        self.publisher.publish(object_msg)

            else:
                # If no objects are detected, publish an Object message with type 'None'
                object_msg = Object()
                object_msg.type = 'None'
                object_msg.x = 0.0
                object_msg.y = 0.0
                object_msg.z = 0.0
                self.publisher.publish(object_msg)

            # # Draw bounding boxes and class names on the image
            # for detection in results:
            #     if detection.boxes.xyxy.shape[0] > 0:
            #         xyxy = detection.boxes.xyxy[0].numpy()
            #         x1, y1, x2, y2 = map(int, xyxy)
            #         class_name = class_names[int(detection.boxes.cls[0])]
            #         conf = detection.boxes.conf[0]
            #         cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #         cv2.putText(cv_image, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # cv2.imshow("Kinect Camera Image", cv_image)
            # cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error('Could not convert from ROS Image message to OpenCV Image: %s' % str(e))




def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    # When your node is done executing, make sure to destroy it
    image_subscriber.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()  # Close the display window

if __name__ == '__main__':
    main()
