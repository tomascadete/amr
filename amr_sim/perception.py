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




# Load the YOLOv8s PyTorch model (file called yolo5s.pt)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

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

            # Perform object detection using the YOLOv5s model
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
            # Log the robot's position
            # self.get_logger().info(f'Robot position: ({x_robot}, {y_robot}, {z_robot})')
            # Log yaw pitch and roll
            # self.get_logger().info(f'Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}')
            



            # For each detected object, calculate its global coordinates
            for *xyxy, conf, cls in results.xyxy[0]:
                # Extract bb coordinates
                x1, y1, x2, y2 = map(int, xyxy)
                # Get class name
                class_name = results.names[int(cls)]
                # Calculate the center of the bounding box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # Get the distance to the object at the center of the bounding box
                depth = self.depth_map[int(center_y), int(center_x)]

                # Get the local coordinates of the object in the camera frame
                x_camera = (center_x - cx) * depth / fx
                y_camera = (center_y - cy) * depth / fy
                z_camera = depth

                # Get the local coordinates of the object in the robot frame
                x_robot_base = z_camera
                y_robot_base = -x_camera
                z_robot_base = -y_camera

                # Log the local coordinates of the object in the robot frame
                # self.get_logger().info(f'Local coordinates of {class_name} in the robot frame: ({x_robot_base}, {y_robot_base}, {z_robot_base})')

                # Transform the local coordinates of the object in the robot frame to the world frame
                x_world = x_robot + np.cos(yaw) * x_robot_base - np.sin(yaw) * y_robot_base
                y_world = y_robot + np.sin(yaw) * x_robot_base + np.cos(yaw) * y_robot_base
                z_world = z_robot + z_robot_base

                # Log the global coordinates of the object
                #self.get_logger().info(f'Global coordinates of {class_name}: ({x_world}, {y_world}, {z_world})')

                # Create an Object message
                object_msg = Object()
                object_msg.type = class_name
                object_msg.x = x_world
                object_msg.y = y_world
                object_msg.z = z_world

                # Publish the Object message to the /detections topic
                self.publisher.publish(object_msg)

            # If no objects are detected, publish an Object messange with Inf values
            if len(results.xyxy[0]) == 0:
                object_msg = Object()
                object_msg.type = 'None'
                object_msg.x = np.inf
                object_msg.y = np.inf
                object_msg.z = np.inf
                self.publisher.publish(object_msg)
                


            # Draw bounding boxes and labels on the image
            for *xyxy, conf, cls in results.xyxy[0]:
                # Extract bb coordinates
                x1, y1, x2, y2 = map(int, xyxy)
                # Draw rectangle
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Get class name
                class_name = results.names[int(cls)]
                # Draw label
                cv2.putText(cv_image, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Log information about detected objects
            # self.get_logger().info(f'Detected {len(results.xyxy[0])} objects')

            # Display the image
            cv2.imshow("Kinect Camera Image", cv_image)
            cv2.waitKey(1)  # Refresh the display window every 1 millisecond


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
