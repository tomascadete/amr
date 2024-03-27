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
        
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.robot_pose = None
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

            # Perform object detection using the YOLOv5s model
            results = model(cv_image)

            # For each detected object, go to the point cloud data and extract the 3D coordinates
            for obj in results.xyxy[0]:
                x1, y1, x2, y2 = obj[:4]
                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                # Log the centroid coordinates
                # self.get_logger().info(f'Centroid at ({centroid_x}, {centroid_y})')

                # Extract the 3D coordinates of the object from the depth image
                depth = self.depth_map[int(centroid_y), int(centroid_x)]
                # Log the depth value
                # self.get_logger().info(f'Depth value: {depth}')

                # Using the depth value and the camera intrinsics, calculate the 3D coordinates of the detected object
                x_local = (centroid_x - cx) * depth / fx
                y_local = (centroid_y - cy) * depth / fy
                z_local = depth
                # TODO: The above calculation is incorrect. X and Z seem to be swapped.

                # Log the local coordinates
                self.get_logger().info(f'Local coordinates of the object: ({x_local}, {y_local}, {z_local})')

                # Robot's current position
                px, py, pz = self.robot_pose.position.x, self.robot_pose.position.y, self.robot_pose.position.z
                # Robot's current orientation in quaternion format
                quaternion = (
                    self.robot_pose.orientation.x,
                    self.robot_pose.orientation.y,
                    self.robot_pose.orientation.z,
                    self.robot_pose.orientation.w
                )

                # Convert quaternion to rotation matrix
                R = t3d.quaternions.quat2mat(quaternion)

                # Apply the rotation matrix to the local coordinates
                global_coords_rotated = np.dot(R, np.array([x_local, y_local, z_local]))

                # Apply the translation to the rotated coordinates
                global_coords = global_coords_rotated + np.array([px, py, pz])

                # Log the global coordinates
                self.get_logger().info(f'Global coordinates of the object: {global_coords}')

                # TODO: Publish a custom message with the global coordinates of the detected object along with the class name.


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
            self.get_logger().info(f'Detected {len(results.xyxy[0])} objects')

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
