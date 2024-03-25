import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch


# Load the YOLOv8s PyTorch model (file called yolo5s.pt)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/kinect_camera/image_raw',
            self.listener_callback,
            10)
        self.subscription = self.create_subscription(
            PointCloud2,
            '/kinect_camera/points',
            self.points_callback,
            10)
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.robot_pose = None
        self.point_cloud = None


    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def points_callback(self, msg):
        self.point_cloud = msg


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

                # Extract the 3D coordinates of the object from the point cloud data
                result = self.get_3d_coordinates(centroid_x, centroid_y)
                if result is not None:
                    x, y, z = result
                    self.get_logger().info(f'Object at ({x}, {y}, {z})')
                else:
                    self.get_logger().warn("Could not get 3D coordinates for the object.")


            # Log information about detected objects
            self.get_logger().info(f'Detected {len(results.xyxy[0])} objects')

            # Display the image in a window
            cv2.imshow("Kinect Camera Image", cv_image)
            cv2.waitKey(1)  # Refresh the display window every 1 millisecond


        except CvBridgeError as e:
            self.get_logger().error('Could not convert from ROS Image message to OpenCV Image: %s' % str(e))


    
    def get_3d_coordinates(self, centroid_x, centroid_y):
        # Assuming the point cloud is stored in self.point_cloud
        if self.point_cloud is None:
            self.get_logger().warn("No point cloud data available.")
            return None
        
        # Convert the PointCloud2 message to a list of points
        cloud_points = list(point_cloud2.read_points(self.point_cloud, field_names=("x", "y", "z"), skip_nans=True))

        # Calculate the index in the point cloud array
        width = self.point_cloud.width
        height = self.point_cloud.height
        # index = int(centroid_y * width + centroid_x)
        index = max(0, min(len(cloud_points) - 1, int(centroid_y * width + centroid_x)))


        # Ensure the index is within the bounds of the point cloud array
        if index >= len(cloud_points) or index < 0:
            self.get_logger().warn("Computed index out of bounds.")
            return None

        # Extract the 3D coordinates
        x, y, z = cloud_points[index]
        return x, y, z

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
