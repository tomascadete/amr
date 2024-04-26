import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import Twist, PoseStamped
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class TrackedObject:
    def __init__(self, id):
        self.id = id
        self.positions = []
        self.velocity = 0.0
        self.acceleration = 0.0
        self.steps_since_seen = 0
        self.distances_to_robot = []

class CrossingNode(Node):
    def __init__(self):
        super().__init__('crossing_node')
        self.bridge = CvBridge()
        # self.ryg_model = YOLO("semaphorelight.pt")
        self.subscription_image = self.create_subscription(Image, '/camera/image_raw', self.greenlight_callback, 10)
        self.subscription_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.subscription_occ_grid = self.create_subscription(OccupancyGrid, '/occupancy_grid', self.occupancy_grid_callback, 10)
        self.publisher = self.create_publisher(Path, '/path', 10)
        self.robot_pose = None
        self.crosswalk_entry = np.array([12.0, 37.0])
        self.crosswalk_exit = np.array([29.0, 37.0])
        self.grid = None
        self.grid_origin = None
        self.crossing = False
        self.tracked_objects = []
        self.tracked_objects_id = 1
        self.last_time = self.get_clock().now()
        self.times_ran = 0
        self.safe_to_cross = False

        # Initialize the plot for the road part of the occupancy grid
        # plt.ion()
        # self.fig, self.ax = plt.subplots()
        # self.ax.set_title("Occupancy Grid of Road Area")
        # plt.show()



    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose
        self.check_crosswalk_proximity()


    def check_crosswalk_proximity(self):
        if self.robot_pose is None:
            return
        robot_x = self.robot_pose.position.x
        robot_y = self.robot_pose.position.y
        distance_to_entry = np.linalg.norm(np.array([robot_x, robot_y]) - self.crosswalk_entry)
        distance_to_exit = np.linalg.norm(np.array([robot_x, robot_y]) - self.crosswalk_exit)
        if distance_to_entry < 1.0:
            self.crossing = True
            self.get_logger().info('Robot is at the crosswalk entry')
        elif distance_to_exit < 1.0:
            self.crossing = False
            self.get_logger().info('Robot is at the crosswalk exit')



    def occupancy_grid_callback(self, msg):
        if not self.crossing:
            return
        
        # In this grid, the first index is the y coordinate and the second index is the x coordinate
        self.height = msg.info.height # 200 (cells)
        self.width = msg.info.width # 200 (cells)
        self.resolution = msg.info.resolution # 0.5 (m)
        self.grid_origin_x = msg.info.origin.position.x # -50 (m)
        self.grid_origin_y = msg.info.origin.position.y # -50 (m)
        self.grid = np.array(msg.data).reshape((self.height, self.width))
        # The point (0,0) in world coordinates is at the center of the grid (y=100, x=100) grid coordinates

        # Extract the road area from the occupancy grid
        x_min_grid = int((13 - self.grid_origin_x) / self.resolution)
        x_max_grid = int((28 - self.grid_origin_x) / self.resolution)
        road_area = self.grid[:, x_min_grid:x_max_grid]
        occupied = road_area > 0


        # Update the plot with the road area of the occupancy grid
        # self.ax.clear()
        # self.ax.imshow(road_area.T, cmap='gray', origin='lower')
        # self.ax.set_title("Occupancy Grid of Road Area")
        # plt.draw()
        # plt.pause(0.001)


        # Perform DBSCAN clustering on the occupied cells
        if np.any(occupied):
            xy_points = np.column_stack(np.nonzero(occupied))
            clustering = DBSCAN(eps=2, min_samples=2).fit(xy_points)
            labels = clustering.labels_

            # self.ax.clear()
            # self.ax.imshow(road_area.T, cmap='gray', origin='lower')
            # self.ax.set_title("Occupancy Grid with Tracked Objects")

            for k in set(labels):
                if k == -1:
                    continue
                class_member_mask = (labels == k)
                xy_cluster = xy_points[class_member_mask]
                centroid = np.mean(xy_cluster, axis=0)

                # Convert the centroid to world coordinates
                x_world = (centroid[1] + x_min_grid) * self.resolution + self.grid_origin_x
                y_world = centroid[0] * self.resolution + self.grid_origin_y
                centroid_world = np.array([x_world, y_world])
                self.track_objects(centroid_world)
                
        # Log the tracked objects last position, velocity, and acceleration
        # for obj in self.tracked_objects:
        #     self.get_logger().info(f"Object {obj.id}: Position: {obj.positions[-1]}, Velocity: {obj.velocity}, Acceleration: {obj.acceleration}")

        self.times_ran += 1
        self.safe_to_cross_check()
        

        # If the node has run for a while and it is safe to cross, publish a path message
        if self.times_ran > 100 and self.safe_to_cross:
            # self.get_logger().info("Safe to cross the road")
            # Publish a path message with the exit point as the only waypoint
            path_msg = Path()
            pose = PoseStamped()
            pose.pose.position.x = self.crosswalk_exit[0]
            pose.pose.position.y = self.crosswalk_exit[1]
            path_msg.poses.append(pose)
            self.publisher.publish(path_msg)



    # This will continuously check if it is safe to cross the road
    # It should consider safe to cross if any of the following conditions are met:
    # - Objects have very low velocity
    # - Objects are decelerating
    # - If the distance of an object to the robot is increasing
    def safe_to_cross_check(self):
        if self.times_ran < 10:
            return
        for obj in self.tracked_objects:
            if obj.velocity < 0.5 or obj.acceleration < 0 or obj.distances_to_robot[-1] > obj.distances_to_robot[0]:
                self.safe_to_cross = True
                return
        self.safe_to_cross = False
        return



    # The centroid comes in world coordinates and in the proper order (x, y)
    def track_objects(self, centroid):
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # Check if the centroid is close to an existing tracked object
        for obj in self.tracked_objects:
            distance = np.linalg.norm(centroid - obj.positions[-1])
            if distance < 2.0:
                velocity = distance / time_diff
                obj.acceleration = (velocity - obj.velocity) / time_diff
                obj.velocity = velocity
                obj.positions.append(centroid)
                distance_to_robot = np.linalg.norm(centroid - np.array([self.robot_pose.position.x, self.robot_pose.position.y]))
                obj.distances_to_robot.append(distance_to_robot)
                obj.steps_since_seen = 0
                return

        # If the centroid is not close to any existing tracked object, create a new one
        new_obj = TrackedObject(self.tracked_objects_id)
        new_obj.positions.append(centroid)
        self.tracked_objects_id += 1
        self.tracked_objects.append(new_obj)

        # Check if any tracked object has not been seen for a while
        for obj in self.tracked_objects:
            obj.steps_since_seen += 1
            if obj.steps_since_seen > 20:
                self.tracked_objects.remove(obj)


    # TODO: This callback will detect red, yellow, and green traffic lights
    # Only if the traffic light is green and self.safe_to_cross is True, then publish a path message
    def greenlight_callback(self, msg):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = CrossingNode()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
