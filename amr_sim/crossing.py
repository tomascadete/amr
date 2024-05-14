import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import Twist, PoseStamped
import cv2
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from amr_interfaces.msg import LightColour



class TrackedObject:
    def __init__(self, id):
        self.id = id
        self.positions = []
        self.timestamps = []
        self.next_position = None
        self.velocity = 0.0
        self.velocities = []
        self.acceleration = 0.0
        self.accelerations = []
        self.steps_since_seen = 0
        self.distances_to_robot = []
        self.is_approaching = True
        self.y_ascent = None
        self.y_descent = None

    def add_position(self, position, time):
        self.positions.append(position)
        self.timestamps.append(time)
        # Sort positions based on x value, then y value conditionally
        if 12 <= position[0] <= 18:
            # Descending order of y
            self.positions = sorted(self.positions, key=lambda x: (x[0], -x[1]))
            self.y_ascent = False
            self.y_descent = True
        elif 23 < position[0] <= 29:
            # Ascending order of y
            self.positions = sorted(self.positions, key=lambda x: (x[0], x[1]))
            self.y_ascent = True
            self.y_descent = False


    # This should update the velocity and acceleration
    # It should always do the average with the rest of the calculated velocities
    # This will make the velocity more stable and towards the real value
    def update_velocity_and_acceleration(self, time_diff):
        if len(self.positions) > 1:
            velocity = np.linalg.norm(self.positions[-1] - self.positions[-2]) / time_diff
            self.velocities.append(velocity)

            if len(self.velocities) > 10:
                self.velocities.pop(0)
                acceleration = (self.velocities[-1] - self.velocities[0]) / time_diff
                self.accelerations.append(acceleration)

        else:
            self.velocities.append(0.0)

        self.velocity = np.mean(self.velocities)
        self.acceleration = np.mean(self.accelerations)

            



class CrossingNode(Node):
    def __init__(self):
        super().__init__('crossing_node')
        self.bridge = CvBridge()
        self.ryg_model = YOLO("traffic_light.pt")
        self.subscription_traffic_light = self.create_subscription(LightColour, '/light_status', self.traffic_light_callback, 10)
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
        self.traffic_light_state = None
        self.safety_counter = 0

        # Initialize the plot for displaying the trajectories of the tracked objects
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Tracked Objects' Trajectories")
        plt.show()


    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlim(12, 29)
        self.ax.set_ylim(0, 50)
        self.ax.set_title("Tracked Objects' Trajectories")

        for obj in self.tracked_objects:
            if len(obj.positions) > 1:
                data = np.array(obj.positions)
                self.ax.plot(data[:, 0], data[:, 1], marker='o', label=f"Object {obj.id}")

        # Plot a line at y = 37 to represent the robot path in the crosswalk
        self.ax.axhline(y=37, color='r', linestyle='--', label='Robot Desired Path')

        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



    def traffic_light_callback(self, msg):
        if not self.crossing:
            return
        self.traffic_light_state = msg.colour
        self.get_logger().info(f"Traffic light state updated to: {msg.colour}")


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
        elif distance_to_exit < 1.0:
            self.crossing = False


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



        # Perform DBSCAN clustering on the occupied cells
        if np.any(occupied):
            xy_points = np.column_stack(np.nonzero(occupied))
            clustering = DBSCAN(eps=2, min_samples=2).fit(xy_points)
            labels = clustering.labels_


            for k in set(labels):
                if k == -1:
                    continue
                class_member_mask = (labels == k)
                xy_cluster = xy_points[class_member_mask]
                centroid = np.mean(xy_cluster, axis=0)

                # Convert the centroid to world coordinates
                x_world = (centroid[1] + x_min_grid) * self.resolution + self.grid_origin_x
                y_world = centroid[0] * self.resolution + self.grid_origin_y
                if x_world < 19:
                    x_world = 14.5
                elif x_world > 20:
                    x_world = 25.5
                centroid_world = np.array([x_world, y_world])
                self.track_objects(centroid_world)
                
        # Log the tracked objects predicted next position, velocity, and acceleration
        # for obj in self.tracked_objects:
        #     if obj.next_position is not None:
        #         self.get_logger().info(f"Object {obj.id}: next position = {obj.next_position}, velocity = {obj.velocity}, acceleration = {obj.acceleration}")

        self.times_ran += 1
        self.safe_to_cross_check()
        self.update_plot()


        # Check if any tracked object has not been seen for a while
        for obj in self.tracked_objects:
            obj.steps_since_seen += 1
            if obj.steps_since_seen > 40:
                self.tracked_objects.remove(obj)

        
        # Count how many times in a row safe_to_cross is True
        if self.safe_to_cross:
            self.safety_counter += 1
        else:
            self.safety_counter = 0

        if self.times_ran > 100 and self.safety_counter > 10:
            # self.get_logger().info("Safe to cross the road")
            # Publish a path message with the exit point as the only waypoint
            path_msg = Path()
            pose = PoseStamped()
            pose.pose.position.x = self.crosswalk_exit[0]
            pose.pose.position.y = self.crosswalk_exit[1]
            path_msg.poses.append(pose)
            self.publisher.publish(path_msg)



    # This will continuously check if it is safe to cross the road
    # It should consider safe to cross if:
    # Traffic light is green
    # All tracked objects are stopped or their distance to the robot is increasing
    def safe_to_cross_check(self):
        if self.traffic_light_state == 'green':
            self.safe_to_cross = True
            for obj in self.tracked_objects:
                if obj.is_approaching:
                    self.safe_to_cross = False
                    break
        else:
            self.safe_to_cross = False



    # The centroid comes in world coordinates and in the proper order (x, y)
    def track_objects(self, centroid):
        current_time = self.get_clock().now()
        timestamp = current_time.nanoseconds / 1e9
        time_diff = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # Identify whether the centroid is within the specific x range
        x_range_y_descent = 12 <= centroid[0] <= 18
        x_range_y_ascent = 23 < centroid[0] <= 29

        # Check if the centroid is close to an existing tracked object
        for obj in self.tracked_objects:
            # Calculate the distance between the centroid and the last position of the object
            # Take into account that if an object has y_descent, the position vector is sorted in descending order of y
            distance = np.linalg.norm(centroid - obj.positions[-1])
            if distance < 5.0 and ((x_range_y_ascent and obj.y_ascent) or (x_range_y_descent and obj.y_descent)):
                obj.add_position(centroid, timestamp)
                obj.steps_since_seen = 0


                if len(obj.positions) > 10:
                    data = np.array(obj.positions)
                    X, y = data[:, 0].reshape(-1, 1), data[:, 1]
                    model = LinearRegression().fit(X, y)

                    # Predict the next position
                    next_x = data[-1, 0] + np.mean(np.diff(data[:, 0]))
                    next_y = model.predict([[next_x]])[0]
                    next_position = np.array([next_x, next_y])
                    obj.next_position = next_position

                    obj.update_velocity_and_acceleration(time_diff)
                    obj.distances_to_robot.append(np.linalg.norm(next_position - np.array([self.robot_pose.position.x, self.robot_pose.position.y])))
                    if obj.y_ascent:
                        obj.is_approaching = obj.positions[-1][1] < 37
                    elif obj.y_descent:
                        obj.is_approaching = obj.positions[-1][1] > 37



                return

        if x_range_y_ascent or x_range_y_descent:
            if x_range_y_ascent:
                new_obj = TrackedObject(self.tracked_objects_id)
                new_obj.y_ascent = True
            elif x_range_y_descent:
                new_obj = TrackedObject(self.tracked_objects_id)
                new_obj.y_descent = True
            new_obj.positions.append(centroid)
            self.tracked_objects_id += 1
            self.tracked_objects.append(new_obj)




def main(args=None):
    rclpy.init(args=args)
    node = CrossingNode()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()