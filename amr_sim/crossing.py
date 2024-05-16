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

    def add_position(self, position, time):
        self.positions.append(position)
        self.timestamps.append(time)


    def update_velocity_and_acceleration(self, time_diff):
        if len(self.positions) > 1:
            initial_position = self.positions[0]
            last_position = self.positions[-1]
            initial_time = self.timestamps[0]
            last_time = self.timestamps[-1]
            time_diff = last_time - initial_time
            if time_diff > 0:
                # Consider only displacement along the y-axis
                displacement = last_position[1] - initial_position[1]
                velocity = displacement / time_diff
                self.velocities.append(velocity)
                if len(self.velocities) > 1:
                    self.velocity = self.calculate_weighted_average(self.velocities)
            else:
                self.velocity = 0.0
        else:
            self.velocity = 0.0

        if len(self.velocities) > 1:
            previous_velocity = self.velocities[-2]
            current_velocity = self.velocities[-1]
            self.acceleration = (current_velocity - previous_velocity) / time_diff
            self.accelerations.append(self.acceleration)
        else:
            self.acceleration = 0.0


    def calculate_weighted_average(self,data):
        weights = np.exp(-0.5*np.arange(len(data))) # The higher the decay rate, the more recent values will be weighted more heavily
        return np.average(data[-10:], weights=weights[-10:])
            



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
        # self.get_logger().info(f"Traffic light state updated to: {msg.colour}")


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
                centroid_world = np.array([x_world, y_world])
                self.track_objects(centroid_world)


        self.times_ran += 1
        self.safe_to_cross_check()
        self.update_plot()

        # Log the velocities and accelerations of the tracked objects
        for obj in self.tracked_objects:
            if len(obj.velocities) > 0:
                self.get_logger().info(f"Object {obj.id} - Velocity: {obj.velocity:.2f} m/s, Acceleration: {obj.acceleration:.2f} m/s^2")


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
            # self.publisher.publish(path_msg)



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

        max_history = 10
        decay_rate = 0.9
        # The higher the decay rate, the more recent positions will be weighted more heavily

        # Check if the centroid is close to an existing tracked object
        for obj in self.tracked_objects:
            recent_positions = obj.positions[-max_history:]
            num_positions = len(recent_positions)
            if num_positions > 0:
                weights = np.exp(-decay_rate * np.arange(num_positions))[::-1]
                weighted_sum = np.sum(recent_positions * weights[:, np.newaxis], axis=0)
                weighted_average_position = weighted_sum / np.sum(weights)

            distance = np.linalg.norm(centroid - weighted_average_position)
            if distance < 1.0:
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
                    obj.distances_to_robot.append(np.linalg.norm(centroid - np.array([self.robot_pose.position.x, self.robot_pose.position.y])))

                    if len(obj.distances_to_robot) >= 5:
                        distances = np.array(obj.distances_to_robot[-5:])
                        average = np.mean(distances)
                        if obj.distances_to_robot[-1] > average:
                            obj.is_approaching = False
                        else:
                            obj.is_approaching = True


                return

        
        new_obj = TrackedObject(self.tracked_objects_id)
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