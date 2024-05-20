import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
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
        self.predicted_positions = []

    def add_position(self, position, time):
        self.positions.append(position)
        self.timestamps.append(time)
        self.predict_future_positions()

    def update_velocity_and_acceleration(self, time_diff):
        if len(self.positions) > 1:
            recent_positions = self.positions[-20:]
            recent_timestamps = self.timestamps[-20:]
            last_position = recent_positions[-1]
            last_time = recent_timestamps[-1]
            initial_position = recent_positions[0]
            initial_time = recent_timestamps[0]
            time_diff = last_time - initial_time
            if time_diff > 0:
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

    def calculate_weighted_average(self, data):
        weights = np.exp(-0.1 * np.arange(len(data)))
        return np.average(data[-10:], weights=weights[-10:])

    def predict_future_positions(self):
        if len(self.positions) < 2:
            return

        times = np.array(self.timestamps).reshape(-1, 1)
        positions = np.array(self.positions)

        model_y = LinearRegression().fit(times, positions[:, 1])

        future_times = np.linspace(self.timestamps[-1], self.timestamps[-1] + 15, num=15).reshape(-1, 1)
        future_y = model_y.predict(future_times)

        last_x = self.positions[-1][0]
        last_x = np.full((15, 1), last_x)

        self.predicted_positions = np.column_stack((last_x, future_y))


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
        self.stop_robot = False  # New flag to stop the robot
        self.started_crossing = False

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
            if len(obj.predicted_positions) > 0:
                predicted_data = np.array(obj.predicted_positions)
                self.ax.scatter(predicted_data[:, 0], predicted_data[:, 1], marker='x', label=f"Object {obj.id} Prediction")

        self.ax.axhline(y=37, color='r', linestyle='--', label='Robot Desired Path')

        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def traffic_light_callback(self, msg):
        if not self.crossing:
            return
        self.traffic_light_state = msg.colour

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
            self.started_crossing = False

    def occupancy_grid_callback(self, msg):
        if not self.crossing:
            return

        self.height = msg.info.height
        self.width = msg.info.width
        self.resolution = msg.info.resolution
        self.grid_origin_x = msg.info.origin.position.x
        self.grid_origin_y = msg.info.origin.position.y
        self.grid = np.array(msg.data).reshape((self.height, self.width))

        x_min_grid = int((13 - self.grid_origin_x) / self.resolution)
        x_max_grid = int((28 - self.grid_origin_x) / self.resolution)
        road_area = self.grid[:, x_min_grid:x_max_grid]
        occupied = road_area > 0

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

                x_world = (centroid[1] + x_min_grid) * self.resolution + self.grid_origin_x
                y_world = centroid[0] * self.resolution + self.grid_origin_y
                centroid_world = np.array([x_world, y_world])
                self.track_objects(centroid_world)

        self.times_ran += 1
        self.safe_to_cross_check()
        self.update_plot()

        for obj in self.tracked_objects:
            obj.steps_since_seen += 1
            if obj.steps_since_seen > 40:
                self.tracked_objects.remove(obj)

        if self.safe_to_cross:
            self.safety_counter += 1
        else:
            self.safety_counter = 0

        if self.times_ran > 50 and self.safety_counter > 10 and not self.stop_robot:
            path_msg = Path()
            pose = PoseStamped()
            pose.pose.position.x = self.crosswalk_exit[0]
            pose.pose.position.y = self.crosswalk_exit[1]
            path_msg.poses.append(pose)
            self.publisher.publish(path_msg)
            self.started_crossing = True

        if self.started_crossing:
            self.check_and_handle_crossing_objects()

    def safe_to_cross_check(self):
        if self.traffic_light_state == 'green':
            self.safe_to_cross = True
            for obj in self.tracked_objects:
                if obj.is_approaching:
                    self.safe_to_cross = False
                    break
        else:
            self.safe_to_cross = False

    def track_objects(self, centroid):
        current_time = self.get_clock().now()
        timestamp = current_time.nanoseconds / 1e9
        time_diff = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        max_history = 10
        decay_rate = 0.9

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
                    obj.update_velocity_and_acceleration(time_diff)
                    obj.distances_to_robot.append(np.linalg.norm(centroid - np.array([self.robot_pose.position.x, self.robot_pose.position.y])))

                    if len(obj.distances_to_robot) >= 5:
                        distances = np.array(obj.distances_to_robot[-5:])
                        if obj.distances_to_robot[-1] > obj.distances_to_robot[0]:
                            obj.is_approaching = False
                        else:
                            obj.is_approaching = True

                return

        new_obj = TrackedObject(self.tracked_objects_id)
        new_obj.positions.append(centroid)
        new_obj.timestamps.append(timestamp)
        self.tracked_objects_id += 1
        self.tracked_objects.append(new_obj)

    def check_and_handle_crossing_objects(self):
        if self.robot_pose is None:
            return
        
        robot_x = self.robot_pose.position.x
        robot_y = self.robot_pose.position.y
        if not self.stop_robot:
            for obj in self.tracked_objects:
                if len(obj.predicted_positions) > 0:
                    future_positions = np.array(obj.predicted_positions)
                    if np.any(future_positions[:, 1] > 37) and np.any(future_positions[:, 1] < 37) and np.all(future_positions[:, 0] > robot_x + 3):
                        # The robot should stop at its current position or at (20,37) if robot's x position is less than 20
                        path_msg = Path()
                        pose = PoseStamped()
                        pose.pose.position.x = 20.0 if robot_x < 20 else robot_x
                        pose.pose.position.y = robot_y
                        path_msg.poses.append(pose)
                        self.publisher.publish(path_msg)
                        self.get_logger().info(f"Robot stopping due to object {obj.id} crossing path")

                        self.stop_robot = True  # Set flag to stop robot
                        return

        # Check if the robot should resume
        if self.stop_robot:
            all_objects_passed = True
            for obj in self.tracked_objects:
                future_positions = np.array(obj.predicted_positions)
                # If 
                if np.any(future_positions[:, 1] > 37) and np.any(future_positions[:, 1] < 37) and np.all(future_positions[:, 0] > robot_x):
                    all_objects_passed = False
                    break

            if all_objects_passed:
                path_msg = Path()
                pose = PoseStamped()
                pose.pose.position.x = self.crosswalk_exit[0]
                pose.pose.position.y = self.crosswalk_exit[1]
                path_msg.poses.append(pose)
                self.publisher.publish(path_msg)
                self.get_logger().info(f"Robot resuming after all objects passed")
                self.stop_robot = False  # Reset flag to allow robot to move


def main(args=None):
    rclpy.init(args=args)
    node = CrossingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
