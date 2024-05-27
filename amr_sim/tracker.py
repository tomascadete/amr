import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
import transforms3d
from amr_interfaces.msg import Object



class TrackedObject:
    def __init__(self):
        self.id = 0
        self.type = ''
        self.x = 0.0
        self.y = 0.0
        self.steps_since_seen = 0



class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        self.lidar_subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.objects_subscription = self.create_subscription(Object, '/detections', self.incoming_object_callback, 10)
        self.publisher = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)
        self.robot_odom = None
        self.tracked_objects = []
        self.tracked_objects_id = 1
        self.initialize_occupancy_grid()
        self.crosswalk_entry = np.array([12.0, 37.0])
        self.crosswalk_exit = np.array([29.0, 37.0])
        self.crossing = False


    def incoming_object_callback(self, msg):
        if msg.type == 'None':
            return
        
        # If object is too close to the robot (less than 3m), ignore it
        # Robot position in the world frame is self.robot_odom.pose.pose.position
        robot_x = self.robot_odom.pose.pose.position.x
        robot_y = self.robot_odom.pose.pose.position.y
        distance = np.sqrt((robot_x - msg.x)**2 + (robot_y - msg.y)**2)
        if distance < 3.0:
            return

        new_object = True
        for obj in self.tracked_objects:
            distance = np.sqrt((obj.x - msg.x)**2 + (obj.y - msg.y)**2)
            if distance < 1.0:
                obj.x = msg.x
                obj.y = msg.y
                obj.steps_since_seen = 0
                new_object = False
                break

        if new_object:
            obj = TrackedObject()
            obj.id = self.tracked_objects_id
            self.tracked_objects_id += 1
            obj.type = msg.type
            obj.x = msg.x
            obj.y = msg.y
            obj.steps_since_seen = 0
            self.tracked_objects.append(obj)

        # Remove objects that haven't been seen for a while
        for obj in self.tracked_objects:
            obj.steps_since_seen += 1
            if obj.steps_since_seen > 10:
                self.tracked_objects.remove(obj)



    def odom_callback(self, msg):
        self.robot_odom = msg
        # If the robot is within 1m of the crosswalk entry, stop marking the road as occupied
        robot_x = self.robot_odom.pose.pose.position.x
        robot_y = self.robot_odom.pose.pose.position.y
        distance_to_entry = np.linalg.norm(np.array([robot_x, robot_y]) - self.crosswalk_entry)
        distance_to_exit = np.linalg.norm(np.array([robot_x, robot_y]) - self.crosswalk_exit)
        if distance_to_entry < 1.0:
            self.crossing = True
        elif distance_to_exit < 1.0:
            self.crossing = False


    def lidar_callback(self, msg):
        if self.robot_odom is None:
            self.get_logger().info('No odometry data available.')
            return

        # Get robot's current position and orientation
        robot_x = self.robot_odom.pose.pose.position.x
        robot_y = self.robot_odom.pose.pose.position.y
        orientation_q = self.robot_odom.pose.pose.orientation
        _, _, yaw = transforms3d.euler.quat2euler([orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z], axes='sxyz')


        # Compute global coordinates for each LIDAR point
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        adjusted_angles = angles + yaw

        valid_indices = np.where((ranges > msg.range_min) & (ranges < msg.range_max))
        valid_ranges = ranges[valid_indices]
        valid_angles = adjusted_angles[valid_indices]

        global_x = valid_ranges * np.cos(valid_angles) + robot_x
        global_y = valid_ranges * np.sin(valid_angles) + robot_y

        # Clear the occupancy grid
        self.occupancy_grid.data = [0] * (self.occupancy_grid.info.width * self.occupancy_grid.info.height)

        # Update the occupancy grid with the LIDAR data
        for x, y in zip(global_x, global_y):
            if self.crossing and x < 13 and x > 28:
                continue
            grid_x = int((x - self.occupancy_grid.info.origin.position.x) / self.occupancy_grid.info.resolution)
            grid_y = int((y - self.occupancy_grid.info.origin.position.y) / self.occupancy_grid.info.resolution)

            if 0 <= grid_x < self.occupancy_grid.info.width and 0 <= grid_y < self.occupancy_grid.info.height:
                # If the robot is in the crossing procedure, do not add more points to the occupancy grid than necessary
                if not self.crossing:
                    for i in range(-3, 3):
                        for j in range(-3, 3):
                            if 0 <= grid_x + i < self.occupancy_grid.info.width and 0 <= grid_y + j < self.occupancy_grid.info.height:
                                self.occupancy_grid.data[(grid_y + j) * self.occupancy_grid.info.width + grid_x + i] = 100
                else:
                    self.occupancy_grid.data[grid_y * self.occupancy_grid.info.width + grid_x] = 100


        # Update the occupancy grid with the tracked objects if self.crossing is False
        if not self.crossing:
            for obj in self.tracked_objects:
                grid_x = int((obj.x - self.occupancy_grid.info.origin.position.x) / self.occupancy_grid.info.resolution)
                grid_y = int((obj.y - self.occupancy_grid.info.origin.position.y) / self.occupancy_grid.info.resolution)
                if 0 <= grid_x < self.occupancy_grid.info.width and 0 <= grid_y < self.occupancy_grid.info.height:
                    for i in range(-1, 1):
                        for j in range(-1, 1):
                            if 0 <= grid_x + i < self.occupancy_grid.info.width and 0 <= grid_y + j < self.occupancy_grid.info.height:
                                self.occupancy_grid.data[(grid_y + j) * self.occupancy_grid.info.width + grid_x + i] = 100

        # Mark the road as occupied if self.crossing is False
        if not self.crossing:
            x_min_grid = int((12.5 - self.occupancy_grid.info.origin.position.x) / self.occupancy_grid.info.resolution)
            x_max_grid = int((28.5 - self.occupancy_grid.info.origin.position.x) / self.occupancy_grid.info.resolution)
            for i in range(x_min_grid, x_max_grid):
                for j in range(0, self.occupancy_grid.info.height):
                    index = j * self.occupancy_grid.info.width + i
                    if 0 <= index < len(self.occupancy_grid.data):
                        self.occupancy_grid.data[index] = 100


        self.occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.occupancy_grid)

    def initialize_occupancy_grid(self):
        self.occupancy_grid = OccupancyGrid()
        self.occupancy_grid.header.frame_id = 'map'
        self.occupancy_grid.info.resolution = 0.5  # meters per grid cell
        self.occupancy_grid.info.width = 200  # cells
        self.occupancy_grid.info.height = 200  # cells
        self.occupancy_grid.info.origin.position.x = -50.0  # meters
        self.occupancy_grid.info.origin.position.y = -50.0  # meters
        self.occupancy_grid.data = [0] * (self.occupancy_grid.info.width * self.occupancy_grid.info.height)
        # Mark the road area as occupied
        x_min_grid = int((13 - self.occupancy_grid.info.origin.position.x) / self.occupancy_grid.info.resolution)
        x_max_grid = int((28 - self.occupancy_grid.info.origin.position.x) / self.occupancy_grid.info.resolution)
        for i in range(x_min_grid, x_max_grid):
            for j in range(0, self.occupancy_grid.info.height):
                index = j * self.occupancy_grid.info.width + i
                if 0 <= index < len(self.occupancy_grid.data):
                    self.occupancy_grid.data[index] = 100


def main(args=None):
    rclpy.init(args=args)
    tracker = ObjectTracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
