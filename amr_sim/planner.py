import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from scipy.spatial.distance import euclidean
from queue import PriorityQueue
import numpy as np
from amr_interfaces.msg import PredictionArray, Prediction, Emergency, LightColour
from matplotlib import pyplot as plt

class MovingObstacle:
    def __init__(self, position, predicted_position, size):
        self.position = position
        self.predicted_position = predicted_position
        self.size = size

# Helper function: Douglas-Peucker Algorithm
def douglas_peucker(points, epsilon):
    dmax = 0
    index = 0
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        results1 = douglas_peucker(points[:index + 1], epsilon)
        results2 = douglas_peucker(points[index:], epsilon)
        return results1[:-1] + results2
    else:
        return [points[0], points[-1]]

def perpendicular_distance(pt, line_start, line_end):
    if(np.all(line_start == line_end)):
        return np.linalg.norm(pt - line_start)
    else:
        norm = np.linalg.norm(line_end - line_start)
        return np.linalg.norm(np.cross(line_end - line_start, line_start - pt)) / norm

class Planner(Node):
    def __init__(self):
        super().__init__('planner')
        self.create_subscription(OccupancyGrid, '/occupancy_grid', self.occupancy_grid_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PredictionArray, '/predictions', self.predictions_callback, 10)
        self.create_subscription(LightColour, '/light_status', self.traffic_light_callback, 10)
        self.publisher_ = self.create_publisher(Path, '/path', 10)
        self.emergency_publisher = self.create_publisher(Emergency, '/emergency', 10)
        self.crosswalk_entry = np.array([12.0, 37.0])  # Entry point to the crosswalk
        self.crosswalk_exit = np.array([29.0, 37.0])  # Exit point from the crosswalk
        self.goal = self.crosswalk_entry
        self.robot_pose = np.array([0.0, 0.0])  # Robot's initial position
        self.grid = None  # Occupancy grid
        self.resolution = 0.5  # Grid resolution in meters
        self.grid_origin = None  # To be defined based on grid metadata
        self.planning_active = True
        self.moving_obstacles = []
        self.previous_path = None
        self.traffic_light_state = None
        self.started_crossing_flag = False
        self.waypoint_found = False

        plt.ion()
        self.fig, self.ax = plt.subplots()

    def traffic_light_callback(self, msg):
        self.traffic_light_state = msg.colour
        # Log the traffic light state
        # self.get_logger().info(f'Traffic light state: {self.traffic_light_state}')


    def predictions_callback(self, msg):
        self.moving_obstacles = []
        for prediction in msg.predictions:
            current_position = np.array([prediction.current_x, prediction.current_y])
            predicted_position = np.array([prediction.pred_x, prediction.pred_y])
            size = prediction.size
            self.moving_obstacles.append(MovingObstacle(current_position, predicted_position, size))
            # self.get_logger().info(f'Current position: ({prediction.current_x:.2f}, {prediction.current_y:.2f}), Predicted position: ({prediction.pred_x:.2f}, {prediction.pred_y:.2f})')

    def odom_callback(self, msg):
        self.robot_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.check_crosswalk_proximity()
        if self.planning_active:
            self.plan_path()

    def check_crosswalk_proximity(self):
        if self.robot_pose is None:
            return
        distance_to_entry = np.linalg.norm(self.robot_pose - self.crosswalk_entry)
        distance_to_exit = np.linalg.norm(self.robot_pose - self.crosswalk_exit)
        if distance_to_entry < 1.0:
            if not self.started_crossing_flag:
                self.planning_active = False
            self.goal = self.crosswalk_exit
            if self.traffic_light_state == 'green':
                self.planning_active = True
                self.started_crossing_flag = True
        elif distance_to_exit < 1.0:
            self.goal = np.array([40.0, -30.0])
            self.planning_active = True
            self.started_crossing_flag = False

    def occupancy_grid_callback(self, msg):
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.grid_origin = np.array([msg.info.width / 2, msg.info.height / 2])

        # Expand the occupied cells to account for the robot's size
        self.expand_occupied_cells()

        # Mark the road area as occupied if self.planning_active is true and not started crossing
        if self.planning_active and not self.started_crossing_flag:
            x_min_grid = int((12.5 - msg.info.origin.position.x) / msg.info.resolution)
            x_max_grid = int((28.5 - msg.info.origin.position.x) / msg.info.resolution)
            for i in range (x_min_grid, x_max_grid):
                for j in range(0, msg.info.height):
                    if 0 <= i < msg.info.width and 0 <= j < msg.info.height:
                        self.grid[j, i] = 100

        # Mark the moving obstacles' predicted trajectory as occupied
        for obstacle in self.moving_obstacles:
            current_grid_pos = self.world_to_grid(obstacle.position)
            predicted_grid_pos = self.world_to_grid(obstacle.predicted_position)
            extra_cells_to_mark = int(obstacle.size / self.resolution)
            # Extra cells to mark should be at least 1 and smaller than 5
            if extra_cells_to_mark < 1:
                extra_cells_to_mark = 1
            elif extra_cells_to_mark > 4:
                extra_cells_to_mark = 4
            # self.get_logger().info(f'Obstacle from {current_grid_pos} to {predicted_grid_pos} in grid coordinates')
            self.mark_path_as_occupied(current_grid_pos, predicted_grid_pos, extra_cells_to_mark)

        self.update_plot()


    def expand_occupied_cells(self):
        expanded_grid = self.grid.copy()
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 100:
                    for di in range(-4, 4):
                        for dj in range(-4, 4):
                            if 0 <= i + di < self.grid.shape[0] and 0 <= j + dj < self.grid.shape[1]:
                                expanded_grid[i + di, j + dj] = 100

        self.grid = expanded_grid


    def update_plot(self):
        if self.grid is None:
            return
        self.ax.clear()
        self.ax.imshow(self.grid, cmap='gray', origin='lower')
        self.ax.set_title('Occupancy Grid with Obstacles')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        plt.draw()
        plt.pause(0.001)

    def world_to_grid(self, world_position):
        grid_x = int((world_position[1] / self.resolution) + self.grid_origin[1])
        grid_y = int((world_position[0] / self.resolution) + self.grid_origin[0])
        return np.array([grid_x, grid_y])

    def mark_path_as_occupied(self, start, end, extra_cells_to_mark):
        r_min = -extra_cells_to_mark
        r_max = extra_cells_to_mark
        points = self.bresenham_line(start, end)
        for p in points:
            for a in range(r_min, r_max):
                for b in range(r_min, r_max):
                    if 0 <= p[0] + a < self.grid.shape[0] and 0 <= p[1] + b < self.grid.shape[1]:
                        self.grid[p[0] + a, p[1] + b] = 100
                        # self.get_logger().info(f'Grid updated at ({p[0] + a}, {p[1] + b})')

    def bresenham_line(self, start, end):
        points = []
        x1, y1 = start
        x2, y2 = end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return points

    def heuristic(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def plan_path(self):
        if self.grid is None or self.robot_pose is None:
            return

        start = (int(self.robot_pose[1] / self.resolution + self.grid_origin[1]),
                int(self.robot_pose[0] / self.resolution + self.grid_origin[0]))
        goal = (int(self.goal[1] / self.resolution + self.grid_origin[1]),
                int(self.goal[0] / self.resolution + self.grid_origin[0]))

        if not(0 <= goal[0] < self.grid.shape[0] and 0 <= goal[1] < self.grid.shape[1]):
            self.get_logger().warn('Goal position is out of bounds')
            return
        if self.grid[goal] == 100:
            self.get_logger().warn('Goal position is in an obstacle')

        robot_distance_to_crosswalk = np.linalg.norm(self.robot_pose - self.crosswalk_entry)
        if self.grid[start] == 100 and robot_distance_to_crosswalk > 1.0:
            # self.get_logger().warn('Robot is in an obstacles trajectory')

            apf_path = self.generate_apf_path()
            if apf_path is not None:
                self.publisher_.publish(apf_path)       
                msg = Emergency()
                msg.emergency_state = 1
                self.emergency_publisher.publish(msg)         
                # Log all the waypoints in the path
                # for pose in apf_path.poses:
                #     self.get_logger().info(f'Waypoint: ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})')


            return

        else:
            msg = Emergency()
            msg.emergency_state = 0
            self.emergency_publisher.publish(msg)

        frontier = PriorityQueue()
        frontier.put((0, tuple(start)))
        came_from = {}
        cost_so_far = {}
        came_from[tuple(start)] = None
        cost_so_far[tuple(start)] = 0

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        movement_cost = {d: 1 if d[0] == 0 or d[1] == 0 else np.sqrt(2) for d in directions}
        goal_reached = False
        closest_point = start
        closest_distance = self.heuristic(start, goal)

        while not frontier.empty():
            current = frontier.get()[1]
            current_distance = self.heuristic(current, goal)

            if current_distance < closest_distance:
                closest_distance = current_distance
                closest_point = current

            if np.array_equal(current, goal):
                goal_reached = True
                break

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.grid.shape[0] and 0 <= neighbor[1] < self.grid.shape[1] and self.grid[neighbor] != 100:
                    new_cost = cost_so_far[current] + movement_cost[(dx, dy)]

                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + self.heuristic(neighbor, goal)
                        frontier.put((priority, neighbor))
                        came_from[neighbor] = current

        path = []
        end_point = goal if goal_reached else closest_point

        if tuple(end_point) in came_from:
            step = tuple(end_point)
            while step != tuple(start):
                path.append(step)
                step = came_from[step]
            path.append(tuple(start))
            path.reverse()
        else:
            self.get_logger().warn('No path found')
            return

        simplified_path = douglas_peucker(np.array(path), 0.5)

        if len(simplified_path) > 2:
            simplified_path = simplified_path[2:]
        else:
            return

        ros_path = Path()
        ros_path.header.stamp = self.get_clock().now().to_msg()
        ros_path.header.frame_id = "map"
        for p in simplified_path:
            pose = PoseStamped()
            pose.header.stamp = ros_path.header.stamp
            pose.header.frame_id = ros_path.header.frame_id
            world_x, world_y = (p[1] - self.grid_origin[1]) * self.resolution, (p[0] - self.grid_origin[0]) * self.resolution
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            ros_path.poses.append(pose)
            self.previous_path = ros_path

        self.publisher_.publish(ros_path)
        # self.get_logger().info('Path published')


    def generate_apf_path(self):
        current_position = np.copy(self.robot_pose)
        goal_position = self.goal
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"

        eta = 1.0  # Attractive force coefficient
        zeta = 5.0  # Repulsive force coefficient
        Q_star = 10.0  # Distance threshold for repulsive force

        force = np.zeros(2)
        attractive_force = eta * (goal_position - current_position) / np.linalg.norm(goal_position - current_position)
        force += attractive_force

        # Repulsive force calculation
        for obstacle in self.moving_obstacles:
            distance = np.linalg.norm(current_position - obstacle.position)
            if distance < Q_star:
                repulsive_force = zeta * (1.0 / distance - 1.0 / Q_star) * (1.0 / distance ** 2) * (current_position - obstacle.position)
                force += repulsive_force

        # Normalize the force to get the direction
        if np.linalg.norm(force) > 0:
            force_direction = force / np.linalg.norm(force)
        else:
            force_direction = np.array([0.0, 0.0])

        # Check for parallell vectors and add a small deviation if needed
        for obstacle in self.moving_obstacles:
            predicted_direction = (obstacle.predicted_position - obstacle.position) / np.linalg.norm(obstacle.predicted_position - obstacle.position)
            if np.linalg.norm(predicted_direction) > 0:
                predicted_direction /= np.linalg.norm(predicted_direction)
                dot_product = np.dot(force_direction, predicted_direction)
                if abs(dot_product) > 0.99:
                    # Add a small deviation to the force direction
                    deviation = np.array([predicted_direction[1], -predicted_direction[0]]) * 0.05
                    force_direction += deviation
                    force_direction /= np.linalg.norm(force_direction)
                    # self.get_logger().info('Forces are parallel, adding deviation')

        # Search for the first free cell in the direction of the force
        max_distance = 10.0
        step_size = self.resolution
        test_position = current_position
        free_position = None

        for step in range(int(max_distance / step_size)):
            test_position = test_position + force_direction * step_size * step
            grid_position = self.world_to_grid(test_position)

            if 0 <= grid_position[0] < self.grid.shape[0] and 0 <= grid_position[1] < self.grid.shape[1]:
                if self.grid[grid_position[0], grid_position[1]] != 100:
                    free_position = test_position
                    # Break only if the free cell is at least 2m away from the robot
                    if np.linalg.norm(free_position - current_position) > 2.0:
                        break

        # Check if an unoccupied cell was found
        if free_position is not None:
            path_point = PoseStamped()
            path_point.header.stamp = path.header.stamp
            path_point.header.frame_id = path.header.frame_id
            path_point.pose.position.x = free_position[0]
            path_point.pose.position.y = free_position[1]   
            path.poses.append(path_point)

            # Log the waypoint
            # self.get_logger().info(f'Waypoint: ({path_point.pose.position.x:.2f}, {path_point.pose.position.y:.2f})')


        return path








    def nearest_free_cell_in_direction(self, start, goal):
        frontier = PriorityQueue()
        frontier.put((0, start))
        visited = set()
        visited.add(start)
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while not frontier.empty():
            current = frontier.get()[1]

            if self.grid[current] != 100:
                return current

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.grid.shape[0] and 0 <= neighbor[1] < self.grid.shape[1] and neighbor not in visited:
                    frontier.put((self.heuristic(neighbor, goal), neighbor))
                    visited.add(neighbor)

        return None



def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
