import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from scipy.spatial.distance import euclidean
from queue import PriorityQueue
import numpy as np
from amr_interfaces.msg import PredictionArray, Prediction, Emergency
from matplotlib import pyplot as plt

class MovingObstacle:
    def __init__(self, position, predicted_position):
        self.position = position
        self.predicted_position = predicted_position

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

        # plt.ion()
        # self.fig, self.ax = plt.subplots()


    def predictions_callback(self, msg):
        self.moving_obstacles = []
        for prediction in msg.predictions:
            current_position = np.array([prediction.current_x, prediction.current_y])
            predicted_position = np.array([prediction.pred_x, prediction.pred_y])
            self.moving_obstacles.append(MovingObstacle(current_position, predicted_position))
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
            self.planning_active = False
        elif distance_to_exit < 1.0:
            self.goal = np.array([40.0, -30.0])
            self.planning_active = True

    def occupancy_grid_callback(self, msg):
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.grid_origin = np.array([msg.info.width / 2, msg.info.height / 2])

        for obstacle in self.moving_obstacles:
            current_grid_pos = self.world_to_grid(obstacle.position)
            predicted_grid_pos = self.world_to_grid(obstacle.predicted_position)
            # self.get_logger().info(f'Obstacle from {current_grid_pos} to {predicted_grid_pos} in grid coordinates')
            self.mark_path_as_occupied(current_grid_pos, predicted_grid_pos)

        # self.update_plot()

    # def update_plot(self):
    #     if self.grid is None:
    #         return
    #     self.ax.clear()
    #     self.ax.imshow(self.grid, cmap='gray', origin='lower')
    #     self.ax.set_title('Occupancy Grid with Obstacles')
    #     self.ax.set_xlabel('X')
    #     self.ax.set_ylabel('Y')
    #     plt.draw()
    #     plt.pause(0.001)

    def world_to_grid(self, world_position):
        grid_x = int((world_position[1] / self.resolution) + self.grid_origin[1])
        grid_y = int((world_position[0] / self.resolution) + self.grid_origin[0])
        return np.array([grid_x, grid_y])

    def mark_path_as_occupied(self, start, end):
        points = self.bresenham_line(start, end)
        for p in points:
            for a in range(-2, 2):
                for b in range(-2, 2):
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

        if self.grid[start] == 100:
            self.get_logger().warn('Robot is in an obstacle')
            start = self.nearest_free_cell_in_direction(start, goal)
            if start is None:
                self.get_logger().warn('No free cell found')
                return
            msg = Emergency()
            msg.emergency_state = 1
            self.emergency_publisher.publish(msg)
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

        self.publisher_.publish(ros_path)
        # self.get_logger().info('Path published')

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
