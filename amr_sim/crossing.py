import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from amr_interfaces.msg import LightColour, PredictionArray, Prediction
import numpy as np
from queue import PriorityQueue
from scipy.spatial.distance import euclidean

class MovingObstacle:
    def __init__(self, position, predicted_position):
        self.position = position
        self.predicted_position = predicted_position

class CrossingNode(Node):
    def __init__(self):
        super().__init__('crossing_node')
        self.subscription_traffic_light = self.create_subscription(LightColour, '/light_status', self.traffic_light_callback, 10)
        self.subscription_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.subscription_occ_grid = self.create_subscription(OccupancyGrid, '/occupancy_grid', self.occupancy_grid_callback, 10)
        self.subscription_predictions = self.create_subscription(PredictionArray, '/predictions', self.predictions_callback, 10)
        self.publisher = self.create_publisher(Path, '/path', 10)
        self.robot_pose = None
        self.crosswalk_entry = np.array([12.0, 37.0])
        self.crosswalk_exit = np.array([29.0, 37.0])
        self.grid = None
        self.grid_origin = None
        self.grid_resolution = None
        self.crossing = False
        self.safe_to_cross = False
        self.traffic_light_state = None
        self.moving_obstacles = []

    def traffic_light_callback(self, msg):
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

    def predictions_callback(self, msg):
        self.moving_obstacles = []
        for prediction in msg.predictions:
            current_position = np.array([prediction.current_x, prediction.current_y])
            predicted_position = np.array([prediction.pred_x, prediction.pred_y])
            self.moving_obstacles.append(MovingObstacle(current_position, predicted_position))

    def occupancy_grid_callback(self, msg):
        if not self.crossing:
            return

        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.grid_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.grid_resolution = msg.info.resolution


        for obstacle in self.moving_obstacles:
            current_grid_pos = self.world_to_grid(obstacle.position)
            predicted_grid_pos = self.world_to_grid(obstacle.predicted_position)
            self.mark_path_as_occupied(current_grid_pos, predicted_grid_pos)

        if self.traffic_light_state == 'green':
            self.plan_path()

    def world_to_grid(self, world_position):
        grid_x = int((world_position[0] - self.grid_origin[0]) / self.grid_resolution)
        grid_y = int((world_position[1] - self.grid_origin[1]) / self.grid_resolution)
        return np.array([grid_x, grid_y])

    def mark_path_as_occupied(self, start, end):
        points = self.bresenham_line(start, end)
        for p in points:
            for a in range(-4, 4):
                for b in range(-4, 4):
                    if 0 <= p[0] + a < self.grid.shape[0] and 0 <= p[1] + b < self.grid.shape[1]:
                        self.grid[p[0] + a, p[1] + b] = 100

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

    def plan_path(self):
        if self.grid is None or self.robot_pose is None:
            return

        start = self.world_to_grid(np.array([self.robot_pose.position.x, self.robot_pose.position.y]))
        goal = self.world_to_grid(self.crosswalk_exit)

        if not(0 <= goal[0] < self.grid.shape[0] and 0 <= goal[1] < self.grid.shape[1]):
            self.get_logger().warn('Goal position is out of bounds')
            return
        if self.grid[goal[0], goal[1]] == 100:
            self.get_logger().warn('Goal position is in an obstacle')
            return

        frontier = PriorityQueue()
        frontier.put((0, tuple(start)))
        came_from = {}
        cost_so_far = {}
        came_from[tuple(start)] = None
        cost_so_far[tuple(start)] = 0

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        movement_cost = {d: 1 if d[0] == 0 or d[1] == 0 else np.sqrt(2) for d in directions}
        goal_reached = False

        while not frontier.empty():
            current = frontier.get()[1]

            if np.array_equal(current, goal):
                goal_reached = True
                break

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.grid.shape[0] and 0 <= neighbor[1] < self.grid.shape[1] and self.grid[neighbor[0], neighbor[1]] != 100:
                    new_cost = cost_so_far[current] + movement_cost[(dx, dy)]

                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + self.heuristic(neighbor, goal)
                        frontier.put((priority, neighbor))
                        came_from[neighbor] = current

        if not goal_reached:
            self.get_logger().warn('Goal is unreachable')
            return

        path = []
        if tuple(goal) in came_from:
            step = tuple(goal)
            while step != tuple(start):
                path.append(step)
                step = came_from[step]
            path.append(tuple(start))
            path.reverse()
        else:
            self.get_logger().warn('No path found')
            return

        simplified_path = self.douglas_peucker(np.array(path), 0.5)

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
            world_x, world_y = (p[0] * self.grid_resolution + self.grid_origin[0]), (p[1] * self.grid_resolution + self.grid_origin[1])
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            ros_path.poses.append(pose)

        self.publisher.publish(ros_path)

    def heuristic(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def douglas_peucker(self, points, epsilon):
        dmax = 0
        index = 0
        for i in range(1, len(points) - 1):
            d = self.perpendicular_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d
        if dmax > epsilon:
            results1 = self.douglas_peucker(points[:index + 1], epsilon)
            results2 = self.douglas_peucker(points[index:], epsilon)
            return results1[:-1] + results2
        else:
            return [points[0], points[-1]]

    def perpendicular_distance(self, pt, line_start, line_end):
        if np.all(line_start == line_end):
            return np.linalg.norm(pt - line_start)
        else:
            norm = np.linalg.norm(line_end - line_start)
            return np.linalg.norm(np.cross(line_end - line_start, line_start - pt)) / norm

def main(args=None):
    rclpy.init(args=args)
    node = CrossingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
