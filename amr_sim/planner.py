import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from scipy.spatial.distance import euclidean
from queue import PriorityQueue
import numpy as np


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
        self.publisher_ = self.create_publisher(Path, '/path', 10)
        self.crosswalk_entry = np.array([12.0, 37.0])  # Entry point to the crosswalk
        self.crosswalk_exit = np.array([29.0, 37.0])  # Exit point from the crosswalk
        self.goal = self.crosswalk_entry
        self.robot_pose = np.array([0.0, 0.0])  # Robot's initial position
        self.grid = None  # Occupancy grid
        self.resolution = 0.5  # Grid resolution in meters
        self.grid_origin = None  # To be defined based on grid metadata
        self.planning_active = True

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
            # self.get_logger().info('Robot is at the crosswalk entry')
        elif distance_to_exit < 1.0:
            self.goal = np.array([46.0, -20.0])
            self.planning_active = True
            # self.get_logger().info('Robot is at the crosswalk exit')




    def occupancy_grid_callback(self, msg):
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))  # Swap the order of height and width
        self.grid_origin = np.array([msg.info.width / 2, msg.info.height / 2])  # Swap x and y in grid origin


    def heuristic(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def plan_path(self):
        if self.grid is None or self.robot_pose is None:
            return

        # Convert positions from world coordinates to grid indices, swapping x and y
        start = (int(self.robot_pose[1] / self.resolution + self.grid_origin[1]),
                 int(self.robot_pose[0] / self.resolution + self.grid_origin[0]))
        goal = (int(self.goal[1] / self.resolution + self.grid_origin[1]),
                int(self.goal[0] / self.resolution + self.grid_origin[0]))
        
        if not(0 <= goal[0] < self.grid.shape[0] and 0 <= goal[1] < self.grid.shape[1]):
            self.get_logger().warn('Goal position is out of bounds')
            return
        if self.grid[goal] == 100:
            self.get_logger().warn('Goal position is in an obstacle')
            return
    

        # Priority queue for open set
        frontier = PriorityQueue()
        frontier.put((0, tuple(start)))
        came_from = {}
        cost_so_far = {}
        came_from[tuple(start)] = None
        cost_so_far[tuple(start)] = 0

        # Movement directions (8 directions possible)
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
                if 0 <= neighbor[0] < self.grid.shape[0] and 0 <= neighbor[1] < self.grid.shape[1] and self.grid[neighbor] != 100:
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
        
        simplified_path = douglas_peucker(np.array(path), 0.5)
        
        # Log all waypoints of the simplified path in world coordinates
        # for p in simplified_path:
        #     world_x, world_y = (p[1] - self.grid_origin[1]) * self.resolution, (p[0] - self.grid_origin[0]) * self.resolution
        #     self.get_logger().info(f'Waypoint: ({world_x:.2f}, {world_y:.2f})')

        # Remove the first 2 waypoints if there are at least 3 waypoints
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



def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
