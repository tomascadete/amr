import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from amr_interfaces.msg import ObstacleArray
import numpy as np
import heapq
from math import sqrt

class Planner(Node):
    def __init__(self):
        super().__init__('planner')
        self.create_subscription(ObstacleArray, '/tracked_objects', self.tracked_objects_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher_ = self.create_publisher(Path, '/path', 10)
        self.goal = [10.0, 0.0]
        self.robot_pose = [0.0, 0.0]
        self.obstacles = []
        self.grid_resolution = 0.5
        self.grid_size = 100  # 50 / 0.1 = 500 cells on each side
        self.grid = np.zeros((self.grid_size, self.grid_size))

    def odom_callback(self, msg):
        self.robot_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]

    def tracked_objects_callback(self, msg):
        self.obstacles = []
        for obstacle in msg.obstacles:
            self.obstacles.append([obstacle.x, obstacle.y])

        # Reset grid
        self.grid.fill(0)
        
        # Update grid for obstacles
        for obstacle in self.obstacles:
            ox, oy = int(obstacle[0] / self.grid_resolution + self.grid_size / 2), int(obstacle[1] / self.grid_resolution + self.grid_size / 2)
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                self.grid[ox-4:ox+4, oy-4:oy+4] = 1  # Mark obstacles in grid
        
        path = self.a_star()
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for waypoint in path:
            pose = PoseStamped()
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1]
            path_msg.poses.append(pose)
            # Log path points
            # self.get_logger().info(f'Path point: {waypoint}')
        self.publisher_.publish(path_msg)
        self.get_logger().info('Published path')


    def a_star(self):
        start = (int(self.robot_pose[0] / self.grid_resolution + self.grid_size / 2), int(self.robot_pose[1] / self.grid_resolution + self.grid_size / 2))
        goal = (int(self.goal[0] / self.grid_resolution + self.grid_size / 2), int(self.goal[1] / self.grid_resolution + self.grid_size / 2))

        # Priority queue for A* algorithm
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start, None))

        # A* state
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while open_set:
            _, current_cost, current, parent = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            for neighbor in self.neighbors(current):
                new_cost = current_cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor, current))
                    came_from[neighbor] = current

        return [] # No path found
    
    def neighbors(self, node):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)] # 4-way connectivity
        result = []
        x, y = node
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and not self.grid[nx, ny]:
                result.append((nx, ny))
        return result
    
    def heuristic(self, a, b):
        return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def reconstruct_path(self, came_from, current):
        path = []
        while current:
            x, y = current
            real_x = (x - self.grid_size / 2) * self.grid_resolution
            real_y = (y - self.grid_size / 2) * self.grid_resolution
            path.append((real_x, real_y))
            current = came_from[current]

        # Remove unnecessary initial points
        if path:
            path.pop()
            path.pop()

        return path[::-1] # Return reversed path



def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
