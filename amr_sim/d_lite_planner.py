import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from queue import PriorityQueue
import numpy as np

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
        self.grid = None
        self.resolution = 0.5  # Grid resolution in meters
        self.grid_origin = None  # Represents the origin of the world (0,0) in grid coordinates
        self.planning_active = True
        self.priority_queue = PriorityQueue()
        self.visited = set()
        self.g_values = {}
        self.rhs_values = {}
        self.initialized = False

    def odom_callback(self, msg):
        self.robot_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.check_crosswalk_proximity()

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


    def initialize_grid_values(self):
        # """Initialize grid values for all nodes in the grid."""
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                node = (y, x)
                self.g_values[node] = float('inf')
                self.rhs_values[node] = float('inf')
        # Initialize the start and goal nodes explicitly
        self.rhs_values[self.world_to_grid(self.goal)] = 0


    def initialize(self):
        goal_idx = self.world_to_grid(self.goal)
        start_idx = self.world_to_grid(self.robot_pose)
        self.rhs_values[goal_idx] = 0
        self.priority_queue.put((self.calculate_key(goal_idx), goal_idx))
        self.g_values[goal_idx] = float('inf')
        self.g_values[start_idx] = float('inf')

    def calculate_key(self, node):
        heuristic_cost = self.heuristic(node, self.world_to_grid(self.robot_pose))
        return (min(self.g_values.get(node, float('inf')), self.rhs_values.get(node, float('inf'))) + heuristic_cost,
                min(self.g_values.get(node, float('inf')), self.rhs_values.get(node, float('inf'))))

    def compute_shortest_path(self):
        while not self.priority_queue.empty():
            top_key, top_node = self.priority_queue.get()
            if self.g_values[top_node] > self.rhs_values[top_node]:
                self.g_values[top_node] = self.rhs_values[top_node]
                for neighbor in self.get_neighbors(top_node):
                    self.update_vertex(neighbor)
            else:
                self.g_values[top_node] = float('inf')
                self.update_vertex(top_node)
                for neighbor in self.get_neighbors(top_node):
                    self.update_vertex(neighbor)

    def update_vertex(self, node):
        if node != self.world_to_grid(self.robot_pose):
            self.rhs_values[node] = min([self.cost(node, neighbor) + self.g_values[neighbor] for neighbor in self.get_neighbors(node)])
        if node in self.visited:
            self.visited.remove(node)
        if self.g_values.get(node, float('inf')) != self.rhs_values.get(node):
            self.priority_queue.put((self.calculate_key(node), node))
            self.visited.add(node)

    
    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def get_neighbors(self, node):
        # Assuming 8-directional movement
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        neighbors = []
        for dy, dx in directions:
            neighbor = (node[0] + dy, node[1] + dx)
            if 0 <= neighbor[0] < self.grid.shape[0] and 0 <= neighbor[1] < self.grid.shape[1] and self.grid[neighbor] != 100:
                neighbors.append(neighbor)
        return neighbors
    
    def cost(self, from_node, to_node):
        if self.grid[to_node] == 100:
            return float('inf')
        return np.linalg.norm(np.array(from_node) - np.array(to_node))


    def occupancy_grid_callback(self, msg):
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.grid_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])        
        if not self.initialized:
            self.initialize_grid_values()
            self.initialize()
            self.initialized = True
        self.plan_path()

    def world_to_grid(self, world_pos):
        return (int(world_pos[1] - self.grid_origin[1] / self.resolution), int(world_pos[0] - self.grid_origin[0] / self.resolution))

    def plan_path(self):
        if self.planning_active and self.grid is not None:
            self.compute_shortest_path()
            path = []
            current = self.world_to_grid(self.robot_pose)
            goal = self.world_to_grid(self.goal)
            path = [current]
            while current != goal:
                current = self.get_next_step(current)
                if current is None:
                    break
                path.append(current)

        # Log the path
        self.get_logger().info(f'Path: {path}')

    
    def get_next_step(self, current):
        neighbors = self.get_neighbors(current)
        next_step = None
        min_cost = float('inf')
        for neighbor in neighbors:
            cost = self.g_values.get(neighbor, float('inf')) 
            if cost < min_cost:
                min_cost = cost
                next_step = neighbor
        return next_step
    

    def grid_to_world(self, grid_pos):
        y, x = grid_pos
        world_x = x * self.resolution + self.grid_origin[0]
        world_y = y * self.resolution + self.grid_origin[1]
        return world_x, world_y

def main(args=None):
    rclpy.init(args=args)
    planner = Planner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
