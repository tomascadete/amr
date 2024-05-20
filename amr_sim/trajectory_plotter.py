import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt

class TrajectoryPlotter(Node):
    def __init__(self):
        super().__init__('trajectory_tracker')
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.trajectory = []
        self.threshold = 0.1  # 0.1 meters
        self.setup_plot()

    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-30, 50)
        self.ax.set_ylim(-20, 50)
        self.ax.set_xlabel('Y')
        self.ax.set_ylabel('X')
        self.ax.set_title('Robot Trajectory')
        self.ax.invert_xaxis()  # Invert the X axis to show 50 on the left and -30 on the right
        self.line, = self.ax.plot([], [], 'b-')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def odom_callback(self, msg):
        current_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        if not self.trajectory or np.linalg.norm(current_position - self.trajectory[-1]) > self.threshold:
            self.trajectory.append(current_position)
            self.update_plot()

    def update_plot(self):
        self.line.set_xdata([p[1] for p in self.trajectory])  # Use Y values for x-axis
        self.line.set_ydata([p[0] for p in self.trajectory])  # Use X values for y-axis
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    trajectory_tracker = TrajectoryPlotter()
    rclpy.spin(trajectory_tracker)
    trajectory_tracker.destroy_node()
    rclpy.shutdown()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
