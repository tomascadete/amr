# Context: This script handles a road crossing situation for an autonomous robot. The robot stops at (11.5, 37.0),
# which marks the entry of the crosswalk. The goal is to use this separate node to manage the crossing process.
# Once the road is crossed, the robot will continue functioning with an updated final goal, and this node will
# not run anymore.
#
# Implementation: The crossing decision is based on a semaphore's light state, detected using a YOLOv8 model,
# and vehicle movement, monitored by clustering techniques applied on the occupancy grid map. The node will assess if the semaphore
# is green and if the vehicles on the road, within world coordinates x=13 to x=28 (same for the grid map), are stopped or moving slowly
# enough to ensure safe crossing. This script will only focus on this specific interval of the occupancy grid map (the road)
# to determine the safety of crossing.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np


class CrossingNode(Node):
    def __init__(self):
        super().__init__('crossing_node')
        self.bridge = CvBridge()
        self.ryg_model = YOLO("semaphorelight.pt")
        self.subscription_image = self.create_subscription(Image, '/camera/image_raw', self.greenlight_callback, 10)
        self.subscription_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.subscription_occ_grid = self.create_subscription(OccupancyGrid, '/occupancy_grid', self.occupancy_grid_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.robot_pose = None
        self.grid = None
        self.grid_origin = None
        self.crossing = False


    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose
        
        # If the robot is between x=11 and x=28.5, and y=31.5 and y=42, enable the crossing logic throughout the node
        if 11 <= self.robot_pose.position.x <= 28.5 and 31.5 <= self.robot_pose.position.y <= 42:
            self.crossing = True
        else:
            self.crossing = False



    def occupancy_grid_callback(self, msg):
        if self.crossing:
            self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            self.grid_origin = np.array([msg.info.width / 2, msg.info.height / 2])

        




def main(args=None):
    rclpy.init(args=args)
    node = CrossingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()