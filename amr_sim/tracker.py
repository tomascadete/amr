import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import numpy as np
from collections import defaultdict

class Tracker(Node):
    def __init__(self):
        super().__init__('tracker')
        self.subscription = self.create_subscription(
            String,
            '/detected_objects_with_id',
            self.object_callback,
            10)
        self.prev_positions = defaultdict(lambda: None)
        self.velocities = defaultdict(lambda: np.array([0.0, 0.0]))

    def object_callback(self, msg):
        # Parse the JSON string
        detected_objects = json.loads(msg.data)
        current_positions = {}
        for obj in detected_objects:
            obj_id = obj["id"]
            centroid = np.array(obj["centroid"])
            current_positions[obj_id] = centroid
            
            # Check if we have a previous position for this object
            if obj_id in self.prev_positions and self.prev_positions[obj_id] is not None:
                # Calculate velocity based on the change in position
                velocity = centroid - self.prev_positions[obj_id]
                self.velocities[obj_id] = velocity
                # Log current position and velocity
                self.get_logger().info(f'Object ID: {obj_id}, Position: {centroid.tolist()}, Velocity: {velocity.tolist()}')
            else:
                # This is a new object we haven't seen before
                self.velocities[obj_id] = np.array([0.0, 0.0])
                self.get_logger().info(f'Object ID: {obj_id}, Position: {centroid.tolist()}, Velocity: [0.0, 0.0]')
            
            # Predict future position using the current velocity
            future_position = centroid + self.velocities[obj_id]
            self.get_logger().info(f'Predicted future position for Object ID {obj_id}: {future_position.tolist()}')
        
        # Update the dictionary of previous positions
        self.prev_positions = current_positions


def main(args=None):
    rclpy.init(args=args)
    tracker = Tracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
