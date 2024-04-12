import rclpy
from rclpy.node import Node
from amr_interfaces.msg import Object, ObstacleArray
import numpy as np

class TrackedObject:
    def __init__(self, object_id, object_type, position):
        self.id = object_id
        self.type = object_type
        # Store only x and y from the position
        self.positions = [position[:2]]  # Keep x and y, discard z
        self.steps_since_last_seen = 0

class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        self.subscription = self.create_subscription(
            Object,
            '/detections',
            self.detection_callback,
            10)
        self.publisher = self.create_publisher(ObstacleArray, '/tracked_objects', 10)
        self.tracked_objects = []
        self.next_id = 1
        # Max steps missing should be large enough to account for occlusions
        self.max_steps_missing = 20  # Number of steps to keep tracking an object without updates
        self.min_update_distance = 0.1  # Minimum distance to consider an update valid, in meters


    def detection_callback(self, msg):
        detected_position = np.array([msg.x, msg.y])  # Use only x and y
        detected_type = msg.type  # The type of the detected object
        matched = False
        

        # Attempt to match the detected object to an existing tracked object
        for tracked_object in self.tracked_objects:
            if tracked_object.type == detected_type:  # Check if types match
                last_known_position = np.array(tracked_object.positions[-1])
                distance = np.linalg.norm(detected_position - last_known_position)

                # Check if the detected object matches and the distance exceeds the minimum update distance
                if distance < 1.0 and distance > self.min_update_distance:  # Threshold for matching, adjust as necessary
                    # Update the object's position only if the change is significant
                    tracked_object.positions.append(detected_position.tolist())
                    tracked_object.steps_since_last_seen = 0
                    matched = True
                    self.get_logger().info(f'Updated tracked object ID {tracked_object.id} with significant position change.')
                    break
                elif distance <= self.min_update_distance:
                    # Detected change is too small, likely noise, so don't update the position but reset the missing counter
                    tracked_object.steps_since_last_seen = 0
                    matched = True
                    break

        # If no match was found, start tracking a new object
        if not matched and detected_type != 'None':
            new_tracked_object = TrackedObject(self.next_id, detected_type, detected_position.tolist())
            self.tracked_objects.append(new_tracked_object)
            self.get_logger().info(f'Started tracking new object ID {self.next_id} (Type: {detected_type}).')
            self.next_id += 1

        # Increment steps_since_last_seen for all tracked objects and remove any that exceed the limit
        self.tracked_objects = [obj for obj in self.tracked_objects if obj.steps_since_last_seen < self.max_steps_missing]
        for obj in self.tracked_objects:
            obj.steps_since_last_seen += 1

        # Publish the latest position of the tracked objects
        self.publish_tracked_objects()

    def publish_tracked_objects(self):
        msg = ObstacleArray()
        for tracked_object in self.tracked_objects:
            obstacle = Object()
            obstacle.type = tracked_object.type
            # Publish the last known position
            obstacle.x = tracked_object.positions[-1][0]
            obstacle.y = tracked_object.positions[-1][1]
            msg.obstacles.append(obstacle)
        # Log the ammount of tracked objects
        self.get_logger().info(f'Tracked object count: {len(self.tracked_objects)}')
        self.publisher.publish(msg)
        # If no objects are being tracked and the for loop above doesn't run, the message will be empty

def main(args=None):
    rclpy.init(args=args)
    object_tracker = ObjectTracker()
    rclpy.spin(object_tracker)
    object_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
