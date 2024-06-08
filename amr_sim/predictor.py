import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from amr_interfaces.msg import PredictionArray, Prediction


class TrackedObject:
    def __init__(self, obj_id, position, timestamp, size):
        self.id = obj_id
        self.positions = [position]
        self.timestamps = [timestamp]
        self.predicted_positions = []
        self.steps_since_seen = 0
        self.size = size

    def update(self, position, timestamp, size):
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.size = size
        # Keep only a maximum of 20 positions
        if len(self.positions) > 20:
            self.positions.pop(0)
            self.timestamps.pop(0)
        self.steps_since_seen = 0
        self.predict_future_positions()

    def predict_future_positions(self):
        if len(self.positions) < 5:
            return None
        
        times = np.array(self.timestamps).reshape(-1, 1)
        positions = np.array(self.positions)

        model_x = LinearRegression().fit(times, positions[:, 0])
        model_y = LinearRegression().fit(times, positions[:, 1])
        future_times = np.linspace(self.timestamps[-1], self.timestamps[-1] + 10, num=10).reshape(-1, 1)
        future_x = model_x.predict(future_times)
        future_y = model_y.predict(future_times)

        # If movement along one of the axis is way more significant than the other, keep the prediction for that axis and replace the other with the last known position constant
        if abs(future_x[-1] - future_x[0]) > 2 * abs(future_y[-1] - future_y[0]):
            future_positions = np.column_stack((future_x, np.full_like(future_x, self.positions[-1][1])))
        elif abs(future_y[-1] - future_y[0]) > 2 * abs(future_x[-1] - future_x[0]):
            future_positions = np.column_stack((np.full_like(future_y, self.positions[-1][0]), future_y))
        else:
            future_positions = np.column_stack((future_x, future_y))

        self.predicted_positions = future_positions.tolist()
        
        

class Predictor(Node):
    def __init__(self):
        super().__init__('local_planner')
        self.create_subscription(OccupancyGrid, '/occupancy_grid', self.occupancy_grid_callback, 10)
        self.publisher_ = self.create_publisher(PredictionArray, '/predictions', 10)
        self.grid = None
        self.resolution = 0.5
        self.grid_origin = None
        self.tracked_objects = []
        self.object_id = 0

        # plt.ion()
        # self.fig, self.ax = plt.subplots()
        # self.scatter = None


    def occupancy_grid_callback(self, msg):
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.grid_origin = np.array([msg.info.width / 2, msg.info.height / 2])
        
        occupied_indices = np.argwhere(self.grid == 100)
        if occupied_indices.size == 0:
            return
        
        clustering = DBSCAN(eps=2.0, min_samples=3).fit(occupied_indices)
        unique_labels = np.unique(clustering.labels_)
        current_time = rclpy.clock.Clock().now().nanoseconds / 1e9

        for label in unique_labels:
            if label == -1:
                continue
            cluster_indices = occupied_indices[clustering.labels_ == label]
            cluster_center = np.mean(cluster_indices, axis=0)
            world_x, world_y = (cluster_center[1] - self.grid_origin[1]) * self.resolution, (cluster_center[0] - self.grid_origin[0]) * self.resolution
            position = np.array([world_x, world_y])

            # Estimate the size of the object along both axes and take the larger of the two
            pca = PCA(n_components=2)
            pca.fit(cluster_indices)
            size = np.max(pca.explained_variance_)
            size = int(size * self.resolution)
            # self.get_logger().info(f'Object with size {size}')

            
            matched = False
            for obj in self.tracked_objects:
                if euclidean(position, obj.positions[-1]) < 1.0:
                    obj.update(position, current_time, size)
                    matched = True
                    break

            if not matched:
                self.tracked_objects.append(TrackedObject(self.object_id, position, current_time, size))
                self.object_id += 1

        # Remove objects that haven't been seen for a while
        for obj in self.tracked_objects:
            obj.steps_since_seen += 1
            if obj.steps_since_seen > 10:
                self.tracked_objects.remove(obj)

        # self.update_plot()

        # If the distance between an object's current position and its last predicted position is greater than 1.0 meters, publish an array of predictions
        predictions = PredictionArray()
        for obj in self.tracked_objects:
            if len(obj.predicted_positions) < 2:
                continue
            if euclidean(obj.positions[-1], obj.predicted_positions[-1]) > 2.0:
                prediction = Prediction()
                prediction.current_x = obj.positions[-1][0]
                prediction.current_y = obj.positions[-1][1]
                prediction.pred_x = obj.predicted_positions[-1][0]
                prediction.pred_y = obj.predicted_positions[-1][1]
                prediction.size = obj.size
                predictions.predictions.append(prediction)
        self.publisher_.publish(predictions)

        

    # def update_plot(self):
    #     self.ax.clear()
    #     positions = [obj.positions[-1] for obj in self.tracked_objects]
    #     predicted_positions = [pos for obj in self.tracked_objects for pos in obj.predicted_positions]
        
    #     if positions:
    #         x, y = zip(*positions)
    #         self.ax.scatter(x, y, c='blue', label='Current Positions')
        
    #     if predicted_positions:
    #         px, py = zip(*predicted_positions)
    #         self.ax.scatter(px, py, c='red', marker='x', label='Predicted Positions')
    #     self.ax.set_xlim(-30, 50)
    #     self.ax.set_ylim(-30, 50)
    #     self.ax.set_xlabel('X')
    #     self.ax.set_ylabel('Y')
    #     self.ax.set_title('Tracked Objects')
    #     plt.draw()
    #     plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    local_planner = Predictor()
    rclpy.spin(local_planner)
    local_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
