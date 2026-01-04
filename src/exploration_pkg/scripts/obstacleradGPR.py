#!/usr/bin/env python3

import rospy
import tf
import csv
import os
import numpy as np

from gazebo_radiation_plugins.msg import Simulated_Radiation_Msg
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

################################################################################
# 1) OBSTACLE-AWARE KERNEL
################################################################################

def bresenham_line_points(x0, y0, x1, y1):
    """
    Returns integer grid indices along the line from (x0, y0) to (x1, y1).
    Simple Bresenham implementation for line-of-sight checking.
    """
    points = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0

    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

    return points

def is_line_of_sight_free(xA, yA, xB, yB,
                          map_data, map_width, map_height,
                          origin_x, origin_y, resolution,
                          occupied_thresh=50):
    """
    Check if line-of-sight from (xA, yA) to (xB, yB) crosses an occupied cell in the occupancy map.

    (xA, yA) and (xB, yB) are in continuous "world" coordinates.
    map_data is the list or np.array of occupancy [-1..100], with shape = (map_height, map_width).

    Return True if free, False if blocked.
    """
    # Convert world coords -> map grid indices
    # For Bresenham, we need integer indices in the occupancy grid
    def world_to_map_ix(x_world, y_world):
        mx = int((x_world - origin_x) / resolution)
        my = int((y_world - origin_y) / resolution)
        return mx, my

    (ix0, iy0) = world_to_map_ix(xA, yA)
    (ix1, iy1) = world_to_map_ix(xB, yB)

    # If outside map bounds, treat it as blocked or handle separately
    if not (0 <= ix0 < map_width and 0 <= iy0 < map_height):
        return False
    if not (0 <= ix1 < map_width and 0 <= iy1 < map_height):
        return False

    line_pts = bresenham_line_points(ix0, iy0, ix1, iy1)
    for (cx, cy) in line_pts:
        # Check bounds again (line might step out of bounds)
        if 0 <= cx < map_width and 0 <= cy < map_height:
            # map_data is [row=y, col=x] => map_data[cy, cx]
            val = map_data[cy, cx]
            if val >= occupied_thresh:  # or val == 100 for fully occupied
                return False
        else:
            # If out of bounds, treat as blocked
            return False
    return True

def obstacle_aware_rbf_kernel(X, Y, gamma, 
                              map_data, map_width, map_height,
                              origin_x, origin_y, resolution,
                              occupied_thresh=50):
    """
    Custom kernel function that:
     - Checks line-of-sight between points in X and Y using the occupancy map.
     - If blocked, K=0. If free, K = exp(-gamma * ||x - y||^2).

    :param X: shape [n_samples_X, 2]
    :param Y: shape [n_samples_Y, 2]
    :param gamma: 1 / (2 * length_scale^2) in usual RBF terms
    """
    K = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            xA, yA = X[i, 0], X[i, 1]
            xB, yB = Y[j, 0], Y[j, 1]

            # Check line-of-sight
            los_free = is_line_of_sight_free(
                xA, yA, xB, yB,
                map_data, map_width, map_height,
                origin_x, origin_y, resolution,
                occupied_thresh
            )
            if los_free:
                dist_sq = (xA - xB)**2 + (yA - yB)**2
                K[i, j] = np.exp(-gamma * dist_sq)
            else:
                K[i, j] = 0.0

    return K

################################################################################
# 2) THE ROS NODE
################################################################################

from sklearn.base import BaseEstimator, TransformerMixin

class ObstacleAwareKernelTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn "transformer" that, given X (training or test),
    produces the kernel matrix K(X, self.X_fit_) or K(X, X).

    We store the "training points" in self.X_fit_ after .fit(X).
    Then .transform(X) returns K(X, self.X_fit_).

    This is so we can feed a "precomputed" kernel to Nystroem.
    However, Nystroem doesn't fully support 'precomputed' by default.
    We'll manually build the kernel matrix and pass that into Nystroem's .fit
    or we do a 2-step approach below.
    """
    def __init__(self, gamma=1.0, map_data=None, map_width=0, map_height=0,
                 origin_x=0.0, origin_y=0.0, resolution=1.0, occupied_thresh=50):
        self.gamma = gamma
        self.map_data = map_data
        self.map_width = map_width
        self.map_height = map_height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.resolution = resolution
        self.occupied_thresh = occupied_thresh

    def fit(self, X, y=None):
        self.X_fit_ = X
        return self

    def transform(self, X):
        return obstacle_aware_rbf_kernel(
            X, 
            self.X_fit_, 
            self.gamma,
            self.map_data,
            self.map_width,
            self.map_height,
            self.origin_x,
            self.origin_y,
            self.resolution,
            self.occupied_thresh
        )

class SparseGPRNode:
    def __init__(self):
        rospy.init_node('sparse_gpr_node', anonymous=True)

        # ROS Parameters
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.radiation_sensor_topic = rospy.get_param('~radiation_sensor_topic', '/radiation_sensor_plugin/sensor_0')
        self.map_topic = rospy.get_param('~map_topic', '/map')
        self.radiation_map_topic = rospy.get_param('~radiation_map_topic', '/radiation_map')
        self.radiation_marker_topic = rospy.get_param('~radiation_marker_topic', '/radiation_markers')
        self.log_file_path = rospy.get_param('~log_file_path', "/tmp/radiation_log.csv")
        self.reading_rate = rospy.get_param('~reading_rate', 0.5)

        # Threshold for normalizing the final OccupancyGrid
        self.radiation_threshold = rospy.get_param('~radiation_threshold', 10.0)
        self.max_radiation = rospy.get_param('~max_radiation', 200.0)

        # "Gamma" = 1 / (2 * length_scale^2)
        length_scale = rospy.get_param('~length_scale', 5.0)
        self.gamma = 1.0 / (2.0 * (length_scale ** 2))

        self.min_distance = rospy.get_param('~min_distance', 0.5)
        self.max_training_points = rospy.get_param('~max_training_points', 1000)
        self.n_components = rospy.get_param('~n_components', 200)  # number of basis functions in Nystroem
        self.retrain_interval = rospy.get_param('~retrain_interval', 10)  # retrain after this many new readings
        self.occupied_thresh = rospy.get_param('~occupied_thresh', 50)  # occupancy value considered a wall

        self.marker_pub = rospy.Publisher(self.radiation_marker_topic, Marker, queue_size=10)
        self.radiation_map_pub = rospy.Publisher(self.radiation_map_topic, OccupancyGrid, queue_size=1)
        self.listener = tf.TransformListener()

        self.latest_radiation_level = None
        self.marker_id = 0
        self.reading_count = 0
        self.map_received = False
        self.map_info = None
        self.reading_positions = []

        rospy.Subscriber(self.radiation_sensor_topic, Simulated_Radiation_Msg, self.radiation_callback)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)

        # Prepare logging
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        self.log_file = open(self.log_file_path, 'a', newline='')
        self.csv_writer = csv.writer(self.log_file)
        if os.stat(self.log_file_path).st_size == 0:
            self.csv_writer.writerow(['Timestamp', 'Radiation_Level', 'X_Position', 'Y_Position'])

        # We will store the data points:
        self.X_train = []
        self.y_train = []

        # We build a scikit-learn pipeline:
        # 1. "ObstacleAwareKernelTransformer": builds the NxN kernel matrix
        # 2. "Nystroem": approximates that kernel with n_components features
        # 3. "Ridge" regression in that feature space.
        # 
        # Because Nystroem doesn't have a native "precomputed kernel" mode
        # that automatically calls a custom function, we do a 2-step approach:
        #
        # Step A: We store X in "ObstacleAwareKernelTransformer" and use it to
        #         produce K(X, X). Then we manually call nystroem.fit(Kxx).
        # Step B: For predictions, we compute K(X*, X), then nystroem.transform(K(X*, X)).
        #
        # To keep it simpler for the example, we wrap it in a custom fit/predict routine.

        self.n_components = max(2, self.n_components)  # ensure >= 2

        # The pipeline's last step:
        self.regressor = Ridge(alpha=1.0)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.reading_rate), self.timer_callback)

    def map_callback(self, msg):
        if not self.map_received:
            self.map_info = msg.info
            self.map_width = self.map_info.width
            self.map_height = self.map_info.height
            self.map_resolution = self.map_info.resolution
            self.map_origin = self.map_info.origin

            # Convert map.data (1D) => 2D array [row=y, col=x]
            # OccupancyGrid.data is in row-major order,
            # meaning data[0] is the first cell in the top-left corner of the map
            # (or bottom-left, depending on your convention).
            # We'll assume it's top-left => row=0, col=0 is top-left
            # You may need to invert if your map is stored differently.
            map_1d = np.array(msg.data, dtype=np.int8).reshape(self.map_height, self.map_width)
            self.map_data = map_1d  # store as is, but be mindful of indexing

            self.x_min = self.map_origin.position.x
            self.y_min = self.map_origin.position.y
            self.x_max = self.x_min + self.map_width * self.map_resolution
            self.y_max = self.y_min + self.map_height * self.map_resolution

            self.map_received = True
            rospy.loginfo("Map received. Grid size: {}x{}, Resolution: {}".format(
                self.map_width, self.map_height, self.map_resolution))

    def radiation_callback(self, msg):
        self.latest_radiation_level = msg.value
        self.reading_count += 1

    def find_nearby_reading_index(self, x, y):
        if not self.reading_positions:
            return None
        positions = np.array(self.reading_positions)
        dists = np.sqrt(np.sum((positions - np.array([x, y]))**2, axis=1))
        min_dist = np.min(dists)
        if min_dist < self.min_distance:
            return int(np.argmin(dists))
        return None

    def timer_callback(self, event):
        if not self.map_received or self.latest_radiation_level is None:
            return

        try:
            self.listener.waitForTransform(self.map_frame, "base_link", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform(self.map_frame, "base_link", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("TF Error: Unable to get robot's position.")
            return

        # Current robot location in map frame
        px, py = trans[0], trans[1]

        idx = self.find_nearby_reading_index(px, py)
        if idx is None:
            # Log a new reading
            t = rospy.get_time()
            self.csv_writer.writerow([t, self.latest_radiation_level, px, py])
            self.log_file.flush()

            self.X_train.append([px, py])
            self.y_train.append(self.latest_radiation_level)
            self.reading_positions.append([px, py])

            # Truncate if too large
            if len(self.X_train) > self.max_training_points:
                self.X_train.pop(0)
                self.y_train.pop(0)
                self.reading_positions.pop(0)
        else:
            # Update existing reading
            existing_pos = self.reading_positions[idx]
            for i, pos in enumerate(self.X_train):
                if pos == existing_pos:
                    self.y_train[i] = self.latest_radiation_level
                    break

        # Only re-fit occasionally
        if self.reading_count % self.retrain_interval == 0 and len(self.X_train) > 1:
            rospy.loginfo("Refitting approximate GP (Nystroem + Ridge) with {} points...".format(len(self.X_train)))

            X = np.array(self.X_train)
            y = np.array(self.y_train)

            # === A) Compute K(X, X) with obstacle awareness:
            Kxx = obstacle_aware_rbf_kernel(
                X, X, self.gamma,
                self.map_data,
                self.map_width,
                self.map_height,
                self.map_origin.position.x,
                self.map_origin.position.y,
                self.map_resolution,
                self.occupied_thresh
            )

            # === B) Fit a Nystroem feature map on K(X, X):
            self.nystroem_ = Nystroem(
                kernel='precomputed',
                n_components=self.n_components,
                random_state=42
            )
            # For 'precomputed' kernel in Nystroem, we pass Kxx to .fit
            # Then .fit_transform also needs Kxx
            Zx = self.nystroem_.fit_transform(Kxx)

            # === C) Fit a standard ridge regressor in the approximate feature space
            self.regressor.fit(Zx, y)

            rospy.loginfo("Refit done.")

            # Predict across the entire map for the OccupancyGrid
            self.publish_radiation_map()

        # Publish a marker for visualization
        self.publish_radiation_marker(px, py, rot)

        rospy.loginfo_throttle(10, "Total readings: {}".format(self.reading_count))

    def predict_radiation(self, X_star):
        """
        Predict radiation at query points X_star using the approximate kernel model.
        X_star: shape [N, 2]
        Returns: predicted mean radiation, shape [N]
        """
        if not hasattr(self, 'nystroem_'):
            # Not fitted yet
            return np.zeros(len(X_star))

        X = np.array(self.X_train)

        # 1) Compute K(X*, X)
        Kxstar_x = obstacle_aware_rbf_kernel(
            X_star, X, self.gamma,
            self.map_data,
            self.map_width,
            self.map_height,
            self.map_origin.position.x,
            self.map_origin.position.y,
            self.map_resolution,
            self.occupied_thresh
        )

        # 2) Transform using the fitted Nystroem
        Zxstar = self.nystroem_.transform(Kxstar_x)

        # 3) Predict using the regressor
        y_pred = self.regressor.predict(Zxstar)
        return y_pred

    def publish_radiation_map(self):
        """
        Predict radiation at each cell in the map and publish as an OccupancyGrid.
        """
        # Build a mesh of all map cells
        # [row y in [0..map_height], col x in [0..map_width]]
        # Convert cell indices to map frame coordinates
        coords = []
        for row in range(self.map_height):
            wy = self.map_origin.position.y + row * self.map_resolution
            for col in range(self.map_width):
                wx = self.map_origin.position.x + col * self.map_resolution
                coords.append([wx, wy])

        coords = np.array(coords)
        y_pred = self.predict_radiation(coords)

        # Reshape back into map_height x map_width
        y_pred_2d = y_pred.reshape(self.map_height, self.map_width)

        # Simple threshold normalization for OccupancyGrid
        y_normalized = np.where(
            y_pred_2d >= self.radiation_threshold,
            np.clip(
                (y_pred_2d - self.radiation_threshold) / (self.max_radiation - self.radiation_threshold),
                0.0, 1.0
            ),
            0.0
        )

        # Convert to int8 occupancy range [0..100)
        scaled_data = (y_normalized * 100).astype(np.int8)
        scaled_data = np.clip(scaled_data, 0, 100)

        # Publish
        rad_map = OccupancyGrid()
        rad_map.header.stamp = rospy.Time.now()
        rad_map.header.frame_id = self.map_frame
        rad_map.info = self.map_info
        # Flatten in row-major order
        rad_map.data = scaled_data.flatten().tolist()

        self.radiation_map_pub.publish(rad_map)

    def publish_radiation_marker(self, px, py, rot):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "radiation_markers"
        marker.id = self.marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(px, py, 0.0)
        marker.pose.orientation.x = rot[0]
        marker.pose.orientation.y = rot[1]
        marker.pose.orientation.z = rot[2]
        marker.pose.orientation.w = rot[3]
        marker.scale.x = marker.scale.y = marker.scale.z = 0.3

        if self.latest_radiation_level < 5.0:
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
        elif self.latest_radiation_level < 10.0:
            marker.color = ColorRGBA(1.0, 1.0, 0.0, 0.8)
        else:
            marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)

        marker.lifetime = rospy.Duration(0)
        self.marker_pub.publish(marker)
        self.marker_id += 1

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = SparseGPRNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
