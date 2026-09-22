#!/usr/bin/env python3
"""Headless 2D reference simulator for the original ROS exploration policy.

Holonomic constant-speed execution, not Gazebo or a robot dynamics model.
Truth is used only for sensing, collision checking and scoring; the policy gets
raycast occupancy observations and a point radiation measurement.
"""
import json
import math
from pathlib import Path as FilePath
import threading

import numpy as np
import rospkg
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool, Float32, String
from scipy.ndimage import label

from occmap.grid import clearance, line_cells, load_map, segment_free


class ResearchDemo:
    def __init__(self):
        rospy.init_node('research_demo')
        self.lock = threading.RLock()
        name = rospy.get_param('~map', 'reactor_room_sim')
        root = FilePath(rospkg.RosPack().get_path('exploration_pkg')) / 'maps'
        if name not in {p.name for p in root.iterdir() if p.is_dir()}:
            raise ValueError('Unknown bundled map: ' + name)
        self.map = load_map(root / name, int(rospy.get_param('~factor', 5)))
        self.radius = float(rospy.get_param('~robot_radius_m', 0.12))
        self.speed = float(rospy.get_param('~speed_mps', 0.6))
        self.range = float(rospy.get_param('~sensor_range_m', 3.5))
        self.dt = 0.1
        # Include a half-cell diagonal so centre distances cover obstacle cells.
        self.blocked = clearance(self.map.occupancy != 0) * self.map.resolution <= (
            self.radius + self.map.resolution / math.sqrt(2))
        components, _ = label(~self.blocked)
        counts = np.bincount(components.ravel())
        counts[0] = 0
        if counts.max() == 0:
            raise ValueError('Map has no traversable space for this footprint')
        eligible = np.argwhere(components == counts.argmax())
        # Deterministic valid start nearest the original map-frame (0, 0).
        world = np.array([self.map.grid_to_world(int(c), int(r)) for r, c in eligible])
        index = np.argmin(np.sum(world ** 2, axis=1))
        self.x, self.y = world[index]
        self.yaw = 0.0
        self.belief = np.full(self.map.occupancy.shape, -1, dtype=np.int8)
        self.waypoints = []
        self.route_count = self.routes_received = self.rejected = 0
        self.distance = self.dose = self.elapsed = 0.0
        self.reached = False
        self.tf = tf2_ros.TransformBroadcaster()
        self.pub_map = rospy.Publisher('/map', OccupancyGrid, queue_size=1, latch=True)
        self.pub_sensor = rospy.Publisher('/sim_rad_sensor', Float32, queue_size=1)
        self.pub_active = rospy.Publisher('/navigation/nav_active', Bool, queue_size=1)
        self.pub_reached = rospy.Publisher('/navigation/nav_goal_reached', Bool, queue_size=1)
        self.pub_progress = rospy.Publisher('/navigation/nav_progress', Float32, queue_size=1)
        self.pub_metrics = rospy.Publisher('/demo/metrics', String, queue_size=1)
        rospy.Subscriber('/exploration/corridor', Path, self.corridor, queue_size=1)
        self.ray_ends = [(round(self.range / self.map.resolution * math.cos(a)),
                          round(self.range / self.map.resolution * math.sin(a)))
                         for a in np.linspace(0, 2 * math.pi, 240, endpoint=False)]
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.tick)

    def corridor(self, msg):
        with self.lock:
            if msg.header.frame_id.strip('/') != 'map' or not msg.poses:
                self.waypoints = []
                return
            points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
            if not np.isfinite(points).all():
                self.rejected += 1
                return
            self.waypoints = points
            self.route_count = len(points)
            self.routes_received += 1
            self.reached = False

    def tick(self, _event):
        try:
            self._tick()
        except rospy.ROSException:
            if not rospy.is_shutdown():
                raise

    def _tick(self):
        with self.lock:
            if rospy.is_shutdown():
                return
            self.elapsed += self.dt
            # Observe first. Unknown ground-truth cells stop rays and remain unknown.
            cx, cy = self.map.world_to_grid(self.x, self.y)
            h, w = self.belief.shape
            for dx, dy in self.ray_ends:
                for x, y in line_cells((cx, cy), (cx + dx, cy + dy)):
                    if not (0 <= x < w and 0 <= y < h):
                        break
                    self.belief[y, x] = self.map.occupancy[y, x]
                    if self.map.occupancy[y, x] != 0:
                        break
            while self.waypoints and math.hypot(self.waypoints[0][0] - self.x, self.waypoints[0][1] - self.y) < 0.025:
                self.waypoints.pop(0)
                if not self.waypoints:
                    self.reached = True
            if self.waypoints:
                tx, ty = self.waypoints[0]
                length = math.hypot(tx - self.x, ty - self.y)
                step = min(self.speed * self.dt, length)
                nx, ny = self.x + step * (tx - self.x) / length, self.y + step * (ty - self.y) / length
                end = self.map.world_to_grid(nx, ny)
                if segment_free(self.blocked | (self.belief != 0), (cx, cy), end):
                    self.yaw = math.atan2(ny - self.y, nx - self.x)
                    self.x, self.y = nx, ny
                    self.distance += step
                else:
                    self.waypoints = []
                    self.rejected += 1
            cx, cy = self.map.world_to_grid(self.x, self.y)
            rate = float(self.map.radiation[cy, cx])
            self.dose += rate * self.dt
            stamp = rospy.Time.now()
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id, transform.child_frame_id = 'map', 'base_link'
            transform.transform.translation.x, transform.transform.translation.y = self.x, self.y
            transform.transform.rotation.z = math.sin(self.yaw / 2)
            transform.transform.rotation.w = math.cos(self.yaw / 2)
            self.tf.sendTransform(transform)
            msg = OccupancyGrid()
            msg.header.stamp, msg.header.frame_id = stamp, 'map'
            msg.info.width, msg.info.height = w, h
            msg.info.resolution = self.map.resolution
            msg.info.origin.position.x, msg.info.origin.position.y = self.map.origin[:2]
            msg.info.origin.orientation.w = 1.0
            msg.data = self.belief.ravel().tolist()
            self.pub_map.publish(msg)
            self.pub_sensor.publish(Float32(rate))
            self.pub_active.publish(Bool(bool(self.waypoints)))
            self.pub_reached.publish(Bool(self.reached))
            progress = 1.0 - len(self.waypoints) / max(1, self.route_count)
            self.pub_progress.publish(Float32(progress))
            self.pub_metrics.publish(String(json.dumps({
                'simulation_time_s': self.elapsed, 'distance_m': self.distance,
                'dose_relative_rate_seconds': self.dose, 'dose_rate_relative': rate,
                'observed_free_fraction': float(np.count_nonzero(self.belief == 0) / np.count_nonzero(self.map.occupancy == 0)),
                'rejected_motion_segments': self.rejected, 'routes_received': self.routes_received,
                'x': self.x, 'y': self.y, 'map': self.map.name,
            }, allow_nan=False)))


if __name__ == '__main__':
    ResearchDemo()
    rospy.spin()
