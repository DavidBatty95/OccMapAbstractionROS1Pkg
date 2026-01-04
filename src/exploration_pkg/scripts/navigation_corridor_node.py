#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List, Optional, Tuple

import rospy
import actionlib

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, Float32

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def now_s() -> float:
    return rospy.Time.now().to_sec()


def dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def yaw_to_quat(yaw: float):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.z = math.sin(0.5 * yaw)
    q.w = math.cos(0.5 * yaw)
    return q


class CorridorNavigator:
    def __init__(self):
        rospy.init_node("navigation_corridor", anonymous=False)

        # Frames
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # Topics
        self.goal_topic = rospy.get_param("~goal_topic", "/exploration/goal")
        self.corridor_topic = rospy.get_param("~corridor_topic", "/exploration/corridor")

        self.publish_markers = bool(rospy.get_param("~publish_markers", True))
        self.marker_topic = rospy.get_param("~marker_topic", "/navigation_markers")

        # move_base
        self.move_base_action = rospy.get_param("~move_base_action", "/move_base")

        # Waypoint tolerance
        self.waypoint_reached_m = float(rospy.get_param("~waypoint_reached_m", 0.55))
        self.final_goal_reached_m = float(rospy.get_param("~final_goal_reached_m", 0.85))
        self.min_send_period_s = float(rospy.get_param("~min_send_period_s", 0.8))

        # Waypoint skipping
        self.skip_ahead_margin_m = float(rospy.get_param("~skip_ahead_margin_m", 0.15))

        # Corridor processing
        self.drop_near_robot_waypoints = bool(rospy.get_param("~drop_near_robot_waypoints", True))
        self.trim_start_dist_m = float(rospy.get_param("~trim_start_dist_m", 0.75))
        self.resample_min_dist = float(rospy.get_param("~resample_min_dist", 0.8))
        self.resample_max_dist = float(rospy.get_param("~resample_max_dist", 2.6))
        self._inv_resample_max = 1.0 / max(self.resample_max_dist, 1e-6)

        # Nav feedback topics
        self.pub_nav_active = rospy.Publisher("/navigation/nav_active", Bool, queue_size=10, latch=True)
        self.pub_nav_goal_reached = rospy.Publisher("/navigation/nav_goal_reached", Bool, queue_size=10, latch=True)
        self.pub_nav_progress = rospy.Publisher("/navigation/nav_progress", Float32, queue_size=10)

        self.pub_markers = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1) if self.publish_markers else None

        # Internal state
        self._corridor: Optional[Path] = None
        self._goal: Optional[PoseStamped] = None

        self._waypoints: List[PoseStamped] = []
        self._waypoints_xy: List[Tuple[float, float]] = []
        self._active_idx = 0

        self._last_send_t = 0.0

        # TF
        import tf2_ros
        self.tf_buf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        self._robot_xy_cache: Optional[Tuple[float, float]] = None
        self._robot_xy_cache_t = 0.0

        # move_base client
        self.client = actionlib.SimpleActionClient(self.move_base_action, MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server: %s", self.move_base_action)
        self.client.wait_for_server()
        rospy.loginfo("Navigation Corridor ready (optimised, behaviour-identical).")

        # Subscribers
        rospy.Subscriber(self.corridor_topic, Path, self._on_corridor, queue_size=1)
        rospy.Subscriber(self.goal_topic, PoseStamped, self._on_goal, queue_size=1)

        rospy.Timer(rospy.Duration(0.10), self._tick)

    # ------------------------------------------------------------------

    def _get_robot_xy(self) -> Optional[Tuple[float, float]]:
        now = now_s()
        if self._robot_xy_cache and (now - self._robot_xy_cache_t) < 0.05:
            return self._robot_xy_cache

        try:
            tr = self.tf_buf.lookup_transform(
                self.map_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(0.2),
            )
            xy = (float(tr.transform.translation.x), float(tr.transform.translation.y))
            self._robot_xy_cache = xy
            self._robot_xy_cache_t = now
            return xy
        except Exception:
            return None

    def _on_goal(self, msg: PoseStamped):
        self._goal = msg

    def _on_corridor(self, msg: Path):
        self._corridor = msg
        self._prepare_waypoints()

    # ------------------------------------------------------------------

    def _prepare_waypoints(self):
        if not self._corridor or not self._corridor.poses:
            return

        wps = list(self._corridor.poses)
        robot = self._get_robot_xy()

        if robot and self.drop_near_robot_waypoints:
            trim2 = self.trim_start_dist_m ** 2
            while wps and dist2(
                (wps[0].pose.position.x, wps[0].pose.position.y),
                robot
            ) <= trim2:
                wps.pop(0)

        if not wps:
            wps = [self._goal] if self._goal else [self._corridor.poses[-1]]

        wps = self._resample_waypoints(wps)

        self._waypoints = wps
        self._waypoints_xy = [
            (p.pose.position.x, p.pose.position.y) for p in wps
        ]
        self._active_idx = 0

        rospy.loginfo("[CORRIDOR] %d waypoints prepared.", len(self._waypoints))

    def _resample_waypoints(self, wps: List[PoseStamped]) -> List[PoseStamped]:
        if len(wps) <= 1:
            return wps

        out = [wps[0]]
        for p in wps[1:]:
            ax, ay = out[-1].pose.position.x, out[-1].pose.position.y
            bx, by = p.pose.position.x, p.pose.position.y
            d = math.hypot(bx - ax, by - ay)

            if d < self.resample_min_dist:
                continue

            if d > self.resample_max_dist:
                n = int(math.ceil(d * self._inv_resample_max))
                for i in range(1, n):
                    t = float(i) / float(n)
                    q = PoseStamped()
                    q.header = p.header
                    q.pose.position.x = ax + t * (bx - ax)
                    q.pose.position.y = ay + t * (by - ay)
                    q.pose.orientation.w = 1.0
                    out.append(q)

            out.append(p)

        return out

    # ------------------------------------------------------------------

    def _send_goal(self, idx: int):
        wp = self._waypoints[idx]
        g = MoveBaseGoal()
        g.target_pose.header.stamp = rospy.Time.now()
        g.target_pose.header.frame_id = self.map_frame
        g.target_pose.pose.position = wp.pose.position

        if idx < len(self._waypoints) - 1:
            nx, ny = self._waypoints_xy[idx + 1]
            wx, wy = self._waypoints_xy[idx]
            yaw = math.atan2(ny - wy, nx - wx)
            g.target_pose.pose.orientation = yaw_to_quat(yaw)
        else:
            g.target_pose.pose.orientation = wp.pose.orientation

        self.client.send_goal(g)
        self._last_send_t = now_s()
        self.pub_nav_active.publish(Bool(data=True))
        self.pub_nav_goal_reached.publish(Bool(data=False))

    # ------------------------------------------------------------------

    def _tick(self, _evt):
        if not self._waypoints:
            self.pub_nav_active.publish(Bool(data=False))
            return

        robot = self._get_robot_xy()
        if robot is None:
            self.pub_nav_active.publish(Bool(data=False))
            return

        # Skip-ahead logic
        while self._active_idx + 1 < len(self._waypoints_xy):
            cx, cy = self._waypoints_xy[self._active_idx]
            nx, ny = self._waypoints_xy[self._active_idx + 1]
            d_curr = math.hypot(cx - robot[0], cy - robot[1])
            d_next = math.hypot(nx - robot[0], ny - robot[1])
            if d_next + self.skip_ahead_margin_m < d_curr:
                self._active_idx += 1
            else:
                break

        wx, wy = self._waypoints_xy[self._active_idx]
        d = math.hypot(wx - robot[0], wy - robot[1])

        tol = self.final_goal_reached_m if self._active_idx == len(self._waypoints_xy) - 1 else self.waypoint_reached_m
        if d <= tol:
            if self._active_idx == len(self._waypoints_xy) - 1:
                self.pub_nav_goal_reached.publish(Bool(data=True))
                self.pub_nav_active.publish(Bool(data=False))
                self._waypoints.clear()
                self._waypoints_xy.clear()
                rospy.loginfo("[NAV] final goal reached.")
                return
            self._active_idx += 1
            return

        if (now_s() - self._last_send_t) >= self.min_send_period_s:
            state = self.client.get_state()
            if state not in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
                self._send_goal(self._active_idx)
                self._publish_markers(wx, wy)

    # ------------------------------------------------------------------

    def _publish_markers(self, wx: float, wy: float):
        if self.pub_markers is None:
            return
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = rospy.Time.now()
        m.ns = "nav"
        m.id = 1
        m.type = Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = 0.20
        m.pose.position.x = wx
        m.pose.position.y = wy
        m.pose.orientation.w = 1.0
        m.color.r = 1.0
        m.color.g = 0.8
        m.color.b = 0.1
        m.color.a = 0.95
        ma.markers.append(m)
        self.pub_markers.publish(ma)


if __name__ == "__main__":
    try:
        CorridorNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass