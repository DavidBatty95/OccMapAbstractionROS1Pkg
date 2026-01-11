#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exploration_policy_node.py (v20-map-frame + mission_state + rad_avoid_max) — COMPLETE DROP-IN

Goal of this revision:
- Keep *all* your current behaviour and interfaces (topics/params/state/markers/telemetry)
- Increase “avoid radiation as much as feasibly possible” without breaking coverage:
  - Adds optional radiation hard/soft caps for candidate goals (with connectivity-safe fallback)
  - Makes A* truly radiation-aware by integrating radiation along edges (not just midpoint)
  - Adds optional "radiation exclusion" inflation (softly) with automatic fallback if it would stall exploration

ROS Noetic + Python3.
"""

import math
import time
import json
import threading
import heapq
from collections import deque, defaultdict

import numpy as np

import rospy
from std_msgs.msg import Bool, Float32, String
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

# TF2
import tf2_ros
from tf2_geometry_msgs import PoseStamped as TfPoseStamped

try:
    from tf.transformations import euler_from_quaternion, quaternion_from_euler
except Exception:
    def euler_from_quaternion(q):
        x, y, z, w = q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (0.0, 0.0, yaw)

    def quaternion_from_euler(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return (x, y, z, w)

try:
    from scipy.spatial import cKDTree
    _HAVE_KDTREE = True
except Exception:
    cKDTree = None
    _HAVE_KDTREE = False


# -------------------------- helpers -------------------------- #

def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def hypot2(dx, dy):
    return math.sqrt(dx * dx + dy * dy)

def world_to_grid(x, y, origin_x, origin_y, res):
    gx = int((x - origin_x) / res)
    gy = int((y - origin_y) / res)
    return gx, gy

def grid_to_world(gx, gy, origin_x, origin_y, res):
    x = origin_x + (gx + 0.5) * res
    y = origin_y + (gy + 0.5) * res
    return x, y

def grid_to_world_vec(gx, gy, origin_x, origin_y, res):
    gx = np.asarray(gx, dtype=np.float32)
    gy = np.asarray(gy, dtype=np.float32)
    x = origin_x + (gx + 0.5) * res
    y = origin_y + (gy + 0.5) * res
    return x, y

def bresenham(x0, y0, x1, y1):
    x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

def fast_chamfer_distance(obstacle_mask):
    """
    Fast-ish 2-pass chamfer distance in grid cells (not meters).
    obstacle_mask True => distance 0.
    """
    h, w = obstacle_mask.shape
    INF = 1e9
    dist = np.full((h, w), INF, dtype=np.float32)
    dist[obstacle_mask] = 0.0

    # forward
    for y in range(h):
        row = dist[y]
        prev = dist[y - 1] if y > 0 else None
        for x in range(w):
            if row[x] == 0.0:
                continue
            best = row[x]
            if x > 0:
                best = min(best, row[x - 1] + 1.0)
            if y > 0:
                best = min(best, prev[x] + 1.0)
                if x > 0:
                    best = min(best, prev[x - 1] + 1.4142)
                if x + 1 < w:
                    best = min(best, prev[x + 1] + 1.4142)
            row[x] = best

    # backward
    for y in range(h - 1, -1, -1):
        row = dist[y]
        nxt = dist[y + 1] if y + 1 < h else None
        for x in range(w - 1, -1, -1):
            if row[x] == 0.0:
                continue
            best = row[x]
            if x + 1 < w:
                best = min(best, row[x + 1] + 1.0)
            if y + 1 < h:
                best = min(best, nxt[x] + 1.0)
                if x > 0:
                    best = min(best, nxt[x - 1] + 1.4142)
                if x + 1 < w:
                    best = min(best, nxt[x + 1] + 1.4142)
            row[x] = best

    return dist

def compute_yaw_from_quat(q):
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw

def make_quat_yaw(yaw):
    x, y, z, w = quaternion_from_euler(0.0, 0.0, float(yaw))
    q = Quaternion()
    q.x, q.y, q.z, q.w = x, y, z, w
    return q


# -------------------------- mission states -------------------------- #

MISSION_EXPLORING = "EXPLORING"
MISSION_RETURNING = "RETURNING"
MISSION_COMPLETE = "COMPLETE"


# -------------------------- ExplorerPolicy -------------------------- #

class ExplorerPolicy:
    def __init__(self):
        # ---------- params ----------
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # pose_source kept, but pose is always transformed into map frame
        self.pose_source = rospy.get_param("~pose_source", "odom")  # "odom" or "amcl"
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry/filtered")
        self.amcl_topic = rospy.get_param("~amcl_topic", "/amcl_pose")

        self.free_max = int(rospy.get_param("~free_max", 25))
        self.occ_min = int(rospy.get_param("~occ_min", 65))

        self.wall_buffer_m = float(rospy.get_param("~wall_buffer_m", 0.15))
        self.frontier_extra_clearance_m = float(rospy.get_param("~frontier_extra_clearance_m", 0.10))

        self.target_node_spacing_m = float(rospy.get_param("~target_node_spacing_m", 0.5))
        self.max_total_nodes = int(rospy.get_param("~max_total_nodes", 5000))
        self.max_neighbors = int(rospy.get_param("~max_neighbors", 10))
        self.edge_dist_max_m = float(rospy.get_param("~edge_dist_max_m", 6.0))

        self.graph_min_update_period_s = float(rospy.get_param("~graph_min_update_period_s", 0.8))
        self.graph_force_rebuild_period_s = float(rospy.get_param("~graph_force_rebuild_period_s", 3.0))
        self.map_change_min_new_free_cells = int(rospy.get_param("~map_change_min_new_free_cells", 6))

        # frontiers
        self.frontier_limit = int(rospy.get_param("~frontier_limit", 600))
        self.frontier_min_goal_separation_m = float(rospy.get_param("~frontier_min_goal_separation_m", 1.4))
        self.frontier_local_radius_m = float(rospy.get_param("~frontier_local_radius_m", 1.6))
        self.frontier_unknown_ratio_min = float(rospy.get_param("~frontier_unknown_ratio_min", 0.05))
        self.frontier_unknown_ratio_max = float(rospy.get_param("~frontier_unknown_ratio_max", 0.98))
        self.frontier_min_denom_cells = int(rospy.get_param("~frontier_min_denom_cells", 25))

        self.frontier_anchor_ok_m = float(rospy.get_param("~frontier_anchor_ok_m", 3.0))
        self.frontier_anchor_soft_penalty = float(rospy.get_param("~frontier_anchor_soft_penalty", 0.15))
        self.frontier_anchor_hard_cap_m = float(rospy.get_param("~frontier_anchor_hard_cap_m", 999.0))

        # clustering
        self.enable_frontier_clustering = bool(rospy.get_param("~enable_frontier_clustering", True))
        self.cluster_conn8 = bool(rospy.get_param("~cluster_conn8", True))
        self.cluster_min_size = int(rospy.get_param("~cluster_min_size", 8))
        self.cluster_max_reps = int(rospy.get_param("~cluster_max_reps", 2))
        self.cluster_rep_stride_cells = int(rospy.get_param("~cluster_rep_stride_cells", 2))
        self.cluster_bias_to_farther = float(rospy.get_param("~cluster_bias_to_farther", 0.10))

        # scoring
        self.unknown_gain_weight = float(rospy.get_param("~unknown_gain_weight", 1.0))
        self.robot_frontier_distance_weight = float(rospy.get_param("~robot_frontier_distance_weight", 0.38))
        self.heading_weight = float(rospy.get_param("~heading_weight", 0.80))
        self.visited_weight = float(rospy.get_param("~visited_weight", 0.40))

        # radiation
        self.use_radiation = bool(rospy.get_param("~use_radiation", True))
        self.radiation_topic = rospy.get_param("~radiation_topic", "/radiation_costmap")
        self.radiation_sigma_topic = rospy.get_param("~radiation_sigma_topic", "/radiation_sigma")
        self.radiation_weight = float(rospy.get_param("~radiation_weight", 0.70))
        self.variance_weight = float(rospy.get_param("~variance_weight", 0.40))
        self.radiation_edge_weight = float(rospy.get_param("~radiation_edge_weight", 0.15))

        self.radiation_heading_avoid_weight = float(rospy.get_param("~radiation_heading_avoid_weight", 0.35))
        self.radiation_heading_sample_dist_m = float(rospy.get_param("~radiation_heading_sample_dist_m", 2.0))

        # ---------------- NEW: "avoid as much as feasible" controls ----------------
        # Candidate rejection / heavy penalisation, but with automatic fallback when it would stall exploration.
        self.radiation_goal_soft_cap = float(rospy.get_param("~radiation_goal_soft_cap", 0.65))
        self.radiation_goal_hard_cap = float(rospy.get_param("~radiation_goal_hard_cap", 0.95))
        self.radiation_softcap_penalty = float(rospy.get_param("~radiation_softcap_penalty", 2.5))
        self.radiation_hardcap_reject = bool(rospy.get_param("~radiation_hardcap_reject", True))

        # Inflate high-radiation cells into a "virtual obstacle" mask for frontier vetting (NOT for walls),
        # with fallback if it becomes too restrictive.
        self.enable_radiation_exclusion = bool(rospy.get_param("~enable_radiation_exclusion", True))
        self.radiation_exclusion_threshold = float(rospy.get_param("~radiation_exclusion_threshold", 0.90))
        self.radiation_exclusion_inflate_m = float(rospy.get_param("~radiation_exclusion_inflate_m", 0.30))
        self.radiation_exclusion_blocks_frontiers_only = bool(rospy.get_param("~radiation_exclusion_blocks_frontiers_only", True))

        # A* radiation integration (sample along edges)
        self.astar_rad_sample_step_m = float(rospy.get_param("~astar_rad_sample_step_m", 0.25))
        self.astar_rad_integral_weight = float(rospy.get_param("~astar_rad_integral_weight", 1.0))
        # -------------------------------------------------------------------------

        # anti-churn
        self.goal_hold_min_s = float(rospy.get_param("~goal_hold_min_s", 3.0))
        self.replan_period_s = float(rospy.get_param("~replan_period_s", 1.0))
        self.min_corridor_length_m = float(rospy.get_param("~min_corridor_length_m", 0.35))
        self.enable_preplan = bool(rospy.get_param("~enable_preplan", True))
        self.preplan_min_progress = float(rospy.get_param("~preplan_min_progress", 0.70))
        self.preplan_period_s = float(rospy.get_param("~preplan_period_s", 0.8))

        # corridor publish suppression
        self.suppress_duplicate_dispatch = bool(rospy.get_param("~suppress_duplicate_dispatch", True))
        self.corridor_sig_quant_m = float(rospy.get_param("~corridor_sig_quant_m", 0.05))

        # visited / blacklist
        self.visited_radius_m = float(rospy.get_param("~visited_radius_m", 1.10))
        self.visited_decay = float(rospy.get_param("~visited_decay", 0.985))
        self.visited_decay_period_s = float(rospy.get_param("~visited_decay_period_s", 1.2))
        self.blacklist_threshold = int(rospy.get_param("~blacklist_threshold", 3))
        self.blacklist_timeout_s = float(rospy.get_param("~blacklist_timeout_s", 15.0))
        self.frontier_attempt_cooldown_s = float(rospy.get_param("~frontier_attempt_cooldown_s", 35.0))
        self.frontier_repeat_penalty = float(rospy.get_param("~frontier_repeat_penalty", 0.35))

        # completion
        self.completion_unknown_ratio = float(rospy.get_param("~completion_unknown_ratio", 0.020))
        self.completion_confirm_cycles = int(rospy.get_param("~completion_confirm_cycles", 20))
        self.completion_min_total_distance_m = float(rospy.get_param("~completion_min_total_distance_m", 15.0))

        # nav watchdog
        self.nav_stale_reset_s = float(rospy.get_param("~nav_stale_reset_s", 4.0))
        self.nav_progress_eps = float(rospy.get_param("~nav_progress_eps", 0.01))

        # outputs
        self.publish_markers = bool(rospy.get_param("~publish_markers", True))
        self.marker_topic = rospy.get_param("~marker_topic", "/exploration_markers")
        self.goal_topic = rospy.get_param("~goal_topic", "/exploration/goal")
        self.corridor_topic = rospy.get_param("~corridor_topic", "/exploration/corridor")
        self.marker_rate_hz = float(rospy.get_param("~marker_rate_hz", 2.0))
        self.trail_rate_hz = float(rospy.get_param("~trail_rate_hz", 4.0))
        self.trail_max_poses = int(rospy.get_param("~trail_max_poses", 3000))
        self.trail_min_step_m = float(rospy.get_param("~trail_min_step_m", 0.05))

        # goal behaviour
        self.goal_snap_to_node = bool(rospy.get_param("~goal_snap_to_node", True))
        self.goal_face_frontier = bool(rospy.get_param("~goal_face_frontier", True))

        # UNKNOWN handling
        self.frontier_clearance_blocks_unknown = bool(rospy.get_param("~frontier_clearance_blocks_unknown", False))
        self.graph_clearance_blocks_unknown = bool(rospy.get_param("~graph_clearance_blocks_unknown", True))

        # STRICT corridor verification
        self.strict_corridor_verification = bool(rospy.get_param("~strict_corridor_verification", True))
        self.strict_verify_segment_los = bool(rospy.get_param("~strict_verify_segment_los", True))
        self.strict_verify_pose_clearance = bool(rospy.get_param("~strict_verify_pose_clearance", True))
        self.strict_min_clearance_m = float(rospy.get_param("~strict_min_clearance_m", 0.22))
        self.strict_segment_step_m = float(rospy.get_param("~strict_segment_step_m", 0.10))
        self.strict_max_backtrack_frac = float(rospy.get_param("~strict_max_backtrack_frac", 0.25))

        # LOS precheck to frontier
        self.enable_frontier_los_precheck = bool(rospy.get_param("~enable_frontier_los_precheck", False))
        self.frontier_los_use_inflated = bool(rospy.get_param("~frontier_los_use_inflated", True))

        # angular hysteresis + chain
        self.frontier_chain_len = int(rospy.get_param("~frontier_chain_len", 4))
        self.frontier_chain_dist_m = float(rospy.get_param("~frontier_chain_dist_m", 6.0))
        self.frontier_chain_weight = float(rospy.get_param("~frontier_chain_weight", 0.35))
        self.angular_hysteresis_weight = float(rospy.get_param("~angular_hysteresis_weight", 0.25))
        self.angular_hysteresis_tau_s = float(rospy.get_param("~angular_hysteresis_tau_s", 45.0))

        # debug rejected markers
        self.debug_rejected_markers = bool(rospy.get_param("~debug_rejected_markers", True))
        self.debug_rejected_text = bool(rospy.get_param("~debug_rejected_text", False))
        self.debug_rejected_keep = int(rospy.get_param("~debug_rejected_keep", 250))

        # heading cone
        self.viz_heading_cones = bool(rospy.get_param("~viz_heading_cones", True))
        self.viz_heading_cone_half_angle_deg = float(rospy.get_param("~viz_heading_cone_half_angle_deg", 35.0))
        self.viz_heading_cone_len_m = float(rospy.get_param("~viz_heading_cone_len_m", 2.5))

        # optional extra viz
        self.viz_nodes = bool(rospy.get_param("~viz_nodes", False))
        self.viz_frontier_candidates = bool(rospy.get_param("~viz_frontier_candidates", False))

        # fail logic
        self.frontier_fail_soft_timeout_s = float(rospy.get_param("~frontier_fail_soft_timeout_s", 30.0))
        self.frontier_fail_hard_timeout_s = float(rospy.get_param("~frontier_fail_hard_timeout_s", 180.0))
        self.frontier_fail_progress_min = float(rospy.get_param("~frontier_fail_progress_min", 0.15))

        # nav feedback topics
        self.nav_active_topic = rospy.get_param("~nav_active_topic", "/navigation/nav_active")
        self.nav_goal_reached_topic = rospy.get_param("~nav_goal_reached_topic", "/navigation/nav_goal_reached")
        self.nav_progress_topic = rospy.get_param("~nav_progress_topic", "/navigation/nav_progress")

        # anti-churn gating (nav spin-up grace)
        self.nav_spinup_grace_s = float(rospy.get_param("~nav_spinup_grace_s", 6.0))
        self.require_nav_active_for_preplan = bool(rospy.get_param("~require_nav_active_for_preplan", True))
        self.require_progress_update_for_preplan = bool(rospy.get_param("~require_progress_update_for_preplan", True))

        # ---------- internal state ----------
        self._lock = threading.RLock()

        # map frame is taken from OccupancyGrid header unless you override with ~map_frame
        self.map_frame = rospy.get_param("~map_frame", "")  # "" => use msg.header.frame_id
        self.odom_frame = rospy.get_param("~odom_frame", "odom")

        self.map_msg = None
        self.map_w = 0
        self.map_h = 0
        self.map_res = 0.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_data = None
        self.free_mask = None
        self.unknown_mask = None
        self.occ_mask = None

        self.blocked_for_graph = None
        self.blocked_for_frontier = None
        self.inflated_blocked_graph = None
        self.dist_to_blocked_graph_cells = None
        self.wall_buf_cells = 0

        self.radiation = None
        self.radiation_sigma = None
        self._radiation_shape_ok = False
        self._radiation_sigma_shape_ok = False

        # NEW: radiation exclusion mask (optional)
        self._rad_exclusion_mask = None
        self._rad_exclusion_mask_inflated = None
        self._rad_exclusion_cells = 0

        # robot pose ALWAYS stored in MAP frame
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = 0.0
        self.last_robot_pose_t = None

        # mission lifecycle
        self.start_x = None
        self.start_y = None
        self.start_yaw = None

        self.mission_state = MISSION_EXPLORING
        self.mission_complete = False
        self.return_home_sent = False

        # nav
        self.nav_active = False
        self.nav_goal_reached = False
        self.nav_progress = 0.0
        self.nav_last_progress_t = None
        self.nav_last_progress_val = 0.0

        # nav_active tracking for churn gating
        self._have_seen_nav_active = False
        self._last_nav_active_change_t = None

        # graph in MAP frame
        self.nodes_xy = np.zeros((0, 2), dtype=np.float32)
        self.adj = []
        self.edges = []
        self.kdtree = None

        self.graph_version = 0
        self.last_graph_build_t = 0.0
        self.last_free_count = 0

        self.frontier_strength = defaultdict(float)
        self.frontier_blacklist = {}
        self.frontier_fail = {}
        self.frontier_recent_attempt = {}

        self.visited = defaultdict(float)
        self.last_visited_decay_t = 0.0

        self.active_goal = None
        self.last_plan_attempt_t = 0.0
        self.last_plan_attempt_ok = False
        self.last_plan_attempt_reason = ""
        self._completion_ok_cycles = 0

        self.goal_history = deque(maxlen=max(1, self.frontier_chain_len))
        self.last_motion_heading = None
        self.last_motion_heading_t = None

        # trail (MAP frame)
        self.trail_path = Path()
        self.trail_path.header.frame_id = self._effective_map_frame()
        self._last_trail_pub_t = 0.0
        self._last_marker_pub_t = 0.0
        self._last_trail_xy = None
        self._total_distance = 0.0

        # debug rejected candidates buffer
        self._rej = deque(maxlen=max(50, self.debug_rejected_keep))
        self._last_frontiers_n = 0
        self._last_frontier_candidates_for_viz = []  # additive viz only

        # last-dispatched signature (used to suppress duplicate publish spam)
        self._last_dispatch_fkey = None
        self._last_dispatch_corridor_sig = None
        self._last_dispatch_goal_xy = None

        # TF2
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # publishers (all MAP frame messages)
        self.pub_goal = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1, latch=True)
        self.pub_corridor = rospy.Publisher(self.corridor_topic, Path, queue_size=1, latch=True)
        self.pub_state = rospy.Publisher("~debug_state", String, queue_size=1)
        self.pub_markers = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)
        self.pub_trail = rospy.Publisher("~trail", Path, queue_size=1, latch=True)

        # mission state publisher (latched)
        self.pub_mission_state = rospy.Publisher("/mission_state", String, queue_size=1, latch=True)

        # ---------------- TELEMETRY ----------------
        self.pub_telemetry = rospy.Publisher("/mission/telemetry", String, queue_size=10)

        self._mission_start_t = rospy.Time.now().to_sec()
        self._last_tick_t = self._mission_start_t

        # motion-derived metrics
        self._last_motion_t = None
        self._inst_speed_mps = 0.0
        self._avg_speed_mps = 0.0

        # radiation-derived metrics (dose proxy)
        self._inst_dose_rate = 0.0
        self._total_dose = 0.0

        # time accounting
        self._time_exploring = 0.0
        self._time_returning = 0.0
        self._time_idle = 0.0

        # idle threshold (m/s)
        self._idle_speed_thresh = 0.02

        # telemetry publish throttling
        self._last_telemetry_pub_t = 0.0
        self._telemetry_rate_hz = float(rospy.get_param("~telemetry_rate_hz", 4.0))
        # ---------------------------------------------------

        # subscribers
        rospy.Subscriber(self.map_topic, OccupancyGrid, self._on_map, queue_size=1)

        if self.pose_source == "amcl":
            rospy.Subscriber(self.amcl_topic, PoseWithCovarianceStamped, self._on_amcl, queue_size=5)
        else:
            rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=20)

        rospy.Subscriber(self.nav_active_topic, Bool, self._on_nav_active, queue_size=10)
        rospy.Subscriber(self.nav_goal_reached_topic, Bool, self._on_nav_reached, queue_size=10)
        rospy.Subscriber(self.nav_progress_topic, Float32, self._on_nav_progress, queue_size=20)

        if self.use_radiation:
            rospy.Subscriber(self.radiation_topic, OccupancyGrid, self._on_radiation, queue_size=1)
            rospy.Subscriber(self.radiation_sigma_topic, OccupancyGrid, self._on_radiation_sigma, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(0.25), self._tick)

        self._publish_mission_state()

        rospy.loginfo(
            "ExplorerPolicy v20 COMPLETE started: map_frame='%s' base_frame='%s' strict=%s los_precheck=%s clustering=%s kdtree=%s use_radiation=%s rad_caps(soft=%.2f hard=%.2f) rad_excl=%s",
            self._effective_map_frame(),
            self.base_frame,
            self.strict_corridor_verification,
            self.enable_frontier_los_precheck,
            self.enable_frontier_clustering,
            _HAVE_KDTREE,
            self.use_radiation,
            self.radiation_goal_soft_cap,
            self.radiation_goal_hard_cap,
            self.enable_radiation_exclusion,
        )

    def _effective_map_frame(self):
        return self.map_frame if self.map_frame else "map"

    # ---------------- mission helpers ---------------- #

    def _publish_mission_state(self):
        self.pub_mission_state.publish(String(data=self.mission_state))

    def _ensure_start_pose(self):
        if self.start_x is None and self.robot_x is not None and self.robot_y is not None:
            self.start_x = float(self.robot_x)
            self.start_y = float(self.robot_y)
            self.start_yaw = float(self.robot_yaw)
            rospy.loginfo(
                "[MISSION] Start pose recorded (MAP): (%.2f, %.2f, %.1f deg)",
                self.start_x, self.start_y, math.degrees(self.start_yaw)
            )

    # ----------- anti-churn: signatures ----------- #

    def _quant(self, v, q):
        return int(round(float(v) / max(1e-9, float(q))))

    def _corridor_signature(self, corridor: Path):
        if corridor is None or not corridor.poses:
            return ("empty", 0)
        q = self.corridor_sig_quant_m
        n = len(corridor.poses)
        p0 = corridor.poses[0].pose.position
        p1 = corridor.poses[-1].pose.position
        pm = corridor.poses[n // 2].pose.position
        return (
            n,
            self._quant(p0.x, q), self._quant(p0.y, q),
            self._quant(pm.x, q), self._quant(pm.y, q),
            self._quant(p1.x, q), self._quant(p1.y, q),
        )

    def _should_publish_dispatch(self, fkey, corridor: Path, goal: PoseStamped):
        if not self.suppress_duplicate_dispatch:
            return True

        gx = float(goal.pose.position.x)
        gy = float(goal.pose.position.y)
        goal_xy = (self._quant(gx, self.corridor_sig_quant_m), self._quant(gy, self.corridor_sig_quant_m))
        sig = self._corridor_signature(corridor) if corridor is not None else None

        if (fkey == self._last_dispatch_fkey and
            sig == self._last_dispatch_corridor_sig and
            goal_xy == self._last_dispatch_goal_xy):
            return False

        return True

    def _record_dispatch(self, fkey, corridor: Path, goal: PoseStamped):
        self._last_dispatch_fkey = fkey
        self._last_dispatch_corridor_sig = self._corridor_signature(corridor) if corridor is not None else None
        gx = float(goal.pose.position.x)
        gy = float(goal.pose.position.y)
        self._last_dispatch_goal_xy = (self._quant(gx, self.corridor_sig_quant_m), self._quant(gy, self.corridor_sig_quant_m))

    # ---------------- return-home dispatch ---------------- #

    def _dispatch_return_home(self, now):
        if self.start_x is None or self.start_y is None:
            rospy.logwarn("[MISSION] Start pose unknown; cannot return home.")
            return False

        if self.active_goal is not None:
            fkey = self.active_goal.get("fkey", None)
            if fkey == ("return", "home"):
                return False

        if self.nodes_xy is None or self.nodes_xy.shape[0] < 2:
            rospy.logwarn("[MISSION] No graph available for return-to-home; falling back to direct goal.")
            return self._dispatch_return_home_direct(now, reason="return_home_direct_no_graph")

        rx = float(self.robot_x)
        ry = float(self.robot_y)
        start_i = self._nearest_node(rx, ry)
        goal_i = self._nearest_node(float(self.start_x), float(self.start_y))

        idxs = self._astar(start_i, goal_i)
        if not idxs:
            rospy.logwarn("[MISSION] Return-to-home A* failed; falling back to direct goal.")
            return self._dispatch_return_home_direct(now, reason="return_home_direct_no_path")

        corridor = self._nodes_to_path(idxs)

        if corridor.poses:
            corridor.poses[-1].pose.position.x = float(self.start_x)
            corridor.poses[-1].pose.position.y = float(self.start_y)
            corridor.poses[-1].pose.orientation = make_quat_yaw(float(self.start_yaw))

        if self.strict_corridor_verification:
            ok, why = self._strict_verify_corridor(corridor)
            if not ok:
                rospy.logwarn("[MISSION] Return-to-home strict verification failed (%s); falling back to direct goal.", why)
                return self._dispatch_return_home_direct(now, reason="return_home_direct_strict_fail:" + str(why))

        goal = PoseStamped()
        goal.header.frame_id = self._effective_map_frame()
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = float(self.start_x)
        goal.pose.position.y = float(self.start_y)
        goal.pose.orientation = make_quat_yaw(float(self.start_yaw))

        fkey = ("return", "home")
        if self._should_publish_dispatch(fkey, corridor, goal):
            self.pub_corridor.publish(corridor)
            self.pub_goal.publish(goal)
            self._record_dispatch(fkey, corridor, goal)

        self.active_goal = {
            "fkey": fkey,
            "goal": goal,
            "corridor": corridor,
            "t_sent": now,
            "score": 0.0,
        }

        self.return_home_sent = True
        self.last_plan_attempt_ok = True
        self.last_plan_attempt_reason = "return_home_astar" + ("" if self.use_radiation else "_no_radiation")
        rospy.loginfo("[MISSION] Return-to-start corridor dispatched (A*, radiation=%s).", self.use_radiation)
        return True

    def _dispatch_return_home_direct(self, now, reason="return_home_direct"):
        goal = PoseStamped()
        goal.header.frame_id = self._effective_map_frame()
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = float(self.start_x)
        goal.pose.position.y = float(self.start_y)
        goal.pose.orientation = make_quat_yaw(float(self.start_yaw))

        fkey = ("return", "home")
        if self._should_publish_dispatch(fkey, None, goal):
            self.pub_corridor.publish(Path())
            self.pub_goal.publish(goal)
            self._record_dispatch(fkey, None, goal)

        self.active_goal = {
            "fkey": fkey,
            "goal": goal,
            "corridor": None,
            "t_sent": now,
            "score": 0.0,
        }

        self.return_home_sent = True
        self.last_plan_attempt_ok = True
        self.last_plan_attempt_reason = reason
        rospy.loginfo("[MISSION] Return-to-start goal dispatched (DIRECT fallback).")
        return True

    # ---------------- TF pose fetch (MAP frame) ---------------- #

    def _update_robot_pose_map(self):
        target = self._effective_map_frame()
        src = self.base_frame

        ps = TfPoseStamped()
        ps.header.frame_id = src
        ps.header.stamp = rospy.Time(0)
        ps.pose.position.x = 0.0
        ps.pose.position.y = 0.0
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0

        try:
            out = self.tf_buffer.transform(ps, target, rospy.Duration(0.2))
            self.robot_x = float(out.pose.position.x)
            self.robot_y = float(out.pose.position.y)
            self.robot_yaw = compute_yaw_from_quat(out.pose.orientation)
            self.last_robot_pose_t = rospy.Time.now().to_sec()
            return True
        except Exception:
            return False

    def _pose_to_map(self, pose_stamped_in):
        target = self._effective_map_frame()
        try:
            out = self.tf_buffer.transform(pose_stamped_in, target, rospy.Duration(0.2))
            x = float(out.pose.position.x)
            y = float(out.pose.position.y)
            yaw = compute_yaw_from_quat(out.pose.orientation)
            return x, y, yaw
        except Exception:
            return None

    # ---------------- subscribers ---------------- #

    def _recompute_rad_exclusion(self):
        # Safe no-op until both map + radiation exist and align.
        if not (self.enable_radiation_exclusion and self.use_radiation and self._radiation_shape_ok):
            self._rad_exclusion_mask = None
            self._rad_exclusion_mask_inflated = None
            self._rad_exclusion_cells = 0
            return

        thr = float(self.radiation_exclusion_threshold)
        base = (self.radiation >= thr)
        self._rad_exclusion_mask = base

        r_cells = max(0, int(round(float(self.radiation_exclusion_inflate_m) / max(self.map_res, 1e-6))))
        self._rad_exclusion_cells = int(r_cells)
        if r_cells > 0:
            self._rad_exclusion_mask_inflated = self._inflate_mask(base, r_cells)
        else:
            self._rad_exclusion_mask_inflated = base

    def _on_map(self, msg: OccupancyGrid):
        with self._lock:
            self.map_msg = msg
            if not self.map_frame:
                self.map_frame = msg.header.frame_id if msg.header.frame_id else "map"
                self.trail_path.header.frame_id = self.map_frame

            self.map_w = msg.info.width
            self.map_h = msg.info.height
            self.map_res = msg.info.resolution
            self.map_origin_x = msg.info.origin.position.x
            self.map_origin_y = msg.info.origin.position.y

            data = np.asarray(msg.data, dtype=np.int16).reshape((self.map_h, self.map_w))
            self.map_data = data
            self.free_mask = (data >= 0) & (data <= self.free_max)
            self.unknown_mask = (data < 0)
            self.occ_mask = (data >= self.occ_min)

            if self.graph_clearance_blocks_unknown:
                self.blocked_for_graph = self.occ_mask | self.unknown_mask
            else:
                self.blocked_for_graph = self.occ_mask

            if self.frontier_clearance_blocks_unknown:
                self.blocked_for_frontier = self.occ_mask | self.unknown_mask
            else:
                self.blocked_for_frontier = self.occ_mask

            self.wall_buf_cells = max(1, int(round(self.wall_buffer_m / max(self.map_res, 1e-6))))
            self.inflated_blocked_graph = self._inflate_mask(self.blocked_for_graph, self.wall_buf_cells)
            self.dist_to_blocked_graph_cells = fast_chamfer_distance(self.blocked_for_graph)

            # Validate radiation alignment to map
            self._radiation_shape_ok = (self.radiation is not None and
                                        self.radiation.shape[0] == self.map_h and
                                        self.radiation.shape[1] == self.map_w)
            self._radiation_sigma_shape_ok = (self.radiation_sigma is not None and
                                              self.radiation_sigma.shape[0] == self.map_h and
                                              self.radiation_sigma.shape[1] == self.map_w)

            # NEW
            self._recompute_rad_exclusion()

    def _on_radiation(self, msg: OccupancyGrid):
        with self._lock:
            try:
                arr = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
                self.radiation = arr.astype(np.float32) / 100.0
                self._radiation_shape_ok = (self.map_h > 0 and self.map_w > 0 and
                                            self.radiation.shape[0] == self.map_h and
                                            self.radiation.shape[1] == self.map_w)
            except Exception:
                self.radiation = None
                self._radiation_shape_ok = False
            self._recompute_rad_exclusion()

    def _on_radiation_sigma(self, msg: OccupancyGrid):
        with self._lock:
            try:
                arr = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
                self.radiation_sigma = arr.astype(np.float32) / 100.0
                self._radiation_sigma_shape_ok = (self.map_h > 0 and self.map_w > 0 and
                                                  self.radiation_sigma.shape[0] == self.map_h and
                                                  self.radiation_sigma.shape[1] == self.map_w)
            except Exception:
                self.radiation_sigma = None
                self._radiation_sigma_shape_ok = False

    def _on_odom(self, msg: Odometry):
        with self._lock:
            ps = PoseStamped()
            ps.header.frame_id = msg.header.frame_id if msg.header.frame_id else self.odom_frame
            ps.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time(0)
            ps.pose = msg.pose.pose
            res = self._pose_to_map(ps)
            if res is not None:
                self.robot_x, self.robot_y, self.robot_yaw = res
                self.last_robot_pose_t = rospy.Time.now().to_sec()

    def _on_amcl(self, msg: PoseWithCovarianceStamped):
        with self._lock:
            ps = PoseStamped()
            ps.header.frame_id = msg.header.frame_id if msg.header.frame_id else self._effective_map_frame()
            ps.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time(0)
            ps.pose = msg.pose.pose
            res = self._pose_to_map(ps)
            if res is not None:
                self.robot_x, self.robot_y, self.robot_yaw = res
                self.last_robot_pose_t = rospy.Time.now().to_sec()

    def _on_nav_active(self, msg: Bool):
        with self._lock:
            v = bool(msg.data)
            if v != self.nav_active:
                self._last_nav_active_change_t = rospy.Time.now().to_sec()
            self.nav_active = v
            self._have_seen_nav_active = True

    def _on_nav_reached(self, msg: Bool):
        with self._lock:
            self.nav_goal_reached = bool(msg.data)

    def _on_nav_progress(self, msg: Float32):
        with self._lock:
            self.nav_progress = float(msg.data)
            now = rospy.Time.now().to_sec()
            if self.nav_last_progress_t is None or abs(self.nav_progress - self.nav_last_progress_val) > self.nav_progress_eps:
                self.nav_last_progress_t = now
                self.nav_last_progress_val = self.nav_progress
            if self.active_goal is not None:
                fkey = self.active_goal.get("fkey")
                if fkey is not None and fkey in self.frontier_fail:
                    self.frontier_fail[fkey]["progress_max"] = max(
                        float(self.frontier_fail[fkey].get("progress_max", 0.0)), self.nav_progress
                    )

    # ---------------- tick loop ---------------- #

    def _tick(self, _evt):
        try:
            with self._lock:
                now = rospy.Time.now().to_sec()

                if self.map_data is None or self.map_res <= 0.0:
                    self._publish_state(now, "waiting_map")
                    self._telemetry_tick(now)
                    return

                self._update_robot_pose_map()
                if self.robot_x is None or self.robot_y is None:
                    self._publish_state(now, "waiting_pose_tf")
                    self._telemetry_tick(now)
                    return

                self._ensure_start_pose()

                if self.last_visited_decay_t == 0.0:
                    self.last_visited_decay_t = now
                if now - self.last_visited_decay_t >= self.visited_decay_period_s:
                    self._decay_visited()
                    self.last_visited_decay_t = now

                self._decay_blacklist(now)
                self._update_trail(now)
                self._nav_watchdog(now)

                if self._should_rebuild_graph(now):
                    self._build_graph(now)

                if self.active_goal is not None and self.nav_goal_reached:
                    self._on_active_goal_reached(now)
                    self.nav_goal_reached = False

                if self.mission_state == MISSION_COMPLETE:
                    self._maybe_publish_viz(now)
                    self._publish_state(now, "mission_complete")
                    self._telemetry_tick(now)
                    return

                if self.mission_state == MISSION_RETURNING:
                    if self.active_goal is None:
                        self._dispatch_return_home(now)
                    self._maybe_publish_viz(now)
                    self._publish_state(now, "returning_home")
                    self._telemetry_tick(now)
                    return

                if not self._should_plan(now):
                    self._publish_state(now, "replan_wait")
                    self._maybe_publish_viz(now)
                    self._telemetry_tick(now)
                    return

                ok = self._plan_and_dispatch(now)
                self._maybe_publish_viz(now)
                self._publish_state(now, "goal_dispatched" if ok else "no_admissible_plan")
                self._telemetry_tick(now)
        except Exception as e:
            rospy.logerr("Tick error: %s", str(e))

    # ---------------- TELEMETRY ---------------- #

    def _telemetry_tick(self, now):
        if self._telemetry_rate_hz > 0.0:
            if now - self._last_telemetry_pub_t < (1.0 / max(0.1, self._telemetry_rate_hz)):
                return
        self._last_telemetry_pub_t = now

        dt = max(0.0, now - self._last_tick_t)
        self._last_tick_t = now

        if self._inst_speed_mps < self._idle_speed_thresh:
            self._time_idle += dt
        else:
            if self.mission_state == MISSION_EXPLORING:
                self._time_exploring += dt
            elif self.mission_state == MISSION_RETURNING:
                self._time_returning += dt

        mission_dur = max(0.0, now - self._mission_start_t)
        self._avg_speed_mps = (self._total_distance / mission_dur) if mission_dur > 1e-3 else 0.0
        dose_per_m = (self._total_dose / self._total_distance) if self._total_distance > 1e-6 else 0.0

        goal_age = None
        last_goal_key = None
        if self.active_goal is not None:
            last_goal_key = str(self.active_goal.get("fkey"))
            goal_age = round(float(now - float(self.active_goal.get("t_sent", now))), 3)

        telem = {
            "t": round(now, 3),
            "mission_state": self.mission_state,
            "map_frame": self._effective_map_frame(),

            "robot_xy": None if self.robot_x is None else [float(self.robot_x), float(self.robot_y)],
            "robot_yaw_deg": round(math.degrees(float(self.robot_yaw)), 2) if self.robot_x is not None else None,

            "start_xy": None if self.start_x is None else [float(self.start_x), float(self.start_y)],
            "distance_m": round(float(self._total_distance), 4),

            "speed_mps": round(float(self._inst_speed_mps), 4),
            "speed_avg_mps": round(float(self._avg_speed_mps), 4),

            "dose_rate": round(float(self._inst_dose_rate), 6),
            "dose_total": round(float(self._total_dose), 6),
            "dose_per_meter": round(float(dose_per_m), 6),

            "time_exploring_s": round(float(self._time_exploring), 2),
            "time_returning_s": round(float(self._time_returning), 2),
            "time_idle_s": round(float(self._time_idle), 2),
            "mission_duration_s": round(float(mission_dur), 2),

            "nav_active": bool(self.nav_active),
            "nav_progress": round(float(self.nav_progress), 4),
            "nav_last_progress_age_s": None if self.nav_last_progress_t is None else round(now - self.nav_last_progress_t, 3),

            "active_goal": last_goal_key,
            "goal_age_s": goal_age,

            "nodes": int(self.nodes_xy.shape[0]),
            "edges": int(len(self.edges)),
            "graph_version": int(self.graph_version),

            "frontiers_last_n": int(self._last_frontiers_n),
            "rejected_last_n": int(len(self._rej)),
            "last_plan_ok": bool(self.last_plan_attempt_ok),
            "last_plan_reason": self.last_plan_attempt_reason,
            "last_corridor_sig": None if self._last_dispatch_corridor_sig is None else str(self._last_dispatch_corridor_sig),

            # new knobs (useful for logs)
            "rad_soft_cap": float(self.radiation_goal_soft_cap),
            "rad_hard_cap": float(self.radiation_goal_hard_cap),
            "rad_excl": bool(self.enable_radiation_exclusion),
        }

        self.pub_telemetry.publish(String(data=json.dumps(telem, separators=(",", ":"))))

    # ---------------- state / decay ---------------- #

    def _publish_state(self, now, reason):
        d = {
            "t": round(now, 3),
            "reason": reason,
            "mission_state": self.mission_state,
            "map_frame": self._effective_map_frame(),
            "nodes": int(self.nodes_xy.shape[0]),
            "edges": int(len(self.edges)),
            "robot_xy": None if self.robot_x is None else [float(self.robot_x), float(self.robot_y)],
            "start_xy": None if self.start_x is None else [float(self.start_x), float(self.start_y)],
            "active_goal": None if self.active_goal is None else self.active_goal.get("fkey"),
            "graph_version": int(self.graph_version),
            "last_plan_ok": bool(self.last_plan_attempt_ok),
            "last_plan_reason": self.last_plan_attempt_reason,
            "nav_progress": float(self.nav_progress),
            "total_distance_m": float(self._total_distance),
            "strict": bool(self.strict_corridor_verification),
            "los_precheck": bool(self.enable_frontier_los_precheck),
            "clustering": bool(self.enable_frontier_clustering),
        }
        self.pub_state.publish(String(data=json.dumps(d, separators=(",", ":"))))

    def _decay_visited(self):
        kill = []
        for k, v in self.visited.items():
            nv = v * self.visited_decay
            if nv < 0.01:
                kill.append(k)
            else:
                self.visited[k] = nv
        for k in kill:
            self.visited.pop(k, None)

    def _decay_blacklist(self, now):
        for k in list(self.frontier_strength.keys()):
            self.frontier_strength[k] *= 0.975
            if self.frontier_strength[k] < 0.01:
                self.frontier_strength.pop(k, None)
        for k in list(self.frontier_blacklist.keys()):
            if now >= self.frontier_blacklist[k]:
                self.frontier_blacklist.pop(k, None)
        for k in list(self.frontier_recent_attempt.keys()):
            if now - self.frontier_recent_attempt[k] > max(5.0, self.frontier_attempt_cooldown_s * 2.0):
                self.frontier_recent_attempt.pop(k, None)
        for k in list(self.frontier_fail.keys()):
            rec = self.frontier_fail[k]
            if now - rec.get("t0", now) > self.frontier_fail_hard_timeout_s * 2.0:
                self.frontier_fail.pop(k, None)

    # ---------------- watchdog ---------------- #

    def _nav_watchdog(self, now):
        if self.active_goal is None:
            return

        is_return = (self.active_goal.get("fkey", None) == ("return", "home"))

        if self.nav_last_progress_t is not None and (now - self.nav_last_progress_t) > self.nav_stale_reset_s:
            if not is_return:
                fkey = self.active_goal.get("fkey")
                if fkey is not None:
                    self._register_frontier_fail(fkey, now, "nav_stale")
                    self.frontier_blacklist[fkey] = now + max(self.blacklist_timeout_s, 10.0)
            self.active_goal = None
            self.nav_goal_reached = False
            return

        age = now - float(self.active_goal.get("t_sent", now))
        if age > self.frontier_fail_hard_timeout_s:
            if not is_return:
                fkey = self.active_goal.get("fkey")
                if fkey is not None:
                    self._register_frontier_fail(fkey, now, "goal_timeout")
                    self.frontier_blacklist[fkey] = now + max(self.blacklist_timeout_s, 30.0)
            self.active_goal = None
            self.nav_goal_reached = False

    # ---------------- trail (MAP frame) ---------------- #

    def _update_trail(self, now):
        x = float(self.robot_x)
        y = float(self.robot_y)

        if self._last_motion_t is None:
            self._last_motion_t = now

        if self._last_trail_xy is None:
            self._last_trail_xy = (x, y)

        dx = x - self._last_trail_xy[0]
        dy = y - self._last_trail_xy[1]
        ds = hypot2(dx, dy)

        dt_motion = max(1e-6, now - self._last_motion_t)

        if ds >= self.trail_min_step_m:
            self._total_distance += ds
            self._last_trail_xy = (x, y)

            self._inst_speed_mps = float(ds / dt_motion)
            self._last_motion_t = now

            if self.use_radiation and self._radiation_shape_ok:
                self._inst_dose_rate = float(self._radiation_at_world(x, y))
                self._total_dose += self._inst_dose_rate * ds
            else:
                self._inst_dose_rate = 0.0

            ps = PoseStamped()
            ps.header.frame_id = self._effective_map_frame()
            ps.header.stamp = rospy.Time.now()
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation = make_quat_yaw(self.robot_yaw)
            self.trail_path.header = ps.header
            self.trail_path.poses.append(ps)
            if len(self.trail_path.poses) > self.trail_max_poses:
                self.trail_path.poses = self.trail_path.poses[-self.trail_max_poses:]
        else:
            self._inst_speed_mps = 0.0
            if self.use_radiation and self._radiation_shape_ok:
                self._inst_dose_rate = float(self._radiation_at_world(x, y))
            else:
                self._inst_dose_rate = 0.0

        if now - self._last_trail_pub_t >= (1.0 / max(self.trail_rate_hz, 0.1)):
            self._last_trail_pub_t = now
            self.pub_trail.publish(self.trail_path)

    # ---------------- graph ---------------- #

    def _should_rebuild_graph(self, now):
        if now - self.last_graph_build_t < self.graph_min_update_period_s:
            return False
        if now - self.last_graph_build_t > self.graph_force_rebuild_period_s:
            return True
        free_count = int(np.count_nonzero(self.free_mask)) if self.free_mask is not None else 0
        if abs(free_count - self.last_free_count) >= self.map_change_min_new_free_cells:
            return True
        return False

    def _inflate_mask(self, mask, r):
        if r <= 0:
            return mask.copy()
        h, w = mask.shape
        m = mask.astype(np.uint8)
        ii = np.pad(m, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        out = np.zeros_like(mask, dtype=bool)
        r = int(r)
        xs = np.arange(w, dtype=np.int32)
        for y in range(h):
            y0 = max(0, y - r); y1 = min(h - 1, y + r)
            iy0 = y0
            iy1 = y1 + 1
            x0 = np.clip(xs - r, 0, w - 1)
            x1 = np.clip(xs + r, 0, w - 1)
            ix0 = x0
            ix1 = x1 + 1
            s = ii[iy1, ix1] - ii[iy0, ix1] - ii[iy1, ix0] + ii[iy0, ix0]
            out[y, :] = s > 0
        return out

    def _map_to_grid(self, x_map, y_map):
        return world_to_grid(x_map, y_map, self.map_origin_x, self.map_origin_y, self.map_res)

    def _build_graph(self, now):
        t0 = time.time()
        h, w = self.map_h, self.map_w
        res = self.map_res
        if h == 0 or w == 0:
            return

        buf = self.wall_buf_cells
        clearance_ok = self.free_mask & (~self.inflated_blocked_graph)
        clearance_ok &= (self.dist_to_blocked_graph_cells >= float(buf))

        # NOTE: we do NOT block graph nodes by radiation by default (coverage first).
        # Radiation is handled in scoring and A* cost. This avoids disconnecting areas.
        spacing_cells = max(2, int(round(self.target_node_spacing_m / max(res, 1e-6))))
        ys = np.arange(0, h, spacing_cells, dtype=np.int32)
        xs = np.arange(0, w, spacing_cells, dtype=np.int32)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        cand = clearance_ok[grid_y, grid_x]
        gy = grid_y[cand].astype(np.int32)
        gx = grid_x[cand].astype(np.int32)

        if gx.size > self.max_total_nodes:
            d = self.dist_to_blocked_graph_cells[gy, gx]
            idx = np.argsort(-d)[: self.max_total_nodes]
            gx = gx[idx]; gy = gy[idx]

        xs_map, ys_map = grid_to_world_vec(gx, gy, self.map_origin_x, self.map_origin_y, res)
        nodes = np.stack([xs_map, ys_map], axis=1).astype(np.float32)

        kd = cKDTree(nodes) if (_HAVE_KDTREE and nodes.shape[0] > 0) else None
        adj = [[] for _ in range(nodes.shape[0])]
        edges = []

        if nodes.shape[0] > 1:
            max_d = float(self.edge_dist_max_m)

            if kd is not None:
                try:
                    pair_arr = kd.query_pairs(r=max_d, output_type='ndarray')
                    pair_list = pair_arr.tolist() if pair_arr.size else []
                except Exception:
                    pair_list = list(kd.query_pairs(r=max_d))
            else:
                pair_list = []
                max_d2 = max_d * max_d
                for i in range(nodes.shape[0]):
                    xi, yi = float(nodes[i, 0]), float(nodes[i, 1])
                    for j in range(i + 1, nodes.shape[0]):
                        dx = xi - float(nodes[j, 0])
                        dy = yi - float(nodes[j, 1])
                        if dx * dx + dy * dy <= max_d2:
                            pair_list.append((i, j))

            # precompute node grid coords
            ng = np.zeros((nodes.shape[0], 2), dtype=np.int32)
            for i in range(nodes.shape[0]):
                gx_i, gy_i = self._map_to_grid(float(nodes[i, 0]), float(nodes[i, 1]))
                ng[i, 0] = gx_i; ng[i, 1] = gy_i

            inflated = self.inflated_blocked_graph

            def los_ok(i, j):
                x0, y0 = ng[i]
                x1, y1 = ng[j]
                for x, y in bresenham(x0, y0, x1, y1):
                    if x < 0 or x >= w or y < 0 or y >= h:
                        return False
                    if inflated[y, x]:
                        return False
                return True

            neigh = [[] for _ in range(nodes.shape[0])]
            for i, j in pair_list:
                if los_ok(i, j):
                    dx = float(nodes[i, 0] - nodes[j, 0])
                    dy = float(nodes[i, 1] - nodes[j, 1])
                    d2 = dx * dx + dy * dy
                    neigh[i].append((d2, j))
                    neigh[j].append((d2, i))

            max_n = int(self.max_neighbors)
            for i in range(nodes.shape[0]):
                if not neigh[i]:
                    continue
                neigh[i].sort(key=lambda t: t[0])
                for _, j in neigh[i][:max_n]:
                    adj[i].append(j)

            seen = set()
            for i in range(nodes.shape[0]):
                for j in adj[i]:
                    a, b = (i, j) if i < j else (j, i)
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    edges.append((a, b))

        self.nodes_xy = nodes
        self.kdtree = kd
        self.adj = adj
        self.edges = edges
        self.graph_version += 1
        self.last_graph_build_t = now
        self.last_free_count = int(np.count_nonzero(self.free_mask)) if self.free_mask is not None else 0

        rospy.loginfo("[GRAPH] nodes=%d edges=%d build=%.3fs frame=%s", nodes.shape[0], len(edges), time.time() - t0, self._effective_map_frame())

    # ---------------- planning ---------------- #

    def _should_plan(self, now):
        if self.mission_state in (MISSION_COMPLETE, MISSION_RETURNING):
            return False

        if self.active_goal is None:
            return (now - self.last_plan_attempt_t) >= self.replan_period_s

        sent = float(self.active_goal.get("t_sent", now))
        age = now - sent

        if (not self.nav_active) and (age < self.nav_spinup_grace_s):
            return False

        if age < self.goal_hold_min_s:
            if not self.enable_preplan:
                return False
            if age < self.preplan_period_s:
                return False
            if self.require_nav_active_for_preplan and (not self.nav_active):
                return False
            if self.require_progress_update_for_preplan and (self.nav_last_progress_t is None):
                return False
            if float(self.nav_progress) < float(self.preplan_min_progress):
                return True
            return False

        return (now - self.last_plan_attempt_t) >= self.replan_period_s

    def _plan_and_dispatch(self, now):
        self.last_plan_attempt_t = now
        self._rej.clear()
        self._last_frontier_candidates_for_viz = []

        if self.mission_state == MISSION_COMPLETE:
            self.last_plan_attempt_ok = False
            self.last_plan_attempt_reason = "mission_complete"
            return False
        if self.mission_state == MISSION_RETURNING:
            return self._dispatch_return_home(now)

        if self.nodes_xy.shape[0] < 2:
            self.last_plan_attempt_ok = False
            self.last_plan_attempt_reason = "no_graph"
            return False

        frontier_candidates = self._compute_frontiers_maximal()
        self._last_frontiers_n = len(frontier_candidates)
        if self.viz_frontier_candidates:
            self._last_frontier_candidates_for_viz = [(fx, fy, float(unk)) for _, _, fx, fy, unk in frontier_candidates]

        if not frontier_candidates:
            self._completion_ok_cycles += 1
            if (not self.mission_complete and
                self._completion_ok_cycles >= self.completion_confirm_cycles and
                self._completion_check()):
                self.mission_complete = True
                self.mission_state = MISSION_RETURNING
                self._publish_mission_state()
                rospy.loginfo("[MISSION] Exploration complete. Transition to RETURNING.")
                return self._dispatch_return_home(now)

            self.last_plan_attempt_ok = False
            self.last_plan_attempt_reason = "no_frontiers"
            return False

        self._completion_ok_cycles = 0

        choice = self._choose_frontier(frontier_candidates, now)
        if choice is None:
            # If we were too strict on radiation caps/exclusion, we fall back once by disabling rejection,
            # but we keep penalties so it still “tries” to avoid.
            if self.use_radiation and self._radiation_shape_ok and self.radiation_hardcap_reject:
                rospy.logwarn("[PLAN] All candidates rejected. Temporarily relaxing hard-cap rejection for this cycle.")
                old = self.radiation_hardcap_reject
                self.radiation_hardcap_reject = False
                try:
                    choice = self._choose_frontier(frontier_candidates, now)
                finally:
                    self.radiation_hardcap_reject = old

            if choice is None:
                self.last_plan_attempt_ok = False
                self.last_plan_attempt_reason = "all_rejected"
                return False

        fkey, fx, fy, gx, gy, score = choice

        rx = float(self.robot_x)
        ry = float(self.robot_y)
        start_i = self._nearest_node(rx, ry)
        goal_i = self._nearest_node(fx, fy)

        path_nodes = self._astar(start_i, goal_i)
        if not path_nodes:
            self._register_frontier_fail(fkey, now, "no_path")
            self._rej.append({"x": fx, "y": fy, "reason": "no_path", "score": score})
            self.last_plan_attempt_ok = False
            self.last_plan_attempt_reason = "graph_no_path"
            return False

        corridor = self._nodes_to_path(path_nodes)

        if self.strict_corridor_verification:
            ok, why = self._strict_verify_corridor(corridor)
            if not ok:
                self._register_frontier_fail(fkey, now, "strict_fail:" + why)
                self.frontier_blacklist[fkey] = now + max(self.blacklist_timeout_s, 15.0)
                self._rej.append({"x": fx, "y": fy, "reason": "strict:" + why, "score": score})
                self.last_plan_attempt_ok = False
                self.last_plan_attempt_reason = "strict_corridor_fail"
                return False

        if self._path_length(corridor) < self.min_corridor_length_m:
            self._register_frontier_fail(fkey, now, "too_short")
            self._rej.append({"x": fx, "y": fy, "reason": "too_short", "score": score})
            self.last_plan_attempt_ok = False
            self.last_plan_attempt_reason = "corridor_short"
            return False

        goal = PoseStamped()
        goal.header.frame_id = self._effective_map_frame()
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = corridor.poses[-1].pose.position.x
        goal.pose.position.y = corridor.poses[-1].pose.position.y

        yaw = float(self.robot_yaw)
        if self.goal_face_frontier:
            if len(corridor.poses) >= 2:
                x0 = corridor.poses[-2].pose.position.x
                y0 = corridor.poses[-2].pose.position.y
                x1 = corridor.poses[-1].pose.position.x
                y1 = corridor.poses[-1].pose.position.y
                yaw = math.atan2(y1 - y0, x1 - x0)
            else:
                yaw = math.atan2(goal.pose.position.y - ry, goal.pose.position.x - rx)
        goal.pose.orientation = make_quat_yaw(yaw)

        if self._should_publish_dispatch(fkey, corridor, goal):
            self.pub_corridor.publish(corridor)
            self.pub_goal.publish(goal)
            self._record_dispatch(fkey, corridor, goal)

        self.active_goal = {"fkey": fkey, "goal": goal, "corridor": corridor, "t_sent": now, "score": score}
        self.frontier_recent_attempt[fkey] = now
        self.last_motion_heading = yaw
        self.last_motion_heading_t = now

        self.last_plan_attempt_ok = True
        self.last_plan_attempt_reason = "ok"
        return True

    # ---------------- completion ---------------- #

    def _completion_check(self):
        free = int(np.count_nonzero(self.free_mask)) if self.free_mask is not None else 0
        unk = int(np.count_nonzero(self.unknown_mask)) if self.unknown_mask is not None else 0
        denom = max(1, free + unk)
        unk_ratio = float(unk) / float(denom)
        if unk_ratio > self.completion_unknown_ratio:
            return False
        if self._total_distance < self.completion_min_total_distance_m:
            return False
        return True

    def _on_active_goal_reached(self, now):
        ag = self.active_goal
        if ag is None:
            return

        if ag.get("fkey", None) == ("return", "home") and self.mission_state == MISSION_RETURNING:
            rospy.loginfo("[MISSION] Returned to start. Mission COMPLETE.")
            self.mission_state = MISSION_COMPLETE
            self._publish_mission_state()
            self.active_goal = None
            return

        gx, gy = self._map_to_grid(float(ag["goal"].pose.position.x), float(ag["goal"].pose.position.y))
        self._mark_visited_disk(gx, gy, self.visited_radius_m)
        self.goal_history.append((float(ag["goal"].pose.position.x), float(ag["goal"].pose.position.y)))
        self.frontier_strength[ag["fkey"]] = max(self.frontier_strength[ag["fkey"]], 0.55)
        self.active_goal = None

    # ---------------- maximal frontier generation ---------------- #

    def _compute_frontiers_maximal(self):
        free = self.free_mask
        unk = self.unknown_mask
        if free is None or unk is None:
            return []

        unk_n = np.zeros_like(unk, dtype=bool)
        unk_n[:-1, :] |= unk[1:, :]
        unk_n[1:, :] |= unk[:-1, :]
        unk_n[:, :-1] |= unk[:, 1:]
        unk_n[:, 1:] |= unk[:, :-1]
        frontier_mask = free & unk_n

        extra_cells = max(0, int(round(self.frontier_extra_clearance_m / max(self.map_res, 1e-6))))
        if extra_cells > 0:
            inflated_frontier_blocked = self._inflate_mask(self.blocked_for_frontier, self.wall_buf_cells + extra_cells)
            frontier_mask &= (~inflated_frontier_blocked)

        # NEW: optional radiation exclusion (frontiers only), with automatic fallback if too aggressive.
        if (self.enable_radiation_exclusion and self.use_radiation and self._radiation_shape_ok and
            self._rad_exclusion_mask_inflated is not None):
            masked = frontier_mask & (~self._rad_exclusion_mask_inflated)
            if np.count_nonzero(masked) > 0:
                frontier_mask = masked
            else:
                # Keep behaviour: never stall exploration purely due to radiation exclusion.
                rospy.logwarn_throttle(2.0, "[RAD] exclusion mask would remove all frontiers; ignoring exclusion this cycle.")

        ys, xs = np.nonzero(frontier_mask)
        if xs.size == 0:
            return []

        if self.enable_frontier_clustering:
            clusters = self._cluster_frontiers(frontier_mask)
            reps = self._cluster_representatives(clusters)
            if not reps:
                return []
            gx_list, gy_list = zip(*reps)
            xs_s = np.array(gx_list, dtype=np.int32)
            ys_s = np.array(gy_list, dtype=np.int32)
        else:
            min_sep_cells = max(1, int(round(self.frontier_min_goal_separation_m / max(self.map_res, 1e-6))))
            stride = max(1, min_sep_cells)
            bx = (xs // stride).astype(np.int32)
            by = (ys // stride).astype(np.int32)
            key = by.astype(np.int64) * (self.map_w // stride + 1) + bx.astype(np.int64)
            order = np.argsort(key)
            xs_s = xs[order]
            ys_s = ys[order]
            keep = []
            last = None
            for i in range(xs_s.size):
                k = int(key[order[i]])
                if k != last:
                    keep.append(i)
                    last = k
                if len(keep) >= self.frontier_limit:
                    break
            xs_s = xs_s[keep]
            ys_s = ys_s[keep]

        r_cells = max(1, int(round(self.frontier_local_radius_m / max(self.map_res, 1e-6))))
        out = []

        rx = float(self.robot_x)
        ry = float(self.robot_y)
        r_gx, r_gy = self._map_to_grid(rx, ry)

        los_block = self.inflated_blocked_graph if self.frontier_los_use_inflated else self.blocked_for_frontier

        for gx, gy in zip(xs_s.tolist(), ys_s.tolist()):
            gx = int(gx); gy = int(gy)

            x0 = max(0, gx - r_cells); x1 = min(self.map_w - 1, gx + r_cells)
            y0 = max(0, gy - r_cells); y1 = min(self.map_h - 1, gy + r_cells)
            window_unk = self.unknown_mask[y0:y1+1, x0:x1+1]
            denom = int(window_unk.size)
            if denom < self.frontier_min_denom_cells:
                continue
            unk_ratio = float(np.count_nonzero(window_unk)) / float(denom)
            if unk_ratio < self.frontier_unknown_ratio_min or unk_ratio > self.frontier_unknown_ratio_max:
                continue

            fx, fy = grid_to_world(gx, gy, self.map_origin_x, self.map_origin_y, self.map_res)

            if self.enable_frontier_los_precheck:
                if not self._los_grid_ok(r_gx, r_gy, gx, gy, los_block):
                    self._rej.append({"x": fx, "y": fy, "reason": "los_precheck", "score": 0.0})
                    continue

            out.append((gx, gy, fx, fy, unk_ratio))
            if len(out) >= self.frontier_limit:
                break

        return out

    def _los_grid_ok(self, gx0, gy0, gx1, gy1, blocked_mask):
        for x, y in bresenham(gx0, gy0, gx1, gy1):
            if x < 0 or x >= self.map_w or y < 0 or y >= self.map_h:
                return False
            if blocked_mask[y, x]:
                return False
        return True

    # ---------------- clustering ---------------- #

    def _cluster_frontiers(self, frontier_mask):
        h, w = frontier_mask.shape
        seen = np.zeros_like(frontier_mask, dtype=np.uint8)
        clusters = []

        if self.cluster_conn8:
            nbrs = [(-1, -1), (0, -1), (1, -1),
                    (-1,  0),          (1,  0),
                    (-1,  1), (0,  1), (1,  1)]
        else:
            nbrs = [(0, -1), (-1, 0), (1, 0), (0, 1)]

        ys, xs = np.nonzero(frontier_mask)
        for gy, gx in zip(ys.tolist(), xs.tolist()):
            if seen[gy, gx]:
                continue
            q = [(gx, gy)]
            seen[gy, gx] = 1
            cl = []
            while q:
                x, y = q.pop()
                cl.append((x, y))
                for dx, dy in nbrs:
                    nx = x + dx; ny = y + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if seen[ny, nx]:
                        continue
                    if frontier_mask[ny, nx]:
                        seen[ny, nx] = 1
                        q.append((nx, ny))
            if len(cl) >= self.cluster_min_size:
                clusters.append(cl)

        if not clusters:
            clusters = [[(int(x), int(y))] for y, x in zip(ys.tolist(), xs.tolist())][: self.frontier_limit]
        return clusters

    def _cluster_representatives(self, clusters):
        reps = []
        last_goal = self.goal_history[-1] if len(self.goal_history) > 0 else None
        rx = float(self.robot_x)
        ry = float(self.robot_y)
        r_gx, r_gy = self._map_to_grid(rx, ry)

        for cl in clusters:
            cl_s = cl[:: max(1, self.cluster_rep_stride_cells)]
            if not cl_s:
                continue

            scored = []
            for gx, gy in cl_s:
                d = hypot2(gx - r_gx, gy - r_gy)
                s = 0.01 * d
                if last_goal is not None:
                    lgx, lgy = self._map_to_grid(float(last_goal[0]), float(last_goal[1]))
                    dd = hypot2(gx - lgx, gy - lgy)
                    s += self.cluster_bias_to_farther * (dd / max(1.0, d + 1e-6))
                scored.append((s, gx, gy))

            scored.sort(key=lambda t: -t[0])
            for i in range(min(self.cluster_max_reps, len(scored))):
                reps.append((int(scored[i][1]), int(scored[i][2])))

            if len(reps) >= self.frontier_limit:
                break

        return reps[: self.frontier_limit]

    # ---------------- scoring + choosing ---------------- #

    def _choose_frontier(self, frontiers, now):
        rx = float(self.robot_x)
        ry = float(self.robot_y)
        ryaw = float(self.robot_yaw)

        last_goal = self.goal_history[-1] if len(self.goal_history) > 0 else None

        hysteresis_ref = None
        if self.last_motion_heading is not None and self.last_motion_heading_t is not None:
            dt = now - self.last_motion_heading_t
            decay = math.exp(-dt / max(1e-6, self.angular_hysteresis_tau_s))
            hysteresis_ref = (self.last_motion_heading, decay)

        best = None
        best_score = -1e18

        for gx, gy, fx, fy, unk_ratio in frontiers:
            fkey = (int(gx), int(gy))

            if fkey in self.frontier_blacklist:
                self._rej.append({"x": fx, "y": fy, "reason": "blacklist", "score": 0.0})
                continue

            lt = self.frontier_recent_attempt.get(fkey)
            if lt is not None and (now - lt) < self.frontier_attempt_cooldown_s:
                self._rej.append({"x": fx, "y": fy, "reason": "cooldown", "score": 0.0})
                continue

            if self._frontier_is_stuck(fkey, now):
                self._rej.append({"x": fx, "y": fy, "reason": "stuck", "score": 0.0})
                continue

            anchor_pen = 0.0
            if self.goal_snap_to_node:
                ni = self._nearest_node(fx, fy)
                nx, ny = float(self.nodes_xy[ni, 0]), float(self.nodes_xy[ni, 1])
                dn = hypot2(fx - nx, fy - ny)
                if dn > self.frontier_anchor_hard_cap_m:
                    self._rej.append({"x": fx, "y": fy, "reason": "anchor_cap", "score": 0.0})
                    continue
                if dn > self.frontier_anchor_ok_m:
                    anchor_pen = self.frontier_anchor_soft_penalty * (dn - self.frontier_anchor_ok_m)
                fx2, fy2 = nx, ny
            else:
                fx2, fy2 = fx, fy

            dx = fx2 - rx
            dy = fy2 - ry
            dist = hypot2(dx, dy)
            target_yaw = math.atan2(dy, dx)

            score = 0.0
            score += self.unknown_gain_weight * float(unk_ratio)
            score -= self.robot_frontier_distance_weight * dist

            dtheta = abs(wrap_pi(target_yaw - ryaw))
            score += self.heading_weight * (1.0 - dtheta / math.pi)

            score -= self.visited_weight * self._visited_penalty_world(fx2, fy2)

            if last_goal is not None and self.frontier_chain_weight > 0.0:
                dd = hypot2(fx2 - float(last_goal[0]), fy2 - float(last_goal[1]))
                if dd <= self.frontier_chain_dist_m:
                    score += self.frontier_chain_weight * (1.0 - dd / max(1e-6, self.frontier_chain_dist_m))

            if hysteresis_ref is not None and self.angular_hysteresis_weight > 0.0:
                href, decay = hysteresis_ref
                dh = abs(wrap_pi(target_yaw - href))
                score -= self.angular_hysteresis_weight * decay * (dh / math.pi)

            # Radiation: strong bias, plus soft/hard caps
            if self.use_radiation and self._radiation_shape_ok:
                r_here = self._radiation_at_world(fx2, fy2)
                score -= self.radiation_weight * r_here
                if self._radiation_sigma_shape_ok:
                    score -= self.variance_weight * self._radiation_sigma_at_world(fx2, fy2)
                if self.radiation_heading_avoid_weight > 0.0:
                    score -= self.radiation_heading_avoid_weight * self._radiation_heading_grad(rx, ry, target_yaw)

                # Soft cap: steep penalty above soft_cap (pushes planner away aggressively)
                sc = float(self.radiation_goal_soft_cap)
                if sc > 0.0 and r_here > sc:
                    score -= self.radiation_softcap_penalty * (r_here - sc) / max(1e-6, (1.0 - sc))

                # Hard cap: reject (but global fallback exists if everything would be rejected)
                hc = float(self.radiation_goal_hard_cap)
                if self.radiation_hardcap_reject and hc > 0.0 and r_here >= hc:
                    self._rej.append({"x": fx2, "y": fy2, "reason": "rad_hardcap", "score": score})
                    continue

            if last_goal is not None:
                dd = hypot2(fx2 - float(last_goal[0]), fy2 - float(last_goal[1]))
                if dd < self.frontier_min_goal_separation_m * 0.75:
                    score -= self.frontier_repeat_penalty

            score -= anchor_pen

            if score > best_score:
                best_score = score
                best = (fkey, fx2, fy2, gx, gy, score)

        return best

    def _frontier_is_stuck(self, fkey, now):
        rec = self.frontier_fail.get(fkey)
        if not rec:
            return False
        if (now - rec.get("t0", now)) > self.frontier_fail_hard_timeout_s:
            self.frontier_blacklist[fkey] = now + max(self.blacklist_timeout_s, 90.0)
            return True
        if rec.get("progress_max", 0.0) < self.frontier_fail_progress_min:
            if (now - rec.get("t0", now)) > self.frontier_fail_soft_timeout_s:
                return True
        return False

    def _register_frontier_fail(self, fkey, now, reason="fail"):
        rec = self.frontier_fail.get(fkey)
        if rec is None:
            rec = {"t0": now, "n": 0, "progress_max": 0.0, "last_reason": ""}
            self.frontier_fail[fkey] = rec
        rec["n"] += 1
        rec["last_reason"] = reason
        if rec["n"] >= self.blacklist_threshold:
            self.frontier_blacklist[fkey] = now + max(self.blacklist_timeout_s, 30.0)
            rec["n"] = 0

    # ---------------- visited ---------------- #

    def _visited_penalty_world(self, x_map, y_map):
        gx, gy = self._map_to_grid(x_map, y_map)
        r_cells = max(1, int(round(self.visited_radius_m / max(self.map_res, 1e-6))))
        x0 = max(0, gx - r_cells); x1 = min(self.map_w - 1, gx + r_cells)
        y0 = max(0, gy - r_cells); y1 = min(self.map_h - 1, gy + r_cells)
        acc = 0.0
        n = 0
        vis = self.visited
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                acc += vis.get((xx, yy), 0.0)
                n += 1
        return acc / float(max(1, n))

    def _mark_visited_disk(self, gx, gy, radius_m):
        r_cells = max(1, int(round(radius_m / max(self.map_res, 1e-6))))
        x0 = max(0, gx - r_cells); x1 = min(self.map_w - 1, gx + r_cells)
        y0 = max(0, gy - r_cells); y1 = min(self.map_h - 1, gy + r_cells)
        r2 = r_cells * r_cells
        vis = self.visited
        for yy in range(y0, y1 + 1):
            dy = yy - gy
            for xx in range(x0, x1 + 1):
                dx = xx - gx
                if dx * dx + dy * dy <= r2:
                    vis[(xx, yy)] = 1.0

    # ---------------- radiation (MAP frame) ---------------- #

    def _radiation_at_world(self, x_map, y_map):
        if self.radiation is None or not self._radiation_shape_ok:
            return 0.0
        gx, gy = self._map_to_grid(x_map, y_map)
        if gx < 0 or gx >= self.map_w or gy < 0 or gy >= self.map_h:
            return 0.0
        return float(self.radiation[gy, gx])

    def _radiation_sigma_at_world(self, x_map, y_map):
        if self.radiation_sigma is None or not self._radiation_sigma_shape_ok:
            return 0.0
        gx, gy = self._map_to_grid(x_map, y_map)
        if gx < 0 or gx >= self.map_w or gy < 0 or gy >= self.map_h:
            return 0.0
        return float(self.radiation_sigma[gy, gx])

    def _radiation_heading_grad(self, rx, ry, yaw):
        d = self.radiation_heading_sample_dist_m
        x1 = rx + d * math.cos(yaw)
        y1 = ry + d * math.sin(yaw)
        r0 = self._radiation_at_world(rx, ry)
        r1 = self._radiation_at_world(x1, y1)
        return max(0.0, r1 - r0)

    def _radiation_along_segment_mean(self, x0, y0, x1, y1):
        """
        Mean radiation sampled along a straight segment in world coords.
        This makes A* avoidance *much* more faithful than midpoint-only.
        """
        if not (self.use_radiation and self._radiation_shape_ok):
            return 0.0

        seg = hypot2(x1 - x0, y1 - y0)
        if seg <= 1e-6:
            return self._radiation_at_world(x0, y0)

        step = max(0.05, float(self.astar_rad_sample_step_m))
        n = max(2, int(math.ceil(seg / step)) + 1)
        acc = 0.0
        for k in range(n):
            t = float(k) / float(n - 1)
            xs = x0 + t * (x1 - x0)
            ys = y0 + t * (y1 - y0)
            acc += self._radiation_at_world(xs, ys)
        return float(acc / float(n))

    # ---------------- nearest + A* (optimised) ---------------- #

    def _nearest_node(self, x, y):
        if self.kdtree is not None:
            _, idx = self.kdtree.query([x, y], k=1)
            return int(idx)
        d2 = np.sum((self.nodes_xy - np.array([[x, y]], dtype=np.float32)) ** 2, axis=1)
        return int(np.argmin(d2))

    def _heur(self, i, j):
        dx = float(self.nodes_xy[i, 0] - self.nodes_xy[j, 0])
        dy = float(self.nodes_xy[i, 1] - self.nodes_xy[j, 1])
        return math.sqrt(dx * dx + dy * dy)

    def _astar(self, start_i, goal_i):
        if start_i == goal_i:
            return [start_i]

        nodes = self.nodes_xy
        adj = self.adj

        use_rad = (self.use_radiation and self._radiation_shape_ok and
                   (self.radiation_edge_weight > 0.0 or self.astar_rad_integral_weight > 0.0))

        g = defaultdict(lambda: 1e18)
        g[start_i] = 0.0
        came = {}

        h0 = self._heur(start_i, goal_i)
        heap = [(h0, 0.0, start_i)]
        closed = set()

        while heap:
            fcur, gcur, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            if cur == goal_i:
                return self._reconstruct(came, cur)
            closed.add(cur)

            cx = float(nodes[cur, 0])
            cy = float(nodes[cur, 1])

            for nb in adj[cur]:
                if nb in closed:
                    continue

                nx = float(nodes[nb, 0])
                ny = float(nodes[nb, 1])
                dx = cx - nx
                dy = cy - ny
                base = math.sqrt(dx * dx + dy * dy)

                cost = base

                if use_rad:
                    # Keep your existing midpoint penalty...
                    if self.radiation_edge_weight > 0.0:
                        mx = 0.5 * (cx + nx)
                        my = 0.5 * (cy + ny)
                        cost += self.radiation_edge_weight * self._radiation_at_world(mx, my)

                    # ...and add an integrated segment mean penalty (stronger/cleaner avoidance).
                    if self.astar_rad_integral_weight > 0.0:
                        rmean = self._radiation_along_segment_mean(cx, cy, nx, ny)
                        cost += self.astar_rad_integral_weight * rmean * base

                tentative = gcur + cost
                if tentative < g[nb]:
                    g[nb] = tentative
                    came[nb] = cur
                    fnb = tentative + self._heur(nb, goal_i)
                    heapq.heappush(heap, (fnb, tentative, nb))

        return None

    def _reconstruct(self, came, cur):
        out = [cur]
        while cur in came:
            cur = came[cur]
            out.append(cur)
        out.reverse()
        return out

    # ---------------- corridor building + strict verification ---------------- #

    def _nodes_to_path(self, idxs):
        path = Path()
        path.header.frame_id = self._effective_map_frame()
        path.header.stamp = rospy.Time.now()

        ps0 = PoseStamped()
        ps0.header = path.header
        ps0.pose.position.x = float(self.robot_x)
        ps0.pose.position.y = float(self.robot_y)
        ps0.pose.orientation = make_quat_yaw(self.robot_yaw)
        path.poses.append(ps0)

        for i in idxs:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(self.nodes_xy[i, 0])
            ps.pose.position.y = float(self.nodes_xy[i, 1])
            ps.pose.orientation = make_quat_yaw(0.0)
            path.poses.append(ps)

        if len(path.poses) >= 2:
            for k in range(len(path.poses) - 1):
                x0 = path.poses[k].pose.position.x
                y0 = path.poses[k].pose.position.y
                x1 = path.poses[k + 1].pose.position.x
                y1 = path.poses[k + 1].pose.position.y
                yaw = math.atan2(y1 - y0, x1 - x0)
                path.poses[k].pose.orientation = make_quat_yaw(yaw)
            path.poses[-1].pose.orientation = path.poses[-2].pose.orientation

        return path

    def _strict_verify_corridor(self, path: Path):
        if path is None or len(path.poses) < 2:
            return False, "empty"

        min_clear_cells = max(1, int(round(self.strict_min_clearance_m / max(self.map_res, 1e-6))))
        step_m = max(0.02, float(self.strict_segment_step_m))

        sx = float(path.poses[0].pose.position.x)
        sy = float(path.poses[0].pose.position.y)
        ex = float(path.poses[-1].pose.position.x)
        ey = float(path.poses[-1].pose.position.y)
        total = hypot2(ex - sx, ey - sy)
        if total < 1e-6:
            return False, "degenerate"

        worst_backtrack = 0.0
        inflated = self.inflated_blocked_graph
        distc = self.dist_to_blocked_graph_cells

        def clearance_ok_world(xm, ym):
            gx, gy = self._map_to_grid(xm, ym)
            if gx < 0 or gx >= self.map_w or gy < 0 or gy >= self.map_h:
                return False
            if inflated[gy, gx]:
                return False
            if distc is not None:
                if float(distc[gy, gx]) < float(min_clear_cells):
                    return False
            return True

        for i in range(len(path.poses) - 1):
            x0 = float(path.poses[i].pose.position.x)
            y0 = float(path.poses[i].pose.position.y)
            x1 = float(path.poses[i + 1].pose.position.x)
            y1 = float(path.poses[i + 1].pose.position.y)

            seg = hypot2(x1 - x0, y1 - y0)
            n = max(1, int(math.ceil(seg / step_m)))

            if self.strict_verify_segment_los:
                g0x, g0y = self._map_to_grid(x0, y0)
                g1x, g1y = self._map_to_grid(x1, y1)
                if not self._los_grid_ok(g0x, g0y, g1x, g1y, inflated):
                    return False, "segment_los"

            if self.strict_verify_pose_clearance:
                for k in range(n + 1):
                    t = float(k) / float(n)
                    xs = x0 + t * (x1 - x0)
                    ys = y0 + t * (y1 - y0)
                    if not clearance_ok_world(xs, ys):
                        return False, "clearance"

            d0 = hypot2(ex - x0, ey - y0)
            d1 = hypot2(ex - x1, ey - y1)
            back = max(0.0, d1 - d0)
            worst_backtrack += back

        if worst_backtrack > self.strict_max_backtrack_frac * total:
            return False, "backtrack"

        return True, "ok"

    def _path_length(self, path: Path):
        if path is None or len(path.poses) < 2:
            return 0.0
        acc = 0.0
        for i in range(len(path.poses) - 1):
            x0 = path.poses[i].pose.position.x
            y0 = path.poses[i].pose.position.y
            x1 = path.poses[i + 1].pose.position.x
            y1 = path.poses[i + 1].pose.position.y
            acc += hypot2(x1 - x0, y1 - y0)
        return acc

    # ---------------- viz ---------------- #

    def _maybe_publish_viz(self, now):
        if not self.publish_markers:
            return
        if now - self._last_marker_pub_t < (1.0 / max(self.marker_rate_hz, 0.1)):
            return
        self._last_marker_pub_t = now
        self._publish_markers(now)

    def _publish_markers(self, now):
        ma = MarkerArray()
        stamp = rospy.Time.now()
        frame = self._effective_map_frame()

        m0 = Marker()
        m0.header.frame_id = frame
        m0.header.stamp = stamp
        m0.action = Marker.DELETEALL
        ma.markers.append(m0)

        # graph edges
        if self.nodes_xy.shape[0] > 0 and self.edges:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = "graph"
            m.id = 1
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.02
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 1.0
            m.color.a = 0.55
            pts = []
            for i, j in self.edges:
                p = Point(); p.x = float(self.nodes_xy[i, 0]); p.y = float(self.nodes_xy[i, 1]); p.z = 0.03
                q = Point(); q.x = float(self.nodes_xy[j, 0]); q.y = float(self.nodes_xy[j, 1]); q.z = 0.03
                pts.append(p); pts.append(q)
            m.points = pts
            ma.markers.append(m)

        if self.viz_nodes and self.nodes_xy.shape[0] > 0:
            mn = Marker()
            mn.header.frame_id = frame
            mn.header.stamp = stamp
            mn.ns = "nodes"
            mn.id = 20
            mn.type = Marker.POINTS
            mn.action = Marker.ADD
            mn.scale.x = 0.06
            mn.scale.y = 0.06
            mn.color.r = 1.0
            mn.color.g = 0.0
            mn.color.b = 0.0
            mn.color.a = 0.55
            mn.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in self.nodes_xy.tolist()]
            ma.markers.append(mn)

        if self.viz_frontier_candidates and self._last_frontier_candidates_for_viz:
            mf = Marker()
            mf.header.frame_id = frame
            mf.header.stamp = stamp
            mf.ns = "frontier_candidates"
            mf.id = 21
            mf.type = Marker.POINTS
            mf.action = Marker.ADD
            mf.scale.x = 0.08
            mf.scale.y = 0.08
            mf.color.r = 0.2
            mf.color.g = 0.9
            mf.color.b = 1.0
            mf.color.a = 0.45
            mf.points = [Point(x=float(x), y=float(y), z=0.04) for x, y, _ in self._last_frontier_candidates_for_viz[:800]]
            ma.markers.append(mf)

        # active goal + corridor
        if self.active_goal is not None:
            g = self.active_goal["goal"].pose.position
            mg = Marker()
            mg.header.frame_id = frame
            mg.header.stamp = stamp
            mg.ns = "goal"
            mg.id = 2
            mg.type = Marker.SPHERE
            mg.action = Marker.ADD
            mg.pose.position.x = g.x
            mg.pose.position.y = g.y
            mg.pose.position.z = 0.08
            mg.scale.x = 0.18
            mg.scale.y = 0.18
            mg.scale.z = 0.18
            mg.color.r = 1.0
            mg.color.g = 0.6
            mg.color.b = 0.1
            mg.color.a = 0.85
            ma.markers.append(mg)

            path = self.active_goal.get("corridor")
            if path is not None and len(path.poses) >= 2:
                ml = Marker()
                ml.header.frame_id = frame
                ml.header.stamp = stamp
                ml.ns = "corridor"
                ml.id = 3
                ml.type = Marker.LINE_STRIP
                ml.action = Marker.ADD
                ml.scale.x = 0.05
                ml.color.r = 1.0
                ml.color.g = 0.0
                ml.color.b = 1.0
                ml.color.a = 0.75
                pts = []
                for ps in path.poses:
                    p = Point()
                    p.x = ps.pose.position.x
                    p.y = ps.pose.position.y
                    p.z = 0.05
                    pts.append(p)
                ml.points = pts
                ma.markers.append(ml)

            if self.viz_heading_cones:
                self._add_heading_cone(ma, stamp)

        if self.debug_rejected_markers and len(self._rej) > 0:
            self._add_rejected_markers(ma, stamp)

        self.pub_markers.publish(ma)

    def _add_heading_cone(self, ma: MarkerArray, stamp):
        rx = float(self.robot_x)
        ry = float(self.robot_y)
        yaw = float(self.robot_yaw)
        half = math.radians(self.viz_heading_cone_half_angle_deg)
        L = float(self.viz_heading_cone_len_m)

        p0 = Point(); p0.x = rx; p0.y = ry; p0.z = 0.05
        p1 = Point(); p1.x = rx + L * math.cos(yaw - half); p1.y = ry + L * math.sin(yaw - half); p1.z = 0.05
        p2 = Point(); p2.x = rx + L * math.cos(yaw + half); p2.y = ry + L * math.sin(yaw + half); p2.z = 0.05

        m = Marker()
        m.header.frame_id = self._effective_map_frame()
        m.header.stamp = stamp
        m.ns = "heading_cone"
        m.id = 10
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.03
        m.color.r = 0.2
        m.color.g = 0.8
        m.color.b = 0.2
        m.color.a = 0.6
        m.points = [p0, p1, p0, p2, p1, p2]
        ma.markers.append(m)

    def _add_rejected_markers(self, ma: MarkerArray, stamp):
        buckets = defaultdict(list)
        for r in list(self._rej):
            buckets[r["reason"]].append(r)

        base_id = 100
        for idx, (reason, items) in enumerate(buckets.items()):
            m = Marker()
            m.header.frame_id = self._effective_map_frame()
            m.header.stamp = stamp
            m.ns = "rejected"
            m.id = base_id + idx
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.10
            m.scale.y = 0.10

            if "blacklist" in reason:
                m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.0, 0.0, 0.55)
            elif "cooldown" in reason:
                m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.5, 0.0, 0.55)
            elif "los_precheck" in reason:
                m.color.r, m.color.g, m.color.b, m.color.a = (0.6, 0.6, 0.6, 0.55)
            elif "rad_hardcap" in reason:
                m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.2, 0.2, 0.55)
            elif "strict" in reason:
                m.color.r, m.color.g, m.color.b, m.color.a = (0.9, 0.0, 0.9, 0.55)
            elif "no_path" in reason:
                m.color.r, m.color.g, m.color.b, m.color.a = (0.0, 0.2, 1.0, 0.55)
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = (0.8, 0.8, 0.2, 0.55)

            pts = []
            for it in items[: self.debug_rejected_keep]:
                p = Point()
                p.x = float(it["x"])
                p.y = float(it["y"])
                p.z = 0.06
                pts.append(p)
            m.points = pts
            ma.markers.append(m)

            if self.debug_rejected_text and pts:
                t = Marker()
                t.header.frame_id = self._effective_map_frame()
                t.header.stamp = stamp
                t.ns = "rejected_text"
                t.id = 500 + idx
                t.type = Marker.TEXT_VIEW_FACING
                t.action = Marker.ADD
                t.scale.z = 0.18
                t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0; t.color.a = 0.8
                t.pose.position.x = pts[0].x
                t.pose.position.y = pts[0].y
                t.pose.position.z = 0.25
                t.text = reason
                ma.markers.append(t)


def main():
    rospy.init_node("exploration_policy", anonymous=False)
    ExplorerPolicy()
    rospy.spin()


if __name__ == "__main__":
    main()