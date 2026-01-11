#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ExploreSim – Offline Exploration + Radiation Simulation (FULL)

Features (preserved):
- Loads ground-truth map (PGM + YAML)
- Correct ROS map origin handling
- Robot starts at MAP ORIGIN (0,0) in map frame
- Fog-of-war belief map via raycasting
- Inflated obstacle grid
- A* planning on belief map
- Smooth waypoint following
- Offline precomputed radiation field (loaded from disk)
- Radiation visualisation as OccupancyGrid
- Radiation sensor output (/sim_rad_sensor)
- Full TF tree: map -> odom -> base_link

Upgrades (added, no feature removals):
- Fix clamp() NameError
- Adaptive inflation radius: prefer larger clearance, back off only if necessary
- Path smoothing (line-of-sight pruning) to remove grid zig-zag
- Much smoother, more realistic robot motion:
  * pure pursuit style tracking of the path
  * acceleration-limited unicycle dynamics
  * stable dt from ROS timer event
"""

import os
import math
import heapq
import yaml
from collections import deque

import numpy as np
import rospy
import rospkg
import tf2_ros

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32
from tf.transformations import quaternion_from_euler


# ============================================================
# Small helpers
# ============================================================

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


# ============================================================
# Bresenham line tracing (fog-of-war + LOS checks)
# ============================================================

def bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0

    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


# ============================================================
# A* planner (8-connected, no corner cutting)
# ============================================================

def astar(grid, start, goal):
    """
    grid: bool/uint8 where 1=True means occupied
    start, goal: (x,y) in grid coords
    """
    h, w = grid.shape
    sx, sy = start
    gx, gy = goal

    # bounds safety
    if not (0 <= sx < w and 0 <= sy < h):
        return None
    if not (0 <= gx < w and 0 <= gy < h):
        return None

    if grid[sy, sx] or grid[gy, gx]:
        return None

    moves = [
        (1, 0, 1.0), (-1, 0, 1.0),
        (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.414), (-1, 1, 1.414),
        (1, -1, 1.414), (-1, -1, 1.414),
    ]

    def heuristic(x, y):
        return math.hypot(gx - x, gy - y)

    gscore = np.full((h, w), np.inf)
    parent = np.full((h, w, 2), -1, dtype=int)
    closed = np.zeros((h, w), dtype=bool)

    gscore[sy, sx] = 0.0
    pq = [(heuristic(sx, sy), sx, sy)]

    while pq:
        _, x, y = heapq.heappop(pq)
        if closed[y, x]:
            continue
        closed[y, x] = True

        if (x, y) == (gx, gy):
            path = [(x, y)]
            while (x, y) != (sx, sy):
                x, y = parent[y, x]
                path.append((x, y))
            return path[::-1]

        for dx, dy, cost in moves:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue

            if grid[ny, nx]:
                continue

            # forbid cutting corners on diagonals
            if dx != 0 and dy != 0:
                if grid[y, nx] or grid[ny, x]:
                    continue

            ng = gscore[y, x] + cost
            if ng < gscore[ny, nx]:
                gscore[ny, nx] = ng
                parent[ny, nx] = [x, y]
                heapq.heappush(pq, (ng + heuristic(nx, ny), nx, ny))

    return None


# ============================================================
# Obstacle inflation (grid-radius, BFS distance)
# ============================================================

def inflate(grid, radius_cells):
    """
    grid: uint8/bool occupancy (1=occupied)
    radius_cells: Manhattan-ish expansion in cells
    """
    if radius_cells <= 0:
        return grid.copy()

    h, w = grid.shape
    inflated = grid.copy()
    dist = np.full((h, w), -1, dtype=int)
    q = deque()

    ys, xs = np.where(grid == 1)
    for y, x in zip(ys, xs):
        q.append((x, y))
        dist[y, x] = 0

    neigh = [(1,0),(-1,0),(0,1),(0,-1)]

    while q:
        x, y = q.popleft()
        d = dist[y, x]
        if d >= radius_cells:
            continue
        for dx, dy in neigh:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if dist[ny, nx] == -1:
                    dist[ny, nx] = d + 1
                    inflated[ny, nx] = 1
                    q.append((nx, ny))
    return inflated


# ============================================================
# Path smoothing (LOS pruning on inflated grid)
# ============================================================

def smooth_path_los(path, occ_bool):
    """
    Removes zig-zags by skipping intermediate points when line-of-sight is free.
    occ_bool: True means obstacle.
    """
    if not path or len(path) < 3:
        return path

    smoothed = [path[0]]
    i = 0
    n = len(path)

    while i < n - 1:
        j = n - 1
        # find farthest j with clear LOS
        while j > i + 1:
            ok = True
            x0, y0 = path[i]
            x1, y1 = path[j]
            for x, y in bresenham(x0, y0, x1, y1):
                if occ_bool[y, x]:
                    ok = False
                    break
            if ok:
                break
            j -= 1

        smoothed.append(path[j])
        i = j

    return smoothed


# ============================================================
# Main node
# ============================================================

class ExploreSimNode:
    def __init__(self):
        rospy.init_node("explore_sim_node")

        # ---------------- Frames ----------------
        self.map_frame  = "map"
        self.odom_frame = "odom"
        self.base_frame = "base_link"

        # ---------------- Parameters ----------------
        self.map_name = rospy.get_param("~map", "reactor_room_sim")
        self.sensor_range = float(rospy.get_param("~sensor_range_m", 15.0))
        self.num_rays = int(rospy.get_param("~num_rays", 360))

        # Planning / clearance
        self.inflation_radius_m = float(rospy.get_param("~inflation_radius_m", 0.15))
        self.inflation_max_factor = float(rospy.get_param("~inflation_max_factor", 2.5))
        self.inflation_min_factor = float(rospy.get_param("~inflation_min_factor", 0.35))

        # Path following / motion
        self.goal_tol = float(rospy.get_param("~goal_tolerance", 0.25))

        # Unicycle dynamics + realism knobs
        self.max_v = float(rospy.get_param("~max_v", 0.6))            # m/s
        self.max_omega = float(rospy.get_param("~max_omega", 1.5))    # rad/s
        self.lin_acc = float(rospy.get_param("~lin_acc", 0.8))        # m/s^2
        self.ang_acc = float(rospy.get_param("~ang_acc", 2.0))        # rad/s^2

        # Pure-pursuit-ish tracking
        self.lookahead_m = float(rospy.get_param("~lookahead_m", 0.8))
        self.k_yaw = float(rospy.get_param("~k_yaw", 2.2))
        self.k_curve_slow = float(rospy.get_param("~k_curve_slow", 1.0))

        # Update rates
        self.tf_rate = float(rospy.get_param("~tf_rate_hz", 20.0))     # TF publishing
        self.tick_rate = float(rospy.get_param("~tick_rate_hz", 10.0)) # planning + raycasts

        # ---------------- Paths ----------------
        rospack = rospkg.RosPack()
        base = rospack.get_path("exploration_pkg")
        self.map_dir = f"{base}/maps/{self.map_name}"

        # ==================================================
        # Load map
        # ==================================================
        with open(f"{self.map_dir}/{self.map_name}.yaml") as f:
            cfg = yaml.safe_load(f)

        self.res = float(cfg["resolution"])
        self.origin = cfg["origin"]

        import cv2
        img = cv2.imread(f"{self.map_dir}/{cfg['image']}", cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("Failed to load map image")
        img = np.flipud(img)

        self.h, self.w = img.shape
        norm = img.astype(np.float32) / 255.0

        self.gt = np.full((self.h, self.w), -1, np.int8)
        self.gt[norm < 0.2] = 100
        self.gt[norm > 0.8] = 0

        rospy.loginfo("[ExploreSim] Map loaded (%dx%d), origin=(%.2f, %.2f)",
                      self.w, self.h, self.origin[0], self.origin[1])

        # ==================================================
        # Load radiation field (offline)
        # ==================================================
        rad_path = f"{self.map_dir}/radiation_field.npy"
        if not os.path.exists(rad_path):
            raise RuntimeError("radiation_field.npy not found")

        self.rad_map = np.load(rad_path).astype(np.float32)
        vmax = float(np.max(self.rad_map))
        rospy.loginfo("[ExploreSim] Radiation field loaded (max=%.3g)", vmax)

        # Prepare RViz-friendly radiation occupancy
        rad = self.rad_map.copy()
        rad[self.gt == 100] = 0.0
        rad = np.log1p(rad)
        if np.max(rad) > 1e-12:
            rad = rad / np.max(rad) * 100.0
        self.rad_occ = rad.astype(np.int8)
        self.rad_occ[self.gt == -1] = -1
        self.rad_occ[self.gt == 100] = 100

        # ==================================================
        # Robot pose — MAP FRAME ORIGIN (FIXED)
        # ==================================================
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # motion state
        self.v = 0.0
        self.omega = 0.0

        # ==================================================
        # State
        # ==================================================
        self.belief = np.full((self.h, self.w), -1, np.int8)
        self.goal = None

        self.plan = None           # list[(gx,gy)]
        self.plan_world = None     # list[(wx,wy)]
        self.plan_idx = 0
        self.used_infl_cells = None

        # ==================================================
        # Publishers / Subscribers
        # ==================================================
        self.pub_map      = rospy.Publisher("/map", OccupancyGrid, queue_size=1, latch=True)
        self.pub_inflated = rospy.Publisher("/explore_sim/inflated_map", OccupancyGrid, queue_size=1, latch=True)
        self.pub_path     = rospy.Publisher("/explore_sim/plan", Path, queue_size=1)
        self.pub_radmap   = rospy.Publisher("/explore_sim/radiation_map", OccupancyGrid, queue_size=1, latch=True)
        self.pub_rad      = rospy.Publisher("/sim_rad_sensor", Float32, queue_size=1)

        rospy.Subscriber("/exploration/goal", PoseStamped, self._goal_cb)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._goal_cb)

        # ==================================================
        # TF
        # ==================================================
        self.tf_static = tf2_ros.StaticTransformBroadcaster()
        self.tf_dyn = tf2_ros.TransformBroadcaster()
        self._publish_static_tf()

        self._publish_radiation_map()

        # Timers
        self._last_tick_wall = rospy.Time.now().to_sec()
        rospy.Timer(rospy.Duration(1.0 / self.tf_rate), self._publish_tf)
        rospy.Timer(rospy.Duration(1.0 / self.tick_rate), self._tick)

        rospy.loginfo("[ExploreSim] Initialised correctly")

    # ==================================================
    # TF
    # ==================================================
    def _publish_static_tf(self):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.rotation.w = 1.0
        self.tf_static.sendTransform(t)

    def _publish_tf(self, _):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        q = quaternion_from_euler(0, 0, self.yaw)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tf_dyn.sendTransform(t)

    # ==================================================
    # Helpers
    # ==================================================
    def world_to_grid(self, x, y):
        return int((x - self.origin[0]) / self.res), int((y - self.origin[1]) / self.res)

    def grid_to_world(self, gx, gy):
        return (
            self.origin[0] + (gx + 0.5) * self.res,
            self.origin[1] + (gy + 0.5) * self.res
        )

    # ==================================================
    # Callbacks
    # ==================================================
    def _goal_cb(self, msg):
        if msg.header.frame_id.strip("/") != self.map_frame:
            rospy.logwarn_throttle(1.0, "[ExploreSim] Goal ignored (frame='%s', expected '%s')",
                                   msg.header.frame_id, self.map_frame)
            return
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.plan = None
        self.plan_world = None
        self.plan_idx = 0
        self.used_infl_cells = None

    # ==================================================
    # Main loop
    # ==================================================
    def _tick(self, _):
        # --- fog-of-war perception ---
        self._cast_rays()
        self._publish_map()

        # --- planning + motion ---
        if self.goal:
            self._plan_if_needed()
            self._follow_plan_smooth()

        # --- radiation sensor ---
        self._publish_radiation_sensor()

    # ==================================================
    # Fog-of-war
    # ==================================================
    def _cast_rays(self):
        gx, gy = self.world_to_grid(self.x, self.y)
        maxc = int(self.sensor_range / self.res)

        # If robot out of bounds, nothing to do
        if not (0 <= gx < self.w and 0 <= gy < self.h):
            return

        for i in range(self.num_rays):
            ang = 2 * math.pi * i / self.num_rays
            ex = gx + int(maxc * math.cos(ang))
            ey = gy + int(maxc * math.sin(ang))

            for cx, cy in bresenham(gx, gy, ex, ey):
                if not (0 <= cx < self.w and 0 <= cy < self.h):
                    break
                if self.gt[cy, cx] == 100:
                    self.belief[cy, cx] = 100
                    break
                self.belief[cy, cx] = 0

    # ==================================================
    # Planning (adaptive inflation + smoothing)
    # ==================================================
    def _plan_if_needed(self):
        if self.plan is not None:
            return

        sx, sy = self.world_to_grid(self.x, self.y)
        gx, gy = self.world_to_grid(*self.goal)

        # use only "known occupied" cells from belief
        occ = (self.belief == 100).astype(np.uint8)

        base_cells = int(self.inflation_radius_m / self.res)
        base_cells = max(0, base_cells)

        # Prefer larger clearance first, back off only if no path
        max_cells = int(math.ceil(base_cells * self.inflation_max_factor))
        min_cells = int(math.floor(base_cells * self.inflation_min_factor))
        min_cells = max(0, min_cells)

        chosen = None
        chosen_infl = None
        chosen_r = None

        # Start big -> go smaller until solvable
        for r in range(max_cells, min_cells - 1, -1):
            infl = inflate(occ, r)
            p = astar(infl.astype(bool), (sx, sy), (gx, gy))
            if p:
                chosen = p
                chosen_infl = infl
                chosen_r = r
                break

        # If still none, try no inflation as last resort
        if chosen is None:
            infl = occ.copy()
            p = astar(infl.astype(bool), (sx, sy), (gx, gy))
            if p:
                chosen = p
                chosen_infl = infl
                chosen_r = 0

        if chosen is None:
            rospy.logwarn_throttle(1.0, "[ExploreSim] A* failed (even after backing off inflation).")
            self.plan = None
            self.plan_world = None
            self.plan_idx = 0
            return

        # Smooth the grid path using LOS pruning on the chosen inflated grid
        smoothed = smooth_path_los(chosen, chosen_infl.astype(bool))

        self.plan = smoothed
        self.plan_idx = 0
        self.used_infl_cells = chosen_r

        # Publish inflated map (for debugging in RViz)
        self._publish_occgrid((chosen_infl * 100).astype(np.int8), self.pub_inflated)

        # Publish path in RViz
        self._publish_path(self.plan)

        # Precompute world coords for smoother tracking
        self.plan_world = [self.grid_to_world(px, py) for (px, py) in self.plan]

        rospy.loginfo("[ExploreSim] Planned with inflation=%d cells (base=%d, max=%d). Path len: %d -> %d",
                      chosen_r, base_cells, max_cells, len(chosen), len(smoothed))

    # ==================================================
    # Smooth motion (pure pursuit + accel-limited unicycle)
    # ==================================================
    def _follow_plan_smooth(self):
        if not self.plan_world or self.plan_idx >= len(self.plan_world):
            self.v = 0.0
            self.omega = 0.0
            return

        # robust dt from wall clock
        now = rospy.Time.now().to_sec()
        dt = now - self._last_tick_wall
        self._last_tick_wall = now

        # clamp dt to avoid huge jumps if callback stalls
        dt = clamp(dt, 1e-3, 0.2)

        # If near goal, stop cleanly
        gx, gy = self.goal
        if math.hypot(gx - self.x, gy - self.y) <= self.goal_tol:
            self.v = 0.0
            self.omega = 0.0
            return

        # Advance plan index if we've reached current waypoint
        while self.plan_idx < len(self.plan_world):
            wx, wy = self.plan_world[self.plan_idx]
            if math.hypot(wx - self.x, wy - self.y) <= self.goal_tol:
                self.plan_idx += 1
            else:
                break

        if self.plan_idx >= len(self.plan_world):
            self.v = 0.0
            self.omega = 0.0
            return

        # Lookahead target: walk forward along path until lookahead distance
        target_x, target_y = self.plan_world[self.plan_idx]
        accum = 0.0
        px, py = self.x, self.y

        # Start from current index, march forward
        for k in range(self.plan_idx, len(self.plan_world)):
            nx, ny = self.plan_world[k]
            seg = math.hypot(nx - px, ny - py)
            accum += seg
            px, py = nx, ny
            target_x, target_y = nx, ny
            if accum >= self.lookahead_m:
                break

        # Heading control
        desired_yaw = math.atan2(target_y - self.y, target_x - self.x)
        yaw_err = wrap_pi(desired_yaw - self.yaw)

        # Curvature proxy: big yaw error -> slow down
        curve_slow = clamp(1.0 - self.k_curve_slow * abs(yaw_err), 0.0, 1.0)
        v_des = self.max_v * curve_slow
        omega_des = clamp(self.k_yaw * yaw_err, -self.max_omega, self.max_omega)

        # Acceleration limits on v and omega
        dv = clamp(v_des - self.v, -self.lin_acc * dt, self.lin_acc * dt)
        domega = clamp(omega_des - self.omega, -self.ang_acc * dt, self.ang_acc * dt)
        self.v += dv
        self.omega += domega

        # Integrate unicycle motion (smooth, continuous)
        self.yaw = wrap_pi(self.yaw + self.omega * dt)
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt

    # ==================================================
    # Publishing
    # ==================================================
    def _publish_occgrid(self, grid, pub):
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = self.res
        msg.info.width = self.w
        msg.info.height = self.h
        msg.info.origin.position.x = self.origin[0]
        msg.info.origin.position.y = self.origin[1]
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.reshape(-1).tolist()
        pub.publish(msg)

    def _publish_map(self):
        self._publish_occgrid(self.belief, self.pub_map)

    def _publish_path(self, path):
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = rospy.Time.now()
        for gx, gy in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = self.grid_to_world(gx, gy)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def _publish_radiation_map(self):
        self._publish_occgrid(self.rad_occ, self.pub_radmap)

    def _publish_radiation_sensor(self):
        gx, gy = self.world_to_grid(self.x, self.y)
        if 0 <= gx < self.w and 0 <= gy < self.h:
            self.pub_rad.publish(Float32(float(self.rad_map[gy, gx])))


if __name__ == "__main__":
    ExploreSimNode()
    rospy.spin()