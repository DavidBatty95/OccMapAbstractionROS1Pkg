#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rad_GPR_node.py (v5.2-reachable-local-gp)

Fixes vs v5.1:
- Prevents predicting through / behind walls by ONLY updating cells that are
  reachable from the robot within a local radius (BFS flood-fill on free cells).
- Reduces smoothing for sharper, more localised structure.
- Improves dynamic range so high intensity reaches 100 (helps RViz show "hot" colours).

Model:
- Persistent global_heat (monotonic envelope; no fading).
- Local GPR update only (bounded compute).
- IDW floor as conservative baseline in the local region.
- Reachability mask gates both query cells and training samples (optional).

ROS1 (Noetic) + sklearn + scipy.
"""

import rospy
import numpy as np
from collections import deque

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

from gazebo_radiation_plugins.msg import Simulated_Radiation_Msg

import tf2_ros

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.ndimage import gaussian_filter

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None


class RadiationGPMapper:
    def __init__(self):
        rospy.init_node("rad_GPR_node", anonymous=False)

        # ---------------- Topics / frames ----------------
        self.map_topic  = rospy.get_param("~map_topic", "/map")
        self.meas_topic = rospy.get_param("~meas_topic", "/radiation_sensor_plugin/sensor_0")
        self.frame_id   = rospy.get_param("~frame_id", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # ---------------- Occupancy thresholds ----------------
        self.free_max = int(rospy.get_param("~free_max", 25))
        self.occ_min  = int(rospy.get_param("~occ_min", 65))

        # ---------------- Timing ----------------
        self.publish_period_s  = float(rospy.get_param("~publish_period_s", 0.35))
        self.gp_refit_period_s = float(rospy.get_param("~gp_refit_period_s", 1.5))
        self.last_fit_t = 0.0

        # ---------------- Local region ----------------
        self.local_radius_m  = float(rospy.get_param("~local_radius_m", 6.0))
        self.local_margin_m  = float(rospy.get_param("~local_margin_m", 2.0))
        self.coarse_factor   = int(rospy.get_param("~coarse_factor", 2))
        self.max_query_points = int(rospy.get_param("~max_query_points", 6000))

        # Reachability gating
        self.use_reachable_mask = bool(rospy.get_param("~use_reachable_mask", True))

        # ---------------- Sample compression ----------------
        self.voxel_size_m = float(rospy.get_param("~voxel_size_m", 0.30))
        self.max_voxels   = int(rospy.get_param("~max_voxels", 15000))  # <=0 disables cap

        # ---------------- Field shaping (make highs hit 100) ----------------
        # Use a LOWER percentile for scaling so high values saturate (gives "red/hot" in RViz).
        self.scale_percentile = float(rospy.get_param("~scale_percentile", 80.0))
        self.scale_ema_alpha  = float(rospy.get_param("~scale_ema_alpha", 0.96))
        self.global_scale = 1.0

        # Contrast curve: y = x/(x+k) (boosts high-end without nuking low-end)
        self.contrast_k = float(rospy.get_param("~contrast_k", 0.25))

        self.gamma       = float(rospy.get_param("~heatmap_gamma", 1.0))   # keep near 1 for realism
        self.min_visible = float(rospy.get_param("~min_visible", 0.01))

        # Smoothing: keep small (or disable) to avoid leaking behind walls / over-smoothing structure
        self.smooth_sigma_cells = float(rospy.get_param("~smooth_sigma_cells", 0.5))  # 0 disables
        self.idw_floor_gain = float(rospy.get_param("~idw_floor_gain", 0.90))

        # Sigma publish
        self.publish_sigma = bool(rospy.get_param("~publish_sigma", True))

        # ---------------- GP hyperparameters ----------------
        length_scale = float(rospy.get_param("~length_scale", 1.2))
        noise_level  = float(rospy.get_param("~noise_level", 0.20))

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=length_scale, nu=1.5)
            + WhiteKernel(noise_level=noise_level)
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            optimizer=None
        )

        # ---------------- Map state ----------------
        self.map_info  = None
        self.free_mask = None  # float32 1 for free, 0 otherwise
        self.global_heat  = None
        self.global_sigma = None

        # ---------------- Voxel store: key -> [sum_x, sum_y, sum_v, count] ----------------
        self.vox = {}

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ---------------- Publishers ----------------
        self.cost_pub   = rospy.Publisher("/radiation_costmap", OccupancyGrid, queue_size=1, latch=True)
        self.sig_pub    = rospy.Publisher("/radiation_sigma",   OccupancyGrid, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("/radiation_markers", MarkerArray,   queue_size=1)

        rospy.Subscriber(self.map_topic,  OccupancyGrid,           self.map_cb,  queue_size=1)
        rospy.Subscriber(self.meas_topic, Simulated_Radiation_Msg, self.meas_cb, queue_size=300)

        rospy.Timer(rospy.Duration(self.publish_period_s), self._timer)

        rospy.loginfo("rad_GPR_node v5.2 running (reachable-gated local GP, no wall-leak, hotter highs)")

    # ======================================================
    # Map
    # ======================================================

    def map_cb(self, msg):
        self.map_info = msg.info
        raw = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        free = (raw >= 0) & (raw <= self.free_max) & (raw < self.occ_min)
        self.free_mask = free.astype(np.float32)

        h, w = msg.info.height, msg.info.width
        if self.global_heat is None or self.global_heat.shape != (h, w):
            self.global_heat  = np.zeros((h, w), dtype=np.float32)
            self.global_sigma = np.zeros((h, w), dtype=np.float32)

    # ======================================================
    # Measurements -> voxel aggregate
    # ======================================================

    def meas_cb(self, msg):
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        v = max(0.0, float(msg.value))

        vs = self.voxel_size_m
        ix = int(np.floor(x / vs))
        iy = int(np.floor(y / vs))
        key = (ix, iy)

        if key in self.vox:
            sx, sy, sv, c = self.vox[key]
            self.vox[key] = [sx + x, sy + y, sv + v, c + 1]
        else:
            self.vox[key] = [x, y, v, 1]

        if self.max_voxels > 0 and len(self.vox) > self.max_voxels:
            self.vox.pop(next(iter(self.vox)))

        # Update global scale (use lower percentile -> saturates highs -> "red")
        vals = self._voxel_values()
        if vals.size >= 10:
            p = float(np.percentile(vals, self.scale_percentile))
            p = max(p, 1e-3)
            self.global_scale = self.scale_ema_alpha * self.global_scale + (1.0 - self.scale_ema_alpha) * p

    def _voxel_points(self):
        if not self.vox:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        pts = np.empty((len(self.vox), 2), dtype=np.float32)
        vals = np.empty((len(self.vox),), dtype=np.float32)
        for i, (sx, sy, sv, c) in enumerate(self.vox.values()):
            pts[i, 0] = sx / c
            pts[i, 1] = sy / c
            vals[i]   = sv / c
        return pts, vals

    def _voxel_values(self):
        if not self.vox:
            return np.zeros((0,), dtype=np.float32)
        out = np.empty((len(self.vox),), dtype=np.float32)
        for i, (sx, sy, sv, c) in enumerate(self.vox.values()):
            out[i] = sv / c
        return out

    # ======================================================
    # TF helper
    # ======================================================

    def _get_robot_xy(self):
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.frame_id, self.base_frame, rospy.Time(0), rospy.Duration(0.1)
            )
            return float(tfm.transform.translation.x), float(tfm.transform.translation.y)
        except Exception:
            return None

    # ======================================================
    # Grid helpers
    # ======================================================

    def _xy_to_rc(self, x, y):
        info = self.map_info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        r = int((y - oy) / res)
        c = int((x - ox) / res)
        return r, c

    def _rc_to_xy(self, rc):
        info = self.map_info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        return np.column_stack([
            (rc[:, 1].astype(np.float32) + 0.5) * res + ox,
            (rc[:, 0].astype(np.float32) + 0.5) * res + oy
        ]).astype(np.float32)

    # ======================================================
    # Reachable mask (BFS flood-fill in free space within radius)
    # ======================================================

    def _reachable_within_radius(self, rr0, cc0, rad_cells):
        h, w = self.map_info.height, self.map_info.width

        if rr0 < 0 or rr0 >= h or cc0 < 0 or cc0 >= w:
            return None
        if self.free_mask[rr0, cc0] < 0.5:
            return None

        # Bounding box to keep BFS local
        r0 = max(0, rr0 - rad_cells)
        r1 = min(h - 1, rr0 + rad_cells)
        c0 = max(0, cc0 - rad_cells)
        c1 = min(w - 1, cc0 + rad_cells)

        reachable = np.zeros((h, w), dtype=np.uint8)

        q = deque()
        q.append((rr0, cc0))
        reachable[rr0, cc0] = 1

        rad2 = rad_cells * rad_cells

        while q:
            r, c = q.popleft()
            dr = r - rr0
            dc = c - cc0
            if dr * dr + dc * dc > rad2:
                continue

            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nr < r0 or nr > r1 or nc < c0 or nc > c1:
                    continue
                if reachable[nr, nc]:
                    continue
                if self.free_mask[nr, nc] < 0.5:
                    continue
                reachable[nr, nc] = 1
                q.append((nr, nc))

        return reachable

    # ======================================================
    # Local query cells (reachable + within radius)
    # ======================================================

    def _local_query_rc(self, cx, cy):
        info = self.map_info
        res = info.resolution
        rr, cc = self._xy_to_rc(cx, cy)

        rad_cells = int(max(0.5, self.local_radius_m) / res)
        cf = max(1, int(self.coarse_factor))

        reachable = None
        if self.use_reachable_mask:
            reachable = self._reachable_within_radius(rr, cc, rad_cells)
            if reachable is None:
                return np.zeros((0, 2), dtype=np.int32), None

        # build candidate rc in disc (downsampled), then optionally gate by reachable
        r0 = max(0, rr - rad_cells)
        r1 = min(info.height - 1, rr + rad_cells)
        c0 = max(0, cc - rad_cells)
        c1 = min(info.width - 1, cc + rad_cells)

        rs = np.arange(r0, r1 + 1, cf, dtype=np.int32)
        cs = np.arange(c0, c1 + 1, cf, dtype=np.int32)
        if rs.size == 0 or cs.size == 0:
            return np.zeros((0, 2), dtype=np.int32), reachable

        grid_r, grid_c = np.meshgrid(rs, cs, indexing="ij")
        dr = (grid_r - rr).astype(np.float32)
        dc = (grid_c - cc).astype(np.float32)
        disk = (dr * dr + dc * dc) <= float(rad_cells * rad_cells)
        rc = np.column_stack([grid_r[disk].ravel(), grid_c[disk].ravel()])

        # free-only
        rc = rc[self.free_mask[rc[:, 0], rc[:, 1]] > 0.5]

        # reachable-only (prevents “behind walls”)
        if reachable is not None and rc.size:
            rc = rc[reachable[rc[:, 0], rc[:, 1]] > 0]

        if rc.shape[0] > self.max_query_points:
            rc = rc[np.random.choice(rc.shape[0], self.max_query_points, replace=False)]

        return rc, reachable

    # ======================================================
    # IDW floor (local)
    # ======================================================

    def _idw_patch(self, xy_query, xy_samples, v_samples):
        if xy_samples.shape[0] == 0:
            return np.zeros((xy_query.shape[0],), dtype=np.float32)

        if KDTree is not None and xy_samples.shape[0] > 25:
            tree = KDTree(xy_samples)
            k = min(25, xy_samples.shape[0])
            d, idx = tree.query(xy_query, k=k)
            if k == 1:
                d = d[:, None]
                idx = idx[:, None]
            vv = v_samples[idx]
        else:
            # cap sample count
            if xy_samples.shape[0] > 100:
                sel = np.random.choice(xy_samples.shape[0], 100, replace=False)
                xy_s = xy_samples[sel]
                v_s  = v_samples[sel]
            else:
                xy_s = xy_samples
                v_s  = v_samples

            dx = xy_query[:, None, 0] - xy_s[None, :, 0]
            dy = xy_query[:, None, 1] - xy_s[None, :, 1]
            d = np.sqrt(dx * dx + dy * dy) + 1e-6
            vv = v_s[None, :]

        w = 1.0 / (d * d + 0.20)
        num = np.sum(w * vv, axis=1)
        den = np.sum(w, axis=1) + 1e-6
        return (num / den).astype(np.float32)

    # ======================================================
    # Contrast mapping (makes highs saturate -> "hot/red")
    # ======================================================

    def _contrast(self, x01):
        # x in [0..inf), output in [0..1)
        k = max(self.contrast_k, 1e-6)
        return np.clip(x01 / (x01 + k), 0.0, 1.0).astype(np.float32)

    # ======================================================
    # Timer loop
    # ======================================================

    def _timer(self, _):
        if self.map_info is None or self.free_mask is None or self.global_heat is None:
            return

        # Always republish cached global maps for smooth RViz
        self._publish_grids()
        self._publish_markers()

        now = rospy.Time.now().to_sec()
        if (now - self.last_fit_t) < self.gp_refit_period_s:
            return

        pose = self._get_robot_xy()
        if pose is None:
            return
        cx, cy = pose

        xy_all, v_all = self._voxel_points()
        if xy_all.shape[0] < 6:
            return

        # Build local query cells gated by reachable region
        rc, reachable = self._local_query_rc(cx, cy)
        if rc.shape[0] == 0:
            return
        xy_q = self._rc_to_xy(rc)

        # Select training samples near robot (and optionally in same reachable component)
        train_rad = float(self.local_radius_m + self.local_margin_m)

        if KDTree is not None and xy_all.shape[0] >= 25:
            tree = KDTree(xy_all)
            idx = tree.query_ball_point([cx, cy], r=train_rad)
            if len(idx) < 6:
                d, nn = tree.query([cx, cy], k=min(80, xy_all.shape[0]))
                idx = np.atleast_1d(nn).tolist()
            X_train = xy_all[idx]
            y_train = v_all[idx]
        else:
            d2 = (xy_all[:, 0] - cx) ** 2 + (xy_all[:, 1] - cy) ** 2
            m = d2 <= (train_rad * train_rad)
            if np.sum(m) < 6:
                nn = np.argsort(d2)[:min(80, xy_all.shape[0])]
                X_train = xy_all[nn]
                y_train = v_all[nn]
            else:
                X_train = xy_all[m]
                y_train = v_all[m]

        # Optional: drop training samples that are not in reachable region (prevents cross-wall influence)
        if reachable is not None and X_train.shape[0] > 0:
            keep = []
            for i, (x, y) in enumerate(X_train):
                rr_s, cc_s = self._xy_to_rc(float(x), float(y))
                if 0 <= rr_s < reachable.shape[0] and 0 <= cc_s < reachable.shape[1] and reachable[rr_s, cc_s] > 0:
                    keep.append(i)
            if len(keep) >= 6:
                X_train = X_train[keep]
                y_train = y_train[keep]

        if X_train.shape[0] < 6:
            return

        # Fit/predict locally
        try:
            self.gp.fit(X_train, y_train)
            mean, std = self.gp.predict(xy_q, return_std=True)
            mean = np.clip(mean.astype(np.float32), 0.0, None)
            std  = np.clip(std.astype(np.float32),  0.0, None)
        except Exception:
            return

        self.last_fit_t = now

        # IDW floor in local region
        idw = self._idw_patch(xy_q, X_train, y_train)

        # Fuse (conservative): never below IDW floor
        patch_raw = np.maximum(mean, self.idw_floor_gain * idw)

        # Stable scale -> then contrast -> then optional gamma
        scale = max(self.global_scale, 1e-3)
        patch = patch_raw / scale
        patch = self._contrast(patch)
        if abs(self.gamma - 1.0) > 1e-3:
            patch = np.power(patch, self.gamma).astype(np.float32)

        # Write patch into grid
        patch_grid = np.zeros_like(self.global_heat, dtype=np.float32)
        patch_sig  = np.zeros_like(self.global_sigma, dtype=np.float32)
        patch_grid[rc[:, 0], rc[:, 1]] = patch
        patch_sig[rc[:, 0], rc[:, 1]]  = np.clip(std / (scale + 1e-6), 0.0, 1.0)

        # Minimal smoothing only (keeps it clean + localised)
        if self.smooth_sigma_cells > 0.0:
            patch_grid = gaussian_filter(patch_grid, sigma=self.smooth_sigma_cells)
            if self.publish_sigma:
                patch_sig = gaussian_filter(patch_sig, sigma=max(0.8, 0.6 * self.smooth_sigma_cells))

        # Mask to free space (and reachable already applied for rc)
        patch_grid *= self.free_mask
        patch_sig  *= self.free_mask

        upd_mask = patch_grid > (self.min_visible * 0.25)
        if not np.any(upd_mask):
            return

        # MONOTONIC envelope update (no fading, no backwards leaks)
        self.global_heat[upd_mask] = np.maximum(self.global_heat[upd_mask], patch_grid[upd_mask])

        if self.publish_sigma:
            # uncertainty conservative max
            self.global_sigma[upd_mask] = np.maximum(self.global_sigma[upd_mask], patch_sig[upd_mask])

        # Small floor where radiation exists
        ever = self.global_heat > 0.0
        self.global_heat[ever] = np.maximum(self.global_heat[ever], self.min_visible * self.free_mask[ever])

        self._publish_grids()

    # ======================================================
    # Publish grids
    # ======================================================

    def _publish_grids(self):
        info = self.map_info
        stamp = rospy.Time.now()

        # OccupancyGrid wants int8-like values in [-1..100]; we use [0..100]
        heat = np.clip(self.global_heat * 100.0, 0.0, 100.0).astype(np.int8)

        cost_msg = OccupancyGrid()
        cost_msg.header.stamp = stamp
        cost_msg.header.frame_id = self.frame_id
        cost_msg.info = info
        cost_msg.data = heat.ravel().tolist()
        self.cost_pub.publish(cost_msg)

        if self.publish_sigma:
            sig = np.clip(self.global_sigma * 100.0, 0.0, 100.0).astype(np.int8)
            sig_msg = OccupancyGrid()
            sig_msg.header.stamp = stamp
            sig_msg.header.frame_id = self.frame_id
            sig_msg.info = info
            sig_msg.data = sig.ravel().tolist()
            self.sig_pub.publish(sig_msg)

    # ======================================================
    # Markers
    # ======================================================

    def _publish_markers(self):
        xy, vv = self._voxel_points()
        ma = MarkerArray()
        if vv.size == 0:
            self.marker_pub.publish(ma)
            return

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "radiation_samples_vox"
        m.id = 0
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.16

        if xy.shape[0] > 2000:
            sel = np.random.choice(xy.shape[0], 2000, replace=False)
            xy = xy[sel]
            vv = vv[sel]

        lo, hi = np.percentile(vv, [10, 90]) if vv.size > 5 else (float(np.min(vv)), float(np.max(vv) + 1e-6))

        for (x, y), v in zip(xy, vv):
            t = np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            m.points.append(Point(x=float(x), y=float(y), z=0.05))
            m.colors.append(ColorRGBA(r=float(t), g=float(1.0 - t), b=0.0, a=0.9))

        ma.markers.append(m)
        self.marker_pub.publish(ma)


if __name__ == "__main__":
    try:
        RadiationGPMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass