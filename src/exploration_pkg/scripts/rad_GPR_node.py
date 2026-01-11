#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rad_GPR_node.py (v7.1-gpytorch-corridor-aware-svgp) — FIXED DROP-IN

Fixes:
- std_msgs/Float32 has .data only (no .value, no .header) -> was preventing sample collection.
- TF lookup now uses latest available transform (Time(0)) for Float32 sensor.
- Publishing OccupancyGrid uses int8 as expected.
- Extra guards + throttled logs so failure modes are visible.

Keeps all original functionality:
- voxel aggregation
- reachable local updates (no predicting through walls)
- corridor-aware geodesic landmark embedding
- SVGP GPyTorch (CUDA if available)
- IDW conservative floor
- monotonic global envelope (no fading)
- publishes: /radiation_costmap, /radiation_sigma, /radiation_markers
"""

import rospy
import numpy as np
from collections import deque

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Float32
from geometry_msgs.msg import Point

import tf2_ros

# Optional speed helpers (CPU)
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

# GPU GP
import torch
import gpytorch


# ======================================================
# GPyTorch SVGP model
# ======================================================

class CorridorSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, nu=1.5):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=nu)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ======================================================
# Node
# ======================================================

class RadiationGPMapper:
    def __init__(self):
        rospy.init_node("rad_GPR_node", anonymous=False)

        # ---------------- Topics / frames ----------------
        self.map_topic  = rospy.get_param("~map_topic", "/map")
        self.meas_topic = rospy.get_param("~meas_topic", "/sim_rad_sensor")
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
        self.local_radius_m     = float(rospy.get_param("~local_radius_m", 6.0))
        self.local_margin_m     = float(rospy.get_param("~local_margin_m", 2.0))
        self.coarse_factor      = int(rospy.get_param("~coarse_factor", 2))
        self.max_query_points   = int(rospy.get_param("~max_query_points", 8000))
        self.use_reachable_mask = bool(rospy.get_param("~use_reachable_mask", True))

        # ---------------- Sample compression ----------------
        self.voxel_size_m = float(rospy.get_param("~voxel_size_m", 0.30))
        self.max_voxels   = int(rospy.get_param("~max_voxels", 20000))  # <=0 disables cap

        # ---------------- Corridor-aware embedding ----------------
        self.num_landmarks = int(rospy.get_param("~num_landmarks", 5))  # 4–8 typical
        self.landmark_pick_stride_cells = int(rospy.get_param("~landmark_pick_stride_cells", 2))
        self.geo_feature_scale_m = float(rospy.get_param("~geo_feature_scale_m", 3.0))

        # ---------------- Field shaping / realism ----------------
        self.scale_percentile = float(rospy.get_param("~scale_percentile", 85.0))
        self.scale_ema_alpha  = float(rospy.get_param("~scale_ema_alpha", 0.965))
        self.global_scale = 1.0

        self.abs_scale_uSv = float(rospy.get_param("~abs_scale_uSv", -1.0))  # <=0 disables

        self.contrast_k  = float(rospy.get_param("~contrast_k", 0.18))
        self.gamma       = float(rospy.get_param("~heatmap_gamma", 1.0))
        self.min_visible = float(rospy.get_param("~min_visible", 0.008))

        self.smooth_sigma_cells = float(rospy.get_param("~smooth_sigma_cells", 0.0))  # OFF by default

        # Conservative floor fusion
        self.idw_floor_gain = float(rospy.get_param("~idw_floor_gain", 0.95))
        self.idw_k   = int(rospy.get_param("~idw_k", 25))
        self.idw_eps = float(rospy.get_param("~idw_eps", 0.20))

        # Sigma publish
        self.publish_sigma = bool(rospy.get_param("~publish_sigma", True))

        # ---------------- GPytorch controls ----------------
        self.inducing_points = int(rospy.get_param("~inducing_points", 128))  # 64–256 typical
        self.train_steps = int(rospy.get_param("~train_steps", 25))           # per refit
        self.train_lr    = float(rospy.get_param("~train_lr", 0.03))

        self.nu = float(rospy.get_param("~matern_nu", 1.5))
        self.init_lengthscale = float(rospy.get_param("~length_scale", 1.2))
        self.init_noise       = float(rospy.get_param("~noise_level", 0.25))
        self.init_outputscale = float(rospy.get_param("~signal_scale", 1.0))

        self.lengthscale_min = float(rospy.get_param("~lengthscale_min", 0.15))
        self.lengthscale_max = float(rospy.get_param("~lengthscale_max", 10.0))
        self.noise_min       = float(rospy.get_param("~noise_min", 1e-4))
        self.noise_max       = float(rospy.get_param("~noise_max", 5.0))

        # Device selection
        self.force_cpu = bool(rospy.get_param("~force_cpu", False))
        self.device = torch.device("cpu")
        if (not self.force_cpu) and torch.cuda.is_available():
            self.device = torch.device("cuda:0")

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

        rospy.Subscriber(self.map_topic,  OccupancyGrid, self.map_cb,  queue_size=1)
        rospy.Subscriber(self.meas_topic, Float32,       self.meas_cb, queue_size=300)

        rospy.Timer(rospy.Duration(self.publish_period_s), self._timer)

        # ---------------- GP objects (lazy init) ----------------
        self.model = None
        self.likelihood = None
        self.optimizer = None
        self.mll = None
        self._feat_dim = None

        rospy.loginfo(
            "rad_GPR_node v7.1 (GPyTorch SVGP corridor-aware) running on device=%s (cuda=%s) meas_topic=%s",
            str(self.device), str(torch.cuda.is_available()), self.meas_topic
        )

    # ======================================================
    # Map
    # ======================================================

    def map_cb(self, msg):
        self.map_info = msg.info
        raw = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))

        # free: known and below free_max; occupied: >= occ_min; unknown: < 0
        free = (raw >= 0) & (raw <= self.free_max) & (raw < self.occ_min)
        self.free_mask = free.astype(np.float32)

        h, w = msg.info.height, msg.info.width
        if self.global_heat is None or self.global_heat.shape != (h, w):
            self.global_heat  = np.zeros((h, w), dtype=np.float32)
            self.global_sigma = np.zeros((h, w), dtype=np.float32)

        rospy.loginfo_throttle(2.0, "[rad_GPR] map received: %dx%d res=%.3f",
                               w, h, float(msg.info.resolution))

    # ======================================================
    # Measurements -> voxel aggregate
    # ======================================================

    def meas_cb(self, msg: Float32):
        """
        /sim_rad_sensor is std_msgs/Float32:
          - value is msg.data
          - no header timestamp -> use TF latest (Time(0))
        """
        if self.map_info is None or self.free_mask is None:
            return

        # value
        try:
            v = max(0.0, float(msg.data))
        except Exception:
            return

        # TF lookup: map <- base_link (latest)
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.frame_id,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(0.1)
            )
            x = float(tfm.transform.translation.x)
            y = float(tfm.transform.translation.y)
        except Exception:
            rospy.logwarn_throttle(2.0, "[rad_GPR] TF lookup failed for radiation measurement (map<-base_link).")
            return

        # reject samples not in free space (prevents garbage points on walls/unknown)
        rr, cc = self._xy_to_rc(x, y)
        if rr < 0 or rr >= self.map_info.height or cc < 0 or cc >= self.map_info.width:
            return
        if self.free_mask[rr, cc] < 0.5:
            return

        # voxel aggregation
        vs = float(self.voxel_size_m)
        ix = int(np.floor(x / vs))
        iy = int(np.floor(y / vs))
        key = (ix, iy)

        if key in self.vox:
            sx, sy, sv, c = self.vox[key]
            self.vox[key] = [sx + x, sy + y, sv + v, c + 1]
        else:
            self.vox[key] = [x, y, v, 1]

        # cap memory
        if self.max_voxels > 0 and len(self.vox) > self.max_voxels:
            self.vox.pop(next(iter(self.vox)))

        # scale tracking
        vals = self._voxel_values()
        if self.abs_scale_uSv > 0.0:
            self.global_scale = float(self.abs_scale_uSv)
        elif vals.size >= 10:
            p = float(np.percentile(vals, self.scale_percentile))
            p = max(p, 1e-3)
            self.global_scale = (
                self.scale_ema_alpha * self.global_scale
                + (1.0 - self.scale_ema_alpha) * p
            )

        rospy.loginfo_throttle(
            2.0,
            "[rad_GPR] meas @ (%.2f, %.2f) = %.3f | voxels=%d scale=%.3f",
            x, y, v, len(self.vox), float(self.global_scale)
        )

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
    # Reachable mask within radius (BFS in free space)
    # ======================================================

    def _reachable_within_radius(self, rr0, cc0, rad_cells):
        h, w = self.map_info.height, self.map_info.width

        if rr0 < 0 or rr0 >= h or cc0 < 0 or cc0 >= w:
            return None
        if self.free_mask[rr0, cc0] < 0.5:
            return None

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
    # Geodesic distance map (BFS with unit edge cost)
    # ======================================================

    def _geodesic_distmap(self, seed_r, seed_c, reachable, max_cells):
        info = self.map_info
        res = info.resolution
        h, w = info.height, info.width

        dist = np.full((h, w), np.inf, dtype=np.float32)
        if reachable[seed_r, seed_c] == 0:
            return dist

        q = deque()
        q.append((seed_r, seed_c))
        dist[seed_r, seed_c] = 0.0

        cap_m = max_cells * res
        while q:
            r, c = q.popleft()
            d0 = dist[r, c]
            if d0 > cap_m:
                continue

            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if reachable[nr, nc] == 0:
                    continue
                nd = d0 + res
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    q.append((nr, nc))

        return dist

    # ======================================================
    # Pick geodesic landmarks (farthest-point sampling)
    # ======================================================

    def _pick_landmarks(self, reachable, rr0, cc0, max_cells):
        stride = max(1, int(self.landmark_pick_stride_cells))
        rc_all = np.argwhere(reachable > 0)
        if rc_all.shape[0] == 0:
            return []

        if stride > 1:
            rc_all = rc_all[(rc_all[:, 0] % stride == 0) & (rc_all[:, 1] % stride == 0)]
            if rc_all.shape[0] == 0:
                rc_all = np.argwhere(reachable > 0)

        landmarks = [(rr0, cc0)]
        last_dist = self._geodesic_distmap(rr0, cc0, reachable, max_cells)

        for _ in range(max(0, self.num_landmarks - 1)):
            dvals = last_dist[rc_all[:, 0], rc_all[:, 1]]
            ok = np.isfinite(dvals)
            if not np.any(ok):
                break
            rc_ok = rc_all[ok]
            d_ok = dvals[ok]
            far_idx = int(np.argmax(d_ok))
            seed = (int(rc_ok[far_idx, 0]), int(rc_ok[far_idx, 1]))
            if seed in landmarks:
                break
            landmarks.append(seed)
            last_dist = self._geodesic_distmap(seed[0], seed[1], reachable, max_cells)

        return landmarks

    # ======================================================
    # Build local query rc set
    # ======================================================

    def _local_query_rc(self, cx, cy):
        info = self.map_info
        res = info.resolution
        rr, cc = self._xy_to_rc(cx, cy)

        rad_cells = int(max(0.5, self.local_radius_m) / res)
        reachable = None
        if self.use_reachable_mask:
            reachable = self._reachable_within_radius(rr, cc, rad_cells)
            if reachable is None:
                return np.zeros((0, 2), dtype=np.int32), None, rr, cc, rad_cells

        cf = max(1, int(self.coarse_factor))
        r0 = max(0, rr - rad_cells)
        r1 = min(info.height - 1, rr + rad_cells)
        c0 = max(0, cc - rad_cells)
        c1 = min(info.width - 1, cc + rad_cells)

        rs = np.arange(r0, r1 + 1, cf, dtype=np.int32)
        cs = np.arange(c0, c1 + 1, cf, dtype=np.int32)
        if rs.size == 0 or cs.size == 0:
            return np.zeros((0, 2), dtype=np.int32), reachable, rr, cc, rad_cells

        grid_r, grid_c = np.meshgrid(rs, cs, indexing="ij")
        dr = (grid_r - rr).astype(np.float32)
        dc = (grid_c - cc).astype(np.float32)
        disk = (dr * dr + dc * dc) <= float(rad_cells * rad_cells)
        rc = np.column_stack([grid_r[disk].ravel(), grid_c[disk].ravel()])

        rc = rc[self.free_mask[rc[:, 0], rc[:, 1]] > 0.5]
        if reachable is not None and rc.size:
            rc = rc[reachable[rc[:, 0], rc[:, 1]] > 0]

        if rc.shape[0] > self.max_query_points:
            rc = rc[np.random.choice(rc.shape[0], self.max_query_points, replace=False)]

        return rc, reachable, rr, cc, rad_cells

    # ======================================================
    # IDW floor
    # ======================================================

    def _idw_patch(self, xy_query, xy_samples, v_samples):
        if xy_samples.shape[0] == 0:
            return np.zeros((xy_query.shape[0],), dtype=np.float32)

        k = min(int(self.idw_k), xy_samples.shape[0])
        if KDTree is not None and xy_samples.shape[0] >= 25:
            tree = KDTree(xy_samples)
            d, idx = tree.query(xy_query, k=k)
            if k == 1:
                d = d[:, None]
                idx = idx[:, None]
            vv = v_samples[idx]
        else:
            if xy_samples.shape[0] > 150:
                sel = np.random.choice(xy_samples.shape[0], 150, replace=False)
                xy_s = xy_samples[sel]
                v_s  = v_samples[sel]
            else:
                xy_s = xy_samples
                v_s  = v_samples

            dx = xy_query[:, None, 0] - xy_s[None, :, 0]
            dy = xy_query[:, None, 1] - xy_s[None, :, 1]
            d = np.sqrt(dx * dx + dy * dy) + 1e-6

            if d.shape[1] > k:
                nn = np.argpartition(d, kth=k-1, axis=1)[:, :k]
                d = np.take_along_axis(d, nn, axis=1)
                vv = v_s[nn]
            else:
                vv = v_s[None, :]

        w = 1.0 / (d * d + float(self.idw_eps))
        num = np.sum(w * vv, axis=1)
        den = np.sum(w, axis=1) + 1e-6
        return (num / den).astype(np.float32)

    # ======================================================
    # Contrast mapping
    # ======================================================

    def _contrast(self, x01):
        k = max(float(self.contrast_k), 1e-6)
        x01 = np.clip(x01, 0.0, None).astype(np.float32)
        y = x01 / (x01 + k)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    # ======================================================
    # Corridor-aware feature embedding
    # ======================================================

    def _embed_features(self, rc, xy, distmaps):
        if not distmaps:
            return xy.astype(np.float32)

        feats = [xy.astype(np.float32)]
        scale = max(self.geo_feature_scale_m, 1e-3)
        for dm in distmaps:
            d = dm[rc[:, 0], rc[:, 1]].astype(np.float32)
            d = np.where(np.isfinite(d), d, 1e6).astype(np.float32)
            feats.append((d[:, None] / scale).astype(np.float32))
        return np.hstack(feats).astype(np.float32)

    # ======================================================
    # GP init / refresh
    # ======================================================

    def _ensure_model(self, feat_dim: int, inducing_X: np.ndarray):
        if self.model is not None and self._feat_dim == feat_dim:
            return

        inducing = torch.from_numpy(inducing_X).to(self.device).float()
        self.model = CorridorSVGP(inducing, nu=self.nu).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        with torch.no_grad():
            self.model.covar_module.base_kernel.lengthscale = torch.tensor(
                float(self.init_lengthscale), device=self.device
            )
            self.model.covar_module.outputscale = torch.tensor(
                float(self.init_outputscale), device=self.device
            )
            self.likelihood.noise = torch.tensor(float(self.init_noise), device=self.device)

        params = list(self.model.parameters()) + list(self.likelihood.parameters())
        self.optimizer = torch.optim.Adam(params, lr=float(self.train_lr))
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=1)

        self._feat_dim = feat_dim
        rospy.loginfo("[rad_GPR] SVGP model created feat_dim=%d inducing=%d device=%s",
                      feat_dim, inducing_X.shape[0], str(self.device))

    def _clamp_hypers(self):
        try:
            with torch.no_grad():
                ls = self.model.covar_module.base_kernel.lengthscale
                ls.clamp_(min=float(self.lengthscale_min), max=float(self.lengthscale_max))
                nz = self.likelihood.noise
                nz.clamp_(min=float(self.noise_min), max=float(self.noise_max))
        except Exception:
            pass

    # ======================================================
    # Timer loop
    # ======================================================

    def _timer(self, _):
        if self.map_info is None or self.free_mask is None or self.global_heat is None:
            return

        # republish cached for RViz smoothness
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
        if xy_all.shape[0] < 8:
            rospy.loginfo_throttle(2.0, "[rad_GPR] waiting for samples: voxels=%d", xy_all.shape[0])
            return

        # Local query region
        rc_q, reachable, rr0, cc0, rad_cells = self._local_query_rc(cx, cy)
        if rc_q.shape[0] == 0:
            return
        xy_q = self._rc_to_xy(rc_q)

        # Training reachable region (slightly larger)
        res = self.map_info.resolution
        train_cells = int(max(0.5, self.local_radius_m + self.local_margin_m) / res)
        if reachable is None and self.use_reachable_mask:
            reachable = self._reachable_within_radius(rr0, cc0, train_cells)
            if reachable is None:
                return

        # Landmarks & distmaps
        distmaps = []
        if reachable is not None and self.num_landmarks > 0:
            landmarks = self._pick_landmarks(reachable, rr0, cc0, train_cells)
            for (lr, lc) in landmarks:
                distmaps.append(self._geodesic_distmap(lr, lc, reachable, train_cells))

        # Select training samples in train region + reachable
        X_train_xy = []
        y_train = []

        r2 = (self.local_radius_m + self.local_margin_m) ** 2
        for (x, y), v in zip(xy_all, v_all):
            rr_s, cc_s = self._xy_to_rc(float(x), float(y))
            if rr_s < 0 or rr_s >= self.map_info.height or cc_s < 0 or cc_s >= self.map_info.width:
                continue
            if self.free_mask[rr_s, cc_s] < 0.5:
                continue
            if reachable is not None and reachable[rr_s, cc_s] == 0:
                continue
            if (float(x) - cx) ** 2 + (float(y) - cy) ** 2 > r2:
                continue
            X_train_xy.append([float(x), float(y)])
            y_train.append(float(v))

        if len(y_train) < 8:
            # Keep radiation alive using IDW-only field
            idw = self._idw_patch(xy_q, X_train_xy, y_train)
            scale = max(float(self.global_scale), 1e-3)
            patch = self._contrast(idw / scale)
            mean = patch
            std = np.zeros_like(patch)

        X_train_xy = np.asarray(X_train_xy, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)

        # Embed in corridor-aware feature space
        rc_train = np.array([self._xy_to_rc(float(x), float(y)) for x, y in X_train_xy], dtype=np.int32)
        X_train = self._embed_features(rc_train, X_train_xy, distmaps)
        X_query = self._embed_features(rc_q, xy_q, distmaps)

        # Inducing points selection
        M = min(int(self.inducing_points), X_train.shape[0])
        if M < 8:
            return
        sel = np.random.choice(X_train.shape[0], M, replace=False)
        Z = X_train[sel].astype(np.float32)

        # Ensure / refresh model if needed
        self._ensure_model(X_train.shape[1], Z)

        # Standardise targets
        y_mu = float(np.mean(y_train))
        y_sd = float(np.std(y_train) + 1e-6)
        y_std = ((y_train - y_mu) / y_sd).astype(np.float32)

        # Torch tensors
        Xtr = torch.from_numpy(X_train).to(self.device).float()
        ytr = torch.from_numpy(y_std).to(self.device).float()
        Xq  = torch.from_numpy(X_query).to(self.device).float()

        # Update inducing points (online adaptation)
        try:
            with torch.no_grad():
                Zt = torch.from_numpy(Z).to(self.device).float()
                self.model.variational_strategy.inducing_points.data.copy_(Zt)
        except Exception:
            pass

        # Train
        self.model.train()
        self.likelihood.train()
        self.mll.num_data = Xtr.size(0)

        for _ in range(max(1, int(self.train_steps))):
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(Xtr)
            loss = -self.mll(out, ytr)
            loss.backward()
            self.optimizer.step()
            self._clamp_hypers()

        # Predict
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(Xq))
            mean_std = pred.mean
            std_std  = pred.variance.clamp_min(1e-12).sqrt()

        mean = (mean_std.detach().cpu().numpy().astype(np.float32) * y_sd + y_mu)
        std  = (std_std.detach().cpu().numpy().astype(np.float32) * y_sd)

        mean = np.clip(mean, 0.0, None).astype(np.float32)
        std  = np.clip(std,  0.0, None).astype(np.float32)

        self.last_fit_t = now

        # IDW floor
        idw = self._idw_patch(xy_q, X_train_xy, y_train)
        patch_raw = np.maximum(mean, self.idw_floor_gain * idw)

        # Map physical -> heat [0..1]
        scale = max(float(self.global_scale), 1e-3)
        patch = patch_raw / scale
        patch = self._contrast(patch)
        if abs(self.gamma - 1.0) > 1e-3:
            patch = np.power(patch, self.gamma).astype(np.float32)

        # -------------------------------------------------------
        # SOLID FIELD CONSTRUCTION (fills + diffuses radiation)
        # -------------------------------------------------------

        patch_grid = np.zeros_like(self.global_heat, dtype=np.float32)
        patch_sig  = np.zeros_like(self.global_sigma, dtype=np.float32)

        # Seed GP predictions
        patch_grid[rc_q[:, 0], rc_q[:, 1]] = patch

        denom = max(np.percentile(std, 90), 1e-6)
        patch_sig[rc_q[:, 0], rc_q[:, 1]] = np.clip(std / denom, 0.0, 1.0)

        # Fill reachable holes using IDW
        if reachable is not None:
            hole_mask = (patch_grid == 0) & (reachable > 0)
        else:
            hole_mask = (patch_grid == 0) & (self.free_mask > 0)

        if np.any(hole_mask):
            rc_holes = np.argwhere(hole_mask)
            xy_holes = self._rc_to_xy(rc_holes)
            idw_fill = self._idw_patch(xy_holes, X_train_xy, y_train) / scale
            patch_grid[hole_mask] = idw_fill

        # Physical diffusion (radiation spreads in air)
        if gaussian_filter is not None:
            sigma_cells = max(1.0, self.local_radius_m / self.map_info.resolution / 12.0)
            patch_grid = gaussian_filter(patch_grid, sigma=sigma_cells)
            if self.publish_sigma:
                patch_sig = gaussian_filter(patch_sig, sigma=max(1.0, 0.7 * sigma_cells))

        # Respect walls
        patch_grid *= self.free_mask
        patch_sig  *= self.free_mask

        upd_mask = patch_grid > (self.min_visible * 0.25)
        if not np.any(upd_mask):
            return

        # Monotonic envelope update
        self.global_heat[upd_mask] = np.maximum(self.global_heat[upd_mask], patch_grid[upd_mask])
        if self.publish_sigma:
            self.global_sigma[upd_mask] = np.maximum(self.global_sigma[upd_mask], patch_sig[upd_mask])

        ever = self.global_heat > 0.0
        self.global_heat[ever] = np.maximum(self.global_heat[ever], self.min_visible * self.free_mask[ever])

        self._publish_grids()

        rospy.loginfo_throttle(
            2.0,
            "[rad_GPR] FIT ok: trainN=%d queryQ=%d M=%d device=%s ls=%.3f noise=%.3f scale=%.3f",
            X_train.shape[0], X_query.shape[0], M, str(self.device),
            float(self.model.covar_module.base_kernel.lengthscale.mean().detach().cpu().item()),
            float(self.likelihood.noise.detach().cpu().item()),
            float(self.global_scale),
        )

    # ======================================================
    # Publish grids
    # ======================================================

    def _publish_grids(self):
        if self.map_info is None:
            return

        info = self.map_info
        stamp = rospy.Time.now()

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

        if xy.shape[0] > 2500:
            sel = np.random.choice(xy.shape[0], 2500, replace=False)
            xy = xy[sel]
            vv = vv[sel]

        lo, hi = np.percentile(vv, [10, 90]) if vv.size > 8 else (float(np.min(vv)), float(np.max(vv) + 1e-6))

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