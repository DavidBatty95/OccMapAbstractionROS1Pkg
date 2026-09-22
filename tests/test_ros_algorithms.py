"""Numerical regression tests importing real ROS nodes; no master required."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def node(name):
    path = ROOT / 'src/exploration_pkg/scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simulator_astar_cannot_cut_corners():
    sim = node('explore_sim_node')
    blocked = np.array([[False, True], [True, False]])
    assert sim.astar(blocked, (0, 0), (1, 1)) is None
    path = sim.astar(np.zeros((10, 10), bool), (0, 0), (9, 9))
    assert path[0] == (0, 0) and path[-1] == (9, 9)


def test_gp_landmarks_cover_detours_without_huge_sentinel_features():
    gp = node('rad_GPR_node')
    mapper = gp.RadiationGPMapper.__new__(gp.RadiationGPMapper)
    mapper.map_info = SimpleNamespace(width=25, height=25, resolution=0.1)
    mapper.geo_feature_scale_m = 3.0
    mapper.landmark_pick_stride_cells = 1
    mapper.num_landmarks = 5
    free = np.ones((25, 25), np.uint8)
    free[:20, 12] = 0
    landmarks = mapper._pick_landmarks(free, 2, 2, 25)
    assert len(landmarks) == 5
    distmaps = [mapper._geodesic_distmap(r, c, free, 25) for r, c in landmarks]
    rc = np.argwhere(free)[::10]
    xy = rc[:, ::-1].astype(np.float32) * 0.1
    features = mapper._embed_features(rc, xy, distmaps)
    assert np.isfinite(features).all() and np.max(abs(features)) < 20
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = gp.CorridorSVGP(torch.from_numpy(features[:16]))
    likelihood = gp.gpytorch.likelihoods.GaussianLikelihood()
    objective = gp.gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(features))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.03)
    inputs = torch.from_numpy(features)
    targets = torch.sin(inputs[:, 0])
    for _ in range(8):
        optimizer.zero_grad()
        loss = -objective(model(inputs), targets)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(inputs)
        assert torch.isfinite(prediction.mean).all()
        assert torch.isfinite(prediction.variance).all()


def test_gp_reachable_mask_is_bounded_and_cannot_cross_walls():
    gp = node('rad_GPR_node')
    mapper = gp.RadiationGPMapper.__new__(gp.RadiationGPMapper)
    mapper.map_info = SimpleNamespace(height=15, width=15)
    mapper.free_mask = np.ones((15, 15))
    mapper.free_mask[:, 8] = 0
    reachable = mapper._reachable_within_radius(7, 5, 5)
    rows, cols = np.nonzero(reachable)
    assert np.all((rows-7)**2 + (cols-5)**2 <= 25)
    assert not reachable[:, 8:].any()


def test_gp_sparse_local_samples_keep_cached_estimate_without_crashing(monkeypatch):
    gp = node('rad_GPR_node')
    mapper = gp.RadiationGPMapper.__new__(gp.RadiationGPMapper)
    mapper.map_info = SimpleNamespace(height=5, width=5, resolution=1.0)
    mapper.free_mask = np.ones((5, 5))
    mapper.global_heat = np.full((5, 5), 0.25)
    mapper.last_fit_t = 0
    mapper.gp_refit_period_s = 1
    mapper.local_radius_m = 1
    mapper.local_margin_m = 1
    mapper.use_reachable_mask = False
    mapper.num_landmarks = 0
    mapper._publish_grids = mapper._publish_markers = lambda: None
    mapper._get_robot_xy = lambda: (0, 0)
    mapper._local_query_rc = lambda *_: (np.array([[0, 0]]), None, 0, 0, 1)
    mapper._rc_to_xy = lambda *_: np.array([[0, 0]])
    mapper._xy_to_rc = lambda x, y: (int(y), int(x))
    # Enough global samples to enter the update, but only one in the local crop.
    mapper._voxel_points = lambda: (np.array([[0, 0]] + [[4, 4]] * 7), np.ones(8))
    monkeypatch.setattr(gp.rospy.Time, 'now', lambda: gp.rospy.Time.from_sec(10))
    gp.RadiationGPMapper._timer.__wrapped__(mapper, None)
    np.testing.assert_array_equal(mapper.global_heat, np.full((5, 5), 0.25))
