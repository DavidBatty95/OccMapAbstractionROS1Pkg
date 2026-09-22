from pathlib import Path
import numpy as np
import pytest
from scipy.ndimage import binary_dilation
from PIL import Image
import yaml
from occmap.grid import ResearchMap, clearance, geodesic_distances, inflate_manhattan, load_map, segment_free


def test_coordinates_use_floor_and_cell_centres():
    grid = ResearchMap(np.zeros((3, 4)), np.zeros((3, 4)), 0.1, (-2, -1, 0), 'test')
    assert grid.world_to_grid(-2.001, -1.001) == (-1, -1)
    for y in range(3):
        for x in range(4):
            assert grid.world_to_grid(*grid.grid_to_world(x, y)) == (x, y)


def test_clearance_exact_and_empty_maps():
    blocked = np.zeros((5, 5), bool)
    assert np.isinf(clearance(blocked)).all()
    blocked[0, 0] = True
    assert clearance(blocked)[3, 4] == 5
    assert clearance(np.ones((5, 5), bool)).max() == 0


def test_geodesic_obstacle_detour_and_cap():
    free = np.ones((7, 7), bool)
    free[:5, 3] = False
    distance = geodesic_distances(free, (0, 0), 30, 0.1)
    assert distance[0, 6] == pytest.approx(1.6)
    assert np.isinf(distance[0, 3])
    capped = geodesic_distances(free, (0, 0), 15, 0.1)
    assert np.isinf(capped[0, 6])


def test_geodesic_decimal_resolution_on_large_open_grid():
    free = np.ones((101, 101), bool)
    actual = geodesic_distances(free, (50, 50), 200, 0.02)
    rows, cols = np.indices(free.shape)
    np.testing.assert_allclose(actual, (abs(rows-50) + abs(cols-50)) * 0.02, rtol=1e-6)


@pytest.mark.parametrize('radius', [0, 1, 2, 5])
def test_inflation_matches_independent_morphology(radius):
    grid = np.random.default_rng(42).random((31, 47)) < 0.1
    expected = binary_dilation(grid, iterations=radius) if radius else grid
    np.testing.assert_array_equal(inflate_manhattan(grid, radius), expected)
    assert not inflate_manhattan(np.zeros((3, 3), bool), radius).any()


def test_segment_rejects_corner_cutting_and_outside_bounds():
    blocked = np.array([[False, True], [True, False]])
    assert not segment_free(blocked, (0, 0), (1, 1))
    assert not segment_free(blocked, (-1, 0), (0, 0))
    assert segment_free(np.zeros((5, 5), bool), (0, 1), (4, 3))


@pytest.mark.parametrize('factor', [1, 5])
def test_all_bundled_maps_have_valid_fields(factor):
    root = Path(__file__).resolve().parents[1] / 'src/exploration_pkg/maps'
    for directory in sorted(root.iterdir()):
        grid = load_map(directory, factor)
        assert grid.occupancy.shape == grid.radiation.shape
        assert np.any(grid.occupancy == 0)
        assert set(np.unique(grid.occupancy)) <= {-1, 0, 100}
        assert grid.radiation.max() > 0


def write_fixture(tmp_path, pixels, field, **settings):
    config = dict(image='map.pgm', resolution=0.1, origin=[-1, -1, 0],
                  negate=0, occupied_thresh=0.65, free_thresh=0.2)
    config.update(settings)
    (tmp_path / (tmp_path.name + '.yaml')).write_text(yaml.safe_dump(config))
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(tmp_path / 'map.pgm')
    np.save(tmp_path / 'radiation_field.npy', np.asarray(field, dtype=np.float32))


def test_map_loading_preserves_ros_orientation_and_conservative_pooling(tmp_path):
    pixels = np.full((4, 4), 255, dtype=np.uint8)
    pixels[0, 0] = 0  # Upper-left image pixel becomes the last ROS row.
    pixels[3, 3] = 200  # Occupancy 0.216: between the configured thresholds.
    write_fixture(tmp_path, pixels, np.arange(16).reshape(4, 4))
    original = load_map(tmp_path)
    assert original.occupancy[3, 0] == 100
    assert original.occupancy[0, 3] == -1
    pooled = load_map(tmp_path, 2)
    np.testing.assert_array_equal(pooled.occupancy, [[0, -1], [100, 0]])
    np.testing.assert_array_equal(pooled.radiation, [[5, 7], [13, 15]])
    assert pooled.resolution == 0.2


@pytest.mark.parametrize('problem', ['shape', 'nan', 'negative', 'rotated'])
def test_map_loading_rejects_invalid_data(tmp_path, problem):
    field = np.ones((2, 2))
    if problem == 'shape':
        field = np.ones((3, 2))
    elif problem == 'nan':
        field[0, 0] = np.nan
    elif problem == 'negative':
        field[0, 0] = -1
    origin = [0, 0, 0.5 if problem == 'rotated' else 0]
    write_fixture(tmp_path, np.full((2, 2), 255), field, origin=origin)
    with pytest.raises(ValueError):
        load_map(tmp_path)
