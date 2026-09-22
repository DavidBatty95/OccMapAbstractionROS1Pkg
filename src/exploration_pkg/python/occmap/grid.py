"""Map-server compatible loading and conservative grid geometry (no ROS imports)."""
from dataclasses import dataclass
from pathlib import Path
import math
from collections import deque

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, distance_transform_cdt
import yaml


def clearance(obstacles):
    """Exact Euclidean centre-to-centre clearance in cells; infinity if no obstacle."""
    obstacles = np.asarray(obstacles, dtype=bool)
    if obstacles.ndim != 2:
        raise ValueError('Expected a two-dimensional obstacle mask')
    if not obstacles.any():
        return np.full(obstacles.shape, np.inf, dtype=np.float32)
    return distance_transform_edt(~obstacles).astype(np.float32)


def inflate_manhattan(grid, radius):
    """Compiled equivalent of the original four-connected BFS inflation."""
    occupied = np.asarray(grid, dtype=bool)
    if radius <= 0 or not occupied.any():
        return grid.copy()
    distance = distance_transform_cdt(~occupied, metric='taxicab')
    return (distance <= radius).astype(grid.dtype)


def geodesic_distances(reachable, seed, max_steps, resolution):
    """Four-connected shortest distances, visiting each cell at most once.

    Integer hop counts avoid float32 rounding repeatedly relaxing equal paths.
    Unreachable/out-of-radius cells remain infinity.
    """
    h, w = reachable.shape
    r, c = seed
    hops = np.full((h, w), -1, dtype=np.int32)
    result = np.full((h, w), np.inf, dtype=np.float32)
    if not (0 <= r < h and 0 <= c < w) or not reachable[r, c]:
        return result
    queue = deque([(r, c)])
    hops[r, c] = 0
    while queue:
        r, c = queue.popleft()
        n = hops[r, c] + 1
        if n > max_steps:
            continue
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= nr < h and 0 <= nc < w and reachable[nr, nc] and hops[nr, nc] < 0:
                hops[nr, nc] = n
                queue.append((nr, nc))
    valid = hops >= 0
    result[valid] = hops[valid] * resolution
    return result


def line_cells(start, end):
    """Bresenham centre line, inclusive of both endpoints."""
    x, y = start
    x1, y1 = end
    dx, dy = abs(x1 - x), -abs(y1 - y)
    sx, sy = (1 if x < x1 else -1), (1 if y < y1 else -1)
    error = dx + dy
    while True:
        yield x, y
        if (x, y) == (x1, y1):
            return
        twice = 2 * error
        if twice >= dy:
            error += dy
            x += sx
        if twice <= dx:
            error += dx
            y += sy


def segment_free(blocked, start, end):
    """Check bounds, occupied cells and both neighbours at every diagonal step."""
    height, width = blocked.shape
    previous = None
    for x, y in line_cells(start, end):
        if not (0 <= x < width and 0 <= y < height) or blocked[y, x]:
            return False
        if previous is not None:
            px, py = previous
            if x != px and y != py and (blocked[py, x] or blocked[y, px]):
                return False
        previous = x, y
    return True


@dataclass
class ResearchMap:
    occupancy: np.ndarray
    radiation: np.ndarray
    resolution: float
    origin: tuple
    name: str

    def world_to_grid(self, x, y):
        return (math.floor((x - self.origin[0]) / self.resolution),
                math.floor((y - self.origin[1]) / self.resolution))

    def grid_to_world(self, x, y):
        return (self.origin[0] + (x + 0.5) * self.resolution,
                self.origin[1] + (y + 0.5) * self.resolution)


def load_map(directory, factor=1):
    """Load trinary PGM and relative-dose field, conservatively pooling blocks.

    Occupied wins over unknown; unknown wins over free. Radiation is max-pooled.
    Rotated maps are rejected rather than silently assigned wrong coordinates.
    """
    directory = Path(directory)
    if not isinstance(factor, int) or factor < 1:
        raise ValueError('factor must be a positive integer')
    with (directory / (directory.name + '.yaml')).open() as stream:
        config = yaml.safe_load(stream)
    resolution = float(config['resolution'])
    origin = tuple(float(v) for v in config['origin'])
    if resolution <= 0 or not math.isfinite(resolution) or len(origin) != 3:
        raise ValueError('Invalid map geometry')
    if not all(math.isfinite(v) for v in origin) or abs(origin[2]) > 1e-9:
        raise ValueError('Only finite, axis-aligned maps are supported')
    with Image.open(directory / config['image']) as image:
        grey = np.flipud(np.asarray(image.convert('L'))).astype(np.float64) / 255.0
    probability = grey if config.get('negate', 0) else 1.0 - grey
    occupancy = np.full(grey.shape, -1, dtype=np.int8)
    occupancy[probability < config['free_thresh']] = 0
    occupancy[probability > config['occupied_thresh']] = 100
    radiation = np.load(directory / 'radiation_field.npy', allow_pickle=False)
    if radiation.shape != occupancy.shape or not np.isfinite(radiation).all() or (radiation < 0).any():
        raise ValueError('Radiation must be finite, nonnegative and match the map shape')
    if factor > 1:
        h, w = occupancy.shape
        pad = ((0, (-h) % factor), (0, (-w) % factor))
        padded = np.pad(occupancy, pad, constant_values=-1)
        shape = (padded.shape[0] // factor, factor, padded.shape[1] // factor, factor)
        blocks = padded.reshape(shape)
        occupancy = np.zeros((shape[0], shape[2]), dtype=np.int8)
        occupancy[(blocks < 0).any(axis=(1, 3))] = -1
        occupancy[(blocks == 100).any(axis=(1, 3))] = 100
        radiation = np.pad(radiation, pad).reshape(shape).max(axis=(1, 3))
    return ResearchMap(occupancy, radiation.astype(np.float32), resolution * factor, origin, directory.name)
