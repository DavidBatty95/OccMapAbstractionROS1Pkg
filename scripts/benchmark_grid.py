#!/usr/bin/env python3
"""Compare retained legacy kernels against compiled replacements on bundled maps."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import time
import hashlib
from datetime import datetime, timezone

import numpy as np
from occmap.grid import clearance, inflate_manhattan, load_map

ROOT = Path(__file__).resolve().parents[1]


def load_node(name):
    path = ROOT / 'src/exploration_pkg/scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(function, *args, repeats=3):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*args)
        samples.append(time.perf_counter() - start)
    return result, {'seconds': samples, 'median_s': statistics.median(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/results/grid-benchmark.json'))
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    policy = load_node('exploration_policy_node')
    sim = load_node('explore_sim_node')
    rows = []
    for name in ('hut_interior', 'reactor_room_sim'):
        grid = load_map(ROOT / 'src/exploration_pkg/maps' / name)
        mask = grid.occupancy != 0
        old, before = measure(policy.legacy_chamfer_distance, mask, repeats=args.repeats)
        new, after = measure(clearance, mask, repeats=args.repeats)
        # Octile approximation and Euclidean metric intentionally differ.
        # The legacy diagonal constant is 1.4142, slightly below sqrt(2).
        assert np.all(new <= old * 1.00002 + 1e-3)
        rows.append({'map': name, 'shape': list(mask.shape), 'kernel': 'clearance',
                     'before': before, 'after': after, 'speedup': before['median_s'] / after['median_s'],
                     'max_metric_difference_cells': float(np.max(abs(new-old))),
                     'max_increase_cells': float(np.max(new-old)),
                     'semantics': 'octile approximation (diagonal 1.4142) -> exact Euclidean'})
        old, before = measure(sim.legacy_inflate, mask.astype(np.uint8), 5, repeats=args.repeats)
        new, after = measure(inflate_manhattan, mask.astype(np.uint8), 5, repeats=args.repeats)
        np.testing.assert_array_equal(old, new)
        rows.append({'map': name, 'shape': list(mask.shape), 'kernel': 'Manhattan inflation, radius 5',
                     'before': before, 'after': after, 'speedup': before['median_s'] / after['median_s'],
                     'semantics': 'identical masks'})
    result = {'schema_version': 1, 'platform': platform.platform(), 'processor': platform.processor(),
              'numpy': np.__version__, 'repeats': args.repeats, 'results': rows,
              'recorded_at': datetime.now(timezone.utc).isoformat(),
              'cpu_model': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                 if line.startswith('model name')), platform.processor()),
              'files_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  Path(__file__).resolve(), ROOT / 'src/exploration_pkg/python/occmap/grid.py',
                  ROOT / 'src/exploration_pkg/scripts/exploration_policy_node.py',
                  ROOT / 'src/exploration_pkg/scripts/explore_sim_node.py')},
              'scope': 'Kernel microbenchmark, not end-to-end exploration speedup; same process, serial timings.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
