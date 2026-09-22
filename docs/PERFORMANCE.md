# Grid-kernel performance

Generated from [raw timings](results/grid-benchmark.json). Measurements were
collected in the Docker environment using the native-resolution bundled maps.
The retained legacy functions provide the baseline in the same Python process.

**Host CPU:** 13th Gen Intel(R) Core(TM) i7-13800H. **NumPy:** 1.24.4. **Repeats:** 3 per kernel.

| Map | Kernel | Legacy median (ms) | Updated median (ms) | Kernel speedup |
| --- | --- | ---: | ---: | ---: |
| hut_interior | clearance | 1131.56 | 5.41 | 209.3× |
| hut_interior | Manhattan inflation, radius 5 | 38.36 | 0.89 | 43.1× |
| reactor_room_sim | clearance | 5884.33 | 29.66 | 198.4× |
| reactor_room_sim | Manhattan inflation, radius 5 | 370.06 | 5.04 | 73.5× |

## Interpretation

- Inflation masks are checked for exact equality against the original BFS.
- Clearance deliberately changes from a two-pass octile approximation to
  exact Euclidean distance. Most distances are lower; the old rounded diagonal
  constant can also produce very small underestimates. Raw results include both
  maximum metric differences and maximum increases. This is not a bitwise-equivalent
  planner optimisation and can change graph clearance decisions.
- Results measure these kernels, not total exploration speed, mapping accuracy
  or scientific quality. Host scheduling, CPU scaling and background work affect
  timings; raw samples are included instead of only a headline ratio.
- The GP geodesic fix bounds each BFS to one visit per reachable cell. Its
  correctness is regression-tested; no timing ratio is claimed for the formerly
  pathological floating-point relaxation loop.

## Reproduce

```bash
mkdir -p results
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/results:/results" \
  occmap-research:noetic python3 scripts/benchmark_grid.py --repeats 3
python3 scripts/refresh_results.py --benchmark results/grid-benchmark.json
```

Build the image first using the [installation instructions](INSTALLATION.md).
Run serially without a simultaneous demo when collecting new performance evidence.
