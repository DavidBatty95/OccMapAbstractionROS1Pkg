#!/usr/bin/env python3
"""Build the published result tables from retained machine-readable evidence."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path)
    args = parser.parse_args()
    result_dir = ROOT / 'docs/results'
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.benchmark:
        shutil.copyfile(args.benchmark, result_dir / 'grid-benchmark.json')
    lines = ['# Recorded reference runs', '',
             'Generated from the retained summaries by `python3 scripts/refresh_results.py`.', '',
             'These are bounded ROS integration demonstrations. All three runs used seed 42,',
             'map factor 5 and a 120-second wall-clock recording. Full missions were not',
             'completed. The paired reactor runs are descriptive examples, not a multi-seed',
             'evaluation. Their simulated durations and achieved coverage differ.', '',
             '| Run | Simulated time (s) | Observed free (%) | Path (m) | Exposure (relative-rate s) | Rejected motion | Pipeline check |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for name in ('reactor', 'barrel', 'unaware'):
        summary = json.loads((result_dir / name / 'summary.json').read_text())
        m = summary['metrics']
        lines.append('| [%s](results/%s/summary.json) | %.1f | %.2f | %.2f | %.1f | %d | %s |' % (
            name, name, m['simulation_time_s'], 100 * m['observed_free_fraction'], m['distance_m'],
            m['dose_relative_rate_seconds'], m['rejected_motion_segments'], 'pass' if summary['smoke_pass'] else 'fail'))
    aware = json.loads((result_dir / 'reactor/summary.json').read_text())['metrics']
    unaware = json.loads((result_dir / 'unaware/summary.json').read_text())['metrics']
    lines += ['', 'In this recorded reactor pair, the aware policy accumulated **%.1f relative-rate s**' % aware['dose_relative_rate_seconds'],
              'against **%.1f** for the unaware baseline, with **%.2f%% vs %.2f%%** observed-free coverage.' % (
                  unaware['dose_relative_rate_seconds'], 100 * aware['observed_free_fraction'], 100 * unaware['observed_free_fraction']),
              'These outcomes are retained as measured; the demo is not evidence that radiation',
              'awareness improves the exploration objective. Estimator calibration and policy',
              'evaluation remain research work.', '',
              'The unaware run disables the radiation estimator and radiation-based policy',
              'penalties. The simulator still measures exposure. A lower exposure in one',
              'recording does not establish general superiority; see [methodology](METHODOLOGY.md).', '',
              '## Radiation-unaware baseline recording', '',
              '![Recorded radiation-unaware reactor exploration](media/unaware.gif)', '',
              'The right panel has no estimate in this mode. The trajectory and dose integral',
              'still come from the simulator. The other recordings are embedded in the',
              '[README](../README.md). GIFs are accelerated; the image timestamp shows simulation time.', '',
              '## Raw evidence', '',
              'Each `docs/results/<run>/` directory contains `summary.json`, `manifest.json`,',
              '`metrics.csv` and `ros.log`. Manifests record source/map hashes and dependency',
              'versions; image/CSV hashes tie the published media to the run. These recorded',
              'source hashes identify the implementation at capture time, even if documentation',
              'or development tooling changes later.', '',
              'Regenerate with the commands in [installation](INSTALLATION.md), then publish:', '',
              '```bash',
              'python3 scripts/publish_demo.py results/<recorded-run-directory> --name reactor',
              '# Use --name barrel or --name unaware for the other cases.',
              'python3 scripts/refresh_results.py',
              '```', '',
              'Failed development runs remain in the local ignored `results/` directory. CI',
              'uploads diagnostics even on failure. The three published records are selected',
              'pipeline demonstrations, not a complete experimental dataset.', '']
    (ROOT / 'docs/RESULTS.md').write_text('\n'.join(lines))
    data = json.loads((result_dir / 'grid-benchmark.json').read_text())
    lines = ['# Grid-kernel performance', '',
             'Generated from [raw timings](results/grid-benchmark.json). Measurements were',
             'collected in the Docker environment using the native-resolution bundled maps.',
             'The retained legacy functions provide the baseline in the same Python process.', '',
             '**Host CPU:** %s. **NumPy:** %s. **Repeats:** %d per kernel.' %
             (data['cpu_model'], data['numpy'], data['repeats']), '',
             '| Map | Kernel | Legacy median (ms) | Updated median (ms) | Kernel speedup |',
             '| --- | --- | ---: | ---: | ---: |']
    for row in data['results']:
        lines.append('| %s | %s | %.2f | %.2f | %.1f× |' % (row['map'], row['kernel'],
            1000 * row['before']['median_s'], 1000 * row['after']['median_s'], row['speedup']))
    lines += ['', '## Interpretation', '',
              '- Inflation masks are checked for exact equality against the original BFS.',
              '- Clearance deliberately changes from a two-pass octile approximation to',
              '  exact Euclidean distance. Most distances are lower; the old rounded diagonal',
              '  constant can also produce very small underestimates. Raw results include both',
              '  maximum metric differences and maximum increases. This is not a bitwise-equivalent',
              '  planner optimisation and can change graph clearance decisions.',
              '- Results measure these kernels, not total exploration speed, mapping accuracy',
              '  or scientific quality. Host scheduling, CPU scaling and background work affect',
              '  timings; raw samples are included instead of only a headline ratio.',
              '- The GP geodesic fix bounds each BFS to one visit per reachable cell. Its',
              '  correctness is regression-tested; no timing ratio is claimed for the formerly',
              '  pathological floating-point relaxation loop.', '',
              '## Reproduce', '', '```bash',
              'mkdir -p results',
              'docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/results:/results" \\',
              '  occmap-research:noetic python3 scripts/benchmark_grid.py --repeats 3',
              'python3 scripts/refresh_results.py --benchmark results/grid-benchmark.json',
              '```', '', 'Build the image first using the [installation instructions](INSTALLATION.md).',
              'Run serially without a simultaneous demo when collecting new performance evidence.', '']
    (ROOT / 'docs/PERFORMANCE.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
