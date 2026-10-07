# Verification record

## GitHub-only recovery — 7 October 2026

Validated the recovered branch and new `occmap` launcher on Linux x86-64 with
Docker 29.8.1 and Compose 5.5.1. The Docker build reused existing dependency layers;
this was not a cache-free dependency-availability test.

| Check | Outcome |
| --- | --- |
| Installed catkin Docker build | Passed |
| Existing numerical and ROS-node regression suite | 20 passed in 2.65 s |
| Documentation links and historical evidence hashes | Passed |
| Radiation-aware reactor run, seed 42, target 30 simulated seconds | Passed; 30.3 s, 18.02 m, 31.59% observed free space, no rejected segments |
| Radiation-disabled reactor run, same target and seed | Passed; 30.0 s, 17.70 m, 28.96% observed free space, no rejected segments |
| Browser HTTP session with 10 simulated-second target | HTML HTTP 200, PNG served, recorded summary passed |
| Browser container shutdown | Exit code 0 after Docker stop |
| Deliberately unattainable 300 s target / 5 s wall limit | Exit code 1; summary failed with explicit timeout |

The two fresh runs remain partial exploration sessions (`EXPLORING`), not mission
completion. Their relative exposure integrals were 1338.55 and 4253.75 relative-rate
seconds respectively. This single pair with slightly different stopping times
does not establish a general reduction in dose. Retained evidence:
[aware summary](results/reactor-20261007/summary.json),
[disabled summary](results/unaware-20261007/summary.json), with manifests, metrics
and ROS logs alongside them. Fresh animations are under `docs/media/` with matching
date-stamped names; the README's original recordings remain historical evidence.

## Original refresh — 22 September 2026

The refresh was verified locally on 22 September 2026, using x86-64 Linux and
the repository's Docker image. The Noetic base image is pinned by digest in
`docker/Dockerfile`; runtime dependency versions are stored in each run manifest.

| Check | Outcome |
| --- | --- |
| Docker build, including `catkin_make install` | Passed |
| Numerical and ROS-node regression tests | 20 passed |
| `python3 -m pip check` inside the image | No broken requirements |
| Python compilation of `src/` and `scripts/` | Passed |
| Compose configuration and shell syntax | Passed |
| Browser HTTP view and JSON endpoints | Served successfully from the container |
| Reactor, barrel-store and radiation-unaware ROS runs | Recorded; pipeline checks passed; see retained summaries |
| Inflation baseline equivalence | Identical masks on both benchmark maps |
| Documentation targets and published artifact hashes | Verified by `scripts/check_repository.py` |

Tests cover negative-origin coordinates, analytic Euclidean clearance, empty
obstacle maps, Manhattan inflation, line-of-sight corner cutting, bounded
geodesic distances, ROS map orientation, conservative pooling, invalid map/field
inputs, all five bundled map/field pairs, legacy A* corner behaviour, finite GP
training with geodesic detours, local reachability and sparse-sample fallback.

The GitHub workflow repeats the Docker build, tests and a 60-second integration
smoke test when the branch is published. This document records local execution,
not an assertion that a remote GitHub Actions run has already completed.

## Remaining validation boundaries

- Only the stated x86-64 Docker route is verified; there is no ARM/GPU CI matrix.
- The recorded runs are partial exploration sessions, not completed missions.
- Full Gazebo navigation, the original unicycle simulator and hardware operation
  are not covered by the new integration check.
- There is no multi-seed evaluation establishing dose-reduction performance or
  calibrated estimation uncertainty. Read [methodology](METHODOLOGY.md) before
  using these demonstrations as scientific evidence.
