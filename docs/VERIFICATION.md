# Verification record

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
