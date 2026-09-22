# Changelog

## Research refresh — 0.1.0

### Reproducible execution

- Added a Docker-first Noetic CPU environment with an installed catkin package.
- Added a headless, collision-checked 2D simulator exercising the original ROS
  policy and online estimator through their actual topics and TF interfaces.
- Added a browser dashboard, bounded experiments, GIF/PNG recording, CSV metrics,
  source/map hashes and machine-readable run outcomes.
- Added a radiation-unaware baseline switch and pinned demo parameters.

### Correctness and performance

- Replaced Python chamfer scans with exact Euclidean distance transforms; this
  deliberately changes the clearance metric and can change graph connectivity.
- Replaced legacy four-connected BFS inflation with equivalent compiled distance
  transforms; retained baseline implementations for benchmarking.
- Fixed negative-origin grid conversion and diagonal corner cutting in legacy
  simulator path smoothing.
- Fixed repeated equal-path relaxations caused by float32 geodesic distances;
  BFS now uses integer hop counts and visits each cell once.
- Corrected farthest-point landmark selection, crop-radius leakage and training
  reachability; local geodesics can follow detours longer than the crop radius.
- Removed huge unreachable-feature sentinels that destabilised GP covariance.
- Reinitialised local GP fits when their feature basis/normalisation changes.
- Serialised estimator callbacks, initialised state before timers, rejected
  invalid measurements, fixed sparse-local-sample crashes, respected smoothing
  settings and applied consistent contrast to IDW hole filling.
- Added bounded seeded sampling and CPU thread controls.

### Repository and documentation

- Fixed install rules referencing nonexistent scripts and completed package
  metadata, dependencies and Python-module installation.
- Removed tracked generated files and broken gitlinks; preserved their hashes and
  recovered all three upstream URLs in an optional pinned vcstool manifest.
- Added README animations, architecture diagrams, research assumptions, unit
  definitions, a compatibility matrix, citation metadata and BSD-3-Clause licence.
- Added numerical regression tests and Docker/ROS verification CI.

The original commit history and research map/field contents are preserved. Full
Gazebo experiment restoration remains distinct from the verified 2D workflow.
