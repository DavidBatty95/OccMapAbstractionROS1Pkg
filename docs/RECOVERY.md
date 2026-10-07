# GitHub-only recovery and release preparation

## Recovery checkpoint — 7 October 2026

The default branch at `ea6e360` was an incomplete workspace export. A full clone
also recovered `origin/refresh/reproducible-research-demo`, ending at `df56e91`.
That branch already contained three committed milestones: generated-file cleanup,
a Docker reference simulator, and documented recordings. This refresh builds on
that existing work rather than reconstructing it from scratch.

No files from the original development machine were needed. The five map/field
pairs and research nodes come from GitHub. The three original external commit
references have public source locations recorded in `legacy.repos`; see
[legacy recovery](LEGACY.md). Their availability does not establish that custom
Gazebo worlds, controllers or experimental settings are recoverable.

## Supported release scope

The supported demonstration runs the ROS exploration policy and online estimator
in a 2D reference simulator, with a browser view and persistent measurements.
The simulator executes the policy's corridors, publishes navigation feedback,
and checks motion against observed space and simulator geometry. It is not a
restored Gazebo experiment or a quadruped locomotion simulation.

The existing refresh changes clearance calculations and estimator implementation;
these are documented in [performance](PERFORMANCE.md) and
[methodology](METHODOLOGY.md). New recordings must not be described as exact
reproductions of historical thesis results.

## Release acceptance

- Build the installed catkin workspace in Docker.
- Run numerical, simulator and packaging regression checks.
- Run bounded radiation-aware and radiation-disabled ROS sessions.
- Require movement, graph publication, map publication and an online radiation
  estimate where enabled; retain failed runs and logs.
- Check downloadable documentation and historical evidence references.
- Record fresh verification separately from historical results.

For scientific comparisons, use matched simulated-time or coverage stopping
criteria and repeated trials. The current recording duration is wall-clock time,
so equal duration arguments alone do not provide a controlled dose comparison.
