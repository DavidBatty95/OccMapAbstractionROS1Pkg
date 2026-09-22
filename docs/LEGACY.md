# Original workspace and Gazebo compatibility

The imported public tree at commit `ea6e360be177ff992091eeab21a9b4103d208593`
contained catkin build products and three gitlinks without `.gitmodules` URLs.
Generated files have been removed from tracking; original history remains intact.
The original gitlink hashes are preserved in
[`archive/external-dependencies.json`](../archive/external-dependencies.json).

All three original commits were verified against public upstream repositories:

| Source | Revision |
| --- | --- |
| [EEEManchester/gazebo_radiation_plugin](https://github.com/EEEManchester/gazebo_radiation_plugin) | `5c75bf18d1de9a70f4e8f21baaef9e09736dfd79` |
| [EEEManchester/simple_radiation_layer](https://github.com/EEEManchester/simple_radiation_layer) | `931d71d7b0853d1fd10f6e81b7e2ab83ebef28fd` |
| [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) | `29cc27f7578e010165042e1fe45cc18f3a4dd2ca` |

[`legacy.repos`](../legacy.repos) is an optional vcstool manifest for recovering
these sources into a **separate** Noetic workspace:

```bash
# Requires vcstool; run from this repository's root.
mkdir -p /tmp/occmap-gazebo/src
vcs import /tmp/occmap-gazebo/src < legacy.repos
```

This downloads upstream sources; it does not reconstruct custom local changes
that were never committed inside those gitlinks. The default Docker image uses
only `exploration_pkg` and needs none of these downloads.

## Compatibility table

| Entry point | Status |
| --- | --- |
| `demo.launch` | Containerised reference pipeline; recorded end to end |
| `exploration.launch` | Original policy + estimator + corridor navigator; expects an external map, TF and running `move_base` |
| `explore_sim_node.launch` | Original interactive grid simulator retained; not the validated reference demo |
| `navigation_corridor_node.py` | Original `move_base` action client retained; server/integration not bundled in the demo |
| `obstacleradGPR.py` | Experimental older estimator; requires the external Gazebo message package; not installed as a supported demo node |
| `precompute_rad.py` | Original field-generation utility; existing fields are used directly by the demo |
| `mission_tui_node.py` | Original terminal UI; historical unit labels are not used for the new demo's scientific metrics |

## Work required for full Gazebo reproduction

Recover the exact world/model assets and simulation launch sequence used in the
experiments, the navigation stack configuration, robot controller setup, and any
uncommitted plugin changes. Verify their provenance against the pinned upstreams.
Then build and record a clean Gazebo run with the same metadata and regression
checks as the 2D pipeline. This repository does not currently claim to reproduce
that complete experiment from a fresh clone.

The original maps, radiation source YAML, field arrays and generation metadata
remain in `src/exploration_pkg/maps/`. They are not regenerated or rescaled on disk
by the refresh. An exact earlier code baseline can be inspected with
`git show ea6e360:src/exploration_pkg/scripts/exploration_policy_node.py`.
