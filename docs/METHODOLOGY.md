# Research methodology and scope

## What is reproduced

The quick-start runs the original `exploration_policy_node.py` and
`rad_GPR_node.py` with documented fixes in a new self-contained ROS reference
simulator. It uses the repository's original occupancy maps and precomputed
radiation fields. It is a runnable integration and inspection baseline, not an
exact reconstruction of the PhD's Gazebo experiments or a new validated result.

## Information boundary

`research_demo_node.py` alone loads ground-truth occupancy and radiation. It
chooses a valid start nearest map-frame (0, 0) within the largest traversable
component. This start placement uses truth; subsequent planning sees only:

- `/map`: initially unknown occupancy, progressively revealed by 240 rays over
  360 degrees with a default 3.5 m range. Rays stop at occupied or unknown truth.
- `/sim_rad_sensor`: the current cell's dose-rate value as `std_msgs/Float32`.
- TF `map → base_link`: ideal localisation.

The policy receives no truth arrays. The estimator receives point measurements,
TF and observed occupancy. The simulator uses truth again for collision checks
and scoring. A nominal 0.12 m circular footprint is conservatively inflated by a
half-cell diagonal; the executed segment must also remain in observed free cells.
The recorder uses the truth map's known bounding box only to frame the display and
remove empty image padding; this camera viewport is not an input to the policy.

## Abstraction, estimation and control

The policy samples free-space graph nodes, checks clearances/visibility, clusters
frontiers and uses graph A* to construct corridors. Frontier scoring incorporates
information gain, heading, travel distance, visits and optional radiation costs.
The radiation-aware mode also applies radiation soft/hard goal caps and an
exclusion mask with the policy's existing fallback behaviour.

The estimator aggregates measurements into spatial voxels, builds local geodesic
landmark features and fits a Matérn sparse variational GP. The demo uses 32 inducing
points and eight training steps per fit on CPU. Each local fit is reinitialised:
changing landmarks and target normalisation changes the feature basis, so carrying
over an old variational posterior is not valid. These settings are a lightweight
demo configuration, not tuned or cross-validated hyperparameters.

The estimator's published costs retain the legacy IDW floor, contrast transform
and monotonic global envelope. Consequently `/radiation_costmap` is a **heuristic
normalised planning cost**, not the raw posterior mean. `/radiation_sigma` is a
normalised visualisation/penalty channel, not calibrated uncertainty. Zero cost
in an unestimated area does not establish low radiation. Straight-line IDW can
influence points around a wall within a connected reachable region; it is not a
radiation transport solver. Smoothing is disabled in the demo.

`Float32` measurements lack timestamps; the mapper associates them with the latest
TF pose. Use a timestamped sensor interface before drawing conclusions about
high-speed sensing accuracy. Map origins must be axis aligned in this demo.

The simulator follows the published corridor directly with holonomic, constant
speed motion (0.6 m/s) and a 0.1 s fixed integration step. It has no dynamics,
localisation drift, scan noise, turning-time penalty or `move_base` local planner.
It stops a rejected motion segment and reports it rather than passing through a
wall. It intentionally does not use the legacy unicycle simulator's unchecked
motion or bypass the graph route with an independent shortest path.

## Map processing

PGM images are vertically flipped to ROS row order. Occupancy thresholds and
`negate` come from each map YAML. World-to-grid conversion uses floor, so points
just below an origin cannot truncate incorrectly into cell zero.

Downsampling pools blocks conservatively: occupied dominates unknown, which
dominates free. Incomplete edge blocks are padded unknown. Radiation uses the
block maximum. Changing `factor` changes geometry and exposure, so comparisons
must hold it fixed. The loader rejects mismatched, negative or nonfinite fields.

## Metrics

The authoritative demo metrics come from `/demo/metrics`, not the historical TUI
or the policy's internal cost-integral telemetry.

| Field | Definition |
| --- | --- |
| `simulation_time_s` | Number of executed timer steps × 0.1 s, including stationary steps |
| `distance_m` | Sum of executed translational step lengths |
| `dose_relative_rate_seconds` | Sum of sampled dose rate × 0.1 s; includes stationary exposure |
| `observed_free_fraction` | Observed free cells / all free cells in the processed truth map |
| `dose_rate_relative` | Current processed truth-field cell value |
| `rejected_motion_segments` | Motion attempts stopped by collision/unknown-space checks, plus invalid incoming paths |
| `routes_received` | Number of nonempty, finite map-frame corridors received |

The observed-free denominator includes disconnected free regions. It is not a
reachable-space completion criterion. `mission_state` is the original policy's
own completion/return logic; a timed-out recording may still be `EXPLORING`.

The original field metadata explicitly says `relative_dose_rate`. No conversion
to μSv, mSv or Sv is justified. The legacy parameter name `abs_scale_uSv` is
retained for compatibility but acts only as a normalisation scale here.

## Reproducibility

Each recording stores parameters, source/map SHA-256 hashes, package versions,
per-sample metrics and the ROS log. NumPy and Torch sampling are seeded. ROS
callbacks and wall-clock planning are asynchronous; equal seeds do **not** make
whole runs bitwise deterministic. A wall-clock duration can yield different
simulation times on different hosts. GIFs play at 100 ms per sampled frame;
they are accelerated, not real-time recordings.

For research comparisons:

1. Hold map, factor, start, sensor range, speed and compute allocation fixed.
2. Record both radiation-aware and unaware modes across multiple seeds/maps.
3. Compare dose at matched simulated duration **and** achieved coverage, including
   stalled and failed runs. A slower or less complete run can have a misleadingly
   low exposure total.
4. Distinguish estimator error, planner performance and simulator artefacts.
5. Report distributions and failures, not just favourable animations.

The published runs establish that the pipeline executes and produces observations.
They do not establish statistical superiority, full exploration, hardware safety,
or physical radiation accuracy.
