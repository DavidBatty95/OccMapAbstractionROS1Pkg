# Download and demonstrate the PhD research

## First run

Use x86-64 Linux with Docker Engine and Docker Compose v2 or newer. No host ROS,
GPU, X11 display or robot model download is needed for the browser demonstration.

Clone the repository, or use GitHub's **Code → Download ZIP** and extract it:

```bash
git clone https://github.com/DavidBatty95/OccMapAbstractionROS1Pkg.git
cd OccMapAbstractionROS1Pkg
bash occmap setup
bash occmap demo
```

Open http://localhost:8080. The first build requires internet access and several
GB of disk space. After the two-minute recording, the page stays available until
Ctrl+C. Run only one browser demo on port 8080 at a time.

For another scenario:

```bash
bash occmap demo --map barrel_store_sim --duration 120
```

## Explain the demonstration

1. **Map discovery:** light areas represent observed free space; dark areas remain
   unknown. The simulator raycasts the bundled map, rather than providing the
   complete map to the policy.
2. **Abstraction:** graph edges summarise traversable connections in observed
   space. Orange corridors show the route selected by the policy.
3. **Radiation inference:** the right panel shows the online cost estimate from
   point measurements, not the ground-truth field. It is a normalised cost display,
   not calibrated dose or uncertainty.
4. **Execution:** the blue trail records actual reference-simulator movement.
   Motion is holonomic and collision checked; it is not physical Go2 locomotion.
5. **Evidence:** open the summary, manifest and CSV links. A successful pipeline
   check establishes active mapping, graph publication and movement, not mission
   completion or a statistically significant reduction in exposure.

## Radiation-aware and disabled runs

```bash
bash occmap run --sim-duration 30 --duration 120 --seed 42
bash occmap run --sim-duration 30 --duration 120 --seed 42 --no-radiation
```

These target the same simulated duration, while allowing up to 120 wall-clock
seconds for startup and recording. The recorder samples asynchronously, so small
stopping overshoots are possible: use the actual simulated timestamps in each
CSV when comparing. Seeds do not make ROS scheduling deterministic. Compare
coverage, distance, exposure and route rejections together; one pair is a demo,
not a reliability study. A timeout or interrupted checked run fails explicitly.

## Results and sharing

Each run writes a separate directory under `results/`: browser page, final image,
animation, metrics, manifest, summary and ROS log. Share the whole directory to
retain provenance. Recorded results can be viewed without ROS or Docker:

```bash
python3 -m http.server 8081 --bind 127.0.0.1 --directory results
```

Open http://localhost:8081 and choose a run directory. Serving results does not
start a simulation. For an offline demonstration machine, save a built image:

```bash
docker save occmap-research:noetic | gzip > occmap-noetic.tar.gz
# On the receiving Linux x86-64 Docker host:
docker load < occmap-noetic.tar.gz
mkdir -p results
docker run --rm --init --user "$(id -u):$(id -g)" \
  -p 127.0.0.1:8080:8080 -v "$PWD/results:/results" occmap-research:noetic
```

The image contains code, maps and runtime dependencies. No source checkout is
needed for this last command. Transfer a checksum alongside the image archive.
The repository does not yet publish a downloadable prebuilt image automatically.

## Research context

The thesis is [Radiation Aware Navigation and Planning for Mobile Robot
Exploration of Unknown Hazardous Nuclear Environments](https://doi.org/10.17638/03199094).
This demo illustrates occupancy-map abstraction, radiation inference and
radiation-informed exploration. It does not reproduce the thesis's hardware
experiments, calibrated radiation measurements or historical Gazebo results.
See [methodology](METHODOLOGY.md), [GitHub recovery](RECOVERY.md),
[legacy restoration](LEGACY.md) and [verification](VERIFICATION.md).
