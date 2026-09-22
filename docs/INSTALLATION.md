# Installation and operation

## Supported container workflow

Tested on x86-64 Ubuntu Linux with Docker Engine 29 and Compose v2. Docker contains
Ubuntu 20.04, ROS Noetic, Python 3.8, CPU PyTorch and GPyTorch. Host ROS packages are
not used. Windows/WSL2, Docker Desktop and ARM hosts have not been validated; the
CPU Torch wheel pin currently targets x86-64 Linux.

Install Docker Engine and the Compose plugin using the
[official instructions](https://docs.docker.com/engine/install/). Your account
must be able to run `docker info` and `docker compose version` successfully.

```bash
git clone https://github.com/DavidBatty95/OccMapAbstractionROS1Pkg.git
cd OccMapAbstractionROS1Pkg
bash scripts/demo.sh
```

Visit <http://localhost:8080>. The first build downloads several GB; later builds
reuse layers. The page updates during the recording, then exposes its GIF, CSV,
summary, log and manifest. The default session records for 120 wall-clock seconds
after ROS startup. Ctrl+C stops the service. Run `docker compose down` to remove
the stopped Compose container/network; recorded files remain on the host.

The launcher creates `results/` and sets the container user to your UID/GID. It
does not install host packages, change Docker permissions, or require privileged
containers, host networking, X11 or GPU passthrough.

## Batch mode

```bash
bash scripts/demo.sh --duration 120 --map reactor_room_sim --seed 42 --check
bash scripts/demo.sh --duration 120 --map barrel_store_sim --seed 42 --check
bash scripts/demo.sh --duration 120 --no-radiation --check
```

Passing arguments selects a bounded headless run that exits when finished. Each
run creates a uniquely named subdirectory of `results/`; earlier runs are retained.
`--check` requires map and graph messages, more than one metre of travel, and a
nonzero radiation estimate when enabled. A logged Python exception fails the run.
It does not require full exploration or lower dose than a baseline.

`--map` accepts any bundled map name. `--factor` conservatively downsamples the
occupancy map and maximum-pools the dose field. The default factor 5 gives 0.1 m
cells for the simulated maps with 0.02 m native resolution. The hut maps have
0.05 m native resolution: use `--factor 2` to obtain 0.1 m cells there. Use factor 1
to retain native resolution, with correspondingly higher compute cost.

To serve a different live case, use the Compose port mapping:

```bash
mkdir -p results
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
docker compose run --rm --service-ports demo \
  python3 scripts/run_demo.py --map barrel_store_sim --duration 180 --output /results --serve
```

Run only one service on port 8080. Independent batch containers have isolated ROS
masters and can run concurrently, but shared CPU load affects scheduling and
performance measurements.

## Development and tests

```bash
docker build -t occmap-research:noetic -f docker/Dockerfile .
docker run --rm occmap-research:noetic python3 -m pytest -q tests
docker run --rm occmap-research:noetic python3 -m compileall -q src scripts
```

Rebuild after editing source or configuration. The image runs from catkin's
**install space**, not a bind-mounted developer workspace. This verifies the
package's module, node, map, configuration and launch installation rules.

For an existing native Noetic installation, use `rosdep install --from-paths src
--ignore-src -r -y`, install `docker/requirements.txt` in your chosen Python 3.8
environment, build with `catkin_make install`, source `install/setup.bash` and run
`roslaunch exploration_pkg demo.launch`. The verified route remains Docker.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Cannot connect to Docker daemon | Check `docker info` and your Docker installation/account permissions. |
| Port 8080 already allocated | Stop the earlier demo or change the host side of the mapping in `compose.yaml`. |
| Permission denied writing `/results` | Use `bash scripts/demo.sh`; ensure `results/` is writable by your account. |
| No radiation colour initially | The GP needs at least eight local voxels; inspect `rad_GPR` messages in `ros.log`. |
| `--check` fails | Inspect `summary.json` and `ros.log`; preserve the failed run. Slow hosts may need a longer recording. |
| Map loads but little motion occurs | Inspect policy rejection reasons and map factor; coarsening can close narrow passages. |
| Cannot find original Gazebo worlds | Follow [legacy restoration](LEGACY.md); they are not part of the self-contained demo. |

ROS Noetic is end-of-life. The image deliberately preserves that ROS API for
legacy research reproducibility; a future ROS 2 port should be validated against
these artifacts rather than silently changing the recorded environment.
