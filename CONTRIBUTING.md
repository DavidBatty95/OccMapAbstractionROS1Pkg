# Contributing

Use the Docker environment in [the README](README.md). It builds and runs the
installed catkin package, so missing install rules are caught during development.

```bash
docker build -t occmap-research:noetic -f docker/Dockerfile .
docker run --rm occmap-research:noetic python3 -m pytest -q tests
bash scripts/demo.sh --duration 60 --check
```

For numerical or planning changes, add a regression test that captures the failure
or invariant, retain a baseline and record the changed assumptions. Keep source
maps and original experiment files intact. New runs belong in ignored `results/`;
publish representative artifacts using `scripts/publish_demo.py`.

Include the relevant `summary.json`, `manifest.json` and ROS log when reporting a
bug. Performance claims should include raw timings, hardware/runtime details and
the exact configuration. A smoke-test pass is not evidence of mission completion
or improved radiation avoidance. See [methodology](docs/METHODOLOGY.md).

Contributions to this project's own code are under [BSD-3-Clause](LICENSE).
Preserve separate notices for external packages and assets.
