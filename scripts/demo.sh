#!/usr/bin/env bash
set -euo pipefail
ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
mkdir -p "$ROOT/results"
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
if [ "$#" -eq 0 ]; then
  exec docker compose --project-directory "$ROOT" up --build
fi
ports=()
for arg in "$@"; do
  if [ "$arg" = "--serve" ]; then ports=(--service-ports); fi
done
exec docker compose --project-directory "$ROOT" run --rm --build "${ports[@]}" demo \
  python3 /opt/occmap/scripts/run_demo.py --output /results "$@"
