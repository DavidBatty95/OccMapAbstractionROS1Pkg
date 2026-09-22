#!/usr/bin/env bash
set -eo pipefail
source /opt/occmap/install/setup.bash
set -u
exec "$@"
