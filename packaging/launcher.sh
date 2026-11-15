#!/bin/sh
# The override supports extracted-package smoke tests without a system install.
# Installed launchers use the immutable packaged application directory by default.
set -eu
app_dir=${NEXUS_DESKTOP_APP_DIR:-/usr/share/nexus-desktop}
PYTHONPATH="$app_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
exec /usr/bin/python3 -P -m nexus_desktop "$@"
