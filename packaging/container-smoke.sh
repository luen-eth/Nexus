#!/bin/sh
set -eu
# Intended for a disposable ubuntu:26.04 or debian:13 Docker image, never the host.
[ -f /.dockerenv ] || { echo "Run this smoke script inside Docker" >&2; exit 2; }
export DEBIAN_FRONTEND=noninteractive
apt-get update
set -- /packages/debian/nexus-desktop_*_all.deb
[ "$#" -eq 1 ] && [ -f "$1" ] || { echo "Provide exactly one GUI .deb in /packages/debian" >&2; exit 2; }
apt-get install -y --no-install-recommends "$1"
mkdir -p /tmp/nexus-smoke-runtime /tmp/nexus-smoke-cwd
chmod 700 /tmp/nexus-smoke-runtime
export QT_QPA_PLATFORM=offscreen XDG_RUNTIME_DIR=/tmp/nexus-smoke-runtime XDG_CONFIG_HOME=/tmp/nexus-smoke-config
nexus-desktop --version
nexus-desktop --help
printf '%s\n' 'raise RuntimeError("untrusted cwd module executed")' > /tmp/nexus-smoke-cwd/nexus_desktop.py
cd /tmp/nexus-smoke-cwd
nexus-desktop --version
PYTHONPATH=/usr/share/nexus-desktop python3 -P - <<'PY'
from PySide6 import __version__
from PySide6.QtCore import QTimer,QEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from pathlib import Path
import nexus_desktop
from nexus_desktop.window import MainWindow
from nexus_desktop.config import Settings
app=QApplication(['nexus-deb-smoke'])
assert not QIcon(str(Path(nexus_desktop.__file__).parent/'assets/nexus.svg')).pixmap(40,40).isNull()
w=MainWindow(Settings());w.show();app.processEvents()
assert w.isVisible()
import os
label=os.environ.get('NEXUS_SMOKE_LABEL','UBUNTU26')
assert w.grab().save('/evidence/'+label.lower()+'-deb.png')
QTimer.singleShot(10,w.close);QTimer.singleShot(100,app.quit)
app.exec()
w.deleteLater();app.sendPostedEvents(None,QEvent.Type.DeferredDelete)
print(label+'_PACKAGED_QT_WINDOW_OK',__version__)
import subprocess,time
process=subprocess.Popen(['/usr/bin/nexus-desktop'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
try:
    time.sleep(1)
    assert process.poll() is None
    process.terminate()
    stdout,stderr=process.communicate(timeout=5)
    assert process.returncode==0,(process.returncode,stderr)
    print(label+'_IDLE_SIGTERM_EXIT_0')
finally:
    if process.poll() is None:
        process.kill();process.communicate()
PY
dpkg-query -W python3 python3-pyside6.qtwidgets python3-pyside6.qtnetwork libqt6svg6 qt6-svg-plugins nexus-desktop

# Test-only binding; production GUI does not need QtTest.
if [ -d /source/desktop/tests ]; then
    apt-get install -y --no-install-recommends python3-pyside6.qttest
    # The hostile-CWD launcher fixture must not contaminate dev-mode child
    # Python invocations in the GUI unit suite (which intentionally use -m).
    cd /tmp
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/usr/share/nexus-desktop python3 -P -m unittest discover -s /source/desktop/tests -v
fi
