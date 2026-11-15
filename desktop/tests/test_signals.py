"""Linux subprocess regressions for idle/owned Qt shutdown by SIGTERM."""
import os
from pathlib import Path
import selectors
import socket
import subprocess
import sys
import tempfile
import time
import unittest


class SignalTests(unittest.TestCase):
    def test_idle_cli_exits_promptly_on_sigterm(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "QT_QPA_PLATFORM":"offscreen", "XDG_CONFIG_HOME":directory}
            process = subprocess.Popen([sys.executable,"-m","nexus_desktop"],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                time.sleep(.65)
                self.assertIsNone(process.poll())
                process.terminate()
                process.wait(timeout=3)
                self.assertEqual(process.returncode,0,process.stderr.read().decode())
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=3)

    def test_sigterm_closes_only_the_apps_owned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1",0))
                port=sock.getsockname()[1]
            fake=str(Path(__file__).with_name("fake_server.py"))
            script=r'''
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from nexus_desktop.__main__ import install_signal_handlers
from nexus_desktop.window import MainWindow
from nexus_desktop.config import Settings
app=QApplication([])
window=MainWindow(Settings(binary=sys.argv[1],port=int(sys.argv[2])))
window.show()
timer=install_signal_handlers(app,window)
reported=[False]
def ready():
    if window.controller.health and window.controller.owned and not reported[0]:
        reported[0]=True
        print(window.controller.process.processId(),flush=True)
window.controller.changed.connect(ready)
QTimer.singleShot(0,window.start_server)
raise SystemExit(app.exec())
'''
            env={**os.environ,"QT_QPA_PLATFORM":"offscreen","XDG_CONFIG_HOME":directory}
            process=subprocess.Popen([sys.executable,"-c",script,fake,str(port)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            child=None
            try:
                with selectors.DefaultSelector() as ready:
                    ready.register(process.stdout,selectors.EVENT_READ)
                    self.assertTrue(ready.select(6),"owned GUI did not become ready")
                    line=process.stdout.readline().strip()
                    self.assertTrue(line.isdigit(),line)
                    child=int(line)
                process.terminate()
                process.wait(timeout=6)
                self.assertEqual(process.returncode,0,process.stderr.read())
                with self.assertRaises(ProcessLookupError):
                    os.kill(child,0)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=3)
                if child:
                    try:
                        os.kill(child,15)  # Only the PID reported by this test's own GUI child.
                    except ProcessLookupError:
                        pass
