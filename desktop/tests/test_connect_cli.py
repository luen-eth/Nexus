"""Exercise the real --connect entrypoint against an authenticated HTTP server."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


class ConnectCliTests(unittest.TestCase):
    def test_environment_key_authenticates_and_reconnects_without_persistence(self):
        key = "connect-cli-test-secret"
        metrics_requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                authorized = self.headers.get("Authorization") == "Bearer " + key
                payload = {
                    "/health": {"status": "ok", "model_loaded": True, "model": "fixture"},
                    "/metrics": {"runtime": {"active_sequences": 0}},
                    "/v1/model": {"loaded": True, "path": "/fixture.gguf"},
                    "/v1/models": {"data": [{"id": "fixture"}]},
                }.get(self.path, {})
                if not authorized and self.path != "/health":
                    payload = {"error": {"message": "Unauthorized"}}
                if self.path == "/metrics" and authorized:
                    metrics_requests.append(True)
                body = json.dumps(payload).encode()
                self.send_response(200 if authorized or self.path == "/health" else 401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        script = r'''
import os, sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLineEdit
import nexus_desktop.window as windows
from nexus_desktop.__main__ import main
expected_key = os.environ["NEXUS_API_KEY"]
class ObservedWindow(windows.MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = 0
        self.check = QTimer(self)
        self.check.timeout.connect(self.observe)
        self.check.start(20)
        QTimer.singleShot(5000, lambda: QApplication.instance().exit(2))
    def observe(self):
        if not self.controller.metrics:
            return
        assert "NEXUS_API_KEY" not in os.environ
        assert self.url.text() == sys.argv[1]
        assert self.key.text() == expected_key
        assert self.key.echoMode() == QLineEdit.EchoMode.Password
        assert not self.controller.owned
        assert self.controller.authorized
        if not self.phase:
            self.phase = 1
            self.controller.disconnect()
            self.connect_server()
        else:
            self.check.stop()
            print("AUTHENTICATED_RECONNECT_OK", flush=True)
            self.close()
windows.MainWindow = ObservedWindow
raise SystemExit(main(["--connect", sys.argv[1]]))
'''
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen",
                           "XDG_CONFIG_HOME": directory, "NEXUS_API_KEY": key}
                    url = f"http://127.0.0.1:{server.server_port}"
                    result = subprocess.run([sys.executable, "-c", script, url], env=env,
                                            capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("AUTHENTICATED_RECONNECT_OK", result.stdout)
                    self.assertGreaterEqual(len(metrics_requests), 2)
                    self.assertNotIn(key, result.stdout + result.stderr)
                    settings = (Path(directory) / "nexus-desktop/settings.json").read_text()
                    self.assertNotIn(key, settings)
                    self.assertEqual(json.loads(settings)["server_url"], url)
            finally:
                server.shutdown()
                worker.join(timeout=3)
