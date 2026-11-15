import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest
from PySide6.QtTest import QTest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication
from nexus_desktop.api import ApiClient

APP = QApplication.instance() or QApplication(["nexus-api-tests"])
APP.setQuitOnLastWindowClosed(False)


def wait(predicate, seconds=3):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        APP.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    return bool(predicate())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        bodies = {"/ok": b'{"value":"ok"}', "/empty": b"", "/invalid": b"not-json", "/array": b"[]", "/huge": b'{"text":"'+b"x"*8192+b'"}'}
        payload = bodies.get(self.path, b'{"slow":"'+b"x"*100+b'"}')
        compressed = self.path == "/gzip"
        if compressed:
            payload = gzip.compress(b'{"text":"'+b"x"*65536+b'"}')
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        if self.path != "/huge":  # Exercise decoded running-total enforcement too.
            self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            if self.path == "/drip":
                for byte in payload:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(0.04)
            else:
                self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=lambda:self.server.serve_forever(poll_interval=.02), daemon=True)
        self.thread.start()
        self.client = ApiClient()
        self.client.configure(f"http://127.0.0.1:{self.server.server_port}")
        self.client.MAX_RESPONSE_BYTES = 1024

    def tearDown(self):
        self.client.cancel()
        self.client.deleteLater()
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)

    def request(self, path, timeout=1000):
        seen = []
        self.client.request("test", path, lambda *args: seen.append(args), timeout=timeout)
        self.assertTrue(wait(lambda:bool(seen)), path)
        self.assertFalse(self.client.pending)
        self.assertFalse(self.client._deadlines)
        return seen[0]

    def test_invalid_and_empty_success_responses_are_errors(self):
        for path in ["/empty", "/invalid", "/array"]:
            code, _data, error, _network = self.request(path)
            self.assertEqual(code, 0, path)
            self.assertIn("JSON", error)
        code, data, error, _network = self.request("/ok")
        self.assertEqual(code, 200)
        self.assertEqual(data, {"value":"ok"})
        self.assertEqual(error, "")

    def test_plain_and_gzip_decoded_body_limits_abort_instead_of_success(self):
        for path in ["/huge", "/gzip"]:
            code, data, error, _network = self.request(path)
            self.assertEqual(code, 0, path)
            self.assertEqual(data, {})
            self.assertIn("size limit", error)

    def test_absolute_deadline_limits_a_continuous_slow_drip(self):
        start = time.monotonic()
        code, _data, error, _network = self.request("/drip", timeout=180)
        self.assertEqual(code, 0)
        self.assertIn("timed out", error)
        self.assertLess(time.monotonic()-start, 1.5)

    def test_cancellation_stops_timers_and_drops_stale_callbacks(self):
        stale = []
        self.client.request("old", "/drip", lambda *args:stale.append(args), timeout=400)
        QTest.qWait(50)
        self.client.cancel()
        self.assertFalse(self.client._deadlines)
        code, _data, _error, _network = self.request("/ok")
        self.assertEqual(code, 200)
        QTest.qWait(450)
        APP.processEvents()
        self.assertEqual(stale, [])

    def test_local_credentials_never_follow_an_application_proxy(self):
        from PySide6.QtNetwork import QNetworkProxy
        previous = QNetworkProxy.applicationProxy()
        QNetworkProxy.setApplicationProxy(QNetworkProxy(QNetworkProxy.ProxyType.HttpProxy, "127.0.0.1", 1))
        try:
            self.client.key = "local-only-test-key"
            code, data, error, _network = self.request("/ok")
            self.assertEqual(code, 200, error)
            self.assertEqual(data, {"value":"ok"})
        finally:
            QNetworkProxy.setApplicationProxy(previous)
