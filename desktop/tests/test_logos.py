"""Publisher logos use public metadata, bounded images and safe asynchronous UI updates."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PySide6.QtCore import QByteArray, QBuffer, QEvent, QIODevice, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nexus_desktop.hub import HubError, LogoCache, ModelHubPanel, PublisherLogoClient, _decode_logo


APP = QApplication.instance() or QApplication(["nexus-logo-tests"])
APP.setQuitOnLastWindowClosed(False)


def wait(predicate, seconds=4):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        APP.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    return bool(predicate())


def png(width=12, height=12):
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(0xff5f43dc)
    data = QByteArray()
    stream = QBuffer(data)
    stream.open(QIODevice.OpenModeFlag.WriteOnly)
    assert image.save(stream, "PNG")
    stream.close()
    return bytes(data)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, status, body, mime="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self.server.requests.append((self.path, dict(self.headers)))
        if self.path.startswith("/api/models?"):
            self.respond(200, json.dumps([{"id": "alice/model", "downloads": 12}]).encode())
            return
        if self.path.startswith("/api/models/alice/model/revision/main?"):
            self.respond(200, json.dumps({"sha": "a" * 40, "private": False, "gated": False,
                "siblings": [{"rfilename": "model.gguf", "size": 24,
                              "lfs": {"size": 24, "sha256": "b" * 64}}]}).encode())
            return
        if self.path.startswith(("/api/users/", "/api/organizations/")):
            author = self.path.split("/")[3]
            if author == "orglogo" and self.path.startswith("/api/users/"):
                self.respond(404, b"{}")
                return
            identity = "different-publisher" if author == "mismatch" else author
            profile = {"name": identity, "user": identity}
            if author != "missing":
                profile["avatarUrl"] = ("https://untrusted.invalid/avatar.png" if author == "untrusted"
                                        else self.server.base + "/avatars/" + author + ".png")
            self.respond(200, json.dumps(profile).encode())
            return
        if self.path.startswith("/avatars/"):
            author = self.path.rsplit("/", 1)[-1].removesuffix(".png")
            if author == "slow":
                self.server.started.set()
                self.server.release.wait(2)
            if author == "redirect":
                self.send_response(302)
                self.send_header("Location", "https://untrusted.invalid/avatar.png")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if author == "svg":
                self.respond(200, b'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12"/>', "image/svg+xml")
            elif author == "oversize":
                self.respond(200, self.server.image + b"x" * (1024 * 1024), "image/png")
            elif author == "pixels":
                self.respond(200, self.server.wide_image, "image/png")
            elif author == "invalid":
                self.respond(200, b"this is not an image", "image/png")
            else:
                self.respond(200, self.server.image, "image/png")
            return
        self.respond(404, b"{}")


class LogoTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.env = patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.root / "cache"),
                                          "NEXUS_API_KEY": "private-nexus-key", "HF_TOKEN": "private-hf-key"})
        self.env.start()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.base = f"http://127.0.0.1:{self.server.server_port}"
        self.server.requests = []
        self.server.image = png()
        self.server.wide_image = png(1025, 1)
        self.server.started = threading.Event()
        self.server.release = threading.Event()
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.02), daemon=True)
        self.thread.start()
        self.client = PublisherLogoClient(self.server.base, allow_local_http=True)
        self.caches = []
        self.panels = []

    def tearDown(self):
        for panel in self.panels:
            panel.shutdown()
        for cache in self.caches:
            cache.shutdown()
        self.server.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        for panel in self.panels:
            panel.deleteLater()
        for cache in self.caches:
            cache.deleteLater()
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        self.env.stop()
        self.directory.cleanup()

    def cache(self, directory=None):
        cache = LogoCache(directory or self.root / "icons")
        cache.client = self.client
        self.caches.append(cache)
        return cache

    def test_public_profile_provenance_and_org_fallback_never_send_credentials(self):
        for author in ["alice", "orglogo"]:
            data = self.client.avatar(author, threading.Event())
            self.assertIsNotNone(_decode_logo(data))
        paths = [path for path, _ in self.server.requests]
        self.assertIn("/api/users/orglogo/overview", paths)
        self.assertIn("/api/organizations/orglogo/overview", paths)
        for _path, headers in self.server.requests:
            lowered = {key.lower(): value for key, value in headers.items()}
            self.assertNotIn("authorization", lowered)
            self.assertNotIn("cookie", lowered)
            self.assertNotIn("private-nexus-key", str(headers))
            self.assertNotIn("private-hf-key", str(headers))
        before = len(self.server.requests)
        try:
            self.assertIsNone(self.client.avatar("mismatch", threading.Event()))
        except HubError:
            pass
        self.assertFalse(any(path.startswith("/avatars/") for path, _ in self.server.requests[before:]))

    def test_missing_unsafe_invalid_oversized_and_svg_images_keep_fallback(self):
        cache = self.cache()
        self.assertFalse(cache.fallback_icon.isNull())
        self.assertFalse(cache.fallback_icon.pixmap(32, 32).isNull())
        original_resolver = socket.getaddrinfo
        external_hosts = []
        def resolve(host, *args, **kwargs):
            if host != "127.0.0.1":
                external_hosts.append(host)
                raise OSError("External DNS is forbidden in this fixture")
            return original_resolver(host, *args, **kwargs)
        with patch("socket.getaddrinfo", side_effect=resolve):
            for author in ["missing", "untrusted", "redirect", "svg", "oversize", "pixels", "invalid"]:
                with self.subTest(author=author):
                    try:
                        data = self.client.avatar(author, threading.Event())
                    except HubError:
                        data = None
                    self.assertTrue(data is None or _decode_logo(data) is None)
                    self.assertFalse(cache.has_logo(author))
        self.assertEqual(external_hosts, [])
        self.assertIsNone(_decode_logo(b'<svg xmlns="http://www.w3.org/2000/svg"/>'))
        self.assertIsNone(_decode_logo(self.server.wide_image))

    def test_async_avatar_cache_is_reused_offline_and_corruption_is_rejected(self):
        cache = self.cache()
        initial = cache.icon("alice")
        self.assertFalse(initial.isNull())
        self.assertTrue(wait(lambda: cache.has_logo("alice")))
        self.assertTrue(wait(lambda: len(list((self.root / "icons").glob("*.png"))) == 1))
        files = list((self.root / "icons").glob("*.png"))
        self.assertEqual(len(files), 1)
        self.assertIsNotNone(_decode_logo(files[0].read_bytes()))
        cache.shutdown()

        class Offline:
            MAX_IMAGE_BYTES = PublisherLogoClient.MAX_IMAGE_BYTES
            def __init__(self):
                self.calls = 0
            def avatar(self, *_):
                self.calls += 1
                raise HubError("Offline fixture")

        offline = Offline()
        reused = self.cache()
        reused.client = offline
        self.assertFalse(reused.icon("alice").isNull())
        self.assertTrue(wait(lambda: reused.has_logo("alice")))
        self.assertEqual(offline.calls, 0)
        reused.shutdown()
        files[0].write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"/>')
        rejected = self.cache()
        rejected.client = offline
        self.assertFalse(rejected.icon("alice").isNull())
        self.assertTrue(wait(lambda: offline.calls > 0))
        self.assertFalse(rejected.has_logo("alice"))

    def test_publisher_roles_cover_remote_and_downloaded_models_unknown_imports_use_fallback(self):
        models = self.root / "models"
        known = models / "alice" / "model" / ("a" * 40) / "model.gguf"
        known.parent.mkdir(parents=True)
        known.write_bytes(b"GGUF fixture")
        (models / "imported.gguf").write_bytes(b"GGUF fixture")
        panel = ModelHubPanel(None, str(models))
        panel.logo_cache.client = self.client
        panel.client.base_url = self.server.base
        panel.client.allow_local_http = True
        # Use the same fixture opener so both original URLs and redirects remain local.
        panel.client.opener = self.client.opener
        self.panels.append(panel)
        self.assertTrue(wait(lambda: panel.local_files.count() == 2))
        panel.search_input.setText("model")
        panel.search()
        self.assertTrue(wait(lambda: panel.results.count() == 1 and panel.logo_cache.has_logo("alice")))
        remote = panel.results.item(0)
        self.assertEqual(remote.data(Qt.ItemDataRole.UserRole + 1), "alice")
        self.assertEqual(remote.data(Qt.ItemDataRole.UserRole + 2), "publisher")
        self.assertFalse(remote.icon().isNull())
        panel.results.setCurrentRow(0)
        self.assertTrue(wait(lambda: panel.file_list.count() == 1))
        selected_file = panel.file_list.item(0)
        self.assertEqual(selected_file.data(Qt.ItemDataRole.UserRole + 1), "alice")
        self.assertEqual(selected_file.data(Qt.ItemDataRole.UserRole + 2), "publisher")
        self.assertFalse(selected_file.icon().pixmap(32, 32).isNull())
        for index in range(panel.local_files.count()):
            item = panel.local_files.item(index)
            self.assertFalse(item.icon().pixmap(32, 32).isNull())
            expected = "publisher" if item.data(Qt.ItemDataRole.UserRole) == str(known) else "fallback"
            self.assertEqual(item.data(Qt.ItemDataRole.UserRole + 2), expected)
        self.assertFalse(any("imported" in path for path, _ in self.server.requests))

    def test_clearing_lists_and_shutdown_during_avatar_fetch_drop_stale_updates(self):
        models = self.root / "models"
        known = models / "slow" / "model" / ("a" * 40) / "model.gguf"
        known.parent.mkdir(parents=True)
        known.write_bytes(b"GGUF fixture")
        panel = ModelHubPanel(None, str(models))
        panel.logo_cache.client = self.client
        self.panels.append(panel)
        self.assertTrue(wait(self.server.started.is_set))
        errors = io.StringIO()
        with redirect_stderr(errors):
            panel.local_files.clear()
            panel.results.clear()
            panel.file_list.clear()
            self.server.release.set()
            self.assertTrue(wait(lambda: panel.logo_cache.has_logo("slow")))
        self.assertEqual(panel.local_files.count(), 0)
        self.assertEqual(panel.results.count(), 0)
        self.assertNotIn("Traceback", errors.getvalue())
        self.assertNotIn("Internal C++ object", errors.getvalue())
        # Exercise shutdown separately, with another fetch still blocked.
        self.server.started.clear()
        self.server.release.clear()
        cache = self.cache(self.root / "shutdown-icons")
        updated = []
        cache.updated.connect(updated.append)
        cache.icon("slow")
        self.assertTrue(wait(self.server.started.is_set))
        cache.shutdown()
        self.server.release.set()
        QTest.qWait(100)
        APP.processEvents()
        self.assertEqual(updated, [])
        self.assertFalse(cache.has_logo("slow"))


if __name__ == "__main__":
    unittest.main()
