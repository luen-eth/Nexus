import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QListWidgetItem
from nexus_desktop.hub import (
    Cancelled, HubClient, HubError, HubFile, ModelHubPanel,
    _allowed_url, inspect_gguf, list_local_models,
)

APP = QApplication.instance() or QApplication(["nexus-hub-tests"])
APP.setQuitOnLastWindowClosed(False)
REVISION = "a" * 40


def gguf(*, split=1, padding=0):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    metadata += string("split.count") + struct.pack("<II", 4, split)
    return struct.pack("<4sIQQ", b"GGUF", 3, 0, 2) + metadata + b"x" * padding


def wait(predicate, seconds=3):
    end = time.monotonic() + seconds
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
        self.server.requests.append((self.path, dict(self.headers)))
        path = urlsplit(self.path).path
        if path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "https://example.com/model.gguf")
            self.end_headers()
            return
        if path == "/api/models":
            body = json.dumps([{"id":"owner/repository", "downloads":123}, {"id":"../bad"}]).encode()
        elif path.endswith("/revision/main"):
            body = json.dumps({"sha":REVISION,"gated":self.server.gated,"gguf":{"architecture":"llama"},"siblings":[{
                "rfilename":"nested/model.gguf", "size":len(self.server.payload),
                "lfs":{"size":len(self.server.payload),"sha256":hashlib.sha256(self.server.payload).hexdigest()}
            }]}).encode()
        else:
            body = self.server.payload
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            if self.server.stall and "/resolve/" in path:
                time.sleep(.5)
            if self.server.slow and "/resolve/" in path:
                for start in range(0, len(body), 1024):
                    self.wfile.write(body[start:start+1024])
                    self.wfile.flush()
                    time.sleep(.01)
            else:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


class HubTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "models"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.payload = gguf()
        self.server.requests = []
        self.server.gated = False
        self.server.slow = False
        self.server.stall = False
        self.thread = threading.Thread(target=lambda:self.server.serve_forever(poll_interval=.02),daemon=True)
        self.thread.start()
        self.client = HubClient(f"http://127.0.0.1:{self.server.server_port}", timeout=.3, allow_local_http=True)
        self.cancel = threading.Event()
        self.panels = []

    def tearDown(self):
        for panel in self.panels:
            panel.shutdown()
            panel.deleteLater()
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)
        self.temp.cleanup()

    def file(self, **changes):
        file = HubFile("owner/repository", REVISION, "nested/model.gguf", len(self.server.payload), hashlib.sha256(self.server.payload).hexdigest(), "llama")
        return replace(file, **changes)

    def downloaded_files(self):
        return list(self.root.rglob("*.gguf")) if self.root.exists() else []

    def test_search_and_pinned_file_metadata_never_use_local_credentials(self):
        with patch.dict(os.environ, {"NEXUS_API_KEY":"must-not-leak", "HF_TOKEN":"also-not-used"}):
            found = self.client.search("small model", self.cancel)
            files = self.client.files(found[0]["id"], self.cancel)
            path = self.client.download(files[0], self.root, self.cancel, lambda *_:None)
        self.assertEqual(found, [{"id":"owner/repository", "downloads":123}])
        self.assertEqual(files[0].revision, REVISION)
        self.assertEqual(path.read_bytes(), self.server.payload)
        self.assertIn("/owner/repository/resolve/"+REVISION+"/nested/model.gguf", [p for p,_ in self.server.requests])
        self.assertIn("filter=gguf", self.server.requests[0][0])
        for _, headers in self.server.requests:
            self.assertNotIn("Authorization", headers)
            self.assertNotIn("Cookie", headers)
        self.assertEqual(inspect_gguf(path)["general.architecture"], "llama")
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_existing_file_is_never_overwritten(self):
        file = self.file()
        path = self.client.download(file, self.root, self.cancel, lambda *_:None)
        count = len(self.server.requests)
        with self.assertRaises(HubError):
            self.client.download(file, self.root, self.cancel, lambda *_:None)
        self.assertEqual(path.read_bytes(), self.server.payload)
        self.assertEqual(len(self.server.requests), count)

    def test_deep_nested_download_remains_visible_in_local_library(self):
        name = "/".join(["nested"] * 15 + ["model.gguf"])
        path = self.client.download(self.file(name=name), self.root, self.cancel, lambda *_:None)
        self.assertIn(path, list_local_models(self.root))

    def test_bad_sha_magic_size_and_shards_leave_no_published_file(self):
        cases = [self.file(sha256="0"*64), self.file(size=len(self.server.payload)+1),
                 self.file(name="model-00001-of-00002.gguf")]
        for file in cases:
            with self.subTest(file=file), self.assertRaises(HubError):
                self.client.download(file, self.root, self.cancel, lambda *_:None)
        for payload in [b"HTML" + self.server.payload[4:], gguf(split=2)]:
            self.server.payload = payload
            with self.assertRaises(HubError):
                self.client.download(self.file(), self.root, self.cancel, lambda *_:None)
        self.assertEqual(self.downloaded_files(), [])
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_cancel_removes_partial_data(self):
        self.server.payload = gguf(padding=128*1024)
        self.server.slow = True
        def progress(done, _total):
            if done:
                self.cancel.set()
        with self.assertRaises(Cancelled):
            self.client.download(self.file(), self.root, self.cancel, progress)
        self.assertEqual(self.downloaded_files(), [])
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_paths_symlinks_and_disk_capacity_are_checked_before_network(self):
        for name in ["../escape.gguf", "/escape.gguf", "a/../../escape.gguf", "a\\b.gguf"]:
            with self.subTest(name=name), self.assertRaises(HubError):
                self.client.download(self.file(name=name), self.root, self.cancel, lambda *_:None)
        self.root.mkdir()
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (self.root / "owner").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(HubError):
            self.client.download(self.file(), self.root, self.cancel, lambda *_:None)
        self.assertEqual(list(outside.iterdir()), [])
        (self.root / "owner").unlink()
        with patch("nexus_desktop.hub.os.fstatvfs",return_value=SimpleNamespace(f_bavail=0,f_frsize=4096)):
            with self.assertRaises(HubError):
                self.client.download(self.file(), self.root, self.cancel, lambda *_:None)
        self.assertEqual(self.server.requests, [])

    def test_redirect_allowlist_metadata_limit_and_gated_models(self):
        with self.assertRaises(HubError):
            self.client._open(self.client.base_url + "/redirect")
        for url in ["http://huggingface.co/file", "https://huggingface.co.evil.test/file", "https://user:pass@huggingface.co/file", "https://huggingface.co:444/file"]:
            with self.subTest(url=url), self.assertRaises(HubError):
                _allowed_url(url)
        for url in ["https://huggingface.co/file", "https://cas-bridge.xethub.hf.co/file", "https://us.aws.cdn.hf.co/file"]:
            _allowed_url(url)
        self.client.MAX_METADATA_BYTES = 10
        with self.assertRaises(HubError):
            self.client.search("small", self.cancel)
        self.client.MAX_METADATA_BYTES = 4*1024*1024
        self.server.gated = "auto"
        with self.assertRaises(HubError):
            self.client.files("owner/repository", self.cancel)

    def panel(self):
        panel = ModelHubPanel(None, str(self.root))
        panel.client = self.client
        self.panels.append(panel)
        self.assertTrue(wait(lambda:"local" not in panel._jobs))
        return panel

    def test_panel_async_search_download_library_and_load_signal(self):
        panel = self.panel()
        self.assertEqual(self.server.requests, [])  # Startup never searches/downloads.
        panel.search_input.setText("small")
        panel.search()
        self.assertTrue(wait(lambda:panel.results.count()==1))
        panel.results.setCurrentRow(0)
        self.assertTrue(wait(lambda:panel.file_list.count()==1))
        panel.file_list.setCurrentRow(0)
        self.assertTrue(panel.download_button.isEnabled())
        panel.start_download()
        self.assertTrue(wait(lambda:panel.local_files.count()==1))
        seen=[]
        panel.model_selected.connect(seen.append)
        panel._load_selected()
        self.assertTrue(wait(lambda:bool(seen)))
        self.assertEqual(Path(seen[0]).read_bytes(), self.server.payload)
        self.assertIn("Download verified", panel.progress.format())

    def test_progress_larger_than_two_gib_and_close_during_download(self):
        panel = self.panel()
        panel._jobs["download"]=(123,threading.Event())
        panel._signals.progress.emit(123,3*1024**3,6*1024**3)
        APP.processEvents()
        self.assertEqual(panel.progress.value(),500)
        panel._jobs.pop("download")
        self.server.payload=gguf(padding=256*1024)
        self.server.slow=True
        item=QListWidgetItem("model.gguf")
        item.setData(Qt.ItemDataRole.UserRole,self.file())
        panel.file_list.addItem(item)
        panel.file_list.setCurrentItem(item)
        panel.start_download()
        self.assertTrue(wait(lambda:any(self.root.rglob("*.part"))))
        panel.shutdown()
        self.assertTrue(wait(lambda:not any(self.root.rglob("*.part"))))
        self.assertEqual(self.downloaded_files(),[])

    def test_local_library_ignores_symlinks_and_split_load_is_blocked(self):
        self.root.mkdir()
        (self.root/"model.gguf").write_bytes(gguf())
        (self.root/"alias.gguf").symlink_to(self.root/"model.gguf")
        split=self.root/"renamed.gguf"
        split.write_bytes(gguf(split=2))
        self.assertEqual(len(list_local_models(self.root)),2)
        panel=self.panel()
        seen=[]
        panel.model_selected.connect(seen.append)
        for index in range(panel.local_files.count()):
            if panel.local_files.item(index).data(Qt.ItemDataRole.UserRole)==str(split):
                panel.local_files.setCurrentRow(index)
        panel._load_selected()
        self.assertTrue(wait(lambda:"loadcheck" not in panel._jobs))
        self.assertEqual(seen,[])
        self.assertIn("Split GGUF",panel.status.text())

    def test_destroy_during_stalled_read_cleans_partial_after_socket_timeout(self):
        panel = self.panel()
        self.server.stall = True
        self.client.timeout = .1
        item = QListWidgetItem("model.gguf")
        item.setData(Qt.ItemDataRole.UserRole, self.file())
        panel.file_list.addItem(item)
        panel.file_list.setCurrentItem(item)
        panel.start_download()
        self.assertTrue(wait(lambda:any(self.root.rglob("*.part"))))
        panel.shutdown()
        panel.deleteLater()
        self.panels.remove(panel)
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertTrue(wait(lambda:not any(self.root.rglob("*.part")), seconds=1.5))
        self.assertEqual(self.downloaded_files(), [])


if __name__ == "__main__":
    unittest.main()
