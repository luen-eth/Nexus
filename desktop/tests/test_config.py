import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from nexus_desktop.config import Settings, load_settings, local_url, save_settings


class ConfigTests(unittest.TestCase):
    def test_settings_are_atomic_private_and_never_persist_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nexus" / "settings.json"
            settings = Settings(model="/tmp/Türkçe model.gguf", port=19001)
            settings.api_key = "must-never-be-written"
            save_settings(settings, path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertNotIn("must-never", path.read_text())
            self.assertNotIn("api_key", json.loads(path.read_text()))
            self.assertEqual(load_settings(path).model, settings.model)
            settings.port = 19002
            save_settings(settings, path)
            self.assertEqual(load_settings(path).port, 19002)
            self.assertEqual(list(path.parent.glob(".settings-*")), [])

    def test_malformed_unknown_or_huge_settings_do_not_break_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            for contents in ["{ broken", "null", '[1,2]', '{"port":"wrong","backend":"fake","context":-1,"api_key":"secret"}', ' '*65537]:
                path.write_text(contents)
                result = load_settings(path)
                self.assertIsInstance(result, Settings)
                self.assertGreater(result.port, 0)
                self.assertGreater(result.context, 0)
                self.assertEqual(result.backend, "cpu")

    def test_only_local_http_management_urls_are_accepted(self):
        self.assertEqual(local_url("http://localhost:8000/"), "http://localhost:8000")
        self.assertEqual(local_url("http://[::1]:8081"), "http://[::1]:8081")
        for url in ["https://example.com", "http://127.0.0.1.evil.test:8080", "http://example.com", "file:///tmp/x", "http://user:key@localhost:8080", "http://localhost:0", "http://localhost:65536", "http://localhost:8080/api", "http://localhost:8080?key=secret"]:
            with self.assertRaises(ValueError, msg=url):
                local_url(url)

    def test_help_and_version_work_without_a_display(self):
        env = dict(os.environ)
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        invalid = subprocess.run([sys.executable, "-m", "nexus_desktop", "--port", "0"], env=env, capture_output=True, timeout=5, text=True)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("port", invalid.stderr)
        for arg in ["--help", "--version"]:
            result = subprocess.run([sys.executable, "-m", "nexus_desktop", arg], env=env, capture_output=True, timeout=5, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Nexus", result.stdout)
