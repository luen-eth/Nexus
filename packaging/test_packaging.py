"""Lightweight packaging contract checks; no system installation or native build."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("linux_packages", ROOT / "scripts/build-linux-packages.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PackagingTests(unittest.TestCase):
    def test_launcher_ignores_same_named_cwd_module_or_package(self):
        for package in [False, True]:
            with self.subTest(package=package), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                trusted = root / "trusted/nexus_desktop"
                trusted.mkdir(parents=True)
                (trusted / "__init__.py").write_text("")
                (trusted / "__main__.py").write_text("import sys;print('TRUSTED',repr(sys.argv[1:]))")
                cwd = root / "untrusted"
                cwd.mkdir()
                marker = root / "cwd-executed"
                source = f"from pathlib import Path;Path({str(marker)!r}).write_text('wrong');print('UNTRUSTED')"
                if package:
                    untrusted = cwd / "nexus_desktop"
                    untrusted.mkdir()
                    (untrusted / "__init__.py").write_text(source)
                    (untrusted / "__main__.py").write_text(source)
                else:
                    (cwd / "nexus_desktop.py").write_text(source)
                environment = {**os.environ, "NEXUS_DESKTOP_APP_DIR": str(trusted.parent), "PYTHONPATH": ""}
                result = subprocess.run([ROOT / "packaging/launcher.sh", "--probe", "space and Türkçe"], env=environment, cwd=cwd, capture_output=True, text=True, check=True)
                self.assertFalse(marker.exists())
                self.assertEqual(result.stdout.strip(), "TRUSTED ['--probe', 'space and Türkçe']")

    @unittest.skipUnless((ROOT / "target/release/nexus").is_file(), "native release is optional for GUI-only packaging")
    def test_elf_dependencies_survive_translated_inherited_locale(self):
        for language in ["tr", "fr"]:
            with self.subTest(language=language), mock.patch.dict(os.environ, {"LANGUAGE": language, "LC_ALL": "en_US.UTF-8"}):
                elf = builder.inspect_elf(ROOT / "target/release/nexus")
                self.assertIn("libc.so.6", elf["needed"])
                self.assertIn("libstdc++.so.6", elf["needed"])
                self.assertIsNotNone(elf["required_versions"]["GLIBC"])
                self.assertIsNotNone(elf["required_versions"]["GLIBCXX"])

    def test_supplied_subprocess_environment_cannot_translate_metadata(self):
        result = builder.command(["/usr/bin/python3", "-c", "import os;print(os.environ['LC_ALL'],os.environ['LANGUAGE'])"], env={**os.environ, "LC_ALL":"en_US.UTF-8", "LANGUAGE":"tr"}, capture_output=True)
        self.assertEqual(result.stdout.strip(), "C C")

    def test_payload_archive_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "rootfs/usr/bin"
            stage.mkdir(parents=True)
            script = stage / "example"
            script.write_text("#!/bin/sh\nexit 0\n")
            script.chmod(0o755)
            a, b = root / "a.tar.gz", root / "b.tar.gz"
            builder.archive(root / "rootfs", a, 1794700800)
            os.utime(script, None)
            builder.archive(root / "rootfs", b, 1794700800)
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_unexpanded_control_placeholder_is_rejected(self):
        with self.assertRaises(ValueError):
            builder.render(ROOT / "packaging/debian/control.in", {})


if __name__ == "__main__":
    unittest.main()
