#!/usr/bin/env python3
"""Validate generated archives by extraction; never install packages on the host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, capture_output=True, text=True, **kwargs)


def gui_smoke(root: Path, name: str, output: Path):
    appdir = root / "usr/share/nexus-desktop"
    runtime = output / f"{name}-runtime"
    runtime.mkdir(mode=0o700, exist_ok=True)
    runtime.chmod(0o700)
    env = {**os.environ, "NEXUS_DESKTOP_APP_DIR": str(appdir), "PYTHONPATH": "", "QT_QPA_PLATFORM": "offscreen", "XDG_CONFIG_HOME": str(output / f"{name}-config"), "XDG_RUNTIME_DIR": str(runtime)}
    result = {"kind": name, "entrypoints": {}}
    for argument in ["--help", "--version"]:
        result["entrypoints"][argument] = run([root / "usr/bin/nexus-desktop", argument], env=env, timeout=10).stdout.strip()
    # Native Qt window lifecycle, without launching a model or an external server.
    screenshot = output / f"{name}.png"
    code = (
        "from PySide6 import __version__;from PySide6.QtCore import QTimer,QEvent;"
        "from PySide6.QtWidgets import QApplication;from PySide6.QtGui import QIcon;from pathlib import Path;import nexus_desktop;"
        "from nexus_desktop.window import MainWindow;from nexus_desktop.config import Settings;"
        "app=QApplication(['nexus-package-smoke']);assert not QIcon(str(Path(nexus_desktop.__file__).parent/'assets/nexus.svg')).pixmap(40,40).isNull();w=MainWindow(Settings());w.show();"
        "app.processEvents();assert w.isVisible();assert w.grab().save(" + repr(str(screenshot)) + ");"
        "QTimer.singleShot(10,w.close);QTimer.singleShot(100,app.quit);app.exec();w.deleteLater();app.sendPostedEvents(None,QEvent.Type.DeferredDelete);print('QT_WINDOW_OK',__version__)"
    )
    env["PYTHONPATH"] = str(appdir)
    window = run(["/usr/bin/python3", "-P", "-c", code], env=env, timeout=20)
    result.update({"window": window.stdout.strip(), "stderr": window.stderr, "screenshot": str(screenshot)})
    # Exercise the actual packaged -m entrypoint while Qt is idle. A timer must
    # let Python handle SIGTERM; never leave a hung GUI behind on failure.
    process = subprocess.Popen([str(root / "usr/bin/nexus-desktop")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(1)
        if process.poll() is not None:
            raise ValueError("Packaged GUI exited before idle shutdown test")
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        if process.returncode != 0:
            raise ValueError(f"Packaged GUI SIGTERM exit {process.returncode}: {stderr}")
        result["idle_sigterm"] = "graceful exit 0"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    # An architecture-independent GUI archive must not silently carry native ELF.
    elf = [str(p.relative_to(root)) for p in (root / "usr").rglob("*") if p.is_file() and p.read_bytes()[:4] == b"\x7fELF"]
    if elf:
        raise ValueError(f"Native ELF found in GUI package: {elf}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.artifacts / "manifest.json").read_text())
    results = []
    with tempfile.TemporaryDirectory(prefix="nexus-package-extract-") as temporary:
        for artifact in manifest["artifacts"]:
            archive = args.artifacts / artifact["path"]
            assert hashlib.sha256(archive.read_bytes()).hexdigest() == artifact["sha256"]
            root = Path(temporary) / artifact["kind"]
            root.mkdir()
            if artifact["kind"] == "deb-gui":
                run(["dpkg-deb", "-x", archive, root])
            else:
                run(["bsdtar", "-xf", archive, "-C", root])
            if artifact["kind"].endswith("gui"):
                results.append(gui_smoke(root, artifact["kind"], args.output))
            else:
                for binary, metadata in manifest["engine"]["elf"].items():
                    path = root / "usr/bin" / binary
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
                    run([path, "--help"], timeout=10)
                results.append({"kind": artifact["kind"], "entrypoints": "all --help passed", "binary_hashes": "matched manifest"})
    (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
