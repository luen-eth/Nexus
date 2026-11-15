#!/usr/bin/env python3
"""Opt-in real-server Qt lifecycle test; no downloads and no GUI-thread waits.
QT_QPA_PLATFORM=offscreen PYTHONPATH=desktop python3 desktop/tests/real_server_smoke.py \
  --server target/release/nexus-server --model /path/to/model-f32.gguf --screenshot /tmp/nexus-desktop.png
"""
import argparse
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import socket
import tempfile
import time
from PySide6.QtCore import QTimer, QEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from nexus_desktop.config import Settings
from nexus_desktop.window import MainWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--screenshot", required=True)
    args = parser.parse_args()
    application = QApplication([])
    application.setStyle("Fusion")
    application.setQuitOnLastWindowClosed(False)
    with tempfile.TemporaryDirectory(prefix="nexus-desktop-real-") as temp:
        os.environ["XDG_CONFIG_HOME"] = temp
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",0))
            port = sock.getsockname()[1]
        settings = Settings(binary=str(Path(args.server).resolve()), context=128, port=port, backend="cpu", kv_quant="none", model="")
        window = MainWindow(settings)
        window.show()
        controller = window.controller
        ticks = []
        heartbeat = QTimer()
        heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
        heartbeat.start(25)
        def wait(predicate, timeout=90):
            deadline = time.monotonic()+timeout
            while time.monotonic() < deadline:
                application.processEvents()
                if predicate():
                    return
                QTest.qWait(15)
            raise AssertionError(f"timeout: {controller.state}: {controller.detail}")
        try:
            window.start_server()
            wait(lambda: bool(controller.health), 15)
            assert controller.owned and not controller.health["model_loaded"]
            window.model.setText(str(Path(args.model).resolve()))
            controller.model_action("load", window.settings())
            wait(lambda: not controller.busy and controller.health.get("model_loaded"))
            wait(lambda: bool(controller.metrics.get("engine",{}).get("host_weight_payload_bytes")),15)
            wait(lambda: bool(controller.models),10)
            window.refresh()
            application.processEvents()
            Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
            assert window.grab().save(args.screenshot)
            # Unload intentionally clears the server's remembered path. Reload
            # is checked while loaded; a later load uses the selected local file.
            for action, loaded in [("reload", True), ("unload", False), ("load", True)]:
                controller.model_action(action, window.settings())
                wait(lambda: not controller.busy and controller.health.get("model_loaded") is loaded)
            first_pid = controller.process.processId()
            restart_settings = window.settings()
            restart_settings.model = ""  # Avoid an unrelated startup reload in the process test.
            controller.restart(restart_settings)
            wait(lambda: controller.owned and controller.process.processId()!=first_pid and bool(controller.health),15)
            assert len(ticks)>4, "Qt event loop did not remain responsive"
            window.close()
            wait(lambda: not controller.owned and not window.isVisible(),10)
            print("PASS: start / health / models / metrics / load / unload / reload / restart / close")
            print(f"Qt heartbeat callbacks: {len(ticks)}; screenshot: {args.screenshot}")
        finally:
            controller._restart = None
            controller.stop()
            wait(lambda: not controller.owned,10)
            controller.disconnect()
            heartbeat.stop()
            heartbeat.deleteLater()
            window.close()
            window.deleteLater()
            application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            application.processEvents()


if __name__ == "__main__":
    main()
