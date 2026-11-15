import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import socket
import tempfile
import time
import unittest
from PySide6.QtCore import QProcess, QTimer, QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from nexus_desktop.config import Settings
from nexus_desktop.controller import ServerController
from nexus_desktop.window import MainWindow

APP = QApplication.instance() or QApplication(["nexus-desktop-tests"])
APP.setQuitOnLastWindowClosed(False)
APP.setStyle("Fusion")
FAKE = str(Path(__file__).with_name("fake_server.py"))


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def until(predicate, seconds=6):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        APP.processEvents()
        if predicate():
            return True
        QTest.qWait(15)
    return bool(predicate())


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.old_xdg = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = self.directory.name
        self.controller = ServerController()
        self.settings = Settings(binary=FAKE, port=free_port(), context=128)
        self.children = []
        self.windows = []

    def tearDown(self):
        for controller in [self.controller] + self.children:
            controller._restart = None
            controller.stop()
            until(lambda: not controller.owned)
            controller.disconnect()
            controller.deleteLater()
        for window in self.windows:
            window.close()
            window.deleteLater()
        # processEvents alone does not dispatch deferred deletes without a full
        # exec loop. Destroy QWidget trees while QApplication is still alive.
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        self.directory.cleanup()
        if self.old_xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self.old_xdg

    def start(self):
        self.controller.start(self.settings)
        self.assertTrue(until(lambda: bool(self.controller.health)), self.controller.detail)
        self.assertTrue(self.controller.owned)
        return self.controller

    def test_owned_lifecycle_keeps_key_out_of_arguments_logs_and_settings(self):
        controller = self.start()
        secret = controller.api.key
        self.assertGreater(len(secret), 32)
        self.assertNotIn(secret, " ".join(controller.process.arguments()))
        self.assertEqual(controller.process.processEnvironment().value("NEXUS_API_KEY"), secret)
        controller.log("Bearer " + secret)
        controller.log("NEXUS_API_KEY=" + secret)
        self.assertNotIn(secret, "\n".join(controller.logs))
        self.assertTrue(until(lambda: bool(controller.metrics)))
        first_pid = controller.process.processId()
        controller.restart(self.settings)
        self.assertTrue(until(lambda: controller.owned and controller.process.processId() != first_pid and bool(controller.health)))
        self.assertNotEqual(controller.api.key, secret)
        controller.stop()
        self.assertTrue(until(lambda: not controller.owned))

    def test_external_connection_cannot_stop_a_foreign_process(self):
        owner = self.start()
        other = ServerController()
        self.children.append(other)
        other.connect_external(owner.api.base, owner.api.key)
        self.assertTrue(until(lambda: bool(other.metrics)))
        self.assertEqual(other.mode, "external")
        self.assertFalse(other.owned)
        other.stop()
        other.disconnect()
        self.assertTrue(owner.owned)
        owner.poll()
        self.assertTrue(until(lambda: bool(owner.health)))

    def test_local_start_cannot_replace_an_external_connection(self):
        owner = self.start()
        client = ServerController()
        self.children.append(client)
        window = MainWindow(Settings(port=free_port()), client)
        self.windows.append(window)
        window.show()
        client.connect_external(owner.api.base, owner.api.key)
        self.assertTrue(until(lambda: bool(client.metrics)))
        self.assertFalse(window.start_button.isEnabled())
        owner_pid = owner.process.processId()
        health, key, base = dict(client.health), client.api.key, client.api.base
        errors = []
        client.error.connect(errors.append)
        # Cover the reported missing-binary error and a valid executable that
        # would otherwise disconnect this client and begin a new port probe.
        for binary in ["/does/not/exist/nexus-server", FAKE]:
            with self.subTest(binary=binary):
                window.binary.setText(binary)
                window.start_server()  # Exercise a programmatic click as well.
                self.assertEqual(client.mode, "external")
                self.assertIsNone(client.process)
                self.assertFalse(client.probing)
                self.assertEqual(client.health, health)
                self.assertEqual(client.api.key, key)
                self.assertEqual(client.api.base, base)
                self.assertTrue(client.timer.isActive())
                self.assertEqual(errors, [])
                self.assertEqual(list(client.logs), [])
        self.assertTrue(owner.owned)
        self.assertEqual(owner.process.processId(), owner_pid)
        client.disconnect()
        self.assertTrue(window.start_button.isEnabled())

    def test_occupied_port_is_not_taken_over(self):
        owner = self.start()
        other = ServerController()
        self.children.append(other)
        other.start(self.settings)
        self.assertTrue(until(lambda: not other.probing))
        self.assertFalse(other.owned)
        self.assertIn("port", other.detail)
        self.assertTrue(owner.owned)

    def test_model_actions_and_bounded_logs(self):
        controller = self.start()
        model = Path(self.directory.name) / "fixture.gguf"
        model.write_bytes(b"fixture")
        self.settings.model = str(model)
        for action, expected in [("load", True), ("unload", False), ("reload", True)]:
            controller.model_action(action, self.settings)
            self.assertTrue(until(lambda: not controller.busy))
            controller.poll()
            self.assertTrue(until(lambda: controller.health.get("model_loaded") is expected))
        controller.log("\n".join("x"*10000 for _ in range(1200)))
        self.assertEqual(len(controller.logs), controller.MAX_LOG_LINES)
        self.assertLess(max(map(len, controller.logs)), 1250)

    def test_window_close_waits_for_its_process_and_remains_responsive(self):
        window = MainWindow(self.settings, self.controller)
        self.windows.append(window)
        window.show()
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(20)
        window.start_server()
        self.assertTrue(until(lambda: bool(self.controller.health)))
        self.assertGreater(len(ticks), 0)
        self.assertTrue(window.stop_button.isEnabled())
        window.copy_key()
        self.assertEqual(APP.clipboard().text(), self.controller.api.key)
        APP.clipboard().clear()
        window.close()
        self.assertTrue(until(lambda: not self.controller.owned and not window.isVisible()))
        timer.stop()
        timer.deleteLater()
        self.assertTrue((Path(self.directory.name)/"nexus-desktop/settings.json").is_file())

    def test_missing_binary_and_unreachable_server_are_clear_errors(self):
        self.settings.binary = "/does/not/exist/nexus-server"
        self.controller.start(self.settings)
        self.assertFalse(self.controller.owned)
        self.assertIn("was not found", self.controller.detail)
        self.controller.connect_external(f"http://127.0.0.1:{free_port()}")
        self.assertTrue(until(lambda: self.controller.state == "No connection"))

    def test_stale_network_callbacks_are_cancelled_on_disconnect(self):
        owner = self.start()
        client = ServerController()
        self.children.append(client)
        client.connect_external(owner.api.base, owner.api.key)
        client.disconnect()
        QTest.qWait(100)
        APP.processEvents()
        self.assertEqual(client.mode, "disconnected")
        self.assertEqual(client.health, {})
        self.assertEqual(client.metrics, {})

    def test_failed_child_and_cancelled_start_probe_do_not_leave_a_process(self):
        self.settings.binary = "/usr/bin/false"
        self.controller.start(self.settings)
        self.assertTrue(until(lambda: self.controller.state == "Server error"))
        self.assertFalse(self.controller.owned)
        self.assertIn("exit code", self.controller.detail)
        self.settings.binary = FAKE
        window = MainWindow(self.settings, self.controller)
        self.windows.append(window)
        window.show()
        window.start_server()
        self.assertTrue(self.controller.probing)
        window.close()
        QTest.qWait(100)
        APP.processEvents()
        self.assertFalse(self.controller.probing)
        self.assertFalse(self.controller.owned)
        self.assertFalse(window.isVisible())
