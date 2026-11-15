"""Trash-only model removal and known-incompatible architecture regressions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QListWidgetItem, QMessageBox
from nexus_desktop.controller import ServerController
from nexus_desktop.hub import (
    HubClient, HubError, HubFile, ModelHubPanel, _delete_file_identity, _trash_model,
)

APP = QApplication.instance() or QApplication(["nexus-delete-tests"])
APP.setQuitOnLastWindowClosed(False)


def wait(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        APP.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    return bool(predicate())


def gguf(architecture="llama"):
    def string(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw
    return (struct.pack("<4sIQQ", b"GGUF", 3, 0, 1)
            + string("general.architecture") + struct.pack("<I", 8) + string(architecture))


class DeleteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "models"
        self.root.mkdir()
        self.model = self.root / "fixture.gguf"
        self.model.write_bytes(gguf())
        self.controller = ServerController()
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.base / "cache")}):
            self.panel = ModelHubPanel(self.controller, str(self.root))
        self.assertTrue(wait(lambda: "local" not in self.panel._jobs))
        self.select(self.model)
        self.deleted = []
        self.panel.model_deleted.connect(self.deleted.append)

    def tearDown(self):
        self.panel.shutdown()
        self.panel.deleteLater()
        self.controller.deleteLater()
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        self.temporary.cleanup()

    def select(self, path):
        for index in range(self.panel.local_files.count()):
            if self.panel.local_files.item(index).data(Qt.ItemDataRole.UserRole) == str(path):
                self.panel.local_files.setCurrentRow(index)
                return
        item = QListWidgetItem(path.name)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        self.panel.local_files.addItem(item)
        self.panel.local_files.setCurrentItem(item)

    def test_cancel_confirmation_is_default_and_leaves_file_unchanged(self):
        observed = []
        def cancel_dialog():
            dialog = APP.activeModalWidget()
            if not isinstance(dialog, QMessageBox):
                QTimer.singleShot(10, cancel_dialog)
                return
            observed.append({"default": dialog.standardButton(dialog.defaultButton()),
                             "text": dialog.text() + dialog.informativeText(),
                             "plain": dialog.textFormat(), "busy": self.panel.deletion_in_progress})
            dialog.reject()
        QTimer.singleShot(0, cancel_dialog)
        with patch("nexus_desktop.hub.QFile.moveToTrash") as trash:
            self.panel._delete_selected()
            trash.assert_not_called()
        self.assertEqual(observed[0]["default"], QMessageBox.StandardButton.Cancel)
        self.assertEqual(observed[0]["plain"], Qt.TextFormat.PlainText)
        self.assertTrue(observed[0]["busy"])
        self.assertIn(str(self.model), observed[0]["text"])
        self.assertIn("Size:", observed[0]["text"])
        self.assertIn("empty Trash", observed[0]["text"])
        self.assertTrue(self.model.exists())
        self.assertFalse(self.panel.deletion_in_progress)
        self.assertEqual(self.deleted, [])

    def test_loaded_unknown_path_and_busy_states_are_rejected_before_confirmation(self):
        self.controller.mode = "external"
        self.controller.health = {"model_loaded": True}
        self.controller.model_info = {"loaded": True, "path": str(self.model)}
        with patch.object(self.panel, "_confirm_delete") as confirm, patch("nexus_desktop.hub.QFile.moveToTrash") as trash:
            self.controller.changed.emit()
            self.assertFalse(self.panel.delete_button.isEnabled())
            self.panel._delete_selected()
            self.assertIn("Unload", self.panel.status.text())
            self.controller.model_info["path"] = None
            self.panel._delete_selected()
            self.assertIn("unknown", self.panel.status.text())
            self.controller.health = {"model_loaded": False}
            self.controller.model_info = {"loaded": False}
            for state in ["busy", "starting", "stopping", "probing"]:
                setattr(self.controller, state, True)
                self.panel._delete_selected()
                self.assertIn("server operation", self.panel.status.text())
                setattr(self.controller, state, False)
            self.controller.health = {}
            self.panel._delete_selected()
            self.assertIn("model status", self.panel.status.text())
            confirm.assert_not_called()
            trash.assert_not_called()
        self.assertTrue(self.model.exists())

    def test_outside_symlink_and_nonregular_or_non_gguf_files_are_rejected(self):
        outside = self.base / "outside.gguf"
        outside.write_bytes(gguf())
        alias = self.root / "alias.gguf"
        alias.symlink_to(outside)
        directory = self.root / "directory.gguf"
        directory.mkdir()
        folder_alias = self.root / "linked"
        folder_alias.symlink_to(self.base, target_is_directory=True)
        invalid = self.root / "invalid.gguf"
        invalid.write_bytes(b"not a model")
        other = self.root / "notes.txt"
        other.write_bytes(gguf())
        for path in [outside, alias, directory, folder_alias / outside.name, invalid, other,
                     self.root / ".." / outside.name]:
            with self.subTest(path=path):
                self.select(path)
                with patch.object(self.panel, "_confirm_delete") as confirm, patch("nexus_desktop.hub.QFile.moveToTrash") as trash:
                    self.panel._delete_selected()
                    confirm.assert_not_called()
                    trash.assert_not_called()
        self.assertTrue(outside.exists())
        self.assertTrue(self.model.exists())

    def test_selection_file_identity_and_loaded_state_are_revalidated_after_confirmation(self):
        alternative = self.root / "other.gguf"
        alternative.write_bytes(gguf())
        def change_selection(*_):
            self.select(alternative)
            return True
        def replace_file(*_):
            replacement = self.root / "replacement.tmp"
            replacement.write_bytes(gguf())
            replacement.replace(self.model)
            return True
        def load_model(*_):
            self.controller.mode = "external"
            self.controller.health = {"model_loaded": True}
            self.controller.model_info = {"loaded": True, "path": str(self.model)}
            return True
        for action in [change_selection, replace_file, load_model]:
            with self.subTest(action=action.__name__):
                self.select(self.model)
                with patch.object(self.panel, "_confirm_delete", side_effect=action), patch("nexus_desktop.hub.QFile.moveToTrash") as trash:
                    self.panel._delete_selected()
                    trash.assert_not_called()
                self.assertFalse(self.panel.deletion_in_progress)
                self.assertTrue(self.model.exists())
        self.assertEqual(self.deleted, [])

    def test_download_loadcheck_and_delete_conflicts_are_guarded(self):
        with patch.object(self.panel, "_confirm_delete") as confirm:
            for kind in ["download", "loadcheck", "delete"]:
                self.panel._jobs[kind] = (17, threading.Event())
                self.panel._delete_selected()
                self.panel._jobs.pop(kind)
            confirm.assert_not_called()
        self.panel._delete_confirming = True
        self.panel._update_local_actions()
        self.assertFalse(self.panel.load_button.isEnabled())
        self.assertFalse(self.panel.delete_button.isEnabled())
        with patch.object(self.panel, "_submit") as submit:
            self.panel._load_selected()
            self.panel.start_download()
            submit.assert_not_called()
        self.panel._delete_confirming = False

    def test_success_emits_original_path_and_refreshes_library(self):
        trash_folder = self.base / "fake-trash"
        trash_folder.mkdir()
        states = []
        self.panel.deletion_state_changed.connect(lambda: states.append(self.panel.deletion_in_progress))
        def move(path):
            destination = trash_folder / Path(path).name
            Path(path).rename(destination)
            return True
        with patch.object(self.panel, "_confirm_delete", return_value=True), patch("nexus_desktop.hub._move_to_trash", side_effect=move):
            self.panel._delete_selected()
            self.assertTrue(wait(lambda: self.deleted == [str(self.model)] and not self.panel._jobs))
        self.assertFalse(self.model.exists())
        self.assertEqual((trash_folder / self.model.name).read_bytes(), gguf())
        self.assertEqual(self.panel.local_files.count(), 0)
        self.assertFalse(self.panel.deletion_in_progress)
        self.assertIn(True, states)
        self.assertFalse(states[-1])
        self.assertIn("empty Trash", self.panel.status.text())

    def test_trash_failure_leaves_file_and_emits_no_deleted_signal(self):
        with patch.object(self.panel, "_confirm_delete", return_value=True), patch("nexus_desktop.hub._move_to_trash", return_value=False):
            self.panel._delete_selected()
            self.assertTrue(wait(lambda: "delete" not in self.panel._jobs))
        self.assertTrue(self.model.exists())
        self.assertEqual(self.deleted, [])
        self.assertIn("No permanent deletion", self.panel.status.text())
        self.assertFalse(self.panel.deletion_in_progress)

    def test_worker_cancellation_and_identity_change_prevent_trash(self):
        identity = _delete_file_identity(self.root, self.model)
        cancel = threading.Event()
        cancel.set()
        with patch("nexus_desktop.hub.QFile.moveToTrash") as trash:
            with self.assertRaises(HubError):
                _trash_model(self.model, self.root, identity, cancel)
            cancel.clear()
            self.model.write_bytes(gguf() + b"changed")
            with self.assertRaises(HubError):
                _trash_model(self.model, self.root, identity, cancel)
            trash.assert_not_called()
        self.assertTrue(self.model.exists())

    def test_real_qt_trash_isolated_in_temporary_xdg_home(self):
        script = r'''
import os, pathlib, struct, threading
from PySide6.QtCore import QCoreApplication
from nexus_desktop.hub import _delete_file_identity, _trash_model
app = QCoreApplication([])
root = pathlib.Path(os.environ["XDG_DATA_HOME"]) / "models"
root.mkdir(parents=True)
path = root / "nexus-disposable-trash-fixture.gguf"
payload = struct.pack("<4sIQQ", b"GGUF", 3, 0, 0)
path.write_bytes(payload)
identity = _delete_file_identity(root, path)
assert _trash_model(path, root, identity, threading.Event()) == str(path)
assert not path.exists()
files = list((pathlib.Path(os.environ["XDG_DATA_HOME"]) / "Trash/files").glob("nexus-disposable-trash-fixture*"))
assert len(files) == 1 and files[0].read_bytes() == payload
print("ISOLATED_QT_TRASH_OK")
'''
        env = {**os.environ, "XDG_DATA_HOME": str(self.base / "isolated-data"),
               "XDG_CACHE_HOME": str(self.base / "isolated-cache"), "QT_QPA_PLATFORM": "offscreen"}
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ISOLATED_QT_TRASH_OK", result.stdout)


class CompatibilityTests(unittest.TestCase):
    def test_known_unsupported_download_is_rejected_before_network_or_filesystem(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models"
            file = HubFile("owner/gemma", "a" * 40, "model.gguf", 4096, architecture="gemma4")
            client = HubClient()
            with patch.object(client, "_open") as network:
                with self.assertRaisesRegex(HubError, "gemma4"):
                    client.download(file, root, threading.Event(), lambda *_: None)
                network.assert_not_called()
            self.assertFalse(root.exists())

    def test_known_unsupported_file_is_disabled_and_local_load_emits_no_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "gemma.gguf"
            path.write_bytes(gguf("gemma4"))
            with patch.dict(os.environ, {"XDG_CACHE_HOME": str(root / "cache")}):
                panel = ModelHubPanel(None, str(root))
            try:
                self.assertTrue(wait(lambda: "local" not in panel._jobs))
                item = QListWidgetItem("model.gguf")
                item.setData(Qt.ItemDataRole.UserRole, HubFile("owner/model", "a" * 40, "model.gguf", 4096, architecture="gemma4"))
                panel.file_list.addItem(item)
                panel.file_list.setCurrentItem(item)
                self.assertFalse(panel.download_button.isEnabled())
                self.assertIn("not currently supported", panel.file_detail.text())
                seen = []
                panel.model_selected.connect(seen.append)
                panel._load_selected()
                self.assertTrue(wait(lambda: "loadcheck" not in panel._jobs))
                self.assertEqual(seen, [])
                self.assertIn("gemma4", panel.status.text())
            finally:
                panel.shutdown()
                panel.deleteLater()
                APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
