"""Real asynchronous HTTP regressions for local chat, history, usage and cancellation."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtNetwork import QNetworkProxy, QNetworkReply
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from nexus_desktop.chat import ChatPanel, ChatTransport
from nexus_desktop.controller import ServerController

APP = QApplication.instance() or QApplication(["nexus-chat-tests"])
APP.setQuitOnLastWindowClosed(False)


def wait(predicate, seconds=4):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        APP.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    return bool(predicate())


def event(value):
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\r\n\r\n").encode()


def lf(value):
    """Same framing with bare LF terminators; the final blank line is added by callers."""
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()


def delta(text):
    return event({"choices": [{"delta": {"content": text}, "finish_reason": None}]})


def final(tokens=7, *, usage=True):
    item = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    if usage:
        item["usage"] = {"prompt_tokens": 11, "completion_tokens": tokens, "total_tokens": 11 + tokens}
    return event(item) + b"data: [DONE]\r\n\r\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((body, self.headers.get("Authorization"), self.path))
        scenario = self.server.scenarios.popleft() if self.server.scenarios else "normal"
        if scenario == "auth" or self.headers.get("Authorization") != "Bearer test-chat-key":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Unauthorized"}}')
            return
        if scenario == "error":
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Prompt and output exceed the context window."}}')
            return
        if scenario == "abort":
            # Announce more bytes than will ever arrive: closing now is a
            # transport error (RemoteHostClosed), not a clean EOF, so the
            # truncated residual line must never be parsed as an event.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", "2048")
            self.end_headers()
            self.wfile.write(b"data: " + json.dumps(
                {"choices": [{"delta": {"content": "half"}, "finish_reason": None}]})[:24].encode())
            self.wfile.flush()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            if scenario == "slow":
                self.wfile.write(delta("Partial"))
                self.wfile.flush()
                self.server.release.wait(1.5)
                self.wfile.write(delta(" stale tail") + final(99))
            elif scenario == "drip":
                for step in range(8):
                    self.wfile.write(delta(f"piece{step} "))
                    self.wfile.flush()
                    time.sleep(.25)
                self.wfile.write(final(5))
            elif scenario == "lf":
                item = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}}
                # The last event ends with a single LF and no blank line, and no [DONE] follows.
                self.wfile.write(lf({"choices": [{"delta": {"content": "Last "}, "finish_reason": None}]})
                                 + lf({"choices": [{"delta": {"content": "event"}, "finish_reason": None}]})
                                 + ("data: " + json.dumps(item) + "\n").encode())
            elif scenario == "no-done":
                item = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15}}
                self.wfile.write(delta("Unmarked ") + event(item))
            elif scenario == "silent":
                self.server.release.wait(2.5)
            elif scenario == "large":
                self.wfile.write(delta("x" * 8192) + final())
            elif scenario == "invalid":
                self.wfile.write(b'data: {invalid}\n\n')
            elif scenario == "truncated":
                self.wfile.write(delta("unfinished"))
            else:
                payload = (b": keep-alive\r\n\r\n" + delta("Héllo ") + delta("🌍 <img src='https://invalid/image'>")
                           + final(7, usage=scenario != "no-usage"))
                # Three-byte writes split both UTF-8 characters and SSE frames.
                for index in range(0, len(payload), 3):
                    self.wfile.write(payload[index:index + 3])
                    self.wfile.flush()
                    if index % 48 == 0:
                        time.sleep(.002)
        except (BrokenPipeError, ConnectionResetError):
            pass


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.requests = []
        self.server.scenarios = deque()
        self.server.release = threading.Event()
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.02), daemon=True)
        self.thread.start()
        self.controller = ServerController()
        self.controller.api.configure(f"http://127.0.0.1:{self.server.server_port}", "test-chat-key")
        self.controller.mode = "external"
        self.controller.health = {"status": "ok", "model_loaded": True, "model": "fixture"}
        self.controller.models = [{"id": "fixture"}]
        self.controller.model_info = {"loaded": True, "id": "fixture", "epoch": 1,
                                      "path": "/fixture.gguf", "backend": "cpu", "max_seq_len": 512}
        self.panel = ChatPanel(self.controller)
        self.panel.resize(900, 800)
        self.panel.show()
        APP.processEvents()

    def tearDown(self):
        self.panel.shutdown()
        self.panel.close()
        self.panel.deleteLater()
        self.controller.disconnect()
        self.controller.deleteLater()
        self.server.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()

    def send(self, text="Hello"):
        self.panel.composer.setPlainText(text)
        self.panel.send()
        self.assertTrue(wait(lambda: not self.panel.is_generating))

    def test_fragmented_utf8_usage_history_and_plain_text_rendering(self):
        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(5)
        self.send()
        timer.stop()
        self.assertGreater(len(ticks), 1)
        self.assertEqual(self.panel.messages[-1]["content"], "Héllo 🌍 <img src='https://invalid/image'>")
        self.assertEqual(self.panel._cards[-1][1].textFormat(), Qt.TextFormat.PlainText)
        stats = self.panel.last_stats
        self.assertEqual(stats["completion_tokens"], 7)  # Two delta events are not two tokens.
        self.assertGreater(stats["elapsed_seconds"], 0)
        self.assertAlmostEqual(stats["tokens_per_second"], 7 / stats["elapsed_seconds"])
        self.assertIn("7 tokens", self.panel._cards[-1][2].text())
        self.send("And then?")
        body, authorization, path = self.server.requests[-1]
        self.assertEqual(authorization, "Bearer test-chat-key")
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual([turn["role"] for turn in body["messages"]], ["user", "assistant", "user"])
        self.assertEqual(body["messages"][1]["content"], self.panel.messages[1]["content"])
        self.assertEqual(body["max_tokens"], 128)
        self.assertTrue(body["stream"])

    def test_cancel_preserves_partial_without_inventing_usage_or_history(self):
        self.server.scenarios.append("slow")
        self.panel.composer.setPlainText("First")
        self.panel.send()
        self.assertTrue(wait(lambda: self.panel.messages[-1]["content"] == "Partial"))
        management_epoch = self.controller.api.epoch
        self.panel.stop()
        self.assertFalse(self.panel.is_generating)
        self.assertEqual(self.panel.messages[-1]["content"], "Partial")
        self.assertEqual(self.panel.last_stats["status"], "cancelled")
        self.assertIsNone(self.panel.last_stats["completion_tokens"])
        self.assertIsNone(self.panel.last_stats["tokens_per_second"])
        self.assertEqual(self.panel._history, [])
        self.assertEqual(self.controller.api.epoch, management_epoch)
        self.server.release.set()
        self.send("Second")
        self.assertEqual(self.server.requests[-1][0]["messages"], [{"role": "user", "content": "Second"}])
        self.assertNotIn("stale tail", self.panel.messages[-1]["content"])
        self.assertEqual(self.panel.last_stats["completion_tokens"], 7)

    def test_disconnect_epoch_reload_and_key_change_clear_history_and_stale_callbacks(self):
        self.send()
        self.controller.api.key = "replacement-key"
        self.controller.changed.emit()
        self.assertEqual(self.panel.messages, [])
        self.assertEqual(self.panel._history, [])
        self.controller.api.key = "test-chat-key"
        self.controller.changed.emit()
        self.send()
        self.controller.model_info["epoch"] = 2
        self.controller.changed.emit()
        self.assertEqual(self.panel.messages, [])
        self.assertEqual(self.panel._history, [])
        self.server.scenarios.append("slow")
        self.panel.composer.setPlainText("In flight")
        self.panel.send()
        self.assertTrue(wait(lambda: self.panel.messages[-1]["content"] == "Partial"))
        self.controller.disconnect()
        self.assertFalse(self.panel.is_generating)
        self.assertEqual(self.panel.messages, [])
        self.assertFalse(self.panel.send_button.isEnabled())
        self.server.release.set()
        QTest.qWait(100)
        APP.processEvents()
        self.assertEqual(self.panel.messages, [])
        self.assertIsNone(self.panel.last_stats)

    def test_failed_model_operation_preserves_history_and_a_new_epoch_clears_it(self):
        self.send()
        self.assertTrue(self.panel._history)
        self.controller.busy = True  # A failed load toggles busy without changing the model epoch.
        self.controller.changed.emit()
        self.assertTrue(self.panel._history)
        self.assertEqual(len(self.panel.messages), 2)
        self.controller.busy = False
        self.controller.changed.emit()
        self.assertTrue(self.panel._history)
        self.controller.model_info["epoch"] = 2  # A successful load reports a new epoch.
        self.controller.changed.emit()
        self.assertEqual(self.panel._history, [])
        self.assertEqual(self.panel.messages, [])
        self.send("Fresh model conversation")
        self.assertEqual(self.server.requests[-1][0]["messages"],
                         [{"role": "user", "content": "Fresh model conversation"}])

    def test_transient_health_failures_keep_generation_and_three_consecutive_clear(self):
        self.server.scenarios.append("slow")
        self.panel.composer.setPlainText("In flight")
        self.panel.send()
        self.assertTrue(wait(lambda: self.panel.messages[-1]["content"] == "Partial"))
        failure = (0, {}, "unreachable", QNetworkReply.NetworkError.ConnectionRefusedError)
        success = (200, {"status": "ok", "model_loaded": True, "model": "fixture"}, "",
                   QNetworkReply.NetworkError.NoError)
        for _ in range(2):
            self.controller._health(*failure)
            self.controller.changed.emit()  # Other polls keep refreshing the UI meanwhile.
            self.assertTrue(self.panel.is_generating)
            self.assertEqual([m["content"] for m in self.panel.messages], ["In flight", "Partial"])
            self.assertTrue(self.controller.health.get("model_loaded"))
        # A single successful poll resets the grace counter: two more
        # cumulative failures (with a success in between) are not three
        # consecutive ones and must not wipe the kept health.
        self.controller._health(*success)
        self.assertEqual(self.controller.state, "Ready")
        for _ in range(2):
            self.controller._health(*failure)
            self.controller.changed.emit()
            self.assertTrue(self.panel.is_generating)
            self.assertTrue(self.controller.health.get("model_loaded"))
            self.assertNotEqual(self.controller.state, "No connection")
        self.controller._health(*failure)  # The third consecutive failure clears the kept health.
        self.assertEqual(self.controller.health, {})
        self.controller.changed.emit()
        self.assertTrue(self.panel.is_generating)  # A poll-driven refresh never cancels the stream.
        self.server.release.set()
        self.assertTrue(wait(lambda: not self.panel.is_generating))
        self.assertEqual(self.panel.last_stats["status"], "complete")
        self.assertEqual(self.panel._history[-1]["content"], "Partial stale tail")

    def test_constant_activity_renews_the_idle_deadline_but_silence_still_aborts(self):
        # Qt's transfer timeout must strictly outlive the app idle deadline so
        # the friendlier stall message always wins the race.
        self.assertGreater(ChatTransport.TRANSFER_TIMEOUT_MS, ChatTransport.IDLE_TIMEOUT_MS)
        with patch.object(ChatTransport, "IDLE_TIMEOUT_MS", 1000), \
                patch.object(ChatTransport, "ABSOLUTE_TIMEOUT_MS", 6000):
            self.server.scenarios.append("drip")
            self.panel.composer.setPlainText("Slow and steady")
            self.panel.send()
            self.assertTrue(wait(lambda: not self.panel.is_generating, seconds=10))
            self.assertEqual(self.panel.last_stats["status"], "complete")
            self.assertGreater(self.panel.last_stats["elapsed_seconds"], 1.2)
            self.assertEqual(self.panel._history[-1]["role"], "assistant")
            self.assertIn("piece7", self.panel._history[-1]["content"])
            self.server.scenarios.append("silent")
            self.panel.composer.setPlainText("Say nothing")
            self.panel.send()
            self.assertTrue(wait(lambda: not self.panel.is_generating, seconds=6))
            self.assertEqual(self.panel.last_stats["status"], "error")
            self.assertIn("timed out", self.panel.notice.text())
            self.assertNotIn("Say nothing", [m["content"] for m in self.panel._history])

    def test_missing_final_blank_line_and_missing_done_marker_still_commit(self):
        for scenario, tokens, reply in (("lf", 4, "Last event"), ("no-done", 6, "Unmarked ")):
            with self.subTest(scenario=scenario):
                self.server.scenarios.append(scenario)
                self.send("Finish anyway")
                self.assertEqual(self.panel.last_stats["status"], "complete")
                self.assertEqual(self.panel.last_stats["completion_tokens"], tokens)
                self.assertEqual(self.panel.last_stats["finish_reason"], "stop")
                self.assertEqual(self.panel._history[-1]["role"], "assistant")
                self.assertEqual(self.panel._history[-1]["content"], reply)

    def test_transport_error_mid_event_is_not_relabelled_as_an_invalid_event(self):
        self.server.scenarios.append("abort")
        self.send("Cut me off")
        self.assertEqual(self.panel.last_stats["status"], "error")
        self.assertIn("ended unexpectedly", self.panel.notice.text())
        self.assertNotIn("invalid chat event", self.panel.notice.text())
        self.assertEqual(self.panel._history, [])

    def test_auth_context_and_invalid_stream_errors_do_not_commit_history(self):
        for scenario, expected in [("auth", "API key"), ("error", "context window"),
                                   ("invalid", "invalid chat event"), ("truncated", "before completion")]:
            with self.subTest(scenario=scenario):
                self.panel.new_chat()
                self.server.scenarios.append(scenario)
                self.send("Retry me")
                self.assertEqual(self.panel.last_stats["status"], "error")
                self.assertIsNone(self.panel.last_stats["completion_tokens"])
                self.assertIn(expected, self.panel.notice.text())
                self.assertEqual(self.panel._history, [])
                self.assertEqual(self.panel.composer.toPlainText(), "Retry me")

    def test_missing_usage_is_explicit_and_output_limit_tracks_context(self):
        self.server.scenarios.append("no-usage")
        self.controller.model_info["max_seq_len"] = 64
        self.controller.changed.emit()
        self.assertEqual(self.panel.max_output.maximum(), 63)
        self.send()
        self.assertEqual(self.server.requests[-1][0]["max_tokens"], 63)
        self.assertEqual(self.panel.last_stats["status"], "complete")
        self.assertIsNone(self.panel.last_stats["completion_tokens"])
        self.assertIn("Tokens unavailable", self.panel._cards[-1][2].text())

    def test_send_waits_for_coherent_model_details_and_context(self):
        info = self.controller.model_info
        self.controller.model_info = {}
        self.controller.changed.emit()
        self.panel.composer.setPlainText("Wait for metadata")
        self.assertFalse(self.panel.send_button.isEnabled())
        self.panel.send()
        self.assertEqual(self.server.requests, [])
        self.controller.model_info = {**info, "id": "old-model"}
        self.controller.changed.emit()
        self.assertFalse(self.panel.send_button.isEnabled())
        self.controller.model_info = info
        self.controller.changed.emit()
        self.assertTrue(self.panel.send_button.isEnabled())
        self.send()
        self.assertEqual(self.panel.last_stats["status"], "complete")

    def test_response_and_input_bounds_fail_without_blocking_or_sending_oversize_input(self):
        self.panel.composer.setPlainText("a" * (self.panel.MAX_INPUT_CHARS + 1))
        self.panel.send()
        self.assertEqual(self.server.requests, [])
        self.assertIn("characters", self.panel.notice.text())
        self.panel.transport.MAX_RESPONSE_BYTES = 1024
        self.server.scenarios.append("large")
        self.send("bounded")
        self.assertEqual(self.panel.last_stats["status"], "error")
        self.assertIn("size limit", self.panel.notice.text())
        self.assertEqual(self.panel._history, [])

    def test_enter_sends_shift_enter_adds_line_and_shutdown_cancels(self):
        self.panel.composer.setPlainText("Hello")
        QTest.keyClick(self.panel.composer, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        self.assertIn("\n", self.panel.composer.toPlainText())
        self.server.scenarios.append("slow")
        QTest.keyClick(self.panel.composer, Qt.Key.Key_Return)
        self.assertTrue(wait(lambda: bool(self.server.requests)))
        self.assertTrue(self.panel.is_generating)
        self.panel.shutdown()
        self.assertFalse(self.panel.is_generating)
        self.assertIsNone(self.panel.transport.reply)
        self.assertFalse(self.panel.send_button.isEnabled())

    def test_transport_ignores_application_proxy(self):
        previous = QNetworkProxy.applicationProxy()
        QNetworkProxy.setApplicationProxy(QNetworkProxy(QNetworkProxy.ProxyType.HttpProxy, "127.0.0.1", 1))
        try:
            self.send()
            self.assertEqual(self.panel.last_stats["status"], "complete")
        finally:
            QNetworkProxy.setApplicationProxy(previous)
