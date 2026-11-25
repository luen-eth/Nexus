"""Bounded asynchronous local chat with server-reported completion statistics."""
from __future__ import annotations

import json
import time
from typing import Any

from PySide6.QtCore import QByteArray, QObject, QSignalBlocker, Qt, QTimer, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkProxy, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from .config import local_url


class ChatTransport(QObject):
    """A separate network manager keeps chat independent of management polling."""
    delta = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    MAX_RESPONSE_BYTES = 2 * 1024 * 1024
    MAX_EVENT_BYTES = 256 * 1024
    IDLE_TIMEOUT_MS = 60000
    # Qt's own transfer timeout must never fire before the app-level idle
    # deadline: keep it strictly larger so a stall surfaces the friendlier
    # "timed out" message instead of Qt's opaque operation-canceled error.
    TRANSFER_TIMEOUT_MS = IDLE_TIMEOUT_MS + 5000
    ABSOLUTE_TIMEOUT_MS = 30 * 60 * 1000
    READ_BYTES = 65536

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = QNetworkAccessManager(self)
        self.manager.setProxy(QNetworkProxy(QNetworkProxy.ProxyType.NoProxy))
        self.reply = None
        self._epoch = 0

    def cancel(self):
        self._epoch += 1
        reply, self.reply = self.reply, None
        if reply is not None:
            reply.abort()

    def start(self, base, key, payload):
        self.cancel()
        request = QNetworkRequest(QUrl(local_url(base) + "/v1/chat/completions"))
        request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute,
                             QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
        request.setTransferTimeout(self.TRANSFER_TIMEOUT_MS)
        request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
        request.setRawHeader(b"Accept", b"text/event-stream")
        if key:
            request.setRawHeader(b"Authorization", ("Bearer " + key).encode())
        reply = self.manager.post(request, QByteArray(json.dumps(payload).encode()))
        reply.setReadBufferSize(self.READ_BYTES)
        self.reply = reply
        epoch = self._epoch
        state = {"bytes": 0, "buffer": bytearray(), "body": bytearray(), "event": [],
                 "event_bytes": 0, "usage": None, "reason": None, "done": False,
                 "error": "", "draining": False, "finished": False}
        deadline = QTimer(reply)
        deadline.setSingleShot(True)
        absolute = QTimer(reply)
        absolute.setSingleShot(True)

        def active():
            return epoch == self._epoch and self.reply is reply

        def fail(message):
            state["error"] = str(message)[:1000].replace(key, "[redacted]") if key else str(message)[:1000]
            if not state["finished"]:
                reply.abort()

        def arm_idle():
            # The deadline is activity-based: any received byte proves the stream
            # is alive, so a healthy slow reply never hits a wall clock.
            if active():
                deadline.start(self.IDLE_TIMEOUT_MS)

        def parse_event():
            data = b"\n".join(state["event"])
            state["event"], state["event_bytes"] = [], 0
            if not data or state["done"]:
                return
            if data == b"[DONE]":
                state["done"] = True
                return
            try:
                item = json.loads(data)
                if not isinstance(item, dict):
                    raise ValueError()
                if "error" in item:
                    problem = item["error"]
                    fail(problem.get("message", "Generation failed.") if isinstance(problem, dict) else "Generation failed.")
                    return
                usage = item.get("usage")
                if isinstance(usage, dict):
                    value = usage.get("completion_tokens")
                    if type(value) is int and value >= 0:
                        state["usage"] = dict(usage)
                choices = item.get("choices", [])
                if not isinstance(choices, list):
                    raise ValueError()
                if choices:
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        raise ValueError()
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        raise ValueError()
                    content = delta.get("content")
                    if content is not None and not isinstance(content, str):
                        raise ValueError()
                    reason = choice.get("finish_reason")
                    if reason is not None:
                        if not isinstance(reason, str):
                            raise ValueError()
                        state["reason"] = reason
                    if reason == "error":
                        fail("Generation failed before completion.")
                    elif content and active():
                        self.delta.emit(content)
            except (ValueError, UnicodeDecodeError):
                fail("The server returned an invalid chat event.")

        def drain():
            if state["draining"] or not active():
                return
            state["draining"] = True
            try:
                while reply.bytesAvailable() and not state["error"] and active():
                    chunk = bytes(reply.read(self.READ_BYTES))
                    if not chunk:
                        break
                    state["bytes"] += len(chunk)
                    arm_idle()
                    if state["bytes"] > self.MAX_RESPONSE_BYTES:
                        fail("The chat response exceeded the size limit.")
                        return
                    code = int(reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute) or 0)
                    mime = str(reply.header(QNetworkRequest.KnownHeaders.ContentTypeHeader) or "").lower()
                    if code != 200 or "text/event-stream" not in mime:
                        state["body"].extend(chunk)
                        continue
                    state["buffer"].extend(chunk)
                    while b"\n" in state["buffer"] and not state["error"] and active():
                        line, _, rest = state["buffer"].partition(b"\n")
                        state["buffer"] = bytearray(rest)
                        line = line.removesuffix(b"\r")
                        if not line:
                            parse_event()
                        elif line.startswith(b"data:"):
                            value = line[5:]
                            if value.startswith(b" "):
                                value = value[1:]
                            state["event"].append(value)
                            state["event_bytes"] += len(value)
                        if state["event_bytes"] + len(state["buffer"]) > self.MAX_EVENT_BYTES:
                            fail("The chat event exceeded the size limit.")
                            return
                    if len(state["buffer"]) > self.MAX_EVENT_BYTES:
                        fail("The chat event exceeded the size limit.")
            finally:
                state["draining"] = False

        def expired():
            if active():
                fail("The chat request timed out. Try a shorter prompt or output limit.")

        def absolute_expired():
            if active():
                fail("The chat request exceeded the total time limit.")

        def finished():
            if state["finished"]:
                return
            state["finished"] = True
            deadline.stop()
            absolute.stop()
            if not active():
                reply.deleteLater()
                return
            drain()
            # A stream may end without the final blank line; flush a pending
            # data:-bearing event once so the last chunk and [DONE] are not
            # dropped. Only a clean EOF (NoError) may flush: a transport error
            # mid-event means the tail is truncated, and parsing it would
            # mislabel a network cut as "invalid chat event".
            if not state["error"] and reply.error() == QNetworkReply.NetworkError.NoError:
                tail = bytes(state["buffer"]).removesuffix(b"\r")
                if tail.startswith(b"data:"):
                    value = tail[5:]
                    state["event"].append(value[1:] if value.startswith(b" ") else value)
                if state["event"]:
                    parse_event()
            code = int(reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute) or 0)
            network_error = reply.error()
            self.reply = None
            reply.deleteLater()
            error = state["error"]
            if code in {401, 403}:
                error = "Authentication failed. Reconnect with a valid API key."
            elif state["body"]:
                try:
                    body = json.loads(state["body"])
                    if not isinstance(body, dict):
                        raise ValueError()
                    problem = body.get("error")
                    if problem:
                        error = problem.get("message", "Chat request failed.") if isinstance(problem, dict) else "Chat request failed."
                    elif code == 200:
                        # Some compatible servers return JSON despite stream=true.
                        message = body["choices"][0]["message"]["content"]
                        if not isinstance(message, str):
                            raise ValueError()
                        self.delta.emit(message)
                        state["usage"] = body.get("usage")
                        state["reason"] = body["choices"][0].get("finish_reason")
                        state["done"] = True
                    else:
                        error = f"Chat request failed (HTTP {code})."
                except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError):
                    error = error or f"The server returned an invalid chat response (HTTP {code})."
            if not error and code != 200:
                error = f"Chat request failed (HTTP {code})."
            if not error and network_error != QNetworkReply.NetworkError.NoError:
                error = "The chat connection ended unexpectedly."
            if not error and not state["done"] and state["reason"] is None:
                # EOF without [DONE] but with a finish chunk is still a complete reply;
                # a dangling content delta without a finish chunk stays an error.
                error = "The chat stream ended before completion."
            # A delta observer may cancel or replace this request synchronously.
            if epoch != self._epoch:
                return
            if error:
                safe = str(error)[:1000]
                self.failed.emit(safe.replace(key, "[redacted]") if key else safe)
            else:
                self.completed.emit({"usage": state["usage"], "finish_reason": state["reason"]})

        reply.readyRead.connect(drain)
        reply.finished.connect(finished)
        deadline.timeout.connect(expired)
        absolute.timeout.connect(absolute_expired)
        deadline.start(self.IDLE_TIMEOUT_MS)
        absolute.start(self.ABSOLUTE_TIMEOUT_MS)


class Composer(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            event.accept()
            self.submitted.emit()
        else:
            super().keyPressEvent(event)


class ChatPanel(QWidget):
    MAX_INPUT_CHARS = 16384
    MAX_OUTPUT_CHARS = 262144
    MAX_HISTORY_CHARS = 65536
    MAX_HISTORY_MESSAGES = 32
    MAX_VISIBLE_MESSAGES = 64
    STATS_TOOLTIP = ("Completion tokens are reported by the server. Elapsed time includes network, queue, "
                     "prompt processing, and generation. Tokens/s is completion tokens divided by that total time. "
                     "Counts are unavailable for interrupted replies or servers without usage data.")

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.setObjectName("chatPanel")
        self.controller = controller
        self.transport = ChatTransport(self)
        self.messages: list[dict[str, Any]] = []
        self.last_stats = None
        self._history = []
        self._cards = []
        self._identity = None
        self._pending = None
        self._started = 0.0
        self._closed = False
        self._refreshing = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Model"))
        self.model_selector = QComboBox()
        self.model_selector.setObjectName("chatModel")
        self.model_selector.setMinimumWidth(180)
        controls.addWidget(self.model_selector, 1)
        self.new_chat_button = QPushButton("New chat")
        self.new_chat_button.clicked.connect(self.new_chat)
        controls.addWidget(self.new_chat_button)
        layout.addLayout(controls)
        self.notice = QLabel()
        self.notice.setObjectName("chatNotice")
        self.notice.setTextFormat(Qt.TextFormat.PlainText)
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        self.transcript = QScrollArea()
        self.transcript.setObjectName("chatTranscript")
        self.transcript.setWidgetResizable(True)
        self.transcript.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        transcript_body = QWidget()
        transcript_body.setObjectName("chatCanvas")
        self._transcript_layout = QVBoxLayout(transcript_body)
        self._transcript_layout.setContentsMargins(12, 12, 12, 12)
        self._transcript_layout.setSpacing(16)
        self.empty_state = QWidget()
        self.empty_state.setObjectName("chatEmpty")
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setContentsMargins(32, 32, 32, 32)
        empty_layout.setSpacing(10)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_title = QLabel("Start a conversation")
        empty_title.setObjectName("chatEmptyTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_title.setTextFormat(Qt.TextFormat.PlainText)
        empty_text = QLabel("Choose a loaded model and send your first message.")
        empty_text.setObjectName("chatEmptyHint")
        empty_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_text.setTextFormat(Qt.TextFormat.PlainText)
        empty_text.setWordWrap(True)
        empty_layout.addWidget(empty_title)
        empty_layout.addWidget(empty_text)
        self._transcript_layout.addWidget(self.empty_state, 1)
        self._transcript_layout.addStretch(0)
        self.transcript.setWidget(transcript_body)
        layout.addWidget(self.transcript, 1)
        options = QHBoxLayout()
        options.addWidget(QLabel("Max output"))
        self.max_output = QSpinBox()
        self.max_output.setRange(1, 2048)
        self.max_output.setValue(128)
        self.max_output.setSuffix(" tokens")
        self.max_output.setToolTip("Maximum output tokens. The server checks the full conversation and output against the model context window.")
        options.addWidget(self.max_output)
        options.addWidget(QLabel("Temperature"))
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.7)
        options.addWidget(self.temperature)
        options.addStretch()
        layout.addLayout(options)
        self.composer = Composer()
        self.composer.setObjectName("chatComposer")
        self.composer.setPlaceholderText("Message your local model…")
        self.composer.setMinimumHeight(90)
        self.composer.setMaximumHeight(150)
        self.composer.setMaximumBlockCount(1024)
        self.composer.submitted.connect(self.send)
        self.composer.textChanged.connect(self._update_buttons)
        layout.addWidget(self.composer)
        actions = QHBoxLayout()
        hint = QLabel("Enter to send · Shift+Enter for a new line")
        hint.setObjectName("chatHint")
        actions.addWidget(hint, 1)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.send_button = QPushButton("Send")
        self.send_button.setProperty("primary", True)
        self.send_button.clicked.connect(self.send)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.send_button)
        layout.addLayout(actions)
        self.transport.delta.connect(self._delta)
        self.transport.completed.connect(self._completed)
        self.transport.failed.connect(self._failed)
        self.model_selector.currentIndexChanged.connect(self.refresh)
        self.controller.changed.connect(self.refresh)
        self.refresh()

    @property
    def is_generating(self):
        return self._pending is not None

    def _ready(self):
        c = self.controller
        context = c.model_info.get("max_seq_len")
        model_ready = (c.model_info.get("loaded") is True
                       and c.model_info.get("id") == self.model_selector.currentData()
                       and type(context) is int and context > 1)
        return (not self._closed and c.mode != "disconnected" and bool(c.health.get("model_loaded"))
                and c.health.get("status") != "error" and c.authorized and not c.busy and not c.stopping
                and bool(self.model_selector.currentData()) and model_ready)

    def refresh(self, *_):
        if self._closed or self._refreshing:
            return
        self._refreshing = True
        try:
            c = self.controller
            selected = self.model_selector.currentData()
            names = list(dict.fromkeys(m["id"] for m in c.models if isinstance(m.get("id"), str)))[:128]
            if [self.model_selector.itemData(i) for i in range(self.model_selector.count())] != names:
                blocker = QSignalBlocker(self.model_selector)
                self.model_selector.clear()
                for name in names:
                    self.model_selector.addItem(name, name)
                preferred = selected if selected in names else c.health.get("model")
                self.model_selector.setCurrentIndex(max(0, self.model_selector.findData(preferred)))
                del blocker
            selected = self.model_selector.currentData()
            info = c.model_info
            context = info.get("max_seq_len")
            if type(context) is int and context > 1:
                self.max_output.setMaximum(min(65536, context - 1))
            identity = (c.mode != "disconnected", c.api.base, c.api.key, c.api.epoch, selected,
                        info.get("epoch"), info.get("path"), info.get("backend"), context,
                        info.get("id"), c.authorized)
            if self._identity is not None and identity != self._identity:
                had_messages = bool(self.messages)
                self.new_chat()
                if had_messages:
                    self.notice.setText("The connection or model changed. A new conversation is ready.")
            self._identity = identity
            if not self._ready():
                # A poll-driven refresh never cancels an active stream: only a real
                # disconnect (identity change above) or an explicit Stop may do that.
                if not c.authorized:
                    self.notice.setText("Reconnect with a valid API key to chat.")
                elif c.busy:
                    self.notice.setText("Wait for the model operation to finish.")
                elif c.mode == "disconnected":
                    self.notice.setText("Connect to a local server in Server settings to start chatting.")
                elif not c.health.get("model_loaded") or not selected:
                    self.notice.setText("Load a model in Server settings to start chatting.")
                elif info.get("id") != selected or info.get("loaded") is not True or type(context) is not int:
                    self.notice.setText("Reading the loaded model details…")
                else:
                    self.notice.setText("The server is not ready. Check Server settings.")
            elif not self.messages and not self.notice.text().startswith("The connection"):
                self.notice.setText("Your conversation stays on this device. Choose a loaded model and send a message.")
            self._update_buttons()
        finally:
            self._refreshing = False

    def _update_buttons(self):
        text = self.composer.toPlainText().strip()
        self.send_button.setEnabled(self._ready() and not self.is_generating and bool(text))
        self.stop_button.setEnabled(self.is_generating)
        self.model_selector.setEnabled(not self.is_generating)
        self.max_output.setEnabled(not self.is_generating)
        self.temperature.setEnabled(not self.is_generating)

    def _add_message(self, role, content):
        message = {"role": role, "content": content, "status": "complete" if role == "user" else "pending", "stats": None}
        frame = QFrame()
        frame.setObjectName("chatMessage")
        frame.setProperty("role", role)
        box = QVBoxLayout(frame)
        box.setContentsMargins(16, 14, 16, 14)
        box.setSpacing(10)
        title = QLabel("You" if role == "user" else "Nexus")
        title.setObjectName("chatRole")
        body = QLabel(content or "Thinking…")
        body.setObjectName("chatMessageText")
        body.setTextFormat(Qt.TextFormat.PlainText)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        footer = QLabel()
        footer.setObjectName("chatStats")
        footer.setTextFormat(Qt.TextFormat.PlainText)
        footer.setWordWrap(True)
        footer.setToolTip(self.STATS_TOOLTIP)
        box.addWidget(title)
        box.addWidget(body)
        box.addWidget(footer)
        footer.setVisible(role == "assistant")
        self.empty_state.hide()
        self._transcript_layout.setStretch(self._transcript_layout.count() - 1, 1)
        self._transcript_layout.insertWidget(self._transcript_layout.count() - 1, frame)
        self.messages.append(message)
        self._cards.append((frame, body, footer))
        while len(self.messages) > self.MAX_VISIBLE_MESSAGES:
            self.messages.pop(0)
            old, _, _ = self._cards.pop(0)
            self._transcript_layout.removeWidget(old)
            old.deleteLater()
        self._scroll()
        return message, body, footer

    def _scroll(self):
        QTimer.singleShot(0, lambda: self.transcript.verticalScrollBar().setValue(self.transcript.verticalScrollBar().maximum()) if not self._closed else None)

    def send(self):
        if not self._ready() or self.is_generating:
            return
        text = self.composer.toPlainText().strip()
        if not text:
            return
        if len(text) > self.MAX_INPUT_CHARS:
            self.notice.setText(f"Keep each message under {self.MAX_INPUT_CHARS:,} characters.")
            return
        history = self._history + [{"role": "user", "content": text}]
        if len(history) >= self.MAX_HISTORY_MESSAGES or sum(len(m["content"]) for m in history) > self.MAX_HISTORY_CHARS:
            self.notice.setText("This conversation reached its local history limit. Start a new chat to continue.")
            return
        self.notice.setText("Generating…")
        self._add_message("user", text)
        message, body, footer = self._add_message("assistant", "")
        self._pending = {"message": message, "body": body, "footer": footer, "history": history, "user": text}
        self.composer.clear()
        self._started = time.perf_counter()
        self.last_stats = None
        self._update_buttons()
        payload = {"model": self.model_selector.currentData(), "messages": history,
                   "max_tokens": self.max_output.value(), "temperature": self.temperature.value(), "stream": True}
        try:
            self.transport.start(self.controller.api.base, self.controller.api.key, payload)
        except (ValueError, UnicodeError) as error:
            self._failed(str(error))

    def _delta(self, text):
        if self._pending is None or self._closed:
            return
        pending = self._pending
        content = pending["message"]["content"] + text
        if len(content) > self.MAX_OUTPUT_CHARS:
            self.transport.cancel()
            self._failed("The reply exceeded the display limit. Try a smaller output limit.")
            return
        pending["message"]["content"] = content
        pending["body"].setText(content)
        self._scroll()

    def _finish(self, status, result=None):
        pending, self._pending = self._pending, None
        if pending is None:
            return None
        elapsed = max(0.0, time.perf_counter() - self._started)
        usage = (result or {}).get("usage")
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if status != "complete" or type(tokens) is not int or tokens < 0:
            tokens = None
        speed = tokens / elapsed if tokens is not None and elapsed > 0 else None
        stats = {"completion_tokens": tokens, "elapsed_seconds": elapsed, "tokens_per_second": speed,
                 "finish_reason": (result or {}).get("finish_reason"), "status": status}
        self.last_stats = stats
        pending["message"].update(status=status, stats=stats)
        count_text = f"{tokens:,} tokens" if tokens is not None else "Tokens unavailable"
        speed_text = f"{speed:.1f} tokens/s" if speed is not None else "tokens/s unavailable"
        suffix = " · Stopped" if status == "cancelled" else " · Failed" if status == "error" else ""
        pending["footer"].setText(f"{count_text} · {elapsed:.2f} s · {speed_text}{suffix}")
        if not pending["message"]["content"]:
            pending["body"].setText("No response received." if status != "complete" else "The model returned no text.")
        self._update_buttons()
        self._scroll()
        return pending

    def _completed(self, result):
        pending = self._finish("complete", result)
        if pending:
            content = pending["message"]["content"]
            history = pending["history"] + [{"role": "assistant", "content": content}]
            if sum(len(message["content"]) for message in history) <= self.MAX_HISTORY_CHARS:
                self._history = history
                self.notice.setText("Reply complete." if result.get("finish_reason") != "length" else "Output limit reached. Send a follow-up or increase Max output.")
            else:
                self._history.clear()
                self.notice.setText("Reply complete. The history limit was reached; the next message will start a new conversation.")

    def _failed(self, message):
        pending = self._finish("error")
        if pending:
            safe = str(message)[:1000]
            key = self.controller.api.key
            self.notice.setText(safe.replace(key, "[redacted]") if key else safe)
            if not self.composer.toPlainText():
                self.composer.setPlainText(pending["user"])

    def stop(self):
        self.transport.cancel()
        if self._finish("cancelled"):
            self.notice.setText("Reply stopped. Partial replies are not added to conversation history.")

    def new_chat(self):
        self.stop()
        self._history.clear()
        self.messages.clear()
        self.last_stats = None
        for frame, _, _ in self._cards:
            self._transcript_layout.removeWidget(frame)
            frame.deleteLater()
        self._cards.clear()
        self.empty_state.show()
        self._transcript_layout.setStretch(self._transcript_layout.count() - 1, 0)
        self.notice.setText("New conversation. Send a message to begin.")
        self._update_buttons()

    def shutdown(self):
        if self._closed:
            return
        self.stop()
        self._closed = True
        try:
            self.controller.changed.disconnect(self.refresh)
        except (RuntimeError, TypeError):
            pass
        self._update_buttons()
