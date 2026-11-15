"""Nonblocking, bounded, generation-scoped localhost HTTP client."""
from __future__ import annotations
import json
from typing import Callable
from PySide6.QtCore import QObject, QByteArray, QTimer, QUrl
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply, QNetworkProxy
from .config import local_url


class ApiClient(QObject):
    MAX_RESPONSE_BYTES = 2 * 1024 * 1024
    READ_CHUNK_BYTES = 65536

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = QNetworkAccessManager(self)
        # Local management credentials must never traverse a system/application proxy.
        self.manager.setProxy(QNetworkProxy(QNetworkProxy.ProxyType.NoProxy))
        self.base = "http://127.0.0.1:8080"
        self.key = ""
        self.epoch = 0
        self.pending: dict[str, QNetworkReply] = {}
        self._deadlines: dict[QNetworkReply, QTimer] = {}

    def configure(self, base: str, key: str = ""):
        base = local_url(base)
        self.cancel()
        self.base, self.key = base, key

    def cancel(self):
        self.epoch += 1
        replies = list(self.pending.values())
        self.pending.clear()
        for reply in replies:
            timer = self._deadlines.pop(reply, None)
            if timer:
                timer.stop()
            reply.abort()

    def request(self, tag: str, path: str, callback: Callable, payload=None,
                *, base: str | None = None, authenticate=True, timeout=2500):
        if tag in self.pending:
            return False
        epoch = self.epoch
        request = QNetworkRequest(QUrl(local_url(base or self.base) + path))
        request.setTransferTimeout(timeout)  # Idle deadline, plus absolute timer below.
        request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute,
                             QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
        request.setRawHeader(b"Accept", b"application/json")
        if self.key and authenticate:
            request.setRawHeader(b"Authorization", ("Bearer " + self.key).encode())
        if payload is None:
            reply = self.manager.get(request)
        else:
            request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
            reply = self.manager.post(request, QByteArray(json.dumps(payload).encode()))
        reply.setReadBufferSize(self.READ_CHUNK_BYTES)
        self.pending[tag] = reply
        body = bytearray()
        state = {"too_large": False, "expired": False, "done": False, "draining": False}
        deadline = QTimer(reply)
        deadline.setSingleShot(True)
        self._deadlines[reply] = deadline

        def drain():
            if state["draining"]:
                return
            state["draining"] = True
            try:
                while reply.bytesAvailable() > 0:
                    remaining = self.MAX_RESPONSE_BYTES + 1 - len(body)
                    if remaining > 0:
                        chunk = bytes(reply.read(min(self.READ_CHUNK_BYTES, remaining)))
                        if not chunk:
                            break
                        body.extend(chunk)
                    if len(body) > self.MAX_RESPONSE_BYTES or remaining <= 0:
                        state["too_large"] = True
                        if not state["done"]:
                            reply.abort()
                        return
            finally:
                state["draining"] = False

        def metadata_received():
            length = reply.header(QNetworkRequest.KnownHeaders.ContentLengthHeader)
            if isinstance(length, int) and length > self.MAX_RESPONSE_BYTES:
                state["too_large"] = True
                reply.abort()

        def expired():
            state["expired"] = True
            reply.abort()

        def finished():
            if state["done"]:
                return
            state["done"] = True
            deadline.stop()
            self._deadlines.pop(reply, None)
            if self.pending.get(tag) is reply:
                self.pending.pop(tag, None)
            drain()
            code = int(reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute) or 0)
            network_error = reply.error()
            error = reply.errorString() if network_error != QNetworkReply.NetworkError.NoError else ""
            reply.deleteLater()
            if epoch != self.epoch:
                return
            data = {}
            if state["expired"]:
                error = "The server request timed out."
            elif state["too_large"]:
                error = "The server response exceeded the size limit."
            elif body:
                try:
                    data = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    error = "The server did not return valid JSON."
            elif 200 <= code < 300:
                error = "The server returned an empty JSON response."
            if 200 <= code < 300 and (error or not isinstance(data, dict)):
                code = 0
                error = error or "The server did not return a valid JSON object."
            callback(code, data, error, network_error)

        reply.metaDataChanged.connect(metadata_received)
        reply.readyRead.connect(drain)
        reply.finished.connect(finished)
        deadline.timeout.connect(expired)
        deadline.start(timeout)
        return True
