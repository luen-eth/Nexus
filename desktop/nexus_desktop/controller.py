"""Own only the QProcess we start. All network/process operations are asynchronous."""
from __future__ import annotations
import codecs
from collections import deque
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
import re
import secrets
import shutil
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal
from PySide6.QtNetwork import QNetworkReply
from .api import ApiClient
from .config import Settings, discover_binary, local_url


class ServerController(QObject):
    changed = Signal()
    log_received = Signal(str)
    error = Signal(str)
    stopped = Signal()
    MAX_LOG_LINES = 600
    MAX_LINE_CHARS = 1200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.api = ApiClient(self)
        self.process: QProcess | None = None
        self.mode = "disconnected"
        self.state = "Bağlı değil"
        self.detail = "Bir sunucu başlatın veya yerel sunucuya bağlanın."
        self.health: dict = {}
        self.models: list = []
        self.metrics: dict = {}
        self.model_info: dict = {}
        self.busy = False
        self.authorized = True
        self.starting = False
        self.stopping = False
        self.probing = False
        self._restart: Settings | None = None
        self._settings: Settings | None = None
        self._secret_history: deque[str] = deque(maxlen=8)
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._tail = ""
        self._drain_pending = False
        self.logs: deque[str] = deque(maxlen=self.MAX_LOG_LINES)
        self._kill_target = None
        self.kill_timer = QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.setInterval(3500)
        self.kill_timer.timeout.connect(self._force_stop)
        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.poll)

    @property
    def owned(self):
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def _redact(self, text: str) -> str:
        for key in self._secret_history:
            if key:
                text = text.replace(key, "[gizlendi]")
        return re.sub(r"(?i)(Bearer\s+|NEXUS_API_KEY\s*[=:]\s*)[^\s\",;]+", r"\1[gizlendi]", text)

    def log(self, text: str):
        for line in self._redact(text).splitlines():
            line = line[:self.MAX_LINE_CHARS]
            formatted = f"{datetime.now():%H:%M:%S}  {line}"
            self.logs.append(formatted)
            self.log_received.emit(formatted)

    def _fail(self, message: str):
        message = self._redact(str(message))
        self.detail = message
        self.log(message)
        self.error.emit(message)
        self.changed.emit()

    def start(self, settings: Settings):
        if self.owned or self.probing or self.stopping:
            return
        binary = settings.binary.strip() or discover_binary()
        resolved = shutil.which(binary) if binary and not Path(binary).is_absolute() else binary
        if not resolved or not Path(resolved).is_file() or not os.access(resolved, os.X_OK):
            self._fail("nexus-server bulunamadı. Sunucu paketini kurun veya çalıştırılabilir dosyayı seçin.")
            return
        if settings.model and not Path(settings.model).is_file():
            self._fail("Seçilen model dosyası bulunamadı. Geçerli bir GGUF dosyası seçin.")
            return
        if settings.backend not in {"cpu", "cuda"} or settings.kv_quant not in {"none", "tq_3b2b", "tq_3b4b", "tq_4b4b", "block_q4", "block_q6"}:
            self._fail("Desteklenmeyen backend veya KV biçimi.")
            return
        if not 1 <= settings.port <= 65535 or settings.context < 1 or not 1 <= settings.batch_size <= 16:
            self._fail("Port, context ve eşzamanlı istek değerlerini kontrol edin.")
            return
        self.disconnect()
        self._settings = replace(settings, binary=str(Path(resolved).resolve()))
        base = f"http://127.0.0.1:{settings.port}"
        self.api.configure(base)
        self.probing = True
        self.state, self.detail = "Port kontrol ediliyor", base
        self.changed.emit()

        def checked(status, _data, error, network_error):
            self.probing = False
            # Refuse to take over an existing endpoint, even if it returns 401/404.
            if status or network_error != QNetworkReply.NetworkError.ConnectionRefusedError:
                self.state = "Başlatılamadı"
                self._fail("Bu port kullanılıyor veya yanıt vermiyor. Var olan sunucuya bağlanın ya da başka port seçin.")
                return
            self._spawn(self._settings)
        self.api.request("probe", "/health", checked, authenticate=False, timeout=1800)

    def _spawn(self, settings: Settings):
        key = secrets.token_urlsafe(32)
        self._secret_history.append(key)
        self.api.configure(f"http://127.0.0.1:{settings.port}", key)
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("NEXUS_API_KEY", key)
        environment.remove("NEXUS_RATE_LIMIT_PER_MINUTE")
        environment.remove("NEXUS_CORS_ORIGIN")
        process.setProcessEnvironment(environment)
        args = ["--host", "127.0.0.1", "--port", str(settings.port), "--backend", settings.backend,
                "--kv-quant", settings.kv_quant, "--ctx-size", str(settings.context),
                "--batch-size", str(settings.batch_size), "--weight-quant", "f32"]
        if settings.model:
            args.extend(["--model", settings.model])
        process.setProgram(settings.binary)
        process.setArguments(args)
        self.process = process
        self.mode, self.state = "owned", "Başlatılıyor"
        self.detail = "Sunucu hazırlanıyor. Model yüklenirken günlükleri izleyebilirsiniz."
        self.starting, self.stopping = True, False
        self.authorized = True
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._tail = ""
        process.readyReadStandardOutput.connect(lambda: self._drain(process))
        process.started.connect(self.poll)
        process.finished.connect(lambda code, _status: self._finished(process, code))
        process.errorOccurred.connect(lambda error: self._process_error(process, error))
        self.log(f"Yerel sunucu başlatılıyor · {settings.backend.upper()} · port {settings.port}")
        process.start()
        self.timer.start()
        self.changed.emit()

    def _drain(self, process):
        if process is not self.process:
            return
        text = self._decoder.decode(bytes(process.read(65536)))
        # Cap partial as well as complete lines; a single giant line cannot grow the tail.
        text = (self._tail + text).replace("\r", "\n")
        lines = text.split("\n")
        self._tail = lines.pop()[-self.MAX_LINE_CHARS:]
        if lines:
            self.log("\n".join(lines))
        if process.bytesAvailable() and not self._drain_pending:
            self._drain_pending = True
            def again():
                self._drain_pending = False
                self._drain(process)
            QTimer.singleShot(0, again)

    def _process_error(self, process, error):
        if process is not self.process:
            return
        if error == QProcess.ProcessError.FailedToStart:
            self._fail("Sunucu çalıştırılamadı: " + process.errorString())
            self._finished(process, -1)

    def _finished(self, process, code):
        if process is not self.process:
            return
        self._drain(process)
        if self._tail:
            self.log(self._tail)
            self._tail = ""
        self.kill_timer.stop()
        self._kill_target = None
        self.process = None
        process.deleteLater()
        self.api.cancel()
        self.api.key = ""
        self.timer.stop()
        expected_stop = self.stopping
        self.starting = self.stopping = False
        self.busy = False
        self.mode = "disconnected"
        self.state = "Kapalı" if expected_stop or code == 0 else "Sunucu hatası"
        self.detail = f"Sunucu kapandı (çıkış kodu {code})."
        if not expected_stop and code != 0:
            self.detail += " Ayrıntılar için sunucu günlüğünü inceleyin."
            self.error.emit(self.detail)
        self.health, self.models, self.metrics, self.model_info = {}, [], {}, {}
        self.log(self.detail)
        restart, self._restart = self._restart, None
        self.changed.emit()
        self.stopped.emit()
        if restart:
            QTimer.singleShot(0, lambda: self.start(restart))

    def stop(self):
        if self.probing:
            self.api.cancel()
            self.probing = False
            self.state = "Bağlı değil"
            self.changed.emit()
        process = self.process
        if not self.owned or self.stopping:
            return
        self.stopping = True
        self.starting = False
        self.state, self.detail = "Durduruluyor", "Bu uygulamanın başlattığı sunucu kapatılıyor."
        self.timer.stop()
        self.api.cancel()
        self._kill_target = process
        process.terminate()
        self.kill_timer.start()
        self.changed.emit()

    def _force_stop(self):
        if self._kill_target is self.process and self.owned:
            self.log("Sunucu kapanışı gecikti; uygulamaya ait işlem sonlandırılıyor.")
            self.process.kill()

    def restart(self, settings: Settings):
        if not self.owned:
            return
        self._restart = replace(settings)
        self.stop()

    def connect_external(self, base: str, key=""):
        if self.owned or self.probing:
            self._fail("Önce bu uygulamanın sunucusunu durdurun.")
            return
        try:
            base = local_url(base)
        except ValueError as error:
            self._fail(str(error))
            return
        self.disconnect()
        self.api.configure(base, key)
        if key:
            self._secret_history.append(key)
        self.mode, self.state = "external", "Bağlanıyor"
        self.detail = "Var olan yerel sunucu kontrol ediliyor."
        self.authorized = True
        self.timer.start()
        self.changed.emit()
        self.poll()

    def disconnect(self):
        if self.owned:
            return
        self.api.cancel()
        self.api.key = ""
        self.timer.stop()
        self.health, self.models, self.metrics, self.model_info = {}, [], {}, {}
        self.mode, self.state = "disconnected", "Bağlı değil"
        self.detail = "Bir sunucu başlatın veya yerel sunucuya bağlanın."
        self.busy = False
        self.changed.emit()

    def poll(self):
        if self.mode == "disconnected" or self.stopping:
            return
        self.api.request("health", "/health", self._health)
        for tag, path in [("models", "/v1/models"), ("metrics", "/metrics"), ("model", "/v1/model")]:
            self.api.request(tag, path, lambda status, data, error, _network, tag=tag: self._data(tag, status, data, error))

    def _health(self, status, data, error, _network):
        if status == 200 and isinstance(data, dict) and isinstance(data.get("model_loaded"), bool):
            self.starting = False
            self.health = data
            if not isinstance(self.health.get("model"), str):
                self.health["model"] = ""
            self.state = "Hazır" if data.get("model_loaded") else "Model bekleniyor"
            if data.get("status") == "error":
                self.state = "Sunucu hatası"
            if not self.busy:
                self.detail = self.api.base + (" · Uygulama tarafından yönetiliyor" if self.mode == "owned" else " · Haricî sunucu; yalnız bağlantı yönetilir")
            if not self.authorized:
                self.state = "API anahtarı gerekli"
                self.detail = "Sunucuya erişiliyor, korunan API için geçerli anahtar girin."
        elif not self.starting:
            self.health = {}
            self.state = "Bağlantı yok"
            self.detail = "Yerel sunucuya ulaşılamıyor. Adres ve portu kontrol edin."
        self.changed.emit()

    def _data(self, tag, status, data, error):
        if status in {401, 403}:
            self.authorized = False
            self.state = "API anahtarı gerekli"
            self.detail = "Sunucuya erişiliyor, korunan API için geçerli anahtar girin."
        elif status == 200 and isinstance(data, dict):
            if tag == "models":
                models = data.get("data", [])
                self.models = [model for model in models if isinstance(model, dict) and isinstance(model.get("id"), str)] if isinstance(models, list) else []
            elif tag == "metrics":
                self.metrics = data
                for key in ["runtime", "engine", "scheduler"]:
                    if not isinstance(self.metrics.get(key), dict):
                        self.metrics[key] = {}
            elif tag == "model":
                self.model_info = data
                if not isinstance(self.model_info.get("path"), str):
                    self.model_info["path"] = None
        self.changed.emit()

    def model_action(self, action: str, settings: Settings):
        if action not in {"load", "reload", "unload"} or self.mode == "disconnected" or self.busy or self.starting or self.stopping:
            return
        payload = {}
        if action == "load":
            if not settings.model or not Path(settings.model).is_file():
                self._fail("Önce geçerli bir GGUF model dosyası seçin.")
                return
            payload = {"model": settings.model, "ctx_size": settings.context, "kv_quant": settings.kv_quant, "backend": settings.backend, "weight_quant": "f32"}
        self.busy = True
        self.detail = {"load":"Model yükleniyor…", "reload":"Model yeniden yükleniyor…", "unload":"Model kaldırılıyor…"}[action]
        self.changed.emit()
        def finished(status, data, error, _network):
            self.busy = False
            if 200 <= status < 300:
                self.log({"load":"Model yüklendi.","reload":"Model yeniden yüklendi.","unload":"Model bellekten kaldırıldı."}[action])
                self.detail = "Model işlemi tamamlandı."
            else:
                problem = data.get("error", {}) if isinstance(data, dict) else {}
                message = problem.get("message", "") if isinstance(problem, dict) else str(problem)
                self._fail(message or error or f"Model işlemi başarısız (HTTP {status}).")
            self.changed.emit()
            self.poll()
        self.api.request("action", "/v1/model/" + action, finished, payload, timeout=120000)
