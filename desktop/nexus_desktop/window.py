"""Turkish Qt Widgets dashboard. No synchronous HTTP/process waits."""
from __future__ import annotations
import json
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QSpinBox, QFileDialog, QFrame, QPlainTextEdit, QTabWidget, QGridLayout,
    QScrollArea, QSizePolicy, QBoxLayout,
)
from .config import Settings, discover_binary, save_settings
from .controller import ServerController


STYLE = """
QWidget {font-family: 'Inter','Noto Sans','DejaVu Sans'; font-size:13px; color:#192339;}
QMainWindow, #mainContent {background:#f3f5fa;}
#sidebar {background:#111b31; border:0;}
#brand {color:#fff; font-size:25px; font-weight:700;}
#brandTag {color:#93a4c7; font-size:10px; font-weight:600; letter-spacing:2px;}
#nav {background:#25324e; color:#eff4ff; border:0; text-align:left; padding:13px 17px; border-radius:9px; font-weight:600;}
#sideText {color:#9aaccd; font-size:12px;}
#pageTitle {font-size:26px; font-weight:700; color:#14213d;}
#subtitle {color:#768199; font-size:13px;}
#status {background:#e7ebf3; color:#53617b; padding:9px 15px; border-radius:16px; font-weight:600;}
#status[ready="true"] {background:#e2f4ee; color:#217859;}
#card {background:white; border:1px solid #e2e7f0; border-radius:13px;}
#cardTitle {font-size:15px; font-weight:700; color:#1e2b45;}
#fieldLabel {font-size:11px; font-weight:600; color:#73819b;}
#metricValue {font-size:20px; font-weight:700; color:#172745;}
#metricHint {font-size:11px; color:#8792a8;}
QLineEdit,QComboBox,QSpinBox {background:#fafbfe; border:1px solid #dce2ed; border-radius:7px; min-height:19px; padding:9px 10px; selection-background-color:#4776ef;}
QLineEdit:focus,QComboBox:focus,QSpinBox:focus {border:1px solid #4776ef; background:white;}
QComboBox {padding-right:28px;}
QComboBox::drop-down {border:0; width:25px;}
QComboBox::down-arrow {image:url("ASSETS/chevron-down.svg"); width:12px; height:8px;}
QSpinBox {padding-right:28px;}
QSpinBox::up-button {subcontrol-origin:border; subcontrol-position:top right; width:24px; height:19px; border:0; border-left:1px solid #dce2ed; background:#f0f4fb; border-top-right-radius:7px;}
QSpinBox::down-button {subcontrol-origin:border; subcontrol-position:bottom right; width:24px; height:19px; border:0; border-left:1px solid #dce2ed; background:#f0f4fb; border-bottom-right-radius:7px;}
QSpinBox::up-arrow {image:url("ASSETS/chevron-up.svg"); width:10px; height:7px;}
QSpinBox::down-arrow {image:url("ASSETS/chevron-down.svg"); width:10px; height:7px;}
QSpinBox::up-button:hover,QSpinBox::down-button:hover {background:#e0e9fa;}
QPushButton {background:#fff; border:1px solid #dbe2ef; border-radius:7px; padding:10px 14px; font-weight:600;}
QPushButton:hover {background:#f0f4fd; border-color:#b6c8ef;}
QPushButton:pressed {background:#e5ecfa;}
QPushButton:disabled {color:#a4adbf; background:#f6f8fb; border-color:#e9edf4;}
QPushButton[primary="true"] {background:#3e6dea; border-color:#3e6dea; color:white;}
QPushButton[primary="true"]:hover {background:#315ed4;}
QPushButton[primary="true"]:disabled {background:#b9caf5; border-color:#b9caf5; color:#f4f7ff;}
QPushButton[danger="true"] {color:#b45d62;}
QPushButton[danger="true"]:disabled {color:#a4adbf;}
#notice {background:#fff1e4; color:#96632d; border:1px solid #f1dac0; border-radius:8px; padding:10px 13px;}
QTabWidget::pane {border:0; background:#fff;}
QTabBar::tab {background:#fff; color:#8994a8; padding:10px 14px; border-bottom:2px solid transparent;}
QTabBar::tab:selected {color:#3e6dea; border-bottom-color:#3e6dea;}
#logView {background:#152039; color:#bbcae6; border:0; border-radius:8px; padding:12px; font-family:'JetBrains Mono','DejaVu Sans Mono'; font-size:11px;}
#metricView {background:#f8faff; border:0; padding:10px; font-family:'DejaVu Sans Mono'; font-size:11px;}
QScrollArea {border:0; background:transparent;}
QScrollBar:vertical {width:8px; background:transparent;}
QScrollBar::handle:vertical {background:#d3dbe9; border-radius:4px; min-height:25px;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical {height:0;}
"""


def label(text, name=""):
    result = QLabel(text)
    if name:
        result.setObjectName(name)
    return result


def card(title: str):
    frame = QFrame()
    frame.setObjectName("card")
    box = QVBoxLayout(frame)
    box.setContentsMargins(20, 18, 20, 18)
    box.setSpacing(12)
    if title:
        box.addWidget(label(title, "cardTitle"))
    return frame, box


def field(box, title, widget):
    box.addWidget(label(title, "fieldLabel"))
    box.addWidget(widget)


def pair(box, a_title, a, b_title, b):
    row = QHBoxLayout()
    row.setSpacing(14)
    for title, widget in [(a_title, a), (b_title, b)]:
        column = QVBoxLayout()
        column.setSpacing(6)
        field(column, title, widget)
        row.addLayout(column, 1)
    box.addLayout(row)


def bytes_label(value):
    if not isinstance(value, (int, float)):
        return "—"
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024
    return "—"


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, controller: ServerController | None = None):
        super().__init__()
        self.controller = controller or ServerController(self)
        self._closing = False
        self.setWindowTitle("Nexus Desktop · Yerel sunucu yönetimi")
        icon_path = Path(__file__).parent / "assets/nexus.svg"
        self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1230, 900)
        self.setMinimumSize(980, 720)
        self.setStyleSheet(STYLE.replace("ASSETS", str(Path(__file__).parent / "assets")))
        container = QWidget()
        root = QHBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(container)
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(214)
        side_box = QVBoxLayout(side)
        side_box.setContentsMargins(24, 30, 22, 26)
        logo = QLabel()
        logo.setPixmap(QIcon(str(icon_path)).pixmap(40, 40))
        side_box.addWidget(logo)
        side_box.addSpacing(10)
        side_box.addWidget(label("nexus", "brand"))
        side_box.addWidget(label("DESKTOP", "brandTag"))
        side_box.addSpacing(36)
        nav = QPushButton("◉   Sunucu yönetimi")
        nav.setObjectName("nav")
        side_box.addWidget(nav)
        side_box.addStretch()
        for text in ["YEREL ÇALIŞMA ALANI", "Modelleriniz bu cihazda.\nKontrol her zaman sizde."]:
            item = label(text, "sideText")
            item.setWordWrap(True)
            side_box.addWidget(item)
        side_box.addSpacing(20)
        side_box.addWidget(label("Nexus Desktop  0.1", "sideText"))
        root.addWidget(side)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("mainContent")
        scroll.setWidget(content)
        root.addWidget(scroll, 1)
        body = QVBoxLayout(content)
        body.setContentsMargins(28, 27, 28, 24)
        body.setSpacing(18)
        heading = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(6)
        title_box.addWidget(label("Sunucu yönetimi", "pageTitle"))
        title_box.addWidget(label("Modellerinizi bu bilgisayarda çalıştırın ve yönetin.", "subtitle"))
        heading.addLayout(title_box)
        heading.addStretch()
        self.status = label("Bağlı değil", "status")
        heading.addWidget(self.status, 0, Qt.AlignmentFlag.AlignTop)
        body.addLayout(heading)
        self.notice = label("", "notice")
        self.notice.setWordWrap(True)
        self.notice.hide()
        body.addWidget(self.notice)
        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.metric_values, self.metric_hints = [], []
        for title, value, hint in [("MODEL", "Yüklü değil", "GGUF modeli bekleniyor"),
                                   ("SUNUCU", "Kapalı", "127.0.0.1"),
                                   ("ETKİN İSTEK", "—", "Kuyruk bilgisi bekleniyor"),
                                   ("MODEL BELLEĞİ", "—", "Host ağırlıkları · F32")]:
            frame, box = card("")
            box.setContentsMargins(16, 14, 16, 14)
            box.setSpacing(5)
            box.addWidget(label(title, "fieldLabel"))
            value_label = label(value, "metricValue")
            value_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            hint_label = label(hint, "metricHint")
            box.addWidget(value_label)
            box.addWidget(hint_label)
            self.metric_values.append(value_label)
            self.metric_hints.append(hint_label)
            metrics.addWidget(frame, 1)
        body.addLayout(metrics)
        middle = QHBoxLayout()
        self.middle = middle
        middle.setSpacing(16)
        settings_card, settings_box = card("Çalıştırma ayarları")
        model_row = QHBoxLayout()
        self.model = QLineEdit(settings.model)
        self.model.setObjectName("modelPath")
        self.model.setPlaceholderText("Bir .gguf model dosyası seçin")
        browse = QPushButton("Seç…")
        browse.clicked.connect(self.browse_model)
        model_row.addWidget(self.model, 1)
        model_row.addWidget(browse)
        settings_box.addWidget(label("MODEL DOSYASI", "fieldLabel"))
        settings_box.addLayout(model_row)
        binary_row = QHBoxLayout()
        self.binary = QLineEdit(settings.binary or discover_binary())
        self.binary.setObjectName("binaryPath")
        self.binary.setPlaceholderText("nexus-server ikilisi veya PATH komutu")
        choose_binary = QPushButton("Seç…")
        choose_binary.clicked.connect(self.browse_binary)
        binary_row.addWidget(self.binary, 1)
        binary_row.addWidget(choose_binary)
        settings_box.addWidget(label("SUNUCU İKİLİSİ", "fieldLabel"))
        settings_box.addLayout(binary_row)
        self.backend = QComboBox()
        self.backend.addItem("CPU", "cpu")
        self.backend.addItem("CUDA · NVIDIA", "cuda")
        self.backend.setCurrentIndex(max(0, self.backend.findData(settings.backend)))
        self.backend.setToolTip("CUDA için CUDA destekli nexus-server ve NVIDIA sürücüsü gerekir.")
        self.kv = QComboBox()
        for text, data in [("F32 · tam hassasiyet", "none"), ("K:3 bit / V:2 bit", "tq_3b2b"), ("K:3 bit / V:4 bit", "tq_3b4b"), ("K:4 bit / V:4 bit", "tq_4b4b"), ("Block Q4", "block_q4"), ("Block Q6", "block_q6")]:
            self.kv.addItem(text, data)
        self.kv.setCurrentIndex(max(0, self.kv.findData(settings.kv_quant)))
        pair(settings_box, "HESAPLAMA", self.backend, "KV ÖNBELLEĞİ", self.kv)
        self.context = QSpinBox()
        self.context.setRange(1, 1048576)
        self.context.setValue(settings.context)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(settings.port)
        pair(settings_box, "CONTEXT · TOKEN", self.context, "YEREL PORT", self.port)
        self.batch = QSpinBox()
        self.batch.setRange(1, 16)
        self.batch.setValue(settings.batch_size)
        batch_row = QHBoxLayout()
        batch_row.addWidget(label("Eşzamanlı istek", "fieldLabel"))
        batch_row.addStretch()
        batch_row.addWidget(self.batch)
        settings_box.addLayout(batch_row)
        actions = QHBoxLayout()
        self.start_button = QPushButton("▶  Sunucuyu başlat")
        self.start_button.setObjectName("startButton")
        self.start_button.setProperty("primary", True)
        self.stop_button = QPushButton("Durdur")
        self.stop_button.setProperty("danger", True)
        self.stop_button.setObjectName("stopButton")
        self.restart_button = QPushButton("Yeniden başlat")
        self.start_button.clicked.connect(self.start_server)
        self.stop_button.clicked.connect(self.controller.stop)
        self.restart_button.clicked.connect(lambda: self.controller.restart(self.settings()))
        for button in [self.start_button, self.stop_button, self.restart_button]:
            actions.addWidget(button)
        settings_box.addLayout(actions)
        settings_box.addStretch()
        middle.addWidget(settings_card, 3)
        right = QVBoxLayout()
        model_card, model_box = card("Model işlemleri")
        self.loaded_model = label("Henüz bir model yüklü değil.", "subtitle")
        self.loaded_model.setWordWrap(True)
        model_box.addWidget(self.loaded_model)
        model_actions = QHBoxLayout()
        self.load_button = QPushButton("Yükle")
        self.reload_button = QPushButton("Yenile")
        self.reload_button.setToolTip("Yüklü modeli aynı dosyadan yeniden yükle.")
        self.unload_button = QPushButton("Kaldır")
        self.unload_button.setProperty("danger", True)
        for button, action in [(self.load_button, "load"), (self.reload_button, "reload"), (self.unload_button, "unload")]:
            button.clicked.connect(lambda _checked=False, action=action: self.run_model_action(action))
            model_actions.addWidget(button)
        model_box.addLayout(model_actions)
        self.operation_hint = label("Sunucu hazır olduğunda modeli değiştirebilirsiniz.", "metricHint")
        self.operation_hint.setWordWrap(True)
        model_box.addWidget(self.operation_hint)
        self.copy_key_button = QPushButton("API anahtarını kopyala")
        self.copy_key_button.setToolTip("Bu uygulamanın sunucusuna istemci bağlamak için oturum anahtarını kopyalayın.")
        self.copy_key_button.clicked.connect(self.copy_key)
        model_box.addWidget(self.copy_key_button)
        right.addWidget(model_card)
        connection_card, connection_box = card("Var olan sunucuya bağlan")
        self.url = QLineEdit(settings.server_url)
        self.url.setObjectName("serverUrl")
        field(connection_box, "YEREL API ADRESİ", self.url)
        self.key = QLineEdit()
        self.key.setObjectName("apiKey")
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("API anahtarı · isteğe bağlı")
        self.key.setToolTip("Yalnız bu oturumda kullanılır; diske kaydedilmez.")
        connection_box.addWidget(self.key)
        connection_actions = QHBoxLayout()
        self.connect_button = QPushButton("Bağlan")
        self.disconnect_button = QPushButton("Bağlantıyı kes")
        self.connect_button.clicked.connect(self.connect_server)
        self.disconnect_button.clicked.connect(self.controller.disconnect)
        connection_actions.addWidget(self.connect_button)
        connection_actions.addWidget(self.disconnect_button)
        connection_box.addLayout(connection_actions)
        hint = label("Haricî sunucu durdurulmaz. API anahtarı yalnız bu oturumda tutulur.", "metricHint")
        hint.setWordWrap(True)
        connection_box.addWidget(hint)
        right.addWidget(connection_card)
        right.addStretch()
        middle.addLayout(right, 2)
        body.addLayout(middle)
        logs_card, log_box = card("")
        logs_title = QHBoxLayout()
        logs_title.addWidget(label("Sunucu günlüğü", "cardTitle"))
        logs_title.addStretch()
        logs_title.addWidget(label("Son 600 satır", "metricHint"))
        clear = QPushButton("Temizle")
        clear.clicked.connect(self.clear_logs)
        clear.setMaximumHeight(30)
        logs_title.addWidget(clear)
        log_box.addLayout(logs_title)
        tabs = QTabWidget()
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(self.controller.MAX_LOG_LINES)
        self.log_view.setPlaceholderText("Sunucu başlatıldığında günlükler burada görünür.")
        self.metric_view = QPlainTextEdit()
        self.metric_view.setObjectName("metricView")
        self.metric_view.setReadOnly(True)
        self.metric_view.setMaximumBlockCount(500)
        tabs.addTab(self.log_view, "Günlükler")
        tabs.addTab(self.metric_view, "Metrikler")
        tabs.setMinimumHeight(190)
        log_box.addWidget(tabs)
        body.addWidget(logs_card, 1)
        self.connection_detail = label("", "subtitle")
        self.connection_detail.setWordWrap(True)
        body.addWidget(self.connection_detail)
        self.controller.changed.connect(self.refresh)
        self.controller.log_received.connect(self.log_view.appendPlainText)
        self.controller.error.connect(self.show_error)
        self.controller.stopped.connect(self._after_stopped)
        nav.clicked.connect(lambda: scroll.verticalScrollBar().setValue(0))
        self.refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "middle"):
            direction = QBoxLayout.Direction.TopToBottom if self.width() < 1120 else QBoxLayout.Direction.LeftToRight
            self.middle.setDirection(direction)

    def settings(self):
        return Settings(binary=self.binary.text().strip(), model=self.model.text().strip(),
                        backend=self.backend.currentData(), kv_quant=self.kv.currentData(),
                        context=self.context.value(), port=self.port.value(), batch_size=self.batch.value(),
                        server_url=self.url.text().strip())

    def connect_server(self):
        self.notice.hide()
        self.controller.connect_external(self.url.text(), self.key.text())

    def run_model_action(self, action):
        self.notice.hide()
        self.controller.model_action(action, self.settings())

    def copy_key(self):
        if self.controller.owned and self.controller.api.key:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(self.controller.api.key)
            self.connection_detail.setText("API anahtarı panoya kopyalandı; yalnız bu sunucu oturumunda geçerlidir.")

    def start_server(self):
        self.notice.hide()
        self.controller.start(self.settings())

    def browse_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "GGUF model seçin", str(Path.home()), "GGUF modelleri (*.gguf);;Tüm dosyalar (*)")
        if filename:
            self.model.setText(filename)

    def browse_binary(self):
        filename, _ = QFileDialog.getOpenFileName(self, "nexus-server ikilisini seçin", "/usr/bin", "Tüm dosyalar (*)")
        if filename:
            self.binary.setText(filename)

    def show_error(self, message):
        self.notice.setText(message)
        self.notice.show()

    def clear_logs(self):
        self.controller.logs.clear()
        self.log_view.clear()

    def refresh(self):
        c = self.controller
        self.status.setText("Model işlemi sürüyor" if c.busy else c.state)
        self.status.setProperty("ready", bool(c.health) and c.health.get("status") != "error" and c.authorized and not c.busy)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.connection_detail.setText(c.detail)
        model = c.health.get("model") or (c.models[0].get("id") if c.models else "")
        loaded = bool(c.health.get("model_loaded"))
        model_text = model if loaded else "Yüklü değil"
        self.metric_values[0].setText(self.metric_values[0].fontMetrics().elidedText(model_text, Qt.TextElideMode.ElideRight, max(80, self.metric_values[0].width())))
        self.metric_values[0].setToolTip(model)
        self.metric_hints[0].setText("Model kullanıma hazır" if loaded else "GGUF modeli bekleniyor")
        self.loaded_model.setText((model + "\n" + (c.model_info.get("path") or "")) if loaded else "Henüz bir model yüklü değil.")
        process = c.process
        self.metric_values[1].setText("Yerel işlem" if c.owned else "Haricî" if c.mode == "external" else "Kapalı")
        self.metric_hints[1].setText(f"PID {process.processId()} · port {c.api.base.rsplit(':',1)[-1]}" if c.owned else c.api.base.replace("http://", "") if c.mode == "external" else "127.0.0.1")
        runtime = c.metrics.get("runtime", {})
        self.metric_values[2].setText(str(runtime.get("active_sequences", "—")))
        self.metric_hints[2].setText(f"{runtime.get('queued_requests', 0)} istek kuyrukta")
        self.metric_values[3].setText(bytes_label(c.metrics.get("engine", {}).get("host_weight_payload_bytes")))
        self.metric_view.setPlainText(json.dumps(c.metrics, ensure_ascii=False, indent=2)[:65536] if c.metrics else "Metrik bekleniyor…")
        self.start_button.setEnabled(not c.owned and not c.probing and not c.stopping)
        self.stop_button.setEnabled(c.owned and not c.stopping)
        self.restart_button.setEnabled(c.owned and not c.stopping and not c.starting)
        self.connect_button.setEnabled(not c.owned and not c.probing)
        self.disconnect_button.setEnabled(c.mode == "external")
        available = bool(c.health) and c.authorized and not c.busy and not c.starting and not c.stopping
        self.copy_key_button.setVisible(c.owned)
        self.copy_key_button.setEnabled(c.owned and bool(c.api.key) and not c.stopping)
        self.load_button.setEnabled(available)
        self.reload_button.setEnabled(available and bool(c.model_info.get("path")))
        self.unload_button.setEnabled(available and loaded)
        self.operation_hint.setText("İşlem sürüyor; günlükleri izleyebilirsiniz." if c.busy else "Modeli kaldırmak sunucuyu kapatmaz.")

    def closeEvent(self, event):
        try:
            save_settings(self.settings())
        except OSError:
            pass
        if self.controller.owned:
            self._closing = True
            self.controller._restart = None
            self.controller.stop()
            event.ignore()
            return
        self.controller.stop()  # Cancels a pending start-port probe, never external processes.
        self.controller.disconnect()
        event.accept()

    def _after_stopped(self):
        if self._closing:
            QTimer.singleShot(0, self.close)
