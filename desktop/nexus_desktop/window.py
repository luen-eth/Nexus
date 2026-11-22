"""Qt Widgets main window: Chat, Models and Server. No synchronous HTTP/process waits."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import traceback
from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QBoxLayout, QButtonGroup, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QStackedWidget, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)
from . import __version__
from .config import Settings, discover_binary, save_settings
from .chat import ChatPanel
from .controller import ServerController
from .hub import ModelHubPanel
from .theme import COLORS, build_stylesheet, nav_icon, set_property

ASSETS = Path(__file__).parent / "assets"
STYLE = build_stylesheet(str(ASSETS))


# --------------------------------------------------------------------- helpers

def label(text, name=""):
    """Plain-text label: server errors, model names and file names are never parsed as HTML."""
    result = QLabel(text)
    result.setTextFormat(Qt.TextFormat.PlainText)
    if name:
        result.setObjectName(name)
    return result


def card(title: str, subtitle: str = ""):
    frame = QFrame()
    frame.setObjectName("card")
    box = QVBoxLayout(frame)
    box.setContentsMargins(22, 20, 22, 20)
    box.setSpacing(12)
    if title:
        heading = QVBoxLayout()
        heading.setSpacing(3)
        heading.addWidget(label(title, "cardTitle"))
        if subtitle:
            sub = label(subtitle, "cardSubtitle")
            sub.setWordWrap(True)
            heading.addWidget(sub)
        box.addLayout(heading)
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


def page_header(title: str, subtitle: str) -> QVBoxLayout:
    box = QVBoxLayout()
    box.setSpacing(4)
    box.addWidget(label(title, "pageTitle"))
    sub = label(subtitle, "pageSubtitle")
    sub.setWordWrap(True)
    box.addWidget(sub)
    return box


def callout(tone: str = "neutral"):
    """Title + text block used for mode explanations and instructions."""
    frame = QFrame()
    frame.setObjectName("callout")
    frame.setProperty("tone", tone)
    row = QHBoxLayout(frame)
    row.setContentsMargins(18, 14, 14, 14)
    row.setSpacing(14)
    text = QVBoxLayout()
    text.setSpacing(3)
    title = label("", "calloutTitle")
    body = label("", "calloutText")
    body.setWordWrap(True)
    text.addWidget(title)
    text.addWidget(body)
    row.addLayout(text, 1)
    return frame, title, body, row


class ElidedLabel(QLabel):
    """Keeps the full text (text()) but paints an elided version that tracks the width."""

    _full = ""  # Class defaults: Qt may deliver change events during base-class construction.
    _mode = Qt.TextElideMode.ElideRight

    def __init__(self, text="", mode=Qt.TextElideMode.ElideRight, parent=None):
        super().__init__(parent)
        self._mode = mode
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def text(self):
        return self._full

    def setText(self, text):
        self._full = str(text)
        self._apply()

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._apply()

    def _apply(self):
        width = max(self.contentsRect().width(), 40)
        metrics = self.fontMetrics()
        shown = "\n".join(metrics.elidedText(line, self._mode, width) for line in self._full.split("\n"))
        if shown != QLabel.text(self):
            QLabel.setText(self, shown)


class NoticeLabel(QLabel):
    """Dismissible error banner. Still a plain QLabel for callers (text/show/hide)."""
    visibility_changed = Signal()

    def __init__(self):
        super().__init__("")
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setObjectName("notice")
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setAccessibleName("Error message")
        self.close_button = QToolButton(self)
        self.close_button.setObjectName("noticeClose")
        self.close_button.setText("×")
        self.close_button.setToolTip("Dismiss")
        self.close_button.setAccessibleName("Dismiss message")
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.setFixedSize(28, 28)
        self.close_button.clicked.connect(self.hide)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.close_button.move(self.width() - 38, 8)

    def event(self, event):
        result = super().event(event)
        if event.type() in (QEvent.Type.ShowToParent, QEvent.Type.HideToParent):
            self.visibility_changed.emit()
        return result


class StateBanner(QFrame):
    """One-line guidance shown above Chat/Models when the server cannot serve them yet."""
    visibility_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("stateBanner")
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 10, 10, 10)
        row.setSpacing(12)
        self.message = label("", "stateBannerText")
        self.message.setWordWrap(True)
        self.button = QPushButton("")
        self.button.setObjectName("bannerButton")
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addWidget(self.message, 1)
        row.addWidget(self.button, 0, Qt.AlignmentFlag.AlignVCenter)

    def event(self, event):
        result = super().event(event)
        if event.type() in (QEvent.Type.ShowToParent, QEvent.Type.HideToParent):
            self.visibility_changed.emit()
        return result


# ----------------------------------------------------------------- main window

class MainWindow(QMainWindow):
    PAGES = ("chat", "models", "server")
    SIDEBAR_WIDTH = 224
    STACK_BELOW = 920  # Server page width under which its two columns stack.

    def __init__(self, settings: Settings, controller: ServerController | None = None):
        super().__init__()
        self.controller = controller or ServerController(self)
        self._closing = False
        self._panels_closed = False
        self._pending_model = ""
        self._initial_models_dir = settings.models_dir or ""
        self.setWindowTitle("Nexus Desktop")
        icon_path = ASSETS / "nexus.svg"
        self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1260, 860)
        self.setMinimumSize(960, 720)
        self.setStyleSheet(STYLE)
        container = QWidget()
        root = QHBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(container)
        root.addWidget(self._build_sidebar(icon_path))
        column = QWidget()
        column.setObjectName("content")
        column_box = QVBoxLayout(column)
        column_box.setContentsMargins(0, 0, 0, 0)
        column_box.setSpacing(0)
        root.addWidget(column, 1)
        # Global strip: errors and "what to do next" guidance, visible on every page.
        self.strip = QWidget()
        self.strip_layout = QVBoxLayout(self.strip)
        self.strip_layout.setContentsMargins(0, 0, 0, 0)
        self.strip_layout.setSpacing(10)
        self.notice = NoticeLabel()
        self.notice.hide()
        self.banner = StateBanner()
        self.banner.hide()
        self.strip_layout.addWidget(self.notice)
        self.strip_layout.addWidget(self.banner)
        column_box.addWidget(self.strip)
        self.pages = QStackedWidget()
        self.pages.setObjectName("pages")
        column_box.addWidget(self.pages, 1)
        self._banner_target = "server"
        self.banner.button.clicked.connect(lambda: self.select_page(self._banner_target))
        self.notice.visibility_changed.connect(self._sync_strip)
        self.banner.visibility_changed.connect(self._sync_strip)

        self.chat_panel = self._create_chat()
        self.hub_panel = self._create_hub()
        self.pages.addWidget(self._chat_page())
        self.pages.addWidget(self._models_page())
        self.pages.addWidget(self._build_server_page(settings))
        self.hub_panel.model_selected.connect(self.load_model_from_hub)
        self.hub_panel.model_deleted.connect(self.model_deleted_from_hub)
        self.hub_panel.deletion_state_changed.connect(self.refresh)
        self.pages.currentChanged.connect(self._page_changed)

        for number, key in enumerate(self.PAGES, 1):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{number}"), self)
            shortcut.activated.connect(lambda key=key: self.select_page(key))

        self.controller.changed.connect(self.refresh)
        self.controller.log_received.connect(self.log_view.appendPlainText)
        self.controller.error.connect(self.show_error)
        self.controller.stopped.connect(self._after_stopped)
        self.select_page("chat")
        self._update_responsive()
        self.refresh()

    # ------------------------------------------------------------ construction

    def _build_sidebar(self, icon_path: Path) -> QFrame:
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(self.SIDEBAR_WIDTH)
        box = QVBoxLayout(side)
        box.setContentsMargins(16, 24, 16, 18)
        box.setSpacing(0)
        brand = QHBoxLayout()
        brand.setContentsMargins(8, 0, 0, 0)
        brand.setSpacing(12)
        logo = QLabel()
        logo.setPixmap(QIcon(str(icon_path)).pixmap(36, 36))
        logo.setFixedSize(36, 36)
        brand.addWidget(logo)
        words = QVBoxLayout()
        words.setSpacing(0)
        words.addWidget(label("nexus", "brand"))
        tag = label("DESKTOP", "brandTag")
        tag_font = QFont(tag.font())
        tag_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.6)
        tag.setFont(tag_font)
        words.addWidget(tag)
        brand.addLayout(words)
        brand.addStretch()
        box.addLayout(brand)
        box.addSpacing(30)
        section = label("WORKSPACE", "sideSection")
        section.setContentsMargins(10, 0, 0, 0)
        box.addWidget(section)
        box.addSpacing(8)
        self.nav_buttons: dict[str, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for number, (key, text, hint) in enumerate([
            ("chat", "Chat", "Talk with the loaded model"),
            ("models", "Models", "Find, download and load models"),
            ("server", "Server", "Start, connect and monitor"),
        ], 1):
            button = QPushButton(text)
            button.setObjectName("navItem")
            button.setCheckable(True)
            button.setIcon(nav_icon(key, COLORS["side_text"], "#ffffff"))
            button.setIconSize(QSize(30, 20))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(f"{hint} (Ctrl+{number})")
            button.setAccessibleName(text)
            button.clicked.connect(lambda _checked=False, key=key: self.select_page(key))
            group.addButton(button)
            self.nav_buttons[key] = button
            box.addWidget(button)
            box.addSpacing(4)
        box.addStretch()
        status_card = QFrame()
        status_card.setObjectName("sideCard")
        card_box = QVBoxLayout(status_card)
        card_box.setContentsMargins(14, 12, 14, 12)
        card_box.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(9)
        self.status_dot = QLabel()
        self.status_dot.setObjectName("statusDot")
        self.status_dot.setFixedSize(10, 10)
        self.status = label("Disconnected", "status")
        self.status.setWordWrap(True)
        top.addWidget(self.status_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.status, 1)
        card_box.addLayout(top)
        self.side_mode = ElidedLabel("No server connected")
        self.side_mode.setObjectName("sideDetail")
        self.side_model = ElidedLabel("No model loaded", Qt.TextElideMode.ElideMiddle)
        self.side_model.setObjectName("sideModel")
        card_box.addWidget(self.side_mode)
        card_box.addWidget(self.side_model)
        status_card.setAccessibleName("Server status")
        box.addWidget(status_card)
        box.addSpacing(14)
        footer = label(f"Runs on this device · v{__version__}", "sideFooter")
        footer.setContentsMargins(4, 0, 0, 0)
        box.addWidget(footer)
        return side

    def _create_chat(self):
        return ChatPanel(self.controller)

    def _create_hub(self):
        return ModelHubPanel(self.controller, download_dir=self._initial_models_dir)

    def _chat_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("page")
        box = QVBoxLayout(page)
        box.setContentsMargins(28, 22, 28, 20)
        box.setSpacing(0)
        box.addWidget(self.chat_panel, 1)
        return page

    @staticmethod
    def _scroll_surface(object_name: str):
        """A vertically scrolling, horizontally fitted page: (scroll area, content layout).

        Content never gets squeezed below its minimum size; the page scrolls instead,
        so lists, details and actions cannot overlap at any window size.
        """
        scroll = QScrollArea()
        scroll.setObjectName(object_name)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        surface = QWidget()
        surface.setObjectName("pageSurface")
        scroll.setWidget(surface)
        outer = QHBoxLayout(surface)
        outer.setContentsMargins(28, 24, 28, 28)
        outer.addStretch(1)
        inner = QWidget()
        inner.setMaximumWidth(1120)
        outer.addWidget(inner, 100)
        outer.addStretch(1)
        body = QVBoxLayout(inner)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(16)
        return scroll, body

    def _models_page(self) -> QScrollArea:
        scroll, body = self._scroll_surface("modelsScroll")
        self.models_scroll = scroll
        # The inner widget's minimum size follows its content (header + whole hub panel).
        body.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        body.addLayout(page_header(
            "Models", "Find GGUF models on Hugging Face, download them to this computer and load them into your server."))
        body.addWidget(self.hub_panel, 1)
        return scroll

    def _build_server_page(self, settings: Settings) -> QScrollArea:
        scroll, body = self._scroll_surface("serverScroll")
        self.server_scroll = scroll
        body.addLayout(page_header(
            "Server", "Run Nexus on this computer, or connect to a server that is already running."))

        # What am I connected to, and what can I do about it?
        self.mode_note, self.mode_title, self.mode_text, _row = callout("neutral")
        body.addWidget(self.mode_note)
        # Set when the Models page selected a file but the server could not load it yet.
        self.instruction, self.instruction_title, self.instruction_text, instruction_row = callout("info")
        self.instruction_button = QPushButton("Load model")
        self.instruction_button.setProperty("primary", True)
        self.instruction_button.clicked.connect(lambda: self.run_model_action("load"))
        instruction_row.addWidget(self.instruction_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.instruction.hide()
        body.addWidget(self.instruction)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.metric_values, self.metric_hints = [], []
        for title, value, hint in [("Model", "Not loaded", "Waiting for a GGUF model"),
                                   ("Server", "Stopped", "127.0.0.1"),
                                   ("Active requests", "—", "Waiting for queue data"),
                                   ("Model memory", "—", "Host weights · F32")]:
            frame, box = card("")
            box.setContentsMargins(16, 14, 16, 14)
            box.setSpacing(4)
            box.addWidget(label(title, "metricTitle"))
            value_label = ElidedLabel(value)
            value_label.setObjectName("metricValue")
            hint_label = ElidedLabel(hint)
            hint_label.setObjectName("metricHint")
            box.addWidget(value_label)
            box.addWidget(hint_label)
            self.metric_values.append(value_label)
            self.metric_hints.append(hint_label)
            metrics.addWidget(frame, 1)
        body.addLayout(metrics)
        self.connection_detail = label("", "hint")
        self.connection_detail.setWordWrap(True)
        self.connection_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.addWidget(self.connection_detail)

        middle = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.middle = middle
        middle.setSpacing(16)
        settings_card, settings_box = card(
            "Local server", "Settings used when Nexus Desktop starts a server on this computer.")
        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        self.model = QLineEdit(settings.model)
        self.model.setObjectName("modelPath")
        self.model.setPlaceholderText("Choose a .gguf model file")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse_model)
        model_row.addWidget(self.model, 1)
        model_row.addWidget(browse)
        settings_box.addWidget(label("Model file", "fieldLabel"))
        settings_box.addLayout(model_row)
        binary_row = QHBoxLayout()
        binary_row.setSpacing(8)
        self.binary = QLineEdit(settings.binary or discover_binary())
        self.binary.setObjectName("binaryPath")
        self.binary.setPlaceholderText("nexus-server executable or command on PATH")
        choose_binary = QPushButton("Browse…")
        choose_binary.clicked.connect(self.browse_binary)
        binary_row.addWidget(self.binary, 1)
        binary_row.addWidget(choose_binary)
        settings_box.addWidget(label("Server executable", "fieldLabel"))
        settings_box.addLayout(binary_row)
        self.backend = QComboBox()
        self.backend.addItem("CPU", "cpu")
        self.backend.addItem("CUDA · NVIDIA", "cuda")
        self.backend.setCurrentIndex(max(0, self.backend.findData(settings.backend)))
        self.backend.setToolTip("CUDA requires a CUDA-enabled nexus-server and an NVIDIA driver.")
        self.kv = QComboBox()
        for text, data in [("F32 · full precision", "none"), ("K:3 bit / V:2 bit", "tq_3b2b"), ("K:3 bit / V:4 bit", "tq_3b4b"), ("K:4 bit / V:4 bit", "tq_4b4b"), ("Block Q4", "block_q4"), ("Block Q6", "block_q6")]:
            self.kv.addItem(text, data)
        self.kv.setCurrentIndex(max(0, self.kv.findData(settings.kv_quant)))
        pair(settings_box, "Compute", self.backend, "KV cache", self.kv)
        self.context = QSpinBox()
        self.context.setRange(1, 1048576)
        self.context.setValue(settings.context)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(settings.port)
        pair(settings_box, "Context (tokens)", self.context, "Local port", self.port)
        self.batch = QSpinBox()
        self.batch.setRange(1, 16)
        self.batch.setValue(settings.batch_size)
        batch_row = QHBoxLayout()
        batch_row.addWidget(label("Concurrent requests", "fieldLabel"))
        batch_row.addStretch()
        batch_row.addWidget(self.batch)
        settings_box.addLayout(batch_row)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.start_button = QPushButton("Start local server")
        self.start_button.setObjectName("startButton")
        self.start_button.setProperty("primary", True)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setProperty("danger", True)
        self.stop_button.setObjectName("stopButton")
        self.restart_button = QPushButton("Restart")
        self.start_button.clicked.connect(self.start_server)
        self.stop_button.clicked.connect(self.controller.stop)
        self.restart_button.clicked.connect(lambda: self.controller.restart(self.settings()))
        for button in [self.start_button, self.stop_button, self.restart_button]:
            actions.addWidget(button)
        actions.addStretch()
        settings_box.addSpacing(4)
        settings_box.addLayout(actions)
        self.local_note = label("", "hint")
        self.local_note.setWordWrap(True)
        settings_box.addWidget(self.local_note)
        settings_box.addStretch()
        middle.addWidget(settings_card, 3)

        right = QVBoxLayout()
        right.setSpacing(16)
        model_card, model_box = card("Loaded model", "Load, reload or unload without restarting the server.")
        self.loaded_model = ElidedLabel("No model is loaded yet.", Qt.TextElideMode.ElideMiddle)
        self.loaded_model.setObjectName("subtitle")
        self.loaded_model.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        model_box.addWidget(self.loaded_model)
        model_actions = QHBoxLayout()
        model_actions.setSpacing(8)
        self.load_button = QPushButton("Load")
        self.reload_button = QPushButton("Reload")
        self.reload_button.setToolTip("Reload the current model from the same file.")
        self.unload_button = QPushButton("Unload")
        self.unload_button.setProperty("danger", True)
        for button, action in [(self.load_button, "load"), (self.reload_button, "reload"), (self.unload_button, "unload")]:
            button.clicked.connect(lambda _checked=False, action=action: self.run_model_action(action))
            model_actions.addWidget(button)
        model_box.addLayout(model_actions)
        self.operation_hint = label("Model actions become available when the server is ready.", "hint")
        self.operation_hint.setWordWrap(True)
        model_box.addWidget(self.operation_hint)
        self.copy_key_button = QPushButton("Copy API key")
        self.copy_key_button.setToolTip("Copy the session key to connect a client to this app’s server.")
        self.copy_key_button.clicked.connect(self.copy_key)
        model_box.addWidget(self.copy_key_button)
        right.addWidget(model_card)
        connection_card, connection_box = card(
            "Connect to a running server", "For example a server running in Docker on this computer.")
        self.url = QLineEdit(settings.server_url)
        self.url.setObjectName("serverUrl")
        field(connection_box, "Local API address", self.url)
        self.key = QLineEdit()
        self.key.setObjectName("apiKey")
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("API key · optional")
        self.key.setToolTip("Used only for this session; never saved to disk.")
        self.key.setAccessibleName("API key")
        connection_box.addWidget(self.key)
        connection_actions = QHBoxLayout()
        connection_actions.setSpacing(8)
        self.connect_button = QPushButton("Connect")
        self.connect_button.setProperty("primary", True)
        self.disconnect_button = QPushButton("Disconnect")
        self.connect_button.clicked.connect(self.connect_server)
        self.disconnect_button.clicked.connect(self.controller.disconnect)
        connection_actions.addWidget(self.connect_button)
        connection_actions.addWidget(self.disconnect_button)
        connection_actions.addStretch()
        connection_box.addLayout(connection_actions)
        hint = label("The external server keeps running. The API key is kept only for this session.", "hint")
        hint.setWordWrap(True)
        connection_box.addWidget(hint)
        right.addWidget(connection_card)
        right.addStretch()
        middle.addLayout(right, 2)
        body.addLayout(middle)

        logs_card, log_box = card("")
        logs_title = QHBoxLayout()
        logs_title.addWidget(label("Server log", "cardTitle"))
        logs_title.addStretch()
        logs_title.addWidget(label("Last 600 lines", "hint"))
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_logs)
        clear.setMaximumHeight(32)
        logs_title.addWidget(clear)
        log_box.addLayout(logs_title)
        tabs = QTabWidget()
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(self.controller.MAX_LOG_LINES)
        self.log_view.setPlaceholderText("Logs appear here when the server starts.")
        self.metric_view = QPlainTextEdit()
        self.metric_view.setObjectName("metricView")
        self.metric_view.setReadOnly(True)
        self.metric_view.setMaximumBlockCount(500)
        tabs.addTab(self.log_view, "Logs")
        tabs.addTab(self.metric_view, "Metrics")
        tabs.setMinimumHeight(240)
        log_box.addWidget(tabs)
        body.addWidget(logs_card, 1)
        return scroll

    # -------------------------------------------------------------- navigation

    @property
    def current_page(self) -> str:
        return self.PAGES[self.pages.currentIndex()]

    def select_page(self, name: str) -> str:
        key = str(name).strip().lower()
        if key not in self.PAGES:
            raise ValueError(f"Unknown page {name!r}; expected one of {', '.join(self.PAGES)}")
        self.pages.setCurrentIndex(self.PAGES.index(key))
        self._page_changed()
        return key

    def _page_changed(self, _index: int = -1):
        key = self.current_page
        self.nav_buttons[key].setChecked(True)
        self._update_banner()

    def _sync_strip(self):
        # Margins exist only while something is shown, so an empty strip takes no space.
        shown = self.notice.isVisibleTo(self.strip) or self.banner.isVisibleTo(self.strip)
        if shown:
            self.strip_layout.setContentsMargins(28, 16, 28, 0)
        else:
            self.strip_layout.setContentsMargins(0, 0, 0, 0)

    def _update_responsive(self):
        narrow = self.width() - self.SIDEBAR_WIDTH < self.STACK_BELOW
        direction = QBoxLayout.Direction.TopToBottom if narrow else QBoxLayout.Direction.LeftToRight
        if self.middle.direction() != direction:
            self.middle.setDirection(direction)
            self.middle.setStretch(0, 0 if narrow else 3)
            self.middle.setStretch(1, 0 if narrow else 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "middle"):
            self._update_responsive()

    # ----------------------------------------------------------------- actions

    def settings(self):
        return Settings(binary=self.binary.text().strip(), model=self.model.text().strip(),
                        backend=self.backend.currentData(), kv_quant=self.kv.currentData(),
                        context=self.context.value(), port=self.port.value(), batch_size=self.batch.value(),
                        server_url=self.url.text().strip(), models_dir=self._models_dir())

    def _models_dir(self) -> str:
        return str(self.hub_panel.download_dir)

    def connect_server(self):
        self.notice.hide()
        self.controller.connect_external(self.url.text(), self.key.text())

    def run_model_action(self, action):
        if self.hub_panel.deletion_in_progress:
            return
        self.notice.hide()
        if action == "load":
            self._pending_model = ""
        self.controller.model_action(action, self.settings())

    def _server_available(self) -> bool:
        c = self.controller
        return bool(c.health) and c.authorized and not c.busy and not c.starting and not c.stopping

    def load_model_from_hub(self, path: str):
        """A downloaded file was chosen on the Models page."""
        if self.hub_panel.deletion_in_progress:
            return
        path = (path or "").strip()
        if not path:
            return
        self.model.setText(path)
        if self._server_available():
            self._pending_model = ""
            self.run_model_action("load")
            return
        # Never start a local server implicitly: explain what is needed instead.
        self._pending_model = path
        self.select_page("server")
        self.refresh()

    def model_deleted_from_hub(self, path: str):
        """Forget only paths referring to the model successfully moved to Trash."""
        deleted = Path(path).resolve(strict=False)
        def matches(value):
            return bool(value) and Path(value).resolve(strict=False) == deleted
        if matches(self.model.text().strip()):
            self.model.clear()
        if matches(self._pending_model):
            self._pending_model = ""
        try:
            save_settings(self.settings())
        except OSError:
            self.show_error("The model was moved to Trash, but settings could not be saved.")
        self.refresh()

    def copy_key(self):
        if self.controller.owned and self.controller.api.key:
            QApplication.clipboard().setText(self.controller.api.key)
            self.connection_detail.setText("API key copied to the clipboard; valid only for this server session.")

    def start_server(self):
        if self.hub_panel.deletion_in_progress:
            return
        self.notice.hide()
        self.controller.start(self.settings())

    def browse_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Choose a GGUF model", str(Path.home()), "GGUF models (*.gguf);;All files (*)", options=QFileDialog.Option.DontUseNativeDialog)
        if filename:
            self.model.setText(filename)

    def browse_binary(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Choose the nexus-server executable", "/usr/bin", "All files (*)", options=QFileDialog.Option.DontUseNativeDialog)
        if filename:
            self.binary.setText(filename)

    def show_error(self, message):
        # Matches both "only llama is implemented" and
        # "only llama and mistral are implemented" server messages.
        if "only llama" in message and "is implemented" in message:
            message += (" Choose a LLaMA- or Mistral-architecture GGUF in Models. "
                        "Changing GPU or Docker settings cannot add support for this architecture.")
        self.notice.setText(message)
        self.notice.show()
        self._sync_strip()

    def clear_logs(self):
        self.controller.logs.clear()
        self.log_view.clear()

    # ----------------------------------------------------------------- refresh

    def _tone(self, ready: bool, loaded: bool) -> str:
        c = self.controller
        if c.busy or c.starting or c.stopping or c.probing or c.state in {"Connecting", "Checking port", "Starting", "Stopping"}:
            return "busy"
        if c.state in {"Server error", "No connection", "API key required", "Failed to start"}:
            return "bad"
        if ready and loaded:
            return "ok"
        return "wait" if ready else "idle"

    def _banner_state(self, page: str):
        """(message, button text, target page) for the guidance strip, or None."""
        c = self.controller
        if page == "server":
            return None
        working = ("Checking the local port…" if c.probing else "Starting the server…" if c.starting
                   else "Stopping the server…" if c.stopping else "Connecting to the server…" if c.state == "Connecting" else "")
        if working:
            return working, "", "server"
        if c.mode == "disconnected":
            if page == "models":
                return ("No server is connected. You can browse and download models now; connect a server to load them.",
                        "Open Server", "server")
            return ("No server is connected. Start a local server or connect to a running one to start chatting.",
                    "Open Server", "server")
        if not c.authorized or c.state == "API key required":
            return "The server needs an API key before it can be used.", "Open Server", "server"
        if not c.health:
            return "Can’t reach the server. Check the address and port.", "Open Server", "server"
        if c.health.get("status") == "error":
            return "The server reported an error. Check the server log for details.", "Open Server", "server"
        if c.busy:
            return c.detail or "A model operation is in progress…", "", "server"
        if page == "chat" and not c.health.get("model_loaded"):
            return "The server is running, but no model is loaded. Load one from Models to start chatting.", "Open Models", "models"
        return None

    def _update_banner(self):
        state = self._banner_state(self.current_page)
        if state is None:
            self.banner.hide()
            return
        message, button, target = state
        self.banner.message.setText(message)
        self.banner.button.setText(button)
        self.banner.button.setVisible(bool(button))
        self._banner_target = target
        self.banner.show()

    def _update_instruction(self, available: bool):
        c = self.controller
        if self._pending_model and self.model.text().strip() != self._pending_model:
            self._pending_model = ""  # The user chose a different file; the hint is stale.
        if not self._pending_model:
            self.instruction.hide()
            return
        name = Path(self._pending_model).name or self._pending_model
        self.instruction_title.setText(f"“{name}” is selected")
        if available:
            text = "The server is ready. Load the model to use it in Chat."
        elif c.busy:
            text = "Another model operation is in progress. When it finishes, load the model to use it."
        elif c.mode == "disconnected":
            text = "To load it, start a local server below or connect to a running server, then press Load model."
        else:
            text = "Press Load model once the server is ready."
        self.instruction_text.setText(text)
        self.instruction_button.setEnabled(available)
        self.instruction.show()

    def refresh(self):
        c = self.controller
        ready = bool(c.health) and c.health.get("status") != "error" and c.authorized and not c.busy
        model = c.health.get("model") or (c.models[0].get("id") if c.models else "")
        loaded = bool(c.health.get("model_loaded"))
        tone = self._tone(ready, loaded)
        self.status.setText("Model operation in progress" if c.busy else c.state)
        set_property(self.status, "ready", ready)
        set_property(self.status, "tone", tone)
        set_property(self.status_dot, "tone", tone)
        self.connection_detail.setText(c.detail)
        port = c.api.base.rsplit(":", 1)[-1]
        endpoint = c.api.base.replace("http://", "")
        self.side_mode.setText(f"Local server · port {port}" if c.owned else f"External · {endpoint}" if c.mode == "external" else "No server connected")
        self.side_model.setText(model if loaded else "No model loaded")
        self.side_model.setToolTip(model if loaded else "")
        model_text = model if loaded else "Not loaded"
        self.metric_values[0].setText(model_text)
        self.metric_values[0].setToolTip(model)
        self.metric_hints[0].setText("Model is ready" if loaded else "Waiting for a GGUF model")
        self.loaded_model.setText((model + "\n" + (c.model_info.get("path") or "")) if loaded else "No model is loaded yet.")
        process = c.process
        self.metric_values[1].setText("Local process" if c.owned else "External" if c.mode == "external" else "Stopped")
        self.metric_hints[1].setText(f"PID {process.processId()} · port {port}" if c.owned else endpoint if c.mode == "external" else "127.0.0.1")
        runtime = c.metrics.get("runtime", {})
        self.metric_values[2].setText(str(runtime.get("active_sequences", "—")))
        self.metric_hints[2].setText(f"{runtime.get('queued_requests', 0)} requests queued")
        self.metric_values[3].setText(bytes_label(c.metrics.get("engine", {}).get("host_weight_payload_bytes")))
        self.metric_view.setPlainText(json.dumps(c.metrics, ensure_ascii=False, indent=2)[:65536] if c.metrics else "Waiting for metrics…")
        deleting = self.hub_panel.deletion_in_progress
        self.start_button.setEnabled(not deleting and c.mode != "external" and not c.owned and not c.probing and not c.stopping)
        self.start_button.setToolTip(
            "Connected to an external server. Manage its process with its original launcher. "
            "Disconnect before starting a separate local server."
            if c.mode == "external" else
            "Start a local nexus-server executable. To use a running Docker server, connect to its API address."
        )
        self.stop_button.setEnabled(c.owned and not c.stopping)
        self.restart_button.setEnabled(not deleting and c.owned and not c.stopping and not c.starting)
        self.connect_button.setEnabled(not c.owned and not c.probing)
        self.disconnect_button.setEnabled(c.mode == "external")
        available = bool(c.health) and c.authorized and not c.busy and not c.starting and not c.stopping
        self.copy_key_button.setVisible(c.owned)
        self.copy_key_button.setEnabled(c.owned and bool(c.api.key) and not c.stopping)
        self.load_button.setEnabled(available and not deleting)
        self.reload_button.setEnabled(available and not deleting and bool(c.model_info.get("path")))
        self.unload_button.setEnabled(available and not deleting and loaded)
        generating = bool(getattr(self.chat_panel, "is_generating", False))
        if deleting:
            hint = "Finish deleting the selected model before changing server models."
        elif c.busy:
            hint = "Operation in progress; check the logs for details."
        elif not available:
            hint = "Model actions become available when the server is ready."
        else:
            hint = "Unloading the model keeps the server running."
            if generating:
                hint += " A chat reply is being generated; reloading or unloading may interrupt it."
        self.operation_hint.setText(hint)
        # Explain the current mode and which actions apply to it.
        if c.mode == "external":
            title = "Connected to an external server"
            text = ("Nexus Desktop only talks to this server. It can’t start, stop or restart it — use its original launcher, "
                    "such as Docker. Disconnect first if you want to start a local server instead.")
            local = "Start is unavailable while connected to an external server. Disconnect to start a local server."
        elif c.mode == "owned" or c.owned:
            title = "Local server managed by Nexus Desktop"
            text = "Stop and Restart apply to this server. Closing Nexus Desktop stops it."
            local = "Stop or restart the server from here."
            if generating:
                local += " A chat reply is being generated; stopping or restarting will end it."
        else:
            title = "No server connected"
            text = "Start a local server on this computer, or connect to one that is already running — for example in Docker."
            local = "Starts a server on 127.0.0.1 using the settings above."
        self.mode_title.setText(title)
        self.mode_text.setText(text)
        set_property(self.mode_note, "tone", "neutral" if c.mode == "disconnected" and not c.owned else "info")
        self.local_note.setText(local)
        self._update_instruction(available)
        self._update_banner()
        for button in self.nav_buttons.values():
            button.setEnabled(not self._closing)
        self._sync_strip()

    # ------------------------------------------------------------------ closing

    def _shutdown_panels(self):
        if self._panels_closed:
            return
        self._panels_closed = True
        for panel in (self.chat_panel, self.hub_panel):
            try:
                panel.shutdown()
            except Exception:  # A failing panel must never keep the window or server alive.
                traceback.print_exc(file=sys.stderr)

    def closeEvent(self, event):
        try:
            save_settings(self.settings())
        except OSError:
            pass
        self._shutdown_panels()
        if self.controller.owned:
            self._closing = True
            self.controller._restart = None
            self.controller.stop()
            # The window stays up until the owned server exits; make that state obvious.
            self.pages.setEnabled(False)
            for button in self.nav_buttons.values():
                button.setEnabled(False)
            event.ignore()
            return
        self.controller.stop()  # Cancels a pending start-port probe, never external processes.
        self.controller.disconnect()
        event.accept()

    def _after_stopped(self):
        if self._closing:
            QTimer.singleShot(0, self.close)
