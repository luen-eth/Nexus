"""Shared visual language for Nexus Desktop: palette, typography, QSS and icons.

Design notes
------------
* Calm, light workspace (cool-grey canvas, white surfaces) with a deep ink sidebar
  as the single dark anchor. One accent (indigo) is used for primary actions,
  selection and focus; status colours are reserved for state only.
* All muted text meets WCAG AA (4.5:1) on the surface it is drawn on.
* Every interactive control has explicit hover, pressed, focus and disabled states.
* Styling is expressed with object names and dynamic properties so feature panels
  (chat, model hub) inherit the look without importing this module.
"""
from __future__ import annotations

from pathlib import Path
import re

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF

COLORS = {
    # Surfaces
    "canvas": "#f5f6fa",
    "surface": "#ffffff",
    "surface_alt": "#f0f2f8",
    "surface_soft": "#f7f8fc",  # Assistant message cards on the white chat canvas.
    "border": "#e0e4ed",
    "border_strong": "#c6cddd",
    "border_hover": "#9da8c0",
    # Text
    "ink": "#151b2c",
    "ink_soft": "#36415b",
    "muted": "#586278",       # 5.9:1 on white, 5.4:1 on canvas
    "faint": "#8791a6",       # disabled / decorative only
    # Accent
    "accent": "#4650d6",
    "accent_hover": "#3b44c0",
    "accent_press": "#3138a8",
    "accent_soft": "#eceefc",
    "accent_line": "#d3d8f5",
    # State
    "ok": "#14724a",
    "ok_bg": "#e3f3ea",
    "warn": "#7d5209",
    "warn_bg": "#fdf2dc",
    "warn_line": "#f1dba6",
    "bad": "#a42f2a",
    "bad_bg": "#fdeceb",
    "bad_line": "#f3c9c6",
    # Sidebar
    "side_bg": "#0e1322",
    "side_card": "#151c31",
    "side_line": "#222b46",
    "side_hover": "#1a2139",
    "side_active": "#252e52",
    "side_text": "#c9d1e6",
    "side_muted": "#98a4c2",
    "side_faint": "#7f8bab",
    # Logs
    "log_bg": "#0e1322",
    "log_text": "#cbd4ec",
}

FONT_FAMILY = '"Inter", "Noto Sans", "Segoe UI", "DejaVu Sans"'
MONO_FAMILY = '"JetBrains Mono", "DejaVu Sans Mono", "Monospace"'

_TEMPLATE = """
/* ---------- Base ---------- */
QWidget {font-family:@font; font-size:13px; color:@ink;}
QMainWindow, #content, #pageSurface, #page {background:@canvas;}
QToolTip {background:@ink; color:#ffffff; border:1px solid @ink; padding:6px 9px;}
QScrollArea {border:0; background:transparent;}

/* ---------- Typography ---------- */
#pageTitle {font-size:24px; font-weight:700; color:@ink;}
#pageSubtitle {font-size:13px; color:@muted;}
#cardTitle {font-size:15px; font-weight:700; color:@ink;}
#cardSubtitle {font-size:12px; color:@muted;}
#fieldLabel {font-size:12px; font-weight:600; color:@ink_soft;}
#hint, #metricHint {font-size:12px; color:@muted;}
#metricTitle {font-size:12px; font-weight:600; color:@muted;}
#metricValue {font-size:19px; font-weight:700; color:@ink;}
#subtitle {font-size:13px; color:@muted;}
#modelName {font-size:14px; font-weight:600; color:@ink;}

/* ---------- Cards and callouts ---------- */
#card {background:@surface; border:1px solid @border; border-radius:14px;}
#callout {background:@surface; border:1px solid @border; border-radius:12px;}
#callout[tone="info"] {background:@accent_soft; border-color:@accent_line;}
#callout[tone="warn"] {background:@warn_bg; border-color:@warn_line;}
#calloutTitle {font-size:13px; font-weight:700; color:@ink;}
#calloutText {font-size:13px; color:@ink_soft;}
#stateBanner {background:@accent_soft; border:1px solid @accent_line; border-radius:12px;}
#stateBannerText {font-size:13px; color:@ink_soft;}
#notice {background:@bad_bg; color:#7d2420; border:1px solid @bad_line; border-radius:12px; padding:11px 46px 11px 14px; font-size:13px;}
#noticeClose {border:0; background:transparent; color:@bad; font-size:18px; font-weight:700; border-radius:6px;}
#noticeClose:hover {background:@bad_line;}
#noticeClose:focus {border:2px solid @bad;}

/* ---------- Sidebar ---------- */
#sidebar {background:@side_bg; border:0;}
#brand {color:#ffffff; font-size:21px; font-weight:700;}
#brandTag {color:@side_muted; font-size:10px; font-weight:600;}
#sideSection {color:@side_muted; font-size:11px; font-weight:700;}
#navItem {background:transparent; border:1px solid transparent; border-radius:10px; color:@side_text; text-align:left; padding:10px 14px; font-size:14px; font-weight:600;}
#navItem:hover {background:@side_hover; color:#ffffff; border-color:transparent;}
#navItem:pressed {background:@side_active;}
#navItem:checked {background:@side_active; color:#ffffff;}
#navItem:checked:hover {background:@side_active;}
#navItem:focus {border:1px solid @accent_line; padding:10px 14px;}
#navItem:disabled {background:transparent; color:@side_faint; border-color:transparent;}
#sideCard {background:@side_card; border:1px solid @side_line; border-radius:12px;}
#status {color:#ffffff; font-size:13px; font-weight:700;}
#sideDetail {color:@side_muted; font-size:12px;}
#sideModel {color:@side_text; font-size:12px; font-weight:600;}
#sideFooter {color:@side_faint; font-size:11px;}
#statusDot {background:@side_faint; border:0; border-radius:5px; min-width:10px; max-width:10px; min-height:10px; max-height:10px;}
#statusDot[tone="ok"] {background:#3ecf8e;}
#statusDot[tone="wait"] {background:#f2b84b;}
#statusDot[tone="busy"] {background:#8c97ff;}
#statusDot[tone="bad"] {background:#f46d66;}

/* ---------- Inputs ---------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {background:@surface; border:1px solid @border_strong; border-radius:8px; min-height:20px; padding:7px 10px; selection-background-color:@accent; selection-color:#ffffff;}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {border-color:@border_hover;}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {border:2px solid @accent; padding:6px 9px;}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {background:@surface_alt; color:@faint; border-color:@border;}
QLineEdit:read-only {background:@surface_alt;}
QComboBox {padding-right:30px;}
QComboBox:focus {padding-right:29px;}
QComboBox::drop-down {subcontrol-origin:padding; subcontrol-position:center right; width:28px; border:0;}
QComboBox::down-arrow {image:url("@assets/chevron-down.svg"); width:12px; height:8px;}
QComboBox QAbstractItemView {background:@surface; border:1px solid @border_strong; border-radius:8px; padding:4px; outline:0; selection-background-color:@accent_soft; selection-color:@ink;}
QComboBox QAbstractItemView::item {min-height:26px; padding:2px 8px;}
QSpinBox, QDoubleSpinBox {padding-right:30px;}
QSpinBox:focus, QDoubleSpinBox:focus {padding-right:29px;}
QSpinBox::up-button, QDoubleSpinBox::up-button {subcontrol-origin:border; subcontrol-position:top right; width:24px; border:0; border-left:1px solid @border; background:@surface_alt; border-top-right-radius:7px;}
QSpinBox::down-button, QDoubleSpinBox::down-button {subcontrol-origin:border; subcontrol-position:bottom right; width:24px; border:0; border-left:1px solid @border; background:@surface_alt; border-bottom-right-radius:7px;}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {image:url("@assets/chevron-up.svg"); width:10px; height:7px;}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {image:url("@assets/chevron-down.svg"); width:10px; height:7px;}
QSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {background:@accent_soft;}
QPlainTextEdit, QTextEdit {selection-background-color:@accent; selection-color:#ffffff;}

/* ---------- Buttons ---------- */
QPushButton {background:@surface; border:1px solid @border_strong; border-radius:8px; padding:8px 14px; font-weight:600; color:@ink;}
QPushButton:hover {background:@surface_alt; border-color:@border_hover;}
QPushButton:pressed {background:#e4e8f3;}
QPushButton:focus {border:2px solid @accent; padding:7px 13px;}
QPushButton:disabled {color:@faint; background:@surface_alt; border-color:@border;}
QPushButton[primary="true"] {background:@accent; border-color:@accent; color:#ffffff;}
QPushButton[primary="true"]:hover {background:@accent_hover; border-color:@accent_hover;}
QPushButton[primary="true"]:pressed {background:@accent_press; border-color:@accent_press;}
QPushButton[primary="true"]:focus {border:2px solid @ink; padding:7px 13px;}
QPushButton[primary="true"]:disabled {background:#c5caf0; border-color:#c5caf0; color:#ffffff;}
QPushButton[danger="true"] {color:@bad;}
QPushButton[danger="true"]:hover {background:@bad_bg; border-color:@bad_line;}
QPushButton[danger="true"]:pressed {background:@bad_line;}
QPushButton[danger="true"]:focus {border:2px solid @bad; padding:7px 13px;}
QPushButton[danger="true"]:disabled {color:@faint; background:@surface_alt; border-color:@border;}
QPushButton[quiet="true"] {background:transparent; border:1px solid transparent; color:@accent;}
QPushButton[quiet="true"]:hover {background:@accent_soft; border-color:transparent;}
QPushButton[quiet="true"]:focus {border:2px solid @accent; padding:7px 13px;}
QPushButton[quiet="true"]:disabled {background:transparent; color:@faint;}
#bannerButton {background:@surface; border:1px solid @accent_line; color:@accent;}
#bannerButton:hover {background:@surface_alt; border-color:@accent;}
QToolButton {background:transparent; border:0; border-radius:6px;}
QToolButton:hover {background:@surface_alt;}
QToolButton:focus {border:2px solid @accent;}

/* ---------- Tabs ---------- */
QTabWidget::pane {border:0; background:transparent; top:-1px;}
QTabBar::tab {background:transparent; color:@muted; padding:9px 14px; margin-right:6px; border:0; border-bottom:2px solid transparent; font-weight:600;}
QTabBar::tab:hover {color:@ink;}
QTabBar::tab:selected {color:@accent; border-bottom:2px solid @accent;}
QTabBar::tab:focus {color:@accent;}

/* ---------- Logs ---------- */
#logView {background:@log_bg; color:@log_text; border:0; border-radius:10px; padding:12px; font-family:@mono; font-size:12px;}
#metricView {background:@surface_alt; color:@ink_soft; border:1px solid @border; border-radius:10px; padding:12px; font-family:@mono; font-size:12px;}

/* ---------- Lists, progress, menus (inherited by hub and chat) ---------- */
QListView, QListWidget, QTreeView, QTableView {background:@surface; border:1px solid @border; border-radius:10px; outline:0; selection-background-color:@accent_soft; selection-color:@ink;}
QListView::item, QListWidget::item {padding:6px 8px;}
QListView::item:hover, QListWidget::item:hover {background:@surface_alt;}
QListView::item:selected, QListWidget::item:selected {background:@accent_soft; color:@ink;}
QProgressBar {background:@surface_alt; border:1px solid @border; border-radius:6px; text-align:center; min-height:14px; color:@ink; font-size:11px;}
QProgressBar::chunk {background:@accent; border-radius:5px;}
QCheckBox {spacing:8px;}
QMenu {background:@surface; border:1px solid @border_strong; border-radius:8px; padding:4px;}
QMenu::item {padding:7px 18px; border-radius:6px;}
QMenu::item:selected {background:@accent_soft; color:@ink;}
QMenu::item:disabled {color:@faint;}
QMenu::separator {height:1px; background:@border; margin:4px 8px;}

/* ---------- Scrollbars ---------- */
QScrollBar:vertical {width:11px; background:transparent; margin:2px;}
QScrollBar::handle:vertical {background:#c3cadb; border-radius:3px; min-height:30px;}
QScrollBar::handle:vertical:hover {background:#9fa9c0;}
QScrollBar:horizontal {height:11px; background:transparent; margin:2px;}
QScrollBar::handle:horizontal {background:#c3cadb; border-radius:3px; min-width:30px;}
QScrollBar::handle:horizontal:hover {background:#9fa9c0;}
QScrollBar::add-line, QScrollBar::sub-line {height:0; width:0; border:0; background:transparent;}
QScrollBar::add-page, QScrollBar::sub-page {background:transparent;}

/* ---------- Feature panel hooks (chat / model hub) ---------- */
#chatPanel, #hubPanel {background:transparent;}
/* The transcript is a white canvas. Its padding keeps the square viewport clear of the rounded border. */
#chatTranscript {background:@surface; border:1px solid @border; border-radius:14px; padding:6px;}
#chatCanvas {background:@surface;}
#chatEmptyState, #chatEmpty {background:transparent;}
#chatEmptyTitle {font-size:20px; font-weight:700; color:@ink;}
#chatEmptyText, #chatEmptyHint {font-size:13px; color:@muted;}
/* Messages: tinted accent card for you, quiet neutral card for the model. */
QFrame#chatMessage {background:@surface_soft; border:1px solid @border; border-radius:12px;}
QFrame#chatMessage[role="user"] {background:@accent_soft; border-color:@accent_line;}
QFrame#chatMessage[role="assistant"] {background:@surface_soft; border-color:@border;}
#chatRole {font-size:12px; font-weight:700; color:@ink_soft;}
QFrame#chatMessage[role="user"] QLabel#chatRole {color:@accent;}
#chatMessageText {font-size:14px; color:@ink;}
#chatStats {font-size:12px; color:@muted;}
#chatNotice, #chatHint {font-size:12px; color:@muted;}
#chatComposer {background:@surface; border:1px solid @border_strong; border-radius:14px; padding:10px 12px; font-size:14px;}
#chatComposer:focus {border:2px solid @accent; padding:9px 11px;}
#hubStatus {color:@muted;}
/* The hub's compatibility note is guidance, not a failure: amber, never error red. Real errors keep #notice. */
#hubPanel QLabel#notice {background:@warn_bg; color:@warn; border:1px solid @warn_line; border-radius:12px; padding:11px 14px; font-size:13px;}
#hubPanel QFrame[frameShape="4"] {background:@border; border:0; min-height:1px; max-height:1px;}
"""


def build_stylesheet(assets_dir: str = "") -> str:
    """Return the application stylesheet with palette tokens resolved."""
    tokens = dict(COLORS)
    tokens["font"] = FONT_FAMILY
    tokens["mono"] = MONO_FAMILY
    tokens["assets"] = Path(assets_dir).as_posix() if assets_dir else "."
    return re.sub(r"@([a-z_]+)", lambda match: tokens[match.group(1)], _TEMPLATE)


def set_property(widget, name: str, value) -> None:
    """Set a dynamic property and re-polish only when it actually changed."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


# ---------------------------------------------------------------- icons

def _glyph(kind: str, color: str, size: int, gap: int) -> QPixmap:
    ratio = 2
    pixmap = QPixmap((size + gap) * ratio, size * ratio)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 20, size / 20)
    pen = QPen(QColor(color), 1.7)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    def line(*points):
        painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in points]))

    if kind == "chat":
        painter.drawRoundedRect(QRectF(3, 3.5, 14, 11), 3.5, 3.5)
        line((7.5, 14.5), (7.5, 17.3), (10.8, 14.5))
        line((7, 7.6), (13, 7.6))
        line((7, 10.6), (10.6, 10.6))
    elif kind == "models":
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in
                                       [(10, 2.6), (16.5, 6.2), (16.5, 13.8), (10, 17.4), (3.5, 13.8), (3.5, 6.2)]]))
        line((3.5, 6.2), (10, 9.8), (16.5, 6.2))
        line((10, 9.8), (10, 17.4))
    else:  # server
        painter.drawRoundedRect(QRectF(3, 3.5, 14, 5.2), 2, 2)
        painter.drawRoundedRect(QRectF(3, 11.3, 14, 5.2), 2, 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(6.4, 6.1), 0.95, 0.95)
        painter.drawEllipse(QPointF(6.4, 13.9), 0.95, 0.95)
    painter.end()
    return pixmap


def nav_icon(kind: str, idle: str, active: str, size: int = 20, gap: int = 10) -> QIcon:
    """Vector-drawn navigation icon. Checked buttons use the `active` colour.

    The pixmap includes `gap` transparent pixels on the right so the icon sits a
    comfortable distance from the button text regardless of the Qt style.
    """
    icon = QIcon()
    for color, state in ((idle, QIcon.State.Off), (active, QIcon.State.On)):
        pixmap = _glyph(kind, color, size, gap)
        for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected):
            icon.addPixmap(pixmap, mode, state)
    return icon
