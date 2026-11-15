"""CLI parsing is Qt-free so --help/--version also work without a display."""
import argparse
from . import __version__


def install_signal_handlers(application, window):
    """Wake Python while Qt is idle so SIGTERM follows the normal close path."""
    from PySide6.QtCore import QTimer
    import signal
    timer = QTimer(application)
    timer.setInterval(200)
    timer.timeout.connect(lambda: None)
    timer.start()
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    return timer


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nexus-desktop", description="Nexus yerel sunucu yöneticisi · Linux / Qt 6")
    parser.add_argument("--version", action="version", version=f"Nexus Desktop {__version__}")
    parser.add_argument("--server", help="nexus-server çalıştırılabilir dosyası")
    parser.add_argument("--model", help="GGUF model dosyası")
    parser.add_argument("--port", type=int, help="Yerel sunucu portu")
    parser.add_argument("--connect", metavar="URL", help="Var olan localhost sunucuya bağlan; yeni süreç başlatma")
    args = parser.parse_args(argv)
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("port 1..65535 aralığında olmalı")
    from PySide6.QtCore import QTimer, QEvent
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from .config import load_settings
    from .window import MainWindow
    from pathlib import Path
    settings = load_settings()
    if args.server:
        settings.binary = args.server
    if args.model:
        settings.model = args.model
    if args.port is not None:
        settings.port = args.port
    application = QApplication(["nexus-desktop"])
    application.setApplicationName("Nexus Desktop")
    application.setOrganizationName("Nexus")
    application.setStyle("Fusion")
    application.setWindowIcon(QIcon(str(Path(__file__).parent / "assets/nexus.svg")))
    window = MainWindow(settings)
    window.show()
    signal_timer = install_signal_handlers(application, window)
    if args.connect:
        QTimer.singleShot(0, lambda: window.controller.connect_external(args.connect))
    result = application.exec()
    signal_timer.stop()
    window.deleteLater()
    application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
