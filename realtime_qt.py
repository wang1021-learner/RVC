#!/usr/bin/env python3
"""RVC 实时变声 - 桌面客户端入口。"""
import os, sys, logging, traceback, threading, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from tools.app_paths import ensure_user_data, log_dir
from ui.common import NL, STYLE_QSS, _friendly_error
from ui.main_window import MainWindow


def setup_logging():
    try:
        d = log_dir()
        root = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            fh = logging.FileHandler(d / "app.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(fh)
            root.setLevel(logging.INFO)
    except Exception:
        pass


_last_crash_popup = 0.0


def _excepthook(exc_type, exc, tb):
    if exc_type in (KeyboardInterrupt, SystemExit):
        return
    logging.getLogger("crash").critical("未捕获异常", exc_info=(exc_type, exc, tb))
    crash_path = log_dir() / "crash.log"
    try:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        with open(crash_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n" + text)
    except Exception:
        pass
    try:
        app = QApplication.instance()
        if app is None:
            return
        def _popup():
            global _last_crash_popup
            now = time.time()
            if now - _last_crash_popup < 4.0:
                return
            _last_crash_popup = now
            QMessageBox.critical(
                None, "程序错误",
                "发生未处理的错误，变声可能已中断。" + NL + NL
                + "详细日志：" + str(crash_path) + NL + NL
                + _friendly_error(exc))
        QTimer.singleShot(0, _popup)
    except Exception:
        pass


def _thread_excepthook(args):
    if getattr(args, "exc_type", None) in (KeyboardInterrupt, SystemExit):
        return
    _excepthook(args.exc_type, args.exc_value, args.exc_traceback)


if __name__ == "__main__":
    ensure_user_data()
    setup_logging()
    sys.excepthook = _excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
