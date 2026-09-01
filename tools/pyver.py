"""Refuse to run on anything but Python 3.11 (the only supported interpreter)."""
import sys

REQUIRED = (3, 11)


def require_python_311():
    if getattr(sys, "frozen", False):
        return
    if sys.version_info[:2] == REQUIRED:
        return
    ver = ".".join(map(str, sys.version_info[:3]))
    sys.stderr.write(
        "This project requires Python 3.11 x64 (got %s).\n"
        "RVC 只支持 Python 3.11 x64。\n"
        "Windows: py -3.11 -m venv .venv  or  install_local.bat (runtime\\)\n"
        "Linux:   python3.11 -m venv .venv\n" % ver
    )
    raise SystemExit(1)
