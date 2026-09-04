"""客户端可写目录。冻结版用 AppData，源码版仍用仓库根目录。"""
import os
import shutil
import sys
from pathlib import Path

APP_DIR_NAME = "RVC实时变声"


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def package_root():
    """包根：冻结版为 exe 所在目录，源码版为仓库根目录。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundled_dir():
    """只读资源：冻结版是 _internal，源码版是仓库根。"""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        internal = package_root() / "_internal"
        if internal.is_dir():
            return internal
    return package_root()


def user_data_dir():
    """角色列表、设置、日志。冻结版不写进安装目录，避免覆盖安装冲掉。"""
    if not is_frozen():
        return package_root()
    appdata = os.environ.get("APPDATA")
    if appdata:
        d = Path(appdata) / APP_DIR_NAME
    else:
        d = Path.home() / "AppData" / "Roaming" / APP_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_dir():
    d = user_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def speakers_path():
    return user_data_dir() / "speakers.json"


def settings_path():
    return user_data_dir() / "user_settings.json"


def presets_path():
    return user_data_dir() / "presets.json"


def _copy_if_needed(src, dst):
    try:
        if not src.is_file() or dst.is_file():
            return False
        if src.resolve() == dst.resolve():
            return False
    except Exception:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _first_existing(*paths):
    for p in paths:
        try:
            if p is not None and p.is_file():
                return p
        except Exception:
            continue
    return None


def ensure_user_data():
    """首次启动：把安装目录/内置默认拷到 AppData。已有用户文件则不动。"""
    data = user_data_dir()
    pkg = package_root()
    bundled = bundled_dir()
    internal = pkg / "_internal"
    mapping = (
        (speakers_path(), ("speakers.json",)),
        (settings_path(), ("user_settings.json",)),
        (presets_path(), ("presets.json",)),
    )
    for dst, names in mapping:
        if dst.is_file():
            continue
        cands = []
        for name in names:
            cands.extend((pkg / name, internal / name, bundled / name))
        src = _first_existing(*cands)
        if src is not None:
            _copy_if_needed(src, dst)
    return data
