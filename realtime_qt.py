#!/usr/bin/env python3
"""
RVC 实时变声 - 桌面客户端
============================
架构: MainWindow(UI) -> VCEngine(音频+信号) -> RVCClient(服务器推理)
"""
import os, sys, json, queue, time, subprocess, logging, traceback
from pathlib import Path
import numpy as np
import wave as wave_mod
import sounddevice as sd

from PySide6.QtCore import Qt, Signal, QObject, QThread, QSize, QTimer
from PySide6.QtWidgets import (
    QStyledItemDelegate, QStyle,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QRadioButton, QFileDialog, QGroupBox, QMessageBox,
    QLineEdit, QStatusBar, QSplitter, QSlider,
    QDialog, QDialogButtonBox, QFormLayout, QFrame, QListView,
    QInputDialog,
)
from PySide6.QtGui import (QDragEnterEvent, QDropEvent, QColor,
    QStandardItemModel, QStandardItem)


# ==============================================================================
# 音频设备选择下拉框 - 分组显示 + 图标 + 两行信息
# ==============================================================================
class DeviceItemDelegate(QStyledItemDelegate):
    """自定义绘制: 分组标题 / 设备项(两行)"""

    def __init__(self, direction="input", parent=None):
        super().__init__(parent)
        self.direction = direction
        self.icon = "🎤 " if direction == "input" else "🔊 "

    def paint(self, painter, option, index):
        is_group = index.data(Qt.UserRole + 1) == "group"

        if is_group:
            # 分组标题: 浅灰底 + 小字
            painter.save()
            painter.fillRect(option.rect, QColor("#f0f3f7"))
            painter.setPen(QColor("#7f8c8d"))
            f = painter.font(); f.setPointSize(9); f.setBold(True)
            painter.setFont(f)
            painter.drawText(option.rect.x() + 10, option.rect.y() + 16, str(index.data() or ""))
            painter.restore()
            return

        # 设备项
        name = index.data(Qt.DisplayRole) or ""
        detail = index.data(Qt.UserRole + 2) or ""
        is_default = index.data(Qt.UserRole + 3) or False

        painter.save()
        # 选中/悬停背景
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, QColor("#d6eaf8"))
        elif option.state & QStyle.State_MouseOver:
            painter.fillRect(option.rect, QColor("#eef4fa"))

        # 图标 + 主名 (Line 1)
        painter.setPen(QColor("#2c3e50"))
        f = painter.font(); f.setPointSize(10); f.setBold(False)
        painter.setFont(f)
        text = self.icon + name
        if is_default:
            text += "  ⭐"
        painter.drawText(option.rect.x() + 8, option.rect.y() + 17, text)

        # 副行: 采样率/声道 (Line 2)
        if detail:
            painter.setPen(QColor("#95a5a6"))
            f2 = painter.font(); f2.setPointSize(8)
            painter.setFont(f2)
            painter.drawText(option.rect.x() + 28, option.rect.y() + 33, detail)
        painter.restore()

    def sizeHint(self, option, index):
        if index.data(Qt.UserRole + 1) == "group":
            return QSize(option.rect.width(), 24)
        return QSize(option.rect.width(), 40)


class DeviceCombo(QComboBox):
    """音频设备下拉框: 按 API 分组, 显示设备详情"""

    def __init__(self, direction="input", parent=None):
        super().__init__(parent)
        self.direction = direction
        self.setView(QListView(self))
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self.setItemDelegate(DeviceItemDelegate(direction, self))
        self.setMaxVisibleItems(8)
        self._device_ids = {}   # row -> device index
        self._device_names = {}  # row -> 完整设备名（防 ID 漂移）
        self._device_apis = {}   # row -> hostapi（同名设备区分 MME/WASAPI）

    @staticmethod
    def _api_rank(api_name):
        n = (api_name or "").upper()
        if "WASAPI" in n:
            return 0
        if "WDM" in n or "KS" in n:
            return 1
        if "DIRECTSOUND" in n:
            return 2
        if "MME" in n:
            return 3
        return 4

    def populate(self, devs, apis, ch):
        """按 hostapi 分组填充，WASAPI 优先。"""
        self._model.clear()
        self._device_ids = {}
        self._device_names = {}
        self._device_apis = {}
        row = 0

        default_item = QStandardItem("默认设备")
        default_item.setData("device", Qt.UserRole + 1)
        default_item.setData("", Qt.UserRole + 2)
        default_item.setData(True, Qt.UserRole + 3)
        default_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self._model.appendRow(default_item)
        self._device_ids[row] = None
        row += 1

        buckets = {}
        for i, d in enumerate(devs):
            if d[ch] <= 0:
                continue
            buckets.setdefault(d.get("hostapi", 0), []).append((i, d))

        for api_idx in sorted(buckets, key=lambda a: (
            self._api_rank(apis[a]["name"] if a < len(apis) else ""),
            a,
        )):
            api_name = apis[api_idx]["name"] if api_idx < len(apis) else "其他"
            head = QStandardItem(api_name)
            head.setData("group", Qt.UserRole + 1)
            head.setFlags(Qt.ItemIsEnabled)
            self._model.appendRow(head)
            row += 1
            for i, d in buckets[api_idx]:
                name = d["name"]
                show = name if len(name) <= 36 else name[:34] + "..."
                sr = int(d.get("default_samplerate", 0))
                chs = d[ch]
                detail = f"{sr // 1000}kHz · {chs}ch" if sr > 0 else f"{chs}ch"
                item = QStandardItem(show)
                item.setData("device", Qt.UserRole + 1)
                item.setData(detail, Qt.UserRole + 2)
                item.setData(False, Qt.UserRole + 3)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._model.appendRow(item)
                self._device_ids[row] = i
                self._device_names[row] = d["name"]
                self._device_apis[row] = api_idx
                row += 1

    def currentDeviceId(self):
        """返回当前选中的 sounddevice 设备 ID"""
        return self._device_ids.get(self.currentIndex())

    def currentDeviceName(self):
        """当前选中设备的完整名称（启动时按名字重解析 ID，防设备索引漂移）"""
        return self._device_names.get(self.currentIndex())

    def currentDeviceApi(self):
        """当前选中设备的 hostapi（区分同名 MME/WASAPI 设备）"""
        return self._device_apis.get(self.currentIndex())

    def selectByNameApi(self, name, api=None):
        if not name:
            self.setCurrentIndex(0)
            return False
        for row, n in self._device_names.items():
            if n == name and (api is None or self._device_apis.get(row) == api):
                self.setCurrentIndex(row)
                return True
        for row, n in self._device_names.items():
            if n == name:
                self.setCurrentIndex(row)
                return True
        return False

    def selectFirstWithApi(self, api):
        for row, a in self._device_apis.items():
            if a == api:
                self.setCurrentIndex(row)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        view = self.view()
        h = 6
        # PySide6 不暴露 protected 的 viewOptions()，改用 public 的 sizeHintForRow
        for i in range(self.count()):
            h += view.sizeHintForRow(i)
        h = min(h, 380)
        view.setFixedHeight(h)
        parent = view.parentWidget()
        if parent is not None and parent is not self:
            parent.setFixedHeight(h + 4)

def create_styled_combo(min_width=0, max_visible=8):
    cb = QComboBox()
    if min_width > 0:
        cb.setMinimumWidth(min_width)
    cb.setMaxVisibleItems(max_visible)
    return cb

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")

from worker.rvc_client import RVCClient
from worker.local_server import is_frozen, package_root, runtime_installed
from tools.audio_meter import VUMeterWidget, calc_rms_db
from tools.audio_process import AutoGain


def setup_logging():
    """应用日志落盘：logs/app.log（源码版=项目根，冻结版=exe 目录）。"""
    try:
        log_dir = package_root() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(fh)
            root.setLevel(logging.INFO)
    except Exception:
        pass


def _excepthook(exc_type, exc, tb):
    """未捕获异常：写入 crash.log 并弹窗提示（日志路径随包定位）。"""
    logging.getLogger("crash").critical("未捕获异常", exc_info=(exc_type, exc, tb))
    try:
        log_dir = package_root() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        with open(log_dir / "crash.log", "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n" + text)
    except Exception:
        pass
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is not None:
            QMessageBox.critical(
                None, "程序错误",
                "发生未处理的错误，日志已保存到 logs 目录（app.log / crash.log）\n\n"
                + str(exc)[:300])
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc, tb)


class LazyLocalPipeline:
    """启动时不立刻 import torch，第一次加载角色再进本机推理。"""
    is_remote = False

    def __init__(self, on_status):
        self._on_status = on_status
        self._real = None

    def _ensure(self):
        if self._real is None:
            from worker.rvc_pipeline import RVCPipeline
            self._real = RVCPipeline(on_status=self._on_status)
        return self._real

    @property
    def is_loaded(self):
        return bool(self._real and self._real.is_loaded)

    @property
    def samplerate(self):
        return self._real.samplerate if self._real is not None else 48000

    @property
    def channels(self):
        return self._real.channels if self._real is not None else 1

    @property
    def _block_frame(self):
        if self._real is not None and hasattr(self._real, "_block_frame"):
            return self._real._block_frame
        return None

    def is_connected(self):
        return True

    def abort(self):
        return

    def set_server_url(self, url):
        return

    def stop(self):
        if self._real is not None:
            self._real.stop()

    def unload(self):
        if self._real is not None:
            self._real.unload()

    def disconnect(self):
        return

    def __getattr__(self, name):
        return getattr(self._ensure(), name)


def make_pipeline(mode, server_url, on_status):
    if mode == "local":
        if not os.environ.get("RVC_DIRECT_LOCAL"):
            # 统一走本机子进程推理（崩溃隔离、单一代码路径、UI 启动不加载 torch）；
            # 设置环境变量 RVC_DIRECT_LOCAL=1 可回退为进程内直连（旧行为）
            from worker.local_server import LocalServerPipeline
            return LocalServerPipeline(on_status=on_status)
        return LazyLocalPipeline(on_status)
    client = RVCClient(server_url=server_url, on_status=on_status)
    client.connect(timeout=3)
    return client

NL = chr(10)
SETTINGS_FILE = PROJECT_ROOT / "user_settings.json"
PRESETS_FILE = PROJECT_ROOT / "presets.json"
DEFAULT_SERVER_URL = "ws://192.168.1.28:8765"
SERVER_ROOT = "/home/songwang/Retrieval-based-Voice-Conversion-WebUI"
SERVER_MODEL_DIR = SERVER_ROOT + "/assets/weights"
SERVER_INDEX_DIR = SERVER_ROOT + "/logs/thchs_v2"
RESTART_KEYS = ("block_time", "crossfade_time", "extra_time", "I_noise_reduce", "O_noise_reduce")
DEFAULT_PARAMS = {
    "block_time": 0.08,
    "crossfade_time": 0.02,
    "extra_time": 1.5,
    "f0method": "rmvpe",
    "I_noise_reduce": False,
    "O_noise_reduce": False,
    "rms_mix_rate": 0.3,
    "threhold": -50,
    "limiter_enable": True,
    "limiter_threshold_db": -1.0,
}

# 场景预设：低延迟 / 高音质 / 游戏语音 / 唱歌
BUILTIN_PRESETS = [
    {
        "name": "低延迟",
        "params": {
            "block_time": 0.05, "crossfade_time": 0.01, "extra_time": 0.8,
            "f0method": "rmvpe", "rms_mix_rate": 0.5, "threhold": -50,
            "I_noise_reduce": False, "O_noise_reduce": False,
        },
    },
    {
        "name": "高音质",
        "params": {
            "block_time": 0.1, "crossfade_time": 0.03, "extra_time": 2.0,
            "f0method": "rmvpe", "rms_mix_rate": 0.3, "threhold": -55,
            "I_noise_reduce": True, "O_noise_reduce": False,
        },
    },
    {
        "name": "游戏语音",
        "params": {
            "block_time": 0.06, "crossfade_time": 0.02, "extra_time": 1.0,
            "f0method": "rmvpe", "rms_mix_rate": 0.6, "threhold": -45,
            "I_noise_reduce": True, "O_noise_reduce": False,
        },
    },
    {
        "name": "唱歌",
        "params": {
            "block_time": 0.06, "crossfade_time": 0.02, "extra_time": 1.5,
            "f0method": "rmvpe", "rms_mix_rate": 0.0, "threhold": -60,
            "I_noise_reduce": False, "O_noise_reduce": True,
        },
    },
]


def _wasapi_fail_reason(exc):
    s = str(exc or "").lower()
    if "illegal combination" in s:
        return "输入输出不是同一组 API（请选择同组设备，或点「刷新设备」重试）"
    if "sample" in s or "rate" in s:
        return "采样率需为 48k（设备不支持当前采样率）"
    if "channel" in s:
        return "通道数不匹配（请更换设备或声道设置）"
    if "unsupported" in s or "format" in s:
        return "设备不支持当前音频格式（尝试切换共享模式）"
    if "full duplex" in s or "duplex" in s:
        return "设备不支持同时输入输出（全双工）"
    if any(k in s for k in ("unavail", "invalid device", "exclusive", "busy", "in use")):
        return "设备被占用或不支持独占（关闭占用程序，或改用共享模式）"
    if "not found" in s or "doesn't exist" in s or "no such device" in s:
        return "设备不存在，请点「刷新设备」后重选"
    text = str(exc).strip()
    return text[:80] if text else "未知原因"


def _friendly_net_error(e):
    """把底层网络异常翻译成用户可行动的提示。"""
    s = str(e or "").lower()
    if "refused" in s or "10061" in s:
        return "服务器拒绝连接（未启动或端口不对）"
    if "timed out" in s or "timeout" in s or "10060" in s:
        return "连接超时（检查 IP 地址与防火墙）"
    if "getaddrinfo" in s or "nodename" in s or "name or service" in s:
        return "地址无法解析（请检查服务器地址写法）"
    if "unreachable" in s or "10065" in s:
        return "网络不可达（请检查网络连接）"
    if "handshake" in s or "10054" in s or "reset" in s:
        return "连接被重置（服务器可能异常退出）"
    return (str(e) or "未知错误")[:120]


def _device_fingerprint():
    try:
        devs = sd.query_devices()
        return tuple(
            (d["name"], d.get("hostapi", 0), d["max_input_channels"], d["max_output_channels"])
            for d in devs
        )
    except Exception:
        return ()


def load_user_settings():
    if SETTINGS_FILE.is_file():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_user_settings(data):
    SETTINGS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_presets():
    """内置预设 + 用户 presets.json（追加，不覆盖内置）。"""
    presets = list(BUILTIN_PRESETS)
    try:
        if PRESETS_FILE.is_file():
            data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
            for p in data.get("presets", []):
                if isinstance(p, dict) and p.get("name") and isinstance(p.get("params"), dict):
                    presets.append({"name": p["name"], "params": p["params"]})
    except Exception:
        pass
    return presets


def save_presets(presets):
    """保存用户预设（仅存非内置条目）。"""
    try:
        PRESETS_FILE.write_text(
            json.dumps({"presets": presets}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except Exception:
        pass


def to_server_path(local_path: str) -> str:
    """模型进 weights/，索引进 logs/thchs_v2/（服务器端还会再搜一遍）。"""
    name = Path(str(local_path)).name
    if name.lower().endswith(".index"):
        return SERVER_INDEX_DIR + "/" + name
    return SERVER_MODEL_DIR + "/" + name

SPEAKERS_FILE = PROJECT_ROOT / "speakers.json"
WEIGHTS_DIR   = PROJECT_ROOT / "assets" / "weights"

STYLE_QSS = """
QMainWindow, QDialog { background-color: #e7ecf1; color: #1c2833; }
QWidget { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 13px; }
QFrame#header {
    background-color: #f7f9fb; border: 1px solid #d5dde6;
    border-radius: 12px;
}
QGroupBox {
    border: 1px solid #d5dde6; border-radius: 12px; margin-top: 12px;
    padding: 16px 12px 12px 12px; background-color: #ffffff;
    font-weight: 600; color: #3d4f5f;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 8px; color: #0f766e; }
QLabel#appTitle { font-size: 18px; font-weight: 700; color: #12352f; }
QLabel#fieldLabel { color: #5d6d7e; font-size: 12px; font-weight: 600; }
QLabel#chip { padding: 2px 0; }
QComboBox {
    combobox-popup: 0;
    background-color: #ffffff; border: 1px solid #cfd8e3;
    border-radius: 8px; padding: 5px 10px; min-height: 22px;
    font-size: 13px; color: #1c2833;
}
QComboBox:hover { border-color: #0f766e; }
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: top right;
    width: 22px; border: none;
}
QComboBox QAbstractItemView {
    background-color: #ffffff; border: 1px solid #cfd8e3;
    border-radius: 8px; padding: 4px; outline: none;
    max-height: 280px;
    selection-background-color: #d1fae5; selection-color: #134e4a;
}
QComboBox QAbstractItemView::item { min-height: 24px; padding: 3px 8px; }
QScrollBar:vertical { border: none; background: #eef2f6; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #b7c3ce; border-radius: 4px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #8fa0ae; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #fff; border: 1px solid #cfd8e3;
    border-radius: 8px; padding: 4px 8px; min-height: 22px;
}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus {
    border-color: #0f766e;
}
QCheckBox { spacing: 8px; color: #3d4f5f; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 2px solid #b8c4d0;
    border-radius: 4px; background-color: #fff;
}
QCheckBox::indicator:checked { background-color: #0f766e; border-color: #0f766e; }
QPushButton {
    background-color: #eef3f6; color: #1c2833;
    border: 1px solid #cfd8e3; border-radius: 8px;
    padding: 7px 12px; font-weight: 600;
}
QPushButton:hover { background-color: #e1eaef; border-color: #b7c6d1; }
QPushButton:pressed { background-color: #d4e0e7; }
QPushButton#btnGhost { background: transparent; }
QPushButton#btnConnect { background-color: #0f766e; color: #fff; border: none; min-width: 72px; }
QPushButton#btnConnect:hover { background-color: #0d9488; }
QPushButton#btnStart { font-size: 16px; padding: 14px 20px; border-radius: 10px; border: none; }
QPushButton#btnStart[state="off"] { background-color: #15803d; color: #fff; }
QPushButton#btnStart[state="off"]:hover { background-color: #16a34a; }
QPushButton#btnStart[state="on"] { background-color: #b91c1c; color: #fff; }
QPushButton#btnStart[state="on"]:hover { background-color: #dc2626; }
QPushButton#btnStart:disabled { background-color: #b8c4ce; color: #fff; }
QLabel { color: #1c2833; }
QStatusBar { background-color: #f7f9fb; color: #6b7c8a; border-top: 1px solid #d5dde6; }
QFrame#roleCard {
    background: #f0fdfa; border: 1px solid #99f6e4; border-radius: 10px;
}
QSplitter::handle { background: #d5dde6; width: 1px; }
"""

# 状态灯颜色
LIGHT_GRAY  = "#bdc3c7"   # 未加载
LIGHT_YELLOW = "#f39c12"  # 加载中
LIGHT_GREEN = "#27ae60"   # 运行中
LIGHT_RED   = "#e74c3c"   # 错误

# ==============================================================================
# 角色配置
# ==============================================================================
class SpeakerConfig:
    __slots__ = (
        "name", "model_path", "index_path", "speaker_id", "pitch", "index_rate",
        "formant", "f0method", "I_noise_reduce", "O_noise_reduce",
        "rms_mix_rate", "threhold",
    )

    def __init__(self, name="", model_path="", index_path="", speaker_id=0,
                 pitch=0, index_rate=0.0, formant=0.0, f0method="rmvpe",
                 I_noise_reduce=False, O_noise_reduce=False,
                 rms_mix_rate=0.3, threhold=-50):
        self.name = name; self.model_path = model_path; self.index_path = index_path
        self.speaker_id = speaker_id; self.pitch = pitch; self.index_rate = index_rate
        self.formant = formant; self.f0method = f0method
        self.I_noise_reduce = I_noise_reduce; self.O_noise_reduce = O_noise_reduce
        self.rms_mix_rate = rms_mix_rate; self.threhold = threhold

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def pipeline_overrides(self):
        """加载/启动时覆盖全局高级参数的角色级参数。"""
        return {
            "f0method": self.f0method or "rmvpe",
            "I_noise_reduce": bool(self.I_noise_reduce),
            "O_noise_reduce": bool(self.O_noise_reduce),
            "rms_mix_rate": float(self.rms_mix_rate),
            "threhold": int(self.threhold),
        }

    @classmethod
    def from_dict(cls, d):
        defaults = {
            "name": "", "model_path": "", "index_path": "", "speaker_id": 0,
            "pitch": 0, "index_rate": 0.0, "formant": 0.0, "f0method": "rmvpe",
            "I_noise_reduce": False, "O_noise_reduce": False,
            "rms_mix_rate": 0.3, "threhold": -50,
        }
        defaults.update({k: v for k, v in (d or {}).items() if k in cls.__slots__})
        return cls(**defaults)

class SpeakerManager:
    def __init__(self, path=SPEAKERS_FILE):
        self.path = Path(path); self.speakers = []; self.load()
    def load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.speakers = [SpeakerConfig.from_dict(s) for s in data.get("speakers", [])]
    def save(self):
        self.path.write_text(
            json.dumps({"speakers": [s.to_dict() for s in self.speakers]},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    def add(self, s): self.speakers.append(s); self.save()
    def remove(self, i):
        if 0 <= i < len(self.speakers): self.speakers.pop(i); self.save()
    def update(self, i, s):
        if 0 <= i < len(self.speakers): self.speakers[i] = s; self.save()


# ==============================================================================
# 后台模型加载线程 - 避免界面卡死
# ==============================================================================
class ModelLoader(QThread):
    finished_ok = Signal(object, int)   # speaker, 加载代数
    failed = Signal(str, int)

    def __init__(self, engine, speaker, gen, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.speaker = speaker
        self.gen = gen
        self.pipeline = engine.pipeline

    def run(self):
        try:
            mp, ip = self.engine.resolve_paths(self.speaker)
            params = self.engine.merged_params(self.speaker)
            ok = self.pipeline.load_speaker(
                mp, ip,
                self.speaker.pitch, self.speaker.index_rate,
                formant=self.speaker.formant,
                **params)
            if ok:
                self.finished_ok.emit(self.speaker, self.gen)
            else:
                self.failed.emit("模型加载失败", self.gen)
        except Exception as e:
            self.failed.emit(str(e), self.gen)

# ==============================================================================
# 异步推理工作线程 - 彻底解耦音频 I/O 与 PyTorch 深度学习计算
# ==============================================================================
class InferenceWorkerThread(QThread):
    infer_done = Signal(int, float, float)  # elapsed_ms, in_rms_db, out_rms_db
    stage_stats = Signal(dict)              # 本地模式分阶段耗时 {feature,index,pitch,model}
    xrun_occurred = Signal()
    need_recover = Signal()

    def __init__(self, pipeline, input_queue, output_queue, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = False

    def _emit_out(self, out_block, elapsed_ms, in_rms):
        try:
            self.output_queue.put_nowait(out_block)
        except queue.Full:
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(out_block)
            except queue.Full:
                pass
            self.xrun_occurred.emit()
        self.infer_done.emit(elapsed_ms, in_rms, calc_rms_db(out_block))

    def run(self):
        from collections import deque
        self.running = True
        inflight = deque()
        depth = 2 if getattr(self.pipeline, "is_remote", False) else 1
        while self.running:
            while self.running and len(inflight) < depth:
                try:
                    indata = self.input_queue.get(timeout=0.01 if inflight else 0.05)
                except queue.Empty:
                    break
                token = self.pipeline.send_audio(indata)
                if token is None:
                    if not self.pipeline.is_connected():
                        self.need_recover.emit()
                    break
                inflight.append((token, calc_rms_db(indata)))
            if not inflight:
                continue
            token, in_rms = inflight.popleft()
            out_block, elapsed_ms = self.pipeline.recv_audio(*token)
            if not self.pipeline.is_connected():
                self.need_recover.emit()
            if not self.running:
                break
            stage = getattr(self.pipeline, "last_stage_ms", None)
            if stage:
                self.stage_stats.emit(dict(stage))
            self._emit_out(out_block, elapsed_ms, in_rms)

    def stop(self):
        self.running = False
        if self.isRunning():
            self.quit()
            self.wait(1800)


class RecThread(QThread):
    """录音测试线程：录 N 秒 → 用当前角色变声 → 保存 wav"""
    done = Signal(str, str)   # 保存路径, 错误信息

    def __init__(self, engine, seconds=10, device=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.seconds = seconds
        self.device = device

    def run(self):
        import sounddevice as sd
        import numpy as np
        import time, os
        try:
            c = self.engine.pipeline
            if c is None:
                self.done.emit("", "推理引擎未就绪")
                return
            if not c.is_connected():
                if not c.connect(timeout=5):
                    self.done.emit("", "无法连接服务器")
                    return
            try:
                started = c.start(**self.engine.merged_params())
                if started is False:
                    self.done.emit("", "无法启动推理")
                    return
            except Exception as e:
                self.done.emit("", "无法启动推理: " + str(e))
                return
            SR = c.samplerate
            BLOCK = getattr(c, "_block_frame", None)
            if not BLOCK:
                self.done.emit("", "推理块大小未知，请先成功加载角色")
                return
            frames = []
            rec_log = []
            def cb(indata, frames_, times, status):
                frames.append(indata[:, 0].copy())
            kwargs = dict(samplerate=SR, channels=1, dtype='float32',
                          blocksize=BLOCK, callback=cb)
            if self.device is not None:
                kwargs['device'] = self.device
            with sd.InputStream(**kwargs):
                time.sleep(self.seconds)
            if not frames:
                self.done.emit("", "没有录到音频")
                return
            raw = np.concatenate(frames)
            outs = []
            for i in range(0, len(raw) - BLOCK + 1, BLOCK):
                out, _ = c.process_chunk(raw[i:i + BLOCK])
                outs.append(np.asarray(out).reshape(-1))
            if not outs:
                self.done.emit("", "录音太短")
                return
            os.makedirs('record_out', exist_ok=True)
            out = np.concatenate(outs)
            spk = self.engine.current_speaker
            fname = os.path.join('record_out', f'rec_{spk.name}_{int(time.time())}.wav')
            pcm = np.clip(out * 32767, -32768, 32767).astype(np.int16)
            with wave_mod.open(fname, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
                wf.writeframes(pcm.tobytes())
            self.done.emit(fname, "")
        except Exception as e:
            self.done.emit("", str(e))


# ==============================================================================
# 推理引擎（本地）
# ==============================================================================
class VCEngine(QObject):
    status_msg = Signal(str); infer_time = Signal(int)
    started_ok = Signal(); stopped_ok = Signal()
    load_failed = Signal(str)
    rms_levels = Signal(float, float)  # in_db, out_db
    xrun_signal = Signal(int)         # total_xruns
    fade_done = Signal()
    loop_latency = Signal(float)      # 端到端延迟(ms)：output DAC time - input ADC time
    stage_stats = Signal(dict)        # 本地模式分阶段耗时

    def __init__(self, mode="local", server_url=DEFAULT_SERVER_URL):
        super().__init__()
        self.mode = "local" if mode == "local" else "server"
        self.server_url = server_url or DEFAULT_SERVER_URL
        self.pipeline = make_pipeline(
            self.mode, self.server_url, lambda m: self.status_msg.emit(m))
        self.stream = None; self.running = False
        self.current_speaker = None
        self.input_device = None; self.output_device = None
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self._in_residual = None      # 输入弹性缓冲（变长块拼接）
        self._out_residual = np.array([], dtype=np.float32)   # 输出弹性缓冲
        self.worker_thread = None
        self._zombie_threads = []   # 超时未退出的线程，等 finished 再删，避免销毁运行中的线程
        self.xrun_count = 0
        self._params = dict(DEFAULT_PARAMS)
        self._formant = 0.0
        self.dry_mix = 0.0
        self.bypass = False
        self._fade_in_left = 0
        self._fade_out_left = 0
        self._fade_out_total = 0
        self._last_recover = 0.0
        self._fade_epoch = 0
        self._pending_fade_epoch = -1
        # 输入 AGC / 监听混音 / 端到端延迟
        self.input_agc = False
        self.agc = None
        self.monitor_enabled = False
        self.monitor_device = None
        self.monitor_volume = 0.8
        self.monitor_stream = None
        self.monitor_queue = queue.Queue(maxsize=8)
        self._loop_lat_ema = None
        self.fade_done.connect(self._on_fade_done, Qt.QueuedConnection)

    def merged_params(self, speaker=None):
        """全局高级参数 + 指定角色的角色级覆盖（默认当前角色）。"""
        params = dict(self._params)
        spk = speaker if speaker is not None else self.current_speaker
        if spk is not None:
            try:
                params.update(spk.pipeline_overrides())
            except Exception:
                pass
        return params

    def _drain_queue(self, q):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _enqueue_input(self, chunk):
        try:
            self.input_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.input_queue.put_nowait(chunk)
            except queue.Full:
                pass
            self._on_worker_xrun()

    def _prop(name):
        def setter(s, v):
            s._params[name] = v
            if name in RESTART_KEYS:
                if s.running:
                    s.status_msg.emit("该参数将在下次启动后生效")
                return
            pipe = getattr(s, "pipeline", None)
            if pipe is None:
                return
            if s.mode == "server" and not pipe.is_connected():
                return
            try:
                pipe.configure(**{name: v})
            except Exception:
                pass
        return property(lambda s: s._params[name], setter)
    block_time = _prop("block_time"); crossfade_time = _prop("crossfade_time")
    extra_time = _prop("extra_time"); f0method = _prop("f0method")
    I_noise_reduce = _prop("I_noise_reduce"); O_noise_reduce = _prop("O_noise_reduce")
    rms_mix_rate = _prop("rms_mix_rate"); threhold = _prop("threhold")
    limiter_enable = _prop("limiter_enable")
    limiter_threshold_db = _prop("limiter_threshold_db")

    def change_pitch(self, val):
        if self.pipeline is not None:
            self.pipeline.change_pitch(val)

    def change_index_rate(self, val):
        if self.pipeline is not None:
            self.pipeline.change_index_rate(val)

    def change_formant(self, val):
        self._formant = float(val)
        if self.pipeline is not None:
            self.pipeline.change_formant(val)

    def set_dry_mix(self, val):
        self.dry_mix = float(val)

    def set_bypass(self, on):
        self.bypass = bool(on)

    def set_input_agc(self, on):
        self.input_agc = bool(on)

    def set_monitor(self, on, device, volume):
        """监听混音：第二输出设备播放变声结果（耳机监听）。

        仅开关/设备变化时重建流；音量在推入监听队列时实时生效。
        """
        need_rebuild = bool(on) != self.monitor_enabled or device != self.monitor_device
        self.monitor_enabled = bool(on)
        self.monitor_device = device
        self.monitor_volume = float(volume)
        if self.running and need_rebuild:
            ok, msg = self._open_monitor()
            if not ok:
                self.status_msg.emit("监听开启失败: " + msg)

    def set_server_url(self, url):
        self.server_url = (url or "").strip() or DEFAULT_SERVER_URL
        if self.mode == "server":
            self.pipeline.set_server_url(self.server_url)

    def _dispose_pipeline(self):
        pipe = self.pipeline
        self.pipeline = None
        if pipe is None:
            return
        try:
            pipe.stop()
        except Exception:
            pass
        if getattr(pipe, "is_remote", False):
            try:
                pipe.disconnect()
            except Exception:
                pass
        else:
            try:
                pipe.unload()
            except Exception:
                pass

    def set_mode(self, mode):
        mode = "local" if mode == "local" else "server"
        if mode == self.mode:
            return
        if self.running:
            self._hard_stop()
        self._dispose_pipeline()
        self.mode = mode
        self.current_speaker = None
        self.pipeline = make_pipeline(
            self.mode, self.server_url, lambda m: self.status_msg.emit(m))
        self.status_msg.emit("已切换到" + ("本地推理" if mode == "local" else "服务器"))

    def resolve_paths(self, speaker):
        if self.mode == "server":
            return (
                to_server_path(speaker.model_path),
                to_server_path(speaker.index_path) if speaker.index_path else "",
            )
        # 本地模式（直连或本机子进程）：路径原样透传。
        # 绝对路径由服务端直接打开；只有文件名时由服务端在自己的目录解析。
        return speaker.model_path or "", speaker.index_path or ""

    def _ensure_connected(self):
        if self.mode == "local" or self.pipeline.is_connected():
            return True
        self.status_msg.emit("重新连接服务器...")
        if not self.pipeline.connect(timeout=5):
            self.status_msg.emit("连接服务器失败")
            return False
        return True

    def _ensure_model(self):
        if self.pipeline.is_loaded:
            return True
        if self.current_speaker is None:
            self.status_msg.emit("请先加载角色模型")
            return False
        self.status_msg.emit("恢复模型加载...")
        mp, ip = self.resolve_paths(self.current_speaker)
        ok = self.pipeline.load_speaker(
            mp, ip,
            self.current_speaker.pitch, self.current_speaker.index_rate,
            formant=self.current_speaker.formant, **self.merged_params())
        if not ok:
            self.status_msg.emit("请先加载角色模型")
        return ok

    def start(self):
        self._fade_epoch += 1
        if self.stream is not None or self.running:
            self._hard_stop()
        if not self._ensure_connected() or not self._ensure_model():
            return
        try:
            params = self.merged_params()
            self.pipeline.configure(**params)
            started = self.pipeline.start(**params)
            if started is False:
                raise RuntimeError("推理未能启动")
            self._in_residual = None
            self._out_residual = np.array([], dtype=np.float32)
            self.xrun_count = 0
            self._drain_queue(self.input_queue)
            self._drain_queue(self.output_queue)

            # 输入 AGC（按当前采样率重建，静音/增益状态清零）
            if self.input_agc:
                self.agc = AutoGain(sample_rate=self.pipeline.samplerate)
            else:
                self.agc = None

            self.worker_thread = InferenceWorkerThread(
                self.pipeline, self.input_queue, self.output_queue, self)
            self.worker_thread.infer_done.connect(self._on_worker_infer_done)
            self.worker_thread.stage_stats.connect(self.stage_stats)
            self.worker_thread.xrun_occurred.connect(self._on_worker_xrun)
            self.worker_thread.need_recover.connect(self._try_recover)
            self.worker_thread.start()

            self.running = True
            ok, msg = self._open_stream()
            if not ok:
                self._hard_stop()
                self.status_msg.emit("启动失败: " + msg)
                self.load_failed.emit(msg)
                return
            self._fade_in_left = int(0.04 * self.pipeline.samplerate)
            self._fade_out_left = 0
            self.stream.start()
            self._open_monitor()
            self.started_ok.emit()
            self.status_msg.emit("实时转换已启动 · " + msg)
        except Exception as e:
            self.status_msg.emit("启动失败: " + str(e))
            self.load_failed.emit(str(e))

    def _close_stream_only(self):
        if self.stream:
            try:
                self.stream.abort()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _open_stream(self):
        block = getattr(self.pipeline, "_block_frame", None)
        if not block:
            raise RuntimeError("推理块大小未知，请先加载角色再启动")
        kwargs = dict(
            callback=self._on_audio,
            blocksize=block,
            samplerate=self.pipeline.samplerate,
            channels=self.pipeline.channels,
            dtype="float32",
            device=(self.input_device, self.output_device),
            latency="low",
        )
        errors = []
        try:
            self.stream = sd.Stream(
                extra_settings=sd.WasapiSettings(exclusive=True), **kwargs)
            return True, "WASAPI 独占 (最低延迟)"
        except Exception as e:
            errors.append(e)
        try:
            try:
                extra = sd.WasapiSettings(exclusive=False, auto_convert=True)
            except TypeError:
                extra = sd.WasapiSettings(exclusive=False)
            self.stream = sd.Stream(extra_settings=extra, **kwargs)
            return True, "WASAPI 共享（独占失败: %s）" % _wasapi_fail_reason(errors[-1])
        except Exception as e:
            errors.append(e)
        try:
            self.stream = sd.Stream(**kwargs)
            return True, "系统共享（%s）" % _wasapi_fail_reason(errors[-1])
        except Exception as e:
            errors.append(e)
        try:
            kwargs["channels"] = 2
            self.stream = sd.Stream(**kwargs)
            return True, "共享模式 2 声道"
        except Exception as e:
            return False, str(e)

    def reopen_stream(self, input_device, output_device):
        """运行中只重建声卡流，不断开服务器、不重载模型。"""
        self.input_device = input_device
        self.output_device = output_device
        if not self.running:
            return True, ""
        self._close_stream_only()
        ok, msg = self._open_stream()
        if not ok:
            self.status_msg.emit("切换设备失败: " + msg)
            self._hard_stop()
            return False, msg
        self._fade_in_left = int(0.04 * self.pipeline.samplerate)
        self._fade_out_left = 0
        try:
            self.stream.start()
        except Exception as e:
            self.status_msg.emit("切换设备失败: " + str(e))
            self._hard_stop()
            return False, str(e)
        self.status_msg.emit("已切换设备 · " + msg)
        return True, msg

    def _on_monitor(self, outdata, frames, times, status):
        """监听流回调：从队列取变声结果播放，空则补零（绝不阻塞）。"""
        try:
            block = self.monitor_queue.get_nowait()
            n = min(len(block), frames)
            outdata[:n, 0] = block[:n]
            if n < frames:
                outdata[n:, 0] = 0
            if outdata.shape[1] > 1:
                outdata[:, 1:] = outdata[:, :1]
        except queue.Empty:
            outdata.fill(0)

    def _open_monitor(self):
        self._close_monitor()
        if not self.monitor_enabled:
            return True, ""
        if self.monitor_device is None:
            return False, "未选择监听输出设备（请在右侧「音质与监听」中选择耳机）"
        block = getattr(self.pipeline, "_block_frame", None)
        if not block:
            return False, "推理块大小未知"
        try:
            self.monitor_stream = sd.OutputStream(
                samplerate=self.pipeline.samplerate,
                blocksize=block,
                channels=1,
                dtype="float32",
                device=self.monitor_device,
                callback=self._on_monitor,
                latency="low",
            )
            self.monitor_stream.start()
            return True, "监听已开启"
        except Exception as e:
            self.monitor_enabled = False
            self.monitor_stream = None
            return False, str(e)

    def _close_monitor(self):
        if self.monitor_stream is not None:
            try:
                self.monitor_stream.abort()
                self.monitor_stream.close()
            except Exception:
                pass
            self.monitor_stream = None
        self._drain_queue(self.monitor_queue)

    def _push_monitor(self, mono_out):
        if not self.monitor_enabled or self.monitor_stream is None:
            return
        try:
            self.monitor_queue.put_nowait(
                (mono_out * np.float32(self.monitor_volume)).astype(np.float32))
        except queue.Full:
            pass

    def _report_loop_latency(self, times):
        """端到端延迟 = 输出 DAC 时刻 - 输入 ADC 时刻（PortAudio 时间戳）。"""
        if times is None:
            return
        try:
            adc = getattr(times, "inputBufferAdcTime", 0.0) or 0.0
            dac = getattr(times, "outputBufferDacTime", 0.0) or 0.0
            if adc <= 0 or dac <= 0:
                return
            ms = (dac - adc) * 1000.0
            if not (0 < ms < 5000):
                return
            if self._loop_lat_ema is None:
                self._loop_lat_ema = ms
            else:
                self._loop_lat_ema += (ms - self._loop_lat_ema) * 0.05
            self.loop_latency.emit(self._loop_lat_ema)
        except Exception:
            pass

    def stop(self):
        if self.running and self.stream is not None and self._fade_out_left <= 0:
            self._fade_epoch += 1
            self._pending_fade_epoch = self._fade_epoch
            self._fade_out_total = max(1, int(0.04 * self.pipeline.samplerate))
            self._fade_out_left = self._fade_out_total
            return
        self._hard_stop()

    def _on_fade_done(self):
        if self._pending_fade_epoch != self._fade_epoch:
            return
        if not self.running:
            return
        self._hard_stop()

    def _try_recover(self):
        if not self.running:
            return
        # 纯本地直连管线（LazyLocalPipeline）不涉及网络，无需恢复；
        # 冻结版本地子进程/远程服务器是网络管线，允许走重连逻辑
        if self.mode == "local" and not getattr(self.pipeline, "is_network", False):
            return
        # 网络管线先强制断开（清掉失效连接），再重连恢复
        if getattr(self.pipeline, "is_network", False):
            try:
                self.pipeline.abort()
            except Exception:
                pass
        if self.pipeline.is_connected() and self.pipeline.is_loaded:
            return
        now = time.time()
        if now - self._last_recover < 2.0:
            return
        self._last_recover = now
        self.status_msg.emit("连接中断，正在重连...")
        if not self._ensure_connected() or not self._ensure_model():
            return
        try:
            self.pipeline.start(**self.merged_params())
            self.status_msg.emit("已重新连上服务器")
        except Exception as e:
            self.status_msg.emit("重连失败: " + str(e))

    def _hard_stop(self):
        self._fade_epoch += 1
        self.running = False
        self._fade_out_left = 0
        self._close_stream_only()
        self._close_monitor()
        if self.worker_thread:
            self.worker_thread.running = False
            if self.worker_thread.isRunning():
                self.worker_thread.wait(1800)
            if self.worker_thread.isRunning() and self.mode == "server":
                try:
                    self.pipeline.abort()
                except Exception:
                    pass
                self.worker_thread.wait(400)
                try:
                    self.pipeline.connect(timeout=3)
                except Exception:
                    pass
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            if self.worker_thread.isRunning():
                # 线程没退出：保留引用等它自然结束，不能销毁运行中的线程
                t = self.worker_thread
                t.finished.connect(lambda: t.deleteLater())
                self._zombie_threads.append(t)
            else:
                self.worker_thread.deleteLater()
            self.worker_thread = None
        self.stopped_ok.emit()
        self.status_msg.emit("已停止")

    def _on_worker_infer_done(self, elapsed_ms, in_db, out_db):
        self.infer_time.emit(elapsed_ms)
        self.rms_levels.emit(in_db, out_db)

    def _on_worker_xrun(self):
        self.xrun_count += 1
        self.xrun_signal.emit(self.xrun_count)

    def _apply_edge_fade(self, outdata):
        n = len(outdata)
        if self._fade_in_left > 0:
            total = max(1, int(0.04 * self.pipeline.samplerate))
            done = total - self._fade_in_left
            gains = np.clip((np.arange(n) + done) / total, 0.0, 1.0).astype(np.float32)
            outdata[:, 0] *= gains
            self._fade_in_left = max(0, self._fade_in_left - n)
        if self._fade_out_left > 0:
            total = max(1, self._fade_out_total)
            remaining = self._fade_out_left
            gains = np.clip((remaining - np.arange(n)) / total, 0.0, 1.0).astype(np.float32)
            outdata[:, 0] *= gains
            self._fade_out_left = max(0, self._fade_out_left - n)
            if self._fade_out_left <= 0:
                self.fade_done.emit()

    def _on_audio(self, indata, outdata, frames, times, status):
        """音频回调：弹性缓冲输入凑整块，输出按需切分，推理不及时输出静音。"""
        if not self.running:
            outdata.fill(0)
            return
        try:
            mono = indata[:, 0] if indata.ndim > 1 else indata
            n_needed = len(outdata)
            if self.bypass:
                n = min(len(mono), n_needed)
                outdata[:n, 0] = mono[:n]
                if n < n_needed:
                    outdata[n:, 0] = 0
                if outdata.shape[1] > 1:
                    outdata[:, 1:] = outdata[:, :1]
                self._apply_edge_fade(outdata)
                self._push_monitor(outdata[:, 0].copy())
                self._report_loop_latency(times)
                return

            # 输入 AGC：电平归一化（本地/服务器模式都在发送前生效）
            if self.input_agc and self.agc is not None:
                mono = self.agc.process(mono)

            in_block = self.pipeline._block_frame
            if len(mono) != in_block:
                self._in_residual = (
                    np.concatenate([self._in_residual, mono])
                    if self._in_residual is not None else mono.copy()
                )
                while len(self._in_residual) >= in_block:
                    chunk = self._in_residual[:in_block].astype(np.float32)
                    self._in_residual = self._in_residual[in_block:]
                    self._enqueue_input(chunk)
            else:
                self._enqueue_input(mono.astype(np.float32, copy=True))

            while len(self._out_residual) < n_needed:
                try:
                    block = self.output_queue.get_nowait()
                    if block.ndim > 1:
                        block = block[:, 0]
                    self._out_residual = np.concatenate(
                        [self._out_residual, block.astype(np.float32)]
                    )
                except queue.Empty:
                    break
            if len(self._out_residual) >= n_needed:
                outdata[:, 0] = self._out_residual[:n_needed]
                self._out_residual = self._out_residual[n_needed:]
                if outdata.shape[1] > 1:
                    outdata[:, 1:] = outdata[:, :1]
            else:
                outdata.fill(0.0)
                self._on_worker_xrun()
            dry = float(self.dry_mix)
            if dry > 0:
                n = min(len(mono), n_needed)
                outdata[:n, 0] = outdata[:n, 0] * (1.0 - dry) + mono[:n] * dry
                if outdata.shape[1] > 1:
                    outdata[:, 1:] = outdata[:, :1]
            self._apply_edge_fade(outdata)
            self._push_monitor(outdata[:, 0].copy())
            self._report_loop_latency(times)
        except Exception:
            outdata.fill(0)


# ==============================================================================
# 主窗口 - 三栏布局
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC 实时变声")
        self.setMinimumSize(1040, 660)
        self.setAcceptDrops(True)
        self.speaker_mgr = SpeakerManager()
        self._settings = load_user_settings()
        self._live_guard = False
        self._device_guard = False
        self._dev_fp = ()
        self._load_gen = 0
        url = self._settings.get("server_url") or DEFAULT_SERVER_URL
        mode = self._settings.get("infer_mode") or "local"
        self.engine = VCEngine(mode=mode, server_url=url)
        self.engine.status_msg.connect(self._on_status)
        self.engine.infer_time.connect(self._on_infer_time)
        self.engine.started_ok.connect(self._on_started)
        self.engine.stopped_ok.connect(self._on_stopped)
        self.engine.load_failed.connect(self._on_start_failed)
        self.engine.rms_levels.connect(self._on_rms_levels)
        self.engine.xrun_signal.connect(self._on_xrun)
        self.engine.loop_latency.connect(self._on_loop_latency)
        self.engine.stage_stats.connect(self._on_stage_stats)
        self._build_ui()
        self._apply_saved_params()
        self._rd(restore=False)
        self._restore_devices()
        self._rl()
        self._set_light(LIGHT_GRAY, "未加载模型")
        self._dev_timer = QTimer(self)
        self._dev_timer.setInterval(2500)
        self._dev_timer.timeout.connect(self._poll_devices)
        self._dev_timer.start()
        # 冻结版：定时刷新本地推理安装状态
        self._install_timer = QTimer(self)
        self._install_timer.setInterval(5000)
        self._install_timer.timeout.connect(self._refresh_local_install)
        self._install_timer.start()

    def _on_rms_levels(self, in_db, out_db):
        self.in_meter.set_level(in_db)
        self.out_meter.set_level(out_db)

    def _on_xrun(self, xruns):
        self.xrun_label.setText(f"卡顿 {xruns}")
        self.xrun_label.setStyleSheet("font-size:12px;font-weight:700;color:#b91c1c;" if xruns > 0 else "font-size:12px;font-weight:700;color:#6b7c8a;")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        pth_file = None
        index_file = None
        for u in urls:
            path = u.toLocalFile()
            if path.lower().endswith(".pth"):
                pth_file = path
            elif path.lower().endswith(".index"):
                index_file = path
        if pth_file:
            name = Path(pth_file).stem
            cfg = SpeakerConfig(name=name, model_path=pth_file, index_path=index_file or "", pitch=0, index_rate=0.6)
            dlg = SpeakerDialog(self, cfg)
            if dlg.exec():
                self.speaker_mgr.add(dlg.result)
                self._rl()
                self.sc.setCurrentIndex(len(self.speaker_mgr.speakers) - 1)

    def _lbl(self, text):
        w = QLabel(text)
        w.setObjectName("fieldLabel")
        return w

    def _build_ui(self):
        self.resize(1240, 760)
        self.setMinimumSize(1120, 680)
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(10)

        header = QFrame(); header.setObjectName("header")
        top = QHBoxLayout(header)
        top.setContentsMargins(14, 10, 14, 10)
        top.setSpacing(12)
        title = QLabel("RVC 实时变声")
        title.setObjectName("appTitle")
        top.addWidget(title)
        self.in_meter = VUMeterWidget(title="输入")
        self.out_meter = VUMeterWidget(title="输出")
        top.addWidget(self.in_meter)
        top.addWidget(self.out_meter)
        top.addStretch()
        self.light = QLabel()
        self.light.setFixedSize(10, 10)
        self.light.setStyleSheet(f"background:{LIGHT_GRAY};border-radius:5px;")
        top.addWidget(self.light)
        self.state_label = QLabel("未加载模型")
        self.state_label.setObjectName("chip")
        self.state_label.setStyleSheet("font-size:12px;font-weight:700;color:#6b7c8a;")
        top.addWidget(self.state_label)
        self.latency_label = QLabel("延迟 --ms")
        self.latency_label.setStyleSheet("font-size:12px;font-weight:700;color:#15803d;")
        top.addWidget(self.latency_label)
        self.e2e_label = QLabel("端到端 --ms")
        self.e2e_label.setStyleSheet("font-size:12px;font-weight:700;color:#7c3aed;")
        top.addWidget(self.e2e_label)
        self.xrun_label = QLabel("卡顿 0")
        self.xrun_label.setStyleSheet("font-size:12px;font-weight:700;color:#6b7c8a;")
        top.addWidget(self.xrun_label)
        root.addWidget(header)

        sp = QSplitter(Qt.Horizontal)
        sp.addWidget(self._build_left())
        sp.addWidget(self._build_mid())
        sp.addWidget(self._build_right())
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        sp.setStretchFactor(2, 0)
        sp.setSizes([300, 580, 330])
        root.addWidget(sp, 1)

        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)

    def _build_left(self):
        g = QGroupBox("角色")
        l = QVBoxLayout(g)
        l.setContentsMargins(10, 16, 10, 10)
        l.setSpacing(8)

        self.sc = create_styled_combo(max_visible=12)
        self.sc.setMinimumHeight(34)
        self.sc.currentIndexChanged.connect(self._sel)
        l.addWidget(self.sc)

        br = QHBoxLayout(); br.setSpacing(6)
        for t, fn in [("添加", self._a), ("编辑", self._e), ("删除", self._d)]:
            b = QPushButton(t); b.setObjectName("btnGhost"); b.clicked.connect(fn)
            br.addWidget(b)
        l.addLayout(br)

        self.cur_card = QFrame(); self.cur_card.setObjectName("roleCard")
        cv = QVBoxLayout(self.cur_card)
        cv.setContentsMargins(10, 10, 10, 10); cv.setSpacing(4)
        self.cur_name = QLabel("未选择角色")
        self.cur_name.setStyleSheet("font-size:14px;font-weight:700;color:#134e4a;")
        self.cur_model = QLabel("")
        self.cur_model.setStyleSheet("font-size:11px;color:#6b7c8a;")
        self.cur_info = QLabel("")
        self.cur_info.setStyleSheet("font-size:11px;color:#6b7c8a;")
        cv.addWidget(self.cur_name); cv.addWidget(self.cur_model); cv.addWidget(self.cur_info)
        l.addWidget(self.cur_card)

        live = QGroupBox("实时调节")
        gl = QGridLayout(live)
        gl.setContentsMargins(8, 14, 8, 8)
        gl.setHorizontalSpacing(8); gl.setVerticalSpacing(6)
        self.live_pitch = QSpinBox(); self.live_pitch.setRange(-36, 36)
        self.live_pitch.setSuffix(" 半音")
        self.live_pitch.valueChanged.connect(self._on_live_pitch)
        self.live_index = QDoubleSpinBox(); self.live_index.setRange(0.0, 1.0)
        self.live_index.setSingleStep(0.1)
        self.live_index.valueChanged.connect(self._on_live_index)
        self.live_formant = QDoubleSpinBox(); self.live_formant.setRange(-12.0, 12.0)
        self.live_formant.setSingleStep(0.5)
        self.live_formant.valueChanged.connect(self._on_live_formant)
        gl.addWidget(self._lbl("音高"), 0, 0); gl.addWidget(self.live_pitch, 0, 1)
        gl.addWidget(self._lbl("检索"), 1, 0); gl.addWidget(self.live_index, 1, 1)
        gl.addWidget(self._lbl("共振峰"), 2, 0); gl.addWidget(self.live_formant, 2, 1)
        self.live_dry = QDoubleSpinBox(); self.live_dry.setRange(0.0, 1.0)
        self.live_dry.setSingleStep(0.1)
        self.live_dry.setToolTip("0=只听变声，1=只听原声")
        self.live_dry.valueChanged.connect(self._on_live_dry)
        gl.addWidget(self._lbl("原声混合"), 3, 0); gl.addWidget(self.live_dry, 3, 1)
        self.bypass = QCheckBox("旁通（听原声）")
        self.bypass.setToolTip("快捷键 Ctrl+B")
        self.bypass.toggled.connect(self._on_bypass)
        gl.addWidget(self.bypass, 4, 0, 1, 2)
        l.addWidget(live)
        l.addStretch(1)
        return g

    def _build_mid(self):
        g = QGroupBox("转换")
        l = QVBoxLayout(g)
        l.setContentsMargins(12, 16, 12, 12)
        l.setSpacing(10)

        ml = QHBoxLayout(); ml.setSpacing(12)
        ml.addWidget(self._lbl("模式"))
        self.mode_local = QRadioButton("本地推理")
        self.mode_server = QRadioButton("服务器")
        if self.engine.mode == "server":
            self.mode_server.setChecked(True)
        else:
            self.mode_local.setChecked(True)
        self.mode_local.toggled.connect(lambda on: on and self._apply_mode("local"))
        self.mode_server.toggled.connect(lambda on: on and self._apply_mode("server"))
        ml.addWidget(self.mode_local)
        ml.addWidget(self.mode_server)
        ml.addStretch(1)
        l.addLayout(ml)

        self.server_row = QWidget()
        sl = QHBoxLayout(self.server_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)
        sl.addWidget(self._lbl("服务器"))
        self.server_edit = QLineEdit(self._settings.get("server_url") or DEFAULT_SERVER_URL)
        self.server_edit.setPlaceholderText("ws://主机:8765")
        sl.addWidget(self.server_edit, 1)
        self.conn_btn = QPushButton("连接")
        self.conn_btn.setObjectName("btnConnect")
        self.conn_btn.setToolTip("改完地址后点这里，按新地址重连（不重新加载角色）")
        self.conn_btn.clicked.connect(self._connect_server)
        sl.addWidget(self.conn_btn)
        l.addWidget(self.server_row)
        self._sync_mode_ui()

        # 冻结版：本地推理一键安装入口（源码运行时隐藏）
        self.local_install_row = QWidget()
        il = QHBoxLayout(self.local_install_row)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(8)
        self.local_install_lbl = QLabel("")
        self.local_install_lbl.setWordWrap(True)
        self.local_install_lbl.setStyleSheet("font-size:12px;color:#b45309;")
        self.local_install_btn = QPushButton("安装本地推理")
        self.local_install_btn.setObjectName("btnConnect")
        self.local_install_btn.setToolTip("下载并安装本机推理环境（需 NVIDIA 显卡与网络）")
        self.local_install_btn.clicked.connect(self._install_local)
        il.addWidget(self.local_install_lbl, 1)
        il.addWidget(self.local_install_btn)
        l.addWidget(self.local_install_row)
        self._refresh_local_install()

        dl = QGridLayout(); dl.setHorizontalSpacing(8); dl.setVerticalSpacing(8)
        dl.addWidget(self._lbl("输入"), 0, 0)
        self.ic = DeviceCombo(direction="input")
        self.ic.setToolTip("WASAPI 独占要求输入输出同一组 API；选择后会自动对齐另一侧")
        self.ic.currentIndexChanged.connect(lambda: self._on_device_changed("input"))
        dl.addWidget(self.ic, 0, 1)
        dl.addWidget(self._lbl("输出"), 1, 0)
        self.oc = DeviceCombo(direction="output")
        self.oc.setToolTip("WASAPI 独占要求输入输出同一组 API；选择后会自动对齐另一侧")
        self.oc.currentIndexChanged.connect(lambda: self._on_device_changed("output"))
        dl.addWidget(self.oc, 1, 1)
        rb = QPushButton("刷新设备")
        rb.setObjectName("btnGhost")
        rb.clicked.connect(lambda: self._rd())
        dl.addWidget(rb, 2, 1, Qt.AlignRight)
        dl.setColumnStretch(1, 1)
        l.addLayout(dl)

        # 设备错位/异常提示（常驻显示，比状态栏更醒目）
        self.dev_hint = QLabel("")
        self.dev_hint.setWordWrap(True)
        self.dev_hint.setVisible(False)
        l.addWidget(self.dev_hint)
        l.addStretch(1)

        self.sb = QPushButton("启动变声")
        self.sb.setObjectName("btnStart")
        self.sb.setProperty("state", "off")
        self.sb.setMinimumHeight(48)
        self.sb.clicked.connect(self._tg)
        l.addWidget(self.sb)
        self.rec_btn = QPushButton("录音测试 10 秒")
        self.rec_btn.setToolTip("录 10 秒，用当前角色变声后保存并播放")
        self.rec_btn.clicked.connect(self._rec)
        l.addWidget(self.rec_btn)
        return g

    def _build_right(self):
        box = QWidget()
        root = QVBoxLayout(box)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # ── 高级参数 ──
        g1 = QGroupBox("高级参数")
        l = QGridLayout(g1)
        l.setContentsMargins(10, 16, 10, 10)
        l.setHorizontalSpacing(8); l.setVerticalSpacing(6)

        self.fc = create_styled_combo(max_visible=10); self.fc.addItems(["rmvpe", "fcpe", "pm"])
        self.fc.currentTextChanged.connect(lambda v: setattr(self.engine, "f0method", v))
        self.fc.setToolTip("基频提取: rmvpe 最准, pm 最快")
        self.bs = QDoubleSpinBox(); self.bs.setRange(0.05, 2.0); self.bs.setSingleStep(0.05)
        self.bs.setValue(0.08)
        self.bs.setToolTip("越小延迟越低。运行中修改将在下次启动后生效")
        self.bs.valueChanged.connect(lambda v: setattr(self.engine, "block_time", v))
        self.xs = QDoubleSpinBox(); self.xs.setRange(0.01, 0.5); self.xs.setSingleStep(0.01)
        self.xs.setValue(0.02)
        self.xs.setToolTip("运行中修改将在下次启动后生效")
        self.xs.valueChanged.connect(lambda v: setattr(self.engine, "crossfade_time", v))
        self.es = QDoubleSpinBox(); self.es.setRange(0.5, 10.0); self.es.setSingleStep(0.5)
        self.es.setValue(1.5)
        self.es.setToolTip("越大音色越稳、延迟越高。运行中修改将在下次启动后生效")
        self.es.valueChanged.connect(lambda v: setattr(self.engine, "extra_time", v))
        self.ts = QSpinBox(); self.ts.setRange(-80, 0); self.ts.setValue(-50)
        self.ts.setToolTip("低于此音量视为静音。-80 关闭门限")
        self.ts.valueChanged.connect(lambda v: setattr(self.engine, "threhold", v))
        self.rs = QDoubleSpinBox(); self.rs.setRange(0.0, 1.0); self.rs.setSingleStep(0.1)
        self.rs.setValue(0.3)
        self.rs.setToolTip("0=完全跟随输入音量，1=只保留变声自身音量")
        self.rs.valueChanged.connect(lambda v: setattr(self.engine, "rms_mix_rate", v))
        rows = [
            ("F0", self.fc),
            ("块大小", self.bs),
            ("交叉淡入", self.xs),
            ("额外推理", self.es),
            ("静音阈值", self.ts),
            ("音量保留", self.rs),
        ]
        for i, (name, w) in enumerate(rows):
            l.addWidget(self._lbl(name), i, 0)
            l.addWidget(w, i, 1)
        self.inc = QCheckBox("输入降噪")
        self.inc.setToolTip("运行中修改将在下次启动后生效")
        self.inc.toggled.connect(lambda v: setattr(self.engine, "I_noise_reduce", v))
        self.onc = QCheckBox("输出降噪")
        self.onc.setToolTip("运行中修改将在下次启动后生效")
        self.onc.toggled.connect(lambda v: setattr(self.engine, "O_noise_reduce", v))
        nr = QHBoxLayout(); nr.setSpacing(12)
        nr.addWidget(self.inc); nr.addWidget(self.onc); nr.addStretch()
        l.addLayout(nr, len(rows), 0, 1, 2)

        # 输出保护（直流高通 + 软限幅）
        pr = QHBoxLayout(); pr.setSpacing(8)
        self.limiter_cb = QCheckBox("输出保护")
        self.limiter_cb.setToolTip("直流高通 + 软限幅，防止爆音/直流偏移（实时生效）")
        self.limiter_cb.setChecked(True)
        self.limiter_cb.toggled.connect(lambda v: setattr(self.engine, "limiter_enable", v))
        pr.addWidget(self.limiter_cb)
        self.limiter_th = QDoubleSpinBox(); self.limiter_th.setRange(-12.0, 0.0)
        self.limiter_th.setSingleStep(0.5); self.limiter_th.setSuffix(" dB")
        self.limiter_th.setValue(-1.0)
        self.limiter_th.setToolTip("起限阈值，-1 dB 为推荐值")
        self.limiter_th.valueChanged.connect(lambda v: setattr(self.engine, "limiter_threshold_db", v))
        pr.addWidget(self.limiter_th); pr.addStretch()
        l.addLayout(pr, len(rows) + 1, 0, 1, 2)

        # 分阶段耗时（本地模式）
        self.st_lbl = QLabel("阶段耗时: --")
        self.st_lbl.setStyleSheet("font-size:11px;color:#6b7c8a;")
        l.addWidget(self.st_lbl, len(rows) + 2, 0, 1, 2)
        l.setColumnStretch(1, 1)
        root.addWidget(g1)

        # ── 音质与监听 ──
        g2 = QGroupBox("音质与监听")
        m = QGridLayout(g2)
        m.setContentsMargins(10, 16, 10, 10)
        m.setHorizontalSpacing(8); m.setVerticalSpacing(6)

        self.agc_cb = QCheckBox("输入自动增益 (AGC)")
        self.agc_cb.setToolTip("输入电平归一化，说话人远近变化时音色更稳定（实时生效）")
        self.agc_cb.toggled.connect(self._on_agc)
        m.addWidget(self.agc_cb, 0, 0, 1, 2)

        m.addWidget(self._lbl("预设"), 1, 0)
        pbtn_row = QHBoxLayout(); pbtn_row.setSpacing(6)
        self._preset_map = load_presets()
        self.preset_cb = create_styled_combo()
        for p in self._preset_map:
            self.preset_cb.addItem(p["name"])
        pbtn_row.addWidget(self.preset_cb, 1)
        pb_apply = QPushButton("应用"); pb_apply.setObjectName("btnGhost")
        pb_apply.setToolTip("应用所选预设（需重启生效的参数会提示）")
        pb_apply.clicked.connect(self._apply_preset)
        pbtn_row.addWidget(pb_apply)
        pb_save = QPushButton("保存"); pb_save.setObjectName("btnGhost")
        pb_save.setToolTip("把当前参数保存为用户预设")
        pb_save.clicked.connect(self._save_preset)
        pbtn_row.addWidget(pb_save)
        m.addLayout(pbtn_row, 1, 1)

        mr = QHBoxLayout(); mr.setSpacing(8)
        self.monitor_cb = QCheckBox("监听")
        self.monitor_cb.setToolTip("把变声结果同时播放到第二输出设备（如耳机）")
        self.monitor_cb.toggled.connect(self._on_monitor_toggle)
        mr.addWidget(self.monitor_cb)
        self.monitor_vol = QSlider(Qt.Horizontal)
        self.monitor_vol.setRange(0, 100); self.monitor_vol.setValue(80)
        self.monitor_vol.setToolTip("监听音量")
        self.monitor_vol.valueChanged.connect(self._on_monitor_vol)
        mr.addWidget(self.monitor_vol, 1)
        m.addLayout(mr, 2, 0, 1, 2)

        self.mc = DeviceCombo(direction="output")
        self.mc.setToolTip("监听输出设备（可不同于主输出）")
        self.mc.currentIndexChanged.connect(lambda: self._on_monitor_changed())
        m.addWidget(self.mc, 3, 0, 1, 2)
        m.setColumnStretch(1, 1)
        root.addWidget(g2)
        root.addStretch(1)
        return box

    # ── 事件处理 ──
    def _show_dev_hint(self, text, warn=True):
        icon = "⚠ " if warn else "ℹ "
        color = "#b45309" if warn else "#1d4ed8"
        bg = "#fef3c7" if warn else "#dbeafe"
        self.dev_hint.setText(icon + text)
        self.dev_hint.setStyleSheet(
            f"font-size:12px;color:{color};background:{bg};"
            "border:1px solid #e5e7eb;border-radius:6px;padding:6px 8px;")
        self.dev_hint.setVisible(True)

    def _clear_dev_hint(self):
        self.dev_hint.setVisible(False)

    def _on_device_changed(self, which="input"):
        if self._device_guard:
            return
        self._align_device_apis(which)
        if not self.engine.running:
            self._persist_settings()
            return
        in_id, out_id, err = self._resolve_selected(reinit=False)
        if err:
            self._show_dev_hint(err)
            self.status_bar.showMessage(err, 8000)
            return
        ok, msg = self.engine.reopen_stream(in_id, out_id)
        if not ok:
            self._show_dev_hint(msg)
            self.status_bar.showMessage("切换设备失败: " + msg, 8000)
        else:
            self._clear_dev_hint()
            self.status_bar.showMessage("已切换设备 · " + msg, 5000)
        self._persist_settings()

    def _align_device_apis(self, changed):
        """输入/输出 API 不同时自动对齐另一侧；无法对齐时给出明确提示。"""
        in_api = self.ic.currentDeviceApi()
        out_api = self.oc.currentDeviceApi()
        if in_api is None or out_api is None or in_api == out_api:
            return
        self._device_guard = True
        try:
            if changed == "input":
                name = self.oc.currentDeviceName()
                ok = self.oc.selectByNameApi(name, in_api) or self.oc.selectFirstWithApi(in_api)
                shown = self.oc.currentDeviceName() or "默认设备"
                if ok:
                    self._show_dev_hint(f"输入与输出不在同一驱动组，输出已自动对齐为「{shown}」", warn=False)
                    self.status_bar.showMessage("输出已自动改为同组设备: " + shown, 5000)
                else:
                    self._show_dev_hint(
                        "输入/输出不在同一驱动组，且找不到可对齐的输出设备。"
                        "请手动把输入输出选为同组设备（WASAPI 独占要求同组 API）。")
                    self.status_bar.showMessage("输入输出 API 不一致且无法自动对齐", 8000)
            else:
                name = self.ic.currentDeviceName()
                ok = self.ic.selectByNameApi(name, out_api) or self.ic.selectFirstWithApi(out_api)
                shown = self.ic.currentDeviceName() or "默认设备"
                if ok:
                    self._show_dev_hint(f"输入与输出不在同一驱动组，输入已自动对齐为「{shown}」", warn=False)
                    self.status_bar.showMessage("输入已自动改为同组设备: " + shown, 5000)
                else:
                    self._show_dev_hint(
                        "输入/输出不在同一驱动组，且找不到可对齐的输入设备。"
                        "请手动把输入输出选为同组设备（WASAPI 独占要求同组 API）。")
                    self.status_bar.showMessage("输入输出 API 不一致且无法自动对齐", 8000)
        finally:
            self._device_guard = False

    def _resolve_selected(self, reinit=False):
        if reinit:
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
        try:
            devs = sd.query_devices()
            apis = sd.query_hostapis()
        except Exception as e:
            return None, None, "无法读取音频设备: " + str(e)

        def find(name, api, ch):
            if not name:
                return None
            for i, d in enumerate(devs):
                if d["name"] == name and d.get("hostapi") == api and d[ch] > 0:
                    return i
            for i, d in enumerate(devs):
                if d["name"] == name and d[ch] > 0:
                    return i
            return "missing"

        in_id = find(self.ic.currentDeviceName(), self.ic.currentDeviceApi(), "max_input_channels")
        out_id = find(self.oc.currentDeviceName(), self.oc.currentDeviceApi(), "max_output_channels")
        if in_id == "missing":
            return None, None, f"输入设备「{self.ic.currentDeviceName()}」已不存在，请点「刷新设备」后重选"
        if out_id == "missing":
            return None, None, f"输出设备「{self.oc.currentDeviceName()}」已不存在，请点「刷新设备」后重选"
        if in_id is not None and out_id is not None:
            if devs[in_id]["hostapi"] != devs[out_id]["hostapi"]:
                ain = apis[devs[in_id]["hostapi"]]["name"]
                aout = apis[devs[out_id]["hostapi"]]["name"]
                return None, None, (
                    f"输入（{ain}）与输出（{aout}）不是同一组 API，"
                    "WASAPI 独占要求同组；请手动选择同组设备")
        return in_id, out_id, None

    def _rd(self, restore=True, reinit=True):
        in_name, in_api = self.ic.currentDeviceName(), self.ic.currentDeviceApi()
        out_name, out_api = self.oc.currentDeviceName(), self.oc.currentDeviceApi()
        mon_name, mon_api = self.mc.currentDeviceName(), self.mc.currentDeviceApi()
        self.ic.blockSignals(True)
        self.oc.blockSignals(True)
        self.mc.blockSignals(True)
        try:
            if reinit:
                sd._terminate()
                sd._initialize()
            devs = sd.query_devices()
            apis = sd.query_hostapis()
            self.ic.populate(devs, apis, "max_input_channels")
            self.oc.populate(devs, apis, "max_output_channels")
            self.mc.populate(devs, apis, "max_output_channels")
            if restore:
                self.ic.selectByNameApi(in_name, in_api)
                self.oc.selectByNameApi(out_name, out_api)
                self.mc.selectByNameApi(mon_name, mon_api)
            self._dev_fp = _device_fingerprint()
        except Exception as e:
            print("Refresh devices error:", e)
            self.status_bar.showMessage("刷新设备失败: " + str(e), 6000)
        finally:
            self.ic.blockSignals(False)
            self.oc.blockSignals(False)
            self.mc.blockSignals(False)
        if restore:
            self._apply_monitor()

    def _poll_devices(self):
        fp = _device_fingerprint()
        if not fp or fp == self._dev_fp:
            return
        self._rd(restore=True, reinit=False)
        if not self.engine.running:
            return
        _in, _out, err = self._resolve_selected(reinit=False)
        if err:
            self.engine._hard_stop()
            self._show_dev_hint("音频设备已断开: " + err)
            self.status_bar.showMessage("音频设备已断开: " + err, 8000)

    def _preferred_speaker_index(self):
        saved = self._settings.get("speaker")
        names = [s.name for s in self.speaker_mgr.speakers]
        if saved in names:
            return names.index(saved)
        return 0

    def _rl(self):
        self.sc.blockSignals(True)
        self.sc.clear()
        for s in self.speaker_mgr.speakers:
            self.sc.addItem("  " + s.name)
        if self.speaker_mgr.speakers:
            if self.engine.current_speaker:
                names = [s.name for s in self.speaker_mgr.speakers]
                idx = names.index(self.engine.current_speaker.name) if self.engine.current_speaker.name in names else 0
            else:
                idx = self._preferred_speaker_index()
            idx = max(0, min(idx, len(self.speaker_mgr.speakers) - 1))
            self.sc.setCurrentIndex(idx)
        self.sc.blockSignals(False)
        if self.speaker_mgr.speakers:
            self._sel(idx)
        else:
            self.cur_name.setText("未选择角色")
            self.cur_model.setText("")
            self.cur_info.setText("")

    def _sel(self, row):
        if row < 0 or row >= len(self.speaker_mgr.speakers): return
        s = self.speaker_mgr.speakers[row]
        self.cur_name.setText(s.name)
        self.cur_model.setText("模型: " + Path(s.model_path).name)
        self.cur_info.setText(
            f"音高 {s.pitch:+d}  检索 {s.index_rate:.1f}  共振峰 {s.formant:+.1f}"
            if getattr(s, "formant", 0.0) != 0.0
            else f"音高 {s.pitch:+d}  检索 {s.index_rate:.1f}")
        self._sync_live_sliders(s)
        if self.engine.current_speaker is s:
            self._set_light(LIGHT_GREEN, "就绪")
            self.sb.setEnabled(True)
            return
        if self.engine.running:
            self.engine._hard_stop()
        self._set_light(LIGHT_YELLOW, "加载模型中...")
        self.sb.setEnabled(False)
        self._start_loading(s)

    def _start_loading(self, speaker):
        # 旧加载线程保留引用防 GC（PySide6 中 deleteLater 与信号竞争会崩溃）
        if not hasattr(self, "_loaders"):
            self._loaders = []
        if hasattr(self, "_loader") and self._loader is not None and self._loader.isRunning():
            try:
                self._loader.quit()
            except Exception:
                pass
            self._loaders.append(self._loader)   # 防 GC，不删
            if len(self._loaders) > 3:
                self._loaders.pop(0)
        self._load_gen += 1
        self._loader = ModelLoader(self.engine, speaker, self._load_gen, parent=self)
        self._loader.finished_ok.connect(self._on_loaded)
        self._loader.failed.connect(self._on_load_failed)
        self._loader.start()

    def _on_loaded(self, speaker, gen=0):
        if gen != self._load_gen:
            return
        self.engine.current_speaker = speaker
        self._sync_live_sliders(speaker)
        self.engine.change_pitch(speaker.pitch)
        self.engine.change_index_rate(speaker.index_rate)
        self._set_light(LIGHT_GREEN, "就绪")
        self.sb.setEnabled(True)
        self.status_bar.showMessage("模型已加载: " + speaker.name, 5000)
        self._persist_settings()

    def _on_load_failed(self, err, gen=0):
        if gen != self._load_gen:
            return
        self._set_light(LIGHT_RED, "加载失败")
        self.sb.setEnabled(True)
        text = (err or "未知错误").strip()
        if len(text) > 300:
            text = text[:300] + "…"
        self.status_bar.showMessage("加载失败: " + text, 8000)
        QMessageBox.warning(
            self, "模型加载失败", text + NL + NL +
            "请确认：模型文件完整且路径正确；服务器模式下请确认服务器已启动。")

    def _on_start_failed(self, msg):
        """启动变声失败：常驻提示 + 弹窗。"""
        text = (msg or "未知错误").strip()
        self._show_dev_hint("启动失败: " + text[:200])
        self.status_bar.showMessage("启动失败: " + text, 8000)
        QMessageBox.warning(self, "启动失败", text[:300])

    def _set_light(self, color, text):
        self.light.setStyleSheet(f"background:{color};border-radius:5px;")
        self.state_label.setText(text)
        self.state_label.setStyleSheet(
            "font-size:12px;font-weight:700;color:" + ("#134e4a" if color != LIGHT_GRAY else "#6b7c8a"))

    def _on_started(self):
        self._set_light(LIGHT_GREEN, "运行中")
        self._clear_dev_hint()
        self.sb.setText("停止变声")
        self.sb.setProperty("state", "on")
        self.sb.style().unpolish(self.sb); self.sb.style().polish(self.sb)

    def _on_stopped(self):
        self._set_light(LIGHT_GRAY, "已停止")
        self.sb.setText("启动变声")
        self.sb.setProperty("state", "off")
        self.sb.style().unpolish(self.sb); self.sb.style().polish(self.sb)

    def _on_status(self, m):
        self.status_bar.showMessage(m, 5000)

    def _rec(self):
        """录音测试：录 10 秒 → 当前角色变声 → 保存 wav 并播放"""
        if not self.engine.pipeline.is_loaded:
            QMessageBox.warning(self, "提示", "请先加载角色模型")
            return
        if self.engine.running:
            QMessageBox.warning(self, "提示", "请先停止实时转换")
            return
        if hasattr(self, "_rec_thread") and self._rec_thread is not None and self._rec_thread.isRunning():
            QMessageBox.information(self, "提示", "录音正在进行中，请稍候...")
            return
        # 用界面选的输入设备（按 名字+API 重解析，防同名设备匹配错）
        rec_dev = None
        in_name = self.ic.currentDeviceName()
        in_api = self.ic.currentDeviceApi()
        if in_name:
            import sounddevice as sd
            sd._terminate(); sd._initialize()
            for i, d in enumerate(sd.query_devices()):
                if d['name'] == in_name and d['hostapi'] == in_api and d['max_input_channels'] > 0:
                    rec_dev = i; break
        self.rec_btn.setEnabled(False)
        self.rec_btn.setText("录音中…")
        self._rec_thread = RecThread(self.engine, 10, rec_dev, self)
        self._rec_thread.done.connect(self._rec_done)
        self._rec_thread.start()

    def _rec_done(self, path, err):
        self.rec_btn.setEnabled(True)
        self.rec_btn.setText("录音测试 10 秒")
        if err:
            QMessageBox.warning(self, "录音失败", err)
            return
        msg = "已保存: " + path + NL + NL + "正在用系统播放器打开，请听变声效果。"
        QMessageBox.information(self, "录音完成", msg)
        try:
            import os
            os.startfile(os.path.abspath(path))
        except Exception:
            pass

    def _on_infer_time(self, ms):
        c = "#27ae60" if ms < 50 else ("#f39c12" if ms < 100 else "#e74c3c")
        self.latency_label.setText(f"延迟 {ms}ms")
        self.latency_label.setStyleSheet(f"font-size:12px;font-weight:700;color:{c};")

    def _tg(self):
        if self.engine.running:
            self.engine.stop()
            return
        if not self.engine.pipeline.is_loaded:
            QMessageBox.warning(self, "提示", "请先选择角色模型")
            return
        in_id, out_id, err = self._resolve_selected(reinit=True)
        if err:
            self._show_dev_hint(err)
            QMessageBox.warning(self, "设备", err)
            return
        self.engine.input_device = in_id
        self.engine.output_device = out_id
        self.engine.start()
        self._persist_settings()

    # ── 角色增删改 ──
    def _a(self):
        d = SpeakerDialog(self)
        if d.exec() and d.result:
            self.speaker_mgr.add(d.result)
            self._rl()
            self.sc.setCurrentIndex(len(self.speaker_mgr.speakers) - 1)

    def _e(self):
        r = self.sc.currentIndex()
        if r < 0 or r >= len(self.speaker_mgr.speakers):
            return QMessageBox.warning(self, "提示", "请先选择一个角色")
        d = SpeakerDialog(self, self.speaker_mgr.speakers[r])
        if d.exec() and d.result:
            self.speaker_mgr.update(r, d.result)
            self._rl()
            self.sc.setCurrentIndex(r)

    def _d(self):
        r = self.sc.currentIndex()
        if r < 0 or r >= len(self.speaker_mgr.speakers):
            return QMessageBox.warning(self, "提示", "请先选择一个角色")
        n = self.speaker_mgr.speakers[r].name
        if QMessageBox.question(self, "确认", f"确定删除角色「{n}」?") == QMessageBox.Yes:
            self.speaker_mgr.remove(r)
            self._rl()

    def closeEvent(self, e):
        self._persist_settings()
        if hasattr(self, "_loader") and self._loader is not None and self._loader.isRunning():
            try:
                self._loader.quit()
                if not self._loader.wait(1500):
                    t = self._loader
                    t.finished.connect(lambda: t.deleteLater())
                    if not hasattr(self, "_zombie_loaders"):
                        self._zombie_loaders = []
                    self._zombie_loaders.append(t)
            except Exception:
                pass
        if self.engine.running:
            self.engine._hard_stop()
        # 冻结版：退出时回收本机子进程推理服务
        pipe = getattr(self.engine, "pipeline", None)
        if pipe is not None and getattr(pipe, "stop_server", None) is not None:
            try:
                pipe.stop_server()
            except Exception:
                pass
        e.accept()

    def keyPressEvent(self, e):
        if e.modifiers() & Qt.ControlModifier and e.key() == Qt.Key_B:
            self.bypass.toggle()
            e.accept()
            return
        if e.key() == Qt.Key_F5:
            self._tg()
            e.accept()
            return
        super().keyPressEvent(e)

    def _sync_live_sliders(self, speaker):
        if not hasattr(self, "live_pitch"):
            return
        self._live_guard = True
        self.live_pitch.setValue(int(speaker.pitch))
        self.live_index.setValue(float(speaker.index_rate))
        formant = float(getattr(speaker, "formant", 0.0) or 0.0)
        self.live_formant.setValue(formant)
        if hasattr(self, "live_dry"):
            self.live_dry.setValue(float(self._settings.get("dry_mix", 0.0)))
        self._live_guard = False
        self.engine.change_formant(formant)
        self.engine.set_dry_mix(float(self._settings.get("dry_mix", 0.0)))

    def _on_live_pitch(self, val):
        if self._live_guard:
            return
        self.engine.change_pitch(val)
        s = self.engine.current_speaker
        if s is not None:
            s.pitch = int(val)
            self.speaker_mgr.save()
            self.cur_info.setText(f"音高 {s.pitch:+d}  检索 {s.index_rate:.1f}")
        self._persist_settings()

    def _on_live_index(self, val):
        if self._live_guard:
            return
        self.engine.change_index_rate(val)
        s = self.engine.current_speaker
        if s is not None:
            s.index_rate = float(val)
            self.speaker_mgr.save()
            self.cur_info.setText(f"音高 {s.pitch:+d}  检索 {s.index_rate:.1f}")
        self._persist_settings()

    def _on_live_formant(self, val):
        if self._live_guard:
            return
        self.engine.change_formant(val)
        self._settings["formant"] = float(val)
        self._persist_settings()

    def _on_live_dry(self, val):
        if self._live_guard:
            return
        self.engine.set_dry_mix(val)
        self._settings["dry_mix"] = float(val)
        self._persist_settings()

    def _on_bypass(self, on):
        self.engine.set_bypass(on)
        self.status_bar.showMessage("已旁通，输出原声" if on else "旁通已关，输出变声", 3000)

    # ── 音质产品化：AGC / 监听 / 预设 / 阶段耗时 ──
    def _on_agc(self, on):
        self.engine.set_input_agc(on)
        self._persist_settings()

    def _on_monitor_toggle(self, on):
        self._apply_monitor()

    def _on_monitor_vol(self, val):
        self._apply_monitor()

    def _on_monitor_changed(self):
        self._apply_monitor()

    def _apply_monitor(self):
        dev_id = self._monitor_device_id()
        self.engine.set_monitor(
            self.monitor_cb.isChecked(), dev_id, self.monitor_vol.value() / 100.0)
        self._persist_settings()
        if self.monitor_cb.isChecked() and dev_id is None:
            self.status_bar.showMessage(
                "请先在「监听」下方选择监听输出设备（如耳机）", 8000)

    def _monitor_device_id(self):
        name = self.mc.currentDeviceName()
        api = self.mc.currentDeviceApi()
        if not name:
            return None
        try:
            devs = sd.query_devices()
        except Exception:
            return None
        for i, d in enumerate(devs):
            if d["name"] == name and d.get("hostapi") == api and d["max_output_channels"] > 0:
                return i
        for i, d in enumerate(devs):
            if d["name"] == name and d["max_output_channels"] > 0:
                return i
        return None

    def _on_loop_latency(self, ms):
        self.e2e_label.setText(f"端到端 {ms:.0f}ms")

    def _on_stage_stats(self, s):
        try:
            self.st_lbl.setText(
                "阶段耗时: 特征 %.1f · 检索 %.1f · 音高 %.1f · 模型 %.1f ms"
                % (s.get("feature", 0.0), s.get("index", 0.0),
                   s.get("pitch", 0.0), s.get("model", 0.0)))
        except Exception:
            pass

    def _apply_preset(self):
        idx = self.preset_cb.currentIndex()
        if idx < 0 or idx >= len(self._preset_map):
            return
        params = self._preset_map[idx].get("params", {})
        mapping = {
            "block_time": self.bs, "crossfade_time": self.xs,
            "extra_time": self.es, "threhold": self.ts,
            "rms_mix_rate": self.rs,
        }
        for k, w in mapping.items():
            if k in params:
                try:
                    w.setValue(params[k])
                except Exception:
                    pass
        if "f0method" in params:
            i = self.fc.findText(str(params["f0method"]))
            if i >= 0:
                self.fc.setCurrentIndex(i)
        if "I_noise_reduce" in params:
            self.inc.setChecked(bool(params["I_noise_reduce"]))
        if "O_noise_reduce" in params:
            self.onc.setChecked(bool(params["O_noise_reduce"]))
        if "formant" in params and hasattr(self, "live_formant"):
            self.live_formant.setValue(float(params["formant"]))
        if "dry_mix" in params and hasattr(self, "live_dry"):
            self.live_dry.setValue(float(params["dry_mix"]))
        self._persist_settings()
        self.status_bar.showMessage("已应用预设: " + self._preset_map[idx]["name"], 4000)

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "保存预设", "预设名称:")
        if not ok or not name.strip():
            return
        params = self._current_preset_params()
        entry = {"name": name.strip(), "params": params}
        names = [p["name"] for p in self._preset_map]
        if entry["name"] in names:
            self._preset_map[names.index(entry["name"])] = entry
            sel = names.index(entry["name"])
        else:
            self._preset_map.append(entry)
            sel = len(self._preset_map) - 1
        builtin_names = {p["name"] for p in BUILTIN_PRESETS}
        save_presets([p for p in self._preset_map if p["name"] not in builtin_names])
        self.preset_cb.blockSignals(True)
        self.preset_cb.clear()
        for p in self._preset_map:
            self.preset_cb.addItem(p["name"])
        self.preset_cb.setCurrentIndex(sel)
        self.preset_cb.blockSignals(False)
        self.status_bar.showMessage("预设已保存: " + entry["name"], 4000)

    def _current_preset_params(self):
        return {
            "block_time": self.engine.block_time,
            "crossfade_time": self.engine.crossfade_time,
            "extra_time": self.engine.extra_time,
            "f0method": self.engine.f0method,
            "I_noise_reduce": self.engine.I_noise_reduce,
            "O_noise_reduce": self.engine.O_noise_reduce,
            "rms_mix_rate": self.engine.rms_mix_rate,
            "threhold": self.engine.threhold,
            "formant": float(self.live_formant.value()) if hasattr(self, "live_formant") else 0.0,
            "dry_mix": float(self.live_dry.value()) if hasattr(self, "live_dry") else 0.0,
        }

    def _apply_mode(self, mode):
        if self.engine.mode == mode:
            self._sync_mode_ui()
            return
        if self.engine.running:
            self.engine._hard_stop()
        self._load_gen += 1
        self.engine.set_mode(mode)
        self._sync_mode_ui()
        self._persist_settings()
        if self.speaker_mgr.speakers:
            self.engine.current_speaker = None
            row = self.sc.currentIndex()
            if row >= 0:
                self._sel(row)
        self.status_bar.showMessage(
            "已切换到本地推理" if mode == "local" else "已切换到服务器", 4000)

    def _sync_mode_ui(self):
        server = self.engine.mode == "server"
        if hasattr(self, "server_row"):
            self.server_row.setVisible(server)

    def _connect_server(self):
        url = self.server_edit.text().strip() or DEFAULT_SERVER_URL
        self.engine.set_server_url(url)
        self._persist_settings()
        ok = False
        try:
            ok = bool(self.engine.pipeline.connect(timeout=5))
        except Exception as e:
            ok = False
            self.status_bar.showMessage("连接失败: " + _friendly_net_error(e), 8000)
            QMessageBox.warning(
                self, "连接服务器",
                "无法连接 " + url + NL + NL + _friendly_net_error(e))
            return
        if ok:
            self.status_bar.showMessage("已连接服务器: " + url, 5000)
        else:
            QMessageBox.warning(
                self, "连接服务器",
                "无法连接 " + url + NL + NL +
                "请确认：服务器已启动、地址与端口正确（默认 8765）、防火墙已放行。")

    # ── 冻结版：本地推理安装 ──
    def _refresh_local_install(self):
        if not is_frozen():
            self.local_install_row.setVisible(False)
            return
        self.local_install_row.setVisible(True)
        if runtime_installed():
            self.local_install_lbl.setText("本地推理环境已安装（约 3.5GB）")
            self.local_install_btn.setText("重新安装")
        else:
            self.local_install_lbl.setText(
                "本地推理未安装：需要 NVIDIA 显卡，点击右侧按钮开始（需联网，约 3.5GB）")
            self.local_install_btn.setText("安装本地推理")

    def _install_local(self):
        root = package_root()
        bat = root / "install_local.bat"
        if not bat.is_file():
            QMessageBox.warning(self, "安装本地推理",
                                "未找到 install_local.bat（安装包不完整）")
            return
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "RVC 本地推理安装", str(bat)],
                cwd=str(root))
        except Exception as e:
            QMessageBox.warning(self, "安装本地推理", "无法打开安装窗口: " + str(e))
            return
        self.status_bar.showMessage(
            "安装窗口已打开。完成后本程序会自动检测，若未生效请重启本程序", 12000)

    def _apply_saved_params(self):
        s = self._settings
        mapping = [
            ("block_time", self.bs.setValue, float),
            ("crossfade_time", self.xs.setValue, float),
            ("extra_time", self.es.setValue, float),
            ("threhold", self.ts.setValue, int),
            ("rms_mix_rate", self.rs.setValue, float),
        ]
        for key, setter, cast in mapping:
            if key in s:
                try:
                    setter(cast(s[key]))
                except Exception:
                    pass
        if "f0method" in s:
            i = self.fc.findText(str(s["f0method"]))
            if i >= 0:
                self.fc.setCurrentIndex(i)
        self.inc.setChecked(bool(s.get("I_noise_reduce", False)))
        self.onc.setChecked(bool(s.get("O_noise_reduce", False)))
        if hasattr(self, "live_dry"):
            self.live_dry.setValue(float(s.get("dry_mix", 0.0)))
        self.live_formant.setValue(float(s.get("formant", 0.0)))
        self.engine.set_dry_mix(float(s.get("dry_mix", 0.0)))
        self.engine.change_formant(float(s.get("formant", 0.0)))
        # 音质产品化：AGC / 输出保护 / 监听
        self.agc_cb.setChecked(bool(s.get("input_agc", False)))
        self.limiter_cb.setChecked(bool(s.get("limiter_enable", True)))
        if "limiter_threshold_db" in s:
            try:
                self.limiter_th.setValue(float(s["limiter_threshold_db"]))
            except Exception:
                pass
        self.monitor_cb.setChecked(bool(s.get("monitor_enabled", False)))
        if "monitor_volume" in s:
            try:
                self.monitor_vol.setValue(int(float(s["monitor_volume"]) * 100))
            except Exception:
                pass

    def _restore_devices(self):
        self.ic.blockSignals(True)
        self.oc.blockSignals(True)
        self.mc.blockSignals(True)
        self.ic.selectByNameApi(self._settings.get("input_name"), self._settings.get("input_api"))
        self.oc.selectByNameApi(self._settings.get("output_name"), self._settings.get("output_api"))
        self.mc.selectByNameApi(self._settings.get("monitor_name"), self._settings.get("monitor_api"))
        self.ic.blockSignals(False)
        self.oc.blockSignals(False)
        self.mc.blockSignals(False)
        self._apply_monitor()

    def _persist_settings(self):
        data = dict(self._settings)
        if hasattr(self, "server_edit"):
            data["server_url"] = self.server_edit.text().strip() or DEFAULT_SERVER_URL
        data["infer_mode"] = self.engine.mode
        # 设备字段：下拉框未填充（启动早期）时不写回 None，避免覆盖已保存的选择
        in_name = self.ic.currentDeviceName()
        out_name = self.oc.currentDeviceName()
        mon_name = self.mc.currentDeviceName()
        device_update = {}
        if in_name:
            device_update["input_name"] = in_name
            device_update["input_api"] = self.ic.currentDeviceApi()
        if out_name:
            device_update["output_name"] = out_name
            device_update["output_api"] = self.oc.currentDeviceApi()
        if mon_name:
            device_update["monitor_name"] = mon_name
            device_update["monitor_api"] = self.mc.currentDeviceApi()
        data.update(device_update)
        data.update({
            "speaker": self.engine.current_speaker.name if self.engine.current_speaker else data.get("speaker", ""),
            "block_time": self.engine.block_time,
            "crossfade_time": self.engine.crossfade_time,
            "extra_time": self.engine.extra_time,
            "f0method": self.engine.f0method,
            "threhold": self.engine.threhold,
            "rms_mix_rate": self.engine.rms_mix_rate,
            "I_noise_reduce": self.engine.I_noise_reduce,
            "O_noise_reduce": self.engine.O_noise_reduce,
            "formant": float(self.live_formant.value()) if hasattr(self, "live_formant") else 0.0,
            "dry_mix": float(self.live_dry.value()) if hasattr(self, "live_dry") else 0.0,
            "input_agc": bool(self.agc_cb.isChecked()),
            "monitor_enabled": bool(self.monitor_cb.isChecked()),
            "monitor_volume": self.monitor_vol.value() / 100.0,
            "limiter_enable": bool(self.limiter_cb.isChecked()),
            "limiter_threshold_db": float(self.limiter_th.value()),
        })
        try:
            save_user_settings(data)
            self._settings = data
        except Exception:
            pass

# ==============================================================================
# 角色编辑对话框
# ==============================================================================
class SpeakerDialog(QDialog):
    def __init__(self, parent=None, speaker=None):
        super().__init__(parent)
        self.setWindowTitle("编辑角色" if speaker else "添加角色")
        self.setMinimumWidth(520); self.result = None
        s = speaker or SpeakerConfig()
        l = QFormLayout(self); l.setSpacing(10)

        self.ne = QLineEdit(s.name)
        self.ne.setPlaceholderText("例如: 女声A")
        l.addRow("角色名称:", self.ne)

        mr = QHBoxLayout()
        self.me = QLineEdit(s.model_path)
        self.me.setPlaceholderText("选择 .pth 模型文件")
        mr.addWidget(self.me, 1)
        mb = QPushButton("浏览...")
        mb.clicked.connect(lambda: self._br("模型文件 (*.pth)", WEIGHTS_DIR, self.me))
        mr.addWidget(mb)
        ms = QPushButton("从服务器获取")
        ms.setToolTip("列出当前模式可用的 .pth（本地目录或服务器）")
        ms.clicked.connect(self._from_server)
        mr.addWidget(ms)
        l.addRow("模型文件:", mr)

        ir = QHBoxLayout()
        self.ie = QLineEdit(s.index_path)
        self.ie.setPlaceholderText("可选 .index 文件")
        ir.addWidget(self.ie, 1)
        ib = QPushButton("浏览...")
        ib.clicked.connect(lambda: self._br("索引文件 (*.index)", PROJECT_ROOT, self.ie))
        ir.addWidget(ib)
        l.addRow("索引文件:", ir)

        self.si = QSpinBox(); self.si.setRange(0, 200); self.si.setValue(s.speaker_id)
        l.addRow("说话人ID:", self.si)

        self.ps = QSpinBox(); self.ps.setRange(-36, 36); self.ps.setValue(s.pitch)
        self.ps.setSuffix(" 半音")
        self.ps.setToolTip("男转女 +12, 女转男 -12")
        l.addRow("音高偏移:", self.ps)

        self.irs = QDoubleSpinBox(); self.irs.setRange(0.0, 1.0); self.irs.setSingleStep(0.1)
        self.irs.setValue(s.index_rate)
        l.addRow("检索比例:", self.irs)

        self.fs = QDoubleSpinBox(); self.fs.setRange(-12.0, 12.0); self.fs.setSingleStep(0.5)
        self.fs.setValue(getattr(s, "formant", 0.0) or 0.0)
        self.fs.setToolTip("共振峰偏移（半音），0 为不偏移")
        l.addRow("共振峰:", self.fs)

        self.fmc = QComboBox(); self.fmc.addItems(["rmvpe", "fcpe", "pm"])
        i = self.fmc.findText(getattr(s, "f0method", "rmvpe") or "rmvpe")
        self.fmc.setCurrentIndex(i if i >= 0 else 0)
        l.addRow("F0 方法:", self.fmc)

        nr = QHBoxLayout(); nr.setSpacing(12)
        self.inc2 = QCheckBox("输入降噪"); self.inc2.setChecked(bool(getattr(s, "I_noise_reduce", False)))
        self.onc2 = QCheckBox("输出降噪"); self.onc2.setChecked(bool(getattr(s, "O_noise_reduce", False)))
        nr.addWidget(self.inc2); nr.addWidget(self.onc2); nr.addStretch()
        l.addRow("降噪:", nr)

        b = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        b.accepted.connect(self._ok); b.rejected.connect(self.reject)
        l.addRow(b)

    def _from_server(self):
        """从服务器拉取模型文件名列表，点选填入"""
        engine = getattr(self.parent(), "engine", None)
        if engine is None:
            return
        try:
            models = engine.pipeline.list_models()
        except Exception:
            models = []
        if not models:
            QMessageBox.warning(self, "提示",
                "无法获取服务器模型列表" + NL + "请确认已连接服务器（状态栏显示已连接）")
            return
        name, ok = QInputDialog.getItem(self, "服务器模型", "选择模型文件:", models, 0, False)
        if ok and name:
            self.me.setText(name)

    @staticmethod
    def _br(filt, start, target):
        p, _ = QFileDialog.getOpenFileName(None, "选择文件", str(start), filt)
        if p: target.setText(p)

    def _ok(self):
        n = self.ne.text().strip(); mp = self.me.text().strip()
        if not n:
            return QMessageBox.warning(self, "提示", "请输入角色名称")
        if not mp:
            # 网络模式：模型在服务器上，本地只需填服务器上的模型文件名
            return QMessageBox.warning(self, "提示",
                "请输入模型文件名（可点「从服务器获取」选择，或填服务器上的文件名，如 thchs_v2.pth）")
        # 不检查本地文件存在：推理在服务器，本地路径只取文件名发送
        self.result = SpeakerConfig(
            n, mp, self.ie.text().strip(),
            self.si.value(), self.ps.value(), self.irs.value(),
            formant=self.fs.value(), f0method=self.fmc.currentText(),
            I_noise_reduce=self.inc2.isChecked(),
            O_noise_reduce=self.onc2.isChecked())
        self.accept()

# ==============================================================================
# 入口
# ==============================================================================
if __name__ == "__main__":
    setup_logging()
    sys.excepthook = _excepthook
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_QSS)
    w = MainWindow(); w.show()
    sys.exit(app.exec())
