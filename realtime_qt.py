#!/usr/bin/env python3
"""
RVC 实时变声 - 桌面客户端
============================
架构: MainWindow(UI) -> VCEngine(音频+信号) -> 本地 RVCPipeline / 远程 RVCClient
源码本地默认进程内推理；打包 exe 才走本机子进程。
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
    QInputDialog, QScrollArea, QTabWidget,
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
        if "ASIO" in n:
            return 0
        if "WASAPI" in n:
            return 1
        if "WDM" in n or "KS" in n:
            return 2
        if "DIRECTSOUND" in n:
            return 3
        if "MME" in n:
            return 4
        return 5

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


class SpeakerCardList(QWidget):
    """角色卡片列表，接口对齐 QComboBox 的常用方法。"""
    currentIndexChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self._index = -1
        self._blocked = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setMinimumHeight(140)
        self._scroll.setMaximumHeight(220)
        self._inner = QWidget()
        self._box = QVBoxLayout(self._inner)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(6)
        self._box.addStretch(1)
        self._scroll.setWidget(self._inner)
        root.addWidget(self._scroll)

    def blockSignals(self, on):
        self._blocked = bool(on)
        return super().blockSignals(on)

    def clear(self):
        while self._box.count() > 1:
            item = self._box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._items = []
        self._index = -1

    def addItem(self, text):
        card = QFrame()
        card.setObjectName("speakerCard")
        card.setProperty("selected", False)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)
        title = QLabel(str(text).strip())
        title.setObjectName("cardTitle")
        title.setStyleSheet("font-size:13px;font-weight:700;")
        lay.addWidget(title)
        idx = len(self._items)
        card.mousePressEvent = lambda e, i=idx: self.setCurrentIndex(i)
        self._box.insertWidget(self._box.count() - 1, card)
        self._items.append(card)

    def currentIndex(self):
        return self._index

    def setCurrentIndex(self, idx):
        if idx < 0 or idx >= len(self._items):
            return
        if idx == self._index:
            return
        self._index = idx
        for i, card in enumerate(self._items):
            card.setProperty("selected", i == idx)
            card.style().unpolish(card)
            card.style().polish(card)
        if not self._blocked:
            self.currentIndexChanged.emit(idx)

    def count(self):
        return len(self._items)


class CableWizard(QDialog):
    """检测虚拟声卡；没有则给出安装入口。"""

    def __init__(self, parent, input_combo, output_combo):
        super().__init__(parent)
        self.setWindowTitle("虚拟声卡向导")
        self.setMinimumWidth(560)
        self.ic = input_combo
        self.oc = output_combo
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        # 1. 状态
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#334155;font-size:13px;")
        lay.addLayout(self._section("状态", self.hint))

        # 2. 检测到的虚拟声卡
        self.list_lbl = QLabel()
        self.list_lbl.setWordWrap(True)
        self.list_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addLayout(self._section("检测到的虚拟声卡", self.list_lbl))

        # 3. 路由自检（卡片样式，单独成块）
        self.route_lbl = QLabel()
        self.route_lbl.setWordWrap(True)
        lay.addLayout(self._section("路由自检", self.route_lbl))

        # 安装入口
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        for name, url in INSTALL_URLS:
            b = QPushButton("打开 " + name)
            b.setObjectName("btnGhost")
            b.clicked.connect(lambda _, u=url: open_install_page(u))
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        # 应用 / 刷新
        apply_row = QHBoxLayout()
        apply_row.setSpacing(8)
        self.use_out = QPushButton("设为输出")
        self.use_out.setObjectName("btnConnect")
        self.use_out.setToolTip("把 RVC 输出设备设为这条虚拟线")
        self.use_out.clicked.connect(self._apply_out)
        self.use_in = QPushButton("设为输入")
        self.use_in.setObjectName("btnGhost")
        self.use_in.setToolTip("把 RVC 输入设备设为这条虚拟线")
        self.use_in.clicked.connect(self._apply_in)
        refresh_btn = QPushButton("重新检测")
        refresh_btn.setObjectName("btnGhost")
        refresh_btn.clicked.connect(self.refresh)
        apply_row.addWidget(self.use_out)
        apply_row.addWidget(self.use_in)
        apply_row.addWidget(refresh_btn)
        lay.addLayout(apply_row)

        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        close.accepted.connect(self.accept)
        lay.addWidget(close)
        self._found = []
        self.refresh()

    def _section(self, title, widget):
        """区块：小标题 + 内容，视觉分隔，避免多个标签挤成一团。"""
        box = QVBoxLayout()
        box.setSpacing(5)
        hdr = QLabel(title)
        hdr.setStyleSheet("font-size:12px;font-weight:700;color:#64748b;")
        box.addWidget(hdr)
        box.addWidget(widget)
        return box

    def refresh(self):
        try:
            devs = sd.query_devices()
            apis = sd.query_hostapis()
        except Exception as e:
            self.hint.setText("无法读取音频设备: " + str(e))
            self._found = []
            return
        self._found = find_virtual_devices(devs, apis)
        check = route_self_check(devs, apis)
        color = "#059669" if check["ok"] else "#d97706"
        self.route_lbl.setText(check["message"])
        self.route_lbl.setStyleSheet(
            "color:%s;font-weight:600;background:#f8fafc;border:1px solid #e2e8f0;"
            "border-radius:6px;padding:8px 10px;" % color
        )
        if self._found:
            lines = []
            for d in self._found:
                lines.append(
                    "· %s  [%s]  入%d / 出%d"
                    % (d["name"], d["api"] or "?", d["in_ch"], d["out_ch"])
                )
            self.hint.setText("已检测到虚拟声卡。把它设为「输出」，游戏/Discord 里选同一条虚拟线当麦克风。")
            self.list_lbl.setText("\n".join(lines))
            self.use_out.setEnabled(True)
            self.use_in.setEnabled(any(d["in_ch"] > 0 for d in self._found))
        else:
            self.hint.setText(
                "没有检测到 VB-Cable / VoiceMeeter 等虚拟声卡。\n"
                "安装后点「刷新设备」，再把输出选成 CABLE Input，其它软件选 CABLE Output 当麦克风。"
            )
            self.list_lbl.setText("推荐：VB-Audio Cable（免费）。")
            self.use_out.setEnabled(False)
            self.use_in.setEnabled(False)

    def _pick(self, need_in=False, need_out=False):
        for d in self._found:
            if need_out and d["out_ch"] > 0:
                return d
            if need_in and d["in_ch"] > 0:
                return d
        return self._found[0] if self._found else None

    def _apply_out(self):
        d = self._pick(need_out=True)
        if not d:
            return
        self.oc.selectByNameApi(d["name"])
        self.accept()

    def _apply_in(self):
        d = self._pick(need_in=True)
        if not d:
            return
        self.ic.selectByNameApi(d["name"])
        self.accept()

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")

from worker.rvc_client import RVCClient
from worker.local_server import is_frozen, package_root, runtime_installed
from tools.audio_meter import VUMeterWidget, SpectrumWidget, calc_rms_db, spec_bins
from tools.virtual_cable import find_virtual_devices, is_virtual_name, INSTALL_URLS, open_install_page, route_self_check
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
    is_network = False

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

    @property
    def last_stage_ms(self):
        if self._real is None:
            return {}
        return getattr(self._real, "last_stage_ms", {}) or {}

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


def _use_local_subprocess():
    """打包 exe 默认子进程隔离；源码默认进程内。环境变量可强制切换。"""
    if os.environ.get("RVC_DIRECT_LOCAL") == "1":
        return False
    if os.environ.get("RVC_LOCAL_SUBPROCESS") == "1":
        return True
    return bool(is_frozen())


def make_pipeline(mode, server_url, on_status):
    if mode == "local":
        if _use_local_subprocess():
            from worker.local_server import LocalServerPipeline
            return LocalServerPipeline(on_status=on_status)
        return LazyLocalPipeline(on_status)
    # 不在构造时 connect：远程未启动会卡住 UI 数秒
    return RVCClient(server_url=server_url, on_status=on_status)

NL = chr(10)
SETTINGS_FILE = PROJECT_ROOT / "user_settings.json"
PRESETS_FILE = PROJECT_ROOT / "presets.json"
DEFAULT_SERVER_URL = "ws://192.168.1.28:8765"
SERVER_ROOT = "/home/songwang/Retrieval-based-Voice-Conversion-WebUI"
SERVER_MODEL_DIR = SERVER_ROOT + "/assets/weights"
SERVER_INDEX_DIR = SERVER_ROOT + "/logs/thchs_v2"
RESTART_KEYS = ("block_time", "crossfade_time", "extra_time", "I_noise_reduce", "O_noise_reduce")
DEFAULT_PARAMS = {
    "block_time": 0.06,
    "crossfade_time": 0.02,
    "extra_time": 0.8,
    "f0method": "rmvpe",
    "I_noise_reduce": False,
    "O_noise_reduce": False,
    "rms_mix_rate": 0.3,
    "threhold": -50,
    "limiter_enable": True,
    "limiter_threshold_db": -1.0,
    "hf_mix_rate": 0.2,
    "presence": 0.10,
    "deesser_enable": False,
    "vad_enable": False,
    "vad_threshold": 0.50,
}

# 场景预设：低延迟 / 高音质 / 游戏语音 / 唱歌
BUILTIN_PRESETS = [
    {
        "name": "低延迟",
        "params": {
            "block_time": 0.04, "crossfade_time": 0.01, "extra_time": 0.6,
            "f0method": "rmvpe", "rms_mix_rate": 0.5, "threhold": -50,
            "I_noise_reduce": False, "O_noise_reduce": False,
        },
    },
    {
        "name": "高音质",
        "params": {
            "block_time": 0.08, "crossfade_time": 0.03, "extra_time": 1.2,
            "f0method": "rmvpe", "rms_mix_rate": 0.3, "threhold": -55,
            "I_noise_reduce": False, "O_noise_reduce": False,
        },
    },
    {
        "name": "游戏语音",
        "params": {
            "block_time": 0.05, "crossfade_time": 0.015, "extra_time": 0.7,
            "f0method": "rmvpe", "rms_mix_rate": 0.6, "threhold": -45,
            "I_noise_reduce": False, "O_noise_reduce": False,
        },
    },
    {
        "name": "通话",
        "params": {
            "block_time": 0.06, "crossfade_time": 0.02, "extra_time": 0.8,
            "f0method": "rmvpe", "rms_mix_rate": 0.5, "threhold": -50,
            "I_noise_reduce": False, "O_noise_reduce": False,
            "hf_mix_rate": 0.2, "presence": 0.10,
            "deesser_enable": False, "vad_enable": False,
        },
    },
    {
        "name": "唱歌",
        "params": {
            "block_time": 0.06, "crossfade_time": 0.02, "extra_time": 1.2,
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

LIGHT_QSS = """
QMainWindow, QDialog {
    background-color: #f8fafc;
    color: #0f172a;
}
QWidget {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 13px;
    color: #1e293b;
}

/* 顶部 Header */
QFrame#header {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}
QLabel#appTitle {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.2px;
}

/* 核心面板卡片 */
QGroupBox {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 12px 10px 12px;
    background-color: #ffffff;
    font-weight: 600;
    font-size: 13px;
    color: #0f172a;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    background-color: #ffffff;
    color: #0f172a;
    font-weight: 600;
}

QLabel#fieldLabel {
    color: #64748b;
    font-size: 12px;
    font-weight: 500;
}

/* 下拉菜单 */
QComboBox {
    combobox-popup: 0;
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 24px;
    font-size: 13px;
    color: #111827;
}
QComboBox:hover {
    border-color: #9ca3af;
}
QComboBox:focus {
    border-color: #0f766e;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 4px;
    outline: none;
    max-height: 280px;
    selection-background-color: #f3f4f6;
    selection-color: #111827;
}
QComboBox QAbstractItemView::item {
    min-height: 24px;
    padding: 3px 6px;
    border-radius: 4px;
}

/* 滚动区域与滚动条 */
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 6px;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #e2e8f0;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #cbd5e1;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* 输入框与数值调节器 */
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 4px 6px;
    min-height: 24px;
    color: #111827;
    font-size: 13px;
}
QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #9ca3af;
}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #0f766e;
}

/* 滑块 */
QSlider::groove:horizontal {
    height: 4px;
    background: #e5e7eb;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #0f766e;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 1.5px solid #0f766e;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

/* 单选与复选框 */
QCheckBox, QRadioButton {
    spacing: 6px;
    color: #374151;
    font-weight: 500;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #9ca3af;
    border-radius: 4px;
    background-color: #ffffff;
}
QCheckBox::indicator:hover {
    border-color: #0f766e;
}
QCheckBox::indicator:checked {
    background-color: #0f766e;
    border-color: #0f766e;
}
QRadioButton::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #9ca3af;
    border-radius: 8px;
    background-color: #ffffff;
}
QRadioButton::indicator:hover {
    border-color: #0f766e;
}
QRadioButton::indicator:checked {
    background-color: #0f766e;
    border-color: #0f766e;
}

/* 按钮系统 */
QPushButton {
    background-color: #ffffff;
    color: #374151;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 5px 12px;
    font-weight: 500;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #f9fafb;
    border-color: #9ca3af;
    color: #111827;
}
QPushButton:pressed {
    background-color: #f3f4f6;
}
QPushButton#btnGhost {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    color: #4b5563;
    padding: 4px 10px;
}
QPushButton#btnGhost:hover {
    background-color: #f3f4f6;
    border-color: #d1d5db;
    color: #111827;
}
QPushButton#btnConnect {
    background-color: #0f766e;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 5px 12px;
    font-weight: 600;
    min-width: 64px;
}
QPushButton#btnConnect:hover {
    background-color: #115e59;
}

/* 启动/停止主操作按钮 */
QPushButton#btnStart {
    font-size: 15px;
    font-weight: 700;
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
}
QPushButton#btnStart[state="off"] {
    background-color: #15803d;
    color: #ffffff;
}
QPushButton#btnStart[state="off"]:hover {
    background-color: #16a34a;
}
QPushButton#btnStart[state="on"] {
    background-color: #dc2626;
    color: #ffffff;
}
QPushButton#btnStart[state="on"]:hover {
    background-color: #ef4444;
}
QPushButton#btnStart:disabled {
    background-color: #e5e7eb;
    color: #9ca3af;
}

/* 角色卡片 */
QFrame#roleCard {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}
QFrame#speakerCard {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}
QFrame#speakerCard[selected="true"] {
    background-color: #ecfdf5;
    border: 1px solid #0f766e;
}

/* 状态栏 */
QStatusBar {
    background-color: #ffffff;
    color: #6b7280;
    border-top: 1px solid #e5e7eb;
}
QSplitter::handle {
    background: #e5e7eb;
    width: 1px;
}

/* 分段页签 QTabWidget */
QTabWidget::pane {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background-color: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background-color: #f1f5f9;
    color: #64748b;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 4px;
}
QTabBar::tab:hover {
    background-color: #e2e8f0;
    color: #0f172a;
}
QTabBar::tab:selected {
    background-color: #ffffff;
    color: #0f766e;
    border-bottom: 2px solid #0f766e;
}
"""

STYLE_QSS = LIGHT_QSS
UI_DARK = False

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
    spectrum = Signal(object)
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
        try:
            self.spectrum.emit(spec_bins(out_block))
        except Exception:
            pass

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


class EngineStartThread(QThread):
    """后台执行启动/重建/停机的阻塞部分（网络等待、CUDA 预热、声卡打开、等推理退出）。

    UI 线程只负责发起与信号响应，绝不进入 wait——杜绝"未响应"。
    """

    def __init__(self, engine, action, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.action = action

    def run(self):
        if self.action == "start":
            self.engine._start_blocking()
        elif self.action == "reopen":
            self.engine._reopen_blocking()
        elif self.action == "stop":
            self.engine._hard_stop()


# ==============================================================================
# 推理引擎（本地）
# ==============================================================================
class VCEngine(QObject):
    status_msg = Signal(str); infer_time = Signal(int)
    started_ok = Signal(); stopped_ok = Signal()
    load_failed = Signal(str)
    rms_levels = Signal(float, float)  # in_db, out_db
    spectrum = Signal(object)
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
        self.input_queue = queue.Queue(maxsize=4)
        self.output_queue = queue.Queue(maxsize=4)
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._in_n = 0
        self._out_buf = np.zeros(0, dtype=np.float32)
        self._out_n = 0
        self._in_pool = []
        self._pool_i = 0
        self._last_out = np.zeros(0, dtype=np.float32)
        self._last_out_n = 0
        self._mon_scratch = np.zeros(0, dtype=np.float32)
        self.worker_thread = None
        self._zombie_threads = []   # 超时未退出的线程，等 finished 再删，避免销毁运行中的线程
        self.xrun_count = 0
        self._xrun_emitted = 0
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
        self._stop_requested = False
        self._emit_stopped = True
        self._fade_done_flag = False
        self._true_e2e_ms = 0.0
        self._slow_streak = 0
        self._adapted = False
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
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(50)
        self._stats_timer.timeout.connect(self._flush_callback_stats)
        self._stats_timer.start()

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
    hf_mix_rate = _prop("hf_mix_rate")
    presence = _prop("presence")
    deesser_enable = _prop("deesser_enable")
    vad_enable = _prop("vad_enable")
    vad_threshold = _prop("vad_threshold")

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
        if self.running or self.stream is not None:
            self._emit_stopped = False
            try:
                self._hard_stop()
            finally:
                self._emit_stopped = True
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

    def _engine_busy(self):
        t = getattr(self, "_start_thread", None)
        return t is not None and t.isRunning()

    def start(self):
        """异步启动：网络等待 / CUDA 预热 / 声卡打开全部在后台线程，UI 永不阻塞。"""
        self._fade_epoch += 1
        self._stop_requested = False
        if self._engine_busy():
            self.status_msg.emit("正在处理，请稍候...")
            return
        self._start_thread = EngineStartThread(self, "start", parent=self)
        self._start_thread.start()

    def _alloc_rt_buffers(self, block):
        """音频回调用的预分配缓冲：回调内不再 numpy 拼接/新建数组。"""
        cap = max(int(block) * 4, 1024)
        self._in_buf = np.zeros(cap, dtype=np.float32)
        self._in_n = 0
        self._out_buf = np.zeros(cap, dtype=np.float32)
        self._out_n = 0
        # 槽位多于队列深度，避免回调复用工人尚未读完的数组
        self._in_pool = [np.zeros(block, dtype=np.float32) for _ in range(32)]
        self._pool_i = 0
        self._last_out = np.zeros(max(block, 1), dtype=np.float32)
        self._last_out_n = 0
        self._mon_scratch = np.zeros(max(block, 1), dtype=np.float32)

    def _start_blocking(self):
        if self.stream is not None or self.running:
            self._emit_stopped = False
            try:
                self._hard_stop()
            finally:
                self._emit_stopped = True
        if self._stop_requested:
            return
        if not self._ensure_connected() or not self._ensure_model():
            self.status_msg.emit("无法启动：请先加载角色模型")
            self.load_failed.emit("模型未加载")
            return
        try:
            params = self.merged_params()
            self.pipeline.configure(**params)
            started = self.pipeline.start(**params)
            if started is False:
                raise RuntimeError("推理未能启动")
            if self._stop_requested:
                self._hard_stop()
                return
            self._alloc_rt_buffers(int(getattr(self.pipeline, "_block_frame", 0) or 4800))
            self.xrun_count = 0
            self._xrun_emitted = 0
            self._true_e2e_ms = 0.0
            self._loop_lat_ema = None
            self._slow_streak = 0
            self._adapted = False
            self._drain_queue(self.input_queue)
            self._drain_queue(self.output_queue)

            # 输入 AGC（按当前采样率重建，静音/增益状态清零，动态绑定当前静音门限）
            if self.input_agc:
                self.agc = AutoGain(sample_rate=self.pipeline.samplerate, gate_db=float(self.threhold))
            else:
                self.agc = None

            self.worker_thread = InferenceWorkerThread(
                self.pipeline, self.input_queue, self.output_queue, self)
            self.worker_thread.infer_done.connect(self._on_worker_infer_done)
            self.worker_thread.stage_stats.connect(self.stage_stats)
            self.worker_thread.spectrum.connect(self.spectrum)
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
            if self._stop_requested:
                self._hard_stop()
                return
            self.stream.start()
            self._open_monitor()
            if self._stop_requested:
                self._hard_stop()
                return
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

        # 1. 优先探测是否为 ASIO 硬件设备（极低硬件延迟，不使用 WasapiSettings）
        is_asio = False
        try:
            in_info = sd.query_devices(self.input_device) if self.input_device is not None else None
            if in_info:
                api_idx = in_info.get("hostapi", -1)
                apis = sd.query_hostapis()
                api_name = apis[api_idx]["name"].upper() if 0 <= api_idx < len(apis) else ""
                if "ASIO" in api_name:
                    is_asio = True
        except Exception:
            pass

        if is_asio:
            try:
                self.stream = sd.Stream(**kwargs)
                return True, "ASIO 硬件直通 (极低延迟)"
            except Exception as e:
                errors.append(e)

        # 2. WASAPI 独占（次低延迟）
        try:
            self.stream = sd.Stream(
                extra_settings=sd.WasapiSettings(exclusive=True), **kwargs)
            return True, "WASAPI 独占 (最低延迟)"
        except Exception as e:
            errors.append(e)

        # 3. WASAPI 共享模式
        try:
            try:
                extra = sd.WasapiSettings(exclusive=False, auto_convert=True)
            except TypeError:
                extra = sd.WasapiSettings(exclusive=False)
            self.stream = sd.Stream(extra_settings=extra, **kwargs)
            return True, "WASAPI 共享（独占失败: %s）" % _wasapi_fail_reason(errors[-1])
        except Exception as e:
            errors.append(e)

        # 4. 系统默认共享回退
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
        """运行中只重建声卡流（后台执行），不断开服务器、不重载模型。"""
        self.input_device = input_device
        self.output_device = output_device
        if not self.running:
            return True, ""
        if self._engine_busy():
            self.status_msg.emit("正在切换设备，请稍候...")
            return True, ""
        self.status_msg.emit("正在切换设备...")
        self._start_thread = EngineStartThread(self, "reopen", parent=self)
        self._start_thread.start()
        return True, ""

    def _reopen_blocking(self):
        self._close_stream_only()
        ok, msg = self._open_stream()
        if not ok:
            self.status_msg.emit("切换设备失败: " + msg)
            self._hard_stop()
            return
        self._fade_in_left = int(0.04 * self.pipeline.samplerate)
        self._fade_out_left = 0
        try:
            self.stream.start()
        except Exception as e:
            self.status_msg.emit("切换设备失败: " + str(e))
            self._hard_stop()
            return
        self.status_msg.emit("已切换设备 · " + msg)

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

    def _report_loop_latency(self, times, frames):
        """嘴到耳 ≈ PortAudio(DAC−ADC) + 一块算法延迟 + 队列积压。

        回调里只写数值，由 UI 定时器取出，避免 PortAudio 线程发 Qt 信号。
        """
        if times is None:
            return
        try:
            adc = getattr(times, "inputBufferAdcTime", 0.0) or 0.0
            dac = getattr(times, "outputBufferDacTime", 0.0) or 0.0
            if adc <= 0 or dac <= 0:
                return
            pa_ms = (dac - adc) * 1000.0
            if not (0 < pa_ms < 5000):
                return
            sr = float(getattr(self.pipeline, "samplerate", 0) or 0)
            # 算法延迟地板是管线块，不是声卡回调帧；队列里每个元素也是一块管线块
            alg = int(getattr(self.pipeline, "_block_frame", 0) or frames)
            block_ms = (1000.0 * alg / sr) if sr > 0 else 0.0
            queued = 0
            try:
                queued = self.input_queue.qsize() + self.output_queue.qsize()
            except Exception:
                pass
            true_ms = pa_ms + block_ms + queued * block_ms
            if self._loop_lat_ema is None:
                self._loop_lat_ema = true_ms
            else:
                self._loop_lat_ema += (true_ms - self._loop_lat_ema) * 0.08
            self._true_e2e_ms = self._loop_lat_ema
        except Exception:
            pass

    def _flush_callback_stats(self):
        if self.xrun_count != self._xrun_emitted:
            self._xrun_emitted = self.xrun_count
            self.xrun_signal.emit(self.xrun_count)
        if self.running and self._true_e2e_ms > 0:
            self.loop_latency.emit(self._true_e2e_ms)
        if self._fade_done_flag:
            self._fade_done_flag = False
            self.fade_done.emit()

    def request_hard_stop(self):
        """后台停机，UI 线程只发起。"""
        self._stop_requested = True
        self._fade_epoch += 1
        if self._engine_busy():
            action = getattr(self._start_thread, "action", "")
            if action == "stop":
                return
            # start/reopen 线程会在关键点看到 _stop_requested
            return
        if not self.running and self.stream is None and self.worker_thread is None:
            return
        self._start_thread = EngineStartThread(self, "stop", parent=self)
        self._start_thread.start()

    def wait_idle(self, timeout_ms=2500):
        t = getattr(self, "_start_thread", None)
        if t is not None and t.isRunning():
            t.wait(timeout_ms)
        if self.running or self.stream is not None or self.worker_thread is not None:
            self._hard_stop()

    def stop(self):
        if self.running and self.stream is not None and self._fade_out_left <= 0:
            self._fade_epoch += 1
            self._pending_fade_epoch = self._fade_epoch
            self._fade_out_total = max(1, int(0.04 * self.pipeline.samplerate))
            self._fade_out_left = self._fade_out_total
            return
        self.request_hard_stop()

    def _on_fade_done(self):
        if self._pending_fade_epoch != self._fade_epoch:
            return
        if not self.running:
            return
        self.request_hard_stop()

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
        if self._emit_stopped:
            self.stopped_ok.emit()
            self.status_msg.emit("已停止")

    def _on_worker_infer_done(self, elapsed_ms, in_db, out_db):
        self.infer_time.emit(elapsed_ms)
        self.rms_levels.emit(in_db, out_db)
        self._adapt_if_slow(elapsed_ms)

    def _adapt_if_slow(self, elapsed_ms):
        budget = max(20.0, float(self.block_time) * 1000.0)
        if elapsed_ms > budget * 0.95:
            self._slow_streak += 1
        else:
            self._slow_streak = max(0, self._slow_streak - 1)
        if self._adapted or self._slow_streak < 6:
            return
        self._adapted = True
        if self.vad_enable:
            self.vad_enable = False
        if self.deesser_enable:
            self.deesser_enable = False
        try:
            cur = float(self.current_speaker.index_rate) if self.current_speaker else 0.0
        except Exception:
            cur = 0.0
        if cur > 0.35:
            self.change_index_rate(0.35)
            if self.current_speaker is not None:
                self.current_speaker.index_rate = 0.35
        self.status_msg.emit("推理偏慢：已自动关 VAD/去齿音并降低检索，稳住实时")

    def _on_worker_xrun(self):
        self.xrun_count += 1

    def _apply_edge_fade(self, outdata):
        n = len(outdata)
        if self._fade_in_left > 0:
            total = max(1, int(0.04 * self.pipeline.samplerate))
            done = total - self._fade_in_left
            # 线性斜坡，避免每块 np.arange 分配
            if n == 1:
                outdata[0, 0] *= min(1.0, (done + 1) / total)
            else:
                g0 = done / total
                g1 = (done + n) / total
                outdata[:, 0] *= np.linspace(g0, g1, n, endpoint=False, dtype=np.float32).clip(0.0, 1.0)
            self._fade_in_left = max(0, self._fade_in_left - n)
        if self._fade_out_left > 0:
            total = max(1, self._fade_out_total)
            remaining = self._fade_out_left
            if n == 1:
                outdata[0, 0] *= max(0.0, remaining / total)
            else:
                g0 = remaining / total
                g1 = (remaining - n) / total
                outdata[:, 0] *= np.linspace(g0, g1, n, endpoint=False, dtype=np.float32).clip(0.0, 1.0)
            self._fade_out_left = max(0, self._fade_out_left - n)
            if self._fade_out_left <= 0:
                self._fade_done_flag = True

    def _cb_push_in(self, mono):
        n = int(mono.shape[0])
        if n <= 0 or self._in_buf.size == 0:
            return
        if self._in_n + n > self._in_buf.shape[0]:
            self._in_n = 0
            self.xrun_count += 1
        self._in_buf[self._in_n:self._in_n + n] = mono[:n]
        self._in_n += n

    def _cb_take_in(self, n):
        if self._in_n < n or not self._in_pool:
            return None
        slot = self._in_pool[self._pool_i]
        self._pool_i = (self._pool_i + 1) % len(self._in_pool)
        if slot.shape[0] != n:
            slot = np.zeros(n, dtype=np.float32)
            self._in_pool[self._pool_i - 1] = slot
        slot[:n] = self._in_buf[:n]
        remain = self._in_n - n
        if remain:
            self._in_buf[:remain] = self._in_buf[n:self._in_n]
        self._in_n = remain
        return slot

    def _cb_push_out(self, block):
        if block is None or self._out_buf.size == 0:
            return
        src = block[:, 0] if getattr(block, "ndim", 1) > 1 else block
        n = int(src.shape[0])
        if n <= 0:
            return
        if self._out_n + n > self._out_buf.shape[0]:
            self._out_n = 0
            self.xrun_count += 1
        self._out_buf[self._out_n:self._out_n + n] = np.asarray(src[:n], dtype=np.float32)
        self._out_n += n

    def _cb_fill_from_hold(self, dest):
        n = dest.shape[0]
        if self._last_out_n <= 0 or self._last_out.size == 0:
            dest.fill(0.0)
            return
        src = self._last_out[:self._last_out_n]
        # 连续欠载计数：前 3 块原样重复，之后线性淡到静音，避免循环同一块变成嗡鸣
        self._hold_count = getattr(self, "_hold_count", 0) + 1
        g = 1.0 if self._hold_count <= 3 else max(0.0, 1.0 - 0.25 * (self._hold_count - 3))
        if src.shape[0] >= n:
            dest[:] = src[:n] * g
        else:
            dest[:src.shape[0]] = src * g
            dest[src.shape[0]:] = (src[-1] * g) if src.size else 0.0

    def _on_audio(self, indata, outdata, frames, times, status):
        """音频回调：预分配环缓冲凑整块。欠载重复上一块，绝不在此发 Qt 信号。"""
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
                self._push_monitor_view(outdata[:, 0], n_needed)
                self._report_loop_latency(times, n_needed)
                return

            if self.input_agc and self.agc is not None:
                mono = self.agc.process(mono)

            in_block = int(getattr(self.pipeline, "_block_frame", 0) or 0)
            if in_block <= 0:
                outdata.fill(0)
                return
            if mono.shape[0] == in_block and self._in_n == 0 and self._in_pool:
                slot = self._in_pool[self._pool_i]
                self._pool_i = (self._pool_i + 1) % len(self._in_pool)
                if slot.shape[0] != in_block:
                    slot = np.zeros(in_block, dtype=np.float32)
                    self._in_pool[(self._pool_i - 1) % len(self._in_pool)] = slot
                slot[:in_block] = mono[:in_block]
                self._enqueue_input(slot)
            else:
                self._cb_push_in(mono)
                while True:
                    chunk = self._cb_take_in(in_block)
                    if chunk is None:
                        break
                    self._enqueue_input(chunk)

            while self._out_n < n_needed:
                try:
                    block = self.output_queue.get_nowait()
                except queue.Empty:
                    break
                self._cb_push_out(block)

            if self._out_n >= n_needed:
                outdata[:, 0] = self._out_buf[:n_needed]
                remain = self._out_n - n_needed
                if remain:
                    self._out_buf[:remain] = self._out_buf[n_needed:self._out_n]
                self._out_n = remain
                hold_n = min(n_needed, self._last_out.shape[0])
                self._last_out[:hold_n] = outdata[:hold_n, 0]
                self._last_out_n = hold_n
                self._hold_count = 0
            else:
                n_take = min(self._out_n, n_needed)
                if n_take > 0:
                    outdata[:n_take, 0] = self._out_buf[:n_take]
                    self._out_n = 0
                    self._cb_fill_from_hold(outdata[n_take:, 0])
                else:
                    self._cb_fill_from_hold(outdata[:, 0])
                self.xrun_count += 1
            if outdata.shape[1] > 1:
                outdata[:, 1:] = outdata[:, :1]
            dry = float(self.dry_mix)
            if dry > 0:
                n = min(len(mono), n_needed)
                outdata[:n, 0] = outdata[:n, 0] * (1.0 - dry) + mono[:n] * dry
                if outdata.shape[1] > 1:
                    outdata[:, 1:] = outdata[:, :1]
            self._apply_edge_fade(outdata)
            self._push_monitor_view(outdata[:, 0], n_needed)
            self._report_loop_latency(times, n_needed)
        except Exception:
            outdata.fill(0)

    def _push_monitor_view(self, mono, n):
        if not self.monitor_enabled or self.monitor_stream is None:
            return
        n = min(int(n), int(mono.shape[0]), int(self._mon_scratch.shape[0] or 0))
        if n <= 0:
            return
        self._mon_scratch[:n] = mono[:n]
        self._mon_scratch[:n] *= np.float32(self.monitor_volume)
        try:
            self.monitor_queue.put_nowait(self._mon_scratch[:n].copy())
        except queue.Full:
            pass


# ==============================================================================
# 主窗口 - 三栏布局
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC 实时变声")
        self.setMinimumSize(1040, 640)
        self.setAcceptDrops(True)
        self.speaker_mgr = SpeakerManager()
        self._settings = load_user_settings()
        self._live_guard = False
        self._device_guard = False
        self._dev_fp = ()
        self._load_gen = 0
        self._load_started = None       # 需在 _rl() 之前初始化（_rl 会触发加载）
        self._load_log_offset = 0
        self._pending_after_stop = None
        self._pending_speaker = None
        self._pending_mode = None
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
        self.engine.spectrum.connect(self._on_spectrum)
        self._dark = False
        self._build_ui()
        self._apply_theme()
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
        # 加载进度：每秒刷新「阶段 + 已等待秒数」，避免首次加载看起来像卡死
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._update_load_progress)
        self._progress_timer.start()

    def _on_rms_levels(self, in_db, out_db):
        self.in_meter.set_level(in_db)
        self.out_meter.set_level(out_db)

    def _on_spectrum(self, bins):
        if hasattr(self, "spectrum"):
            self.spectrum.set_bins(bins)

    def _apply_theme(self):
        global UI_DARK
        UI_DARK = False
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(LIGHT_QSS)
        if hasattr(self, "in_meter"):
            self.in_meter.set_dark(False)
        if hasattr(self, "out_meter"):
            self.out_meter.set_dark(False)
        if hasattr(self, "spectrum"):
            self.spectrum.set_dark(False)

    def _cable_wizard(self):
        dlg = CableWizard(self, self.ic, self.oc)
        if dlg.exec():
            self._on_device_changed("output")
            self._persist_settings()

    def _on_xrun(self, xruns):
        self.xrun_label.setText(f"卡顿 {xruns}")
        self.xrun_label.setStyleSheet("font-size:12px;font-weight:700;color:#dc2626;background:#fef2f2;border:1px solid #fecaca;border-radius:6px;padding:4px 8px;" if xruns > 0 else "font-size:12px;font-weight:700;color:#64748b;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:4px 8px;")

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
        self.resize(1200, 760)
        self.setMinimumSize(1060, 680)
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(10)

        # ── 顶部 Header 状态栏 ──
        header = QFrame(); header.setObjectName("header")
        top = QHBoxLayout(header)
        top.setContentsMargins(14, 8, 14, 8)
        top.setSpacing(12)

        title = QLabel("RVC 实时变声")
        title.setObjectName("appTitle")
        top.addWidget(title)

        self.in_meter = VUMeterWidget(title="输入")
        self.out_meter = VUMeterWidget(title="输出")
        top.addWidget(self.in_meter)
        top.addWidget(self.out_meter)
        top.addStretch()

        # 状态微型标签
        self.badge_box = QFrame()
        self.badge_box.setStyleSheet("background:#f3f4f6;border:1px solid #e5e7eb;border-radius:6px;")
        bh = QHBoxLayout(self.badge_box); bh.setContentsMargins(8, 4, 8, 4); bh.setSpacing(6)
        self.light = QLabel()
        self.light.setFixedSize(7, 7)
        self.light.setStyleSheet(f"background:{LIGHT_GRAY};border-radius:3px;")
        bh.addWidget(self.light)
        self.state_label = QLabel("未加载模型")
        self.state_label.setStyleSheet("font-size:12px;font-weight:600;color:#6b7280;")
        bh.addWidget(self.state_label)
        top.addWidget(self.badge_box)

        self.latency_label = QLabel("推理 --ms")
        self.latency_label.setStyleSheet("font-size:12px;font-weight:600;color:#15803d;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;padding:4px 8px;")
        self.latency_label.setToolTip("单块推理耗时。超过块大小就会卡顿")
        top.addWidget(self.latency_label)

        self.e2e_label = QLabel("嘴到耳 --ms")
        self.e2e_label.setStyleSheet("font-size:12px;font-weight:600;color:#6d28d9;background:#f5f3ff;border:1px solid #ddd6fe;border-radius:6px;padding:4px 8px;")
        self.e2e_label.setToolTip("估算的真实听感延迟：声卡缓冲 + 一块算法延迟 + 队列积压")
        top.addWidget(self.e2e_label)

        self.xrun_label = QLabel("卡顿 0")
        self.xrun_label.setStyleSheet("font-size:12px;font-weight:600;color:#6b7280;background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;padding:4px 8px;")
        top.addWidget(self.xrun_label)
        root.addWidget(header)

        # ── 三栏主体布局 ──
        sp = QSplitter(Qt.Horizontal)
        sp.addWidget(self._build_left())
        sp.addWidget(self._build_mid())
        sp.addWidget(self._build_right())
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        sp.setStretchFactor(2, 1)
        sp.setSizes([260, 480, 380])
        root.addWidget(sp, 1)

        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)

    def _build_left(self):
        g = QGroupBox("角色配置")
        l = QVBoxLayout(g)
        l.setContentsMargins(12, 16, 12, 12)
        l.setSpacing(10)

        # 角色下拉选择
        self.sc = create_styled_combo(max_visible=12)
        self.sc.setMinimumHeight(32)
        self.sc.currentIndexChanged.connect(self._sel)
        l.addWidget(self.sc)

        # 角色增删改按钮行
        br = QHBoxLayout()
        br.setSpacing(6)
        for t, fn in [("添加", self._a), ("编辑", self._e), ("删除", self._d)]:
            b = QPushButton(t)
            b.setObjectName("btnGhost")
            b.setMinimumHeight(28)
            b.clicked.connect(fn)
            br.addWidget(b, 1)
        l.addLayout(br)

        # 当前角色信息展示卡片
        self.cur_card = QFrame()
        self.cur_card.setObjectName("roleCard")
        cv = QVBoxLayout(self.cur_card)
        cv.setContentsMargins(12, 10, 12, 10)
        cv.setSpacing(4)
        self.cur_name = QLabel("未选择角色")
        self.cur_name.setStyleSheet("font-size:14px;font-weight:700;color:#111827;")
        self.cur_model = QLabel("")
        self.cur_model.setStyleSheet("font-size:11px;color:#6b7280;")
        self.cur_info = QLabel("")
        self.cur_info.setStyleSheet("font-size:11px;color:#6b7280;")
        cv.addWidget(self.cur_name)
        cv.addWidget(self.cur_model)
        cv.addWidget(self.cur_info)
        l.addWidget(self.cur_card)

        # 实时调节面板
        live = QGroupBox("实时调节")
        gl = QGridLayout(live)
        gl.setContentsMargins(10, 16, 10, 10)
        gl.setHorizontalSpacing(10)
        gl.setVerticalSpacing(8)

        self.live_pitch = QSpinBox()
        self.live_pitch.setRange(-36, 36)
        self.live_pitch.setSuffix(" 半音")
        self.live_pitch.valueChanged.connect(self._on_live_pitch)

        self.live_index = QDoubleSpinBox()
        self.live_index.setRange(0.0, 1.0)
        self.live_index.setSingleStep(0.1)
        self.live_index.valueChanged.connect(self._on_live_index)

        self.live_formant = QDoubleSpinBox()
        self.live_formant.setRange(-12.0, 12.0)
        self.live_formant.setSingleStep(0.5)
        self.live_formant.valueChanged.connect(self._on_live_formant)

        self.live_dry = QDoubleSpinBox()
        self.live_dry.setRange(0.0, 1.0)
        self.live_dry.setSingleStep(0.1)
        self.live_dry.setToolTip("0=只听变声，1=只听原声")
        self.live_dry.valueChanged.connect(self._on_live_dry)

        gl.addWidget(self._lbl("音高"), 0, 0)
        gl.addWidget(self.live_pitch, 0, 1)
        gl.addWidget(self._lbl("检索"), 1, 0)
        gl.addWidget(self.live_index, 1, 1)
        gl.addWidget(self._lbl("共振峰"), 2, 0)
        gl.addWidget(self.live_formant, 2, 1)
        gl.addWidget(self._lbl("原声混合"), 3, 0)
        gl.addWidget(self.live_dry, 3, 1)

        self.bypass = QCheckBox("旁通（听原声）")
        self.bypass.setToolTip("快捷键 Ctrl+B")
        self.bypass.toggled.connect(self._on_bypass)
        gl.addWidget(self.bypass, 4, 0, 1, 2)
        gl.setColumnStretch(1, 1)

        l.addWidget(live)
        l.addStretch(1)
        return g

    def _build_mid(self):
        g = QGroupBox("转换控制")
        l = QVBoxLayout(g)
        l.setContentsMargins(12, 16, 12, 12)
        l.setSpacing(10)

        self.mode_local = QRadioButton("本地推理")
        self.mode_server = QRadioButton("服务器")
        if self.engine.mode == "server":
            self.mode_server.setChecked(True)
        else:
            self.mode_local.setChecked(True)
        self.mode_local.toggled.connect(lambda on: on and self._apply_mode("local"))
        self.mode_server.toggled.connect(lambda on: on and self._apply_mode("server"))
        self.server_edit = QLineEdit(self._settings.get("server_url") or DEFAULT_SERVER_URL)
        self.server_edit.setPlaceholderText("ws://主机:8765")
        self.conn_btn = QPushButton("连接")
        self.conn_btn.setObjectName("btnConnect")
        self.conn_btn.setToolTip("改完地址后点这里，按新地址重连（不重新加载角色）")
        self.conn_btn.clicked.connect(self._connect_server)
        self.server_row = QWidget()
        sl = QHBoxLayout(self.server_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)
        sl.addWidget(self._lbl("服务器"))
        sl.addWidget(self.server_edit, 1)
        sl.addWidget(self.conn_btn)

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

        # 音频设备选择
        dl = QGridLayout()
        dl.setHorizontalSpacing(10)
        dl.setVerticalSpacing(8)

        dl.addWidget(self._lbl("输入设备"), 0, 0)
        self.ic = DeviceCombo(direction="input")
        self.ic.setMinimumHeight(30)
        self.ic.setToolTip("WASAPI 独占要求输入输出同一组 API；选择后会自动对齐另一侧")
        self.ic.currentIndexChanged.connect(lambda: self._on_device_changed("input"))
        dl.addWidget(self.ic, 0, 1)

        dl.addWidget(self._lbl("输出设备"), 1, 0)
        self.oc = DeviceCombo(direction="output")
        self.oc.setMinimumHeight(30)
        self.oc.setToolTip("WASAPI 独占要求输入输出同一组 API；选择后会自动对齐另一侧")
        self.oc.currentIndexChanged.connect(lambda: self._on_device_changed("output"))
        dl.addWidget(self.oc, 1, 1)

        rb = QPushButton("刷新设备")
        rb.setObjectName("btnGhost")
        rb.setMinimumHeight(28)
        rb.clicked.connect(lambda: self._rd())

        cable_btn = QPushButton("虚拟声卡向导")
        cable_btn.setObjectName("btnGhost")
        cable_btn.setMinimumHeight(28)
        cable_btn.setToolTip("检测 VB-Cable / VoiceMeeter，没有则打开安装页")
        cable_btn.clicked.connect(self._cable_wizard)

        row_btns = QHBoxLayout()
        row_btns.setSpacing(8)
        row_btns.addWidget(cable_btn, 1)
        row_btns.addWidget(rb, 1)
        dl.addLayout(row_btns, 2, 1)
        dl.setColumnStretch(1, 1)
        l.addLayout(dl)

        # 设备错位/异常提示（常驻显示）
        self.dev_hint = QLabel("")
        self.dev_hint.setWordWrap(True)
        self.dev_hint.setVisible(False)
        l.addWidget(self.dev_hint)

        # 延迟/音质平衡滑杆
        tq = QHBoxLayout()
        tq.setSpacing(8)
        tq.addWidget(self._lbl("延迟平衡"))
        self.tq_fast = QLabel("更快(低延迟)")
        self.tq_fast.setStyleSheet("font-size:11px;color:#64748b;")
        tq.addWidget(self.tq_fast)
        self.tq_slider = QSlider(Qt.Horizontal)
        self.tq_slider.setRange(0, 100)
        self.tq_slider.setValue(40)
        self.tq_slider.setToolTip("同时调节块大小与额外上下文。偏左更低延迟，偏右更稳音色。下次启动生效。")
        self.tq_slider.valueChanged.connect(self._on_tradeoff)
        tq.addWidget(self.tq_slider, 1)
        self.tq_hq = QLabel("更稳(好音质)")
        self.tq_hq.setStyleSheet("font-size:11px;color:#64748b;")
        tq.addWidget(self.tq_hq)
        l.addLayout(tq)

        # 输出频谱
        self.spectrum = SpectrumWidget()
        self.spectrum.setToolTip("输出频谱实时监视")
        l.addWidget(self.spectrum)
        l.addStretch(1)

        # 核心主按钮
        self.sb = QPushButton("启动变声")
        self.sb.setObjectName("btnStart")
        self.sb.setProperty("state", "off")
        self.sb.setMinimumHeight(46)
        self.sb.clicked.connect(self._tg)
        l.addWidget(self.sb)

        self.rec_btn = QPushButton("录音测试 (10 秒)")
        self.rec_btn.setObjectName("btnGhost")
        self.rec_btn.setMinimumHeight(34)
        self.rec_btn.setToolTip("录 10 秒，用当前角色变声后保存并播放")
        self.rec_btn.clicked.connect(self._rec)
        l.addWidget(self.rec_btn)
        return g

    def _build_right(self):
        tabs = QTabWidget()
        tabs.setObjectName("rightTabs")

        # ── Tab 1: 核心算法 ──
        t1 = QWidget()
        l1 = QVBoxLayout(t1)
        l1.setContentsMargins(10, 14, 10, 10)
        l1.setSpacing(10)

        g1 = QGroupBox("算法核心参数")
        l = QGridLayout(g1)
        l.setContentsMargins(10, 16, 10, 10)
        l.setHorizontalSpacing(10)
        l.setVerticalSpacing(8)

        self.fc = create_styled_combo(max_visible=10)
        self.fc.addItems(["rmvpe", "fcpe", "pm"])
        self.fc.setMinimumHeight(28)
        self.fc.currentTextChanged.connect(lambda v: setattr(self.engine, "f0method", v))
        self.fc.setToolTip("基频提取: rmvpe 最准, pm 最快")

        self.bs = QDoubleSpinBox()
        self.bs.setRange(0.03, 0.5)
        self.bs.setSingleStep(0.01)
        self.bs.setValue(DEFAULT_PARAMS["block_time"])
        self.bs.setDecimals(3)
        self.bs.setSuffix(" s")
        self.bs.setMinimumHeight(28)
        self.bs.setToolTip("音频块时长（秒）。越小嘴到耳越低，GPU 越容易卡顿。运行中修改下次启动生效")
        self.bs.valueChanged.connect(lambda v: setattr(self.engine, "block_time", v))
        self.bs.valueChanged.connect(lambda _: self._sync_tradeoff_slider())

        self.xs = QDoubleSpinBox()
        self.xs.setRange(0.01, 0.5)
        self.xs.setSingleStep(0.01)
        self.xs.setValue(DEFAULT_PARAMS["crossfade_time"])
        self.xs.setSuffix(" s")
        self.xs.setMinimumHeight(28)
        self.xs.setToolTip("运行中修改将在下次启动后生效")
        self.xs.valueChanged.connect(lambda v: setattr(self.engine, "crossfade_time", v))

        self.es = QDoubleSpinBox()
        self.es.setRange(0.4, 5.0)
        self.es.setSingleStep(0.1)
        self.es.setValue(DEFAULT_PARAMS["extra_time"])
        self.es.setSuffix(" s")
        self.es.setMinimumHeight(28)
        self.es.setToolTip("上下文长度：越大音色越稳，但不增加听感延迟，只增加每块算力。运行中修改下次启动生效")
        self.es.valueChanged.connect(lambda v: setattr(self.engine, "extra_time", v))

        self.ts = QSpinBox()
        self.ts.setRange(-80, 0)
        self.ts.setValue(-50)
        self.ts.setSuffix(" dB")
        self.ts.setMinimumHeight(28)
        self.ts.setToolTip("低于此音量视为静音。-80 关闭门限")
        self.ts.valueChanged.connect(lambda v: setattr(self.engine, "threhold", v))

        self.calib_noise_btn = QPushButton("测底噪")
        self.calib_noise_btn.setObjectName("btnGhost")
        self.calib_noise_btn.setFixedWidth(58)
        self.calib_noise_btn.setMinimumHeight(28)
        self.calib_noise_btn.setToolTip("保持安静 1 秒，自动测定当前环境底噪并计算最优静音阈值")
        self.calib_noise_btn.clicked.connect(self._auto_calibrate_noise)

        self.ts_box = QWidget()
        ts_l = QHBoxLayout(self.ts_box)
        ts_l.setContentsMargins(0, 0, 0, 0)
        ts_l.setSpacing(6)
        ts_l.addWidget(self.ts, 1)
        ts_l.addWidget(self.calib_noise_btn)

        self.rs = QDoubleSpinBox()
        self.rs.setRange(0.0, 1.0)
        self.rs.setSingleStep(0.1)
        self.rs.setValue(0.3)
        self.rs.setMinimumHeight(28)
        self.rs.setToolTip("0=完全跟随输入音量，1=只保留变声自身音量")
        self.rs.valueChanged.connect(lambda v: setattr(self.engine, "rms_mix_rate", v))

        rows = [
            ("F0 算法", self.fc),
            ("块大小", self.bs),
            ("交叉淡入", self.xs),
            ("额外上下文", self.es),
            ("静音阈值", self.ts_box),
            ("音量保留", self.rs),
        ]
        for i, (name, w) in enumerate(rows):
            lbl = self._lbl(name)
            lbl.setMinimumHeight(28)
            l.addWidget(lbl, i, 0)
            l.addWidget(w, i, 1)

        self.inc = QCheckBox("输入降噪")
        self.inc.setMinimumHeight(26)
        self.inc.setToolTip("运行中修改将在下次启动后生效")
        self.inc.toggled.connect(lambda v: setattr(self.engine, "I_noise_reduce", v))
        self.onc = QCheckBox("输出降噪")
        self.onc.setMinimumHeight(26)
        self.onc.setToolTip("运行中修改将在下次启动后生效")
        self.onc.toggled.connect(lambda v: setattr(self.engine, "O_noise_reduce", v))

        nr = QHBoxLayout()
        nr.setSpacing(12)
        nr.addWidget(self.inc)
        nr.addWidget(self.onc)
        nr.addStretch()
        l.addLayout(nr, len(rows), 0, 1, 2)
        l.setColumnStretch(1, 1)
        l1.addWidget(g1)
        l1.addStretch(1)

        # ── Tab 2: 音质增强 ──
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        l2.setContentsMargins(10, 14, 10, 10)
        l2.setSpacing(10)

        g_dsp = QGroupBox("人声修饰与 DSP")
        dl = QGridLayout(g_dsp)
        dl.setContentsMargins(10, 16, 10, 10)
        dl.setHorizontalSpacing(10)
        dl.setVerticalSpacing(8)

        # 输出保护
        pr = QHBoxLayout()
        pr.setSpacing(8)
        self.limiter_cb = QCheckBox("输出保护")
        self.limiter_cb.setMinimumHeight(26)
        self.limiter_cb.setToolTip("直流高通 + 软限幅，防止爆音/直流偏移（实时生效）")
        self.limiter_cb.setChecked(True)
        self.limiter_cb.toggled.connect(lambda v: setattr(self.engine, "limiter_enable", v))
        pr.addWidget(self.limiter_cb)

        self.limiter_th = QDoubleSpinBox()
        self.limiter_th.setRange(-12.0, 0.0)
        self.limiter_th.setSingleStep(0.5)
        self.limiter_th.setSuffix(" dB")
        self.limiter_th.setValue(-1.0)
        self.limiter_th.setFixedWidth(82)
        self.limiter_th.setMinimumHeight(26)
        self.limiter_th.setToolTip("起限阈值，-1 dB 为推荐值")
        self.limiter_th.valueChanged.connect(lambda v: setattr(self.engine, "limiter_threshold_db", v))
        pr.addWidget(self.limiter_th)
        dl.addLayout(pr, 0, 0, 1, 2)

        # 齿音保留
        self.hf_spin = QDoubleSpinBox()
        self.hf_spin.setRange(0.0, 1.0)
        self.hf_spin.setSingleStep(0.05)
        self.hf_spin.setValue(DEFAULT_PARAMS["hf_mix_rate"])
        self.hf_spin.setMinimumHeight(28)
        self.hf_spin.setToolTip("把原声 6kHz 以上的气音混回输出。与「去齿音」互斥。")
        self.hf_spin.valueChanged.connect(self._on_hf_mix)
        dl.addWidget(self._lbl("齿音保留"), 1, 0)
        dl.addWidget(self.hf_spin, 1, 1)

        # 临场感
        self.pres_spin = QDoubleSpinBox()
        self.pres_spin.setRange(0.0, 1.0)
        self.pres_spin.setSingleStep(0.05)
        self.pres_spin.setValue(DEFAULT_PARAMS["presence"])
        self.pres_spin.setMinimumHeight(28)
        self.pres_spin.setToolTip("轻微提升人声穿透力，0=关闭，1=最大")
        self.pres_spin.valueChanged.connect(lambda v: setattr(self.engine, "presence", v))
        dl.addWidget(self._lbl("临场感"), 2, 0)
        dl.addWidget(self.pres_spin, 2, 1)

        # 去齿音
        self.deess_cb = QCheckBox("自适应去齿音")
        self.deess_cb.setMinimumHeight(26)
        self.deess_cb.setToolTip("尖刺超标时软衰减。与「齿音保留」互斥。")
        self.deess_cb.setChecked(DEFAULT_PARAMS["deesser_enable"])
        self.deess_cb.toggled.connect(self._on_deesser)
        dl.addWidget(self.deess_cb, 3, 0, 1, 2)

        # 人声识别 (VAD)
        vr = QHBoxLayout()
        vr.setSpacing(8)
        self.vad_cb = QCheckBox("人声识别 (VAD)")
        self.vad_cb.setMinimumHeight(26)
        self.vad_cb.setToolTip("智能区分人声与环境杂音，非人声自动静音")
        self.vad_cb.setChecked(False)
        self.vad_cb.toggled.connect(lambda v: setattr(self.engine, "vad_enable", v))
        vr.addWidget(self.vad_cb)

        self.vad_th = QDoubleSpinBox()
        self.vad_th.setRange(0.10, 0.90)
        self.vad_th.setSingleStep(0.05)
        self.vad_th.setValue(0.50)
        self.vad_th.setFixedWidth(82)
        self.vad_th.setMinimumHeight(26)
        self.vad_th.setToolTip("人声置信度门限（0.50 为推荐平衡点）")
        self.vad_th.valueChanged.connect(lambda v: setattr(self.engine, "vad_threshold", v))
        vr.addWidget(self.vad_th)
        dl.addLayout(vr, 4, 0, 1, 2)

        # 阶段耗时
        self.st_lbl = QLabel("阶段耗时: --")
        self.st_lbl.setStyleSheet("font-size:11px;color:#6b7c8a;")
        dl.addWidget(self.st_lbl, 5, 0, 1, 2)
        dl.setColumnStretch(1, 1)
        l2.addWidget(g_dsp)
        l2.addStretch(1)

        # ── Tab 3: 预设与监听 ──
        t3 = QWidget()
        l3 = QVBoxLayout(t3)
        l3.setContentsMargins(10, 14, 10, 10)
        l3.setSpacing(10)

        g2 = QGroupBox("预设与监听")
        m = QGridLayout(g2)
        m.setContentsMargins(10, 16, 10, 10)
        m.setHorizontalSpacing(10)
        m.setVerticalSpacing(8)

        self.agc_cb = QCheckBox("输入自动增益 (AGC)")
        self.agc_cb.setMinimumHeight(26)
        self.agc_cb.setToolTip("输入电平归一化，说话人远近变化时音色更稳定（实时生效）")
        self.agc_cb.toggled.connect(self._on_agc)
        m.addWidget(self.agc_cb, 0, 0, 1, 2)

        m.addWidget(self._lbl("预设方案"), 1, 0)
        pbtn_row = QHBoxLayout()
        pbtn_row.setSpacing(6)
        self._preset_map = load_presets()
        self.preset_cb = create_styled_combo()
        self.preset_cb.setMinimumHeight(28)
        for p in self._preset_map:
            self.preset_cb.addItem(p["name"])
        pbtn_row.addWidget(self.preset_cb, 1)

        pb_apply = QPushButton("应用")
        pb_apply.setObjectName("btnGhost")
        pb_apply.setFixedWidth(48)
        pb_apply.setMinimumHeight(28)
        pb_apply.setToolTip("应用所选预设")
        pb_apply.clicked.connect(self._apply_preset)
        pbtn_row.addWidget(pb_apply)

        pb_save = QPushButton("保存")
        pb_save.setObjectName("btnGhost")
        pb_save.setFixedWidth(48)
        pb_save.setMinimumHeight(28)
        pb_save.setToolTip("把当前参数保存为用户预设")
        pb_save.clicked.connect(self._save_preset)
        pbtn_row.addWidget(pb_save)
        m.addLayout(pbtn_row, 1, 1)

        mr = QHBoxLayout()
        mr.setSpacing(8)
        self.monitor_cb = QCheckBox("耳机监听")
        self.monitor_cb.setMinimumHeight(26)
        self.monitor_cb.setToolTip("把变声结果同时播放到第二输出设备（如耳机）")
        self.monitor_cb.toggled.connect(self._on_monitor_toggle)
        mr.addWidget(self.monitor_cb)

        self.monitor_vol = QSlider(Qt.Horizontal)
        self.monitor_vol.setRange(0, 100)
        self.monitor_vol.setValue(80)
        self.monitor_vol.setToolTip("监听音量")
        self.monitor_vol.valueChanged.connect(self._on_monitor_vol)
        mr.addWidget(self.monitor_vol, 1)
        m.addLayout(mr, 2, 0, 1, 2)

        self.mc = DeviceCombo(direction="output")
        self.mc.setMinimumHeight(30)
        self.mc.setToolTip("监听输出设备（可不同于主输出）")
        self.mc.currentIndexChanged.connect(lambda: self._on_monitor_changed())
        m.addWidget(self.mc, 3, 0, 1, 2)
        m.setColumnStretch(1, 1)
        l3.addWidget(g2)

        # 远程服务器模式（可选）
        g3 = QGroupBox("远程服务器模式（可选）")
        g3.setCheckable(True)
        g3.setChecked(self.engine.mode == "server")
        g3.setToolTip("连接局域网/远程 RVC 推理服务器")
        sg = QVBoxLayout(g3)
        sg.setContentsMargins(10, 16, 10, 10)
        sg.setSpacing(8)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        mode_row.addWidget(self.mode_local)
        mode_row.addWidget(self.mode_server)
        mode_row.addStretch(1)
        sg.addLayout(mode_row)
        sg.addWidget(self.server_row)
        self.server_box = g3
        g3.toggled.connect(self._on_server_box_toggled)
        l3.addWidget(g3)
        l3.addStretch(1)

        self._sync_mode_ui()

        tabs.addTab(t1, "核心算法")
        tabs.addTab(t2, "人声增强")
        tabs.addTab(t3, "预设与监听")
        return tabs

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
        # 切换设备在后台执行，结果由引擎状态消息回报
        self._clear_dev_hint()
        self.engine.reopen_stream(in_id, out_id)
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
        if getattr(self.engine, "running", False):
            reinit = False
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
            self.engine.request_hard_stop()
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
            self.sc.addItem(s.name)
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
            self._pending_speaker = s
            self._pending_after_stop = "load"
            self._set_light(LIGHT_YELLOW, "正在停止以便换角色...")
            self.sb.setEnabled(False)
            self.engine.request_hard_stop()
            return
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
        self._load_started = time.time()
        self._load_log_offset = self._server_log_size()
        self._loader = ModelLoader(self.engine, speaker, self._load_gen, parent=self)
        self._loader.finished_ok.connect(self._on_loaded)
        self._loader.failed.connect(self._on_load_failed)
        self._loader.start()

    def _server_log_size(self):
        try:
            p = package_root() / "logs" / "local_server.log"
            return p.stat().st_size if p.is_file() else 0
        except Exception:
            return 0

    def _loading_stage(self):
        """从本地服务日志的「本次加载增量」推断当前阶段（避免读到历史记录）。"""
        try:
            log_path = package_root() / "logs" / "local_server.log"
            if not log_path.is_file():
                return None
            offset = getattr(self, "_load_log_offset", 0) or 0
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(offset)
                lines = f.readlines()
            for line in reversed(lines):
                low = line.lower()
                if "加速就绪" in line or "graph ready" in low:
                    return "预热完成，即将就绪"
                if "正在预热" in line or "warmup" in low:
                    return "正在预热加速图"
                if "rmvpe" in low or "fcpe" in low or "音高" in line:
                    return "正在加载音高模型"
                if "加载模型" in line or "loading" in low:
                    return "正在加载角色模型"
                if "客户端连接" in line or "server 启动" in low or "服务已启动" in line:
                    return "推理服务已启动"
            return None
        except Exception:
            return None

    def _update_load_progress(self):
        if not hasattr(self, "_loader") or self._loader is None or not self._loader.isRunning():
            return
        elapsed = time.time() - (self._load_started or time.time())
        stage = self._loading_stage() or "正在加载"
        self.state_label.setText(f"加载中… {stage}（{elapsed:.0f} 秒）")
        self.state_label.setStyleSheet(
            "font-size:12px;font-weight:700;color:#b45309;")

    def _on_loaded(self, speaker, gen=0):
        if gen != self._load_gen:
            return
        self._load_started = None
        self.engine.current_speaker = speaker
        self._sync_live_sliders(speaker)
        self.engine.change_pitch(speaker.pitch)
        self.engine.change_index_rate(speaker.index_rate)
        self._set_light(LIGHT_GREEN, "就绪")
        self.sb.setEnabled(True)
        extra = ""
        try:
            pipe = self.engine.pipeline
            rvc = getattr(getattr(pipe, "_real", None), "rvc", None) or getattr(pipe, "rvc", None)
            lab = getattr(getattr(rvc, "model", None), "backend_label", None)
            if lab:
                extra = " · 特征 " + lab
        except Exception:
            extra = ""
        self.status_bar.showMessage("模型已加载: " + speaker.name + extra, 6000)
        self._persist_settings()

    def _on_load_failed(self, err, gen=0):
        if gen != self._load_gen:
            return
        self._load_started = None
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
        self.sb.setEnabled(True)
        self.sb.setText("启动变声")
        self.sb.setProperty("state", "off")
        self.sb.style().unpolish(self.sb); self.sb.style().polish(self.sb)
        self._set_light(LIGHT_RED, "启动失败")
        self._show_dev_hint("启动失败: " + text[:200])
        self.status_bar.showMessage("启动失败: " + text, 8000)
        QMessageBox.warning(self, "启动失败", text[:300])

    def _set_light(self, color, text):
        self.light.setStyleSheet(f"background:{color};border-radius:4px;")
        self.state_label.setText(text)
        c = "#0f766e" if color == LIGHT_GREEN else ("#d97706" if color == LIGHT_YELLOW else ("#dc2626" if color == LIGHT_RED else "#64748b"))
        self.state_label.setStyleSheet(f"font-size:12px;font-weight:700;color:{c};")

    def _on_started(self):
        self._set_light(LIGHT_GREEN, "运行中")
        self._clear_dev_hint()
        self.sb.setEnabled(True)
        self.sb.setText("停止变声")
        self.sb.setProperty("state", "on")
        self.sb.style().unpolish(self.sb); self.sb.style().polish(self.sb)

    def _on_stopped(self):
        self._set_light(LIGHT_GRAY, "已停止")
        self.sb.setEnabled(True)
        self.sb.setText("启动变声")
        self.sb.setProperty("state", "off")
        self.sb.style().unpolish(self.sb); self.sb.style().polish(self.sb)
        self.latency_label.setText("推理 --ms")
        self.e2e_label.setText("嘴到耳 --ms")
        action = self._pending_after_stop
        self._pending_after_stop = None
        if action == "load" and self._pending_speaker is not None:
            speaker = self._pending_speaker
            self._pending_speaker = None
            self._set_light(LIGHT_YELLOW, "加载模型中...")
            self.sb.setEnabled(False)
            self._start_loading(speaker)
        elif action == "mode" and self._pending_mode:
            mode = self._pending_mode
            self._pending_mode = None
            self._apply_mode(mode)

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
        rec_dev = None
        in_name = self.ic.currentDeviceName()
        in_api = self.ic.currentDeviceApi()
        if in_name:
            try:
                for i, d in enumerate(sd.query_devices()):
                    if d['name'] == in_name and d['hostapi'] == in_api and d['max_input_channels'] > 0:
                        rec_dev = i
                        break
            except Exception:
                rec_dev = None
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
        budget = max(30, int(float(self.engine.block_time) * 1000))
        if ms < budget * 0.7:
            c, bg, bd = "#059669", "#ecfdf5", "#a7f3d0"
        elif ms < budget:
            c, bg, bd = "#d97706", "#fffbeb", "#fde68a"
        else:
            c, bg, bd = "#dc2626", "#fef2f2", "#fecaca"
        self.latency_label.setText(f"推理 {ms}ms")
        self.latency_label.setStyleSheet(
            f"font-size:12px;font-weight:700;color:{c};background:{bg};border:1px solid {bd};border-radius:6px;padding:4px 8px;"
        )

    def _tg(self):
        if self.engine.running:
            self.engine.stop()
            return
        if self.engine._engine_busy():
            self.status_bar.showMessage("正在处理，请稍候...", 3000)
            return
        if not self.engine.pipeline.is_loaded:
            QMessageBox.warning(self, "提示", "请先选择角色模型")
            return
        in_id, out_id, err = self._resolve_selected(reinit=False)
        if err:
            self._show_dev_hint(err)
            QMessageBox.warning(self, "设备", err)
            return
        self.engine.input_device = in_id
        self.engine.output_device = out_id
        # 启动全程异步：按钮立即反馈"启动中"，后台完成后由信号切回
        self.sb.setEnabled(False)
        self.sb.setText("启动中…")
        self._set_light(LIGHT_YELLOW, "启动中…")
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
        self.engine._stop_requested = True
        if self.engine.running or self.engine._engine_busy():
            self.engine.request_hard_stop()
            self.engine.wait_idle(2000)
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

    def _auto_calibrate_noise(self):
        """1 秒环境底噪自动采样并校准静音阈值"""
        in_id = self.ic.currentDeviceId()
        self.calib_noise_btn.setEnabled(False)
        self.calib_noise_btn.setText("采样中...")
        self.status_bar.showMessage("请保持安静 1 秒，正在测定麦克风环境底噪...", 3000)

        class CalibNoiseThread(QThread):
            calib_done = Signal(object, str)

            def __init__(self, dev_id, parent=None):
                super().__init__(parent)
                self.dev_id = dev_id

            def run(self):
                import sounddevice as sd
                import numpy as np
                import math, time
                frames = []
                try:
                    def cb(indata, f, t, status):
                        frames.append(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())

                    kwargs = dict(
                        samplerate=48000,
                        channels=1,
                        dtype="float32",
                        blocksize=480,
                        callback=cb,
                    )
                    if self.dev_id is not None:
                        kwargs["device"] = self.dev_id
                    with sd.InputStream(**kwargs):
                        time.sleep(1.0)
                    if not frames:
                        self.calib_done.emit(None, "未采集到音频帧")
                        return
                    raw = np.concatenate(frames)
                    rms = float(np.sqrt(np.mean(np.square(raw)) + 1e-12))
                    noise_db = int(round(20.0 * math.log10(rms + 1e-9)))
                    target_thresh = max(-75, min(-30, noise_db + 6))
                    self.calib_done.emit((noise_db, target_thresh), "")
                except Exception as e:
                    self.calib_done.emit(None, str(e))

        self._calib_thread = CalibNoiseThread(in_id, self)
        def _on_calib_finish(res, err):
            self.calib_noise_btn.setEnabled(True)
            self.calib_noise_btn.setText("测底噪")
            if res is not None:
                noise_db, target_thresh = res
                self.ts.setValue(target_thresh)
                self.status_bar.showMessage(
                    f"✓ 底噪测定完成：当前环境底噪 {noise_db} dB，已自动设定静音阈值与 AGC 门限为 {target_thresh} dB",
                    6000,
                )
            else:
                self.status_bar.showMessage(f"底噪测定失败: {err}", 4000)

        self._calib_thread.calib_done.connect(_on_calib_finish)
        self._calib_thread.start()

    # ── 音质产品化：AGC / 监听 / 预设 / 阶段耗时 ──
    def _on_tradeoff(self, val):
        if getattr(self, "_tq_guard", False):
            return
        t = max(0.0, min(1.0, float(val) / 100.0))
        block = round(0.03 + t * 0.07, 3)
        extra = round(0.50 + t * 1.00, 2)
        fade = round(0.01 + t * 0.02, 3)
        self._tq_guard = True
        try:
            self.bs.setValue(block)
            self.es.setValue(extra)
            self.xs.setValue(fade)
        finally:
            self._tq_guard = False
        self._persist_settings()

    def _sync_tradeoff_slider(self):
        if not hasattr(self, "tq_slider"):
            return
        block = float(self.bs.value())
        t = (block - 0.03) / 0.07
        self._tq_guard = True
        try:
            self.tq_slider.setValue(int(round(max(0.0, min(1.0, t)) * 100)))
        finally:
            self._tq_guard = False

    def _on_hf_mix(self, v):
        self.engine.hf_mix_rate = float(v)
        if v > 0 and self.deess_cb.isChecked():
            self.deess_cb.blockSignals(True)
            self.deess_cb.setChecked(False)
            self.deess_cb.blockSignals(False)
            self.engine.deesser_enable = False
        self._persist_settings()

    def _on_deesser(self, on):
        self.engine.deesser_enable = bool(on)
        if on and self.hf_spin.value() > 0:
            self.hf_spin.blockSignals(True)
            self.hf_spin.setValue(0.0)
            self.hf_spin.blockSignals(False)
            self.engine.hf_mix_rate = 0.0
        self._persist_settings()

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
        self.e2e_label.setText(f"嘴到耳 {ms:.0f}ms")

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
        if "deesser_enable" in params and hasattr(self, "deess_cb"):
            self.deess_cb.setChecked(bool(params["deesser_enable"]))
        if "hf_mix_rate" in params and hasattr(self, "hf_spin"):
            self.hf_spin.setValue(float(params["hf_mix_rate"]))
        if "presence" in params and hasattr(self, "pres_spin"):
            self.pres_spin.setValue(float(params["presence"]))
        if "vad_enable" in params and hasattr(self, "vad_cb"):
            self.vad_cb.setChecked(bool(params["vad_enable"]))
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
            "hf_mix_rate": float(self.hf_spin.value()) if hasattr(self, "hf_spin") else 0.2,
            "presence": float(self.pres_spin.value()) if hasattr(self, "pres_spin") else 0.10,
            "deesser_enable": bool(self.deess_cb.isChecked()) if hasattr(self, "deess_cb") else False,
            "vad_enable": bool(self.vad_cb.isChecked()) if hasattr(self, "vad_cb") else False,
            "formant": float(self.live_formant.value()) if hasattr(self, "live_formant") else 0.0,
            "dry_mix": float(self.live_dry.value()) if hasattr(self, "live_dry") else 0.0,
        }

    def _apply_mode(self, mode):
        if self.engine.mode == mode:
            self._sync_mode_ui()
            return
        if self.engine.running:
            self._pending_mode = mode
            self._pending_after_stop = "mode"
            self.engine.request_hard_stop()
            return
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

    def _on_server_box_toggled(self, on):
        if on and self.engine.mode != "server":
            self.mode_server.setChecked(True)
        elif (not on) and self.engine.mode != "local":
            self.mode_local.setChecked(True)

    def _sync_mode_ui(self):
        server = self.engine.mode == "server"
        if hasattr(self, "server_row"):
            self.server_row.setVisible(server)
        if hasattr(self, "server_box"):
            self.server_box.blockSignals(True)
            self.server_box.setChecked(server)
            self.server_box.blockSignals(False)
        if hasattr(self, "mode_local"):
            self.mode_local.blockSignals(True)
            self.mode_server.blockSignals(True)
            self.mode_local.setChecked(not server)
            self.mode_server.setChecked(server)
            self.mode_local.blockSignals(False)
            self.mode_server.blockSignals(False)

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
        # 音质增强（齿音保留/临场感/去齿音）
        if "hf_mix_rate" in s:
            try:
                self.hf_spin.setValue(float(s["hf_mix_rate"]))
            except Exception:
                pass
        if "presence" in s:
            try:
                self.pres_spin.setValue(float(s["presence"]))
            except Exception:
                pass
        if "deesser_enable" in s:
            self.deess_cb.setChecked(bool(s["deesser_enable"]))
        if "vad_enable" in s:
            self.vad_cb.setChecked(bool(s["vad_enable"]))
        if "vad_threshold" in s:
            try:
                self.vad_th.setValue(float(s["vad_threshold"]))
            except Exception:
                pass
        if self.deess_cb.isChecked() and self.hf_spin.value() > 0:
            self.deess_cb.setChecked(False)
        self._sync_tradeoff_slider()

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
        data["theme"] = "dark" if getattr(self, "_dark", True) else "light"
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
            "hf_mix_rate": float(self.hf_spin.value()),
            "presence": float(self.pres_spin.value()),
            "deesser_enable": bool(self.deess_cb.isChecked()),
            "vad_enable": bool(self.vad_cb.isChecked()),
            "vad_threshold": float(self.vad_th.value()),
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

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "保持当前设置",
            "女声角色 (男变女推荐 +12)",
            "男声角色 (同性别推荐 0)",
            "女声高音 (推荐 +14)",
            "男声低音 (女变男推荐 -12)",
        ])
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        l.addRow("音高推荐:", self.preset_combo)

        self.ps = QSpinBox(); self.ps.setRange(-36, 36); self.ps.setValue(s.pitch)
        self.ps.setSuffix(" 半音")
        self.ps.setToolTip("男转女 +12, 女转男 -12")
        l.addRow("音高偏移:", self.ps)

        self.ne.textChanged.connect(self._on_name_auto_preset)

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

    def _on_preset_changed(self, idx):
        if idx == 1:
            self.ps.setValue(12)
        elif idx == 2:
            self.ps.setValue(0)
        elif idx == 3:
            self.ps.setValue(14)
        elif idx == 4:
            self.ps.setValue(-12)

    def _on_name_auto_preset(self, text):
        t = text.lower()
        if any(k in t for k in ["女", "girl", "female", "sister", "妹", "娘"]) and self.ps.value() == 0:
            self.ps.setValue(12)
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(1)
            self.preset_combo.blockSignals(False)

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
