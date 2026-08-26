#!/usr/bin/env python3
"""
RVC 实时变声 - 桌面客户端
============================
架构: MainWindow(UI) -> VCEngine(音频+信号) -> 本地 RVCPipeline / 远程 RVCClient
源码本地默认进程内推理；打包 exe 才走本机子进程。
"""
import os, sys, json, queue, time, subprocess, logging, traceback, threading
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
    QInputDialog, QScrollArea, QTabWidget, QSystemTrayIcon, QMenu,
)
from PySide6.QtGui import (
    QDragEnterEvent, QDropEvent, QColor, QPainter, QFontMetrics,
    QStandardItemModel, QStandardItem, QIcon, QAction, QPixmap,
)


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
            head = QStandardItem(_hostapi_zh(api_name))
            head.setData("group", Qt.UserRole + 1)
            head.setFlags(Qt.ItemIsEnabled)
            self._model.appendRow(head)
            row += 1
            for i, d in buckets[api_idx]:
                name = d["name"]
                show = name if len(name) <= 36 else name[:34] + "..."
                sr = int(d.get("default_samplerate", 0))
                chs = d[ch]
                detail = (
                    f"{sr // 1000} kHz · {chs} 声道" if sr > 0 else f"{chs} 声道"
                )
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

class ComboItemDelegate(QStyledItemDelegate):
    """下拉项：圆角行、选中青绿条；有副标题时两行（模型 / 索引）。"""
    ROW_H = 36
    ROW_H_SUB = 50

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = option.rect.adjusted(5, 2, -6, -2)
        selected = bool(option.state & QStyle.State_Selected)
        hover = bool(option.state & QStyle.State_MouseOver)
        sub = str(index.data(Qt.UserRole + 2) or "")
        if selected:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#ecfdf5"))
            painter.drawRoundedRect(rect, 8, 8)
            bar = rect.adjusted(0, 8, 0, -8)
            bar.setWidth(3)
            painter.setBrush(QColor("#0f766e"))
            painter.drawRoundedRect(bar, 2, 2)
            title_color = QColor("#0f766e")
            sub_color = QColor("#0f766e")
            pad = 12
        elif hover:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#f1f5f9"))
            painter.drawRoundedRect(rect, 8, 8)
            title_color = QColor("#0f172a")
            sub_color = QColor("#64748b")
            pad = 10
        else:
            title_color = QColor("#334155")
            sub_color = QColor("#94a3b8")
            pad = 10
        text = str(index.data(Qt.DisplayRole) or "")
        text_rect = rect.adjusted(pad, 0, -8, 0)
        fm = QFontMetrics(painter.font())
        if sub:
            title_rect = text_rect.adjusted(0, 3, 0, -16)
            sub_rect = text_rect.adjusted(0, 20, 0, -2)
            font = painter.font()
            font.setBold(selected)
            painter.setFont(font)
            painter.setPen(title_color)
            painter.drawText(
                title_rect, int(Qt.AlignVCenter | Qt.AlignLeft),
                fm.elidedText(text, Qt.ElideRight, max(0, title_rect.width())))
            font.setBold(False)
            font.setPointSize(max(8, font.pointSize() - 1))
            painter.setFont(font)
            painter.setPen(sub_color)
            sfm = QFontMetrics(font)
            painter.drawText(
                sub_rect, int(Qt.AlignVCenter | Qt.AlignLeft),
                sfm.elidedText(sub, Qt.ElideMiddle, max(0, sub_rect.width())))
        else:
            font = painter.font()
            font.setBold(selected)
            painter.setFont(font)
            painter.setPen(title_color)
            painter.drawText(
                text_rect, int(Qt.AlignVCenter | Qt.AlignLeft),
                fm.elidedText(text, Qt.ElideRight, max(0, text_rect.width())))
        painter.restore()

    def sizeHint(self, option, index):
        w = option.rect.width() if option.rect.width() > 0 else 180
        if index.data(Qt.UserRole + 2):
            return QSize(w, self.ROW_H_SUB)
        return QSize(w, self.ROW_H)


class StyledCombo(QComboBox):
    """Fusion 下强制自绘弹出列表，不走 Windows 原生菜单。"""

    def __init__(self, min_width=0, max_visible=8, parent=None):
        super().__init__(parent)
        if min_width > 0:
            self.setMinimumWidth(min_width)
        self.setMaxVisibleItems(max_visible)
        view = QListView(self)
        view.setMouseTracking(True)
        view.setSpacing(0)
        view.setFrameShape(QFrame.NoFrame)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setItemDelegate(ComboItemDelegate(view))
        self.setView(view)
        self.setInsertPolicy(QComboBox.NoInsert)

    def showPopup(self):
        super().showPopup()
        view = self.view()
        n = self.count()
        row_h = view.sizeHintForRow(0) if n else ComboItemDelegate.ROW_H
        vis = min(n, max(1, self.maxVisibleItems()))
        h = vis * row_h + 10
        view.setFixedHeight(h)
        box = view.parentWidget()
        if box is not None and box is not self:
            box.setFixedHeight(h + 8)
            box.setMinimumWidth(max(self.width(), 240))


def create_styled_combo(min_width=0, max_visible=8):
    return StyledCombo(min_width=min_width, max_visible=max_visible)


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
from worker.local_server import (
    is_frozen, package_root, runtime_installed, pack_mode, local_infer_ready)

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


_last_crash_popup = 0.0


def _excepthook(exc_type, exc, tb):
    """未捕获异常：写入 crash.log，并在界面线程弹一次提示。"""
    if exc_type in (KeyboardInterrupt, SystemExit):
        return
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
                + "详细日志：logs/crash.log" + NL + NL
                + _friendly_error(exc))
        QTimer.singleShot(0, _popup)
    except Exception:
        pass


def _thread_excepthook(args):
    if getattr(args, "exc_type", None) in (KeyboardInterrupt, SystemExit):
        return
    _excepthook(args.exc_type, args.exc_value, args.exc_traceback)


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
MSG_SERVER_PACK_NO_LOCAL = (
    "当前安装包是「服务器客户端」，只有界面和声卡，不含本机模型和 GPU 推理。"
    + NL + NL
    + "请勾选「远程服务器」，填写 ws://服务器IP:8765 后点「连接」。"
    + NL + NL
    + "若要在这台电脑上变声，请使用「RVC单机版」。"
)
MSG_NEED_INSTALL_LOCAL = (
    "本地推理尚未安装。"
    + NL + NL
    + "需要英伟达显卡，首次安装约 3.5GB（需联网）。"
    + NL
    + "请先点击「安装本地推理」，完成后再启动变声。"
)
SETTINGS_FILE = PROJECT_ROOT / "user_settings.json"
PRESETS_FILE = PROJECT_ROOT / "presets.json"
DEFAULT_SERVER_URL = "ws://127.0.0.1:8765"
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
        return "输入输出不是同一类接口（请选同一类设备，或点「刷新设备」重试）"
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
    """把底层网络异常翻译成用户可行动的提示。对不上则返回 None。"""
    s = str(e or "").lower()
    if "refused" in s or "10061" in s:
        return "服务器拒绝连接（推理服务没开，或端口不是 8765）"
    if "timed out" in s or "timeout" in s or "10060" in s:
        return "连接超时（检查地址、网络和云安全组是否放行 8765）"
    if "getaddrinfo" in s or "nodename" in s or "name or service" in s:
        return "地址无法解析（请检查 ws://IP:8765 写法）"
    if "unreachable" in s or "10065" in s:
        return "网络不可达（请检查本机网络）"
    if "handshake" in s or "10054" in s or "reset" in s or "closed" in s:
        return "连接被断开（服务器可能崩溃或重启了）"
    if "服务器正忙" in str(e or ""):
        return "服务器正忙，请稍后再点连接或加载"
    return None


def _friendly_error(e):
    """统一错误文案：能翻译就翻译，并告诉用户下一步。"""
    raw = str(e or "").strip()
    net = _friendly_net_error(e)
    if net:
        return net
    low = raw.lower()
    if "cuda" in low and ("out of memory" in low or "oom" in low):
        return "显存不足。请关掉其它占显卡的程序后重试。"
    if "no such file" in low or "not found" in low or "找不到" in raw:
        return "找不到模型或索引文件。本地请核对路径；服务器模式只认文件名，远端要有同名文件。"
    if "模型加载失败" in raw or "load failed" in low:
        return "模型加载失败。请确认文件完整；服务器模式下先点「连接」，并确认远端有这个模型。"
    if "未连接" in raw or "not connected" in low:
        return "尚未连上服务器。请先点「连接」。"
    if "推理未能启动" in raw or "未能启动" in raw:
        return "推理未能启动。请先加载角色，并确认设备可用。"
    if "模型未加载" in raw:
        return "请先选择角色并等待加载完成，再启动变声。"
    if raw:
        return raw[:220]
    return "未知错误"


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


def _normalize_ws_url(url):
    url = (url or "").strip()
    if not url:
        return DEFAULT_SERVER_URL
    if "://" not in url:
        url = "ws://" + url
    return url


F0_CHOICES = (
    ("rmvpe", "准确（RMVPE）"),
    ("fcpe", "较快（FCPE）"),
    ("pm", "最快（PM）"),
)


def fill_f0_combo(cb, current="rmvpe"):
    cb.blockSignals(True)
    cb.clear()
    for key, label in F0_CHOICES:
        cb.addItem(label, key)
    i = cb.findData(current or "rmvpe")
    cb.setCurrentIndex(i if i >= 0 else 0)
    cb.blockSignals(False)


def f0_from_combo(cb):
    v = cb.currentData()
    return v if v else "rmvpe"


def set_f0_combo(cb, key):
    i = cb.findData(key or "rmvpe")
    if i >= 0:
        cb.setCurrentIndex(i)


def _hostapi_zh(name):
    n = (name or "").upper()
    if "ASIO" in n:
        return "专业声卡"
    if "WASAPI" in n:
        return "系统低延迟"
    if "WDM" in n or "KS" in n:
        return "内核音频"
    if "DIRECTSOUND" in n:
        return "经典音频"
    if "MME" in n:
        return "兼容模式"
    return name or "其他"


def to_server_path(local_path: str) -> str:
    """只传文件名，由服务器在自己的 assets/weights、logs 下解析。"""
    return Path(str(local_path or "")).name


def _speaker_file_sub(speaker):
    """下拉副行：模型文件 · 索引文件。"""
    pth = Path(str(getattr(speaker, "model_path", "") or "")).name or "无模型"
    idx = Path(str(getattr(speaker, "index_path", "") or "")).name
    if not idx:
        idx = "无索引"
    return pth + "  ·  " + idx

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
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 6px 28px 6px 10px;
    min-height: 28px;
    font-size: 13px;
    font-weight: 600;
    color: #0f172a;
}
QComboBox:hover {
    border-color: #94a3b8;
    background-color: #f8fafc;
}
QComboBox:focus, QComboBox:on {
    border-color: #0f766e;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border: none;
}
QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #64748b;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: none;
    outline: none;
    padding: 4px 0;
    selection-background-color: transparent;
    selection-color: #0f766e;
}
QComboBoxPrivateContainer {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 4px;
}
QComboBox#speakerCombo {
    min-height: 32px;
    font-size: 13px;
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
QPushButton:disabled {
    color: #9ca3af;
    background-color: #f3f4f6;
    border-color: #e5e7eb;
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
QPushButton#btnGhost:pressed {
    background-color: #e5e7eb;
}
QPushButton#btnGhost:disabled {
    color: #94a3b8;
    background-color: #f1f5f9;
    border-color: #e2e8f0;
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
QPushButton#btnConnect:pressed {
    background-color: #134e4a;
}
QPushButton#btnConnect:disabled {
    background-color: #5eead4;
    color: #115e59;
    border: none;
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
                err = getattr(self.pipeline, "last_error", "") or "模型加载失败"
                self.failed.emit(err, self.gen)
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
    crashed = Signal(str)

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
        try:
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
        except Exception as e:
            logging.exception("推理线程异常")
            if self.running:
                self.crashed.emit(str(e)[:200])

    def stop(self):
        self.running = False
        if self.isRunning():
            self.quit()
            self.wait(1800)


# 录音测试暂时关闭
# class RecThread(QThread):
#     """录音测试线程：录 N 秒 → 用当前角色变声 → 保存 wav"""
#     done = Signal(str, str)   # 保存路径, 错误信息
#
#     def __init__(self, engine, seconds=10, device=None, parent=None):
#         super().__init__(parent)
#         self.engine = engine
#         self.seconds = seconds
#         self.device = device
#
#     def run(self):
#         import sounddevice as sd
#         import numpy as np
#         import time, os
#         try:
#             c = self.engine.pipeline
#             if c is None:
#                 self.done.emit("", "推理引擎未就绪")
#                 return
#             if not c.is_connected():
#                 if not c.connect(timeout=5):
#                     self.done.emit("", "无法连接服务器")
#                     return
#             try:
#                 started = c.start(**self.engine.merged_params())
#                 if started is False:
#                     self.done.emit("", "无法启动推理")
#                     return
#             except Exception as e:
#                 self.done.emit("", "无法启动推理: " + str(e))
#                 return
#             SR = c.samplerate
#             BLOCK = getattr(c, "_block_frame", None)
#             if not BLOCK:
#                 self.done.emit("", "推理块大小未知，请先成功加载角色")
#                 return
#             frames = []
#             rec_log = []
#             def cb(indata, frames_, times, status):
#                 frames.append(indata[:, 0].copy())
#             kwargs = dict(samplerate=SR, channels=1, dtype='float32',
#                           blocksize=BLOCK, callback=cb)
#             if self.device is not None:
#                 kwargs['device'] = self.device
#             with sd.InputStream(**kwargs):
#                 time.sleep(self.seconds)
#             if not frames:
#                 self.done.emit("", "没有录到音频")
#                 return
#             raw = np.concatenate(frames)
#             outs = []
#             for i in range(0, len(raw) - BLOCK + 1, BLOCK):
#                 out, _ = c.process_chunk(raw[i:i + BLOCK])
#                 outs.append(np.asarray(out).reshape(-1))
#             if not outs:
#                 self.done.emit("", "录音太短")
#                 return
#             os.makedirs('record_out', exist_ok=True)
#             out = np.concatenate(outs)
#             spk = self.engine.current_speaker
#             fname = os.path.join('record_out', f'rec_{spk.name}_{int(time.time())}.wav')
#             pcm = np.clip(out * 32767, -32768, 32767).astype(np.int16)
#             with wave_mod.open(fname, 'wb') as wf:
#                 wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
#                 wf.writeframes(pcm.tobytes())
#             self.done.emit(fname, "")
#         except Exception as e:
#             self.done.emit("", str(e))


class EngineStartThread(QThread):
    """后台执行启动/重建/停机的阻塞部分（网络等待、CUDA 预热、声卡打开、等推理退出）。

    UI 线程只负责发起与信号响应，绝不进入 wait——杜绝"未响应"。
    """

    def __init__(self, engine, action, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.action = action

    def run(self):
        try:
            if self.action == "start":
                self.engine._start_blocking()
            elif self.action == "reopen":
                self.engine._reopen_blocking()
            elif self.action == "stop":
                self.engine._hard_stop()
            elif self.action == "recover":
                self.engine._recover_blocking()
        except Exception as e:
            logging.exception("引擎线程失败: %s", self.action)
            if self.action == "start":
                self.engine.load_failed.emit(_friendly_error(e))
            else:
                self.engine.status_msg.emit("操作失败: " + _friendly_error(e))


class ServerConnectThread(QThread):
    """后台连接远程推理服务器，避免点「连接」卡死 UI。"""
    done = Signal(bool, str)

    def __init__(self, pipeline, url, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.url = url

    def run(self):
        try:
            self.pipeline.set_server_url(self.url)
            ok = bool(self.pipeline.connect(timeout=5))
            extra = ""
            if ok:
                gpu = getattr(self.pipeline, "gpu_name", "") or ""
                info = {}
                try:
                    info = self.pipeline.loaded_file_info() or {}
                except Exception:
                    info = {}
                model = info.get("model_path") or ""
                idx = info.get("index_path") or ""
                idx_bit = ""
                if idx:
                    idx_bit = "索引 " + Path(str(idx)).name
                    if not info.get("index_loaded", True):
                        idx_bit += "（未启用）"
                elif model:
                    idx_bit = "无索引"
                extra = " · ".join(
                    x for x in (gpu, Path(str(model)).name if model else "未加载模型", idx_bit) if x)
            self.done.emit(ok, extra)
        except Exception as e:
            self.done.emit(False, str(e)[:160])


# ==============================================================================
# 推理引擎（本地）
# ==============================================================================
class VCEngine(QObject):
    status_msg = Signal(str); infer_time = Signal(int)
    started_ok = Signal(); stopped_ok = Signal()
    load_failed = Signal(str)
    recover_progress = Signal(int, int)  # 当前次数, 最多次数
    recover_ok = Signal()
    recover_failed = Signal(str)
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

            # 不能把 self(VCEngine, 主线程对象) 当 parent 传给跨线程创建的 QThread，
            # 否则 Qt 报「Cannot create children for a parent in a different thread」。
            # 线程生命周期已由 _hard_stop 的 wait/deleteLater 手动管理，无需 parent。
            self.worker_thread = InferenceWorkerThread(
                self.pipeline, self.input_queue, self.output_queue)
            self.worker_thread.infer_done.connect(self._on_worker_infer_done)
            self.worker_thread.stage_stats.connect(self.stage_stats)
            self.worker_thread.spectrum.connect(self.spectrum)
            self.worker_thread.xrun_occurred.connect(self._on_worker_xrun)
            self.worker_thread.need_recover.connect(
                self._try_recover, Qt.QueuedConnection)
            self.worker_thread.crashed.connect(
                self._on_worker_crash, Qt.QueuedConnection)
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
                return True, "专业声卡直通（极低延迟）"
            except Exception as e:
                errors.append(e)

        # 2. WASAPI 独占（次低延迟）
        try:
            self.stream = sd.Stream(
                extra_settings=sd.WasapiSettings(exclusive=True), **kwargs)
            return True, "系统低延迟独占"
        except Exception as e:
            errors.append(e)

        # 3. WASAPI 共享模式
        try:
            try:
                extra = sd.WasapiSettings(exclusive=False, auto_convert=True)
            except TypeError:
                extra = sd.WasapiSettings(exclusive=False)
            self.stream = sd.Stream(extra_settings=extra, **kwargs)
            return True, "系统共享（独占失败: %s）" % _wasapi_fail_reason(errors[-1])
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
            return True, "系统共享 · 双声道"
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

    def _on_worker_crash(self, msg):
        if self.mode == "server":
            self._try_recover()
            return
        self.load_failed.emit("推理中断: " + _friendly_error(msg))
        self.request_hard_stop()

    def _try_recover(self):
        if not self.running:
            return
        # 仅远程服务器模式走网络重连；本地进程内推理不碰这条路径
        if self.mode != "server":
            return
        if self._engine_busy():
            return
        now = time.time()
        if now - self._last_recover < 2.0:
            return
        self._last_recover = now
        self._start_thread = EngineStartThread(self, "recover")
        self._start_thread.start()

    def _recover_blocking(self):
        if self.mode != "server":
            return
        max_n = 3
        last_err = "无法连接服务器"
        for i in range(1, max_n + 1):
            if self._stop_requested or not self.running:
                if self._stop_requested:
                    self._hard_stop()
                return
            self.recover_progress.emit(i, max_n)
            self.status_msg.emit("连接中断，正在重连 %d/%d…" % (i, max_n))
            try:
                self.pipeline.abort()
            except Exception:
                pass
            time.sleep(0.5 if i == 1 else min(4.0, float(i)))
            if self._stop_requested or not self.running:
                if self._stop_requested:
                    self._hard_stop()
                return
            try:
                if self._ensure_connected() and self._ensure_model():
                    self.pipeline.start(**self.merged_params())
                    self.recover_ok.emit()
                    self.status_msg.emit("已重新连上服务器")
                    return
                last_err = getattr(self.pipeline, "last_error", "") or "无法连接服务器"
            except Exception as e:
                last_err = str(e)
        self._hard_stop()
        self.recover_failed.emit(last_err)

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
        self.status_msg.emit("推理偏慢：已自动关人声识别/去齿音并降低检索，稳住实时")

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
        self._apply_app_icon()
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
        if pack_mode() == "server":
            mode = "server"
        self.engine = VCEngine(mode=mode, server_url=url)
        self._ui_ready = False
        self._last_local_prompt = 0.0
        self._last_alert = ("", 0.0)
        self.engine.status_msg.connect(self._on_status)
        self.engine.infer_time.connect(self._on_infer_time)
        self.engine.started_ok.connect(self._on_started)
        self.engine.stopped_ok.connect(self._on_stopped)
        self.engine.load_failed.connect(self._on_start_failed)
        self.engine.recover_progress.connect(self._on_recover_progress)
        self.engine.recover_ok.connect(self._on_recover_ok)
        self.engine.recover_failed.connect(self._on_recover_failed)
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
        self._ui_ready = True
        self._setup_tray()
        QTimer.singleShot(400, self._maybe_prompt_local_infer)

    def _on_rms_levels(self, in_db, out_db):
        self.in_meter.set_level(in_db)
        self.out_meter.set_level(out_db)

    def _on_spectrum(self, bins):
        if hasattr(self, "spectrum"):
            self.spectrum.set_bins(bins)

    def _icon_path(self, *names):
        roots = (
            package_root() / "assets" / "icons",
            PROJECT_ROOT / "assets" / "icons",
        )
        for root in roots:
            for name in names:
                p = root / name
                if p.is_file():
                    return str(p)
        return ""

    def _apply_app_icon(self):
        path = self._icon_path("app.ico", "app.png")
        if not path:
            return
        icon = QIcon(path)
        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def _setup_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        path = self._icon_path("tray.ico", "app.ico", "tray.png")
        if not path:
            return
        self.tray = QSystemTrayIcon(QIcon(path), self)
        self.tray.setToolTip("RVC 实时变声")
        menu = QMenu(self)
        act_show = QAction("打开窗口", self)
        act_show.triggered.connect(self._show_from_tray)
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(QApplication.instance().quit)
        menu.addAction(act_show)
        menu.addSeparator()
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _show_from_tray(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_tray_activated(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self._show_from_tray()

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
        self.status_bar.showMessage("正在检测虚拟声卡…", 3000)
        QApplication.processEvents()
        dlg = CableWizard(self, self.ic, self.oc)
        if dlg.exec():
            self._on_device_changed("output")
            self._persist_settings()
            self.status_bar.showMessage("虚拟声卡设置已更新", 4000)

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

        logo = QLabel()
        logo.setFixedSize(36, 36)
        logo.setScaledContents(False)
        logo_path = self._icon_path("app_64.png", "app.png", "app.ico")
        if logo_path:
            pix = QPixmap(logo_path).scaled(
                36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo.setPixmap(pix)
            logo.setToolTip("RVC 实时变声")
        top.addWidget(logo)
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
        self.sc.setObjectName("speakerCombo")
        self.sc.setMinimumHeight(36)
        self.sc.setToolTip("选择角色")
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
        self.cur_model.setWordWrap(True)
        self.cur_model.setTextInteractionFlags(Qt.TextSelectableByMouse)
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
        self.conn_btn.setMinimumWidth(88)
        self.conn_btn.setMinimumHeight(32)
        self.conn_btn.setCursor(Qt.PointingHandCursor)
        self.conn_btn.setToolTip("改完地址后点这里，按新地址重连（不重新加载角色）")
        self.conn_btn.clicked.connect(self._connect_server)
        self.server_row = QWidget()
        sl = QHBoxLayout(self.server_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)
        sl.addWidget(self._lbl("服务器"))
        sl.addWidget(self.server_edit, 1)
        sl.addWidget(self.conn_btn)
        self.server_status = QLabel("未连接")
        self.server_status.setWordWrap(True)
        self._set_server_status("未连接", "idle")

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
        self.local_install_btn.setToolTip("下载并安装本机推理环境（需英伟达显卡与网络）")
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
        self.ic.setToolTip("输入输出需选同一类接口；选择后会自动对齐另一侧")
        self.ic.currentIndexChanged.connect(lambda: self._on_device_changed("input"))
        dl.addWidget(self.ic, 0, 1)

        dl.addWidget(self._lbl("输出设备"), 1, 0)
        self.oc = DeviceCombo(direction="output")
        self.oc.setMinimumHeight(30)
        self.oc.setToolTip("输入输出需选同一类接口；选择后会自动对齐另一侧")
        self.oc.currentIndexChanged.connect(lambda: self._on_device_changed("output"))
        dl.addWidget(self.oc, 1, 1)

        rb = QPushButton("刷新设备")
        rb.setObjectName("btnGhost")
        rb.setMinimumHeight(28)
        rb.setCursor(Qt.PointingHandCursor)
        rb.clicked.connect(self._refresh_devices_clicked)
        self.dev_refresh_btn = rb

        cable_btn = QPushButton("虚拟声卡向导")
        cable_btn.setObjectName("btnGhost")
        cable_btn.setMinimumHeight(28)
        cable_btn.setToolTip("检测虚拟声卡；没有则打开安装页")
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

        # 录音测试暂时关闭
        # self.rec_btn = QPushButton("录音测试 (10 秒)")
        # self.rec_btn.setObjectName("btnGhost")
        # self.rec_btn.setMinimumHeight(34)
        # self.rec_btn.setToolTip("录 10 秒，用当前角色变声后保存并播放")
        # self.rec_btn.clicked.connect(self._rec)
        # l.addWidget(self.rec_btn)
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
        fill_f0_combo(self.fc, DEFAULT_PARAMS["f0method"])
        self.fc.setMinimumHeight(28)
        self.fc.currentIndexChanged.connect(
            lambda: setattr(self.engine, "f0method", f0_from_combo(self.fc)))
        self.fc.setToolTip("音高提取：准确最稳，最快负担最小")

        self.bs = QDoubleSpinBox()
        self.bs.setRange(0.03, 0.5)
        self.bs.setSingleStep(0.01)
        self.bs.setValue(DEFAULT_PARAMS["block_time"])
        self.bs.setDecimals(3)
        self.bs.setSuffix(" 秒")
        self.bs.setMinimumHeight(28)
        self.bs.setToolTip("音频块时长（秒）。越小嘴到耳越低，GPU 越容易卡顿。运行中修改下次启动生效")
        self.bs.valueChanged.connect(lambda v: setattr(self.engine, "block_time", v))
        self.bs.valueChanged.connect(lambda _: self._sync_tradeoff_slider())

        self.xs = QDoubleSpinBox()
        self.xs.setRange(0.01, 0.5)
        self.xs.setSingleStep(0.01)
        self.xs.setValue(DEFAULT_PARAMS["crossfade_time"])
        self.xs.setSuffix(" 秒")
        self.xs.setMinimumHeight(28)
        self.xs.setToolTip("运行中修改将在下次启动后生效")
        self.xs.valueChanged.connect(lambda v: setattr(self.engine, "crossfade_time", v))

        self.es = QDoubleSpinBox()
        self.es.setRange(0.4, 5.0)
        self.es.setSingleStep(0.1)
        self.es.setValue(DEFAULT_PARAMS["extra_time"])
        self.es.setSuffix(" 秒")
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
        self.calib_noise_btn.setMinimumWidth(72)
        self.calib_noise_btn.setMinimumHeight(28)
        self.calib_noise_btn.setCursor(Qt.PointingHandCursor)
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
            ("音高算法", self.fc),
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

        g_dsp = QGroupBox("人声修饰")
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
        self.vad_cb = QCheckBox("人声识别")
        self.vad_cb.setMinimumHeight(26)
        self.vad_cb.setToolTip("区分人声与环境杂音，非人声自动静音")
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

        self.agc_cb = QCheckBox("输入自动增益")
        self.agc_cb.setMinimumHeight(26)
        self.agc_cb.setToolTip("输入音量自动拉齐，远近说话时更稳（实时生效）")
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
        if hasattr(self, "server_status"):
            sg.addWidget(self.server_status)
        hint = QLabel("在带显卡的电脑上启动推理服务后，这里填写 ws://那台机器IP:8765")
        hint.setWordWrap(True)
        hint.setObjectName("fieldLabel")
        sg.addWidget(hint)
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
                        "请手动把输入输出选为同一类接口。")
                    self.status_bar.showMessage("输入输出接口不一致，无法自动对齐", 8000)
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
                        "请手动把输入输出选为同一类接口。")
                    self.status_bar.showMessage("输入输出接口不一致，无法自动对齐", 8000)
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
                    f"输入（{_hostapi_zh(ain)}）与输出（{_hostapi_zh(aout)}）不是同一类接口，"
                    "请选同一类后再启动")
        return in_id, out_id, None

    def _refresh_devices_clicked(self):
        btn = getattr(self, "dev_refresh_btn", None)
        if btn is not None:
            btn.setEnabled(False)
            btn.setText("刷新中…")
        self.status_bar.showMessage("正在刷新音频设备…", 0)
        QApplication.processEvents()
        self._rd()
        self.status_bar.showMessage("设备列表已更新", 4000)
        if btn is not None:
            btn.setEnabled(True)
            btn.setText("已刷新")
            QTimer.singleShot(
                900,
                lambda: btn.setText("刷新设备") if btn.text() == "已刷新" else None,
            )

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
        saved = self._settings.get("speaker") or ""
        names = [s.name for s in self.speaker_mgr.speakers]
        if saved in names:
            return names.index(saved)
        alias = saved[:-1] + "轮" if saved.endswith("e") else ""
        if alias in names:
            return names.index(alias)
        return 0

    def _rl(self):
        self.sc.blockSignals(True)
        self.sc.clear()
        for s in self.speaker_mgr.speakers:
            self.sc.addItem(s.name)
            self.sc.setItemData(self.sc.count() - 1, _speaker_file_sub(s), Qt.UserRole + 2)
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
        self.cur_model.setText(
            "模型: " + (Path(s.model_path).name if s.model_path else "无")
            + NL
            + "索引: " + (Path(s.index_path).name if s.index_path else "无")
        )
        self.cur_info.setText(
            f"音高 {s.pitch:+d}  检索 {s.index_rate:.1f}  共振峰 {s.formant:+.1f}"
            if getattr(s, "formant", 0.0) != 0.0
            else f"音高 {s.pitch:+d}  检索 {s.index_rate:.1f}")
        self._sync_live_sliders(s)
        if self.engine.current_speaker is s:
            self._set_light(LIGHT_GREEN, "就绪")
            self.sb.setEnabled(True)
            self._show_loaded_paths(s)
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
        if self.engine.mode == "local" and not self._guard_local_infer(
                prompt=getattr(self, "_ui_ready", False)):
            if pack_mode() == "server":
                self._set_light(LIGHT_YELLOW, "请连接远程服务器")
            else:
                self._set_light(LIGHT_YELLOW, "请先安装本地推理")
            self.sb.setEnabled(True)
            return
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
        self._show_loaded_paths(speaker)
        info = self._pipeline_file_info()
        pth = info.get("model_path") or Path(speaker.model_path).name
        idx = info.get("index_path") or ""
        idx_txt = Path(idx).name if idx else "无索引"
        if idx and not info.get("index_loaded"):
            idx_txt += "（未启用）"
        self.status_bar.showMessage(
            "已加载 " + speaker.name + " · " + Path(str(pth)).name
            + " · " + idx_txt + extra, 8000)
        self._persist_settings()

    def _pipeline_file_info(self):
        pipe = self.engine.pipeline
        fn = getattr(pipe, "loaded_file_info", None)
        if callable(fn):
            try:
                return fn() or {}
            except Exception:
                return {}
        rvc = getattr(getattr(pipe, "_real", None), "rvc", None) or getattr(pipe, "rvc", None)
        if rvc is None:
            return {}
        pth = getattr(rvc, "pth_path", "") or ""
        idx = getattr(rvc, "index_path", "") or ""
        return {
            "model_path": pth,
            "index_path": idx,
            "index_loaded": getattr(rvc, "index", None) is not None,
        }

    def _show_loaded_paths(self, speaker=None):
        info = self._pipeline_file_info()
        pth = (info.get("model_path") or "").strip()
        idx = (info.get("index_path") or "").strip()
        if not pth and speaker is not None:
            pth = speaker.model_path or ""
        if not pth:
            return
        lines = ["模型: " + pth]
        if idx:
            tag = "" if info.get("index_loaded", True) else "（未启用）"
            lines.append("索引: " + idx + tag)
        else:
            req = ""
            if speaker is not None:
                req = Path(str(speaker.index_path or "")).name
            lines.append("索引: 未找到 " + req if req else "索引: 无")
        self.cur_model.setText(NL.join(lines))
        self.cur_model.setToolTip(NL.join(lines))

    def _alert(self, title, text, kind="warn"):
        """避免同一错误短时间连弹。"""
        text = (text or "").strip()
        key = title + "|" + text[:80]
        now = time.time()
        last_key, last_t = getattr(self, "_last_alert", ("", 0.0))
        if key == last_key and now - last_t < 3.0:
            return
        self._last_alert = (key, now)
        if kind == "info":
            QMessageBox.information(self, title, text)
        else:
            QMessageBox.warning(self, title, text)

    def _on_load_failed(self, err, gen=0):
        if gen != self._load_gen:
            return
        self._load_started = None
        self._set_light(LIGHT_RED, "加载失败")
        self.sb.setEnabled(True)
        text = _friendly_error(err)
        hint = "请核对模型/索引文件是否存在。"
        if self.engine.mode == "server":
            connected = False
            try:
                connected = bool(self.engine.pipeline.is_connected())
            except Exception:
                connected = False
            if connected:
                hint = "服务器已连上。请确认远端有同名的模型/索引文件（只认文件名，不认本机路径）。"
            else:
                hint = "请先点「连接」。服务器只认文件名，远端要有同名 .pth / .index。"
        self.status_bar.showMessage("加载失败: " + text, 8000)
        self._show_dev_hint("加载失败: " + text[:160])
        self._alert("模型加载失败", text + NL + NL + hint)

    def _on_start_failed(self, msg):
        """启动变声失败：常驻提示 + 弹窗。"""
        text = _friendly_error(msg)
        self.sb.setEnabled(True)
        self.sb.setText("启动变声")
        self.sb.setProperty("state", "off")
        self.sb.style().unpolish(self.sb); self.sb.style().polish(self.sb)
        self._set_light(LIGHT_RED, "启动失败")
        self._show_dev_hint("启动失败: " + text[:200])
        self.status_bar.showMessage("启动失败: " + text, 8000)
        self._alert("启动失败", text)

    def _on_recover_progress(self, n, total):
        self._set_light(LIGHT_YELLOW, "正在重连 %d/%d" % (n, total))
        if self.engine.mode == "server":
            self._set_server_status("连接中断，正在重连 %d/%d…" % (n, total), "busy")

    def _on_recover_ok(self):
        self._set_light(LIGHT_GREEN, "运行中")
        self._clear_dev_hint()
        if self.engine.mode == "server":
            self._set_server_status("已重新连上服务器", "ok")
        self.status_bar.showMessage("已重新连上服务器，变声继续", 5000)

    def _on_recover_failed(self, err):
        text = _friendly_error(err)
        self._set_light(LIGHT_RED, "连接中断")
        self._show_dev_hint("服务器已断开，变声已停止")
        if self.engine.mode == "server":
            self._set_server_status("已断开", "fail")
        self.status_bar.showMessage("重连失败，变声已停止", 8000)
        self._alert(
            "服务器中断",
            text + NL + NL +
            "变声已停止。请确认推理服务仍在运行，点「连接」成功后再启动变声。")

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
        hold = 8000 if any(k in str(m) for k in ("失败", "中断", "错误", "断开")) else 5000
        self.status_bar.showMessage(m, hold)

    # 录音测试暂时关闭
    # def _rec(self):
    #     """录音测试：录 10 秒 → 当前角色变声 → 保存 wav 并播放"""
    #     if not self.engine.pipeline.is_loaded:
    #         QMessageBox.warning(self, "提示", "请先加载角色模型")
    #         return
    #     if self.engine.running:
    #         QMessageBox.warning(self, "提示", "请先停止实时转换")
    #         return
    #     if hasattr(self, "_rec_thread") and self._rec_thread is not None and self._rec_thread.isRunning():
    #         QMessageBox.information(self, "提示", "录音正在进行中，请稍候...")
    #         return
    #     rec_dev = None
    #     in_name = self.ic.currentDeviceName()
    #     in_api = self.ic.currentDeviceApi()
    #     if in_name:
    #         try:
    #             for i, d in enumerate(sd.query_devices()):
    #                 if d['name'] == in_name and d['hostapi'] == in_api and d['max_input_channels'] > 0:
    #                     rec_dev = i
    #                     break
    #         except Exception:
    #             rec_dev = None
    #     self.rec_btn.setEnabled(False)
    #     self.rec_btn.setText("录音中…")
    #     self._rec_thread = RecThread(self.engine, 10, rec_dev, self)
    #     self._rec_thread.done.connect(self._rec_done)
    #     self._rec_thread.start()
    #
    # def _rec_done(self, path, err):
    #     self.rec_btn.setEnabled(True)
    #     self.rec_btn.setText("录音测试 10 秒")
    #     if err:
    #         QMessageBox.warning(self, "录音失败", err)
    #         return
    #     msg = "已保存: " + path + NL + NL + "正在用系统播放器打开，请听变声效果。"
    #     QMessageBox.information(self, "录音完成", msg)
    #     try:
    #         import os
    #         os.startfile(os.path.abspath(path))
    #     except Exception:
    #         pass

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
        if not self._guard_local_infer():
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
        self.calib_noise_btn.setText("测定中…")
        self.status_bar.showMessage("请保持安静 1 秒，正在测定麦克风环境底噪…", 0)
        QApplication.processEvents()

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
                    f"测定完成：环境底噪 {noise_db} dB，静音阈值已设为 {target_thresh} dB",
                    6000,
                )
            else:
                self.status_bar.showMessage("测定失败: " + str(err), 4000)

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
            set_f0_combo(self.fc, str(params["f0method"]))
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

    def _local_infer_block_reason(self):
        """冻结包不能本机推理时的原因：server / install；可以跑则 None。"""
        if self.engine.mode != "local":
            return None
        if not is_frozen():
            return None
        if pack_mode() == "server":
            return "server"
        if not local_infer_ready():
            return "install"
        return None

    def _show_local_infer_prompt(self, why):
        now = time.time()
        if now - getattr(self, "_last_local_prompt", 0) < 2.5:
            return
        self._last_local_prompt = now
        if why == "server":
            QMessageBox.information(self, "不能本机推理", MSG_SERVER_PACK_NO_LOCAL)
        elif why == "install":
            QMessageBox.information(self, "请先安装本地推理", MSG_NEED_INSTALL_LOCAL)

    def _maybe_prompt_local_infer(self):
        why = self._local_infer_block_reason()
        if why:
            self._show_local_infer_prompt(why)

    def _guard_local_infer(self, prompt=True):
        why = self._local_infer_block_reason()
        if not why:
            return True
        if prompt and getattr(self, "_ui_ready", False):
            self._show_local_infer_prompt(why)
        return False

    def _apply_mode(self, mode, reload_speaker=True):
        if mode == "local" and is_frozen() and not local_infer_ready():
            why = "server" if pack_mode() == "server" else "install"
            if why == "server":
                self._sync_mode_ui()
                self._show_local_infer_prompt(why)
                return
            self._show_local_infer_prompt(why)
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
        if reload_speaker and self.speaker_mgr.speakers:
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

    def _set_server_status(self, text, kind="idle"):
        if not hasattr(self, "server_status"):
            return
        colors = {
            "idle": ("#6b7280", "#f8fafc", "#e5e7eb"),
            "busy": ("#1d4ed8", "#dbeafe", "#93c5fd"),
            "ok": ("#0f766e", "#ecfdf5", "#99f6e4"),
            "fail": ("#b91c1c", "#fef2f2", "#fecaca"),
        }
        fg, bg, bd = colors.get(kind, colors["idle"])
        self.server_status.setText(text)
        self.server_status.setStyleSheet(
            f"font-size:12px;font-weight:600;color:{fg};background:{bg};"
            f"border:1px solid {bd};border-radius:6px;padding:6px 8px;")

    def _tick_connect_busy(self):
        btn = getattr(self, "conn_btn", None)
        t = getattr(self, "_conn_thread", None)
        if btn is None or t is None or not t.isRunning():
            timer = getattr(self, "_conn_busy_timer", None)
            if timer is not None:
                timer.stop()
            return
        n = getattr(self, "_conn_dots", 0) + 1
        self._conn_dots = n % 3
        btn.setText("连接中" + "." * (self._conn_dots + 1))

    def _connect_server(self):
        url = _normalize_ws_url(self.server_edit.text())
        self.server_edit.setText(url)
        t = getattr(self, "_conn_thread", None)
        if t is not None and t.isRunning():
            self._set_server_status("正在连接，请稍候…", "busy")
            self.status_bar.showMessage("正在连接，请稍候…", 3000)
            return
        self.conn_btn.setEnabled(False)
        self.conn_btn.setText("连接中…")
        self._set_server_status("正在连接 " + url, "busy")
        self.status_bar.showMessage("正在连接 " + url, 0)
        self._conn_dots = 0
        timer = getattr(self, "_conn_busy_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setInterval(400)
            timer.timeout.connect(self._tick_connect_busy)
            self._conn_busy_timer = timer
        timer.start()
        self.engine.server_url = url
        self._persist_settings()
        if self.engine.mode != "server":
            self._apply_mode("server", reload_speaker=False)
        self._conn_thread = ServerConnectThread(self.engine.pipeline, url, self)
        self._conn_thread.done.connect(self._on_server_connected, Qt.UniqueConnection)
        self._conn_thread.start()

    def _on_server_connected(self, ok, extra):
        timer = getattr(self, "_conn_busy_timer", None)
        if timer is not None:
            timer.stop()
        self.conn_btn.setEnabled(True)
        url = _normalize_ws_url(self.server_edit.text())
        if ok:
            self.conn_btn.setText("已连接")
            QTimer.singleShot(1600, lambda: (
                self.conn_btn.setText("连接")
                if self.conn_btn.text() == "已连接" else None
            ))
            msg = "已连接 " + url
            if extra:
                msg += "  (" + extra + ")"
            self._set_server_status(msg, "ok")
            self.status_bar.showMessage(msg, 6000)
            if self.speaker_mgr.speakers and self.sc.currentIndex() >= 0:
                self.engine.current_speaker = None
                self._sel(self.sc.currentIndex())
        else:
            self.conn_btn.setText("连接")
            err = extra or "请确认：那台机器已启动推理服务，地址端口正确，防火墙放行 8765。"
            self._set_server_status("连接失败", "fail")
            self.status_bar.showMessage("连接失败: " + err, 8000)
            QMessageBox.warning(
                self, "连接服务器",
                "无法连接 " + url + NL + NL + _friendly_error(err))

    # ── 冻结版：本地推理安装 ──
    def _refresh_local_install(self):
        if not is_frozen():
            self.local_install_row.setVisible(False)
            return
        self.local_install_row.setVisible(True)
        if pack_mode() == "server":
            self.local_install_lbl.setText(
                "本包为服务器客户端，不能本机推理。请连接远程服务器。")
            self.local_install_btn.setVisible(False)
            return
        self.local_install_btn.setVisible(True)
        if runtime_installed():
            self.local_install_lbl.setText("本地推理环境已安装（约 3.5GB）")
            self.local_install_btn.setText("重新安装")
        else:
            self.local_install_lbl.setText(
                "本地推理未安装：需要英伟达显卡，点击右侧按钮开始（需联网，约 3.5GB）")
            self.local_install_btn.setText("安装本地推理")

    def _install_local(self):
        if pack_mode() == "server":
            self._show_local_infer_prompt("server")
            return
        root = package_root()
        bat = root / "install_local.bat"
        if not bat.is_file():
            QMessageBox.warning(
                self, "安装本地推理",
                "未找到 install_local.bat。单机版请重新解压完整安装包。")
            return
        self.local_install_btn.setEnabled(False)
        self.local_install_btn.setText("正在打开…")
        QApplication.processEvents()
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "RVC 本地推理安装", str(bat)],
                cwd=str(root))
        except Exception as e:
            self.local_install_btn.setEnabled(True)
            self.local_install_btn.setText("安装本地推理")
            QMessageBox.warning(self, "安装本地推理", "无法打开安装窗口: " + str(e))
            return
        self.local_install_btn.setEnabled(True)
        self.local_install_btn.setText("已打开安装")
        QTimer.singleShot(2000, lambda: self._refresh_local_install())
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
            set_f0_combo(self.fc, str(s["f0method"]))
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
        l.addRow("说话人编号:", self.si)

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

        self.fmc = create_styled_combo()
        fill_f0_combo(self.fmc, getattr(s, "f0method", "rmvpe") or "rmvpe")
        l.addRow("音高算法:", self.fmc)

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
        pipe = getattr(engine, "pipeline", None)
        if pipe is None or not getattr(pipe, "is_connected", lambda: False)():
            QMessageBox.warning(self, "提示",
                "请先点「连接」，连上服务器后再获取模型列表")
            return
        self.setCursor(Qt.WaitCursor)

        class _ListThread(QThread):
            got = Signal(list)

            def run(self_t):
                try:
                    self_t.got.emit(list(pipe.list_models() or []))
                except Exception:
                    self_t.got.emit([])

        def _got(models):
            self.unsetCursor()
            if not models:
                QMessageBox.warning(self, "提示",
                    "无法获取服务器模型列表" + NL + "请确认已连接服务器")
                return
            name, ok = QInputDialog.getItem(
                self, "服务器模型", "选择模型文件:", models, 0, False)
            if ok and name:
                self.me.setText(name)

        t = _ListThread(self)
        t.got.connect(_got)
        self._list_models_thread = t
        t.start()

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
            formant=self.fs.value(), f0method=f0_from_combo(self.fmc),
            I_noise_reduce=self.inc2.isChecked(),
            O_noise_reduce=self.onc2.isChecked())
        self.accept()

# ==============================================================================
# 入口
# ==============================================================================
if __name__ == "__main__":
    setup_logging()
    sys.excepthook = _excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_QSS)
    w = MainWindow(); w.show()
    sys.exit(app.exec())
