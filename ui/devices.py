"""音频设备下拉框、虚拟声卡向导。"""
from PySide6.QtCore import Qt, Signal, QSize, QThread
from PySide6.QtWidgets import (
    QStyledItemDelegate, QComboBox, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDialogButtonBox, QListView,
)
from PySide6.QtGui import QColor, QStandardItemModel, QStandardItem
import sounddevice as sd

from tools.virtual_cable import (
    find_virtual_devices, INSTALL_URLS, open_install_page, route_self_check,
)
from ui.common import _hostapi_zh, _fp_from_devs

class DeviceItemDelegate(QStyledItemDelegate):
    """自定义绘制: 分组标题 / 设备项(两行)"""

    def __init__(self, direction="input", parent=None):
        super().__init__(parent)
        self.direction = direction
        self.icon = "🎤 " if direction == "input" else "🔊 "

    def paint(self, painter, option, index):
        is_group = index.data(Qt.UserRole + 1) == "group"

        from ui.theme import device_row_colors
        pal = device_row_colors()
        if is_group:
            painter.save()
            painter.fillRect(option.rect, QColor(pal["group_bg"]))
            painter.setPen(QColor(pal["group_fg"]))
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
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, QColor(pal["selected"]))
        elif option.state & QStyle.State_MouseOver:
            painter.fillRect(option.rect, QColor(pal["hover"]))

        painter.setPen(QColor(pal["title"]))
        f = painter.font(); f.setPointSize(10); f.setBold(False)
        painter.setFont(f)
        text = self.icon + name
        if is_default:
            text += "  ⭐"
        painter.drawText(option.rect.x() + 8, option.rect.y() + 17, text)

        # 副行: 采样率/声道 (Line 2)
        if detail:
            painter.setPen(QColor(pal["sub"]))
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

    def __init__(self, direction="input", parent=None, empty_text="默认设备"):
        super().__init__(parent)
        self.direction = direction
        self.empty_text = empty_text or "默认设备"
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

        default_item = QStandardItem(self.empty_text)
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
        parent = self.parent()
        cache_devs = getattr(parent, "_devs_cache", None) if parent is not None else None
        cache_apis = getattr(parent, "_apis_cache", None) if parent is not None else None
        try:
            if cache_devs:
                devs, apis = cache_devs, cache_apis or []
            else:
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
            self.hint.setText("已检测到虚拟声卡。把它设为「输出」，其它软件选同一条虚拟线当麦克风。")
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


class DeviceQueryThread(QThread):
    """PortAudio 枚举/重启放到后台。sd._terminate 在 UI 上会把窗口卡死。"""
    got = Signal(object, object, str)  # devs, apis, err

    def __init__(self, reinit=False, parent=None):
        super().__init__(parent)
        self.reinit = bool(reinit)

    def run(self):
        try:
            # 禁止 _terminate：和正在跑的声卡流抢 PortAudio 会把进程卡死
            raw_devs = sd.query_devices()
            raw_apis = sd.query_hostapis()
            devs = [dict(d) for d in (raw_devs or [])]
            apis = [dict(a) for a in (raw_apis or [])]
            self.got.emit(devs, apis, "")
        except Exception as e:
            self.got.emit([], [], str(e))
