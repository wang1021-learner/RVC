"""主窗口。"""
import os, sys, json, time, subprocess, logging
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, Signal, QObject, QThread, QSize, QTimer
from PySide6.QtWidgets import (
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

from tools.app_paths import (
    bundled_dir, is_frozen, log_dir, package_root, user_data_dir,
)
from worker.local_server import (
    pack_mode, runtime_installed, local_infer_ready,
)
from worker.engine import (
    VCEngine, ModelLoader, ServerConnectThread, make_pipeline,
)
from tools.audio_meter import VUMeterWidget, SpectrumWidget, calc_rms_db, spec_bins
from tools.virtual_cable import (
    find_virtual_devices, is_virtual_name, is_bluetooth_name,
    INSTALL_URLS, open_install_page, route_self_check,
)
from ui.theme import LIGHT_QSS
from ui.common import (
    NL, PROJECT_ROOT, SETTINGS_FILE, PRESETS_FILE, SPEAKERS_FILE, WEIGHTS_DIR,
    STYLE_QSS, LIGHT_GRAY, LIGHT_YELLOW, LIGHT_GREEN, LIGHT_RED,
    DEFAULT_SERVER_URL, DEFAULT_PARAMS, RESTART_KEYS, BUILTIN_PRESETS,
    MSG_SERVER_PACK_NO_LOCAL, MSG_NEED_INSTALL_LOCAL,
    _friendly_error, _friendly_net_error, _wasapi_fail_reason,
    _fp_from_devs, _device_fingerprint, _normalize_ws_url, _validate_server_url,
    fill_f0_combo, f0_from_combo, set_f0_combo, F0_CHOICES,
    load_user_settings, save_user_settings, load_presets, save_presets,
    to_server_path, _speaker_file_sub, _local_model_path, _hostapi_zh,
)
from ui.devices import DeviceCombo, CableWizard, DeviceQueryThread
from ui.widgets import StyledCombo, SpeakerCardList, create_styled_combo
from ui.speakers import SpeakerConfig, SpeakerManager, SpeakerDialog

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
        self._devs_cache = []
        self._apis_cache = []
        self._dev_query_busy = False
        self._dev_query_restore = False
        self._dev_query_user = False
        self._build_ui()
        self._apply_theme()
        self._apply_saved_params()
        self._rd(restore=False, reinit=False)
        self._rl(load=self.engine.mode != "server")
        self._set_light(LIGHT_GRAY, "未加载模型")
        self._dev_timer = QTimer(self)
        self._dev_timer.setInterval(8000)
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
        if self.engine.mode == "server":
            QTimer.singleShot(300, self._auto_connect_server)

    def _on_rms_levels(self, in_db, out_db):
        self.in_meter.set_level(in_db)
        self.out_meter.set_level(out_db)

    def _on_spectrum(self, bins):
        if hasattr(self, "spectrum"):
            self.spectrum.set_bins(bins)

    def _icon_path(self, *names):
        roots = (
            bundled_dir() / "assets" / "icons",
            package_root() / "assets" / "icons",
            package_root() / "_internal" / "assets" / "icons",
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
        dlg = CableWizard(self, self.ic, self.oc)
        if dlg.exec():
            self._on_device_changed("output")
            self._persist_settings()
            self.status_bar.showMessage("虚拟声卡设置已更新", 4000)

    def _on_xrun(self, xruns):
        text = f"卡顿 {xruns}"
        if text != getattr(self, "_xrun_text", ""):
            self.xrun_label.setText(text)
            self._xrun_text = text
        name = "chipWarn" if xruns > 0 else "chipMute"
        if name != getattr(self, "_xrun_chip", ""):
            self._xrun_chip = name
            self.xrun_label.setObjectName(name)
            self.xrun_label.style().unpolish(self.xrun_label)
            self.xrun_label.style().polish(self.xrun_label)

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
        self.in_meter.setAccessibleName("输入电平")
        self.out_meter = VUMeterWidget(title="输出")
        self.out_meter.setAccessibleName("输出电平")
        top.addWidget(self.in_meter)
        top.addWidget(self.out_meter)
        top.addStretch()

        # 状态微型标签
        self.badge_box = QFrame()
        self.badge_box.setObjectName("badgeBox")
        bh = QHBoxLayout(self.badge_box); bh.setContentsMargins(8, 4, 8, 4); bh.setSpacing(6)
        self.light = QLabel()
        self.light.setFixedSize(8, 8)
        self.light.setStyleSheet(f"background:{LIGHT_GRAY};border-radius:4px;")
        self.light.setAccessibleName("运行状态指示灯")
        bh.addWidget(self.light)
        self.state_label = QLabel("未加载模型")
        self.state_label.setObjectName("stateLabel")
        bh.addWidget(self.state_label)
        top.addWidget(self.badge_box)

        self.latency_label = QLabel("推理 --ms")
        self.latency_label.setObjectName("chipOk")
        self.latency_label.setToolTip("这一块声音算了多久。超过「块大小」就会卡")
        self.latency_label.setAccessibleName("推理耗时")
        top.addWidget(self.latency_label)

        self.e2e_label = QLabel("嘴到耳 --ms")
        self.e2e_label.setObjectName("chipInfo")
        self.e2e_label.setToolTip("从说话到耳机出声的估算延迟")
        self.e2e_label.setAccessibleName("嘴到耳延迟")
        top.addWidget(self.e2e_label)

        self.xrun_label = QLabel("卡顿 0")
        self.xrun_label.setObjectName("chipMute")
        self.xrun_label.setAccessibleName("卡顿次数")
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
        g = QGroupBox("声音角色")
        l = QVBoxLayout(g)
        l.setContentsMargins(12, 16, 12, 12)
        l.setSpacing(10)

        # 角色下拉选择
        self.sc = create_styled_combo(max_visible=12)
        self.sc.setObjectName("speakerCombo")
        self.sc.setMinimumHeight(36)
        self.sc.setToolTip("选择要变成的声音")
        self.sc.setAccessibleName("角色")
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
        self.cur_name.setObjectName("roleName")
        self.cur_model = QLabel("")
        self.cur_model.setObjectName("roleMeta")
        self.cur_model.setWordWrap(True)
        self.cur_model.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cur_info = QLabel("")
        self.cur_info.setObjectName("roleInfo")
        self.cur_warn = QLabel("")
        self.cur_warn.setObjectName("roleWarn")
        self.cur_warn.setWordWrap(True)
        self.cur_warn.setVisible(False)
        cv.addWidget(self.cur_name)
        cv.addWidget(self.cur_model)
        cv.addWidget(self.cur_info)
        cv.addWidget(self.cur_warn)
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
        self.live_pitch.setToolTip("男变女大约 +12，女变男大约 -12")
        self.live_pitch.setAccessibleName("音高")
        self.live_pitch.valueChanged.connect(self._on_live_pitch)

        self.live_index = QDoubleSpinBox()
        self.live_index.setRange(0.0, 1.0)
        self.live_index.setSingleStep(0.1)
        self.live_index.setToolTip("越高越像角色，越低越像你自己。0 表示不用检索")
        self.live_index.setAccessibleName("像角色的程度")
        self.live_index.valueChanged.connect(self._on_live_index)

        self.live_formant = QDoubleSpinBox()
        self.live_formant.setRange(-12.0, 12.0)
        self.live_formant.setSingleStep(0.5)
        self.live_formant.setToolTip("声道长短：正值更亮更女声，负值更厚")
        self.live_formant.setAccessibleName("共鸣")
        self.live_formant.valueChanged.connect(self._on_live_formant)

        self.live_dry = QDoubleSpinBox()
        self.live_dry.setRange(0.0, 1.0)
        self.live_dry.setSingleStep(0.1)
        self.live_dry.setToolTip("0=只听变声，1=只听原声")
        self.live_dry.setAccessibleName("原声混合")
        self.live_dry.valueChanged.connect(self._on_live_dry)

        gl.addWidget(self._lbl("音高"), 0, 0)
        gl.addWidget(self.live_pitch, 0, 1)
        gl.addWidget(self._lbl("像角色"), 1, 0)
        gl.addWidget(self.live_index, 1, 1)
        gl.addWidget(self._lbl("共鸣"), 2, 0)
        gl.addWidget(self.live_formant, 2, 1)
        gl.addWidget(self._lbl("原声混合"), 3, 0)
        gl.addWidget(self.live_dry, 3, 1)

        self.bypass = QCheckBox("旁通（听原声）")
        self.bypass.setToolTip("临时输出原声，不关变声")
        self.bypass.setAccessibleName("旁通听原声")
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
        sl.addWidget(self.server_edit, 1)
        sl.addWidget(self.conn_btn)
        self.server_status = QLabel("未连接")
        self.server_status.setWordWrap(False)
        self.server_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
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
        self.ic.setToolTip("麦克风。尽量和输出选同一类接口（都选系统低延迟）")
        self.ic.setAccessibleName("输入设备")
        self.ic.currentIndexChanged.connect(lambda: self._on_device_changed("input"))
        dl.addWidget(self.ic, 0, 1)

        dl.addWidget(self._lbl("输出设备"), 1, 0)
        self.oc = DeviceCombo(direction="output")
        self.oc.setMinimumHeight(30)
        self.oc.setToolTip("耳机听自己，或选虚拟声卡给游戏/会议软件")
        self.oc.setAccessibleName("输出设备")
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

        # 延迟/音质：单独一行，避免和两端长文案挤在一起
        tq_head = QHBoxLayout()
        tq_head.setSpacing(8)
        tq_head.addWidget(self._lbl("延迟平衡"))
        tq_head.addStretch(1)
        self.tq_value = QLabel("均衡")
        self.tq_value.setObjectName("fieldLabel")
        tq_head.addWidget(self.tq_value)
        l.addLayout(tq_head)

        tq = QHBoxLayout()
        tq.setSpacing(8)
        tq.setContentsMargins(0, 0, 0, 0)
        self.tq_fast = QLabel("快")
        self.tq_fast.setObjectName("fieldLabel")
        self.tq_fast.setFixedWidth(18)
        tq.addWidget(self.tq_fast)
        self.tq_slider = QSlider(Qt.Horizontal)
        self.tq_slider.setRange(0, 100)
        self.tq_slider.setValue(40)
        self.tq_slider.setFixedHeight(22)
        self.tq_slider.setToolTip("偏左更低延迟，偏右更稳。下次启动生效。")
        self.tq_slider.setAccessibleName("延迟平衡")
        self.tq_slider.valueChanged.connect(self._on_tradeoff)
        tq.addWidget(self.tq_slider, 1)
        self.tq_hq = QLabel("稳")
        self.tq_hq.setObjectName("fieldLabel")
        self.tq_hq.setFixedWidth(18)
        tq.addWidget(self.tq_hq)
        l.addLayout(tq)

        # 输出频谱
        self.spectrum = SpectrumWidget()
        self.spectrum.setToolTip("输出频谱实时监视")
        l.addWidget(self.spectrum)

        # 主按钮紧跟频谱，不再被空白顶到底
        self.sb = QPushButton("开始变声")
        self.sb.setObjectName("btnStart")
        self.sb.setProperty("state", "off")
        self.sb.setMinimumHeight(40)
        self.sb.setMaximumHeight(44)
        self.sb.setToolTip("开始或停止实时变声")
        self.sb.setAccessibleName("开始或停止变声")
        self.sb.setDefault(True)
        self.sb.clicked.connect(self._tg)
        l.addWidget(self.sb)
        l.addStretch(1)

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
        l1.setContentsMargins(10, 12, 10, 10)
        l1.setSpacing(12)

        g1 = QGroupBox("参数")
        l = QGridLayout(g1)
        l.setContentsMargins(10, 14, 10, 10)
        l.setHorizontalSpacing(10)
        l.setVerticalSpacing(6)

        self.fc = create_styled_combo(max_visible=10)
        fill_f0_combo(self.fc, DEFAULT_PARAMS["f0method"])
        self.fc.setMinimumHeight(28)
        self.fc.currentIndexChanged.connect(
            lambda: setattr(self.engine, "f0method", f0_from_combo(self.fc)))
        self.fc.setToolTip("音高提取：准确最稳，最快负担最小")
        self.fc.setAccessibleName("音高算法")

        self.bs = QDoubleSpinBox()
        self.bs.setRange(0.03, 0.5)
        self.bs.setSingleStep(0.01)
        self.bs.setValue(DEFAULT_PARAMS["block_time"])
        self.bs.setDecimals(3)
        self.bs.setSuffix(" 秒")
        self.bs.setMinimumHeight(28)
        self.bs.setToolTip("越小反应越快，也越容易卡。给别人听时启动会自动不低于 80ms。")
        self.bs.setAccessibleName("反应速度")
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
            ("反应速度", self.bs),
            ("衔接平滑", self.xs),
            ("音色稳定", self.es),
            ("静音阈值", self.ts_box),
            ("音量贴合", self.rs),
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
        self.cap_ns_cb = QCheckBox("采集降噪")
        self.cap_ns_cb.setMinimumHeight(26)
        self.cap_ns_cb.setToolTip("送出麦克风前压键盘/风扇底噪。给别人听或用蓝牙麦时会自动关掉，避免叠降噪切字、炸麦。")
        self.cap_ns_cb.setChecked(True)
        self.cap_ns_cb.toggled.connect(lambda v: setattr(self.engine, "capture_denoise", v))
        self.inc_hubert_cb = QCheckBox("短窗特征")
        self.inc_hubert_cb.setMinimumHeight(26)
        self.inc_hubert_cb.setToolTip("只算最近一小段特征，更快、更利多人。关掉更稳、更像。")
        self.inc_hubert_cb.setChecked(False)
        self.inc_hubert_cb.toggled.connect(lambda v: setattr(self.engine, "incremental_hubert", v))

        nr = QHBoxLayout()
        nr.setSpacing(10)
        nr.addWidget(self.inc)
        nr.addWidget(self.onc)
        nr.addWidget(self.cap_ns_cb)
        nr.addWidget(self.inc_hubert_cb)
        nr.addStretch()
        l.addLayout(nr, len(rows), 0, 1, 2)
        l.setColumnStretch(1, 1)
        l1.addWidget(g1)

        g3 = QGroupBox("服务器")
        g3.setToolTip("连接局域网/远程推理服务器")
        sg = QVBoxLayout(g3)
        sg.setContentsMargins(10, 14, 10, 10)
        sg.setSpacing(8)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)
        mode_row.addWidget(self.mode_local)
        mode_row.addWidget(self.mode_server)
        mode_row.addStretch(1)
        sg.addLayout(mode_row)
        sg.addWidget(self.server_row)
        sg.addWidget(self.server_status)
        self.server_box = g3
        l1.addWidget(g3)
        l1.addStretch(1)

        # ── Tab 2: 音质增强 ──
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        l2.setContentsMargins(10, 14, 10, 10)
        l2.setSpacing(10)

        g_dsp = QGroupBox("声音修饰")
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
        self.limiter_th.setValue(-3.0)
        self.limiter_th.setFixedWidth(82)
        self.limiter_th.setMinimumHeight(26)
        self.limiter_th.setToolTip("起限阈值，-3 dB 为推荐值，避免硬夹平顶")
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

        self.protect_spin = QDoubleSpinBox()
        self.protect_spin.setRange(0.0, 0.5)
        self.protect_spin.setSingleStep(0.05)
        self.protect_spin.setValue(DEFAULT_PARAMS.get("protect", 0.33))
        self.protect_spin.setMinimumHeight(28)
        self.protect_spin.setToolTip("清辅音保护。越小越留字头/气音，0.33 为原版默认，0.5 关闭。")
        self.protect_spin.valueChanged.connect(lambda v: setattr(self.engine, "protect", v))
        dl.addWidget(self._lbl("辅音保护"), 3, 0)
        dl.addWidget(self.protect_spin, 3, 1)

        # 去齿音
        self.deess_cb = QCheckBox("自适应去齿音")
        self.deess_cb.setMinimumHeight(26)
        self.deess_cb.setToolTip("尖刺超标时软衰减。与「齿音保留」互斥。")
        self.deess_cb.setChecked(DEFAULT_PARAMS["deesser_enable"])
        self.deess_cb.toggled.connect(self._on_deesser)
        dl.addWidget(self.deess_cb, 4, 0, 1, 2)

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
        dl.addLayout(vr, 5, 0, 1, 2)

        # 阶段耗时
        self.st_lbl = QLabel("阶段耗时: --")
        self.st_lbl.setStyleSheet("font-size:11px;color:#6b7c8a;")
        dl.addWidget(self.st_lbl, 6, 0, 1, 2)
        dl.setColumnStretch(1, 1)
        l2.addWidget(g_dsp)
        l2.addStretch(1)

        # ── Tab 3: 预设与监听 ──
        t3 = QWidget()
        l3 = QVBoxLayout(t3)
        l3.setContentsMargins(10, 14, 10, 10)
        l3.setSpacing(10)

        g2 = QGroupBox("场景与监听")
        m = QGridLayout(g2)
        m.setContentsMargins(10, 16, 10, 10)
        m.setHorizontalSpacing(10)
        m.setVerticalSpacing(8)

        self.agc_cb = QCheckBox("输入自动增益")
        self.agc_cb.setMinimumHeight(26)
        self.agc_cb.setToolTip("输入音量自动拉齐。给别人听或用蓝牙麦时会自动关掉，避免和其它软件叠增益炸麦。")
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

        self.monitor_cb = QCheckBox("耳机监听")
        self.monitor_cb.setMinimumHeight(26)
        self.monitor_cb.setToolTip("主输出给别人听时，用第二路耳机听自己")
        self.monitor_cb.setAccessibleName("耳机监听")
        self.monitor_cb.toggled.connect(self._on_monitor_toggle)
        m.addWidget(self.monitor_cb, 2, 0, 1, 2)

        vol_row = QHBoxLayout()
        vol_row.setSpacing(8)
        vol_row.addWidget(self._lbl("音量"))
        self.monitor_vol = QSlider(Qt.Horizontal)
        self.monitor_vol.setRange(0, 100)
        self.monitor_vol.setValue(80)
        self.monitor_vol.setFixedHeight(22)
        self.monitor_vol.setToolTip("监听音量")
        self.monitor_vol.setAccessibleName("监听音量")
        self.monitor_vol.valueChanged.connect(self._on_monitor_vol)
        vol_row.addWidget(self.monitor_vol, 1)
        self.monitor_vol_lbl = QLabel("80%")
        self.monitor_vol_lbl.setObjectName("fieldLabel")
        self.monitor_vol_lbl.setFixedWidth(36)
        self.monitor_vol_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vol_row.addWidget(self.monitor_vol_lbl)
        m.addLayout(vol_row, 3, 0, 1, 2)

        m.addWidget(self._lbl("听筒"), 4, 0)
        self.mc = DeviceCombo(direction="output", empty_text="选择耳机")
        self.mc.setMinimumHeight(30)
        self.mc.setToolTip("监听耳机，可不同于主输出")
        self.mc.setAccessibleName("监听耳机")
        self.mc.currentIndexChanged.connect(lambda: self._on_monitor_changed())
        m.addWidget(self.mc, 4, 1)
        m.setColumnStretch(1, 1)
        self._sync_monitor_enabled()
        l3.addWidget(g2)
        l3.addStretch(1)

        self._sync_mode_ui()

        tabs.addTab(t1, "延迟与音质")
        tabs.addTab(t2, "声音修饰")
        tabs.addTab(t3, "场景与监听")
        tabs.setTabToolTip(0, "块大小、音高算法、远程服务器")
        tabs.setTabToolTip(1, "防爆音、齿音、人声识别")
        tabs.setTabToolTip(2, "场景预设、耳机监听")
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

    def _refresh_route_hint(self):
        """蓝牙麦 / 虚拟声卡路由提示。输入输出不在同一驱动组时不覆盖对齐提示。"""
        if not hasattr(self, "ic") or not hasattr(self, "oc"):
            return
        in_api = self.ic.currentDeviceApi()
        out_api = self.oc.currentDeviceApi()
        if in_api is not None and out_api is not None and in_api != out_api:
            return
        in_name = self.ic.currentDeviceName() or ""
        out_name = self.oc.currentDeviceName() or ""
        if is_bluetooth_name(in_name):
            self._show_dev_hint(
                "当前输入是蓝牙耳机麦，延迟大、容易卡顿炸麦。"
                "请改用电脑自带麦或有线耳机麦。"
            )
            return
        if is_virtual_name(out_name):
            self._show_dev_hint(
                "输出已走虚拟声卡。请把要接收变声的软件，麦克风选成这条虚拟线的录制端"
                "（如 CABLE Output）。给别人听时块长会自动不低于 80ms。",
                warn=False,
            )
            return
        self._clear_dev_hint()

    def _on_device_changed(self, which="input"):
        if self._device_guard:
            return
        self._align_device_apis(which)
        if not self.engine.running:
            self._persist_settings()
            self._refresh_route_hint()
            return
        in_id, out_id, err = self._resolve_selected(reinit=False)
        if err:
            self._show_dev_hint(err)
            self.status_bar.showMessage(err, 8000)
            return
        self._refresh_route_hint()
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
        # reinit 只走刷新线程，UI 上绝不 sd._terminate
        devs = getattr(self, "_devs_cache", None) or []
        apis = getattr(self, "_apis_cache", None) or []
        if not devs:
            return None, None, "正在读取音频设备，请稍候再启动"

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
        self._rd(restore=True, reinit=not getattr(self.engine, "running", False), user=True)

    def _rd(self, restore=True, reinit=True, user=False):
        if getattr(self, "_dev_query_busy", False):
            if user:
                self.status_bar.showMessage("正在刷新音频设备…", 3000)
            return
        if getattr(self.engine, "running", False) or self.engine._engine_busy():
            reinit = False
        self._dev_query_busy = True
        self._dev_query_restore = restore
        self._dev_query_user = user
        t = DeviceQueryThread(reinit=reinit, parent=self)
        t.got.connect(self._on_devices_queried, Qt.QueuedConnection)
        t.finished.connect(t.deleteLater)
        self._dev_query_thread = t
        t.start()

    def _on_devices_queried(self, devs, apis, err):
        self._dev_query_busy = False
        btn = getattr(self, "dev_refresh_btn", None)
        user = getattr(self, "_dev_query_user", False)
        self._dev_query_user = False
        if err:
            self.status_bar.showMessage("刷新设备失败: " + err, 6000)
            if btn is not None:
                btn.setEnabled(True)
                btn.setText("刷新设备")
            return
        fp = _fp_from_devs(devs)
        same = bool(fp) and fp == getattr(self, "_dev_fp", ())
        self._devs_cache = devs
        self._apis_cache = apis
        if same and not user and self.ic.count() > 1:
            if btn is not None and not btn.isEnabled():
                btn.setEnabled(True)
                btn.setText("刷新设备")
            return
        in_name, in_api = self.ic.currentDeviceName(), self.ic.currentDeviceApi()
        out_name, out_api = self.oc.currentDeviceName(), self.oc.currentDeviceApi()
        mon_name, mon_api = self.mc.currentDeviceName(), self.mc.currentDeviceApi()
        self.ic.blockSignals(True)
        self.oc.blockSignals(True)
        self.mc.blockSignals(True)
        try:
            self.ic.populate(devs, apis, "max_input_channels")
            self.oc.populate(devs, apis, "max_output_channels")
            self.mc.populate(devs, apis, "max_output_channels")
            if getattr(self, "_dev_query_restore", True):
                self.ic.selectByNameApi(in_name, in_api)
                self.oc.selectByNameApi(out_name, out_api)
                self.mc.selectByNameApi(mon_name, mon_api)
            else:
                self._restore_devices()
            self._dev_fp = fp
        except Exception as e:
            print("Refresh devices error:", e)
            self.status_bar.showMessage("刷新设备失败: " + str(e), 6000)
        finally:
            self.ic.blockSignals(False)
            self.oc.blockSignals(False)
            self.mc.blockSignals(False)
        if getattr(self, "_dev_query_restore", True):
            self._apply_monitor()
        if user:
            self.status_bar.showMessage("设备列表已更新", 4000)
            if btn is not None:
                btn.setEnabled(True)
                btn.setText("已刷新")
                QTimer.singleShot(
                    900,
                    lambda: btn.setText("刷新设备") if btn.text() == "已刷新" else None,
                )
        elif btn is not None and not btn.isEnabled():
            btn.setEnabled(True)
            btn.setText("刷新设备")
        if self.engine.running:
            _in, _out, lost = self._resolve_selected(reinit=False)
            if lost and "正在读取" not in lost:
                self.engine.request_hard_stop()
                self._show_dev_hint("音频设备已断开: " + lost)
                self.status_bar.showMessage("音频设备已断开: " + lost, 8000)
                return
        self._refresh_route_hint()

    def _poll_devices(self):
        if getattr(self, "_dev_query_busy", False):
            return
        # 变声时不要从另一线程碰 PortAudio，Windows 上会把声卡回调卡死
        if getattr(self.engine, "running", False) or self.engine._engine_busy():
            return
        self._rd(restore=True, reinit=False, user=False)

    def _preferred_speaker_index(self):
        saved = self._settings.get("speaker") or ""
        names = [s.name for s in self.speaker_mgr.speakers]
        if saved in names:
            return names.index(saved)
        alias = saved[:-1] + "轮" if saved.endswith("e") else ""
        if alias in names:
            return names.index(alias)
        return 0

    def _rl(self, load=True):
        self.sc.blockSignals(True)
        self.sc.clear()
        local = self.engine.mode == "local"
        for s in self.speaker_mgr.speakers:
            self.sc.addItem(s.name)
            row = self.sc.count() - 1
            missing = local and not _local_model_path(s.model_path).is_file()
            sub = _speaker_file_sub(s)
            if missing:
                sub = "本机没有这个模型  ·  " + sub
            self.sc.setItemData(row, sub, Qt.UserRole + 2)
            self.sc.setItemData(row, missing, Qt.UserRole + 4)
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
            if load:
                self._sel(idx)
            else:
                s = self.speaker_mgr.speakers[idx]
                self.cur_name.setText(s.name)
                self.cur_model.setText(
                    "模型: " + (Path(s.model_path).name if s.model_path else "无")
                    + NL
                    + "索引: " + (Path(s.index_path).name if s.index_path else "无")
                )
                self.cur_info.setText(f"音高 {s.pitch:+d}  像角色 {s.index_rate:.1f}")
                self._sync_live_sliders(s)
                self._set_light(LIGHT_YELLOW, "正在连接服务器…")
                self.sb.setEnabled(False)
        else:
            self.cur_name.setText("还没有角色")
            self.cur_model.setText("把 .pth 拖进窗口，或点「添加」。")
            self.cur_info.setText("")
            if hasattr(self, "cur_warn"):
                self.cur_warn.setVisible(False)
            self.sb.setEnabled(False)
            self._set_light(LIGHT_GRAY, "待添加角色")

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
            f"音高 {s.pitch:+d}  像角色 {s.index_rate:.1f}  共鸣 {s.formant:+.1f}"
            if getattr(s, "formant", 0.0) != 0.0
            else f"音高 {s.pitch:+d}  像角色 {s.index_rate:.1f}")
        self._sync_live_sliders(s)
        if self.engine.mode == "local" and not _local_model_path(s.model_path).is_file():
            msg = (
                "本机找不到「%s」。把 .pth 放到 assets/weights/，或改用远程服务器。"
                % (Path(s.model_path).name or "模型")
            )
            if hasattr(self, "cur_warn"):
                self.cur_warn.setText(msg)
                self.cur_warn.setVisible(True)
            self._set_light(LIGHT_RED, "模型文件缺失")
            self.sb.setEnabled(False)
            self._show_dev_hint(msg)
            return
        if hasattr(self, "cur_warn"):
            self.cur_warn.setVisible(False)
        if self.engine.current_speaker is s:
            self._set_light(LIGHT_GREEN, "就绪")
            self.sb.setEnabled(True)
            self._show_loaded_paths(s)
            return
        if self.engine.running or self.engine._engine_busy():
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
            old = getattr(self._loader, "speaker", None)
            if (
                old is not None
                and speaker is not None
                and (old.name, Path(str(old.model_path or "")).name)
                == (speaker.name, Path(str(speaker.model_path or "")).name)
            ):
                return
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
        self._set_light(LIGHT_YELLOW, f"加载中… {stage}（{elapsed:.0f} 秒）")

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
        self._refresh_server_loaded_status()
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
        hint = "请核对本机模型/索引文件是否存在。"
        if self.engine.mode == "server":
            connected = False
            try:
                connected = bool(self.engine.pipeline.is_connected())
            except Exception:
                connected = False
            low = (text + " " + str(err or "")).lower()
            closed = (
                (not connected)
                or "连接已关闭" in text
                or "连接已断开" in text
                or "未连接" in text
                or "closed" in low
            )
            if closed:
                hint = "连接已经断了。点一次「连接」，成功后再选角色。"
            elif "找不到" in text or "no such file" in low or "not found" in low:
                hint = "远端没有这个模型。把同名 .pth 放到服务器的 assets/weights/ 后再加载。"
            elif "没回上" in text or "没有返回有效" in str(err or "") or "expecting" in low:
                hint = "连接还在，只是这一次没拿到加载回复。再点一次角色；反复出现就重新「连接」。"
            else:
                hint = "服务器已连上。若文件名对不上，请把 .pth 放到远端 assets/weights/。"
            try:
                if not connected:
                    self._set_server_status("未连接", "fail")
            except Exception:
                pass
        self.status_bar.showMessage("加载失败: " + text, 8000)
        self._show_dev_hint("加载失败: " + text[:160])
        self._alert("模型加载失败", text + NL + NL + hint)

    def _on_start_failed(self, msg):
        """启动变声失败：常驻提示 + 弹窗。"""
        text = _friendly_error(msg)
        self.sb.setEnabled(True)
        self.sb.setText("开始变声")
        self.sb.setProperty("state", "off")
        self.sb.style().unpolish(self.sb); self.sb.style().polish(self.sb)
        self._set_light(LIGHT_RED, "启动失败")
        self._show_dev_hint("启动失败: " + text[:200])
        self.status_bar.showMessage("启动失败: " + text, 8000)
        if self.engine.mode == "server":
            try:
                if not self.engine.pipeline.is_connected():
                    self._set_server_status("未连接", "fail")
            except Exception:
                pass
        self._alert("启动失败", text)

    def _on_recover_progress(self, n, total):
        self._set_light(LIGHT_YELLOW, "正在重连 %d/%d" % (n, total))
        if self.engine.mode == "server":
            self._set_server_status("连接中断，正在重连 %d/%d…" % (n, total), "busy")

    def _on_recover_ok(self):
        self._set_light(LIGHT_GREEN, "运行中")
        self._refresh_route_hint()
        if self.engine.mode == "server":
            self._set_server_status("已重新连上服务器", "ok")
        self.status_bar.showMessage("已重新连上服务器，变声继续", 5000)

    def _on_recover_failed(self, err):
        text = _friendly_error(err)
        self._set_light(LIGHT_RED, "连接中断")
        self.sb.setEnabled(True)
        self.sb.setText("开始变声")
        self.sb.setProperty("state", "off")
        self.sb.style().unpolish(self.sb); self.sb.style().polish(self.sb)
        self._show_dev_hint("服务器已断开，变声已停止。请重新点「连接」。")
        if self.engine.mode == "server":
            self._set_server_status("已断开: " + text, "fail")
        self.status_bar.showMessage("重连失败，变声已停止", 8000)
        self._alert(
            "服务器中断",
            text + NL + NL +
            "变声已停止。请确认推理服务仍在运行，点「连接」成功后再启动变声。")

    def _set_light(self, color, text):
        self.light.setStyleSheet(f"background:{color};border-radius:4px;")
        self.state_label.setText(text)
        if color == LIGHT_GREEN:
            c = "#0f766e"
        elif color == LIGHT_YELLOW:
            c = "#b45309"
        elif color == LIGHT_RED:
            c = "#b91c1c"
        else:
            c = "#475569"
        self.state_label.setStyleSheet(f"font-size:12px;font-weight:700;color:{c};")

    def _on_started(self):
        self._set_light(LIGHT_GREEN, "运行中")
        self.sb.setEnabled(True)
        self.sb.setText("停止变声")
        self.sb.setProperty("state", "on")
        self.sb.setAccessibleName("停止变声")
        self.sb.style().unpolish(self.sb); self.sb.style().polish(self.sb)
        bt = float(getattr(self.engine, "block_time", 0) or 0)
        if hasattr(self, "bs") and abs(float(self.bs.value()) - bt) > 0.0005 and bt > 0:
            self.bs.blockSignals(True)
            try:
                self.bs.setValue(bt)
            finally:
                self.bs.blockSignals(False)
            self._sync_tradeoff_slider()
        if getattr(self.engine, "_agc_held_off", False) and hasattr(self, "agc_cb"):
            self.agc_cb.blockSignals(True)
            self.agc_cb.setChecked(False)
            self.agc_cb.blockSignals(False)
        if getattr(self.engine, "_ns_held_off", False) and hasattr(self, "cap_ns_cb"):
            self.cap_ns_cb.blockSignals(True)
            self.cap_ns_cb.setChecked(False)
            self.cap_ns_cb.blockSignals(False)
        if getattr(self.engine, "_deess_held_off", False) and hasattr(self, "deess_cb"):
            self.deess_cb.blockSignals(True)
            self.deess_cb.setChecked(False)
            self.deess_cb.blockSignals(False)
        if getattr(self.engine, "_vad_held_off", False) and hasattr(self, "vad_cb"):
            self.vad_cb.blockSignals(True)
            self.vad_cb.setChecked(False)
            self.vad_cb.blockSignals(False)
        if hasattr(self, "inc_hubert_cb"):
            self.inc_hubert_cb.blockSignals(True)
            self.inc_hubert_cb.setChecked(False)
            self.inc_hubert_cb.blockSignals(False)
        if hasattr(self, "hf_spin"):
            hf = float(getattr(self.engine, "hf_mix_rate", 0) or 0)
            if abs(float(self.hf_spin.value()) - hf) > 0.001:
                self.hf_spin.blockSignals(True)
                self.hf_spin.setValue(hf)
                self.hf_spin.blockSignals(False)
        if hasattr(self, "pres_spin"):
            pr = float(getattr(self.engine, "presence", 0) or 0)
            if abs(float(self.pres_spin.value()) - pr) > 0.001:
                self.pres_spin.blockSignals(True)
                self.pres_spin.setValue(pr)
                self.pres_spin.blockSignals(False)
        if getattr(self.engine, "_index_bumped", False) and hasattr(self, "live_index"):
            self.live_index.blockSignals(True)
            self.live_index.setValue(0.5)
            self.live_index.blockSignals(False)
        self._refresh_route_hint()
        self._persist_settings(
            save_speakers=bool(getattr(self.engine, "_index_bumped", False)))

    def _on_stopped(self):
        self._set_light(LIGHT_GRAY, "已停止")
        self.sb.setEnabled(True)
        self.sb.setText("开始变声")
        self.sb.setProperty("state", "off")
        self.sb.setAccessibleName("开始或停止变声")
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
        hold = 8000 if any(k in str(m) for k in ("失败", "中断", "错误", "断开", "蓝牙", "80ms", "自动增益")) else 5000
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
            band, name = 0, "chipOk"
        elif ms < budget:
            band, name = 1, "chipMute"
        else:
            band, name = 2, "chipWarn"
        text = f"推理 {ms}ms"
        if text != getattr(self, "_lat_text", ""):
            self.latency_label.setText(text)
            self._lat_text = text
        if band != getattr(self, "_lat_band", None):
            self._lat_band = band
            self.latency_label.setObjectName(name)
            self.latency_label.style().unpolish(self.latency_label)
            self.latency_label.style().polish(self.latency_label)

    def _tg(self):
        if self.engine.running:
            self.engine.stop()
            return
        if self.engine._engine_busy():
            self.engine.request_hard_stop()
            self.sb.setEnabled(True)
            self.sb.setText("正在取消…")
            self.status_bar.showMessage("正在取消…", 3000)
            return
        if not self._guard_local_infer():
            return
        if not self.engine.pipeline.is_loaded:
            if self.engine.mode == "server":
                QMessageBox.warning(
                    self, "提示",
                    "角色还没加载完。请等状态变成「就绪」，或先点「连接」。")
            else:
                QMessageBox.warning(self, "提示", "请先选择角色并等待加载完成")
            return
        in_id, out_id, err = self._resolve_selected(reinit=False)
        if err:
            self._show_dev_hint(err)
            QMessageBox.warning(self, "设备", err)
            return
        self.engine.input_device = in_id
        self.engine.output_device = out_id
        # 启动全程异步；按钮保持可点，用来取消
        self.sb.setEnabled(True)
        self.sb.setText("取消启动")
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
        self._persist_settings(immediate=True)
        self.engine._live_stop = True
        self.engine._live_event.set()
        if hasattr(self, "_loader") and self._loader is not None and self._loader.isRunning():
            try:
                self._loader.quit()
            except Exception:
                pass
        self.engine._stop_requested = True
        pipe = getattr(self.engine, "pipeline", None)
        try:
            if pipe is not None and getattr(pipe, "abort", None) is not None:
                pipe.abort()
        except Exception:
            pass
        self.hide()
        if getattr(self, "tray", None) is not None:
            try:
                self.tray.hide()
            except Exception:
                pass
        self.engine.request_hard_stop()
        self.engine.wait_idle(400)
        if pipe is not None and getattr(pipe, "stop_server", None) is not None:
            try:
                pipe.stop_server()
            except Exception:
                pass
        e.accept()

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
            self.cur_info.setText(f"音高 {s.pitch:+d}  像角色 {s.index_rate:.1f}")
        self._persist_settings(save_speakers=s is not None)

    def _on_live_index(self, val):
        if self._live_guard:
            return
        self.engine.change_index_rate(val)
        s = self.engine.current_speaker
        if s is not None:
            s.index_rate = float(val)
            self.cur_info.setText(f"音高 {s.pitch:+d}  像角色 {s.index_rate:.1f}")
        self._persist_settings(save_speakers=s is not None)

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
        self._update_tradeoff_label(self.tq_slider.value())
        self._persist_settings()

    def _update_tradeoff_label(self, val):
        if not hasattr(self, "tq_value"):
            return
        if val <= 25:
            text = "更低延迟"
        elif val <= 55:
            text = "均衡"
        else:
            text = "更稳音色"
        self.tq_value.setText(text)

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
        self._update_tradeoff_label(self.tq_slider.value())

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
        self._sync_monitor_enabled()
        self._apply_monitor()

    def _sync_monitor_enabled(self):
        on = bool(self.monitor_cb.isChecked())
        self.monitor_vol.setEnabled(on)
        self.mc.setEnabled(on)
        if hasattr(self, "monitor_vol_lbl"):
            self.monitor_vol_lbl.setEnabled(on)

    def _on_monitor_vol(self, val):
        if hasattr(self, "monitor_vol_lbl"):
            self.monitor_vol_lbl.setText("%d%%" % int(val))
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
            devs = getattr(self, "_devs_cache", None) or []
            if not devs:
                return None
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
        text = f"嘴到耳 {ms:.0f}ms"
        if text == getattr(self, "_e2e_text", ""):
            return
        self._e2e_text = text
        self.e2e_label.setText(text)

    def _on_stage_stats(self, s):
        try:
            text = (
                "阶段耗时: 特征 %.1f · 检索 %.1f · 音高 %.1f · 模型 %.1f ms"
                % (s.get("feature", 0.0), s.get("index", 0.0),
                   s.get("pitch", 0.0), s.get("model", 0.0)))
            if text == getattr(self, "_st_text", ""):
                return
            self._st_text = text
            self.st_lbl.setText(text)
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
        if "protect" in params and hasattr(self, "protect_spin"):
            self.protect_spin.setValue(float(params["protect"]))
        if "vad_enable" in params and hasattr(self, "vad_cb"):
            self.vad_cb.setChecked(bool(params["vad_enable"]))
        if "incremental_hubert" in params and hasattr(self, "inc_hubert_cb"):
            self.inc_hubert_cb.setChecked(bool(params["incremental_hubert"]))
        if "capture_denoise" in params and hasattr(self, "cap_ns_cb"):
            self.cap_ns_cb.setChecked(bool(params["capture_denoise"]))
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
            "protect": float(self.protect_spin.value()) if hasattr(self, "protect_spin") else 0.33,
            "deesser_enable": bool(self.deess_cb.isChecked()) if hasattr(self, "deess_cb") else False,
            "vad_enable": bool(self.vad_cb.isChecked()) if hasattr(self, "vad_cb") else False,
            "incremental_hubert": bool(self.inc_hubert_cb.isChecked()) if hasattr(self, "inc_hubert_cb") else True,
            "capture_denoise": bool(self.cap_ns_cb.isChecked()) if hasattr(self, "cap_ns_cb") else True,
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
        if self.engine.running or self.engine._engine_busy():
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
            self._rl()
        self.status_bar.showMessage(
            "已切换到本地推理" if mode == "local" else "已切换到服务器", 4000)

    def _adopt_server_speaker(self):
        """连接成功后：服务器已有当前角色就直接就绪，避免再加载把连接掐断。"""
        if not self.speaker_mgr.speakers:
            return
        row = self.sc.currentIndex()
        if row < 0 or row >= len(self.speaker_mgr.speakers):
            return
        s = self.speaker_mgr.speakers[row]
        pipe = self.engine.pipeline
        info = {}
        try:
            info = pipe.loaded_file_info() or {}
        except Exception:
            info = {}
        got = Path(str(info.get("model_path") or "")).name.lower()
        want = Path(str(s.model_path or "")).name.lower()
        loaded = bool(getattr(pipe, "is_loaded", False))
        if loaded and got and want and got == want:
            self.engine.current_speaker = s
            self._sync_live_sliders(s)
            self._set_light(LIGHT_GREEN, "就绪")
            self.sb.setEnabled(True)
            self._show_loaded_paths(s)
            self._refresh_server_loaded_status()
            return
        self.engine.current_speaker = None
        self._sel(row)

    def _refresh_server_loaded_status(self):
        if self.engine.mode != "server":
            return
        pipe = self.engine.pipeline
        try:
            if not pipe.is_connected():
                return
        except Exception:
            return
        gpu = getattr(pipe, "gpu_name", "") or ""
        info = self._pipeline_file_info()
        model = Path(str(info.get("model_path") or "")).name
        idx = Path(str(info.get("index_path") or "")).name
        bits = [x for x in (gpu, model or "等待加载角色") if x]
        if model:
            bits.append("无索引" if not idx else (
                idx if info.get("index_loaded", True) else idx + "（未启用）"))
        self._set_server_status("已连接 · " + " · ".join(bits), "ok")

    def _sync_mode_ui(self):
        server = self.engine.mode == "server"
        if hasattr(self, "server_row"):
            self.server_row.setVisible(server)
        if hasattr(self, "server_status"):
            self.server_status.setVisible(server)
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
        fm = self.server_status.fontMetrics()
        shown = fm.elidedText(text, Qt.ElideMiddle, max(120, self.server_status.width() - 16))
        self.server_status.setText(shown)
        self.server_status.setToolTip(text if shown != text else "")
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

    def _auto_connect_server(self):
        if self.engine.mode != "server":
            return
        try:
            if self.engine.pipeline.is_connected():
                self._adopt_server_speaker()
                return
        except Exception:
            pass
        self._connect_server(silent=True)

    def _connect_server(self, silent=False):
        self._connect_silent = bool(silent)
        url, bad = _validate_server_url(self.server_edit.text())
        self.server_edit.setText(url)
        if bad:
            self._set_server_status(bad, "fail")
            self.status_bar.showMessage(bad, 8000)
            if not silent:
                QMessageBox.warning(self, "连接服务器", bad)
            return
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
            msg = "已连接"
            if extra:
                msg += " · " + extra
            self._set_server_status(msg, "ok")
            self.status_bar.showMessage("已连接 " + url + ((" · " + extra) if extra else ""), 6000)
            self._connect_silent = False
            self._adopt_server_speaker()
        else:
            self.conn_btn.setText("连接")
            err = extra or "请确认：那台机器已启动推理服务，地址端口正确，防火墙放行 8765。"
            text = _friendly_error(err)
            self._set_server_status("连接失败: " + text, "fail")
            self.status_bar.showMessage("连接失败: " + text, 8000)
            if not getattr(self, "_connect_silent", False):
                QMessageBox.warning(
                    self, "连接服务器",
                    "无法连接 " + url + NL + NL + text + NL + NL +
                    "请确认：" + NL +
                    "1. 那台机器已运行 python server/rvc_server.py（或 start_server.bat）" + NL +
                    "2. 地址是 ws://那台机器的IP:8765" + NL +
                    "3. 防火墙 / 云安全组放行 TCP 8765")
            self._connect_silent = False

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
        if hasattr(self, "monitor_vol_lbl"):
            self.monitor_vol_lbl.setText("%d%%" % int(self.monitor_vol.value()))
        self._sync_monitor_enabled()
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
        if "protect" in s and hasattr(self, "protect_spin"):
            try:
                self.protect_spin.setValue(float(s["protect"]))
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
        if "incremental_hubert" in s and hasattr(self, "inc_hubert_cb"):
            self.inc_hubert_cb.setChecked(bool(s["incremental_hubert"]))
        if "capture_denoise" in s and hasattr(self, "cap_ns_cb"):
            self.cap_ns_cb.setChecked(bool(s["capture_denoise"]))
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

    def _persist_settings(self, immediate=False, save_speakers=False):
        if save_speakers:
            self._persist_speakers_needed = True
        if not immediate:
            t = getattr(self, "_persist_timer", None)
            if t is None:
                t = QTimer(self)
                t.setSingleShot(True)
                t.setInterval(400)
                t.timeout.connect(lambda: self._persist_settings(immediate=True))
                self._persist_timer = t
            t.start()
            return
        if getattr(self, "_persist_speakers_needed", False):
            self._persist_speakers_needed = False
            try:
                self.speaker_mgr.save()
            except Exception:
                pass
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
            "hf_mix_rate": float(self.hf_spin.value()),
            "presence": float(self.pres_spin.value()),
            "protect": float(self.protect_spin.value()) if hasattr(self, "protect_spin") else 0.33,
            "deesser_enable": bool(self.deess_cb.isChecked()),
            "vad_enable": bool(self.vad_cb.isChecked()),
            "vad_threshold": float(self.vad_th.value()),
            "incremental_hubert": bool(self.inc_hubert_cb.isChecked()) if hasattr(self, "inc_hubert_cb") else True,
            "capture_denoise": bool(self.cap_ns_cb.isChecked()) if hasattr(self, "cap_ns_cb") else True,
        })
        try:
            save_user_settings(data)
            self._settings = data
        except Exception:
            pass
