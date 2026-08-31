"""角色配置、列表存储、添加/编辑对话框。"""
import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QHBoxLayout, QDialogButtonBox, QMessageBox,
    QInputDialog, QFileDialog, QComboBox,
)
from PySide6.QtGui import QCursor

from tools.app_paths import speakers_path, package_root
from tools.file_io import write_json_atomic
from ui.common import (
    NL, PROJECT_ROOT, WEIGHTS_DIR, fill_f0_combo, f0_from_combo,
)
from ui.widgets import create_styled_combo

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
    def __init__(self, path=None):
        self.path = Path(path or speakers_path()); self.speakers = []; self.load()
    def load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.speakers = [SpeakerConfig.from_dict(s) for s in data.get("speakers", [])]
    def save(self):
        write_json_atomic(
            self.path, {"speakers": [s.to_dict() for s in self.speakers]})
    def add(self, s): self.speakers.append(s); self.save()
    def remove(self, i):
        if 0 <= i < len(self.speakers): self.speakers.pop(i); self.save()
    def update(self, i, s):
        if 0 <= i < len(self.speakers): self.speakers[i] = s; self.save()


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
        self.irs.setToolTip("越高越像角色，越低越像你自己")
        l.addRow("像角色:", self.irs)

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
