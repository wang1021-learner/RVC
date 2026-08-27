"""
RVC Audio Level Meter & Visualizer
====================================
极简现代音频电平表与状态指示控件。
"""
import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QBrush, QPen


class VUMeterWidget(QWidget):
    def __init__(self, parent=None, title="输入"):
        super().__init__(parent)
        self.title = title
        self.level_db = -60.0
        self.peak_db = -60.0
        self._dark = False
        self.setMinimumSize(130, 26)
        self.setMaximumHeight(28)

    def set_dark(self, dark):
        self._dark = bool(dark)
        self.update()

    def set_level(self, level_db):
        self.level_db = max(-60.0, min(0.0, float(level_db)))
        if self.level_db > self.peak_db:
            self.peak_db = self.level_db
        else:
            self.peak_db = max(-60.0, self.peak_db - 0.7)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        w = self.width()
        h = self.height()

        # 底框 (纯浅色风格)
        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRoundedRect(0, 0, w - 1, h - 1, 5, 5)

        # 标题文本
        painter.setPen(QColor("#475569"))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(8, h - 8, self.title)

        # 电平条区域
        bar_left = painter.fontMetrics().horizontalAdvance(self.title) + 14
        bar_width = w - bar_left - 8
        bar_height = h - 12
        bar_top = 6

        if bar_width <= 0:
            return

        # 槽底
        painter.setBrush(QBrush(QColor("#e2e8f0")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bar_left, bar_top, bar_width, bar_height, 2, 2)

        # 填充
        fill_ratio = (self.level_db + 60.0) / 60.0
        fill_ratio = max(0.0, min(1.0, fill_ratio))
        fill_w = bar_width * fill_ratio

        if fill_w > 0:
            gradient = QLinearGradient(bar_left, 0, bar_left + bar_width, 0)
            gradient.setColorAt(0.0, QColor("#16a34a"))  # 翠绿
            gradient.setColorAt(0.75, QColor("#d97706")) # 暖黄
            gradient.setColorAt(1.0, QColor("#dc2626"))  # 鲜红

            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(bar_left, bar_top, fill_w, bar_height, 2, 2)

        # 峰值针
        peak_ratio = (self.peak_db + 60.0) / 60.0
        peak_x = bar_left + bar_width * max(0.0, min(1.0, peak_ratio))
        if peak_x > bar_left:
            painter.setPen(QPen(QColor("#0f172a"), 1.5))
            painter.drawLine(int(peak_x), bar_top, int(peak_x), bar_top + bar_height)


class SpectrumWidget(QWidget):
    """32 段对数频谱，供实时输出监视。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bins = np.zeros(32, dtype=np.float32)
        self.setMinimumHeight(56)
        self.setMaximumHeight(72)
        self._dark = False

    def set_dark(self, dark):
        self._dark = bool(dark)
        self.update()

    def set_bins(self, bins):
        arr = np.asarray(bins, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            self.bins.fill(0)
        else:
            if arr.size != self.bins.size:
                idx = np.linspace(0, arr.size - 1, self.bins.size)
                arr = np.interp(idx, np.arange(arr.size), arr).astype(np.float32)
            peak = float(arr.max()) if arr.size else 0.0
            if peak > 1e-6:
                arr = arr / peak
            self.bins = np.clip(arr, 0.0, 1.0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        bg = QColor("#f1f5f9")
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        painter.drawRoundedRect(0, 0, w - 1, h - 1, 6, 6)
        n = int(self.bins.size)
        if n <= 0 or w < 8:
            return
        gap = 2
        bar_w = max(2.0, (w - 10 - gap * (n - 1)) / n)
        x0 = 5
        for i, v in enumerate(self.bins):
            bh = max(2.0, float(v) * (h - 10))
            x = x0 + i * (bar_w + gap)
            y = h - 5 - bh
            if v > 0.75:
                c = QColor("#ef4444")
            elif v > 0.4:
                c = QColor("#0d9488")
            else:
                c = QColor("#10b981")
            painter.setBrush(QBrush(c))
            painter.drawRoundedRect(QRectF(x, y, bar_w, bh), 1.5, 1.5)


def spec_bins(audio, n_out=32):
    if audio is None:
        return np.zeros(n_out, dtype=np.float32)
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < 16:
        return np.zeros(n_out, dtype=np.float32)
    n = 1 << int(np.floor(np.log2(min(x.size, 2048))))
    n = max(n, 64)
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
    spec = np.log1p(spec)
    idx = np.linspace(0, spec.size - 1, n_out)
    return np.interp(idx, np.arange(spec.size), spec).astype(np.float32)


def calc_rms_db(audio_data):
    if audio_data is None or len(audio_data) == 0:
        return -60.0
    rms = np.sqrt(np.mean(np.square(audio_data)))
    if rms <= 1e-6:
        return -60.0
    db = 20.0 * np.log10(rms)
    return max(-60.0, min(0.0, float(db)))
