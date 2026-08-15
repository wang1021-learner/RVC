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
        self.setMinimumSize(130, 24)
        self.setMaximumHeight(28)

    def set_level(self, level_db):
        self.level_db = max(-60.0, min(0.0, float(level_db)))
        if self.level_db > self.peak_db:
            self.peak_db = self.level_db
        else:
            self.peak_db = max(-60.0, self.peak_db - 0.7)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # 极简深灰底框
        painter.setBrush(QBrush(QColor("#1f2937")))
        painter.setPen(QPen(QColor("#374151"), 1))
        painter.drawRoundedRect(0, 0, w - 1, h - 1, 5, 5)

        # 标题文本
        painter.setPen(QColor("#9ca3af"))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(8, h - 7, self.title)

        # 电平条区域
        bar_left = painter.fontMetrics().horizontalAdvance(self.title) + 14
        bar_width = w - bar_left - 8
        bar_height = h - 10
        bar_top = 5

        if bar_width <= 0:
            return

        # 槽底
        painter.setBrush(QBrush(QColor("#111827")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bar_left, bar_top, bar_width, bar_height, 2, 2)

        # 填充
        fill_ratio = (self.level_db + 60.0) / 60.0
        fill_ratio = max(0.0, min(1.0, fill_ratio))
        fill_w = bar_width * fill_ratio

        if fill_w > 0:
            gradient = QLinearGradient(bar_left, 0, bar_left + bar_width, 0)
            gradient.setColorAt(0.0, QColor("#22c55e"))  # 翠绿
            gradient.setColorAt(0.75, QColor("#eab308")) # 暖黄
            gradient.setColorAt(1.0, QColor("#ef4444"))  # 浅红

            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(bar_left, bar_top, fill_w, bar_height, 2, 2)

        # 峰值针
        peak_ratio = (self.peak_db + 60.0) / 60.0
        peak_x = bar_left + bar_width * max(0.0, min(1.0, peak_ratio))
        if peak_x > bar_left:
            painter.setPen(QPen(QColor("#f9fafb"), 1.5))
            painter.drawLine(int(peak_x), bar_top, int(peak_x), bar_top + bar_height)


def calc_rms_db(audio_data):
    if audio_data is None or len(audio_data) == 0:
        return -60.0
    rms = np.sqrt(np.mean(np.square(audio_data)))
    if rms <= 1e-6:
        return -60.0
    db = 20.0 * np.log10(rms)
    return max(-60.0, min(0.0, float(db)))
