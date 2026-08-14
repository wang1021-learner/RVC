"""
RVC Audio Level Meter & Visualizer
====================================
提供 PySide6 强实时的 VU 音量电平表与音频指示控件。
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
            self.peak_db = max(-60.0, self.peak_db - 0.8) # Decay peak
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.setBrush(QBrush(QColor("#1e272e")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, 4, 4)

        # Title
        painter.setPen(QColor("#a4b0be"))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(6, h - 7, self.title)

        # Meter bar area
        bar_left = painter.fontMetrics().horizontalAdvance(self.title) + 12
        bar_width = w - bar_left - 6
        bar_height = h - 8
        bar_top = 4

        if bar_width <= 0:
            return

        # Background bar
        painter.setBrush(QBrush(QColor("#2f3640")))
        painter.drawRoundedRect(bar_left, bar_top, bar_width, bar_height, 3, 3)

        # Calculate fill ratio (-60dB ~ 0dB)
        fill_ratio = (self.level_db + 60.0) / 60.0
        fill_ratio = max(0.0, min(1.0, fill_ratio))
        fill_w = bar_width * fill_ratio

        if fill_w > 0:
            gradient = QLinearGradient(bar_left, 0, bar_left + bar_width, 0)
            gradient.setColorAt(0.0, QColor("#2ed573"))  # Green
            gradient.setColorAt(0.7, QColor("#ffa502"))  # Yellow
            gradient.setColorAt(1.0, QColor("#ff4757"))  # Red

            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(bar_left, bar_top, fill_w, bar_height, 3, 3)

        # Peak indicator line
        peak_ratio = (self.peak_db + 60.0) / 60.0
        peak_x = bar_left + bar_width * max(0.0, min(1.0, peak_ratio))
        if peak_x > bar_left:
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(int(peak_x), bar_top, int(peak_x), bar_top + bar_height)


def calc_rms_db(audio_data):
    if audio_data is None or len(audio_data) == 0:
        return -60.0
    rms = np.sqrt(np.mean(np.square(audio_data)))
    if rms <= 1e-6:
        return -60.0
    db = 20.0 * np.log10(rms)
    return max(-60.0, min(0.0, float(db)))
