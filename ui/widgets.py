"""通用下拉框与角色卡片列表。"""
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QStyledItemDelegate, QStyle, QComboBox, QWidget, QVBoxLayout,
    QLabel, QFrame, QListView, QScrollArea,
)
from PySide6.QtGui import QColor, QFontMetrics

class ComboItemDelegate(QStyledItemDelegate):
    """下拉项：圆角行、选中青绿条；有副标题时两行（模型 / 索引）。"""
    ROW_H = 36
    ROW_H_SUB = 50

    def paint(self, painter, option, index):
        from ui.theme import combo_row_colors
        pal = combo_row_colors()
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = option.rect.adjusted(5, 2, -6, -2)
        selected = bool(option.state & QStyle.State_Selected)
        hover = bool(option.state & QStyle.State_MouseOver)
        sub = str(index.data(Qt.UserRole + 2) or "")
        warn = bool(index.data(Qt.UserRole + 4))
        if selected:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(pal["selected_bg"]))
            painter.drawRoundedRect(rect, 8, 8)
            bar = rect.adjusted(0, 8, 0, -8)
            bar.setWidth(3)
            painter.setBrush(QColor(pal["selected_bar"]))
            painter.drawRoundedRect(bar, 2, 2)
            title_color = QColor(pal["selected_title"])
            sub_color = QColor("#b45309" if warn else pal["selected_title"])
            pad = 12
        elif hover:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(pal["hover_bg"]))
            painter.drawRoundedRect(rect, 8, 8)
            title_color = QColor(pal["title"])
            sub_color = QColor("#b45309" if warn else pal["sub"])
            pad = 10
        else:
            title_color = QColor(pal["idle_title"])
            sub_color = QColor("#b45309" if warn else pal.get("idle_sub", pal["sub"]))
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
