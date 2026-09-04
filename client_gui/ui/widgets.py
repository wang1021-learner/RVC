"""Fluent 下拉框：用系统滚动条，滚轮才能正常滚。"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractScrollArea

from ui.fw import ComboBox


def _use_native_scroll(menu):
    view = menu.view
    sd = getattr(view, "scrollDelegate", None)
    if sd is not None:
        try:
            view.viewport().removeEventFilter(sd)
        except Exception:
            pass
        try:
            sd.vScrollBar.setForceHidden(True)
            sd.hScrollBar.setForceHidden(True)
        except Exception:
            pass
    QAbstractScrollArea.setVerticalScrollBarPolicy(view, Qt.ScrollBarAsNeeded)
    QAbstractScrollArea.setHorizontalScrollBarPolicy(view, Qt.ScrollBarAlwaysOff)
    view.verticalScrollBar().setStyleSheet(
        "QScrollBar:vertical{width:8px;background:transparent;margin:0;}"
        "QScrollBar::handle:vertical{background:#C8C6C4;border-radius:4px;min-height:24px;}"
        "QScrollBar::handle:vertical:hover{background:#8A8886;}"
        "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}"
    )
    return menu


class InstantComboBox(ComboBox):
    def _createComboMenu(self):
        return _use_native_scroll(super()._createComboMenu())


def create_styled_combo(min_width=0, max_visible=8):
    cb = InstantComboBox()
    if min_width > 0:
        cb.setMinimumWidth(min_width)
    cb.setMaxVisibleItems(max_visible)
    return cb
