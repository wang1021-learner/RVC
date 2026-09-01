"""浅色工作台：白底、细灰线。控件外观交给 QFluentWidgets。"""

PAPER = "#FFFFFF"
SURFACE = "#FFFFFF"
INK = "#1A1A1A"
MUTED = "#605E5C"
LINE = "#E5E5E5"
LINE_STRONG = "#D1D1D1"
ACCENT = "#0078D4"
OCHRE = "#8A6116"
MOSS = "#0B6A0B"
BRICK = "#C42B1C"

LIGHT_GRAY = "#C8C6C4"
LIGHT_YELLOW = "#CA5010"
LIGHT_GREEN = MOSS
LIGHT_RED = BRICK


def combo_row_colors(dark=None):
    return {
        "selected_bg": "#F3F9FD",
        "selected_bar": ACCENT,
        "selected_title": INK,
        "hover_bg": "#F5F5F5",
        "title": INK,
        "sub": MUTED,
        "idle_title": INK,
        "idle_sub": MUTED,
        "warn": OCHRE,
    }


def device_row_colors(dark=None):
    return {
        "group_bg": "#F5F5F5",
        "group_fg": MUTED,
        "selected": "#F3F9FD",
        "hover": "#F5F5F5",
        "title": INK,
        "sub": MUTED,
    }


def meter_colors(dark=None):
    return {
        "bg": SURFACE,
        "border": LINE,
        "title": MUTED,
        "slot": "#F3F3F3",
        "peak": INK,
        "spec_bg": SURFACE,
        "spec_border": LINE,
        "fill_lo": ACCENT,
        "fill_mid": ACCENT,
        "fill_hi": BRICK,
        "spec_bar": ACCENT,
    }


def status_kind_colors(kind="idle"):
    table = {
        "idle": (MUTED, PAPER, LINE),
        "busy": (ACCENT, "#F3F9FD", "#D0E7F8"),
        "ok": (MOSS, "#F1F8F1", "#D4E6D4"),
        "fail": (BRICK, "#FDF3F1", "#F0D0CC"),
    }
    return table.get(kind, table["idle"])


def hint_colors(warn=True):
    if warn:
        return OCHRE, "#FFF9F0", "#E8D5A8"
    return MUTED, "#F5F5F5", LINE


# 只画窗口骨架，不要覆盖 Fluent 按钮/输入框/下拉框。
LIGHT_QSS = f"""
QMainWindow, QDialog {{
    background-color: {PAPER};
    color: {INK};
}}
QWidget {{
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 13px;
    color: {INK};
}}
QToolTip {{
    background: {INK};
    color: {PAPER};
    border: 1px solid {LINE_STRONG};
    padding: 6px 8px;
}}

QFrame#header {{
    background-color: {SURFACE};
    border: none;
    border-bottom: 1px solid {LINE};
}}
QLabel#appTitle {{
    font-size: 15px;
    font-weight: 600;
    color: {INK};
}}

QGroupBox {{
    border: 1px solid {LINE};
    border-radius: 4px;
    margin-top: 12px;
    padding: 14px 12px 10px 12px;
    background-color: {SURFACE};
    font-weight: 600;
    font-size: 12px;
    color: {MUTED};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    background-color: {SURFACE};
    color: {MUTED};
}}

QLabel#fieldLabel {{
    color: {MUTED};
    font-size: 12px;
    font-weight: 600;
}}
QLabel#fieldLabel:disabled {{
    color: {LINE_STRONG};
}}
QLabel#roleName {{
    font-size: 14px;
    font-weight: 600;
    color: {INK};
}}
QLabel#roleMeta, QLabel#roleInfo {{
    font-size: 11px;
    color: {MUTED};
}}
QLabel#roleWarn {{
    font-size: 12px;
    font-weight: 600;
    color: {OCHRE};
}}

QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollBar:vertical {{
    border: none;
    background: transparent;
    width: 8px;
}}
QScrollBar::handle:vertical {{
    background: {LINE_STRONG};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QFrame#roleCard {{
    background-color: {PAPER};
    border: 1px solid {LINE};
    border-radius: 4px;
}}
QFrame#speakerCard {{
    background-color: {SURFACE};
    border: 1px solid {LINE};
    border-radius: 4px;
}}
QFrame#speakerCard[selected="true"] {{
    background-color: #F3F9FD;
    border: 1px solid {ACCENT};
}}

QFrame#badgeBox {{
    background: {PAPER};
    border: 1px solid {LINE};
    border-radius: 4px;
}}
QLabel#stateLabel {{
    font-size: 12px;
    font-weight: 600;
    color: {MUTED};
}}
QLabel#chipOk {{
    font-size: 12px;
    font-weight: 600;
    color: {MOSS};
    background: #F1F8F1;
    border: 1px solid #D4E6D4;
    border-radius: 4px;
    padding: 4px 8px;
}}
QLabel#chipInfo {{
    font-size: 12px;
    font-weight: 600;
    color: {ACCENT};
    background: #F3F9FD;
    border: 1px solid #D0E7F8;
    border-radius: 4px;
    padding: 4px 8px;
}}
QLabel#chipMute {{
    font-size: 12px;
    font-weight: 600;
    color: {MUTED};
    background: {PAPER};
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 4px 8px;
}}
QLabel#chipWarn {{
    font-size: 12px;
    font-weight: 600;
    color: {BRICK};
    background: #FDF3F1;
    border: 1px solid #F0D0CC;
    border-radius: 4px;
    padding: 4px 8px;
}}

QStatusBar {{
    background-color: {SURFACE};
    color: {MUTED};
    border-top: 1px solid {LINE};
}}
QSplitter::handle {{
    background: {LINE};
    width: 1px;
}}

QTabWidget::pane {{
    border: 1px solid {LINE};
    border-radius: 4px;
    background-color: {SURFACE};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {PAPER};
    color: {MUTED};
    border: 1px solid {LINE};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 3px;
    min-height: 24px;
}}
QTabBar::tab:hover {{
    color: {INK};
    background-color: #F5F5F5;
}}
QTabBar::tab:selected {{
    background-color: {SURFACE};
    color: {INK};
    border-bottom: 1px solid {SURFACE};
}}
"""
