"""浅色控制室主题。签名色是播出青绿。"""


def combo_row_colors(dark=None):
    return {
        "selected_bg": "#ecfdf5",
        "selected_bar": "#0f766e",
        "selected_title": "#0f766e",
        "hover_bg": "#f1f5f9",
        "title": "#0f172a",
        "sub": "#64748b",
        "idle_title": "#334155",
        "idle_sub": "#94a3b8",
    }


def device_row_colors(dark=None):
    return {
        "group_bg": "#f0f3f7",
        "group_fg": "#7f8c8d",
        "selected": "#d6eaf8",
        "hover": "#eef4fa",
        "title": "#2c3e50",
        "sub": "#95a5a6",
    }


def meter_colors(dark=None):
    return {
        "bg": "#f8fafc",
        "border": "#cbd5e1",
        "title": "#475569",
        "slot": "#e2e8f0",
        "peak": "#0f172a",
        "spec_bg": "#f1f5f9",
        "spec_border": "#e2e8f0",
    }


LIGHT_QSS = """
QMainWindow, QDialog {
    background-color: #f1f5f9;
    color: #0f172a;
}
QWidget {
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 13px;
    color: #1e293b;
}
QToolTip {
    background: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
    padding: 6px 8px;
    border-radius: 6px;
}

QFrame#header {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}
QLabel#appTitle {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.2px;
}

QGroupBox {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
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
    color: #475569;
    font-size: 12px;
    font-weight: 600;
}
QLabel#fieldLabel:disabled {
    color: #94a3b8;
}
QLabel#roleName {
    font-size: 14px;
    font-weight: 700;
    color: #0f172a;
}
QLabel#roleMeta, QLabel#roleInfo {
    font-size: 11px;
    color: #475569;
}
QLabel#roleWarn {
    font-size: 12px;
    font-weight: 600;
    color: #b45309;
}

QComboBox {
    combobox-popup: 0;
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 6px 28px 6px 10px;
    min-height: 28px;
    font-size: 13px;
    font-weight: 600;
    color: #0f172a;
}
QComboBox:hover {
    border-color: #64748b;
    background-color: #f8fafc;
}
QComboBox:focus, QComboBox:on {
    border: 2px solid #0f766e;
    padding: 5px 27px 5px 9px;
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
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 26px;
    color: #0f172a;
    font-size: 13px;
}
QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #64748b;
}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border: 2px solid #0f766e;
    padding: 3px 7px;
}

QSlider {
    min-height: 22px;
    max-height: 22px;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #e2e8f0;
    border-radius: 2px;
    margin: 0 4px;
}
QSlider::sub-page:horizontal {
    background: #0f766e;
    border-radius: 2px;
}
QSlider::add-page:horizontal {
    background: #e2e8f0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #0f766e;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}
QSlider:focus {
    outline: none;
}
QSlider:focus::handle:horizontal {
    border-color: #115e59;
    background: #ccfbf1;
}
QSlider:disabled::groove:horizontal,
QSlider:disabled::add-page:horizontal {
    background: #f1f5f9;
}
QSlider:disabled::sub-page:horizontal {
    background: #99f6e4;
}
QSlider:disabled::handle:horizontal {
    border-color: #94a3b8;
    background: #f8fafc;
}

QCheckBox, QRadioButton {
    spacing: 8px;
    color: #1e293b;
    font-weight: 500;
    min-height: 24px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #64748b;
    background-color: #ffffff;
}
QCheckBox::indicator {
    border-radius: 4px;
}
QRadioButton::indicator {
    border-radius: 9px;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #0f766e;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #0f766e;
    border-color: #0f766e;
}
QCheckBox:focus, QRadioButton:focus {
    outline: none;
}

QPushButton {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    padding: 6px 12px;
    min-height: 28px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #f8fafc;
    border-color: #64748b;
}
QPushButton:pressed {
    background-color: #e2e8f0;
}
QPushButton:disabled {
    color: #94a3b8;
    background-color: #f1f5f9;
    border-color: #e2e8f0;
}
QPushButton:focus {
    border: 2px solid #0f766e;
    padding: 5px 11px;
}
QPushButton#btnGhost {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #334155;
}
QPushButton#btnGhost:hover {
    background-color: #f1f5f9;
    border-color: #cbd5e1;
    color: #0f172a;
}
QPushButton#btnGhost:pressed {
    background-color: #e2e8f0;
}
QPushButton#btnConnect {
    background-color: #0f766e;
    color: #ffffff;
    border: none;
    font-weight: 700;
    min-width: 72px;
}
QPushButton#btnConnect:hover {
    background-color: #115e59;
}
QPushButton#btnConnect:pressed {
    background-color: #134e4a;
}
QPushButton#btnConnect:disabled {
    background-color: #99f6e4;
    color: #115e59;
}
QPushButton#btnConnect:focus {
    border: 2px solid #042f2e;
    padding: 4px 10px;
}
QPushButton#btnStart {
    font-size: 14px;
    font-weight: 700;
    padding: 8px 16px;
    min-height: 40px;
    max-height: 44px;
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
    background-color: #e2e8f0;
    color: #94a3b8;
}
QPushButton#btnStart:focus {
    border: 2px solid #042f2e;
    padding: 10px 18px;
}

QFrame#roleCard {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}
QFrame#speakerCard {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}
QFrame#speakerCard[selected="true"] {
    background-color: #ecfdf5;
    border: 1px solid #0f766e;
}

QFrame#badgeBox {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
}
QLabel#stateLabel {
    font-size: 12px;
    font-weight: 700;
    color: #475569;
}
QLabel#chipOk {
    font-size: 12px;
    font-weight: 700;
    color: #166534;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 6px;
    padding: 4px 8px;
}
QLabel#chipInfo {
    font-size: 12px;
    font-weight: 700;
    color: #6d28d9;
    background: #f5f3ff;
    border: 1px solid #ddd6fe;
    border-radius: 6px;
    padding: 4px 8px;
}
QLabel#chipMute {
    font-size: 12px;
    font-weight: 700;
    color: #475569;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 4px 8px;
}
QLabel#chipWarn {
    font-size: 12px;
    font-weight: 700;
    color: #b91c1c;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 4px 8px;
}

QStatusBar {
    background-color: #ffffff;
    color: #475569;
    border-top: 1px solid #e2e8f0;
}
QSplitter::handle {
    background: #e2e8f0;
    width: 1px;
}

QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background-color: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 4px;
    min-height: 24px;
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
QTabBar::tab:focus {
    border: 2px solid #0f766e;
}
"""
