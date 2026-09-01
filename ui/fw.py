"""QFluentWidgets 入口。主色用 Windows Fluent 蓝，浅色模式。"""
from qfluentwidgets import (
    Theme,
    setTheme,
    setThemeColor,
    PushButton,
    PrimaryPushButton,
    ComboBox,
    LineEdit,
    CheckBox,
    RadioButton,
    Slider,
    SpinBox,
    DoubleSpinBox,
    BodyLabel,
    CaptionLabel,
    StrongBodyLabel,
    TitleLabel,
)

ACCENT = "#0078D4"


def _disable_fluent_motion():
    """弹出菜单位置动画瞬间完成；不改滚动条动画，避免下拉滚轮失效。"""
    from qfluentwidgets.components.widgets.menu import MenuAnimationManager

    _init = MenuAnimationManager.__init__

    def init(self, menu):
        _init(self, menu)
        self.ani.setDuration(0)

    MenuAnimationManager.__init__ = init


def apply_fluent_theme():
    setTheme(Theme.LIGHT)
    setThemeColor(ACCENT)
    _disable_fluent_motion()
