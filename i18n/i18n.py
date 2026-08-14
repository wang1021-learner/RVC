import json
import locale
import os
from tools.file_io import read_text


# 基于文件位置定位语言包（兼容源码运行与 PyInstaller 打包，不依赖启动目录）
_LOCALE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locale")


def load_language_list(language):
    return json.loads(read_text(os.path.join(_LOCALE_DIR, f"{language}.json")))


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
        if not os.path.exists(os.path.join(_LOCALE_DIR, f"{language}.json")):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language
