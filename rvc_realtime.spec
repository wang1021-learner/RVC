# -*- mode: python ; coding: utf-8 -*-
"""
RVC 实时变声客户端 (服务器推理模式) 打包配置
构建: pyinstaller rvc_realtime.spec --noconfirm

体积优化说明（199MB -> ~135MB，配合 UPX 可到 ~85MB）：
- 移除 opengl32sw.dll（软件 OpenGL 渲染器，桌面 D3D/硬件 GL 已足够）
- 移除未使用的 Qt 模块（Quick/Qml/Pdf/Network/Svg/OpenGL 等）
- 未安装 UPX 时 upx=True 自动跳过；安装 UPX 后 DLL 可再压 30~40%
"""
import os

project_root = os.path.dirname(os.path.abspath(SPEC))

# 过滤掉的 DLL（全小写 basename 匹配）
_DROP_DLLS = {
    # 软件渲染 / ANGLE（桌面环境不需要）
    "opengl32sw.dll",
    "libegl.dll",
    "libglesv2.dll",
    # 未使用的 Qt 模块
    "qt6quick.dll", "qt6qml.dll", "qt6pdf.dll", "qt6pdfwidgets.dll",
    "qt6quickwidgets.dll", "qt6quickcontrols2.dll", "qt6quicktemplates2.dll",
    "qt6network.dll", "qt6svg.dll", "qt6xml.dll", "qt6printsupport.dll",
    "qt6opengl.dll", "qt6openglwidgets.dll", "qt6dbus.dll", "qt6sql.dll",
    "qt6test.dll", "qt6multimedia.dll", "qt6sensors.dll", "qt6websockets.dll",
    "qt6serialport.dll", "qt6concurrent.dll", "qt6charts.dll",
    "qt6datavisualization.dll", "qt6remoteobjects.dll", "qt6scxml.dll",
    "qt6positioning.dll", "qt6location.dll", "qt6webchannel.dll",
    "qt6websockets.dll", "qt6shadertools.dll",
    "qt6qmlmodels.dll", "qt6qmlmeta.dll", "qt6qmlworkerscript.dll",
    "qt6virtualkeyboard.dll",
}

# basename 包含这些子串的一律去掉
# 注意：不再过滤 openblas —— numpy 2.x 启动时强依赖 OpenBLAS，剔掉会导致 exe 启动崩溃
_DROP_PATTERNS = ("libblas", "liblapack", "libiomp", "libgomp")


def _filter_binaries(binaries):
    out = []
    for name, pth, typecode in binaries:
        base = os.path.basename(name).lower()
        if base in _DROP_DLLS:
            continue
        if any(p in base for p in _DROP_PATTERNS):
            continue
        # Qt 插件只保留 windows 平台插件（Fusion 样式内建，无图片/图标依赖）
        low = pth.lower().replace("/", "\\")
        if "\\plugins\\" in low and not low.endswith("platforms\\qwindows.dll"):
            continue
        out.append((name, pth, typecode))
    return out


def _filter_datas(datas):
    """去掉 Qml/Quick 相关的资源目录。"""
    out = []
    for name, pth, typecode in datas:
        low = name.lower().replace("\\", "/")
        if low.startswith("qml/") or low.startswith("qt6qml") or low.startswith("qt6quick"):
            continue
        out.append((name, pth, typecode))
    return out


a = Analysis(
    [os.path.join(project_root, 'realtime_qt.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        (os.path.join(project_root, 'speakers.json'), '.'),
        (os.path.join(project_root, 'presets.json'), '.'),
        (os.path.join(project_root, 'assets', 'icons'), 'assets/icons'),
    ],
    hiddenimports=[
        'worker.rvc_client',
        'worker.engine',
        'tools.audio_meter',
        'tools.audio_process',
        'tools.client_ns',
        'tools.file_io',
        'tools.app_paths',
        'ui',
        'ui.theme',
        'ui.common',
        'ui.devices',
        'ui.widgets',
        'ui.speakers',
        'ui.main_window',
        'websocket',
        'websocket._abnf',
        'sounddevice',
        'numpy',
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'PySide6.QtGui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 服务器推理，本地不需要
        'torch', 'faiss', 'librosa', 'scipy', 'parselmouth', 'soundfile',
        'infer', 'configs', 'i18n', 'matplotlib', 'tkinter',
        'worker.rvc_pipeline', 'tools.output_protector', 'tools.torchgate',
        'tools.cuda_graph',
        'gradio', 'uvicorn', 'fastapi', 'pytest', 'IPython', 'jupyter',
        # 未使用的 Qt 模块
        'PySide6.QtQuick', 'PySide6.QtQml', 'PySide6.QtPdf',
        'PySide6.QtPdfWidgets', 'PySide6.QtQuickWidgets',
        'PySide6.QtNetwork', 'PySide6.QtSvg', 'PySide6.QtXml',
        'PySide6.QtPrintSupport', 'PySide6.QtOpenGL', 'PySide6.QtOpenGLWidgets',
        'PySide6.QtDBus', 'PySide6.QtSql', 'PySide6.QtTest',
        'PySide6.QtMultimedia', 'PySide6.QtSensors', 'PySide6.QtWebSockets',
        'PySide6.QtSerialPort', 'PySide6.QtConcurrent', 'PySide6.QtCharts',
        'PySide6.QtDataVisualization',
        'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebChannel', 'PySide6.QtWebView',
        # numpy 用不到的重量级子模块
        'numpy.testing', 'numpy.distutils', 'numpy.f2py', 'numpy.polynomial',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

a.binaries = _filter_binaries(a.binaries)
a.datas = _filter_datas(a.datas)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RVC实时变声',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=os.path.join(project_root, 'assets', 'icons', 'app.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RVC实时变声',
)
