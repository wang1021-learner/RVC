# -*- mode: python ; coding: utf-8 -*-
"""
RVC 实时变声客户端 (服务器推理模式) 打包配置
构建: pyinstaller rvc_realtime.spec --noconfirm
"""
import os

project_root = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(project_root, 'realtime_qt.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        (os.path.join(project_root, 'speakers.json'), '.'),
    ],
    hiddenimports=[
        'worker.rvc_client',
        'tools.audio_meter',
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
        'worker.rvc_pipeline',
        'gradio', 'uvicorn', 'fastapi', 'pytest', 'IPython', 'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

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
