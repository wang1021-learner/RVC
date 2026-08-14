# -*- mode: python ; coding: utf-8 -*-
"""
RVC 实时变声（本地 + 服务器）完整包
构建: pyinstaller rvc_full.spec --noconfirm
"""
import os

project_root = os.path.dirname(os.path.abspath(SPEC))

def p(*parts):
    return os.path.join(project_root, *parts)

datas = [
    (p('speakers.json'), '.'),
    (p('worker'), 'worker'),
    (p('infer'), 'infer'),
    (p('configs'), 'configs'),
    (p('i18n'), 'i18n'),
    (p('tools', '__init__.py'), 'tools'),
    (p('tools', 'audio_meter.py'), 'tools'),
    (p('tools', 'cuda_graph.py'), 'tools'),
    (p('tools', 'file_io.py'), 'tools'),
    (p('tools', 'torchgate'), 'tools/torchgate'),
    (p('assets', 'hubert_base'), 'assets/hubert_base'),
    (p('assets', 'rmvpe'), 'assets/rmvpe'),
    (p('assets', 'weights'), 'assets/weights'),
]
index_file = p('logs', 'thchs_v2', 'added_IVF2716_Flat_nprobe_1_thchs_v2_v2.index')
if os.path.isfile(index_file):
    datas.append((index_file, 'logs/thchs_v2'))

a = Analysis(
    [p('realtime_qt.py')],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'worker.rvc_client',
        'worker.rvc_pipeline',
        'websocket',
        'websocket._abnf',
        'infer.rtrvc', 'infer.hubert', 'infer.rmvpe', 'infer.fcpe',
        'infer.module.models', 'infer.module.modules', 'infer.module.attentions',
        'infer.module.commons', 'infer.module.transforms',
        'tools.cuda_graph', 'tools.torchgate', 'tools.audio_meter',
        'tools.file_io',
        'configs.config', 'i18n.i18n',
        'faiss', 'librosa', 'parselmouth', 'soundfile',
        'sounddevice', 'numpy',
        'PySide6.QtCore', 'PySide6.QtWidgets', 'PySide6.QtGui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'gradio', 'uvicorn', 'fastapi', 'pytest', 'IPython', 'jupyter',
        'matplotlib', 'tkinter',
        'tools.pymss', 'tools.pymss_core', 'tools.pymss_webui',
        'infer.vc',
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
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='RVC实时变声',
)
