"""客户端共用常量、错误文案、设置读写。"""
import json
from pathlib import Path

import sounddevice as sd

from tools.app_paths import (
    bundled_dir, ensure_user_data, is_frozen, log_dir, package_root,
    presets_path, settings_path, speakers_path, user_data_dir,
)
from tools.model_assets import writable_asset_dir
from tools.file_io import write_json_atomic
from tools.virtual_cable import is_bluetooth_name
from ui.theme import (
    LIGHT_QSS, LIGHT_GRAY, LIGHT_YELLOW, LIGHT_GREEN, LIGHT_RED,
)

ensure_user_data()

PROJECT_ROOT = package_root()
SETTINGS_FILE = settings_path()
PRESETS_FILE = presets_path()
SPEAKERS_FILE = speakers_path()
WEIGHTS_DIR = writable_asset_dir("weights")
INDICES_DIR = writable_asset_dir("indices")
STYLE_QSS = LIGHT_QSS

NL = chr(10)
MSG_SERVER_PACK_NO_LOCAL = (
    "当前安装包是「服务器客户端」，只有界面和声卡，不含本机模型和 GPU 推理。"
    + NL + NL
    + "请勾选「远程服务器」，填写 ws://服务器IP:8765 后点「连接」。"
    + NL + NL
    + "若要在这台电脑上变声，请使用「RVC单机版」。"
)
MSG_NEED_INSTALL_LOCAL = (
    "本地推理尚未安装。"
    + NL + NL
    + "需要英伟达显卡，首次安装约 3.5GB（需联网）。"
    + NL
    + "请先点击「安装本地推理」，完成后再启动变声。"
)
DEFAULT_SERVER_URL = "ws://127.0.0.1:8765"
RESTART_KEYS = ("block_time", "crossfade_time", "extra_time", "I_noise_reduce", "O_noise_reduce")
DEFAULT_PARAMS = {
    "block_time": 0.06,
    "crossfade_time": 0.02,
    "extra_time": 0.8,
    "f0method": "rmvpe",
    "I_noise_reduce": False,
    "O_noise_reduce": False,
    "rms_mix_rate": 0.3,
    "threhold": -50,
    "limiter_enable": True,
    "limiter_threshold_db": -3.0,
    "hf_mix_rate": 0.25,
    "presence": 0.15,
    "deesser_enable": False,
    "vad_enable": False,
    "vad_threshold": 0.50,
    "incremental_hubert": False,
    "capture_denoise": True,
    "protect": 0.33,
}

# 输出到虚拟声卡给别人听时，块太短会欠载；欠载再复读上一块就是卡+炸麦
VIRTUAL_OUT_MIN_BLOCK = 0.08


def _split_hw_frames(sr):
    """分路采集/播放的声卡回调帧数。两套时钟就用短回调，与麦是否蓝牙、下游是哪款软件无关。"""
    sr = int(sr or 48000)
    n = int(round(sr * 0.02 / 64.0) * 64)
    return max(256, n)

# 场景预设：低延迟 / 高音质 / 游戏语音 / 唱歌
BUILTIN_PRESETS = [
    {
        "name": "低延迟",
        "params": {
            "block_time": 0.04, "crossfade_time": 0.01, "extra_time": 0.6,
            "incremental_hubert": False,
            "f0method": "rmvpe", "rms_mix_rate": 0.5, "threhold": -50,
            "I_noise_reduce": False, "O_noise_reduce": False,
        },
    },
    {
        "name": "高音质",
        "params": {
            "block_time": 0.08, "crossfade_time": 0.03, "extra_time": 1.2,
            "incremental_hubert": False,
            "f0method": "rmvpe", "rms_mix_rate": 0.3, "threhold": -55,
            "I_noise_reduce": False, "O_noise_reduce": False,
            "hf_mix_rate": 0.3, "presence": 0.15,
            "deesser_enable": False, "vad_enable": False,
            "protect": 0.33,
        },
    },
    {
        "name": "游戏语音",
        "params": {
            "block_time": 0.05, "crossfade_time": 0.015, "extra_time": 0.7,
            "f0method": "rmvpe", "rms_mix_rate": 0.6, "threhold": -45,
            "I_noise_reduce": False, "O_noise_reduce": False,
        },
    },
    {
        "name": "通话",
        "params": {
            "block_time": 0.08, "crossfade_time": 0.02, "extra_time": 0.8,
            "f0method": "rmvpe", "rms_mix_rate": 0.5, "threhold": -50,
            "I_noise_reduce": False, "O_noise_reduce": False,
            "hf_mix_rate": 0.2, "presence": 0.10,
            "deesser_enable": False, "vad_enable": False,
            "capture_denoise": False,
            "protect": 0.33,
        },
    },
    {
        "name": "唱歌",
        "params": {
            "block_time": 0.06, "crossfade_time": 0.02, "extra_time": 1.2,
            "f0method": "rmvpe", "rms_mix_rate": 0.0, "threhold": -60,
            "I_noise_reduce": False, "O_noise_reduce": True,
        },
    },
]


def _wasapi_fail_reason(exc):
    s = str(exc or "").lower()
    if "illegal combination" in s:
        return "输入输出不是同一类接口（请选同一类设备，或点「刷新设备」重试）"
    if "sample" in s or "rate" in s:
        return "采样率需为 48k（设备不支持当前采样率）"
    if "channel" in s:
        return "通道数不匹配（请更换设备或声道设置）"
    if "unsupported" in s or "format" in s:
        return "设备不支持当前音频格式（尝试切换共享模式）"
    if "full duplex" in s or "duplex" in s:
        return "设备不支持同时输入输出（全双工）"
    if any(k in s for k in ("unavail", "invalid device", "exclusive", "busy", "in use")):
        return "设备被占用或不支持独占（关闭占用程序，或改用共享模式）"
    if "not found" in s or "doesn't exist" in s or "no such device" in s:
        return "设备不存在，请点「刷新设备」后重选"
    text = str(exc).strip()
    return text[:80] if text else "未知原因"


def _friendly_net_error(e):
    """把底层网络异常翻译成用户可行动的提示。对不上则返回 None。"""
    raw = str(e or "")
    s = raw.lower()
    if "服务器正忙" in raw or "其他客户端" in raw or "路数已满" in raw:
        return "服务器正忙（已有人在变声）。等对方停，或让管理员加大同时路数。"
    if "refused" in s or "10061" in s:
        return "服务器拒绝连接。那台机器上的推理服务没开，或端口不是 8765。"
    if "等待服务器回复" in raw or "加载超时" in raw:
        return None
    if "timed out" in s or "timeout" in s or "10060" in s:
        return "连接超时。检查地址是否写对、网络是否通，以及云服务器安全组是否放行 TCP 8765。"
    if "getaddrinfo" in s or "nodename" in s or "name or service" in s or "11001" in s:
        return "地址无法解析。请写成 ws://IP:8765，不要漏 IP。"
    if "unreachable" in s or "10065" in s or "10051" in s or "10050" in s:
        return "网络不可达。先确认本机能 ping 通那台机器。"
    if "10013" in s or "permission denied" in s or "forbidden" in s:
        return "连接被拒绝（权限/防火墙）。请放行本机或服务器的 8765 端口。"
    if "10064" in s or "host is down" in s:
        return "对方主机无响应。请确认服务器已开机、推理服务已启动。"
    if "10022" in s or "10049" in s:
        return "地址无效。请检查 ws://IP:8765 是否写错。"
    if "bad status" in s or "status 502" in s or "status 503" in s or "status 404" in s:
        return "服务器没有可用的变声服务（HTTP 状态异常）。请确认启动的是 rvc_server，端口 8765。"
    if "handshake" in s or "10054" in s or "10053" in s or "reset" in s or "forcibly" in s or "broken pipe" in s:
        return "连接被断开。服务器可能崩溃、重启，或网络中断了。"
    if "closed" in s and ("connection" in s or "socket" in s or "websocket" in s):
        return "连接已关闭。请重新点「连接」。"
    if "ssl" in s or "certificate" in s:
        return "安全连接失败。局域网请用 ws://，不要用 wss://。"
    return None


def _friendly_error(e):
    """统一错误文案：能翻译就翻译，并告诉用户下一步。"""
    raw = str(e or "").strip()
    if "gpu 正忙" in raw.lower():
        return "服务器 GPU 正忙。请稍等几秒再加载角色。"
    if "加载超时" in raw or "等待服务器回复" in raw:
        return "加载超时。远端换模型要十几秒，请再点一次角色；反复出现请重新「连接」。"
    net = _friendly_net_error(e)
    if net:
        return net
    low = raw.lower()
    if "cuda" in low and ("out of memory" in low or "oom" in low):
        return "显存不足。请关掉其它占显卡的程序后重试。"
    if "cuda" in low and ("illegal" in low or "launch" in low or "device-side" in low):
        return "显卡推理出错。请重启推理服务后再连。"
    if "no such file" in low or "not found" in low or "找不到" in raw:
        return "找不到模型或索引文件。本地请核对路径；服务器模式只认文件名，远端要有同名文件。"
    if "本地推理" in raw:
        return raw[:220]
    if "模型加载失败" in raw or "load failed" in low:
        return "模型加载失败。请确认文件完整；服务器模式下先点「连接」，并确认远端有这个模型。"
    if "未连接" in raw or "not connected" in low or "无法连接" in raw:
        return "尚未连上服务器。请先点「连接」。"
    if "推理未能启动" in raw or "未能启动" in raw:
        return "推理未能启动。请先加载角色，并确认设备可用。"
    if "模型未加载" in raw:
        return "请先选择角色并等待加载完成，再启动变声。"
    if (
        "expecting value" in low
        or "jsondecode" in low
        or "没有返回有效" in raw
        or "回复为空" in raw
    ):
        return "服务器加载角色时没回上结果。请再点一次角色；不行就重新点「连接」。"
    if raw:
        return raw[:220]
    return "未知错误"


def _fp_from_devs(devs):
    try:
        items = []
        for d in (devs or []):
            name = d.get("name") or ""
            if is_bluetooth_name(name):
                continue
            items.append((
                name, d.get("hostapi", 0),
                d.get("max_input_channels", 0), d.get("max_output_channels", 0),
            ))
        return tuple(items)
    except Exception:
        return ()


def _device_fingerprint():
    """仅后台线程调用：UI 线程禁止 query_devices / _terminate。"""
    try:
        return _fp_from_devs(sd.query_devices())
    except Exception:
        return ()


def load_user_settings():
    path = settings_path()
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_user_settings(data):
    write_json_atomic(settings_path(), data)


def load_presets():
    """内置预设 + 用户 presets.json（追加，不覆盖内置）。"""
    presets = list(BUILTIN_PRESETS)
    try:
        if presets_path().is_file():
            data = json.loads(presets_path().read_text(encoding="utf-8"))
            for p in data.get("presets", []):
                if isinstance(p, dict) and p.get("name") and isinstance(p.get("params"), dict):
                    presets.append({"name": p["name"], "params": p["params"]})
    except Exception:
        pass
    return presets


def save_presets(presets):
    """保存用户预设（仅存非内置条目）。"""
    try:
        write_json_atomic(presets_path(), {"presets": presets})
    except Exception:
        pass


def _normalize_ws_url(url):
    url = (url or "").strip()
    if not url:
        return DEFAULT_SERVER_URL
    low = url.lower()
    if low.startswith("http://"):
        url = "ws://" + url[7:]
    elif low.startswith("https://"):
        url = "wss://" + url[8:]
    elif "://" not in url:
        url = "ws://" + url
    url = url.strip().rstrip("/")
    body = url.split("://", 1)[-1]
    host = body.split("/")[0]
    if host.startswith("[") and "]" in host:
        if "]:" not in host:
            url = url.replace(host, host + ":8765", 1)
    elif host and ":" not in host:
        url = url.replace(host, host + ":8765", 1)
    return url


def _validate_server_url(url):
    """返回 (规范化地址, 错误说明)。通过时错误为 None。"""
    url = _normalize_ws_url(url)
    low = url.lower()
    if not low.startswith(("ws://", "wss://")):
        return url, "地址应以 ws:// 开头，例如 ws://192.168.1.10:8765"
    rest = url.split("://", 1)[-1]
    host = rest.split("/")[0]
    if not host or host in (":", ":8765"):
        return url, "缺少服务器地址"
    return url, None


F0_CHOICES = (
    ("rmvpe", "准确（RMVPE）"),
    ("fcpe", "较快（FCPE）"),
    ("pm", "最快（PM）"),
)


def fill_f0_combo(cb, current="rmvpe"):
    cb.blockSignals(True)
    cb.clear()
    for key, label in F0_CHOICES:
        cb.addItem(label, userData=key)
    i = cb.findData(current or "rmvpe")
    cb.setCurrentIndex(i if i >= 0 else 0)
    cb.blockSignals(False)


def f0_from_combo(cb):
    v = cb.currentData()
    return v if v else "rmvpe"


def set_f0_combo(cb, key):
    i = cb.findData(key or "rmvpe")
    if i >= 0:
        cb.setCurrentIndex(i)


def _hostapi_zh(name):
    n = (name or "").upper()
    if "ASIO" in n:
        return "专业声卡"
    if "WASAPI" in n:
        return "系统低延迟"
    if "WDM" in n or "KS" in n:
        return "内核音频"
    if "DIRECTSOUND" in n:
        return "经典音频"
    if "MME" in n:
        return "兼容模式"
    return name or "其他"


def to_server_path(local_path: str) -> str:
    """只传文件名，由服务器在自己的 assets/weights、assets/indices 下解析。"""
    return Path(str(local_path or "")).name


def _speaker_file_sub(speaker):
    """下拉副行：模型文件 · 索引文件。"""
    pth = Path(str(getattr(speaker, "model_path", "") or "")).name or "无模型"
    idx = Path(str(getattr(speaker, "index_path", "") or "")).name
    if not idx:
        idx = "无索引"
    return pth + "  ·  " + idx


def _local_model_path(path):
    """本机模式下解析角色模型路径。找不到则返回空 Path。"""
    raw = str(path or "").strip()
    if not raw:
        return Path()
    p = Path(raw)
    if p.is_file():
        return p
    name = p.name
    roots = [bundled_dir(), package_root(), package_root() / "source",
             writable_asset_dir("weights").parent.parent]
    cands = []
    for root in roots:
        cands.extend((
            root / raw,
            root / "assets" / "weights" / name,
            root / "assets" / "indices" / name,
        ))
    for cand in cands:
        try:
            if cand.is_file():
                return cand
        except Exception:
            continue
    return Path()



