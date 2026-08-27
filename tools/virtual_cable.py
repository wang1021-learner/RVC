"""检测 / 引导安装虚拟声卡（VB-Cable、VoiceMeeter）。"""
import webbrowser

CABLE_HINTS = (
    "cable",
    "vb-audio",
    "voicemeeter",
    "vm-vaio",
    "virtual cable",
    "virtuacable",
    "blackhole",
    "loopback",
)

# 蓝牙免提麦：16 kHz、大缓冲、和虚拟声卡两套时钟，给别人听时极易卡/炸麦
BT_HINTS = (
    "bluetooth",
    "bth",
    "hands-free",
    "handsfree",
    "hfp",
    "hsp",
    "a2dp",
    "tws",
    "airpods",
    "earbuds",
    "galaxy buds",
    "redmi buds",
    "true wireless",
    "蓝牙",
    "免提",
    "headset (",
    "headphones (",
    "耳机 (",
)

INSTALL_URLS = (
    ("VB-Audio Cable（推荐）", "https://vb-audio.com/Cable/"),
    ("VoiceMeeter", "https://vb-audio.com/Voicemeeter/"),
)


def is_virtual_name(name):
    n = (name or "").lower()
    return any(h in n for h in CABLE_HINTS)


def is_bluetooth_name(name):
    n = (name or "").lower()
    return any(h in n for h in BT_HINTS)


def find_virtual_devices(devs, apis):
    found = []
    for i, d in enumerate(devs):
        if not is_virtual_name(d.get("name", "")):
            continue
        api = ""
        try:
            api = apis[d.get("hostapi", 0)]["name"]
        except Exception:
            pass
        found.append({
            "index": i,
            "name": d["name"],
            "api": api,
            "in_ch": int(d.get("max_input_channels", 0) or 0),
            "out_ch": int(d.get("max_output_channels", 0) or 0),
        })
    return found


def open_install_page(url):
    try:
        webbrowser.open(url)
        return True
    except Exception:
        return False


def route_self_check(devs, apis):
    """路由自检：判断虚拟声卡是否具备「RVC 输出 → 软电话麦克风」两侧。

    VB-Cable 命名：播放端(CABLE Input，RVC 输出到这里) / 录制端(CABLE Output，
    软电话把它选作麦克风)。这里按 max_output_channels / max_input_channels 区分。

    返回 dict:
      installed  是否检测到任何虚拟声卡
      out_ok / in_ok  播放端 / 录制端 是否齐
      ok        两者都齐，路由可用
      out_devs / in_devs  对应设备名列表
      message   面向用户的指引文案
    """
    vdevs = find_virtual_devices(devs, apis)
    out_devs = [d["name"] for d in vdevs if d["out_ch"] > 0]
    in_devs = [d["name"] for d in vdevs if d["in_ch"] > 0]
    installed = bool(vdevs)
    out_ok = bool(out_devs)
    in_ok = bool(in_devs)
    ok = out_ok and in_ok

    if not installed:
        message = (
            "未检测到虚拟声卡：请先安装 VB-Audio Cable，"
            "再把 RVC 输出设备选为 CABLE，软电话麦克风选为 CABLE Output。"
        )
    elif ok:
        message = (
            "虚拟声卡就绪：RVC 输出设备选「%s」，"
            "软电话/通话软件里的麦克风选「%s」。"
        ) % (out_devs[0], in_devs[0])
    elif not out_ok:
        message = "虚拟声卡缺「播放端」：找不到可作为 RVC 输出目标的虚拟设备，请检查声卡驱动。"
    else:
        message = "虚拟声卡缺「录制端」：找不到可喂给软电话的虚拟麦克风，请检查声卡驱动。"

    return {
        "installed": installed,
        "ok": ok,
        "out_ok": out_ok,
        "in_ok": in_ok,
        "out_devs": out_devs,
        "in_devs": in_devs,
        "message": message,
    }
