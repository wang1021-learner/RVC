"""
冻结版（打包 exe）本机推理：以子进程方式运行仓库内的 rvc_server.py。

架构：客户端 exe 不内置 torch。打包时仓库源码/模型随包放在
<包根>/source/ 下；客户点击「安装本地推理」后，安装器在 <包根>/runtime/
装好嵌入式 Python + torch 等依赖。本模块负责在本地模式启动/回收该服务，
并复用 RVCClient 的 WebSocket 协议（ws://127.0.0.1:8765）。
"""
import socket
import subprocess
import sys
import time
from pathlib import Path

from tools.app_paths import is_frozen, package_root
from worker.rvc_client import RVCClient

DEFAULT_LOCAL_PORT = 8765


def runtime_python():
    """本机推理用的 Python：冻结版用包内 runtime，源码版用当前解释器。"""
    if is_frozen():
        return package_root() / "runtime" / "python.exe"
    return Path(sys.executable)


def source_dir():
    """服务端源码根目录：冻结版为包内 source/，源码版为项目根目录。"""
    if is_frozen():
        return package_root() / "source"
    return package_root()


def runtime_installed():
    if not is_frozen():
        return True  # 源码模式：当前解释器已具备全部依赖
    return runtime_python().is_file()


def pack_mode():
    """打包类型：source / server（瘦客户端）/ standalone（单机版）。"""
    if not is_frozen():
        return "source"
    marker = package_root() / "pack_mode.txt"
    try:
        if marker.is_file():
            text = marker.read_text(encoding="utf-8", errors="ignore").strip().lower()
            if text in ("server", "standalone"):
                return text
            if text == "local":
                return "standalone"
    except Exception:
        pass
    bat = package_root() / "install_local.bat"
    script = source_dir() / "server" / "rvc_server.py"
    if bat.is_file() or script.is_file():
        return "standalone"
    return "server"


def local_infer_ready():
    """冻结版必须同时具备 runtime Python 和 source 里的推理服务。"""
    if not is_frozen():
        return True
    script = source_dir() / "server" / "rvc_server.py"
    return runtime_installed() and script.is_file()


def port_in_use(port, host="127.0.0.1"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.3)
        result = s.connect_ex((host, port))
        return result == 0
    finally:
        s.close()


class LocalServerPipeline(RVCClient):
    """本地子进程推理：与 RVCClient 同协议，额外管理服务进程生命周期。

    模型路径原样透传：同一台机器上服务端可直接打开客户端选的文件；
    只有文件名时由服务端在自己的 assets/weights、logs 目录解析。
    """

    is_remote = False      # 语义上是"本地模式"（不映射远程服务器路径）
    is_network = True      # 但底层是网络管线，断线时可走重连恢复

    def __init__(self, server_url="ws://127.0.0.1:8765", on_status=None):
        super().__init__(server_url, on_status=on_status)
        self._proc = None
        self._owns_process = False
        self._log_fh = None

    # ── 服务进程管理 ──
    def ensure_server(self, timeout=60.0):
        """确保本机推理服务在运行；必要时拉起子进程并等它监听端口。

        首次启动服务端需导入 torch 并做 CUDA 探测（实测约 15 秒），
        慢机器上可达 30 秒以上，因此默认等待 60 秒。
        """
        if self._connected or port_in_use(DEFAULT_LOCAL_PORT):
            return True
        if not local_infer_ready():
            if pack_mode() == "server":
                msg = "本包为服务器客户端，不能本地推理，请连接远程服务器"
            else:
                msg = "本地推理未安装：请先点击「安装本地推理」"
            self.last_error = msg
            self._on_status(msg)
            return False
        py = runtime_python()
        src = source_dir()
        script = src / "server" / "rvc_server.py"
        if not py.is_file() or not script.is_file():
            self.last_error = "本地推理文件缺失，请重新安装"
            self._on_status(self.last_error)
            return False
        self._on_status("正在启动本地推理服务（首次约 10~30 秒）...")
        # 服务端输出写入日志文件，方便排查（冻结版无控制台）
        try:
            log_dir = package_root() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(log_dir / "local_server.log", "ab", buffering=0)
        except Exception:
            self._log_fh = None
        try:
            self._proc = subprocess.Popen(
                [str(py), "-u", str(script), "--host", "127.0.0.1",
                 "--port", str(DEFAULT_LOCAL_PORT)],
                cwd=str(src),
                stdout=self._log_fh or subprocess.DEVNULL,
                stderr=self._log_fh or subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._owns_process = True
        except Exception as e:
            self.last_error = "本地推理启动失败: %s" % e
            self._on_status(self.last_error)
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            if port_in_use(DEFAULT_LOCAL_PORT):
                return True
            if self._proc is not None and self._proc.poll() is not None:
                self.last_error = "本地推理服务异常退出（检查显卡驱动与安装日志）"
                self._on_status(self.last_error)
                return False
            time.sleep(0.2)
        self.last_error = "本地推理服务启动超时"
        self._on_status(self.last_error)
        return False

    def stop_server(self):
        """结束由本管线拉起的服务进程。"""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=0.4)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._owns_process = False
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    # ── 接口适配 ──
    def connect(self, timeout=5):
        if not self.ensure_server():
            return False
        return super().connect(timeout)

    def unload(self):
        """停止推理并回收本管线拉起的服务进程（模型留在服务端）。"""
        try:
            self.stop()
        except Exception:
            pass
        if self._owns_process:
            self.stop_server()

    def load_speaker(self, model_path, index_path="", pitch=0, index_rate=0.0,
                     formant=0.0, **params):
        if not self.ensure_server():
            return False
        return super().load_speaker(
            model_path, index_path, pitch, index_rate, formant, **params)

    def start(self, **params):
        if not self.ensure_server():
            return False
        if not self._connected and not super().connect(5):
            return False
        return super().start(**params)
