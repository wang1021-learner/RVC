"""
RVC 推理服务器 — 通过 WebSocket 接收音频帧，调用 RVCPipeline 推理，返回结果

启动:
  python server/rvc_server.py --host 0.0.0.0 --port 8765
  Windows: start_server.bat
依赖: websockets, numpy, torch (GPU 推理用)
"""

import asyncio, json, struct, argparse, traceback, sys, socket, os
import concurrent.futures
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# --cpu 必须在 import pipeline / Config 之前生效
if "--cpu" in sys.argv:
    os.environ["RVC_FORCE_CPU"] = "1"

import numpy as np
import websockets

from worker.rvc_pipeline import RVCPipeline

ROOT = Path(__file__).resolve().parent.parent

INFER_KEYS = (
    "block_time", "crossfade_time", "extra_time", "f0method",
    "I_noise_reduce", "O_noise_reduce", "rms_mix_rate", "threhold",
    "limiter_enable", "limiter_threshold_db",
    "hf_mix_rate", "presence", "deesser_enable",
    "vad_enable", "vad_threshold",
)


def _basename(path):
    return Path(str(path or "")).name


def _file_info(pipeline):
    info = pipeline.loaded_file_info() if hasattr(pipeline, "loaded_file_info") else {}
    pth = info.get("model_path") or ""
    idx = info.get("index_path") or ""
    return {
        "model": _basename(pth),
        "model_path": pth,
        "index_path": idx,
        "index_loaded": bool(info.get("index_loaded")),
    }


def _gpu_name():
    if os.environ.get("RVC_FORCE_CPU") == "1":
        return "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


def _lan_urls(port):
    urls = [f"ws://127.0.0.1:{port}"]
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                u = f"ws://{ip}:{port}"
                if u not in urls:
                    urls.append(u)
    except Exception:
        pass
    return urls


def _list_pth():
    weights = ROOT / "assets" / "weights"
    if not weights.is_dir():
        return []
    return sorted(f.name for f in weights.iterdir() if f.is_file() and f.suffix.lower() == ".pth")


def _list_indices():
    found = []
    logs = ROOT / "logs"
    if logs.is_dir():
        for p in logs.rglob("*.index"):
            found.append(p.name)
    extra = ROOT / "assets" / "indices"
    if extra.is_dir():
        for p in extra.glob("*.index"):
            found.append(p.name)
    return sorted(set(found))


def _list_models_payload():
    return _list_pth(), _list_indices()


class RVCServer:
    """WebSocket 服务器，管理多个客户端连接和模型切换"""

    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.pipeline = RVCPipeline(on_status=print)
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rvc-infer")
        self._pump_lock = asyncio.Lock()
        self._pending = None
        self._pump_task = None

    def _apply_infer_params(self, cmd):
        kwargs = {k: cmd[k] for k in INFER_KEYS if k in cmd}
        if kwargs:
            self.pipeline.configure(**kwargs)

    async def _on_infer_thread(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, fn, *args)

    async def handle(self, websocket):
        """处理单个 WebSocket 连接"""
        remote = websocket.remote_address
        print(f"[+] 客户端连接: {remote}")
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # ── 二进制消息 = 音频帧 ──
                    await self._handle_audio(websocket, message)
                else:
                    # ── 文本消息 = 控制命令 (JSON) ──
                    await self._handle_command(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"[-] 客户端断开: {remote}")
        except Exception:
            traceback.print_exc()

    async def _handle_command(self, websocket, message: str):
        """处理控制命令: 加载模型 / 配置参数 / 获取状态"""
        try:
            cmd = json.loads(message)
        except json.JSONDecodeError:
            return

        action = cmd.get("action")
        if action == "start":
            def _start():
                if self.pipeline.rvc is None:
                    return {"error": "模型未加载"}
                self._apply_infer_params(cmd)
                self.pipeline.start()
                return {
                    "status": "ok",
                    "samplerate": self.pipeline.samplerate,
                    "channels": self.pipeline.channels,
                    "block_size": self.pipeline._block_frame,
                }
            await websocket.send(json.dumps(await self._on_infer_thread(_start)))
        elif action == "load":
            model_path = cmd.get("model_path", "")
            index_path = cmd.get("index_path", "")
            pitch = cmd.get("pitch", 0)
            index_rate = cmd.get("index_rate", 0.0)
            formant = float(cmd.get("formant", 0.0))
            print(f"[*] 加载模型: {model_path}")

            def _load():
                rvc = getattr(self.pipeline, "rvc", None)
                same_pth = rvc is not None and (
                    getattr(rvc, "pth_path", None) == model_path
                    or _basename(getattr(rvc, "pth_path", "")) == _basename(model_path)
                )
                same_index = rvc is not None and (
                    _basename(getattr(rvc, "index_path", "")) == _basename(index_path)
                )
                if same_pth and same_index:
                    print("[*] 模型已加载, 跳过重载")
                    self.pipeline.change_pitch(pitch)
                    self.pipeline.change_index_rate(float(index_rate))
                    self.pipeline.change_formant(formant)
                    self._apply_infer_params(cmd)
                else:
                    if self.pipeline.is_active:
                        self.pipeline.stop()
                    ok = self.pipeline.load_speaker(
                        model_path, index_path, pitch, index_rate,
                        last_rvc=rvc,
                    )
                    if not ok:
                        return {"error": "模型加载失败"}
                    self.pipeline.change_formant(formant)
                    self._apply_infer_params(cmd)
                    # 换模型只加载权重。CUDA 预热留给客户端点「启动」，
                    # 否则每次切角色都卡 10–20 秒。
                files = _file_info(self.pipeline)
                print(
                    f"[*] 模型加载完成, sr={self.pipeline.samplerate}"
                    f" pth={files.get('model_path') or '-'}"
                    f" index={files.get('index_path') or '无'}"
                    f" index_loaded={files.get('index_loaded')}"
                )
                return {
                    "status": "ok",
                    "samplerate": self.pipeline.samplerate,
                    "channels": self.pipeline.channels,
                    "block_size": getattr(self.pipeline, "_block_frame", None),
                    "active": bool(self.pipeline.is_active),
                    **files,
                }
            await websocket.send(json.dumps(await self._on_infer_thread(_load)))
        elif action == "configure":
            # 只改字段，绝不进推理线程：否则滑条指令会排在 CUDA 后面，
            # 收包循环卡住，客户端 recv 超时 → 重连 → 看起来像卡死。
            self._apply_infer_params(cmd)
        elif action == "set_live":
            rvc = getattr(self.pipeline, "rvc", None)
            if rvc is not None:
                if "pitch" in cmd:
                    self.pipeline.change_pitch(cmd["pitch"])
                if "formant" in cmd:
                    self.pipeline.change_formant(cmd["formant"])
                if "index_rate" in cmd:
                    rate = float(cmd["index_rate"])
                    # 首次打开索引会读盘，丢进推理线程，别卡住收包循环
                    if rate != 0 and not hasattr(rvc, "index"):
                        self._pool.submit(self.pipeline.change_index_rate, rate)
                    else:
                        self.pipeline.change_index_rate(rate)
        elif action == "list_models":
            loop = asyncio.get_running_loop()
            models, indices = await loop.run_in_executor(None, _list_models_payload)
            await websocket.send(json.dumps({
                "status": "ok",
                "models": models,
                "indices": indices,
            }))
        elif action == "list_indices":
            loop = asyncio.get_running_loop()
            _, indices = await loop.run_in_executor(None, _list_models_payload)
            await websocket.send(json.dumps({
                "status": "ok",
                "indices": indices,
            }))
        elif action == "stop":
            # 只清 active 标志，当前这块推理跑完即停；不要排队等 GPU
            self.pipeline.stop()
            await websocket.send(json.dumps({"status": "stopped"}))
            print("[*] 推理已停止")
        elif action == "ping":
            await websocket.send(json.dumps({"status": "pong"}))
        elif action == "status":
            files = _file_info(self.pipeline)
            await websocket.send(json.dumps({
                "loaded": self.pipeline.is_loaded,
                "active": self.pipeline.is_active,
                "samplerate": self.pipeline.tgt_sr,
                "block_size": getattr(self.pipeline, "_block_frame", None),
                "gpu": _gpu_name(),
                **files,
            }))

    def _parse_audio(self, data: bytes):
        if not data or len(data) < 8:
            return None
        req_seq = struct.unpack(">I", data[:4])[0]
        n_samples = struct.unpack(">I", data[4:8])[0]
        # 边界与 DoS 防护（单块最大 4 秒 48k 采样点）
        if n_samples <= 0 or n_samples > 192000:
            return None
        expected_len = 8 + n_samples * 4
        if len(data) < expected_len:
            print(f"[!] 音频数据不完整: 期望 {n_samples} 采样点 ({expected_len} 字节), 实际 {len(data)} 字节")
            return None
        audio = np.frombuffer(data[8:expected_len], dtype=np.float32)
        return req_seq, np.asarray(audio, dtype=np.float32)

    def _infer_audio(self, audio):
        if not self.pipeline.is_active:
            return None
        out, _ = self.pipeline.process_chunk(audio)
        return np.asarray(out, dtype=np.float32)

    async def _handle_audio(self, websocket, data: bytes):
        parsed = self._parse_audio(data)
        if parsed is None:
            return
        req_seq, audio = parsed
        async with self._pump_lock:
            if self._pump_task is not None and not self._pump_task.done():
                self._pending = (websocket, req_seq, audio)
                return
            self._pending = None
            self._pump_task = asyncio.create_task(
                self._pump_audio(websocket, req_seq, audio))

    async def _pump_audio(self, websocket, req_seq, audio):
        try:
            while True:
                out = await self._on_infer_thread(self._infer_audio, audio)
                if out is None:
                    return
                response = struct.pack(">II", req_seq, out.size) + out.tobytes()
                await websocket.send(response)
                async with self._pump_lock:
                    nxt = self._pending
                    self._pending = None
                    if nxt is None:
                        return
                    websocket, req_seq, audio = nxt
        except Exception:
            traceback.print_exc()

    async def start(self):
        print(f"RVC Server 启动: ws://{self.host}:{self.port}")
        print("GPU:", _gpu_name())
        print("客户端可填:")
        for u in _lan_urls(self.port):
            print("  ", u)
        print("模型:", ", ".join(_list_pth()[:8]) or "(assets/weights 为空)")

        async def _handler(websocket, path=None):
            await self.handle(websocket)

        async with websockets.serve(
            _handler,
            self.host,
            self.port,
            max_size=2 ** 24,
            ping_interval=20,
            ping_timeout=20,
        ):
            await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVC 推理服务器")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址。局域网请用 0.0.0.0（默认）；仅本机用 127.0.0.1",
    )
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制 CPU 推理（无独显的服务器请加这个）",
    )
    args = parser.parse_args()
    asyncio.run(RVCServer(args.host, args.port).start())
