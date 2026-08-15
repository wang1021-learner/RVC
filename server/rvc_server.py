"""
RVC 推理服务器 — 通过 WebSocket 接收音频帧，调用 RVCPipeline 推理，返回结果

启动: python server/rvc_server.py --port 8765
依赖: websockets, numpy, torch (GPU 推理用)
"""

import asyncio, json, struct, time, argparse, traceback, sys
import concurrent.futures
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import websockets

from worker.rvc_pipeline import RVCPipeline

INFER_KEYS = (
    "block_time", "crossfade_time", "extra_time", "f0method",
    "I_noise_reduce", "O_noise_reduce", "rms_mix_rate", "threhold",
    "limiter_enable", "limiter_threshold_db",
    "hf_mix_rate", "presence", "deesser_enable",
    "vad_enable", "vad_threshold",
)


def _basename(path):
    return Path(str(path or "")).name


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
                self.pipeline.start()
                print(f"[*] 模型加载完成, sr={self.pipeline.samplerate}")
                return {
                    "status": "ok",
                    "samplerate": self.pipeline.samplerate,
                    "channels": self.pipeline.channels,
                    "block_size": self.pipeline._block_frame,
                }
            await websocket.send(json.dumps(await self._on_infer_thread(_load)))
        elif action == "configure":
            # 客户端 Fire-and-Forget：不回包，避免客户端等待与响应残留
            await self._on_infer_thread(self._apply_infer_params, cmd)
            print("[*] 参数已更新")
        elif action == "set_live":
            def _live():
                if self.pipeline.rvc is None:
                    return
                if "pitch" in cmd:
                    self.pipeline.change_pitch(cmd["pitch"])
                if "index_rate" in cmd:
                    self.pipeline.change_index_rate(float(cmd["index_rate"]))
                if "formant" in cmd:
                    self.pipeline.change_formant(float(cmd["formant"]))
            await self._on_infer_thread(_live)
        elif action == "list_models":
            # 列出服务器模型目录下的 .pth 文件（客户端添加角色时选择用）
            weights_dir = Path(__file__).resolve().parent.parent / "assets" / "weights"
            try:
                models = sorted(
                    f.name for f in weights_dir.iterdir()
                    if f.is_file() and f.name.lower().endswith(".pth")
                ) if weights_dir.is_dir() else []
            except Exception:
                models = []
            await websocket.send(json.dumps({"status": "ok", "models": models}))
        elif action == "stop":
            def _stop():
                if self.pipeline.is_active:
                    self.pipeline.stop()
            await self._on_infer_thread(_stop)
            await websocket.send(json.dumps({"status": "stopped"}))
            print("[*] 推理已停止")
        elif action == "ping":
            await websocket.send(json.dumps({"status": "pong"}))
        elif action == "status":
            await websocket.send(json.dumps({
                "loaded": self.pipeline.is_loaded,
                "active": self.pipeline.is_active,
                "samplerate": self.pipeline.tgt_sr,
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
        async with websockets.serve(self.handle, self.host, self.port,
                                     max_size=2**24,  # 16MB 消息上限
                                     ping_interval=30,
                                     ping_timeout=10):
            await asyncio.Future()  # 永久运行


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVC 推理服务器")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认 127.0.0.1 本地回环，局域网共享请输入 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    args = parser.parse_args()
    asyncio.run(RVCServer(args.host, args.port).start())
