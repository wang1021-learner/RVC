"""
RVC 推理服务器 — 通过 WebSocket 接收音频帧，调用 RVCPipeline 推理，返回结果

架构:
  连接 = Session（独立 DelayLine / SOLA / 音高缓存）
  ModelHub 共享 HuBERT / RMVPE / 已加载的 net_g
  GPU 仍单线程串行；默认 max_sessions=1（一人一机），以后只改上限

启动:
  python server/rvc_server.py --host 0.0.0.0 --port 8765
  Windows: start_server.bat
依赖: websockets, numpy, torch (GPU 推理用)
"""

import asyncio, json, struct, argparse, traceback, sys, socket, os, threading, time
import concurrent.futures
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.pyver import require_python_311
require_python_311()
# --cpu 必须在 import pipeline / Config 之前生效
if "--cpu" in sys.argv:
    os.environ["RVC_FORCE_CPU"] = "1"

import numpy as np
import websockets

from worker.rvc_pipeline import RVCPipeline, cuda_sync_or_die
from tools.model_assets import list_index_names

ROOT = Path(__file__).resolve().parent.parent

INFER_KEYS = (
    "block_time", "crossfade_time", "extra_time", "f0method",
    "I_noise_reduce", "O_noise_reduce", "rms_mix_rate", "threhold",
    "limiter_enable", "limiter_threshold_db",
    "hf_mix_rate", "presence", "deesser_enable",
    "vad_enable", "vad_threshold",
    "incremental_hubert",
    "protect",
)


def _basename(path):
    return Path(str(path or "")).name


def _name_key(path):
    return _basename(path).lower()


def _file_info(pipeline):
    if pipeline is None or not hasattr(pipeline, "loaded_file_info"):
        return {"model": "", "model_path": "", "index_path": "", "index_loaded": False}
    info = pipeline.loaded_file_info() or {}
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
    return list_index_names(ROOT)


_LIST_CACHE = {"t": 0.0, "models": None, "indices": None}


def _list_models_payload():
    now = time.monotonic()
    if (
        _LIST_CACHE["models"] is not None
        and now - _LIST_CACHE["t"] < 2.0
    ):
        return _LIST_CACHE["models"], _LIST_CACHE["indices"]
    models, indices = _list_pth(), _list_indices()
    _LIST_CACHE["t"] = now
    _LIST_CACHE["models"] = models
    _LIST_CACHE["indices"] = indices
    return models, indices


def _ws_alive(ws):
    if ws is None:
        return False
    try:
        if getattr(ws, "close_code", None) is not None:
            return False
    except Exception:
        pass
    try:
        state = getattr(ws, "state", None)
        if state is not None:
            name = str(getattr(state, "name", state)).upper()
            if name.endswith("OPEN"):
                return True
            if "CLOSE" in name:
                return False
    except Exception:
        pass
    try:
        closed = getattr(ws, "closed", None)
        if isinstance(closed, bool):
            return not closed
    except Exception:
        pass
    return True


def _pack_stage(stage):
    """回包尾 8 字节：特征/检索/音高/模型 毫秒。旧客户端只读 PCM，会忽略。"""
    def u16(key):
        try:
            v = int(round(float((stage or {}).get(key, 0) or 0)))
        except Exception:
            v = 0
        return max(0, min(65535, v))
    return struct.pack(">HHHH", u16("feature"), u16("index"), u16("pitch"), u16("model"))


def _busy_error(max_sessions):
    if int(max_sessions) <= 1:
        return "服务器正被其他客户端使用"
    return "服务器路数已满（%d）" % int(max_sessions)


class ModelHub:
    """共享权重仓：HuBERT / RMVPE / net_g / 索引按角色名缓存，不持有音频状态。"""

    def __init__(self):
        self._by_name = {}

    def donor(self, model_path):
        """同角色返回该合成器；否则返回任意一份（只为复用 HuBERT/RMVPE）。"""
        key = _name_key(model_path)
        if key and key in self._by_name:
            return self._by_name[key]
        if self._by_name:
            return next(iter(self._by_name.values()))
        return None

    def remember(self, rvc):
        if rvc is None:
            return
        key = _name_key(getattr(rvc, "pth_path", "") or "")
        if key:
            self._by_name[key] = rvc


class FairInfer:
    """同一 GPU 线程按会话轮询，一路 pending 满不会饿死另一路。"""

    def __init__(self, pool):
        self._pool = pool
        self._lock = asyncio.Lock()
        self._q = {}
        self._rr = deque()
        self._busy = False
        self._idle = None

    def _ensure_idle(self):
        if self._idle is None:
            self._idle = asyncio.Event()
            self._idle.set()
        return self._idle

    async def wait_idle(self):
        await self._ensure_idle().wait()

    async def submit(self, session, fn, *args):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        sid = id(session)
        async with self._lock:
            self._q.setdefault(sid, deque()).append((fut, fn, args))
            if sid not in self._rr:
                self._rr.append(sid)
            self._kick(loop)
        return await fut

    def drop(self, session):
        sid = id(session)
        q = self._q.pop(sid, None)
        if q:
            for fut, _, _ in q:
                if not fut.done():
                    fut.cancel()
        try:
            self._rr.remove(sid)
        except ValueError:
            pass

    def _kick(self, loop):
        if self._busy:
            return
        while self._rr:
            sid = self._rr[0]
            q = self._q.get(sid)
            if not q:
                self._rr.popleft()
                self._q.pop(sid, None)
                continue
            fut, fn, args = q.popleft()
            if q:
                self._rr.rotate(-1)
            else:
                self._rr.popleft()
                self._q.pop(sid, None)
            if fut.done():
                continue
            self._busy = True
            self._ensure_idle().clear()
            cf = loop.run_in_executor(self._pool, fn, *args)

            def _done(cf, fut=fut, loop=loop):
                def _finish():
                    try:
                        if not fut.done():
                            err = cf.exception()
                            if err is not None:
                                fut.set_exception(err)
                            else:
                                fut.set_result(cf.result())
                    except Exception:
                        if not fut.done():
                            fut.set_result(None)
                    self._busy = False
                    if not self._rr:
                        self._ensure_idle().set()
                    self._kick(loop)
                try:
                    loop.call_soon_threadsafe(_finish)
                except Exception:
                    pass

            cf.add_done_callback(_done)
            return


class Session:
    """一条 WebSocket = 一个会话：独立管线缓冲，占用一个并发名额。"""

    def __init__(self, websocket):
        self.ws = websocket
        self.remote = websocket.remote_address
        self.pipeline = None
        self.claimed = False
        self.pending = deque()
        self.pump_task = None
        self.lock = asyncio.Lock()

    def ensure_pipeline(self):
        if self.pipeline is None:
            self.pipeline = RVCPipeline(on_status=print)
        return self.pipeline


class RVCServer:
    """WebSocket 服务器：每连接一个 Session，权重走 ModelHub。"""

    def __init__(self, host="0.0.0.0", port=8765, max_sessions=1):
        self.host = host
        self.port = port
        self.max_sessions = max(1, int(max_sessions or 1))
        self.hub = ModelHub()
        self._sessions = {}
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rvc-infer")
        self._io_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rvc-io")
        self._fair = FairInfer(self._pool)
        self._slot_lock = asyncio.Lock()

    def _gc_dead(self):
        dead = [ws for ws, s in self._sessions.items() if not _ws_alive(ws)]
        for ws in dead:
            self._forget_session(ws, "原会话已断开")

    def _claimed_count(self):
        n = 0
        for s in self._sessions.values():
            if s.claimed and _ws_alive(s.ws):
                n += 1
        return n

    def _busy_for(self, session):
        if session is not None and session.claimed:
            return False
        return self._claimed_count() >= self.max_sessions

    async def _try_claim(self, session):
        async with self._slot_lock:
            if session.claimed:
                return True
            self._gc_dead()
            if self._claimed_count() >= self.max_sessions:
                return False
            session.claimed = True
            return True

    def _forget_session(self, websocket, reason="会话已释放"):
        session = self._sessions.pop(websocket, None)
        if session is None:
            return
        session.claimed = False
        session.pending.clear()
        p = session.pipeline
        if p is not None:
            try:
                p.stop()
            except Exception:
                pass
        try:
            self._fair.drop(session)
        except Exception:
            pass
        # 等当前推理从线程池出来，再旁路同步 GPU；卡住则退出交给看门狗
        self._wait_infer_idle_or_die(timeout=5.0)
        try:
            cuda_sync_or_die(timeout=5.0)
        except Exception:
            pass
        if p is not None:
            rvc = getattr(p, "rvc", None)
            if rvc is not None:
                self.hub.remember(rvc)
        print("[*] %s: %s" % (reason, session.remote))

    def _wait_infer_idle_or_die(self, timeout=5.0):
        """推理线程若还在跑，排一个空任务等它结束；超时视为 GPU 卡死。"""
        done = threading.Event()
        try:
            self._pool.submit(done.set)
        except Exception:
            return
        if not done.wait(float(timeout)):
            print("[!] 推理线程卡住，退出交给看门狗", flush=True)
            os._exit(1)

    def _prepare_loaded(self, pipeline):
        try:
            if pipeline is None or pipeline.rvc is None:
                return
            if pipeline.is_active or getattr(pipeline, "_keep_active", False):
                print("[*] 已在推理，跳过后台预热")
                return
            print("[*] 后台预热加速图...")
            pipeline.prepare(warmup=True)
            print("[*] 后台预热完成")
        except Exception:
            traceback.print_exc()

    def _apply_infer_params(self, pipeline, cmd):
        if pipeline is None:
            return
        kwargs = {k: cmd[k] for k in INFER_KEYS if k in cmd}
        if kwargs:
            pipeline.configure(**kwargs)

    async def _on_infer_thread(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, fn, *args)

    async def _on_io_thread(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._io_pool, fn, *args)

    async def _send_skip(self, session, seq):
        """丢块/停机时立刻回 n=0，避免客户端空等到超时后连锁错位。"""
        try:
            await session.ws.send(struct.pack(">II", int(seq) & 0xFFFFFFFF, 0))
        except Exception:
            pass

    async def handle(self, websocket):
        """处理单个 WebSocket 连接"""
        session = Session(websocket)
        self._sessions[websocket] = session
        remote = session.remote
        print(f"[+] 客户端连接: {remote}")
        try:
            sock = None
            try:
                trans = getattr(websocket, "transport", None)
                if trans is None:
                    writer = getattr(websocket, "writer", None)
                    trans = getattr(writer, "transport", None) if writer is not None else None
                sock = trans.get_extra_info("socket") if trans is not None else None
                if sock is not None:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            except Exception:
                pass
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._handle_audio(session, message)
                else:
                    await self._handle_command(session, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"[-] 客户端断开: {remote}")
        except Exception:
            traceback.print_exc()
        finally:
            self._forget_session(websocket)

    async def _handle_command(self, session, message: str):
        """处理控制命令: 加载模型 / 配置参数 / 获取状态"""
        try:
            cmd = json.loads(message)
        except json.JSONDecodeError:
            return

        websocket = session.ws
        action = cmd.get("action")
        if action == "start":
            p = session.pipeline
            if p is None or p.rvc is None:
                await websocket.send(json.dumps({"error": "模型未加载"}))
                return
            if not await self._try_claim(session):
                await websocket.send(json.dumps({"error": _busy_error(self.max_sessions)}))
                return
            self._apply_infer_params(p, cmd)
            p._keep_active = True
            new_sig = (
                p.block_time, p.crossfade_time, p.extra_time,
                p.samplerate, p.channels, p.I_noise_reduce, p.O_noise_reduce,
            )
            need_rebuild = getattr(p, "_buf_sig", None) != new_sig
            warming = bool(getattr(p, "_warming", False))
            if (not need_rebuild and hasattr(p, "_in_line")) or warming:
                p._active = True
                await websocket.send(json.dumps({
                    "status": "ok",
                    "samplerate": p.samplerate,
                    "channels": p.channels,
                    "block_size": getattr(p, "_block_frame", None),
                }))
            else:
                def _start():
                    p.start(warmup=False)
                    return {
                        "status": "ok",
                        "samplerate": p.samplerate,
                        "channels": p.channels,
                        "block_size": getattr(p, "_block_frame", None),
                    }
                await websocket.send(json.dumps(await self._on_infer_thread(_start)))
        elif action == "load":
            model_path = cmd.get("model_path", "")
            index_path = cmd.get("index_path", "")
            pitch = cmd.get("pitch", 0)
            index_rate = cmd.get("index_rate", 0.0)
            formant = float(cmd.get("formant", 0.0))
            print(f"[*] 加载模型: {model_path} <- {session.remote}")
            if not await self._try_claim(session):
                await websocket.send(json.dumps({"error": _busy_error(self.max_sessions)}))
                return
            pipeline = session.ensure_pipeline()
            hub = self.hub

            def _load():
                rvc = getattr(pipeline, "rvc", None)
                same_pth = rvc is not None and (
                    getattr(rvc, "pth_path", None) == model_path
                    or _basename(getattr(rvc, "pth_path", "")) == _basename(model_path)
                )
                same_index = rvc is not None and (
                    _basename(getattr(rvc, "index_path", "")) == _basename(index_path)
                )
                if same_pth and same_index:
                    print("[*] 模型已加载, 跳过重载")
                    pipeline.change_pitch(pitch)
                    pipeline.change_index_rate(float(index_rate))
                    pipeline.change_formant(formant)
                    self._apply_infer_params(pipeline, cmd)
                else:
                    if pipeline.is_active:
                        pipeline.stop()
                    donor = hub.donor(model_path) or rvc
                    ok = pipeline.load_speaker(
                        model_path, index_path, pitch, index_rate,
                        last_rvc=donor,
                    )
                    if not ok:
                        return {
                            "error": pipeline.last_error or "模型加载失败",
                        }
                    pipeline.change_formant(formant)
                    self._apply_infer_params(pipeline, cmd)
                    hub.remember(pipeline.rvc)
                pipeline.prepare(warmup=False)
                files = _file_info(pipeline)
                print(
                    f"[*] 模型加载完成, sr={pipeline.samplerate}"
                    f" pth={files.get('model_path') or '-'}"
                    f" index={files.get('index_path') or '无'}"
                    f" index_loaded={files.get('index_loaded')}"
                )
                return {
                    "status": "ok",
                    "samplerate": pipeline.samplerate,
                    "channels": pipeline.channels,
                    "block_size": getattr(pipeline, "_block_frame", None),
                    "active": bool(pipeline.is_active),
                    **files,
                }
            try:
                await asyncio.wait_for(self._fair.wait_idle(), 2.0)
            except asyncio.TimeoutError:
                pass
            resp = await self._on_io_thread(_load)
            await websocket.send(json.dumps(resp))
            if isinstance(resp, dict) and "error" not in resp:
                self._pool.submit(self._prepare_loaded, pipeline)
        elif action == "configure":
            if session.pipeline is None:
                return
            self._apply_infer_params(session.pipeline, cmd)
        elif action == "set_live":
            p = session.pipeline
            rvc = getattr(p, "rvc", None) if p is not None else None
            if rvc is not None:
                if "pitch" in cmd:
                    p.change_pitch(cmd["pitch"])
                if "formant" in cmd:
                    p.change_formant(cmd["formant"])
                if "index_rate" in cmd:
                    rate = float(cmd["index_rate"])
                    if rate != 0 and not hasattr(rvc, "index"):
                        self._pool.submit(p.change_index_rate, rate)
                    else:
                        p.change_index_rate(rate)
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
            if session.pipeline is not None:
                session.pipeline.stop()
            await websocket.send(json.dumps({"status": "stopped"}))
            print("[*] 推理已停止: %s" % (session.remote,))
        elif action == "ping":
            await websocket.send(json.dumps({"status": "pong"}))
        elif action == "status":
            p = session.pipeline
            files = _file_info(p)
            await websocket.send(json.dumps({
                "loaded": bool(p is not None and p.is_loaded),
                "active": bool(p is not None and p.is_active),
                "samplerate": (p.tgt_sr if p is not None else 48000),
                "block_size": getattr(p, "_block_frame", None) if p is not None else None,
                "gpu": _gpu_name(),
                "busy": self._busy_for(session),
                "warming": bool(p is not None and getattr(p, "_warming", False)),
                "sessions": self._claimed_count(),
                "max_sessions": self.max_sessions,
                **files,
            }))

    def _parse_audio(self, data: bytes):
        if not data or len(data) < 8:
            return None
        req_seq = struct.unpack(">I", data[:4])[0]
        n_samples = struct.unpack(">I", data[4:8])[0]
        if n_samples <= 0 or n_samples > 192000:
            return None
        expected_len = 8 + n_samples * 4
        if len(data) < expected_len:
            print(f"[!] 音频数据不完整: 期望 {n_samples} 采样点 ({expected_len} 字节), 实际 {len(data)} 字节")
            return None
        audio = np.frombuffer(data[8:expected_len], dtype=np.float32).copy()
        return req_seq, np.asarray(audio, dtype=np.float32)

    def _infer_audio(self, pipeline, audio):
        if pipeline is None or not pipeline.is_active:
            return None
        out, _ = pipeline.process_chunk(audio)
        stage = getattr(pipeline, "last_stage_ms", None) or {}
        return np.asarray(out, dtype=np.float32), stage

    async def _handle_audio(self, session, data: bytes):
        if not session.claimed or session.pipeline is None:
            return
        parsed = self._parse_audio(data)
        if parsed is None:
            return
        req_seq, audio = parsed
        async with session.lock:
            if session.pump_task is not None and not session.pump_task.done():
                if len(session.pending) >= 3:
                    drop_seq, _ = session.pending.popleft()
                    asyncio.create_task(self._send_skip(session, drop_seq))
                session.pending.append((req_seq, audio))
                return
            session.pending.clear()
            session.pump_task = asyncio.create_task(
                self._pump_audio(session, req_seq, audio))

    async def _pump_audio(self, session, req_seq, audio):
        pipeline = session.pipeline
        websocket = session.ws
        try:
            while True:
                packed = await self._fair.submit(
                    session, self._infer_audio, pipeline, audio)
                if packed is None:
                    await self._send_skip(session, req_seq)
                    async with session.lock:
                        if not session.pending:
                            return
                        req_seq, audio = session.pending.popleft()
                    continue
                out, stage = packed
                if self._sessions.get(websocket) is not session:
                    return
                response = (
                    struct.pack(">II", req_seq, out.size)
                    + out.tobytes()
                    + _pack_stage(stage)
                )
                await websocket.send(response)
                async with session.lock:
                    if not session.pending:
                        return
                    req_seq, audio = session.pending.popleft()
        except asyncio.CancelledError:
            return
        except Exception:
            traceback.print_exc()

    async def start(self):
        print(f"RVC Server 启动: ws://{self.host}:{self.port}")
        print("GPU:", _gpu_name())
        print("最大同时会话:", self.max_sessions, "（一人一机保持 1；多人加 --max-sessions）")
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
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=int(os.environ.get("RVC_MAX_SESSIONS", "1")),
        help="同时变声路数上限，默认 1（一人一机）。多人再加大。",
    )
    args = parser.parse_args()
    asyncio.run(RVCServer(args.host, args.port, max_sessions=args.max_sessions).start())
