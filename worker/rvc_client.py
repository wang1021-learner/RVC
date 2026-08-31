"""
RVC 网络客户端 — 通过 WebSocket 连接推理服务器

替换本地 RVCPipeline，其他代码不变。
依赖: websocket-client, numpy
"""

import json, struct, time, threading, socket
import numpy as np
import websocket

from tools.client_ns import CaptureDenoise

# websocket-client 超时抛 WebSocketTimeoutException（不是 socket.timeout 子类），
# 不识别会落进 except Exception 被误判为断连
try:
    _WS_TIMEOUT_EXC = (socket.timeout, TimeoutError, websocket.WebSocketTimeoutException)
except AttributeError:
    _WS_TIMEOUT_EXC = (socket.timeout, TimeoutError)


class RVCClient:
    """WebSocket 客户端，连接远程 RVC 推理服务器"""

    def __init__(self, server_url="ws://127.0.0.1:8765", on_status=None):
        self.server_url = server_url
        self._on_status = on_status or (lambda _: None)
        self._ws = None
        # 拆分读写锁：发送(指令/音频块)与接收(推流响应)互不干扰，
        # 彻底解决 recv_audio 阻塞导致变调指令静默丢弃的问题
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        self._connected = False
        self._model_loaded = False
        self._active = False          # 服务器推理是否激活（is_active 的真实依据）
        self._seq = 0                 # 音频请求序号（响应配对，防超时后错位）

        # 服务器返回的参数
        self.samplerate = 48000
        self.channels = 1
        self._block_frame = 256    # 默认，连接后更新
        self.loaded_model_path = ""
        self.loaded_index_path = ""
        self.index_loaded = False
        self.gpu_name = ""
        self.last_error = ""
        self._last_good = None   # 超时用 PLC 延拓，避免插零导致发涩、断续
        self._rtt_ema = 0.04     # 秒，用成功回包估 RTT
        self._rtt_m2 = 0.0       # RTT 抖动（平方差 EMA）
        self._conceal_streak = 0
        self._last_stage_ms = {}
        self._ahead = {}
        self._virtual_out = False
        self.capture_denoise = True
        self._cap_ns = CaptureDenoise()

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def is_active(self) -> bool:
        return self._active

    def _ws_alive(self) -> bool:
        ws = self._ws
        if not self._connected or ws is None:
            return False
        try:
            if getattr(ws, "connected", True) is False:
                return False
        except Exception:
            return False
        sock = getattr(ws, "sock", None)
        if sock is None:
            return False
        try:
            if sock.fileno() < 0:
                return False
        except Exception:
            return False
        return True

    @staticmethod
    def _is_closed_err(e) -> bool:
        s = str(e or "").lower()
        return any(
            k in s
            for k in (
                "closed", "10054", "10053", "broken pipe",
                "not connected", "断开", "reset",
            )
        )

    def is_connected(self) -> bool:
        if self._ws_alive():
            return True
        self._connected = False
        return False

    @property
    def is_remote(self):
        return True

    @property
    def is_network(self):
        return True

    @property
    def last_stage_ms(self):
        return dict(self._last_stage_ms)

    def inflight_depth(self):
        """按 RTT+抖动估在途块数：局域网常为 1，弱网最多 3。本机子进程固定 1。"""
        if not getattr(self, "is_remote", True):
            return 1
        if getattr(self, "_virtual_out", False):
            return 1
        sr = float(self.samplerate or 48000)
        block = (float(self._block_frame) / sr) if self._block_frame else 0.06
        if block <= 0:
            block = 0.06
        rtt = float(getattr(self, "_rtt_ema", 0.04) or 0.04)
        std = float(np.sqrt(max(0.0, getattr(self, "_rtt_m2", 0.0))))
        extra = int(round((rtt + 2.0 * std) / block))
        return max(1, min(3, 1 + extra))

    @property
    def tgt_sr(self) -> int:
        return self.samplerate

    def _acquire_io(self, timeout):
        """带超时抢发送/接收锁，避免界面线程和音频线程互相死等。"""
        t0 = time.time()
        if not self._send_lock.acquire(timeout=timeout):
            return False
        remain = timeout - (time.time() - t0)
        if remain <= 0 or not self._recv_lock.acquire(timeout=max(0.05, remain)):
            self._send_lock.release()
            return False
        return True

    def _release_io(self):
        try:
            self._recv_lock.release()
        except Exception:
            pass
        try:
            self._send_lock.release()
        except Exception:
            pass

    def connect(self, timeout=5):
        """建立 WebSocket 连接（已连接则直接返回）"""
        if self._ws_alive():
            return True
        if self._ws is not None:
            self.abort()
        try:
            self._ws = websocket.create_connection(self.server_url, timeout=timeout)
            try:
                sock = getattr(self._ws, "sock", None)
                if sock is not None:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            except Exception:
                pass
            self._connected = True
            self._on_status(f"已连接服务器: {self.server_url}")
            # 查询服务器模型状态，同步本地标志（防服务器重启后状态错乱）
            try:
                if not self._acquire_io(min(3.0, float(timeout))):
                    return True
                try:
                    self._ws.send(json.dumps({"action": "status"}))
                    resp = self._recv_json(min(5.0, float(timeout)))
                finally:
                    self._release_io()
                self._model_loaded = bool(resp.get("loaded", False))
                self._active = bool(resp.get("active", False))
                if resp.get("samplerate"):
                    self.samplerate = resp["samplerate"]
                if resp.get("gpu"):
                    self.gpu_name = resp.get("gpu") or ""
                self._apply_loaded_files(resp)
                if self._model_loaded:
                    self._on_status(self._loaded_status_text(self.samplerate))
                else:
                    self._on_status("服务器无模型")
            except Exception:
                self._model_loaded = False
                self._drain()
            return True
        except Exception as e:
            self._connected = False
            self._ws = None
            self.last_error = str(e)
            self._on_status(f"连接失败: {e}")
            return False

    def _apply_loaded_files(self, resp):
        if not isinstance(resp, dict):
            return
        if "model_path" in resp:
            self.loaded_model_path = resp.get("model_path") or ""
        if "index_path" in resp:
            self.loaded_index_path = resp.get("index_path") or ""
        if "index_loaded" in resp:
            self.index_loaded = bool(resp.get("index_loaded"))
        elif self.loaded_index_path:
            self.index_loaded = True

    def loaded_file_info(self):
        return {
            "model_path": self.loaded_model_path,
            "index_path": self.loaded_index_path,
            "index_loaded": bool(self.index_loaded),
        }

    def _loaded_status_text(self, sr=None):
        pth = self.loaded_model_path or ""
        idx = self.loaded_index_path or ""
        bits = []
        if sr:
            bits.append(f"sr={sr}")
        bits.append(pth or "模型路径未知")
        if idx:
            bits.append("索引 " + idx + ("" if self.index_loaded else "（未启用）"))
        else:
            bits.append("无索引")
        return "模型已加载 " + " | ".join(bits)

    def disconnect(self):
        """断开连接。不阻塞等锁，避免界面卡死。"""
        self._active = False
        self.abort()

    def load_speaker(self, model_path: str, index_path: str = "",
                     pitch: int = 0, index_rate: float = 0.0,
                     formant: float = 0.0, **params) -> bool:
        """发送加载模型命令到服务器（未连接或连接已死则自动重连）"""
        cmd = {
            "action": "load",
            "model_path": model_path,
            "index_path": index_path,
            "pitch": pitch,
            "index_rate": index_rate,
            "formant": formant,
            "block_time": params.get("block_time", 0.06),
            "crossfade_time": params.get("crossfade_time", 0.02),
            "extra_time": params.get("extra_time", 0.8),
            "f0method": params.get("f0method", "rmvpe"),
            "I_noise_reduce": params.get("I_noise_reduce", False),
            "O_noise_reduce": params.get("O_noise_reduce", False),
            "rms_mix_rate": params.get("rms_mix_rate", 0.3),
            "threhold": params.get("threhold", -50),
        }
        for k in (
            "limiter_enable", "limiter_threshold_db",
            "hf_mix_rate", "presence", "deesser_enable",
            "vad_enable", "vad_threshold",
            "incremental_hubert",
            "protect",
        ):
            if k in params:
                cmd[k] = params[k]
        self._on_status("正在加载模型...")
        last_exc = None
        for attempt in (1, 2):
            if not self._ws_alive():
                if not self.connect(timeout=8):
                    return False
            try:
                if not self._acquire_io(8.0):
                    self.last_error = "服务器正忙"
                    self._on_status("加载失败: 服务器正忙")
                    return False
                try:
                    self._ws.send(json.dumps(cmd))
                    resp = self._recv_json(60.0)
                finally:
                    self._release_io()
                if "error" in resp:
                    self.last_error = str(resp["error"])
                    self._on_status(f"加载失败: {resp['error']}")
                    return False
                self.samplerate = resp.get("samplerate", 48000)
                self.channels = resp.get("channels", 1)
                if resp.get("block_size"):
                    self._block_frame = resp.get("block_size", 256)
                self._model_loaded = True
                self._active = bool(resp.get("active", False))
                self._apply_loaded_files(resp)
                self.last_error = ""
                self._on_status(self._loaded_status_text(self.samplerate))
                return True
            except Exception as e:
                last_exc = e
                self.last_error = str(e)
                self._on_status(f"加载失败: {e}")
                self._drain()
                if attempt == 1 and self._is_closed_err(e):
                    self.abort()
                    continue
                return False
        self.last_error = str(last_exc or "连接已关闭")
        return False

    def start(self, **params):
        """通知服务器恢复推理（模型已加载时停止后重启用）"""
        cmd = {"action": "start"}
        cmd.update(params)
        for attempt in (1, 2):
            if not self._ws_alive():
                if not self.connect(timeout=8):
                    self.last_error = self.last_error or "未连接服务器"
                    return False
            try:
                if not self._acquire_io(8.0):
                    self.last_error = "服务器正忙"
                    return False
                try:
                    self._ws.send(json.dumps(cmd))
                    resp = self._recv_json(60.0)
                finally:
                    self._release_io()
                if isinstance(resp, dict) and "error" in resp:
                    self.last_error = str(resp["error"])
                    self._on_status(resp["error"])
                    return False
                if isinstance(resp, dict):
                    if resp.get("samplerate"):
                        self.samplerate = resp["samplerate"]
                    if resp.get("channels"):
                        self.channels = resp["channels"]
                    if resp.get("block_size"):
                        self._block_frame = resp["block_size"]
                self._active = True
                self.last_error = ""
                return True
            except Exception as e:
                self.last_error = str(e)
                self._drain()
                if attempt == 1 and self._is_closed_err(e):
                    self.abort()
                    continue
                return False
        return False

    def abort(self):
        """立刻掐断连接。不要走 ws.close()：默认还要等 3 秒握手，界面会卡死。
        模型仍留在服务器，下次 connect() 用 status 同步。"""
        ws = self._ws
        self._ws = None
        self._connected = False
        self._active = False
        self._last_good = None
        self._last_stage_ms = {}
        self._conceal_streak = 0
        self._ahead = {}
        if ws is None:
            return
        sock = getattr(ws, "sock", None)
        try:
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    sock.settimeout(0.05)
                except Exception:
                    pass
                try:
                    sock.close()
                except Exception:
                    pass
            else:
                try:
                    ws.close(timeout=0)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            ws.connected = False
        except Exception:
            pass

    def stop(self):
        """停止推理（模型保留在服务器，is_loaded 不变）
        非阻塞：worker 正占用连接时放弃（服务器保持 active，状态由重连同步）"""
        if not self._ws:
            return
        if not self._send_lock.acquire(timeout=0.1):
            return
        try:
            self._ws.send(json.dumps({"action": "stop"}))
            self._active = False
            if self._recv_lock.acquire(timeout=0.2):
                try:
                    # 收掉服务器的 stop 响应，避免残留污染下一个请求
                    self._ws.settimeout(0.2)
                    self._ws.recv()
                except Exception:
                    pass
                finally:
                    self._recv_lock.release()
        except Exception:
            pass
        finally:
            self._send_lock.release()

    def _silence(self, n):
        """超时/丢包：自听用周期延拓；喂给虚拟声卡时补静音，避免双重 PLC 卡顿。"""
        n = int(n) if n else (self._block_frame or 256)
        if getattr(self, "_virtual_out", False):
            out = np.zeros(n, dtype=np.float32)
            last = self._last_good
            streak = int(getattr(self, "_conceal_streak", 0)) + 1
            self._conceal_streak = streak
            if streak == 1 and last is not None and last.size:
                fade_n = min(n, int(last.size), max(1, int(0.008 * (self.samplerate or 48000))))
                w = np.linspace(1.0, 0.0, fade_n, dtype=np.float32) * np.float32(0.7)
                out[:fade_n] = np.asarray(last[-fade_n:], dtype=np.float32) * w
            return out.reshape(-1, 1)
        last = self._last_good
        if last is None or last.size < 64:
            return np.zeros(n, dtype=np.float32).reshape(-1, 1)
        x = np.asarray(last, dtype=np.float32).reshape(-1)
        sr = float(self.samplerate or 48000)
        min_lag = max(32, int(sr / 400.0))
        max_lag = min(int(x.size) - 1, int(sr / 70.0))
        if max_lag <= min_lag:
            period = min(int(x.size), n)
        else:
            tail = x[-min(int(x.size), max(max_lag * 2, int(0.04 * sr))):]
            t = tail - float(tail.mean())
            c = np.correlate(t, t, mode="full")
            mid = c.size // 2
            region = c[mid + min_lag: mid + max_lag + 1]
            period = min_lag + int(np.argmax(region)) if region.size else min_lag
        period = max(min_lag, min(period, int(x.size)))
        grain = x[-period:]
        tiled = np.tile(grain, int(np.ceil(n / float(period)) + 2))
        out = tiled[:n].astype(np.float32, copy=True)
        ov = min(n, period // 2, int(x.size))
        if ov > 1:
            w = np.linspace(0.0, 1.0, ov, dtype=np.float32)
            out[:ov] = x[-ov:] * (1.0 - w) + out[:ov] * w
        streak = int(getattr(self, "_conceal_streak", 0)) + 1
        self._conceal_streak = streak
        out *= np.float32(min(0.9, max(0.12, 0.88 ** streak)))
        return out.reshape(-1, 1)

    def send_audio(self, indata: np.ndarray):
        """只发送，不接收。返回 (seq, n_in, t0) 供流水线 recv。"""
        if not self._connected or not self._ws:
            return None
        indata = np.asarray(indata, dtype=np.float32)
        if indata.ndim > 1:
            indata = indata[:, 0]
        if getattr(self, "capture_denoise", True) and getattr(self, "is_remote", False):
            try:
                indata = self._cap_ns.process(indata)
            except Exception:
                pass
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        seq = self._seq
        payload = struct.pack(">II", seq, len(indata)) + indata.tobytes()
        if not self._send_lock.acquire(timeout=0.2):
            return None
        try:
            if not self._ws or not self._connected:
                return None
            self._ws.send_binary(payload)
            return seq, len(indata), time.perf_counter()
        except Exception:
            self._connected = False
            self.last_error = "发送音频失败，连接已断开"
            return None
        finally:
            self._send_lock.release()

    def _audio_timeout(self):
        """2×块长 + 2×RTT，下限 120ms，上限 1.5s。"""
        sr = float(self.samplerate or 48000)
        block = (float(self._block_frame) / sr) if self._block_frame else 0.06
        if block <= 0:
            block = 0.06
        rtt = float(getattr(self, "_rtt_ema", 0.04) or 0.04)
        std = float(np.sqrt(max(0.0, getattr(self, "_rtt_m2", 0.0))))
        return min(1.5, max(0.12, 2.0 * block + 2.0 * rtt + 4.0 * std))

    def recv_audio(self, seq, n_in, t0, timeout=None):
        """接收与 seq 配对的音频响应。"""
        if timeout is None:
            timeout = self._audio_timeout()
        if not self._connected or not self._ws:
            return self._silence(n_in), 0
        deadline = time.time() + timeout
        if not self._recv_lock.acquire(timeout=timeout):
            return self._silence(n_in), 0
        try:
            cached = self._ahead.pop(seq, None)
            if cached is not None:
                response = cached
            else:
                response = None
            while True:
                if response is None:
                    remain = deadline - time.time()
                    if remain <= 0:
                        raise socket.timeout("audio response timeout")
                    self._ws.settimeout(min(remain, 0.5))
                    response = self._ws.recv()
                if isinstance(response, str) or len(response) < 8:
                    response = None
                    continue
                resp_seq = struct.unpack(">I", response[:4])[0]
                if resp_seq != seq:
                    delta = ((resp_seq - seq + 0x80000000) & 0xFFFFFFFF) - 0x80000000
                    if 0 < delta < 8:
                        self._ahead[resp_seq] = response
                        if len(self._ahead) > 8:
                            self._ahead.pop(next(iter(self._ahead)))
                    response = None
                    continue
                n = struct.unpack(">I", response[4:8])[0]
                if n == 0:
                    return self._silence(n_in), 0
                pcm_end = 8 + n * 4
                if len(response) < pcm_end:
                    continue
                out = np.frombuffer(response[8:pcm_end], dtype=np.float32).copy()
                extra = response[pcm_end:]
                if len(extra) >= 8:
                    feat, idx, pitch, model = struct.unpack(">HHHH", extra[:8])
                    self._last_stage_ms = {
                        "feature": feat, "index": idx, "pitch": pitch, "model": model,
                    }
                if self.channels > 1:
                    out = np.tile(out.reshape(-1, 1), (1, self.channels))
                else:
                    out = out.reshape(-1, 1)
                elapsed = int((time.perf_counter() - t0) * 1000)
                e2e = max(0.0, time.perf_counter() - t0)
                est_rtt = max(0.015, min(e2e * 0.5, max(0.015, e2e - 0.02)))
                prev = float(getattr(self, "_rtt_ema", est_rtt) or est_rtt)
                delta = est_rtt - prev
                self._rtt_ema = 0.85 * prev + 0.15 * est_rtt
                self._rtt_m2 = 0.85 * float(getattr(self, "_rtt_m2", 0.0)) + 0.15 * (delta * delta)
                mono = out[:, 0] if out.ndim > 1 else out
                self._last_good = np.asarray(mono, dtype=np.float32).reshape(-1).copy()
                streak = int(getattr(self, "_conceal_streak", 0) or 0)
                if streak > 0:
                    sr = float(self.samplerate or 48000)
                    fade_n = min(int(out.shape[0]), max(1, int(0.005 * sr)))
                    w = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
                    if out.ndim > 1:
                        out[:fade_n, 0] *= w
                    else:
                        out[:fade_n] *= w
                self._conceal_streak = 0
                return out, elapsed
        except _WS_TIMEOUT_EXC:
            out = self._silence(n_in)
            # 连续超时多半是服务挂了或链路死了，标断线让上层重连
            if int(getattr(self, "_conceal_streak", 0) or 0) >= 3:
                self._connected = False
                self.last_error = "服务器长时间无响应"
            return out, 0
        except Exception:
            self._connected = False
            self.last_error = "连接已断开"
            return self._silence(n_in), 0
        finally:
            self._recv_lock.release()

    def process_chunk(self, indata: np.ndarray):
        """兼容录音测试：发一块收一块。"""
        token = self.send_audio(indata)
        if token is None:
            n = len(indata) if indata is not None else self._block_frame
            return self._silence(n), 0
        return self.recv_audio(*token)

    def try_reconnect(self):
        if self._ws_alive():
            return True
        return self.connect(timeout=3)

    def configure(self, **kwargs):
        """参数配置：发了即走（Fire-and-Forget），绝不等待回包卡 UI。

        服务端对 configure 不再回包（与 set_live 一致），
        因此不存在残留响应污染后续请求的问题。
        不改 socket timeout：settimeout 会打断正在 recv 的音频线程。
        """
        if "capture_denoise" in kwargs:
            self.capture_denoise = bool(kwargs.get("capture_denoise"))
            kwargs = {k: v for k, v in kwargs.items() if k != "capture_denoise"}
            if getattr(self, "_cap_ns", None) is not None:
                self._cap_ns.enabled = bool(self.capture_denoise)
        if not kwargs:
            return True
        return self._send_json({"action": "configure", **kwargs}, wait=0.5)

    def _send_json(self, cmd, wait=0.05):
        if not self._ws or not self._connected:
            return False
        if not self._send_lock.acquire(timeout=wait):
            return False
        try:
            if not self._ws or not self._connected:
                return False
            self._ws.send(json.dumps(cmd))
            return True
        except Exception:
            return False
        finally:
            self._send_lock.release()

    def list_models(self):
        """从服务器获取模型文件列表（添加角色时选择用）"""
        data = self._rpc({"action": "list_models"})
        if isinstance(data, dict) and data.get("status") == "ok":
            return data.get("models", [])
        return []

    def list_indices(self):
        data = self._rpc({"action": "list_models"})
        if isinstance(data, dict) and data.get("status") == "ok":
            return data.get("indices", [])
        return []

    def _recv_json(self, timeout):
        """等一条 JSON 控制回复；跳过残留的二进制音频帧。"""
        deadline = time.time() + float(timeout)
        last_err = "服务器没有返回有效结果"
        while True:
            remain = deadline - time.time()
            if remain <= 0:
                raise socket.timeout(last_err)
            self._ws.settimeout(min(remain, 1.0))
            raw = self._ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                last_err = "收到残留音频，继续等待加载结果"
                continue
            text = (raw or "").strip() if isinstance(raw, str) else ""
            if not text:
                last_err = "服务器回复为空"
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                last_err = "服务器加载角色时没有返回有效结果"
                continue

    def _rpc(self, cmd, timeout=5.0):
        if not self._ws:
            if not self.connect(timeout=timeout):
                return None
        try:
            if not self._acquire_io(min(3.0, float(timeout))):
                return None
            try:
                self._ws.send(json.dumps(cmd))
                return self._recv_json(timeout)
            finally:
                self._release_io()
        except Exception:
            self._drain()
            return None

    def _drain(self):
        """清空 socket 里残留的消息（超时/失败后调用，防止响应错位到下一个请求）"""
        ws = self._ws
        if ws is None:
            return
        if not self._recv_lock.acquire(timeout=0.15):
            return
        try:
            ws.settimeout(0.01)
            while True:
                ws.recv()
        except Exception:
            pass
        finally:
            try:
                self._recv_lock.release()
            except Exception:
                pass

    def _send_live(self, **fields):
        cmd = {"action": "set_live"}
        cmd.update(fields)
        return self._send_json(cmd, wait=0.05)

    def change_pitch(self, val):
        return self._send_live(pitch=int(val))

    def change_index_rate(self, val):
        return self._send_live(index_rate=float(val))

    def change_formant(self, val):
        return self._send_live(formant=float(val))

    def set_server_url(self, url):
        url = (url or "").strip()
        if not url:
            return
        if url == self.server_url and self._connected:
            return
        if self._connected:
            self.abort()
        self.server_url = url
