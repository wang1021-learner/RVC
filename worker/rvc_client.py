"""
RVC 网络客户端 — 通过 WebSocket 连接推理服务器

替换本地 RVCPipeline，其他代码不变。
依赖: websocket-client, numpy
"""

import json, struct, time, threading, socket
import numpy as np
import websocket

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

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def is_active(self) -> bool:
        return self._active

    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_remote(self):
        return True

    @property
    def tgt_sr(self) -> int:
        return self.samplerate

    def connect(self, timeout=5):
        """建立 WebSocket 连接（已连接则直接返回）"""
        if self._connected and self._ws is not None:
            return True
        try:
            self._ws = websocket.create_connection(self.server_url, timeout=timeout)
            self._connected = True
            self._on_status(f"已连接服务器: {self.server_url}")
            # 查询服务器模型状态，同步本地标志（防服务器重启后状态错乱）
            try:
                with self._send_lock, self._recv_lock:
                    self._ws.settimeout(5.0)
                    self._ws.send(json.dumps({"action": "status"}))
                    resp = json.loads(self._ws.recv())
                    self._model_loaded = bool(resp.get("loaded", False))
                    self._active = bool(resp.get("active", False))
                    if resp.get("samplerate"):
                        self.samplerate = resp["samplerate"]
                    self._on_status("服务器模型就绪" if self._model_loaded else "服务器无模型")
            except Exception:
                self._model_loaded = False
                self._drain()
            return True
        except Exception as e:
            self._on_status(f"连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self._ws:
            try:
                with self._send_lock:
                    self._ws.send(json.dumps({"action": "stop"}))
            except Exception:
                pass
            with self._send_lock, self._recv_lock:
                try: self._ws.close()
                except Exception: pass
                self._ws = None
        self._connected = False

    def load_speaker(self, model_path: str, index_path: str = "",
                     pitch: int = 0, index_rate: float = 0.0,
                     formant: float = 0.0, **params) -> bool:
        """发送加载模型命令到服务器（未连接则自动重连）"""
        if not self._ws:
            if not self.connect(timeout=5):
                return False
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
        self._on_status("正在加载模型...")
        try:
            with self._send_lock, self._recv_lock:
                self._ws.settimeout(60.0)   # 模型加载可能 30s+
                self._ws.send(json.dumps(cmd))
                resp = json.loads(self._ws.recv())
            if "error" in resp:
                self._on_status(f"加载失败: {resp['error']}")
                return False
            self.samplerate = resp.get("samplerate", 48000)
            self.channels = resp.get("channels", 1)
            self._block_frame = resp.get("block_size", 256)
            self._model_loaded = True
            self._active = True   # 服务器 load 完成后会自动 start
            self._on_status(f"模型已加载 (sr={self.samplerate})")
            return True
        except Exception as e:
            self._on_status(f"加载失败: {e}")
            self._drain()   # 清掉可能残留的响应，防止污染下一个请求
            return False

    def start(self, **params):
        """通知服务器恢复推理（模型已加载时停止后重启用）"""
        if not self._connected:
            raise RuntimeError("未连接服务器")
        if not self._ws:
            return True
        try:
            cmd = {"action": "start"}
            cmd.update(params)
            with self._send_lock, self._recv_lock:
                # 改过参数时服务端要重新预热 CUDA Graph（10~20s），
                # 此调用在后台线程执行，耐心等 60s 不误报失败
                self._ws.settimeout(60.0)
                self._ws.send(json.dumps(cmd))
                resp = json.loads(self._ws.recv())
            if isinstance(resp, dict) and "error" in resp:
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
            return True
        except Exception:
            self._drain()
            return False

    def abort(self):
        """强制中断阻塞中的 recv/send（停止时调用，避免 UI 卡死）
        模型状态保留（服务器上模型还在），重连时由 connect() 状态同步确认"""
        ws = self._ws
        self._ws = None
        if ws:
            try: ws.close()
            except Exception: pass
        self._connected = False

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
        n = int(n) if n else (self._block_frame or 256)
        return np.zeros(n, dtype=np.float32).reshape(-1, 1)

    def send_audio(self, indata: np.ndarray):
        """只发送，不接收。返回 (seq, n_in, t0) 供流水线 recv。"""
        if not self._connected or not self._ws:
            return None
        indata = np.asarray(indata, dtype=np.float32)
        if indata.ndim > 1:
            indata = indata[:, 0]
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
            return None
        finally:
            self._send_lock.release()

    def recv_audio(self, seq, n_in, t0, timeout=1.5):
        """接收与 seq 配对的音频响应。"""
        if not self._connected or not self._ws:
            return self._silence(n_in), 0
        deadline = time.time() + timeout
        if not self._recv_lock.acquire(timeout=timeout):
            return self._silence(n_in), 0
        try:
            while True:
                remain = deadline - time.time()
                if remain <= 0:
                    raise socket.timeout("audio response timeout")
                self._ws.settimeout(min(remain, 0.5))
                response = self._ws.recv()
                if isinstance(response, str) or len(response) < 8:
                    continue
                resp_seq = struct.unpack(">I", response[:4])[0]
                if resp_seq != seq:
                    continue
                n = struct.unpack(">I", response[4:8])[0]
                out = np.frombuffer(response[8:8 + n * 4], dtype=np.float32)
                if self.channels > 1:
                    out = np.tile(out.reshape(-1, 1), (1, self.channels))
                else:
                    out = out.reshape(-1, 1)
                elapsed = int((time.perf_counter() - t0) * 1000)
                return out, elapsed
        except _WS_TIMEOUT_EXC:
            return self._silence(n_in), 0
        except Exception:
            self._connected = False
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
        if self._connected and self._ws is not None:
            return True
        return self.connect(timeout=3)

    def configure(self, **kwargs):
        """参数配置：发了即走（Fire-and-Forget），绝不等待回包卡 UI。

        服务端对 configure 不再回包（与 set_live 一致），
        因此不存在残留响应污染后续请求的问题。
        """
        if not self._ws or not self._connected:
            return False
        if not self._send_lock.acquire(timeout=0.2):
            return False
        try:
            cmd = {"action": "configure"}
            cmd.update(kwargs)
            self._ws.send(json.dumps(cmd))
            return True
        except Exception:
            return False
        finally:
            self._send_lock.release()

    def list_models(self):
        """从服务器获取模型文件列表（添加角色时选择用）"""
        if not self._ws:
            if not self.connect(timeout=5):
                return []
        try:
            with self._send_lock, self._recv_lock:
                self._ws.settimeout(5.0)
                self._ws.send(json.dumps({"action": "list_models"}))
                resp = json.loads(self._ws.recv())
            if isinstance(resp, dict) and resp.get("status") == "ok":
                return resp.get("models", [])
            return []
        except Exception:
            self._drain()
            return []

    def _drain(self):
        """清空 socket 里残留的消息（超时/失败后调用，防止响应错位到下一个请求）"""
        ws = self._ws
        if ws is None:
            return
        try:
            with self._recv_lock:
                ws.settimeout(0.01)
                while True:
                    ws.recv()
        except Exception:
            pass

    def _send_live(self, **fields):
        if not self._ws or not self._connected:
            return False
        if not self._send_lock.acquire(timeout=0.2):
            return False
        try:
            cmd = {"action": "set_live"}
            cmd.update(fields)
            self._ws.send(json.dumps(cmd))
            return True
        except Exception:
            return False
        finally:
            self._send_lock.release()

    def change_pitch(self, val):
        self._send_live(pitch=int(val))

    def change_index_rate(self, val):
        self._send_live(index_rate=float(val))

    def change_formant(self, val):
        self._send_live(formant=float(val))

    def set_server_url(self, url):
        url = (url or "").strip()
        if not url:
            return
        if url == self.server_url and self._connected:
            return
        self.disconnect()
        self.server_url = url
        self.connect(timeout=5)
