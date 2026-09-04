"""音频引擎与后台线程（不含 GPU 推理实现）。"""
import os, sys, json, queue, time, logging, threading
from pathlib import Path
import numpy as np
import sounddevice as sd

from PySide6.QtCore import Qt, Signal, QObject, QThread, QTimer

from worker.rvc_client import RVCClient
from worker.local_server import LocalServerPipeline
from tools.audio_process import AutoGain, PeakLimiter
from tools.audio_meter import calc_rms_db, spec_bins
from tools.virtual_cable import is_virtual_name, is_bluetooth_name
from ui.common import (
    NL, DEFAULT_SERVER_URL, DEFAULT_PARAMS, RESTART_KEYS,
    VIRTUAL_OUT_MIN_BLOCK, _split_hw_frames, _wasapi_fail_reason,
    _friendly_error, to_server_path,
)

def make_pipeline(mode, server_url, on_status):
    if mode == "local":
        # 本地推理也统一走子进程拉起本机 server，客户端不直接 import torch/infer。
        return LocalServerPipeline(on_status=on_status)
    # 不在构造时 connect：远程未启动会卡住 UI 数秒
    return RVCClient(server_url=server_url, on_status=on_status)


# ==============================================================================
# 后台模型加载线程 - 避免界面卡死
# ==============================================================================
class ModelLoader(QThread):
    finished_ok = Signal(object, int)   # speaker, 加载代数
    failed = Signal(str, int)

    def __init__(self, engine, speaker, gen, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.speaker = speaker
        self.gen = gen
        self.pipeline = engine.pipeline

    def run(self):
        try:
            mp, ip = self.engine.resolve_paths(self.speaker)
            params = self.engine.merged_params(self.speaker)
            ok = self.pipeline.load_speaker(
                mp, ip,
                self.speaker.pitch, self.speaker.index_rate,
                formant=self.speaker.formant,
                **params)
            if ok:
                self.finished_ok.emit(self.speaker, self.gen)
            else:
                err = getattr(self.pipeline, "last_error", "") or "模型加载失败"
                self.failed.emit(err, self.gen)
        except Exception as e:
            self.failed.emit(str(e), self.gen)

# ==============================================================================
# 异步推理工作线程 - 彻底解耦音频 I/O 与 PyTorch 深度学习计算
# ==============================================================================
class InferenceWorkerThread(QThread):
    infer_done = Signal(int, float, float)  # elapsed_ms, in_rms_db, out_rms_db
    stage_stats = Signal(dict)              # 分阶段耗时 {feature,index,pitch,model}
    spectrum = Signal(object)
    xrun_occurred = Signal()
    need_recover = Signal()
    crashed = Signal(str)

    def __init__(self, pipeline, input_queue, output_queue, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = False

    def _emit_out(self, out_block, elapsed_ms, in_rms):
        cap = 3 if getattr(self.pipeline, "is_remote", False) else 4
        dropped = False
        while self.output_queue.qsize() >= cap:
            try:
                self.output_queue.get_nowait()
                dropped = True
            except queue.Empty:
                break
        try:
            self.output_queue.put_nowait(out_block)
        except queue.Full:
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(out_block)
            except queue.Full:
                pass
            dropped = True
        if dropped:
            self.xrun_occurred.emit()
        now = time.perf_counter()
        # 电平/频谱跟手一点；文字和样式仍少刷，避免窗口发黏
        if now - getattr(self, "_meter_last", 0.0) >= 0.04:
            self._meter_last = now
            self.infer_done.emit(elapsed_ms, in_rms, calc_rms_db(out_block))
            try:
                self.spectrum.emit(spec_bins(out_block))
            except Exception:
                pass
        if now - getattr(self, "_ui_last", 0.0) >= 0.12:
            self._ui_last = now
            stage = getattr(self.pipeline, "last_stage_ms", None)
            if stage:
                self.stage_stats.emit(dict(stage))

    def run(self):
        from collections import deque
        self.running = True
        inflight = deque()
        try:
            while self.running:
                try:
                    if hasattr(self.pipeline, "inflight_depth"):
                        depth = int(self.pipeline.inflight_depth() or 1)
                    else:
                        depth = 2 if getattr(self.pipeline, "is_remote", False) else 1
                except Exception:
                    depth = 2 if getattr(self.pipeline, "is_remote", False) else 1
                depth = max(1, min(3, depth))
                while self.running and len(inflight) < depth:
                    try:
                        indata = self.input_queue.get(timeout=0.01 if inflight else 0.05)
                    except queue.Empty:
                        break
                    token = self.pipeline.send_audio(indata)
                    if token is None:
                        if not self.pipeline.is_connected():
                            self.need_recover.emit()
                        break
                    inflight.append((token, calc_rms_db(indata)))
                if not inflight:
                    continue
                token, in_rms = inflight.popleft()
                out_block, elapsed_ms = self.pipeline.recv_audio(*token)
                if not self.pipeline.is_connected():
                    self.need_recover.emit()
                if not self.running:
                    break
                self._emit_out(out_block, elapsed_ms, in_rms)
        except Exception as e:
            logging.exception("推理线程异常")
            if self.running:
                self.crashed.emit(str(e)[:200])

    def stop(self):
        self.running = False
        if self.isRunning():
            self.quit()
            self.wait(1800)


# 录音测试暂时关闭
# class RecThread(QThread):
#     """录音测试线程：录 N 秒 → 用当前角色变声 → 保存 wav"""
#     done = Signal(str, str)   # 保存路径, 错误信息
#
#     def __init__(self, engine, seconds=10, device=None, parent=None):
#         super().__init__(parent)
#         self.engine = engine
#         self.seconds = seconds
#         self.device = device
#
#     def run(self):
#         import sounddevice as sd
#         import numpy as np
#         import time, os
#         try:
#             c = self.engine.pipeline
#             if c is None:
#                 self.done.emit("", "推理引擎未就绪")
#                 return
#             if not c.is_connected():
#                 if not c.connect(timeout=5):
#                     self.done.emit("", "无法连接服务器")
#                     return
#             try:
#                 started = c.start(**self.engine.merged_params())
#                 if started is False:
#                     self.done.emit("", "无法启动推理")
#                     return
#             except Exception as e:
#                 self.done.emit("", "无法启动推理: " + str(e))
#                 return
#             SR = c.samplerate
#             BLOCK = getattr(c, "_block_frame", None)
#             if not BLOCK:
#                 self.done.emit("", "推理块大小未知，请先成功加载角色")
#                 return
#             frames = []
#             rec_log = []
#             def cb(indata, frames_, times, status):
#                 frames.append(indata[:, 0].copy())
#             kwargs = dict(samplerate=SR, channels=1, dtype='float32',
#                           blocksize=BLOCK, callback=cb)
#             if self.device is not None:
#                 kwargs['device'] = self.device
#             with sd.InputStream(**kwargs):
#                 time.sleep(self.seconds)
#             if not frames:
#                 self.done.emit("", "没有录到音频")
#                 return
#             raw = np.concatenate(frames)
#             outs = []
#             for i in range(0, len(raw) - BLOCK + 1, BLOCK):
#                 out, _ = c.process_chunk(raw[i:i + BLOCK])
#                 outs.append(np.asarray(out).reshape(-1))
#             if not outs:
#                 self.done.emit("", "录音太短")
#                 return
#             os.makedirs('record_out', exist_ok=True)
#             out = np.concatenate(outs)
#             spk = self.engine.current_speaker
#             fname = os.path.join('record_out', f'rec_{spk.name}_{int(time.time())}.wav')
#             pcm = np.clip(out * 32767, -32768, 32767).astype(np.int16)
#             with wave_mod.open(fname, 'wb') as wf:
#                 wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
#                 wf.writeframes(pcm.tobytes())
#             self.done.emit(fname, "")
#         except Exception as e:
#             self.done.emit("", str(e))


class EngineStartThread(QThread):
    """后台执行启动/重建/停机的阻塞部分（网络等待、CUDA 预热、声卡打开、等推理退出）。

    UI 线程只负责发起与信号响应，绝不进入 wait——杜绝"未响应"。
    """

    def __init__(self, engine, action, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.action = action

    def run(self):
        try:
            if self.action == "start":
                self.engine._start_blocking()
            elif self.action == "reopen":
                self.engine._reopen_blocking()
            elif self.action == "stop":
                self.engine._hard_stop()
            elif self.action == "recover":
                self.engine._recover_blocking()
            elif self.action == "monitor":
                ok, msg = self.engine._open_monitor()
                if not ok:
                    self.engine.status_msg.emit("监听开启失败: " + msg)
                elif msg:
                    self.engine.status_msg.emit(msg)
        except Exception as e:
            logging.exception("引擎线程失败: %s", self.action)
            if self.action == "start":
                self.engine.load_failed.emit(_friendly_error(e))
            else:
                self.engine.status_msg.emit("操作失败: " + _friendly_error(e))


class ServerConnectThread(QThread):
    """后台连接远程推理服务器，避免点「连接」卡死 UI。"""
    done = Signal(bool, str)

    def __init__(self, pipeline, url, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.url = url

    def run(self):
        try:
            self.pipeline.set_server_url(self.url)
            ok = bool(self.pipeline.connect(timeout=8))
            extra = ""
            if ok:
                gpu = getattr(self.pipeline, "gpu_name", "") or ""
                info = {}
                try:
                    info = self.pipeline.loaded_file_info() or {}
                except Exception:
                    info = {}
                model = info.get("model_path") or ""
                idx = info.get("index_path") or ""
                idx_bit = ""
                if idx:
                    idx_bit = "索引 " + Path(str(idx)).name
                    if not info.get("index_loaded", True):
                        idx_bit += "（未启用）"
                elif model:
                    idx_bit = "无索引"
                extra = " · ".join(
                    x for x in (
                        gpu,
                        Path(str(model)).name if model else "等待加载角色",
                        idx_bit,
                    ) if x)
            else:
                extra = (
                    getattr(self.pipeline, "last_error", "")
                    or "连不上推理服务。请确认那台机器已启动服务，地址端口正确，防火墙放行 8765。"
                )
            self.done.emit(ok, extra)
        except Exception as e:
            self.done.emit(False, str(e)[:220])


# ==============================================================================
# 推理引擎（本地）
# ==============================================================================
class VCEngine(QObject):
    status_msg = Signal(str); infer_time = Signal(int)
    started_ok = Signal(); stopped_ok = Signal()
    load_failed = Signal(str)
    recover_progress = Signal(int, int)  # 当前次数, 最多次数
    recover_ok = Signal()
    recover_failed = Signal(str)
    rms_levels = Signal(float, float)  # in_db, out_db
    spectrum = Signal(object)
    xrun_signal = Signal(int)         # total_xruns
    fade_done = Signal()
    loop_latency = Signal(float)      # 端到端延迟(ms)：output DAC time - input ADC time
    stage_stats = Signal(dict)        # 分阶段耗时（服务器回包也会带）

    def __init__(self, mode="local", server_url=DEFAULT_SERVER_URL):
        super().__init__()
        self.mode = "local" if mode == "local" else "server"
        self.server_url = server_url or DEFAULT_SERVER_URL
        self.pipeline = make_pipeline(
            self.mode, self.server_url, lambda m: self.status_msg.emit(m))
        self.stream = None
        self._out_stream = None
        self._last_cap = np.zeros(0, dtype=np.float32)
        self.running = False
        self.current_speaker = None
        self.input_device = None; self.output_device = None
        self.input_queue = queue.Queue(maxsize=4)
        self.output_queue = queue.Queue(maxsize=4)
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._in_n = 0
        self._out_buf = np.zeros(0, dtype=np.float32)
        self._out_n = 0
        self._in_pool = []
        self._pool_i = 0
        self._last_out = np.zeros(0, dtype=np.float32)
        self._last_out_n = 0
        self._hold_count = 0
        self._xrun_fade = np.zeros(0, dtype=np.float32)
        self._mon_scratch = np.zeros(0, dtype=np.float32)
        self._virtual_out = False
        self._bt_in = False
        self._asrc_on = False
        self._asrc_ready = False
        self._asrc_pos = 0.0
        self._asrc_step = 1.0
        self._asrc_int = 0.0
        self._asrc_ramp = np.zeros(0, dtype=np.float32)
        self._asrc_idx = np.zeros(0, dtype=np.float32)
        self._recover_fade = np.zeros(0, dtype=np.float32)
        self._agc_held_off = False
        self._ns_held_off = False
        self._deess_held_off = False
        self._vad_held_off = False
        self._index_bumped = False
        self.worker_thread = None
        self._zombie_threads = []   # 超时未退出的线程，等 finished 再删，避免销毁运行中的线程
        self.xrun_count = 0
        self._xrun_emitted = 0
        self._params = dict(DEFAULT_PARAMS)
        self._formant = 0.0
        self.dry_mix = 0.0
        self.bypass = False
        self._fade_in_left = 0
        self._fade_out_left = 0
        self._fade_out_total = 0
        self._last_recover = 0.0
        self._fade_epoch = 0
        self._pending_fade_epoch = -1
        self._stop_requested = False
        self._emit_stopped = True
        self._fade_done_flag = False
        self._true_e2e_ms = 0.0
        self._slow_streak = 0
        self._adapted = False
        # 输入 AGC / 监听混音 / 端到端延迟
        self.input_agc = False
        self.agc = None
        self._peak_lim = None
        self.monitor_enabled = False
        self.monitor_device = None
        self.monitor_volume = 0.8
        self.monitor_stream = None
        self.monitor_queue = queue.Queue(maxsize=8)
        self._loop_lat_ema = None
        self.fade_done.connect(self._on_fade_done, Qt.QueuedConnection)
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(50)
        self._stats_timer.timeout.connect(self._flush_callback_stats)
        self._stats_timer.start()
        # 滑条/参数只投递最新值，由后台线程发到管线，UI 绝不占 websocket/Faiss 锁
        self._live_lock = threading.Lock()
        self._live_event = threading.Event()
        self._live_stop = False
        self._live_pitch = None
        self._live_index = None
        self._live_formant = None
        self._live_cfg = {}
        self._live_thread = threading.Thread(
            target=self._live_pump, name="rvc-live-params", daemon=True)
        self._live_thread.start()

    def merged_params(self, speaker=None):
        """全局高级参数 + 指定角色的角色级覆盖（默认当前角色）。"""
        params = dict(self._params)
        spk = speaker if speaker is not None else self.current_speaker
        if spk is not None:
            try:
                params.update(spk.pipeline_overrides())
            except Exception:
                pass
        return params

    def _drain_queue(self, q):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _enqueue_input(self, chunk):
        cap = 2 if getattr(self.pipeline, "is_remote", False) else 4
        dropped = False
        while self.input_queue.qsize() >= cap:
            try:
                self.input_queue.get_nowait()
                dropped = True
            except queue.Empty:
                break
        try:
            self.input_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.input_queue.put_nowait(chunk)
            except queue.Full:
                pass
            dropped = True
        if dropped:
            self._on_worker_xrun()

    def _prop(name):
        def setter(s, v):
            s._params[name] = v
            if name in RESTART_KEYS:
                if s.running:
                    s.status_msg.emit("该参数将在下次启动后生效")
                return
            if s.mode == "server":
                pipe = getattr(s, "pipeline", None)
                if pipe is None or not pipe.is_connected():
                    return
            try:
                s._queue_configure({name: v})
            except Exception:
                pass
        return property(lambda s: s._params[name], setter)
    block_time = _prop("block_time"); crossfade_time = _prop("crossfade_time")
    extra_time = _prop("extra_time"); f0method = _prop("f0method")
    I_noise_reduce = _prop("I_noise_reduce"); O_noise_reduce = _prop("O_noise_reduce")
    rms_mix_rate = _prop("rms_mix_rate"); threhold = _prop("threhold")
    limiter_enable = _prop("limiter_enable")
    limiter_threshold_db = _prop("limiter_threshold_db")
    hf_mix_rate = _prop("hf_mix_rate")
    presence = _prop("presence")
    deesser_enable = _prop("deesser_enable")
    vad_enable = _prop("vad_enable")
    vad_threshold = _prop("vad_threshold")
    incremental_hubert = _prop("incremental_hubert")
    capture_denoise = _prop("capture_denoise")
    protect = _prop("protect")

    def _queue_configure(self, kwargs):
        with self._live_lock:
            self._live_cfg.update(kwargs)
        self._live_event.set()

    def change_pitch(self, val):
        with self._live_lock:
            self._live_pitch = int(val)
        self._live_event.set()

    def change_index_rate(self, val):
        with self._live_lock:
            self._live_index = float(val)
        self._live_event.set()

    def change_formant(self, val):
        self._formant = float(val)
        with self._live_lock:
            self._live_formant = float(val)
        self._live_event.set()

    def _live_pump(self):
        while not self._live_stop:
            self._live_event.wait(timeout=0.5)
            if self._live_stop:
                break
            if not self._live_event.is_set():
                continue
            time.sleep(0.04)
            with self._live_lock:
                pitch = self._live_pitch
                index = self._live_index
                formant = self._live_formant
                cfg = dict(self._live_cfg) if self._live_cfg else None
                self._live_pitch = None
                self._live_index = None
                self._live_formant = None
                self._live_cfg.clear()
                self._live_event.clear()
            pipe = getattr(self, "pipeline", None)
            if pipe is None:
                continue
            failed_cfg, failed_pitch, failed_index, failed_formant = None, None, None, None
            try:
                if cfg and pipe.configure(**cfg) is False:
                    failed_cfg = cfg
                if pitch is not None:
                    try:
                        r = pipe.change_pitch(pitch)
                        if r is False:
                            failed_pitch = pitch
                    except Exception:
                        failed_pitch = pitch
                if index is not None:
                    try:
                        r = pipe.change_index_rate(index)
                        if r is False:
                            failed_index = index
                    except Exception:
                        failed_index = index
                if formant is not None:
                    try:
                        r = pipe.change_formant(formant)
                        if r is False:
                            failed_formant = formant
                    except Exception:
                        failed_formant = formant
            except Exception:
                pass
            if failed_cfg or failed_pitch is not None or failed_index is not None or failed_formant is not None:
                connected = True
                try:
                    connected = bool(pipe.is_connected())
                except Exception:
                    connected = False
                if connected:
                    time.sleep(0.12)
                    with self._live_lock:
                        if failed_cfg and not self._live_cfg:
                            self._live_cfg.update(failed_cfg)
                        if failed_pitch is not None and self._live_pitch is None:
                            self._live_pitch = failed_pitch
                        if failed_index is not None and self._live_index is None:
                            self._live_index = failed_index
                        if failed_formant is not None and self._live_formant is None:
                            self._live_formant = failed_formant
                        self._live_event.set()

    def set_dry_mix(self, val):
        self.dry_mix = float(val)

    def set_bypass(self, on):
        self.bypass = bool(on)

    def set_input_agc(self, on):
        self.input_agc = bool(on)

    def set_monitor(self, on, device, volume):
        """监听混音：第二输出设备播放变声结果（耳机监听）。

        仅开关/设备变化时重建流；音量在推入监听队列时实时生效。
        """
        need_rebuild = bool(on) != self.monitor_enabled or device != self.monitor_device
        self.monitor_enabled = bool(on)
        self.monitor_device = device
        self.monitor_volume = float(volume)
        if self.running and need_rebuild:
            if self._engine_busy():
                self.status_msg.emit("正在处理，请稍候...")
                return
            self._start_thread = EngineStartThread(self, "monitor", parent=self)
            self._start_thread.start()

    def set_server_url(self, url):
        self.server_url = (url or "").strip() or DEFAULT_SERVER_URL
        if self.mode == "server":
            self.pipeline.set_server_url(self.server_url)

    def _dispose_pipeline(self):
        pipe = self.pipeline
        self.pipeline = None
        if pipe is None:
            return
        def _bg(p=pipe):
            try:
                p.stop()
            except Exception:
                pass
            if getattr(p, "is_remote", False):
                try:
                    p.disconnect()
                except Exception:
                    pass
            else:
                try:
                    p.unload()
                except Exception:
                    pass
        threading.Thread(target=_bg, daemon=True, name="rvc-dispose-pipe").start()

    def set_mode(self, mode):
        mode = "local" if mode == "local" else "server"
        if mode == self.mode:
            return
        if self.running or self.stream is not None or self._engine_busy():
            self.request_hard_stop()
            return
        self._dispose_pipeline()
        self.mode = mode
        self.current_speaker = None
        self.pipeline = make_pipeline(
            self.mode, self.server_url, lambda m: self.status_msg.emit(m))
        self.status_msg.emit("已切换到" + ("本地推理" if mode == "local" else "服务器"))

    def resolve_paths(self, speaker):
        if self.mode == "server":
            return (
                to_server_path(speaker.model_path),
                to_server_path(speaker.index_path) if speaker.index_path else "",
            )
        # 本地模式（直连或本机子进程）：路径原样透传。
        # 绝对路径由服务端直接打开；只有文件名时由服务端在自己的目录解析。
        return speaker.model_path or "", speaker.index_path or ""

    def _ensure_connected(self):
        if self.mode == "local" or self.pipeline.is_connected():
            return True
        self.status_msg.emit("正在连接服务器...")
        if not self.pipeline.connect(timeout=8):
            err = getattr(self.pipeline, "last_error", "") or "连接服务器失败"
            self.status_msg.emit(err)
            return False
        return True

    def _ensure_model(self):
        if self.pipeline.is_loaded:
            return True
        if self.current_speaker is None:
            self.status_msg.emit("请先加载角色模型")
            return False
        self.status_msg.emit("恢复模型加载...")
        mp, ip = self.resolve_paths(self.current_speaker)
        ok = self.pipeline.load_speaker(
            mp, ip,
            self.current_speaker.pitch, self.current_speaker.index_rate,
            formant=self.current_speaker.formant, **self.merged_params())
        if not ok:
            self.status_msg.emit("请先加载角色模型")
        return ok

    def _engine_busy(self):
        t = getattr(self, "_start_thread", None)
        return t is not None and t.isRunning()

    def start(self):
        """异步启动：网络等待 / CUDA 预热 / 声卡打开全部在后台线程，UI 永不阻塞。"""
        self._fade_epoch += 1
        self._stop_requested = False
        if self._engine_busy():
            self.status_msg.emit("正在处理，请稍候...")
            return
        self._start_thread = EngineStartThread(self, "start", parent=self)
        self._start_thread.start()

    def _alloc_rt_buffers(self, block):
        """音频回调用的预分配缓冲：回调内不再 numpy 拼接/新建数组。"""
        cap = max(int(block) * 8, 2048)
        self._in_buf = np.zeros(cap, dtype=np.float32)
        self._in_n = 0
        self._out_buf = np.zeros(cap, dtype=np.float32)
        self._out_n = 0
        # 槽位多于队列深度，避免回调复用工人尚未读完的数组
        self._in_pool = [np.zeros(block, dtype=np.float32) for _ in range(32)]
        self._pool_i = 0
        self._last_out = np.zeros(max(block, 1), dtype=np.float32)
        self._last_out_n = 0
        self._hold_count = 0
        sr = int(getattr(self.pipeline, "samplerate", 0) or 48000)
        fade_n = max(8, int(0.008 * sr))
        self._xrun_fade = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
        rec_n = max(8, int(0.005 * sr))
        self._recover_fade = np.linspace(0.0, 1.0, rec_n, dtype=np.float32)
        self._mon_scratch = np.zeros(max(block, 1), dtype=np.float32)
        hw = _split_hw_frames(sr)
        ramp_n = max(int(block), int(hw), 1024) + 64
        self._asrc_ramp = np.arange(ramp_n, dtype=np.float32)
        self._asrc_idx = np.zeros(ramp_n, dtype=np.float32)
        self._asrc_pos = 0.0
        self._asrc_step = 1.0
        self._asrc_int = 0.0
        self._asrc_ready = False

    def _start_blocking(self):
        if self.stream is not None or self.running:
            self._emit_stopped = False
            try:
                self._hard_stop()
            finally:
                self._emit_stopped = True
        if self._stop_requested:
            self._hard_stop()
            return
        if not self._ensure_connected():
            if self._stop_requested:
                self._hard_stop()
                return
            err = getattr(self.pipeline, "last_error", "") or "无法连接服务器"
            self.status_msg.emit("无法启动: " + err)
            self.load_failed.emit(err)
            return
        if not self._ensure_model():
            if self._stop_requested:
                self._hard_stop()
                return
            err = getattr(self.pipeline, "last_error", "") or "模型未加载"
            self.status_msg.emit("无法启动: " + err)
            self.load_failed.emit(err)
            return
        try:
            if self._stop_requested:
                self._hard_stop()
                return
            params = self._prepare_call_path(self.merged_params())
            self.pipeline.configure(**params)
            started = self.pipeline.start(**params)
            if started is False:
                if self._stop_requested:
                    self._hard_stop()
                    return
                err = getattr(self.pipeline, "last_error", "") or "推理未能启动"
                raise RuntimeError(err)
            if self._stop_requested:
                self._hard_stop()
                return
            self._alloc_rt_buffers(int(getattr(self.pipeline, "_block_frame", 0) or 4800))
            self.xrun_count = 0
            self._xrun_emitted = 0
            self._true_e2e_ms = 0.0
            self._loop_lat_ema = None
            self._slow_streak = 0
            self._adapted = False
            self._drain_queue(self.input_queue)
            self._drain_queue(self.output_queue)

            # 输入 AGC（按当前采样率重建，静音/增益状态清零，动态绑定当前静音门限）
            if self.input_agc and not getattr(self, "_agc_held_off", False):
                self.agc = AutoGain(sample_rate=self.pipeline.samplerate, gate_db=float(self.threhold))
            else:
                self.agc = None
            self._peak_lim = PeakLimiter(sample_rate=self.pipeline.samplerate)

            # 不能把 self(VCEngine, 主线程对象) 当 parent 传给跨线程创建的 QThread，
            # 否则 Qt 报「Cannot create children for a parent in a different thread」。
            # 线程生命周期已由 _hard_stop 的 wait/deleteLater 手动管理，无需 parent。
            self.worker_thread = InferenceWorkerThread(
                self.pipeline, self.input_queue, self.output_queue)
            self.worker_thread.infer_done.connect(self._on_worker_infer_done)
            self.worker_thread.stage_stats.connect(self.stage_stats)
            self.worker_thread.spectrum.connect(self.spectrum)
            self.worker_thread.xrun_occurred.connect(self._on_worker_xrun)
            self.worker_thread.need_recover.connect(
                self._try_recover, Qt.QueuedConnection)
            self.worker_thread.crashed.connect(
                self._on_worker_crash, Qt.QueuedConnection)
            self.worker_thread.start()

            self.running = True
            ok, msg = self._open_stream()
            if not ok:
                self._hard_stop()
                if self._stop_requested or msg == "已取消":
                    self.status_msg.emit("已取消启动")
                    return
                self.status_msg.emit("启动失败: " + msg)
                self.load_failed.emit(msg)
                return
            self._fade_in_left = int(0.04 * self.pipeline.samplerate)
            self._fade_out_left = 0
            if self._stop_requested:
                self._hard_stop()
                return
            self.stream.start()
            if self._out_stream is not None:
                self._out_stream.start()
            self._open_monitor()
            if self._stop_requested:
                self._hard_stop()
                return
            self.started_ok.emit()
            self.status_msg.emit("实时转换已启动 · " + msg)
        except Exception as e:
            if self._stop_requested:
                try:
                    self._hard_stop()
                except Exception:
                    pass
                self.status_msg.emit("已取消启动")
                return
            self.status_msg.emit("启动失败: " + str(e))
            self.load_failed.emit(str(e))

    def _close_stream_only(self):
        for name in ("stream", "_out_stream"):
            st = getattr(self, name, None)
            if st is None:
                continue
            try:
                st.abort()
                st.close()
            except Exception:
                pass
            setattr(self, name, None)

    def _dev_name(self, idx):
        try:
            if idx is None:
                return ""
            return str(sd.query_devices(idx).get("name") or "")
        except Exception:
            return ""

    def _out_is_virtual(self):
        return is_virtual_name(self._dev_name(self.output_device))

    def _in_is_bluetooth(self):
        return is_bluetooth_name(self._dev_name(self.input_device))

    def _prepare_call_path(self, params):
        """给别人听时稳住块长；同时关掉会把声音做糊的处理。"""
        params = dict(params)
        virtual_out = self._out_is_virtual()
        bt_in = self._in_is_bluetooth()
        self._virtual_out = bool(virtual_out)
        self._bt_in = bool(bt_in)
        if hasattr(self.pipeline, "_virtual_out"):
            self.pipeline._virtual_out = bool(virtual_out)
        notes = []
        bt = float(params.get("block_time") or 0.06)
        if virtual_out and bt < VIRTUAL_OUT_MIN_BLOCK:
            params["block_time"] = VIRTUAL_OUT_MIN_BLOCK
            self._params["block_time"] = VIRTUAL_OUT_MIN_BLOCK
            notes.append("块长提到 80ms")
        self._agc_held_off = bool((virtual_out or bt_in) and self.input_agc)
        if self._agc_held_off:
            notes.append("已关输入自动增益")
        self._ns_held_off = False
        if virtual_out or bt_in:
            params["capture_denoise"] = False
            if hasattr(self.pipeline, "capture_denoise"):
                self.pipeline.capture_denoise = False
            cap_ns = getattr(self.pipeline, "_cap_ns", None)
            if cap_ns is not None:
                cap_ns.enabled = False
            if bool(self._params.get("capture_denoise", False)):
                self._ns_held_off = True
                notes.append("已关采集降噪")

        # 发糊：去齿音会砍高频，人声识别会切字头，短窗特征更糊
        self._deess_held_off = bool(params.get("deesser_enable"))
        if self._deess_held_off:
            params["deesser_enable"] = False
            self._params["deesser_enable"] = False
            notes.append("已关去齿音")
        self._vad_held_off = bool(params.get("vad_enable"))
        if self._vad_held_off:
            params["vad_enable"] = False
            self._params["vad_enable"] = False
            notes.append("已关人声识别")
        if bool(params.get("incremental_hubert", False)):
            notes.append("已关短窗特征")
        params["incremental_hubert"] = False
        self._params["incremental_hubert"] = False
        if float(params.get("hf_mix_rate") or 0) <= 0:
            params["hf_mix_rate"] = 0.25
            self._params["hf_mix_rate"] = 0.25
            notes.append("已开齿音保留")
        if float(params.get("presence") or 0) <= 0:
            params["presence"] = 0.15
            self._params["presence"] = 0.15
            notes.append("已开临场感")
        self._index_bumped = False
        spk = self.current_speaker
        if (spk is not None and str(getattr(spk, "index_path", "") or "").strip()
                and float(getattr(spk, "index_rate", 0) or 0) <= 0.05):
            spk.index_rate = 0.5
            try:
                self.change_index_rate(0.5)
            except Exception:
                pass
            self._index_bumped = True
            notes.append("检索提到 0.5")

        if bt_in:
            notes.append("蓝牙麦延迟大，建议换电脑麦或有线耳机麦")
        if notes:
            prefix = "给别人听：" if virtual_out else "音质："
            self.status_msg.emit(prefix + "，".join(notes))
        return params

    def _wasapi_shared(self):
        try:
            return sd.WasapiSettings(exclusive=False, auto_convert=True)
        except TypeError:
            return sd.WasapiSettings(exclusive=False)

    def _open_stream(self):
        block = getattr(self.pipeline, "_block_frame", None)
        if not block:
            raise RuntimeError("推理块大小未知，请先加载角色再启动")
        virtual_out = self._out_is_virtual()
        bt_in = self._in_is_bluetooth()
        self._virtual_out = bool(virtual_out)
        self._bt_in = bool(bt_in)
        if hasattr(self.pipeline, "_virtual_out"):
            self.pipeline._virtual_out = bool(virtual_out)
        split = self.input_device != self.output_device
        if split:
            return self._open_split_streams(int(block), virtual_out)
        self._asrc_on = False
        kwargs = dict(
            callback=self._on_audio,
            blocksize=block,
            samplerate=self.pipeline.samplerate,
            channels=self.pipeline.channels,
            dtype="float32",
            device=(self.input_device, self.output_device),
            latency="high" if (virtual_out or bt_in) else "low",
        )
        errors = []
        if self._stop_requested:
            return False, "已取消"

        is_asio = False
        try:
            in_info = sd.query_devices(self.input_device) if self.input_device is not None else None
            if in_info:
                api_idx = in_info.get("hostapi", -1)
                apis = sd.query_hostapis()
                api_name = apis[api_idx]["name"].upper() if 0 <= api_idx < len(apis) else ""
                if "ASIO" in api_name:
                    is_asio = True
        except Exception:
            pass

        if is_asio:
            try:
                self.stream = sd.Stream(**kwargs)
                return True, "专业声卡直通（极低延迟）"
            except Exception as e:
                errors.append(e)

        if self._stop_requested:
            return False, "已取消"

        if not virtual_out:
            try:
                self.stream = sd.Stream(
                    extra_settings=sd.WasapiSettings(exclusive=True), **kwargs)
                return True, "系统低延迟独占"
            except Exception as e:
                errors.append(e)

        if self._stop_requested:
            return False, "已取消"

        try:
            extra = self._wasapi_shared()
            self.stream = sd.Stream(extra_settings=extra, **kwargs)
            why = "虚拟声卡共享" if virtual_out else (
                "系统共享（独占失败: %s）" % _wasapi_fail_reason(errors[-1]) if errors else "系统共享"
            )
            return True, why
        except Exception as e:
            errors.append(e)

        try:
            self.stream = sd.Stream(**kwargs)
            return True, "系统共享（%s）" % _wasapi_fail_reason(errors[-1])
        except Exception as e:
            errors.append(e)

        try:
            kwargs["channels"] = 2
            self.stream = sd.Stream(**kwargs)
            return True, "系统共享 · 双声道"
        except Exception as e:
            return False, _wasapi_fail_reason(e)

    def _open_split_streams(self, block, virtual_out):
        """麦和输出不是同一设备：拆开采集/播放，两套时钟用短回调 + ASRC 对钟。"""
        sr = self.pipeline.samplerate
        ch = self.pipeline.channels
        hw = _split_hw_frames(sr)
        in_lat = "low"
        out_lat = "high" if virtual_out else "low"
        in_kw = dict(
            blocksize=hw, samplerate=sr, channels=ch, dtype="float32",
            latency=in_lat, callback=self._on_capture, device=self.input_device,
        )
        out_kw = dict(
            blocksize=hw, samplerate=sr, channels=ch, dtype="float32",
            latency=out_lat, callback=self._on_playback, device=self.output_device,
        )
        errors = []

        def try_pair(in_extra, out_extra, label):
            ik, okw = dict(in_kw), dict(out_kw)
            if in_extra is not None:
                ik["extra_settings"] = in_extra
            if out_extra is not None:
                okw["extra_settings"] = out_extra
            ins = sd.InputStream(**ik)
            try:
                outs = sd.OutputStream(**okw)
            except Exception:
                try:
                    ins.close()
                except Exception:
                    pass
                raise
            self.stream = ins
            self._out_stream = outs
            self._asrc_on = True
            self._asrc_ready = False
            self._asrc_pos = 0.0
            self._asrc_step = 1.0
            self._asrc_int = 0.0
            return True, label

        if not virtual_out:
            try:
                return try_pair(
                    sd.WasapiSettings(exclusive=True),
                    sd.WasapiSettings(exclusive=True),
                    "分路独占（麦/输出分开）",
                )
            except Exception as e:
                errors.append(e)
        try:
            return try_pair(
                sd.WasapiSettings(exclusive=True),
                self._wasapi_shared(),
                "麦独占 · 输出共享",
            )
        except Exception as e:
            errors.append(e)
        try:
            return try_pair(self._wasapi_shared(), self._wasapi_shared(), "分路共享")
        except Exception as e:
            errors.append(e)
        try:
            return try_pair(None, None, "分路默认")
        except Exception as e:
            return False, _wasapi_fail_reason(e)

    def reopen_stream(self, input_device, output_device):
        """运行中只重建声卡流（后台执行），不断开服务器、不重载模型。"""
        self.input_device = input_device
        self.output_device = output_device
        if not self.running:
            return True, ""
        if self._engine_busy():
            self.status_msg.emit("正在切换设备，请稍候...")
            return True, ""
        self.status_msg.emit("正在切换设备...")
        self._start_thread = EngineStartThread(self, "reopen", parent=self)
        self._start_thread.start()
        return True, ""

    def _reopen_blocking(self):
        virtual_out = self._out_is_virtual()
        self._virtual_out = bool(virtual_out)
        self._bt_in = self._in_is_bluetooth()
        if hasattr(self.pipeline, "_virtual_out"):
            self.pipeline._virtual_out = bool(virtual_out)
        if virtual_out and float(getattr(self, "block_time", 0.06) or 0.06) < VIRTUAL_OUT_MIN_BLOCK:
            self.status_msg.emit("给别人听建议停再开，块长会自动提到 80ms")
        self._close_stream_only()
        ok, msg = self._open_stream()
        if not ok:
            self.status_msg.emit("切换设备失败: " + msg)
            self._hard_stop()
            return
        self._fade_in_left = int(0.04 * self.pipeline.samplerate)
        self._fade_out_left = 0
        try:
            self.stream.start()
            if self._out_stream is not None:
                self._out_stream.start()
        except Exception as e:
            self.status_msg.emit("切换设备失败: " + str(e))
            self._hard_stop()
            return
        self.status_msg.emit("已切换设备 · " + msg)

    def _on_monitor(self, outdata, frames, times, status):
        """监听流回调：从队列取变声结果播放，空则补零（绝不阻塞）。"""
        try:
            block = self.monitor_queue.get_nowait()
            n = min(len(block), frames)
            outdata[:n, 0] = block[:n]
            if n < frames:
                outdata[n:, 0] = 0
            if outdata.shape[1] > 1:
                outdata[:, 1:] = outdata[:, :1]
        except queue.Empty:
            outdata.fill(0)

    def _open_monitor(self):
        self._close_monitor()
        if not self.monitor_enabled:
            return True, ""
        if self.monitor_device is None:
            return False, "未选择监听输出设备（请在右侧「音质与监听」中选择耳机）"
        block = getattr(self.pipeline, "_block_frame", None)
        if not block:
            return False, "推理块大小未知"
        try:
            self.monitor_stream = sd.OutputStream(
                samplerate=self.pipeline.samplerate,
                blocksize=block,
                channels=1,
                dtype="float32",
                device=self.monitor_device,
                callback=self._on_monitor,
                latency="low",
            )
            self.monitor_stream.start()
            return True, "监听已开启"
        except Exception as e:
            self.monitor_enabled = False
            self.monitor_stream = None
            return False, _wasapi_fail_reason(e)

    def _close_monitor(self):
        if self.monitor_stream is not None:
            try:
                self.monitor_stream.abort()
                self.monitor_stream.close()
            except Exception:
                pass
            self.monitor_stream = None
        self._drain_queue(self.monitor_queue)

    def _push_monitor(self, mono_out):
        if not self.monitor_enabled or self.monitor_stream is None:
            return
        try:
            self.monitor_queue.put_nowait(
                (mono_out * np.float32(self.monitor_volume)).astype(np.float32))
        except queue.Full:
            pass

    def _report_loop_latency(self, times, frames):
        """嘴到耳 ≈ PortAudio(DAC−ADC) + 一块算法延迟 + 队列积压 + 网络。

        回调里只写数值，由 UI 定时器取出，避免 PortAudio 线程发 Qt 信号。
        """
        if times is None:
            return
        try:
            adc = getattr(times, "inputBufferAdcTime", 0.0) or 0.0
            dac = getattr(times, "outputBufferDacTime", 0.0) or 0.0
            if adc <= 0 or dac <= 0:
                return
            pa_ms = (dac - adc) * 1000.0
            if not (0 < pa_ms < 5000):
                return
            sr = float(getattr(self.pipeline, "samplerate", 0) or 0)
            # 算法延迟地板是管线块，不是声卡回调帧；队列里每个元素也是一块管线块
            alg = int(getattr(self.pipeline, "_block_frame", 0) or frames)
            block_ms = (1000.0 * alg / sr) if sr > 0 else 0.0
            queued = 0
            try:
                queued = self.input_queue.qsize() + self.output_queue.qsize()
            except Exception:
                pass
            extra_ms = 0.0
            pipe = self.pipeline
            if pipe is not None and getattr(pipe, "is_remote", False):
                extra_ms += max(0.0, float(getattr(pipe, "_rtt_ema", 0.0) or 0.0) * 1000.0)
                try:
                    depth = int(pipe.inflight_depth()) if hasattr(pipe, "inflight_depth") else 2
                except Exception:
                    depth = 2
                extra_ms += max(0, min(3, depth) - 1) * block_ms
            true_ms = pa_ms + block_ms + queued * block_ms + extra_ms
            if self._loop_lat_ema is None:
                self._loop_lat_ema = true_ms
            else:
                self._loop_lat_ema += (true_ms - self._loop_lat_ema) * 0.08
            self._true_e2e_ms = self._loop_lat_ema
        except Exception:
            pass

    def _flush_callback_stats(self):
        if self.xrun_count != self._xrun_emitted:
            self._xrun_emitted = self.xrun_count
            self.xrun_signal.emit(self.xrun_count)
        if self.running and self._true_e2e_ms > 0:
            self.loop_latency.emit(self._true_e2e_ms)
        if self._fade_done_flag:
            self._fade_done_flag = False
            self.fade_done.emit()

    def request_hard_stop(self):
        """后台停机，UI 线程只发起。"""
        self._stop_requested = True
        self._fade_epoch += 1
        if self._engine_busy():
            action = getattr(self._start_thread, "action", "")
            if action == "stop":
                return
            # 启动/重连正堵在 websocket recv（最长 60s），必须掐断才能取消
            if self.mode == "server" and action in ("start", "recover"):
                try:
                    if self.pipeline is not None:
                        self.pipeline.abort()
                except Exception:
                    pass
            return
        if not self.running and self.stream is None and self.worker_thread is None:
            return
        self._start_thread = EngineStartThread(self, "stop", parent=self)
        self._start_thread.start()

    def wait_idle(self, timeout_ms=2500):
        t = getattr(self, "_start_thread", None)
        if t is not None and t.isRunning():
            t.wait(timeout_ms)

    def stop(self):
        if self.running and self.stream is not None and self._fade_out_left <= 0:
            self._fade_epoch += 1
            self._pending_fade_epoch = self._fade_epoch
            self._fade_out_total = max(1, int(0.04 * self.pipeline.samplerate))
            self._fade_out_left = self._fade_out_total
            return
        self.request_hard_stop()

    def _on_fade_done(self):
        if self._pending_fade_epoch != self._fade_epoch:
            return
        if not self.running:
            return
        self.request_hard_stop()

    def _on_worker_crash(self, msg):
        if self.mode == "server":
            self._try_recover()
            return
        self.load_failed.emit("推理中断: " + _friendly_error(msg))
        self.request_hard_stop()

    def _try_recover(self):
        if not self.running:
            return
        # 仅远程服务器模式走网络重连；本地进程内推理不碰这条路径
        if self.mode != "server":
            return
        if self._engine_busy():
            return
        now = time.time()
        if now - self._last_recover < 2.0:
            return
        self._last_recover = now
        self._start_thread = EngineStartThread(self, "recover")
        self._start_thread.start()

    def _recover_blocking(self):
        if self.mode != "server":
            return
        max_n = 3
        last_err = "无法连接服务器"
        for i in range(1, max_n + 1):
            if self._stop_requested or not self.running:
                if self._stop_requested:
                    self._hard_stop()
                return
            self.recover_progress.emit(i, max_n)
            self.status_msg.emit("连接中断，正在重连 %d/%d…" % (i, max_n))
            try:
                self.pipeline.abort()
            except Exception:
                pass
            time.sleep(0.5 if i == 1 else min(4.0, float(i)))
            if self._stop_requested or not self.running:
                if self._stop_requested:
                    self._hard_stop()
                return
            try:
                if self._ensure_connected() and self._ensure_model():
                    started = self.pipeline.start(**self.merged_params())
                    if started is False:
                        last_err = getattr(self.pipeline, "last_error", "") or "推理未能启动"
                        continue
                    self.recover_ok.emit()
                    self.status_msg.emit("已重新连上服务器")
                    return
                last_err = getattr(self.pipeline, "last_error", "") or "无法连接服务器"
            except Exception as e:
                last_err = str(e)
        self._hard_stop()
        self.recover_failed.emit(last_err)

    def _hard_stop(self):
        self._fade_epoch += 1
        self.running = False
        self._fade_out_left = 0
        self._close_stream_only()
        self._close_monitor()
        if self.worker_thread:
            self.worker_thread.running = False
        # 先停 worker 再掐 socket，避免 abort 触发 need_recover
        if self.mode == "server" and self.pipeline is not None:
            try:
                self.pipeline.abort()
            except Exception:
                pass
        if self.worker_thread:
            if self.worker_thread.isRunning():
                self.worker_thread.wait(500 if self.mode == "server" else 1800)
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            if self.worker_thread.isRunning():
                # 线程没退出：保留引用等它自然结束，不能销毁运行中的线程
                t = self.worker_thread
                t.finished.connect(lambda: t.deleteLater())
                self._zombie_threads.append(t)
            else:
                self.worker_thread.deleteLater()
            self.worker_thread = None
        elif self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        if self._emit_stopped:
            self.stopped_ok.emit()
            self.status_msg.emit("已停止")

    def _on_worker_infer_done(self, elapsed_ms, in_db, out_db):
        self.infer_time.emit(elapsed_ms)
        self.rms_levels.emit(in_db, out_db)
        self._adapt_if_slow(elapsed_ms)

    def _adapt_if_slow(self, elapsed_ms):
        budget = max(20.0, float(self.block_time) * 1000.0)
        if elapsed_ms > budget * 0.95:
            self._slow_streak += 1
        else:
            self._slow_streak = max(0, self._slow_streak - 1)
        if self._adapted or self._slow_streak < 6:
            return
        self._adapted = True
        if self.vad_enable:
            self.vad_enable = False
        if self.deesser_enable:
            self.deesser_enable = False
        try:
            cur = float(self.current_speaker.index_rate) if self.current_speaker else 0.0
        except Exception:
            cur = 0.0
        if cur > 0.35:
            self.change_index_rate(0.35)
            if self.current_speaker is not None:
                self.current_speaker.index_rate = 0.35
        self.status_msg.emit("推理偏慢：已自动关人声识别/去齿音并降低检索，稳住实时")

    def _on_worker_xrun(self):
        self.xrun_count += 1

    def _apply_edge_fade(self, outdata):
        n = len(outdata)
        if self._fade_in_left > 0:
            total = max(1, int(0.04 * self.pipeline.samplerate))
            done = total - self._fade_in_left
            # 线性斜坡，避免每块 np.arange 分配
            if n == 1:
                outdata[0, 0] *= min(1.0, (done + 1) / total)
            else:
                g0 = done / total
                g1 = (done + n) / total
                outdata[:, 0] *= np.linspace(g0, g1, n, endpoint=False, dtype=np.float32).clip(0.0, 1.0)
            self._fade_in_left = max(0, self._fade_in_left - n)
        if self._fade_out_left > 0:
            total = max(1, self._fade_out_total)
            remaining = self._fade_out_left
            if n == 1:
                outdata[0, 0] *= max(0.0, remaining / total)
            else:
                g0 = remaining / total
                g1 = (remaining - n) / total
                outdata[:, 0] *= np.linspace(g0, g1, n, endpoint=False, dtype=np.float32).clip(0.0, 1.0)
            self._fade_out_left = max(0, self._fade_out_left - n)
            if self._fade_out_left <= 0:
                self._fade_done_flag = True

    def _cb_push_in(self, mono):
        n = int(mono.shape[0])
        if n <= 0 or self._in_buf.size == 0:
            return
        if self._in_n + n > self._in_buf.shape[0]:
            self._in_n = 0
            self.xrun_count += 1
        self._in_buf[self._in_n:self._in_n + n] = mono[:n]
        self._in_n += n

    def _cb_take_in(self, n):
        if self._in_n < n or not self._in_pool:
            return None
        slot = self._in_pool[self._pool_i]
        self._pool_i = (self._pool_i + 1) % len(self._in_pool)
        if slot.shape[0] != n:
            slot = np.zeros(n, dtype=np.float32)
            self._in_pool[self._pool_i - 1] = slot
        slot[:n] = self._in_buf[:n]
        remain = self._in_n - n
        if remain:
            self._in_buf[:remain] = self._in_buf[n:self._in_n]
        self._in_n = remain
        return slot

    def _cb_push_out(self, block):
        if block is None or self._out_buf.size == 0:
            return
        src = block[:, 0] if getattr(block, "ndim", 1) > 1 else block
        n = int(src.shape[0])
        if n <= 0:
            return
        if self._out_n + n > self._out_buf.shape[0]:
            drop = self._out_n + n - self._out_buf.shape[0]
            if drop >= self._out_n:
                self._out_n = 0
                self._asrc_pos = 0.0
            else:
                self._out_buf[: self._out_n - drop] = self._out_buf[drop:self._out_n]
                self._out_n -= drop
                self._asrc_pos = max(0.0, float(getattr(self, "_asrc_pos", 0.0)) - drop)
            self.xrun_count += 1
        self._out_buf[self._out_n:self._out_n + n] = np.asarray(src[:n], dtype=np.float32)
        self._out_n += n

    def _cb_fill_from_hold(self, dest):
        n = dest.shape[0]
        self._hold_count = getattr(self, "_hold_count", 0) + 1
        # 喂给其它软件时：欠载补静音，绝不能把上一块再念一遍（那就是卡+炸麦）
        if getattr(self, "_virtual_out", False):
            dest.fill(0.0)
            fade = getattr(self, "_xrun_fade", None)
            last_n = int(getattr(self, "_last_out_n", 0) or 0)
            if (self._hold_count == 1 and last_n > 0
                    and fade is not None and fade.size):
                fade_n = min(n, last_n, int(fade.shape[0]))
                dest[:fade_n] = self._last_out[last_n - fade_n:last_n] * fade[:fade_n]
            return
        if self._last_out_n <= 0 or self._last_out.size == 0:
            dest.fill(0.0)
            return
        src = self._last_out[:self._last_out_n]
        # 自听：前 3 块原样重复，之后线性淡到静音，避免循环同一块变成嗡鸣
        g = 1.0 if self._hold_count <= 3 else max(0.0, 1.0 - 0.25 * (self._hold_count - 3))
        if src.shape[0] >= n:
            dest[:] = src[:n] * g
        else:
            dest[:src.shape[0]] = src * g
            dest[src.shape[0]:] = (src[-1] * g) if src.size else 0.0

    def _on_capture(self, indata, frames, times, status):
        if not self.running:
            return
        try:
            self._ingest_input(indata)
        except Exception:
            pass

    def _on_playback(self, outdata, frames, times, status):
        if not self.running:
            outdata.fill(0)
            return
        try:
            self._fill_output(outdata, None, frames, times)
        except Exception:
            outdata.fill(0)

    def _ingest_input(self, indata):
        if self.bypass:
            self._last_cap = np.asarray(
                indata[:, 0] if getattr(indata, "ndim", 1) > 1 else indata,
                dtype=np.float32,
            ).copy()
            return
        mono = indata[:, 0] if indata.ndim > 1 else indata
        if self.input_agc and self.agc is not None:
            mono = self.agc.process(mono)
        peak_lim = getattr(self, "_peak_lim", None)
        if peak_lim is not None:
            mono = peak_lim.process(mono)
        self._last_cap = np.asarray(mono, dtype=np.float32).copy()
        in_block = int(getattr(self.pipeline, "_block_frame", 0) or 0)
        if in_block <= 0:
            return
        if mono.shape[0] == in_block and self._in_n == 0 and self._in_pool:
            slot = self._in_pool[self._pool_i]
            self._pool_i = (self._pool_i + 1) % len(self._in_pool)
            if slot.shape[0] != in_block:
                slot = np.zeros(in_block, dtype=np.float32)
                self._in_pool[(self._pool_i - 1) % len(self._in_pool)] = slot
            slot[:in_block] = mono[:in_block]
            self._enqueue_input(slot)
        else:
            self._cb_push_in(mono)
            while True:
                chunk = self._cb_take_in(in_block)
                if chunk is None:
                    break
                self._enqueue_input(chunk)

    def _fill_output(self, outdata, cap_mono, n_needed, times):
        if self.bypass:
            src = cap_mono if cap_mono is not None else getattr(self, "_last_cap", None)
            outdata.fill(0)
            if src is not None and len(src):
                n = min(len(src), n_needed)
                outdata[:n, 0] = src[:n]
            if outdata.shape[1] > 1:
                outdata[:, 1:] = outdata[:, :1]
            self._apply_edge_fade(outdata)
            self._protect_virtual_out(outdata)
            self._push_monitor_view(outdata[:, 0], n_needed)
            self._report_loop_latency(times, n_needed)
            return
        in_block = int(getattr(self.pipeline, "_block_frame", 0) or 0)
        if in_block <= 0:
            outdata.fill(0)
            return
        if getattr(self, "_asrc_on", False):
            self._fill_output_asrc(outdata, n_needed, in_block)
        else:
            self._fill_output_copy(outdata, n_needed, in_block)
        if outdata.shape[1] > 1:
            outdata[:, 1:] = outdata[:, :1]
        dry = float(self.dry_mix)
        src = cap_mono if cap_mono is not None else getattr(self, "_last_cap", None)
        if dry > 0 and src is not None and len(src):
            n = min(len(src), n_needed)
            outdata[:n, 0] = outdata[:n, 0] * (1.0 - dry) + src[:n] * dry
            if outdata.shape[1] > 1:
                outdata[:, 1:] = outdata[:, :1]
        self._apply_edge_fade(outdata)
        self._protect_virtual_out(outdata)
        self._push_monitor_view(outdata[:, 0], n_needed)
        self._report_loop_latency(times, n_needed)

    def _fill_output_copy(self, outdata, n_needed, in_block):
        while self._out_n < n_needed:
            try:
                block = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self._cb_push_out(block)
        was_hold = int(getattr(self, "_hold_count", 0) or 0) > 0
        if self._out_n >= n_needed:
            outdata[:, 0] = self._out_buf[:n_needed]
            remain = self._out_n - n_needed
            if remain:
                self._out_buf[:remain] = self._out_buf[n_needed:self._out_n]
            self._out_n = remain
            hold_n = min(n_needed, self._last_out.shape[0])
            self._last_out[:hold_n] = outdata[:hold_n, 0]
            self._last_out_n = hold_n
            if was_hold:
                self._apply_recover_fade(outdata[:, 0], n_needed)
            self._hold_count = 0
        else:
            n_take = min(self._out_n, n_needed)
            if n_take > 0:
                outdata[:n_take, 0] = self._out_buf[:n_take]
                self._out_n = 0
                self._cb_fill_from_hold(outdata[n_take:, 0])
            else:
                self._cb_fill_from_hold(outdata[:, 0])
            self.xrun_count += 1

    def _asrc_target_n(self, n_needed, in_block):
        n = max(int(n_needed), 1)
        ib = max(int(in_block), n)
        return min(2 * n, ib)

    def _asrc_update_step(self, n_needed, in_block):
        n = max(int(n_needed), 1)
        ib = max(int(in_block), n)
        target = float(self._asrc_target_n(n, ib))
        occ = float(self._out_n)
        try:
            occ += self.output_queue.qsize() * ib
        except Exception:
            pass
        lo = target
        hi = target + ib
        if occ < lo:
            err = (occ - lo) / lo
        elif occ > hi:
            err = (occ - hi) / ib
        else:
            err = 0.0
        integ = 0.995 * float(getattr(self, "_asrc_int", 0.0) or 0.0) + err
        if integ > 40.0:
            integ = 40.0
        elif integ < -40.0:
            integ = -40.0
        self._asrc_int = integ
        step = 1.0 + 0.0004 * err + 5e-5 * integ
        if step < 0.998:
            step = 0.998
        elif step > 1.002:
            step = 1.002
        self._asrc_step = step

    def _apply_recover_fade(self, mono, n):
        fade = getattr(self, "_recover_fade", None)
        if fade is None or fade.size == 0 or mono is None:
            return
        fn = min(int(n), int(mono.shape[0]), int(fade.shape[0]))
        if fn > 0:
            mono[:fn] *= fade[:fn]

    def _fill_output_asrc(self, outdata, n_needed, in_block):
        n = int(n_needed)
        dest = outdata[:n, 0]
        target = self._asrc_target_n(n, in_block)
        while self._out_n < max(target, n) + max(int(in_block), n):
            try:
                block = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self._cb_push_out(block)
        if not getattr(self, "_asrc_ready", False):
            if self._out_n >= target:
                self._asrc_ready = True
            else:
                dest.fill(0.0)
                return
        step = float(getattr(self, "_asrc_step", 1.0) or 1.0)
        pos = float(getattr(self, "_asrc_pos", 0.0) or 0.0)
        if pos < 0.0:
            pos = 0.0
        src_need = int(pos + n * step) + 2
        while self._out_n < src_need:
            try:
                block = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self._cb_push_out(block)
        was_hold = int(getattr(self, "_hold_count", 0) or 0) > 0
        ramp = getattr(self, "_asrc_ramp", None)
        can_interp = (
            ramp is not None and ramp.size >= n and self._out_n >= src_need
            and self._out_n >= 2
        )
        if can_interp:
            idx = pos + ramp[:n] * np.float32(step)
            max_i = self._out_n - 2
            i0 = np.clip(np.floor(idx).astype(np.int32), 0, max_i)
            frac = (idx - i0).astype(np.float32)
            buf = self._out_buf
            dest[:] = buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac
            next_pos = pos + n * step
            consumed = int(next_pos)
            if consumed > 0:
                if consumed >= self._out_n:
                    consumed = max(0, self._out_n - 1)
                    pos = 0.0
                else:
                    pos = next_pos - consumed
                remain = self._out_n - consumed
                if remain > 0:
                    self._out_buf[:remain] = self._out_buf[consumed:self._out_n]
                self._out_n = remain
            else:
                pos = next_pos
            self._asrc_pos = pos
            hold_n = min(n, self._last_out.shape[0])
            self._last_out[:hold_n] = dest[:hold_n]
            self._last_out_n = hold_n
            if was_hold:
                self._apply_recover_fade(dest, n)
            self._hold_count = 0
            self._asrc_update_step(n, in_block)
            return
        n_take = min(self._out_n, n)
        if n_take > 0:
            dest[:n_take] = self._out_buf[:n_take]
            remain = self._out_n - n_take
            if remain > 0:
                self._out_buf[:remain] = self._out_buf[n_take:self._out_n]
            self._out_n = remain
            if n_take < n:
                self._cb_fill_from_hold(dest[n_take:])
        else:
            self._cb_fill_from_hold(dest)
        self._asrc_pos = 0.0
        self.xrun_count += 1
        self._asrc_update_step(n, in_block)

    def _protect_virtual_out(self, outdata):
        if not getattr(self, "_virtual_out", False):
            return
        # 膝点软折，避免硬夹平顶齿波；不绑下游软件
        a = np.abs(outdata)
        knee = 0.70
        lim = 0.89
        over = a > knee
        if not np.any(over):
            return
        t = (a - knee) / (lim - knee)
        folded = knee + (lim - knee) * np.tanh(t)
        np.copyto(outdata, np.sign(outdata) * folded, where=over)

    def _on_audio(self, indata, outdata, frames, times, status):
        """双工回调：同一设备自听时走这里。"""
        if not self.running:
            outdata.fill(0)
            return
        try:
            self._ingest_input(indata)
            mono = indata[:, 0] if indata.ndim > 1 else indata
            self._fill_output(outdata, mono, len(outdata), times)
        except Exception:
            outdata.fill(0)

    def _push_monitor_view(self, mono, n):
        if not self.monitor_enabled or self.monitor_stream is None:
            return
        n = min(int(n), int(mono.shape[0]), int(self._mon_scratch.shape[0] or 0))
        if n <= 0:
            return
        self._mon_scratch[:n] = mono[:n]
        self._mon_scratch[:n] *= np.float32(self.monitor_volume)
        try:
            self.monitor_queue.put_nowait(self._mon_scratch[:n].copy())
        except queue.Full:
            pass

