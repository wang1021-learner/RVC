"""RVC inference pipeline - no UI dependency"""
import os, time, traceback, logging
import numpy as np, torch, torch.nn.functional as F, torchaudio.transforms as tat
from configs.config import Config
from infer import rtrvc as rvc_for_realtime
from tools.dsp import DeEsser, DelayLine, FirHighPass
from tools.vad import VoiceActivityDetector
from tools.output_protector import OutputProtector
from tools.torchgate import TorchGate
from tools.cuda_graph import cuda_graph_enabled, graph_hot_path, run_cuda_graph


class RVCPipeline:
    def __init__(self, on_status=None):
        self.config = Config()
        self.rvc = None
        self.last_error = ""
        self._on_status = on_status or (lambda _: None)
        self.block_time = 0.06
        self.crossfade_time = 0.02
        self.extra_time = 0.8
        self.f0method = "rmvpe"
        self.samplerate = 48000
        self.channels = 1
        self.I_noise_reduce = False
        self.O_noise_reduce = False
        self.rms_mix_rate = 0.3
        self.threhold = -50
        # 输出保护：直流高通 + 软限幅（默认开启，-1 dBFS 起限）
        self.limiter_enable = True
        self.limiter_threshold_db = -1.0
        # 音质增强：高频齿音直通 / 临场感 / 去齿音（实时生效）
        self.hf_mix_rate = 0.2
        self.presence = 0.10
        self.deesser_enable = False
        # 智能人声识别 (VAD)
        self.vad_enable = False
        self.vad_threshold = 0.50
        self._active = False
        self.last_stage_ms = {}

    @property
    def is_loaded(self):
        return self.rvc is not None

    @property
    def is_active(self):
        return self._active

    @property
    def tgt_sr(self):
        return self.rvc.tgt_sr if self.rvc else 48000

    @property
    def is_remote(self):
        return False

    def is_connected(self):
        return True

    def abort(self):
        return

    def set_server_url(self, url):
        return

    def list_models(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights = os.path.join(root, "assets", "weights")
        if not os.path.isdir(weights):
            return []
        return sorted(
            f for f in os.listdir(weights) if f.lower().endswith(".pth")
        )

    def send_audio(self, indata):
        if not self._active:
            return None
        return np.asarray(indata, dtype=np.float32), time.perf_counter()

    def recv_audio(self, indata, t0, timeout=1.5):
        out, elapsed = self.process_chunk(indata)
        return out, elapsed

    def configure(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def change_pitch(self, val):
        if self.rvc:
            self.rvc.change_key(val)

    def change_index_rate(self, val):
        if self.rvc:
            self.rvc.change_index_rate(val)

    def change_formant(self, val):
        if self.rvc:
            self.rvc.change_formant(val)

    def _resolve_path(self, path, is_index=False):
        if not path:
            return ""
        if os.path.isfile(path):
            return os.path.abspath(path)
        name = os.path.basename(path)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [os.path.join(root, path), os.path.join(root, "assets", "weights", name)]
        if is_index or name.lower().endswith(".index"):
            logs = os.path.join(root, "logs")
            extra = []
            if os.path.isdir(logs):
                try:
                    extra = [
                        os.path.join(logs, d, name)
                        for d in os.listdir(logs)
                        if os.path.isdir(os.path.join(logs, d))
                    ]
                except Exception:
                    extra = []
            candidates = [
                os.path.join(root, path),
                os.path.join(logs, name),
                os.path.join(root, "logs", "thchs_v2", name),
                *extra,
                os.path.join(root, "deploy", "logs", "thchs_v2", name),
                os.path.join(root, "assets", "indices", name),
                os.path.join(root, "assets", name),
                os.path.join(root, "assets", "weights", name),
            ]
        for cand in candidates:
            if os.path.isfile(cand):
                return os.path.abspath(cand)
        return path

    def load_speaker(self, model_path, index_path="", pitch=0, index_rate=0.0,
                     formant=0.0, last_rvc=None, **params):
        if params:
            self.configure(**params)
        model_path = self._resolve_path(model_path, is_index=False)
        index_path = self._resolve_path(index_path, is_index=True)
        self._on_status(f"Loading: {os.path.basename(model_path)}")
        old = self.rvc
        try:
            reuse = last_rvc if last_rvc is not None else old
            self.rvc = rvc_for_realtime.RVC(
                pitch, formant, model_path, index_path, index_rate, self.config, reuse)
            self._buf_sig = None
            info = self.loaded_file_info()
            idx = info.get("index_path") or "无索引"
            flag = "" if info.get("index_loaded") else ("（未启用）" if info.get("index_path") else "")
            self._on_status(
                f"Loaded (sr={self.rvc.tgt_sr}) {info.get('model_path') or model_path}"
                f" | {idx}{flag}"
            )
            self.last_error = ""
            return True
        except Exception as e:
            logging.exception("加载模型失败")
            self.last_error = str(e)
            self._on_status("加载失败: " + str(e))
            self.rvc = old
            return False

    def loaded_file_info(self):
        """实际打开的模型/索引路径（解析后的绝对路径）。"""
        rvc = self.rvc
        if rvc is None:
            return {"model_path": "", "index_path": "", "index_loaded": False}
        pth = getattr(rvc, "pth_path", "") or ""
        idx = getattr(rvc, "index_path", "") or ""
        exists = bool(idx) and os.path.isfile(idx)
        loaded = exists and getattr(rvc, "index", None) is not None
        return {
            "model_path": pth,
            "index_path": idx if exists else (idx or ""),
            "index_loaded": bool(loaded),
        }

    def unload(self):
        self.rvc = None
        self._active = False
        self._buf_sig = None

    def start(self, samplerate=None, channels=1, **params):
        if self.rvc is None:
            raise RuntimeError("No model loaded")
        if params:
            self.configure(**params)
        if samplerate is not None:
            self.samplerate = samplerate
        self.samplerate = self.rvc.tgt_sr
        self.channels = channels
        # 模型/参数没变时跳过缓冲重建和 CUDA Graph 重录（stop→start/切设备秒起）
        sig = (self.block_time, self.crossfade_time, self.extra_time,
               self.samplerate, self.channels,
               self.I_noise_reduce, self.O_noise_reduce)
        if getattr(self, "_buf_sig", None) != sig or not hasattr(self, "_in_line"):
            self._setup_buffers()
            self._prewarm()
            self._buf_sig = sig
        else:
            self._zero_buffers()
        self._active = True
        return True

    def stop(self):
        self._active = False

    def process_chunk(self, indata):
        if not self._active:
            return np.zeros_like(indata), 0
        try:
            start_time = time.perf_counter()
            if indata.ndim > 1:
                indata = indata.mean(axis=1)
            in_len = indata.shape[0]
            chunk = torch.from_numpy(np.ascontiguousarray(indata, dtype=np.float32))
            if str(self.config.device).startswith("cuda"):
                chunk = chunk.to(self.config.device, non_blocking=True)
            else:
                chunk = chunk.to(self.config.device)
            self._in_line.push(chunk)

            if self.vad_enable and hasattr(self, "_vad"):
                vad_out, _, _ = self._vad.process(
                    self._in_line.latest(in_len), threshold=self.vad_threshold
                )
                self._in_line.latest(in_len).copy_(vad_out)
            elif self.threhold > -80:
                self._gate_last_block(in_len)

            if self.hf_mix_rate > 0 and not self.deesser_enable:
                self._hf_line.push(self._hp_hf.highpass(self._in_line.latest(in_len)))

            if self.I_noise_reduce:
                iw = self._in_line.latest(self._sola_buffer_frame + self._block_frame)
                ref = self._in_line.latest(self._nr_ref_frame)
                iw = self._tg(iw.unsqueeze(0), ref.unsqueeze(0)).squeeze(0)
                iw[:self._sola_buffer_frame] *= self._fade_in_window
                iw[:self._sola_buffer_frame] += self._nr_buffer * self._fade_out_window
                self._denoise_line.push(iw[:self._block_frame])
                self._nr_buffer.copy_(iw[self._block_frame:])
                ri = self._denoise_line.latest(self._block_frame + 2 * self._zc)
                rs = run_cuda_graph(
                    self._resampler, "in-resample", lambda a: self._resampler(a), ri
                )[160:]
            else:
                ri = self._in_line.latest(in_len + 2 * self._zc)
                rs = run_cuda_graph(
                    self._resampler, "in-resample", lambda a: self._resampler(a), ri
                )[160:]
            self._res_line.push(rs[-self._block_frame_16k:])
            ctx = graph_hot_path() if not getattr(self, "_warming", False) else None
            if ctx is not None:
                ctx.__enter__()
            try:
                infer_wav = self.rvc.infer(
                    self._res_line.latest(),
                    self._block_frame_16k,
                    self._skip_head,
                    self._return_length,
                    self.f0method,
                )
            finally:
                if ctx is not None:
                    ctx.__exit__(None, None, None)
            stage = getattr(self.rvc, "last_stage_ms", None)
            if stage:
                self.last_stage_ms = dict(stage)
            if self._resampler2 is not None:
                infer_wav = run_cuda_graph(
                    self._resampler2, "out-resample", lambda a: self._resampler2(a), infer_wav
                )
            if self.O_noise_reduce:
                self._out_line.push(infer_wav[-self._block_frame:])
                infer_wav = self._tg(
                    infer_wav.unsqueeze(0), self._out_line.latest().unsqueeze(0)
                ).squeeze(0)
            if self.rms_mix_rate < 1:
                infer_wav = self._apply_rms_mix(infer_wav)
            infer_wav, out_block = self._apply_sola(infer_wav)
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return out_block, elapsed_ms
        except Exception:
            traceback.print_exc()
            n = self._block_frame if self._block_frame else indata.shape[0]
            return np.zeros(n), 0

    def _setup_buffers(self):
        zc = self.samplerate // 100
        self._zc = zc
        bf = int(round(self.block_time * self.samplerate / zc)) * zc
        self._block_frame = bf
        self._block_frame_16k = 160 * bf // zc
        cf = int(round(self.crossfade_time * self.samplerate / zc)) * zc
        self._crossfade_frame = cf
        sbf = min(cf, 4 * zc)
        self._sola_buffer_frame = sbf
        self._sola_search_frame = 3 * zc
        self._extra_frame = int(round(self.extra_time * self.samplerate / zc)) * zc
        total = self._extra_frame + cf + self._sola_search_frame + bf
        # 输入降噪的参考窗口：300ms 足够统计噪声，不必送整条缓冲进 TorchGate
        self._nr_ref_frame = min(total, 30 * zc)
        dev, dt = self.config.device, torch.float32
        res_n = 160 * total // zc
        self._in_line = DelayLine(total, dev, dt)
        self._denoise_line = DelayLine(total, dev, dt)
        self._res_line = DelayLine(res_n, dev, dt)
        self._hf_line = DelayLine(total, dev, dt)
        self._out_line = DelayLine(total, dev, dt)
        # 兼容旧属性名（录音测试 / 调试仍可能读 _input_wav）
        self._input_wav = self._in_line.latest()
        self._input_wav_res = self._res_line.latest()
        self._gate_hist = torch.zeros(4 * zc, device=dev, dtype=dt)
        self._sola_buffer = torch.zeros(sbf, device=dev, dtype=dt)
        self._sola_den_kernel = torch.ones(1, 1, sbf, device=dev, dtype=dt)
        self._nr_buffer = self._sola_buffer.clone()
        self._skip_head = self._extra_frame // zc
        self._return_length = (bf + sbf + self._sola_search_frame) // zc
        t = torch.linspace(0.0, 1.0, steps=sbf, device=dev, dtype=dt)
        self._fade_in_window = torch.sin(0.5 * np.pi * t) ** 2
        self._fade_out_window = 1 - self._fade_in_window
        self._resampler = tat.Resample(orig_freq=self.samplerate, new_freq=16000, dtype=dt).to(dev)
        if self.rvc.tgt_sr != self.samplerate:
            self._resampler2 = tat.Resample(orig_freq=self.rvc.tgt_sr, new_freq=self.samplerate, dtype=dt).to(dev)
        else:
            self._resampler2 = None
        self._tg = TorchGate(sr=self.samplerate, n_fft=4 * zc, prop_decrease=0.9).to(dev)
        self._protector = OutputProtector(
            sample_rate=self.samplerate,
            threshold_db=self.limiter_threshold_db,
            device=dev,
        )
        self._hp_hf = FirHighPass(6000.0, self.samplerate, dev)
        self._hp_pres = FirHighPass(3000.0, self.samplerate, dev)
        self._deesser = DeEsser(sample_rate=self.samplerate, device=dev)
        self._vad = VoiceActivityDetector(
            sample_rate=self.samplerate,
            threshold=self.vad_threshold,
            device=dev,
        )
        pin_dev = "cpu"
        pin_kw = {"device": pin_dev, "dtype": torch.float32}
        if str(dev).startswith("cuda"):
            pin_kw["pin_memory"] = True
        self._out_pins = [
            torch.empty(bf, self.channels, **pin_kw) for _ in range(8)
        ]
        self._pin_i = 0

    def _prewarm(self):
        try:
            self._on_status("正在预热推理（录加速图）...")
            tone = 0.05 * torch.sin(
                2 * np.pi * 220.0
                * torch.arange(self._block_frame, device=self.config.device, dtype=torch.float32)
                / float(self.samplerate)
            )
            dummy = tone.detach().cpu().numpy().astype(np.float32)
            if self.rvc is not None:
                self.rvc.reset_feat_cache()
            self._warming = True
            self._active = True
            for _ in range(3):
                self.process_chunk(dummy)
            if str(self.config.device).startswith("cuda"):
                torch.cuda.synchronize(self.config.device)
            self._on_status(
                "加速就绪" if cuda_graph_enabled(self.config.device) else "预热完成"
            )
        except Exception:
            self._on_status(f"预热失败:\n{traceback.format_exc()}")
        finally:
            self._warming = False
            self._active = False
            self._zero_buffers()

    def _zero_buffers(self):
        """清空全部环形缓冲和 f0 缓存（start 跳过重建时也要清，防止旧数据残留）"""
        for a in ("_in_line", "_denoise_line", "_res_line", "_hf_line", "_out_line"):
            obj = getattr(self, a, None)
            if obj is not None:
                obj.reset()
        for a in ("_sola_buffer", "_nr_buffer", "_gate_hist"):
            obj = getattr(self, a, None)
            if obj is not None:
                obj.zero_()
        if self.rvc is not None:
            self.rvc.cache_pitch.zero_()
            self.rvc.cache_pitchf.zero_()
            if hasattr(self.rvc, "reset_feat_cache"):
                self.rvc.reset_feat_cache()
        if hasattr(self, "_protector"):
            self._protector.reset()
        for a in ("_hp_hf", "_hp_pres", "_deesser", "_vad"):
            obj = getattr(self, a, None)
            if obj is not None:
                obj.reset()

    def _gate_last_block(self, n):
        """在 GPU 上按 10ms 帧做静音门。始终算增益，避免 .any() 同步。"""
        wav = self._in_line.latest(n)
        zc = self._zc
        hist = torch.cat([self._gate_hist, wav])
        self._gate_hist = hist[-4 * zc:].detach()
        frames = F.pad(hist.view(1, 1, -1), (2 * zc, 2 * zc)).unfold(-1, 4 * zc, zc)
        rms = frames.pow(2).mean(-1).sqrt()[0, 0]
        db = 20.0 * torch.log10(rms.clamp(min=1e-8))
        gain_frames = (db >= self.threhold).float().view(1, 1, -1)
        gain_samples = F.interpolate(
            gain_frames, size=hist.shape[0], mode="linear", align_corners=False
        )[0, 0]
        wav.mul_(gain_samples[-n:])

    def _rms_torch(self, x):
        """librosa.feature.rms(center=True) 的 GPU 等价实现, 避免每块 .cpu() 同步"""
        pad = 2 * self._zc
        xp = F.pad(x.view(1, 1, -1), (pad, pad), mode="constant")  # 与 librosa 默认 pad_mode 一致
        frames = xp.unfold(-1, 4 * self._zc, self._zc)
        return frames.pow(2).mean(-1).sqrt()

    def _apply_rms_mix(self, infer_wav):
        iw = self._in_line.latest()[self._extra_frame:][:infer_wav.shape[0]]
        r1 = self._rms_torch(iw)
        r2 = self._rms_torch(infer_wav)
        r1 = F.interpolate(r1, size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
        r2 = F.interpolate(r2, size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
        r2 = torch.max(r2, torch.zeros_like(r2) + 1e-3)
        return infer_wav * torch.pow(r1 / r2, 1.0 - self.rms_mix_rate)

    def _copy_out(self, block_1d):
        pin = self._out_pins[self._pin_i]
        self._pin_i = (self._pin_i + 1) % len(self._out_pins)
        view = block_1d.reshape(-1, 1).expand(-1, self.channels).contiguous()
        pin.copy_(view, non_blocking=True)
        if block_1d.is_cuda:
            # 只等本次小拷贝（~60ms 音频）完成即可，开销可忽略；
            # .copy() 让返回数组独立于 pin 槽，杜绝槽位回绕时的别名脏读。
            torch.cuda.current_stream(block_1d.device).synchronize()
        return pin.numpy().copy()

    def _apply_sola(self, infer_wav):
        sbf = self._sola_buffer_frame
        need = self._block_frame + sbf
        if infer_wav.shape[0] < need + self._sola_search_frame:
            infer_wav = F.pad(
                infer_wav, (0, need + self._sola_search_frame - infer_wav.shape[0])
            )
        ci = infer_wav[None, None, : sbf + self._sola_search_frame]
        cn = F.conv1d(ci, self._sola_buffer[None, None, :])
        cd = torch.sqrt(F.conv1d(ci ** 2, self._sola_den_kernel) + 1e-8)
        corr = cn[0, 0] / cd[0, 0]
        windows = infer_wav.unfold(0, need, 1)
        n_off = min(int(corr.shape[0]), int(windows.shape[0]))
        aligned = windows[:n_off][torch.argmax(corr[:n_off])]
        aligned = aligned.clone()
        aligned[:sbf] = aligned[:sbf] * self._fade_in_window + self._sola_buffer * self._fade_out_window
        self._sola_buffer.copy_(aligned[self._block_frame : self._block_frame + sbf])
        block = aligned[: self._block_frame]
        use_hf = self.hf_mix_rate > 0 and not self.deesser_enable
        if use_hf and hasattr(self, "_hf_line"):
            hf_block = self._hf_line.latest(self._block_frame)
            in_tail = self._in_line.latest(self._block_frame)
            ratio = hf_block.square().mean() / in_tail.square().mean().clamp(min=1e-8)
            gate = ((ratio - 0.005) / 0.03).clamp(0.0, 1.0)
            mix = self.hf_mix_rate * gate * 0.5
            block = block * (1.0 - mix) + hf_block * mix
        if self.deesser_enable and hasattr(self, "_deesser") and not use_hf:
            block = self._deesser.process(block)
        if self.presence > 0 and hasattr(self, "_hp_pres"):
            k = float(self.presence) * 0.3
            block = block + k * self._hp_pres.highpass(block)
        if self.limiter_enable and hasattr(self, "_protector"):
            if abs(self._protector.threshold_db - self.limiter_threshold_db) > 0.01:
                self._protector.set_threshold_db(self.limiter_threshold_db)
            block = self._protector.process(block)
        return infer_wav, self._copy_out(block)
