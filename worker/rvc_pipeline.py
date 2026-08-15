"""RVC inference pipeline - no UI dependency"""
import os, time, traceback
import numpy as np, torch, torch.nn.functional as F, torchaudio.transforms as tat
from configs.config import Config
from infer import rtrvc as rvc_for_realtime
from tools.dsp import DeEsser, FirHighPass
from tools.vad import VoiceActivityDetector
from tools.output_protector import OutputProtector
from tools.torchgate import TorchGate
from tools.cuda_graph import cuda_graph_enabled, run_cuda_graph


class RVCPipeline:
    def __init__(self, on_status=None):
        self.config = Config()
        self.rvc = None
        self._on_status = on_status or (lambda _: None)
        self.block_time = 0.08
        self.crossfade_time = 0.02
        self.extra_time = 1.5
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
        self.hf_mix_rate = 0.3
        self.presence = 0.15
        self.deesser_enable = True
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
            candidates = [
                os.path.join(root, path),
                os.path.join(root, "logs", "thchs_v2", name),
                os.path.join(root, "deploy", "logs", "thchs_v2", name),
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
            self._on_status(f"Loaded (sr={self.rvc.tgt_sr})")
            return True
        except Exception:
            self._on_status(f"Load failed:\n{traceback.format_exc()}")
            self.rvc = old
            return False

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
        if getattr(self, "_buf_sig", None) != sig or not hasattr(self, "_input_wav"):
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
            bf = self._block_frame
            bf16 = self._block_frame_16k
            in_len = indata.shape[0]

            # 滚动并写入输入缓冲
            self._input_wav = torch.roll(self._input_wav, -self._block_frame, dims=0)
            self._input_wav[-indata.shape[0]:] = torch.from_numpy(indata).to(self.config.device)

            # 智能人声识别 (VAD) 与静音门控协同（平滑状态包络）
            if self.vad_enable and hasattr(self, "_vad"):
                wav_chunk = self._input_wav[-in_len:]
                vad_out, _, _ = self._vad.process(wav_chunk, threshold=self.vad_threshold)
                self._input_wav[-in_len:] = vad_out
            elif self.threhold > -80:
                self._gate_last_block(in_len)

            # 高频齿音直通：从 48k 原声提取 >6kHz 气音，与输入同延迟线滚动
            if self.hf_mix_rate > 0:
                hf_new = self._hp_hf.highpass(self._input_wav[-in_len:])
                self._hf_wav = torch.roll(self._hf_wav, -self._block_frame, dims=0)
                self._hf_wav[-in_len:] = hf_new

            self._input_wav_res = torch.roll(self._input_wav_res, -self._block_frame_16k, dims=0)
            if self.I_noise_reduce:
                self._input_wav_denoise = torch.roll(self._input_wav_denoise, -self._block_frame, dims=0)
                iw = self._input_wav[-self._sola_buffer_frame - self._block_frame:]
                ref = self._input_wav[-self._nr_ref_frame:]
                iw = self._tg(iw.unsqueeze(0), ref.unsqueeze(0)).squeeze(0)
                iw[:self._sola_buffer_frame] *= self._fade_in_window
                iw[:self._sola_buffer_frame] += self._nr_buffer * self._fade_out_window
                self._input_wav_denoise[-self._block_frame:] = iw[:self._block_frame]
                self._nr_buffer[:] = iw[self._block_frame:]
                ri = self._input_wav_denoise[-self._block_frame - 2 * self._zc:]
                self._input_wav_res[-self._block_frame_16k - 160:] = run_cuda_graph(
                    self._resampler, "in-resample", lambda a: self._resampler(a), ri
                )[160:]
            else:
                ri = self._input_wav[-in_len - 2 * self._zc:]
                self._input_wav_res[-160 * (in_len // self._zc + 1):] = run_cuda_graph(
                    self._resampler, "in-resample", lambda a: self._resampler(a), ri
                )[160:]
            infer_wav = self.rvc.infer(self._input_wav_res, self._block_frame_16k, self._skip_head, self._return_length, self.f0method)
            stage = getattr(self.rvc, "last_stage_ms", None)
            if stage:
                self.last_stage_ms = dict(stage)
            if self._resampler2 is not None:
                infer_wav = run_cuda_graph(self._resampler2, "out-resample", lambda a: self._resampler2(a), infer_wav)
            if self.O_noise_reduce:
                self._output_buffer = torch.roll(self._output_buffer, -self._block_frame, dims=0)
                self._output_buffer[-self._block_frame:] = infer_wav[-self._block_frame:]
                infer_wav = self._tg(infer_wav.unsqueeze(0), self._output_buffer.unsqueeze(0)).squeeze(0)
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
        self._input_wav = torch.zeros(total, device=dev, dtype=dt)
        self._input_wav_denoise = self._input_wav.clone()
        self._input_wav_res = torch.zeros(160 * total // zc, device=dev, dtype=dt)
        self._gate_hist = torch.zeros(4 * zc, device=dev, dtype=dt)
        self._sola_buffer = torch.zeros(sbf, device=dev, dtype=dt)
        self._sola_den_kernel = torch.ones(1, 1, sbf, device=dev, dtype=dt)
        self._nr_buffer = self._sola_buffer.clone()
        self._output_buffer = self._input_wav.clone()
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
        # 音质增强：齿音提取高通 + 同构延迟线 + 临场感 + 去齿音（FIR，单次 conv1d）
        self._hp_hf = FirHighPass(6000.0, self.samplerate, dev)
        self._hf_wav = torch.zeros(total, device=dev, dtype=dt)
        self._hp_pres = FirHighPass(3000.0, self.samplerate, dev)
        self._deesser = DeEsser(sample_rate=self.samplerate, device=dev)
        # 智能人声识别 (VAD)
        self._vad = VoiceActivityDetector(
            sample_rate=self.samplerate,
            threshold=self.vad_threshold,
            device=dev,
        )

    def _prewarm(self):
        if not cuda_graph_enabled(self.config.device):
            return
        try:
            self._on_status("正在预热 CUDA Graph 加速图...")
            n = self._input_wav_res.shape[0]
            phase = torch.arange(n, device=self.config.device, dtype=torch.float32)
            self._input_wav_res.copy_(0.05 * torch.sin(2 * np.pi * 220.0 * phase / 16000.0))
            if self.I_noise_reduce:
                s = self._input_wav[-self._sola_buffer_frame - self._block_frame:].unsqueeze(0)
                self._tg(s, self._input_wav[-self._nr_ref_frame:].unsqueeze(0))
            ri = self._input_wav[-self._block_frame - 2 * self._zc:]
            run_cuda_graph(self._resampler, "warmup-resample", lambda a: self._resampler(a), ri)
            _ = self.rvc.infer(self._input_wav_res, self._block_frame_16k, self._skip_head, self._return_length, self.f0method)
            if self._resampler2 is not None:
                tmp = self.rvc.infer(self._input_wav_res, self._block_frame_16k, self._skip_head, self._return_length, self.f0method)
                run_cuda_graph(self._resampler2, "warmup-output", lambda a: self._resampler2(a), tmp)
            if self.O_noise_reduce:
                tmp = self.rvc.infer(self._input_wav_res, self._block_frame_16k, self._skip_head, self._return_length, self.f0method)
                self._tg(tmp.unsqueeze(0), self._output_buffer.unsqueeze(0))
            torch.cuda.synchronize(self.config.device)
            self._on_status("CUDA Graph 加速就绪")
        except Exception:
            self._on_status(f"CUDA Graph 预热失败:\n{traceback.format_exc()}")
        finally:
            self._zero_buffers()

    def _zero_buffers(self):
        """清空全部环形缓冲和 f0 缓存（start 跳过重建时也要清，防止旧数据残留）"""
        for a in ["_input_wav", "_input_wav_denoise", "_input_wav_res", "_output_buffer", "_sola_buffer", "_nr_buffer", "_gate_hist", "_hf_wav"]:
            obj = getattr(self, a, None)
            if obj is not None:
                obj.zero_()
        if self.rvc is not None:
            self.rvc.cache_pitch.zero_()
            self.rvc.cache_pitchf.zero_()
        if hasattr(self, "_protector"):
            self._protector.reset()
        for a in ("_hp_hf", "_hp_pres", "_deesser", "_vad"):
            obj = getattr(self, a, None)
            if obj is not None:
                obj.reset()

    def _gate_last_block(self, n):
        """在 GPU 上按 10ms 帧做静音门，避免 librosa 每块回 CPU。"""
        wav = self._input_wav[-n:]
        zc = self._zc
        hist = torch.cat([self._gate_hist, wav])
        self._gate_hist = hist[-4 * zc:].detach()
        frames = F.pad(hist.view(1, 1, -1), (2 * zc, 2 * zc)).unfold(-1, 4 * zc, zc)
        rms = frames.pow(2).mean(-1).sqrt()[0, 0]
        db = 20.0 * torch.log10(rms.clamp(min=1e-8))
        # hist 比本块多 4*zc，对齐到 wav 起点
        offset = hist.shape[0] - n
        fade = max(1, int(0.008 * self.samplerate))
        n_frames = db.shape[0]
        for i in range(n_frames):
            if db[i] >= self.threhold:
                continue
            start = i * zc - offset
            end = start + zc
            if end <= 0 or start >= n:
                continue
            a = max(0, start)
            b = min(n, end)
            seg = wav[a:b]
            if seg.numel() > 2 * fade:
                t = torch.linspace(1.0, 0.0, fade, device=wav.device, dtype=wav.dtype)
                seg[:fade] *= t
                seg[fade:-fade] = 0
                seg[-fade:] *= t.flip(0)
            else:
                seg.zero_()

    def _rms_torch(self, x):
        """librosa.feature.rms(center=True) 的 GPU 等价实现, 避免每块 .cpu() 同步"""
        pad = 2 * self._zc
        xp = F.pad(x.view(1, 1, -1), (pad, pad), mode="constant")  # 与 librosa 默认 pad_mode 一致
        frames = xp.unfold(-1, 4 * self._zc, self._zc)
        return frames.pow(2).mean(-1).sqrt()

    def _apply_rms_mix(self, infer_wav):
        iw = self._input_wav[self._extra_frame:][:infer_wav.shape[0]]
        r1 = self._rms_torch(iw)
        r2 = self._rms_torch(infer_wav)
        r1 = F.interpolate(r1, size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
        r2 = F.interpolate(r2, size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
        r2 = torch.max(r2, torch.zeros_like(r2) + 1e-3)
        return infer_wav * torch.pow(r1 / r2, 1.0 - self.rms_mix_rate)

    def _apply_sola(self, infer_wav):
        ci = infer_wav[None, None, :self._sola_buffer_frame + self._sola_search_frame]
        cn = F.conv1d(ci, self._sola_buffer[None, None, :])
        cd = torch.sqrt(F.conv1d(ci ** 2, self._sola_den_kernel) + 1e-8)
        offset = torch.argmax(cn[0, 0] / cd[0, 0]).item()
        infer_wav = infer_wav[offset:]
        infer_wav[:self._sola_buffer_frame] *= self._fade_in_window
        infer_wav[:self._sola_buffer_frame] += self._sola_buffer * self._fade_out_window
        self._sola_buffer[:] = infer_wav[self._block_frame:self._block_frame + self._sola_buffer_frame]
        # ── 高频齿音直通补偿：48k 原声提取的气音按能量门限混回输出 ──
        if self.hf_mix_rate > 0 and hasattr(self, "_hf_wav"):
            end = -self._extra_frame if self._extra_frame > 0 else None
            hf_block = self._hf_wav[-(self._extra_frame + self._block_frame): end]
            in_tail = self._input_wav[-(self._extra_frame + self._block_frame): end]
            hf_energy = hf_block.square().mean()
            full_energy = in_tail.square().mean().clamp(min=1e-8)
            ratio = hf_energy / full_energy
            # 齿音占比门限：气音显著时混回，元音频段不动作
            gate = ((ratio - 0.005) / 0.03).clamp(0.0, 1.0)
            mix = self.hf_mix_rate * gate * 0.5
            infer_wav[:self._block_frame] = (
                infer_wav[:self._block_frame] * (1.0 - mix) + hf_block * mix
            )
        # ── 内置 DSP：自适应去齿音 + 临场感提升（状态跨块连续）──
        if self.deesser_enable and hasattr(self, "_deesser"):
            infer_wav[:self._block_frame] = self._deesser.process(
                infer_wav[:self._block_frame]
            )
        if self.presence > 0 and hasattr(self, "_hp_pres"):
            k = float(self.presence) * 0.3
            infer_wav[:self._block_frame] = (
                infer_wav[:self._block_frame]
                + k * self._hp_pres.highpass(infer_wav[:self._block_frame])
            )
        # 输出保护：对最终输出块做直流高通 + 软限幅（状态跨块连续，
        # 每个输出样本在其输出块上恰好经过一次保护）
        if self.limiter_enable and hasattr(self, "_protector"):
            if abs(self._protector.threshold_db - self.limiter_threshold_db) > 0.01:
                self._protector.set_threshold_db(self.limiter_threshold_db)
            infer_wav[:self._block_frame] = self._protector.process(
                infer_wav[:self._block_frame]
            )
        out = (
            infer_wav[:self._block_frame]
            .unsqueeze(-1)
            .expand(-1, self.channels)
            .contiguous()
            .cpu()
            .numpy()
        )
        return infer_wav, out
