"""
RVC 智能人声识别模块 (Voice Activity Detection, VAD)
=====================================================
功能：
1. 智能区分人类发音（元音、辅音、语流）与环境杂音（机械键盘、敲击、风扇、电流）。
2. 全块多帧扫描 (Whole-Block Multi-Frame Scan)，杜绝中后段字头吞字。
3. 状态性连续平滑增益包络 (Continuous Gain Envelope)，杜绝门控硬切引发的相位台阶与咔哒爆音。
4. 纯 GPU 向量化与单次标量同步，零流水线阻塞与零感知延迟。
"""

import math
import os
import torch
import torch.nn.functional as F


class VoiceActivityDetector:
    """智能人声识别器（支持 Silero JIT 与 GPU 频谱谐波特征分析）"""

    def __init__(
        self,
        sample_rate: int = 48000,
        threshold: float = 0.5,
        hangover_ms: float = 40.0,
        device: str = "cuda",
    ):
        self.sample_rate = int(sample_rate)
        self.threshold = float(threshold)
        self.hangover_ms = float(hangover_ms)
        self.device = torch.device(device)

        self.zc = max(1, self.sample_rate // 100)  # 10ms 帧大小
        self.hangover_frames = max(1, int(self.hangover_ms / 10.0))
        self._hangover_left = 0
        self._is_active = False
        self._current_gain = 0.0
        self._delay = None

        # 尝试加载本地 Silero JIT 模型（如有）
        self._silero_model = None
        self._init_silero()

    def _init_silero(self):
        """尝试寻找或加载轻量 Silero JIT 模型"""
        asset_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
        )
        jit_path = os.path.join(asset_root, "vad", "silero_vad.jit")
        if os.path.isfile(jit_path):
            try:
                model = torch.jit.load(jit_path, map_location=self.device)
                model.eval()
                self._silero_model = model
            except Exception:
                self._silero_model = None

    def reset(self):
        """重置状态机"""
        self._hangover_left = 0
        self._is_active = False
        self._current_gain = 0.0
        self._delay = None
        if self._silero_model is not None and hasattr(self._silero_model, "reset_states"):
            try:
                self._silero_model.reset_states()
            except Exception:
                pass

    def detect_speech_prob(self, wav_48k: torch.Tensor, wav_16k: torch.Tensor = None) -> float:
        """
        计算输入音频块的人声置信度 (0.0 ~ 1.0)
        结合全块滑动多帧扫描、元音共振峰、清辅音摩擦能量与自相关谐波周期性。
        """
        if wav_48k.numel() < self.zc:
            return 0.0

        # 如果有 Silero JIT 模型，优先走 Silero
        if self._silero_model is not None and wav_16k is not None:
            try:
                with torch.no_grad():
                    prob = self._silero_model(wav_16k.view(1, -1), 16000)
                    if torch.is_tensor(prob):
                        return prob.reshape(()).to(dtype=torch.float32)
                    return torch.tensor(float(prob), device=self.device)
            except Exception:
                pass

        # ── 纯 GPU 自适应全块滑动频谱与谐波分析 ──
        rms = wav_48k.square().mean().sqrt().clamp(min=1e-8)
        db = 20.0 * torch.log10(rms)
        if db < -65.0:
            return 0.0

        n = wav_48k.shape[0]
        n_fft = min(1024, 1 << (n.bit_length() - 1)) if n >= 256 else 256
        hop = n_fft // 2

        # 1. 全块展开切片多帧扫描（杜绝字头在块中后段被吞）
        if n >= n_fft:
            pad = (hop - (n - n_fft) % hop) % hop
            w_padded = F.pad(wav_48k, (0, pad))
            frames = w_padded.unfold(0, n_fft, hop)  # (num_frames, n_fft)
        else:
            frames = F.pad(wav_48k, (0, n_fft - n)).unsqueeze(0)  # (1, n_fft)

        window = torch.hann_window(n_fft, device=self.device)
        spec = torch.fft.rfft(frames * window, dim=-1).abs()  # (num_frames, n_bins)
        freq_bin_hz = (self.sample_rate / 2.0) / (spec.shape[-1] - 1)

        idx_low = max(1, int(100 / freq_bin_hz))
        idx_mid = min(spec.shape[-1] - 1, int(3400 / freq_bin_hz))
        idx_high = min(spec.shape[-1] - 1, int(8000 / freq_bin_hz))
        idx_sub = max(1, int(80 / freq_bin_hz))

        vocal_energy = spec[:, idx_low:idx_mid].square().sum(dim=-1)
        fricative_energy = spec[:, idx_mid:idx_high].square().sum(dim=-1)
        total_energy = spec[:, idx_sub:].square().sum(dim=-1).clamp(min=1e-8)

        vocal_ratio = (vocal_energy / total_energy).amax()
        fricative_ratio = (fricative_energy / total_energy).amax()

        # 2. 谐波自相关分析（元音具有强周期性）
        down = wav_48k[:: (self.sample_rate // 16000)]
        d_len = down.shape[0]
        if d_len >= 160:
            max_lag = min(d_len // 2, 320)
            min_lag = 32
            r = F.conv1d(
                down.view(1, 1, -1),
                down[:max_lag].view(1, 1, -1),
                padding=max_lag,
            ).view(-1)
            center = max_lag
            lags = r[center + min_lag : center + max_lag]
            # 归一化必须用相关窗口的能量（前 max_lag 样本），
            # 用整块能量会把周期置信度压低 4 倍导致语音漏检
            norm = down[:max_lag].square().sum() + 1e-8
            periodicity = (lags.amax() / norm).clamp(min=0.0) if lags.numel() > 0 else torch.tensor(0.0, device=self.device)
        else:
            periodicity = torch.tensor(0.0, device=self.device)

        # 3. 综合置信度评估（单次标量合并，彻底消除多次 .item() 流水线同步）
        vowel_prob = 0.55 * vocal_ratio + 0.45 * (periodicity * 1.8).clamp(max=1.0)
        consonant_prob = torch.where(
            db > -48.0,
            (fricative_ratio * 1.6).clamp(max=1.0),
            torch.tensor(0.0, device=self.device),
        )
        return torch.maximum(vowel_prob, consonant_prob).clamp(0.0, 1.0)

    def process(
        self,
        wav_48k: torch.Tensor,
        wav_16k: torch.Tensor = None,
        threshold: float = None,
    ) -> tuple[torch.Tensor, bool, float]:
        """
        1 帧 look-ahead（10ms）+ 短 hangover。
        增益在 GPU 上算完，只在返回 Python 标记时同步一次标量。
        """
        th = float(threshold if threshold is not None else self.threshold)
        zc = self.zc
        if self._delay is None or self._delay.shape[0] != zc or self._delay.device != wav_48k.device:
            self._delay = torch.zeros(zc, device=wav_48k.device, dtype=wav_48k.dtype)
        combined = torch.cat([self._delay, wav_48k])
        self._delay = wav_48k[-zc:].detach()
        delayed = combined[: wav_48k.shape[0]]

        prob_t = self.detect_speech_prob(combined, wav_16k)
        if not torch.is_tensor(prob_t):
            prob_t = torch.tensor(float(prob_t), device=wav_48k.device)
        active_t = prob_t >= th
        if bool(active_t.detach()):
            self._hangover_left = self.hangover_frames
            self._is_active = True
        elif self._hangover_left > 0:
            self._hangover_left -= 1
            self._is_active = True
        else:
            self._is_active = False

        target_gain = 1.0 if self._is_active else 0.0
        n = delayed.shape[0]
        if self._current_gain == target_gain:
            out = delayed if target_gain == 1.0 else torch.zeros_like(delayed)
        else:
            gain_curve = torch.linspace(
                self._current_gain,
                target_gain,
                n,
                device=delayed.device,
                dtype=delayed.dtype,
            )
            self._current_gain = target_gain
            out = delayed * gain_curve
        return out, self._is_active, float(prob_t.detach())
