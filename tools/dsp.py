"""
RVC 内置 DSP：状态连续的单极点滤波器、临场感提升与自适应 De-Esser（GPU）。

设计说明：
- 所有 IIR 均为「一阶低通 + x - lp 得到高通」的组合，块内递推用
  「指数缩放 cumsum」向量化（与 OutputProtector 的直流高通同一技巧），
  状态跨块连续，无块间爆音。
- De-Esser 增益递推采用「攻击即时 + 指数回升」的精确向量化：
  e[m] = max(e_target[m], e[m-1]*c)，与限幅器包络同构，数值有界。
- 数值稳定：低通分块长度自适应，保证 r^(-chunk) 不超过 1e6。
"""

import math

import torch
import torch.nn.functional as F


class OnePoleLP:
    """一阶低通 y[n] = r*y[n-1] + a*x[n]，a = 1 - exp(-2*pi*fc/fs)。

    提供 lowpass(x) 与 highpass(x)=x-lowpass(x)。
    注意：两种输出共享同一内部状态，每个块只能调用其中一个。
    """

    def __init__(self, cutoff_hz, sample_rate, device):
        self.cutoff = float(cutoff_hz)
        self.sample_rate = int(sample_rate)
        self.device = torch.device(device)
        self.r = math.exp(-2.0 * math.pi * self.cutoff / self.sample_rate)
        # 数值安全的分块长度：r^(-chunk) <= 1e6
        if self.r <= 0:
            self.chunk = 8192
        else:
            self.chunk = max(64, min(8192, int(13.8 / -math.log(self.r))))
        self._prev_y = torch.zeros(1, device=self.device, dtype=torch.float32)
        self._cache = {}

    def reset(self):
        self._prev_y.zero_()

    def _rk(self, n, dtype):
        key = (n, str(dtype))
        if key not in self._cache:
            k = torch.arange(n, device=self.device, dtype=dtype)
            rk = torch.pow(
                torch.full((n,), self.r, device=self.device, dtype=dtype), k
            )
            self._cache[key] = (rk, 1.0 / rk)
        return self._cache[key]

    def lowpass(self, x):
        """y[n] = r*y[n-1] + a*x[n]
        => y[n]*r^(-n) = r*y[-1] + sum_{k<=n} a*x[k]*r^(-k)（分段 cumsum）
        """
        r = self.r
        a = 1.0 - r
        n = x.shape[0]
        out = torch.empty_like(x)
        offset = 0
        while offset < n:
            seg = x[offset : offset + self.chunk]
            m = seg.shape[0]
            rk, rk_inv = self._rk(m, x.dtype)
            y = (torch.cumsum(a * seg * rk_inv, dim=0) + r * self._prev_y[0]) * rk
            out[offset : offset + m] = y
            self._prev_y.copy_(y[-1:])
            offset += m
        return out

    def highpass(self, x):
        return x - self.lowpass(x)


class DeEsser:
    """自适应齿音消除：监测 6kHz 以上能量，超阈值时全带软衰减。

    帧级（10ms）峰值包络 + 即时 attack / 指数 release 增益：
    e[m] = max(e_target[m], e[m-1]*c)，帧增益线性插值到样本级。
    """

    def __init__(
        self,
        sample_rate=48000,
        threshold_db=-22.0,
        ratio=0.7,
        release_time=0.06,
        device="cpu",
    ):
        self.sample_rate = int(sample_rate)
        self.threshold_db = float(threshold_db)
        self.ratio = float(ratio)
        self.device = torch.device(device)
        self.zc = max(1, self.sample_rate // 100)
        self.release_time = float(release_time)
        # 每 10ms 帧的 release 衰减系数
        self._c = math.exp(-0.01 / self.release_time)
        self.hp6 = OnePoleLP(6000.0, sample_rate, device)
        self._e_prev = torch.zeros(1, device=self.device, dtype=torch.float32)

    def reset(self):
        self.hp6.reset()
        self._e_prev.zero_()

    def process(self, x):
        zc = self.zc
        n = x.shape[0]
        band = self.hp6.highpass(x)   # 6kHz 以上能量作为齿音检测信号
        if n < zc:
            return x
        n_frames = n // zc
        use = n_frames * zc
        tail = x[use:]
        peaks = band[:use].view(n_frames, zc).abs().amax(dim=1).clamp(min=1e-6)
        db = 20.0 * torch.log10(peaks)
        over = (db - self.threshold_db).clamp(min=0.0)
        # target = 超阈值 12dB 时全衰减
        target = (1.0 - self.ratio * (over / 12.0)).clamp(min=0.0, max=1.0)
        e_target = 1.0 - target
        c = self._c
        m = torch.arange(n_frames, device=x.device, dtype=x.dtype)
        c_m = torch.pow(
            torch.full((n_frames,), c, device=x.device, dtype=x.dtype), m
        )
        # e[m] = max(e_target[m], e[m-1]*c)
        # => e[m]*c^(-m) = max(cummax(e_target*c^(-k)), e[-1]*c)
        # => e[m] = c^m * max(cum, e_prev*c)（精确递推，数值有界）
        cum = torch.cummax(e_target / c_m, dim=0).values
        e = c_m * torch.maximum(cum, self._e_prev[0] * c)
        self._e_prev.copy_(e[-1:])
        gains = (1.0 - e).clamp(min=0.0, max=1.0)
        g_lin = F.interpolate(
            gains.view(1, 1, -1), size=use, mode="linear", align_corners=True
        ).view(-1)
        out = torch.cat([x[:use] * g_lin, tail * gains[-1]])
        return out
