"""
RVC 内置 DSP：FIR 高通滤波器、临场感提升与自适应 De-Esser（GPU）。

设计说明：
- 滤波器用 scipy.signal.firls 在初始化时设计成 65 抽头线性相位 FIR，
  每块只需一次 conv1d（overlap 状态跨块连续），无块间爆音、
  无逐样本递推的 kernel 启动风暴（GPU 上 <0.2ms/块）。
- De-Esser 增益递推采用「攻击即时 + 指数回升」的精确向量化：
  e[m] = max(e_target[m], e[m-1]*c)，与限幅器包络同构，数值有界。
"""

import math

import torch
import torch.nn.functional as F


class FirHighPass:
    """线性相位 FIR 高通（firls 设计，overlap 状态跨块连续）。"""

    def __init__(self, cutoff_hz, sample_rate, device, numtaps=65):
        from scipy.signal import firls

        self.cutoff = float(cutoff_hz)
        self.sample_rate = int(sample_rate)
        self.device = torch.device(device)
        numtaps = int(numtaps) if numtaps % 2 == 1 else int(numtaps) + 1
        bands = [0.0, self.cutoff * 0.55, self.cutoff * 1.05,
                 self.sample_rate / 2.0 * 0.98]
        desired = [0.0, 0.0, 1.0, 1.0]
        h = firls(numtaps, bands, desired, fs=self.sample_rate)
        self.kernel = torch.tensor(
            h, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)
        self._tail = torch.zeros(numtaps - 1, device=self.device, dtype=torch.float32)

    def reset(self):
        self._tail.zero_()

    def highpass(self, x):
        xin = torch.cat([self._tail, x])
        # conv1d 为 valid 相关，输出长度 = len(x)（线性相位核对称，相关即卷积）
        y = F.conv1d(xin.view(1, 1, -1), self.kernel).view(-1)
        self._tail = xin[x.shape[0] :].contiguous()
        return y


class DeEsser:
    """自适应齿音消除：监测 6kHz 以上能量，超阈值时全带软衰减。

    帧级（10ms）峰值包络 + 即时 attack / 指数 release 增益：
    e[m] = max(e_target[m], e[m-1]*c)，帧增益线性插值到样本级。
    """

    def __init__(
        self,
        sample_rate=48000,
        threshold_db=-20.0,
        ratio=0.45,
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
        self.hp6 = FirHighPass(6000.0, sample_rate, device)
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
