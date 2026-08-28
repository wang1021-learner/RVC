"""
RVC 输出保护：直流高通 + 帧级软限幅（torch/GPU）
================================================
仅被 worker/rvc_pipeline.py 导入（推理链路），GUI 不加载本模块，
保持启动时延迟加载 torch 的架构。
"""

import math

import torch
import torch.nn.functional as F


class OutputProtector:
    """直流阻断 + 峰值软限幅（GPU，状态跨块连续）。

    设计要点：
    - 直流高通: y[n] = x[n] - x[n-1] + r*y[n-1]，r = exp(-2*pi*fc/fs)
      块内用「指数缩放 cumsum」向量化递推（分 CHUNK 样本段，数值安全）。
    - 限幅器: 按 10ms 帧算峰值（与静音门 `_gate_last_block` 同粒度），
      帧增益 target = clamp(thresh/peak)，attack 即时、release 单极点，
      帧增益经线性插值到样本级，避免块边界增益跳变产生咔哒声。
    - 不引入 lookahead，零额外延迟。
    """

    _CHUNK = 8192  # 直流递推分段长度（数值稳定）

    def __init__(
        self,
        sample_rate=48000,
        threshold_db=-1.0,
        release_time=0.25,
        dc_hz=20.0,
        device="cpu",
    ):
        self.sample_rate = int(sample_rate)
        self.threshold_db = float(threshold_db)
        self.threshold = 10.0 ** (self.threshold_db / 20.0)
        self.device = torch.device(device)

        self.zc = max(1, self.sample_rate // 100)  # 10ms 帧
        self.release_time = float(release_time)
        self.dc_hz = float(dc_hz)

        # 直流高通系数
        self._r = math.exp(-2.0 * math.pi * self.dc_hz / self.sample_rate)
        # 帧级 release 单极点系数（每 10ms 一帧）
        self._frame_release = math.exp(-1.0 / (self.release_time * 100.0))

        self._prev_x = torch.zeros(1, device=self.device, dtype=torch.float32)
        self._prev_y = torch.zeros(1, device=self.device, dtype=torch.float32)
        self._env = torch.zeros(1, device=self.device, dtype=torch.float32)
        self._gain_last = torch.ones(1, device=self.device, dtype=torch.float32)

    def reset(self):
        self._prev_x.zero_()
        self._prev_y.zero_()
        self._env.zero_()
        self._gain_last.fill_(1.0)

    def set_threshold_db(self, value):
        self.threshold_db = float(value)
        self.threshold = 10.0 ** (self.threshold_db / 20.0)

    def _dc_block_recurrence(self, x):
        """因果递推 y[n] = x[n]-x[n-1] + r*y[n-1] 的块内向量化。

        y[n] = sum_{k=0..n} r^(n-k)*d[k] + r^(n+1)*y[-1]，d = x - prev(x)
        => y[n]*r^(-n) = sum_{k<=n} d[k]*r^(-k) + r*y[-1]
        分 CHUNK 段计算，段间传递 prev_x/prev_y 状态。
        """
        r = self._r
        n = x.shape[0]
        out = torch.empty_like(x)
        offset = 0
        while offset < n:
            seg = x[offset : offset + self._CHUNK]
            m = seg.shape[0]
            prev_x = self._prev_x if offset == 0 else x[offset - 1 : offset]
            d = seg - torch.cat([prev_x, seg[:-1]])
            k = torch.arange(m, device=x.device, dtype=x.dtype)
            rk = torch.pow(torch.full((m,), r, device=x.device, dtype=x.dtype), k)
            y = (torch.cumsum(d / rk, dim=0) + r * self._prev_y[0]) * rk
            out[offset : offset + m] = y
            self._prev_y.copy_(y[-1:])
            offset += m
        self._prev_x.copy_(x[-1:])
        return out

    def _limit(self, x):
        """帧级峰值软限幅，返回处理后的 x。

        峰值包络 env[m] = max(frame_peak[m], env[m-1]*rel)，env 初值 0：
        信号超过阈值时增益即时下降（attack=0），之后包络按 release 衰减、
        增益平滑回升。递推用「指数缩放 cummax」向量化，跨块状态连续。
        """
        zc = self.zc
        n = x.shape[0]
        if n < zc:
            peak = x.abs().max()
            g_now = self._gain_last[0]
            if peak > self.threshold:
                g_now = torch.minimum(g_now, self.threshold / peak.clamp(min=1e-6))
                self._env.fill_(peak)
            else:
                g_now = g_now + (1.0 - g_now) * (1.0 - self._frame_release)
            self._gain_last.fill_(g_now)
            return x * g_now
        n_frames = n // zc
        use = n_frames * zc
        tail = x[use:]
        body = x[:use].view(n_frames, zc)
        peaks = body.abs().amax(dim=1)
        rel = self._frame_release
        m = torch.arange(n_frames, device=x.device, dtype=x.dtype)
        rel_m = torch.pow(
            torch.full((n_frames,), rel, device=x.device, dtype=x.dtype), m
        )
        env = torch.cummax(peaks / rel_m, dim=0).values * rel_m
        env = torch.maximum(env, self._env[0] * rel * rel_m)
        self._env.copy_(env[-1:])
        gains = torch.where(
            env > self.threshold,
            self.threshold / env.clamp(min=1e-6),
            torch.ones_like(env),
        )
        # 逐帧从上一增益 lerp 到本帧，控制点落在帧末，避免整块插值漏峰
        ends = torch.cat([self._gain_last.reshape(1).to(dtype=gains.dtype), gains])
        w = torch.arange(1, zc + 1, device=x.device, dtype=x.dtype) / float(zc)
        g_lin = (ends[:-1].unsqueeze(1) * (1.0 - w) + ends[1:].unsqueeze(1) * w).reshape(-1)
        self._gain_last.copy_(gains[-1:])
        out = body.reshape(-1) * g_lin
        if tail.numel():
            out = torch.cat([out, tail * gains[-1]])
        return out

    def process(self, x):
        """x: 1D float32 tensor（任意设备）。"""
        x = self._dc_block_recurrence(x)
        return self._limit(x)
