"""
RVC 输入 AGC 自动增益控制（仅 numpy，无 torch 依赖）
==================================================
用于 Qt 音频回调（本地/服务器模式通用，改动仅作用于发送前的输入），
保证 GUI 启动时不加载 torch。
"""

import math

import numpy as np


class AutoGain:
    """输入 AGC（numpy，用于音频回调）。

    - 目标 RMS 电平 target_db；低于 gate_db 视为静音，增益冻结（不放大底噪）
    - 增益上下限 max_gain_db / min_gain_db
    - attack（压低过快）按 attack_time 平滑，release 按 release_time
    - 块内用 5ms 渐变应用新增益，避免块边界跳变
    """

    def __init__(
        self,
        sample_rate=48000,
        target_db=-16.0,
        max_gain_db=6.0,
        min_gain_db=-12.0,
        gate_db=-42.0,
        attack_time=0.03,
        release_time=0.5,
    ):
        self.sample_rate = int(sample_rate)
        self.target_db = float(target_db)
        self.max_gain_db = float(max_gain_db)
        self.min_gain_db = float(min_gain_db)
        self.gate_db = float(gate_db)
        self.attack_time = float(attack_time)
        self.release_time = float(release_time)
        self.gain = 1.0
        self.gain_db = 0.0

    def reset(self):
        self.gain = 1.0
        self.gain_db = 0.0

    def _coef(self, block_seconds, time_constant):
        """按块长折算的单极点平滑系数。"""
        return 1.0 - math.exp(-block_seconds / max(0.001, time_constant))

    def process(self, x):
        """x: 1D float32 numpy。返回增益后的 float32。"""
        x = np.asarray(x, dtype=np.float32)
        n = x.shape[0]
        if n == 0:
            return x
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        db = 20.0 * math.log10(rms + 1e-9)
        if db < self.gate_db:
            # 静音冻结增益（信号回来时从当前增益平滑恢复）
            return (x * np.float32(self.gain)).astype(np.float32, copy=False)
        desired_db = min(
            self.max_gain_db, max(self.min_gain_db, self.target_db - db)
        )
        block_seconds = n / self.sample_rate
        if desired_db < self.gain_db:
            # 需要压低：快速（attack）
            coef = self._coef(block_seconds, self.attack_time)
            new_gain_db = desired_db + (self.gain_db - desired_db) * (1.0 - coef)
        else:
            coef = self._coef(block_seconds, self.release_time)
            new_gain_db = self.gain_db + (desired_db - self.gain_db) * coef
        old_gain = self.gain
        new_gain = 10.0 ** (new_gain_db / 20.0)
        if abs(new_gain - old_gain) > 0.02:
            fade = min(n, max(1, int(0.005 * self.sample_rate)))
            ramp = np.linspace(old_gain, new_gain, fade, dtype=np.float32)
            out = x.copy()
            out[:fade] *= ramp
            out[fade:] *= np.float32(new_gain)
        else:
            out = x * np.float32(new_gain)
        self.gain_db = new_gain_db
        self.gain = new_gain
        return out.astype(np.float32, copy=False)
