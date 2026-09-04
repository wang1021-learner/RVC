"""采集端轻量降噪：噪声底跟踪 + 软门。无额外模型，走音频工作线程。"""
import numpy as np


class CaptureDenoise:
    def __init__(self, open_db=-48.0):
        self.open_db = float(open_db)
        self.noise = 10.0 ** (open_db / 20.0)
        self.gain = 1.0
        self.enabled = True

    def reset(self):
        self.noise = 10.0 ** (self.open_db / 20.0)
        self.gain = 1.0

    def process(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if not self.enabled or x.size == 0:
            return x
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms < self.noise * 1.8:
            self.noise = 0.92 * self.noise + 0.08 * rms
        thresh = max(self.noise * 3.5, 10.0 ** (self.open_db / 20.0))
        if rms >= thresh:
            target = 1.0
        else:
            target = max(0.0, (rms / (thresh + 1e-8) - 0.2) / 0.8)
        # 慢一点开关，避免把字头切掉、听起来像卡顿
        self.gain = 0.82 * self.gain + 0.18 * target
        if self.gain >= 0.98:
            return x
        return x * np.float32(self.gain)
