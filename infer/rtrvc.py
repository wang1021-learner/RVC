import traceback
import os
from time import time as ttime
import faiss
import numpy as np
import parselmouth
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from infer.hubert import extract_hubert_features, load_hubert_model
from i18n.i18n import I18nAuto
from tools.cuda_graph import run_cuda_graph
from tools.model_assets import INDEX_TOPK


i18n = I18nAuto()

_ASSET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_RMVPE_PATH = os.path.join(_ASSET_ROOT, "rmvpe", "rmvpe.pt")


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


def _same_weight_file(a, b):
    """同一权重：绝对路径或文件名相同（客户端常只传 basename）。"""
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        if os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b)):
            return True
    except Exception:
        pass
    return os.path.basename(str(a)).lower() == os.path.basename(str(b)).lower()


def _fill_short_uv(f0, max_gap=3):
    """Keep unvoiced frames at 0. Only interpolate holes shorter than max_gap."""
    f0 = np.asarray(f0, dtype=np.float32).copy()
    voiced = f0 > 0
    if not np.any(voiced):
        return f0
    n = f0.shape[0]
    i = 0
    while i < n:
        if voiced[i]:
            i += 1
            continue
        j = i
        while j < n and not voiced[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            f0[i:j] = np.interp(np.arange(i, j), [i - 1, j], [f0[i - 1], f0[j]])
        i = j
    return f0


def _fill_short_uv_torch(f0, max_gap=3):
    """GPU：只填两侧都有浊音、且长度不超过 max_gap 的短洞。"""
    out = f0.reshape(-1)
    inf = out.new_full((), 1.0e6)
    left_val = out.clone()
    left_dist = torch.where(out > 0, torch.zeros_like(out), inf.expand_as(out))
    right_val = out.clone()
    right_dist = left_dist.clone()
    for _ in range(int(max_gap)):
        pv = torch.roll(left_val, 1)
        pd = torch.roll(left_dist, 1)
        pv[0] = 0
        pd[0] = inf
        take = (out <= 0) & (pv > 0) & ((pd + 1) < left_dist)
        left_val = torch.where(take, pv, left_val)
        left_dist = torch.where(take, pd + 1, left_dist)
        nv = torch.roll(right_val, -1)
        nd = torch.roll(right_dist, -1)
        nv[-1] = 0
        nd[-1] = inf
        take = (out <= 0) & (nv > 0) & ((nd + 1) < right_dist)
        right_val = torch.where(take, nv, right_val)
        right_dist = torch.where(take, nd + 1, right_dist)
    gap = left_dist + right_dist - 1
    fill = (out <= 0) & (left_dist < inf) & (right_dist < inf) & (gap <= max_gap)
    w = left_dist / (left_dist + right_dist).clamp(min=1e-6)
    filled = left_val * (1.0 - w) + right_val * w
    return torch.where(fill, filled, out)


def _smooth_f0_torch(f0, k=3, alpha=0.4, jump=0.3):
    """浊音段：3 点中值 + 邻帧限幅 + 一阶平滑。清音保持 0。"""
    x = f0.reshape(-1).float()
    if x.numel() < k:
        return x
    voiced = x > 0
    pad = k // 2
    xp = F.pad(x.view(1, 1, -1), (pad, pad), mode="replicate")
    med = xp.unfold(-1, k, 1).median(dim=-1).values.view(-1)
    med = torch.where(voiced, med, x)
    prev = torch.roll(med, 1)
    prev[0] = med[0]
    ratio = torch.where(
        (prev > 0) & voiced,
        (med - prev).abs() / prev.clamp(min=1.0),
        torch.zeros_like(med),
    )
    scale = (jump / ratio.clamp(min=1e-6)).clamp(max=1.0)
    capped = torch.where(ratio > jump, prev + (med - prev) * scale, med)
    sm = torch.where(voiced & (prev > 0), alpha * capped + (1.0 - alpha) * prev, capped)
    return torch.where(voiced, sm, torch.zeros_like(x))


def _smooth_f0_np(f0, k=3, alpha=0.4, jump=0.3):
    x = np.asarray(f0, dtype=np.float32).reshape(-1).copy()
    if x.size < k:
        return x
    voiced = x > 0
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    med = np.empty_like(x)
    for i in range(x.size):
        med[i] = np.median(xp[i:i + k])
    med = np.where(voiced, med, x)
    prev = np.roll(med, 1)
    prev[0] = med[0]
    ratio = np.zeros_like(med)
    mask = (prev > 0) & voiced
    ratio[mask] = np.abs(med[mask] - prev[mask]) / np.maximum(prev[mask], 1.0)
    capped = med.copy()
    over = ratio > jump
    capped[over] = prev[over] + (med[over] - prev[over]) * (
        jump / np.maximum(ratio[over], 1e-6))
    sm = capped.copy()
    ema = voiced & (prev > 0)
    sm[ema] = alpha * capped[ema] + (1.0 - alpha) * prev[ema]
    sm[~voiced] = 0.0
    return sm


def get_synthesizer(pth_path, device=torch.device("cpu")):
    from infer.module.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    cpt = torch.load(pth_path, map_location=torch.device("cpu"))
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


# config.device=torch.device("cpu")########强制cpu测试
# config.is_half=False########强制cpu测试
class RVC:
    def __init__(
        self,
        key,
        formant,
        pth_path,
        index_path,
        index_rate,
        config,
        last_rvc=None,
    ) :
        """
        初始化
        """
        try:
            # global config
            self.config = config
            # device="cpu"########强制cpu测试
            self.device = config.device
            self.f0_up_key = key
            self.formant_shift = formant
            self.f0_min = 50
            self.f0_max = 1100
            self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
            self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
            self.is_half = config.is_half
            self.index_gpu = None
            self._index_norm = None
            self._gpu_index_checked = False
            self.index_temp = 0.12  # Softmax 温度：略升减轻帧间硬切近邻
            reused_index = (
                last_rvc is not None
                and index_rate != 0
                and index_path
                and os.path.exists(index_path)
                and getattr(last_rvc, "index", None) is not None
                and _same_weight_file(getattr(last_rvc, "index_path", ""), index_path)
            )
            if reused_index:
                self.index = last_rvc.index
                self.big_npy = getattr(last_rvc, "big_npy", None)
                self.index_gpu = getattr(last_rvc, "index_gpu", None)
                self._index_norm = getattr(last_rvc, "_index_norm", None)
                self._gpu_index_checked = bool(
                    getattr(last_rvc, "_gpu_index_checked", False)
                )
                printt(i18n("已启用索引检索"))
            elif index_rate != 0 and index_path and os.path.exists(index_path):
                try:
                    self._load_faiss_index(index_path)
                    printt(i18n("已启用索引检索"))
                except Exception:
                    traceback.print_exc()
                    printt(i18n("索引检索失败"))
                    index_rate = 0
            else:
                index_rate = 0
            self.pth_path = pth_path
            self.index_path = index_path
            self.index_rate = index_rate
            self.cache_pitch = torch.zeros(
                1024, device=self.device, dtype=torch.long
            )
            self.cache_pitchf = torch.zeros(
                1024, device=self.device, dtype=torch.float32
            )
            self.infer_count = 0
            self.last_stage_ms = {}
            # 优化4：F0 提取专用 CUDA 流（与 HuBERT 并行）
            if (
                str(self.device).startswith("cuda")
                and os.environ.get("RVC_NO_F0_STREAM") != "1"
            ):
                self._f0_stream = torch.cuda.Stream(device=self.device)
            else:
                self._f0_stream = None
            self._f0_pending = None
            self._feat_cache = None
            self._index_mix_tail = None
            self._hubert_win = None
            self._p_len_tensor = None
            self._sid_tensor = None
            self.protect = 0.33

            self.resample_kernel = {}

            if last_rvc is None:
                self.model = load_hubert_model(self.device, self.is_half)
            else:
                self.model = last_rvc.model

            self.net_g = None

            def set_synthesizer():
                self.net_g, cpt = get_synthesizer(self.pth_path, self.device)
                self.tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                self.if_f0 = cpt.get("f0", 1)
                self.version = cpt.get("version", "v1")
                if self.is_half:
                    self.net_g = self.net_g.half()
                else:
                    self.net_g = self.net_g.float()

            if last_rvc is None or not _same_weight_file(
                getattr(last_rvc, "pth_path", ""), self.pth_path
            ):
                set_synthesizer()
            else:
                self.tgt_sr = last_rvc.tgt_sr
                self.if_f0 = last_rvc.if_f0
                self.version = last_rvc.version
                self.is_half = last_rvc.is_half
                self.net_g = last_rvc.net_g

            if last_rvc is not None and hasattr(last_rvc, "model_rmvpe"):
                self.model_rmvpe = last_rvc.model_rmvpe
            if last_rvc is not None and hasattr(last_rvc, "model_fcpe"):
                self.model_fcpe = last_rvc.model_fcpe
        except Exception:
            # 不吞异常：加载失败必须让调用方知道，避免半初始化对象后续崩溃
            traceback.print_exc()
            raise

    def reset_feat_cache(self):
        self._feat_cache = None
        self._index_mix_tail = None

    def _blend_index_mix(self, mixed):
        """相邻块检索结果交叉淡化，减轻角色帧间跳变。"""
        prev = getattr(self, "_index_mix_tail", None)
        if (
            prev is not None
            and mixed.shape[0] > 0
            and prev.dim() == mixed.dim()
            and prev.shape[-1] == mixed.shape[-1]
        ):
            n = min(4, int(mixed.shape[0]), int(prev.shape[0]))
            if n > 0:
                w = torch.linspace(
                    0.0, 1.0, n, device=mixed.device, dtype=mixed.dtype
                ).view(n, 1)
                mixed = mixed.clone()
                mixed[:n] = prev[-n:].to(dtype=mixed.dtype) * (1.0 - w) + mixed[:n] * w
        keep = min(4, int(mixed.shape[0]))
        if keep > 0:
            self._index_mix_tail = mixed[-keep:].detach().clone()
        else:
            self._index_mix_tail = None
        return mixed

    def change_key(self, new_key):
        self.f0_up_key = new_key

    def change_formant(self, new_formant):
        self.formant_shift = new_formant

    def _load_faiss_index(self, index_path):
        self.index = faiss.read_index(index_path)
        if hasattr(self.index, "nprobe"):
            try:
                self.index.nprobe = max(4, int(getattr(self.index, "nprobe", 1) or 1))
            except Exception:
                pass
        self.big_npy = None
        try:
            self.index.reconstruct(0)
        except Exception:
            faiss.downcast_index(self.index).make_direct_map()
            self.index.reconstruct(0)
        # 一次性预载全部索引向量，之后每块检索不再逐条 reconstruct_batch
        try:
            ntotal = int(getattr(self.index, "ntotal", 0) or 0)
            if ntotal > 0:
                self.big_npy = np.ascontiguousarray(
                    self.index.reconstruct_n(0, ntotal), dtype=np.float32
                )
        except Exception:
            self.big_npy = None

    def _gather_index_vectors(self, ix_safe, valid):
        if self.big_npy is not None:
            return self.big_npy[ix_safe]
        flat = np.ascontiguousarray(ix_safe.reshape(-1), dtype=np.int64)
        mask = valid.reshape(-1)
        vecs = np.zeros((flat.size, self.index.d), dtype=np.float32)
        ids = flat[mask]
        if ids.size:
            try:
                got = self.index.reconstruct_batch(ids)
            except Exception:
                got = np.stack([self.index.reconstruct(int(i)) for i in ids], axis=0)
            vecs[mask] = got
        return vecs.reshape(ix_safe.shape + (self.index.d,))

    def _ensure_gpu_index(self):
        """把预载的索引向量搬进显存做余弦 Top-K（一次），低显存/异常回退 CPU。"""
        if getattr(self, "_gpu_index_checked", False):
            return
        self._gpu_index_checked = True
        try:
            if os.environ.get("RVC_CPU_INDEX") == "1":
                return
            dev = torch.device(self.device)
            if dev.type != "cuda" or self.big_npy is None:
                return
            mem_gb = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
            if mem_gb < 3.5:
                return  # 显存紧张，保持 CPU 检索
            dtype = torch.float32 if mem_gb >= 6 else torch.float16
            idx_t = torch.from_numpy(self.big_npy).to(dev, dtype=dtype)
            norm = torch.nn.functional.normalize(idx_t.float(), dim=-1).t().contiguous()
            self._index_norm = norm.to(dtype=dtype)
            self.index_gpu = idx_t
        except Exception:
            self.index_gpu = None
            self._index_norm = None

    def _get_f0_gpu(self, x, f0_up_key, method):
        """F0 提取的 GPU 部分（不落 CPU），返回 (hidden, method) 供后续解码。"""
        if method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from infer.rmvpe import RMVPE

                printt(i18n("正在加载RMVPE模型"))
                self.model_rmvpe = RMVPE(
                    _RMVPE_PATH,
                    is_half=self.is_half,
                    device=self.device,
                )
            return self.model_rmvpe.infer_hidden(x), method
        raise ValueError(f"F0 method does not support GPU split: {method}")

    def _finish_f0(self, pending, f0_up_key):
        """F0 解码留在 GPU，避免每块 .cpu()。"""
        hidden, method = pending
        if method == "rmvpe":
            f0 = self.model_rmvpe.decode_torch(hidden, thred=0.03)
            f0 = _fill_short_uv_torch(f0)
            f0 = _smooth_f0_torch(f0)
            f0 = f0 * pow(2, f0_up_key / 12)
            return self.get_f0_post(f0)
        raise ValueError(f"F0 method does not support GPU split: {method}")

    def _feat_dim(self):
        return 768 if getattr(self, "version", "v2") == "v2" else 256

    def _extract_feats_incremental(self, wav_1d, block_frame_16k):
        """只跑最近固定窗的 HuBERT，结果拼进滚动缓存。窗长固定以便 CUDA Graph。"""
        hop = 320
        overlap = 4
        cnn_pad = 640
        new_frames = max(1, int(block_frame_16k) // hop)
        if self._hubert_win is None:
            win = (new_frames + overlap) * hop + cnn_pad
            self._hubert_win = int((win + 159) // 160 * 160)
        win = int(self._hubert_win)
        if self.config.is_half:
            wav = wav_1d.half().view(1, -1)
        else:
            wav = wav_1d.float().view(1, -1)
        full_frames = max(1, wav.shape[1] // hop)
        tail = wav[:, -min(win, wav.shape[1]):]
        if tail.shape[1] < win:
            tail = F.pad(tail, (win - tail.shape[1], 0))
        new_feats = extract_hubert_features(self.model, tail, self.version)
        dtype = new_feats.dtype
        dim = new_feats.shape[-1]
        if (
            self._feat_cache is None
            or self._feat_cache.shape[1] != full_frames
            or self._feat_cache.shape[2] != dim
        ):
            self._feat_cache = torch.zeros(
                1, full_frames, dim, device=self.device, dtype=dtype
            )
        if new_frames < full_frames:
            shifted = self._feat_cache[:, new_frames:, :]
        else:
            shifted = self._feat_cache[:, :0, :]
        need = full_frames - shifted.shape[1]
        if new_feats.shape[1] >= need:
            tail_feats = new_feats[:, -need:, :]
        else:
            # 兜底：正常不触发（HuBERT 对 win 样本的输出帧数总 ≥ need）。
            # 真发生说明 CNN 下采样取整异常，留日志便于定位，前端补零只是占位。
            print(
                "[rtrvc] HuBERT 输出帧数 %d < need=%d，出现占位补齐"
                % (new_feats.shape[1], need)
            )
            tail_feats = F.pad(new_feats, (0, 0, need - new_feats.shape[1], 0))
        self._feat_cache = torch.cat([shifted, tail_feats], dim=1).contiguous()
        return torch.cat((self._feat_cache, self._feat_cache[:, -1:, :]), 1)

    def change_index_rate(self, new_index_rate):
        if new_index_rate != 0 and not hasattr(self, "index"):
            if self.index_path and os.path.exists(self.index_path):
                try:
                    self._load_faiss_index(self.index_path)
                    printt(i18n("已启用索引检索"))
                except Exception:
                    printt(i18n("不支持从该索引重建向量，已禁用索引检索"))
                    new_index_rate = 0
            else:
                printt(i18n("未配置索引文件，忽略检索"))
                new_index_rate = 0
        self.index_rate = new_index_rate

    def get_f0_post(self, f0):
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0

    def get_f0(self, x, f0_up_key, method="rmvpe"):
        if method == "rmvpe":
            return self.get_f0_rmvpe(x, f0_up_key)
        if method == "fcpe":
            return self.get_f0_fcpe(x, f0_up_key)
        if method != "pm":
            raise ValueError(f"Unsupported F0 method: {method}")
        x = x.cpu().numpy()
        p_len = x.shape[0] // 160 + 1
        f0_min = 65
        l_pad = int(np.ceil(1.5 / f0_min * 16000))
        r_pad = l_pad + 1
        s = parselmouth.Sound(np.pad(x, (l_pad, r_pad)), 16000).to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=1100,
        )
        assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
        f0 = s.selected_array["frequency"]
        if len(f0) < p_len:
            f0 = np.pad(f0, (0, p_len - len(f0)))
        f0 = f0[:p_len]
        f0 = _fill_short_uv(f0)
        f0 = _smooth_f0_np(f0)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_rmvpe(self, x, f0_up_key):
        if hasattr(self, "model_rmvpe") == False:
            from infer.rmvpe import RMVPE

            printt(i18n("正在加载RMVPE模型"))
            self.model_rmvpe = RMVPE(
                _RMVPE_PATH,
                is_half=self.is_half,
                device=self.device,
            )
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0 = _fill_short_uv(f0)
        f0 = _smooth_f0_np(f0)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_fcpe(self, x, f0_up_key):
        if hasattr(self, "model_fcpe") == False:
            from infer.fcpe import FCPEInfer

            printt("Loading fcpe model")
            self.model_fcpe = FCPEInfer(self.device)
        f0 = self.model_fcpe.infer(
            x.unsqueeze(0).float(),
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        ).squeeze().detach().cpu().numpy()
        f0 = _fill_short_uv(f0)
        f0 = _smooth_f0_np(f0)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def infer(
        self,
        input_wav,
        block_frame_16k,
        skip_head,
        return_length,
        f0method,
    ) :
        report_status = self.infer_count < 3 or self.infer_count % 100 == 0
        self.infer_count += 1
        t1 = ttime()
        with torch.no_grad():
            # ── F0：固定 2560 点窗（160ms@16k），与 HuBERT 并行 ──
            self._f0_pending = None
            self._f0_event = None
            f0_key = self.f0_up_key - self.formant_shift
            if self.if_f0 == 1:
                if f0method == "rmvpe":
                    f0_win = 2560
                    n = input_wav.shape[0]
                    if n >= f0_win:
                        f0_tail = input_wav[-f0_win:]
                    else:
                        f0_tail = F.pad(input_wav, (f0_win - n, 0))
                else:
                    f0_extractor_frame = block_frame_16k + 800
                    f0_tail = input_wav[-min(f0_extractor_frame, input_wav.shape[0]):]
                if self._f0_stream is not None and f0method == "rmvpe":
                    try:
                        with torch.cuda.stream(self._f0_stream):
                            self._f0_stream.wait_stream(torch.cuda.current_stream())
                            self._f0_pending = self._get_f0_gpu(f0_tail, f0_key, f0method)
                            self._f0_event = torch.cuda.Event()
                            self._f0_event.record(self._f0_stream)
                    except Exception:
                        self._f0_pending = None
                if self._f0_pending is None:
                    pitch, pitchf = self.get_f0(f0_tail, f0_key, f0method)
            # 短窗增量默认关（稳）；RVC_INCREMENTAL_HUBERT=1 或 incremental_hubert=True 才开
            use_inc = getattr(self, "incremental_hubert", None)
            if use_inc is None:
                use_inc = os.environ.get("RVC_INCREMENTAL_HUBERT", "0") == "1"
            if use_inc:
                feats = self._extract_feats_incremental(input_wav, block_frame_16k)
            else:
                src = input_wav.half() if self.config.is_half else input_wav.float()
                feats = extract_hubert_features(self.model, src.view(1, -1), self.version)
                feats = torch.cat((feats, feats[:, -1:, :]), 1)
            protect = float(getattr(self, "protect", 0.33) or 0.33)
            use_protect = bool(self.if_f0 == 1 and protect < 0.5)
            feats0 = feats.clone() if use_protect else None
        t2 = ttime()
        try:
            if hasattr(self, "index") and self.index_rate != 0:
                # HuBERT hop=320@16k。搜本块新帧 + 4 帧重叠，便于跨块淡化
                n_new = max(1, int(block_frame_16k) // 320 + 4)
                start = max(int(skip_head) // 2, feats.shape[1] - n_new)
                # ── 优化3：GPU 余弦 Top-K + 温度 Softmax（低显存自动回退 CPU）──
                self._ensure_gpu_index()
                if self.index_gpu is not None:
                    q = feats[0][start:]
                    qn = torch.nn.functional.normalize(q.float(), dim=-1)
                    if self._index_norm.dtype == torch.float16:
                        qn = qn.half()
                    sim = qn @ self._index_norm
                    top_sim, top_ix = sim.topk(k=INDEX_TOPK, dim=-1)
                    w = torch.softmax(top_sim / self.index_temp, dim=-1)
                    retrieved = (w.unsqueeze(-1) * self.index_gpu[top_ix]).sum(-2)
                    mixed = feats[0][start:]
                    mixed = (
                        retrieved.to(dtype=mixed.dtype) * self.index_rate
                        + (1.0 - self.index_rate) * mixed
                    )
                    mixed = self._blend_index_mix(mixed)
                    feats[0][start:] = mixed
                else:
                    npy = feats[0][start:].cpu().numpy().astype("float32")
                    score, ix = self.index.search(npy, k=INDEX_TOPK)
                    valid = ix >= 0
                    if valid.any():
                        weight = np.square(1.0 / (np.maximum(score, 0.0) + 1e-6))
                        weight = np.where(valid, weight, 0.0)
                        denom = weight.sum(axis=1, keepdims=True)
                        use = denom[:, 0] > 0
                        weight[use] /= denom[use]
                        ix_safe = np.where(valid, ix, 0)
                        retrieved = np.sum(
                            self._gather_index_vectors(ix_safe, valid)
                            * np.expand_dims(weight, axis=2),
                            axis=1,
                        )
                        if self.config.is_half:
                            retrieved = retrieved.astype("float16")
                        mixed = feats[0][start:]
                        retrieved_t = torch.from_numpy(retrieved).to(
                            device=mixed.device, dtype=mixed.dtype
                        )
                        use_t = torch.from_numpy(use).to(mixed.device)
                        mixed[use_t] = (
                            retrieved_t[use_t] * self.index_rate
                            + (1.0 - self.index_rate) * mixed[use_t]
                        )
                        mixed = self._blend_index_mix(mixed)
                        feats[0][start:] = mixed
                    elif report_status:
                        printt(i18n("索引检索失败或未启用"))
            else:
                if report_status:
                    printt(i18n("索引检索失败或未启用"))
        except Exception:
            traceback.print_exc()
            printt(i18n("索引检索失败"))
        t3 = ttime()
        p_len = input_wav.shape[0] // 160
        factor = pow(2, self.formant_shift / 12)
        return_length2 = int(np.ceil(return_length * factor))
        if self.if_f0 == 1:
            if self._f0_pending is not None:
                # 等 F0 辅助流完成，再取回解码（HuBERT/检索已与其并行执行）
                try:
                    torch.cuda.current_stream().wait_event(self._f0_event)
                    pitch, pitchf = self._finish_f0(self._f0_pending, f0_key)
                except Exception:
                    traceback.print_exc()
                    pitch, pitchf = self.get_f0(f0_tail, f0_key, f0method)
            shift = block_frame_16k // 160
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            if pitch.shape[0] >= 4:
                p_slice = pitch[3:-1]
                pf_slice = pitchf[3:-1]
                n_elem = p_slice.shape[0]
                if n_elem > 0:
                    self.cache_pitch[-n_elem:] = p_slice
                    self.cache_pitchf[-n_elem:] = pf_slice
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = self.cache_pitchf[None, -p_len:] * return_length2 / return_length
        t4 = ttime()
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :].contiguous()
        if use_protect and feats0 is not None and self.if_f0 == 1:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            feats0 = feats0[:, :p_len, :].contiguous()
            pf = cache_pitchf.reshape(-1)
            n = min(int(feats.shape[1]), int(feats0.shape[1]), int(pf.shape[0]))
            if n > 0:
                one = torch.ones((), device=feats.device, dtype=feats.dtype)
                prot = torch.tensor(protect, device=feats.device, dtype=feats.dtype)
                w = torch.where(pf[:n] > 0, one, prot).view(1, n, 1)
                mixed = feats[:, :n, :] * w + feats0[:, :n, :] * (1.0 - w)
                if n == feats.shape[1]:
                    feats = mixed.contiguous()
                else:
                    feats = feats.clone()
                    feats[:, :n, :] = mixed
        if getattr(self, "_p_len_value", None) != p_len:
            self._p_len_value = p_len
            self._p_len_tensor = torch.tensor([p_len], device=self.device, dtype=torch.long)
        if self._sid_tensor is None:
            self._sid_tensor = torch.tensor([0], device=self.device, dtype=torch.long)
        p_len_tensor = self._p_len_tensor
        sid = self._sid_tensor
        skip_head_value = int(skip_head)
        return_length_value = int(return_length)
        return_length2_value = int(return_length2)
        with torch.no_grad():
            if self.if_f0 == 1:
                infered_audio = run_cuda_graph(
                    self.net_g,
                    "rvc-realtime-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, coarse, continuous, speaker: self.net_g.infer(
                        phone,
                        lengths,
                        coarse,
                        continuous,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                )
            else:
                infered_audio = run_cuda_graph(
                    self.net_g,
                    "rvc-realtime-no-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, speaker: self.net_g.infer(
                        phone,
                        lengths,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    sid,
                )
        infered_audio = infered_audio.squeeze(1).float()
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            if upp_res not in self.resample_kernel:
                self.resample_kernel[upp_res] = Resample(
                    orig_freq=upp_res,
                    new_freq=self.tgt_sr // 100,
                    dtype=torch.float32,
                ).to(self.device)
            infered_audio = self.resample_kernel[upp_res](
                infered_audio[:, : return_length * upp_res]
            )
        t5 = ttime()
        # 分阶段耗时（毫秒），供 UI 仪表盘/排障使用；print 保持原有节流
        self.last_stage_ms = {
            "feature": round((t2 - t1) * 1000.0, 2),
            "index": round((t3 - t2) * 1000.0, 2),
            "pitch": round((t4 - t3) * 1000.0, 2),
            "model": round((t5 - t4) * 1000.0, 2),
        }
        if report_status:
            printt(
                i18n("耗时：特征=%.3f秒，索引=%.3f秒，音高=%.3f秒，模型=%.3f秒"),
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
            )
        return infered_audio.squeeze()
