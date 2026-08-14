#!/usr/bin/env python3
"""
批量音频转换脚本 - 使用训练好的模型进行音色转换

使用方法:
    python convert_audio.py <输入音频文件> [输出音频文件]

示例:
    python convert_audio.py input.wav output.wav
    python convert_audio.py input.wav (自动命名为input_converted.wav)
"""

import os
import sys
import wave
import argparse
import numpy as np
import torch
import librosa
import faiss

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from infer.module.models import SynthesizerTrnMs768NSFsid
from infer.hubert import extract_hubert_features, load_hubert_model
from infer.rmvpe import RMVPE


def load_model(pth_path: str):
    cpt = torch.load(pth_path, map_location="cpu")
    net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float().cuda().eval()
    net_g.remove_weight_norm()
    return net_g, cpt["config"][-1]


def convert_audio(
    input_path: str,
    output_path: str = None,
    pth_path: str = "assets/weights/thchs_female_100e.pth",
    index_path: str = "logs/thchs_v2/added_IVF2716_Flat_nprobe_1_thchs_v2_v2.index",
    index_rate: float = 0.5,
    pitch: int = 0,
):
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_converted{ext}"

    print(f"加载模型: {pth_path}")
    net_g, tgt_sr = load_model(pth_path)

    print("加载 HuBERT 模型...")
    hubert_model = load_hubert_model(torch.device("cuda"), is_half=False)

    print("加载 RMVPE 模型...")
    rmvpe_model = RMVPE("assets/rmvpe/rmvpe.pt", is_half=False, device=torch.device("cuda"))

    index = None
    index_vectors = None
    if os.path.exists(index_path):
        print(f"加载 FAISS 索引: {index_path}")
        index = faiss.read_index(index_path)
        index_vectors = index.reconstruct_n(0, index.ntotal)
    else:
        print(f"警告: 找不到索引文件 {index_path}，跳过索引检索")

    print(f"加载输入音频: {input_path}")
    sr, audio = _load_audio(input_path)

    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    print("提取 HuBERT 特征...")
    feats = torch.from_numpy(audio_16k).float().cuda().view(1, -1)
    feats = extract_hubert_features(hubert_model, feats, "v2")

    if index is not None and index_rate > 0:
        print(f"执行索引检索 (rate={index_rate})...")
        npy = feats[0].cpu().numpy()
        score, ix = index.search(npy, k=8)
        if (ix >= 0).all():
            weight = np.square(1 / np.maximum(score, 1e-6))
            weight /= weight.sum(axis=1, keepdims=True)
            npy_new = np.sum(index_vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
            npy_mixed = index_rate * npy_new + (1 - index_rate) * npy
            feats[0] = torch.from_numpy(npy_mixed).unsqueeze(0).cuda()

    print("提取 F0...")
    f0_result = rmvpe_model.infer_from_audio(
        torch.from_numpy(audio_16k).float().cuda(), thred=0.03
    )
    if isinstance(f0_result, np.ndarray):
        f0 = torch.from_numpy(f0_result).float().cuda()
    else:
        f0 = f0_result.float().cuda()

    uv = f0 == 0
    if uv.any():
        f0_np = f0.cpu().numpy()
        uv_np = uv.cpu().numpy()
        f0_np[uv_np] = np.interp(
            np.where(uv_np)[0], np.where(~uv_np)[0], f0_np[~uv_np]
        )
        f0 = torch.from_numpy(f0_np).cuda()

    if pitch != 0:
        f0 = f0 * (2 ** (pitch / 12))

    p_len = f0.shape[0]
    feats = torch.nn.functional.interpolate(
        feats.permute(0, 2, 1), scale_factor=2
    ).permute(0, 2, 1)
    if feats.shape[1] > p_len:
        feats = feats[:, :p_len, :]
    elif feats.shape[1] < p_len:
        feats = torch.nn.functional.pad(feats, (0, 0, 0, p_len - feats.shape[1]))

    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + 50 / 700)
    f0_mel_max = 1127 * np.log(1 + 1100 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = torch.round(f0_mel).long()

    print("生成转换音频...")
    with torch.no_grad():
        sid = torch.LongTensor([0]).cuda()
        infered_audio = net_g.infer(
            feats,
            torch.LongTensor([p_len]).cuda(),
            f0_coarse.unsqueeze(0),
            f0.unsqueeze(0),
            sid,
        )[0]

    output = infered_audio.squeeze().cpu().numpy()

    print(f"保存输出: {output_path}")
    _save_audio(output_path, output, tgt_sr)
    print(f"转换完成! 输出文件: {output_path}")
    return output_path


def _load_audio(path: str):
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return sr, audio


def _save_audio(path: str, audio: np.ndarray, sr: int):
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVC 音色转换工具")
    parser.add_argument("input", help="输入音频文件路径")
    parser.add_argument("-o", "--output", help="输出音频文件路径")
    parser.add_argument(
        "-m",
        "--model",
        default="assets/weights/thchs_female_100e.pth",
        help="模型权重文件路径",
    )
    parser.add_argument(
        "-i",
        "--index",
        default="logs/thchs_v2/added_IVF2716_Flat_nprobe_1_thchs_v2_v2.index",
        help="FAISS 索引文件路径",
    )
    parser.add_argument(
        "--index-rate",
        type=float,
        default=0.5,
        help="索引检索比例 (0.0-1.0, 默认0.5)",
    )
    parser.add_argument(
        "--pitch",
        type=int,
        default=0,
        help="音高调整（半音，男转女+12，女转男-12）",
    )

    args = parser.parse_args()

    convert_audio(
        input_path=args.input,
        output_path=args.output,
        pth_path=args.model,
        index_path=args.index,
        index_rate=args.index_rate,
        pitch=args.pitch,
    )
