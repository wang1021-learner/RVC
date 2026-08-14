#!/usr/bin/env python3
"""
RVC 说话人音色模型一键训练脚本
用法: python train_speaker.py --dataset datasets/thchs_female --name thchs_v2 --epochs 200
"""
import os
import sys
import argparse
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd, env=None):
    """运行子进程命令，实时打印输出"""
    print(f"\n{'='*60}")
    print(f"执行: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    result = subprocess.run(cmd, env=merged_env, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"错误: 命令执行失败 (返回码 {result.returncode})")
        sys.exit(1)


def step1_slice(dataset_dir, exp_name, sr=48000):
    """Step 1: 音频切片与预处理"""
    print("\n[Step 1/6] 音频切片与预处理")
    log_dir = os.path.join("logs", exp_name)
    out_48k = os.path.join(log_dir, "0_gt_wavs")
    out_16k = os.path.join(log_dir, "1_16k_wavs")
    os.makedirs(out_48k, exist_ok=True)
    os.makedirs(out_16k, exist_ok=True)

    from scipy import signal
    import librosa

    audio_files = list(Path(dataset_dir).glob("*.wav"))
    if not audio_files:
        print(f"错误: {dataset_dir} 下未找到 wav 文件")
        sys.exit(1)

    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=sr)
    success = 0
    for idx, audio_path in enumerate(audio_files):
        try:
            wav, _ = librosa.load(str(audio_path), sr=sr, mono=True)
            wav = signal.lfilter(bh, ah, wav)
            if len(wav) / sr < 1.0:
                continue
            peak = np.abs(wav).max()
            if peak <= 0 or not np.isfinite(peak):
                continue
            wav_norm = wav / peak * 0.9 * 0.75 + (1 - 0.75) * wav
            sf.write(os.path.join(out_48k, audio_path.name), wav_norm.astype(np.float32), sr)
            wav_16k = librosa.resample(wav_norm, orig_sr=sr, target_sr=16000)
            sf.write(os.path.join(out_16k, audio_path.name), wav_16k.astype(np.float32), 16000)
            success += 1
        except Exception as e:
            print(f"  跳过 {audio_path.name}: {e}")
    print(f"  完成: {success} 个有效音频 → {out_48k}/, {out_16k}/")
    return success


def step2_extract_f0(exp_name, version="v2"):
    """Step 2: F0 基频提取 (使用官方 extract_f0.py)"""
    print("\n[Step 2/6] F0 基频提取")
    exp_dir = os.path.join("logs", exp_name)
    run([
        sys.executable, os.path.join("train", "dataset", "extract_f0.py"),
        "cpu", exp_dir, "1", "rmvpe",
    ], env={"PYTHONPATH": PROJECT_ROOT})
    print(f"  完成 → {exp_dir}/2a_f0/, {exp_dir}/2b-f0nsf/")


def step3_extract_hubert(exp_name, version="v2"):
    """Step 3: HuBERT 内容特征提取"""
    print("\n[Step 3/6] HuBERT 内容特征提取")
    exp_dir = os.path.join("logs", exp_name)
    run([
        sys.executable, os.path.join("train", "dataset", "extract_hubert_feature.py"),
        "cuda", "1", "0", exp_dir, version, "false",
    ], env={"PYTHONPATH": PROJECT_ROOT})
    feat_dir = os.path.join(exp_dir, "3_feature768" if version == "v2" else "3_feature256")
    print(f"  完成 → {feat_dir}/")


def step4_gen_filelist(exp_name, version="v2"):
    """Step 4: 生成训练文件列表"""
    print("\n[Step 4/6] 生成训练文件列表")
    exp_dir = os.path.join("logs", exp_name)
    gt_dir = os.path.join(exp_dir, "0_gt_wavs")
    feat_dir = os.path.join(exp_dir, "3_feature768" if version == "v2" else "3_feature256")
    f0a_dir = os.path.join(exp_dir, "2a_f0")
    f0b_dir = os.path.join(exp_dir, "2b-f0nsf")
    filelist_path = os.path.join(exp_dir, "filelist.txt")

    wav_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".wav")])
    with open(filelist_path, "w") as f:
        for name in wav_files:
            base = name.replace(".wav", "")
            f.write("|".join([
                os.path.join(gt_dir, name),
                os.path.join(feat_dir, base + ".npy"),
                os.path.join(f0a_dir, name + ".npy"),
                os.path.join(f0b_dir, name + ".npy"),
                "0",
            ]) + "\n")
    print(f"  完成: {len(wav_files)} 条 → {filelist_path}")


def step5_train(exp_name, total_epoch, save_every, sr="48k", version="v2",
                pretrain_g=None, pretrain_d=None):
    """Step 5: 模型训练"""
    print("\n[Step 5/6] 模型训练")
    cmd = [
        sys.executable, "-m", "train.train",
        "-se", str(save_every),
        "-te", str(total_epoch),
        "-g", "0",
        "-bs", "4",
        "-e", exp_name,
        "-sr", sr,
        "-sw", "1",
        "-v", version,
        "-f0", "1",
        "-l", "0",
        "-c", "0",
    ]
    if pretrain_g:
        cmd += ["-pg", pretrain_g]
    if pretrain_d:
        cmd += ["-pd", pretrain_d]
    run(cmd, env={"PYTHONPATH": PROJECT_ROOT})


def step6_export(exp_name, total_epoch, sr="48k", version="v2"):
    """Step 6: 导出推理模型 + 训练 FAISS 索引"""
    print("\n[Step 6/6] 导出推理模型 + 训练 FAISS 索引")

    # 6a. 找到最新的 G_*.pth 检查点
    exp_dir = os.path.join("logs", exp_name)
    ckpt_files = sorted(
        [f for f in os.listdir(exp_dir) if f.startswith("G_") and f.endswith(".pth")
         and f != "G_2333333.pth"],
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )
    if not ckpt_files:
        print("错误: 未找到生成器检查点 G_*.pth")
        sys.exit(1)
    latest_ckpt = ckpt_files[-1]
    global_step = int(latest_ckpt.split("_")[1].split(".")[0])
    epoch = global_step // 66  # 66 steps per epoch for batch_size=4
    ckpt_path = os.path.join(exp_dir, latest_ckpt)

    # 6b. 导出推理模型
    from train.process_ckpt import extract_small_model
    model_name = f"{exp_name}_{epoch}e"
    extract_small_model(
        path=ckpt_path,
        name=model_name,
        sr=sr,
        if_f0="1",
        info=f"{epoch}epoch_finetune_fp32",
        version=version,
    )
    model_path = os.path.join("assets", "weights", f"{model_name}.pth")
    print(f"  推理模型: {model_path}")

    # 6c. 训练 FAISS 索引
    print("\n  训练 FAISS 索引...")
    index_root = os.path.join("logs", exp_name)
    run([
        sys.executable, "-m", "train.train_index",
        exp_name, version, index_root, "1",
    ], env={"PYTHONPATH": PROJECT_ROOT})

    # 找到生成的索引文件
    index_files = [f for f in os.listdir(exp_dir) if f.startswith("added_IVF") and f.endswith(".index")]
    index_path = os.path.join(exp_dir, index_files[0]) if index_files else "未找到"

    # 打印结果摘要
    print(f"\n{'='*60}")
    print(f"训练完成！输出文件:")
    print(f"{'='*60}")
    print(f"  推理模型:  {model_path}")
    print(f"  FAISS索引: {index_path}")
    print(f"  检查点:    {ckpt_path}")
    print(f"  文件列表:  {os.path.join(exp_dir, 'filelist.txt')}")
    print(f"\n使用方法:")
    print(f"  RVC_FORCE_FP32=1 python webui.py --port 7865")
    print(f"  在 WebUI 中选择模型 {model_name}.pth 和索引文件")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="RVC 说话人音色模型一键训练")
    parser.add_argument("--dataset", required=True, help="原始音频数据集目录 (含 wav 文件)")
    parser.add_argument("--name", required=True, help="实验名称 (如 thchs_v2)")
    parser.add_argument("--epochs", type=int, default=200, help="训练总轮数 (默认 200)")
    parser.add_argument("--save-every", type=int, default=50, help="每多少轮保存一次 (默认 50)")
    parser.add_argument("--sr", default="48k", choices=["32k", "40k", "48k"], help="采样率 (默认 48k)")
    parser.add_argument("--version", default="v2", choices=["v1", "v2"], help="模型版本 (默认 v2)")
    parser.add_argument("--pretrained-g", default=None, help="预训练生成器路径 (默认使用 assets/pretrained_v2/f0G48k.pth)")
    parser.add_argument("--pretrained-d", default=None, help="预训练判别器路径 (默认使用 assets/pretrained_v2/f0D48k.pth)")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过预处理 (切片+F0+特征)，直接训练")
    args = parser.parse_args()

    # 自动设置预训练模型路径
    if args.pretrained_g is None:
        default_g = os.path.join("assets", "pretrained_v2", f"f0G{args.sr}.pth")
        if os.path.exists(default_g):
            args.pretrained_g = default_g
    if args.pretrained_d is None:
        default_d = os.path.join("assets", "pretrained_v2", f"f0D{args.sr}.pth")
        if os.path.exists(default_d):
            args.pretrained_d = default_d

    print(f"数据集:   {args.dataset}")
    print(f"实验名称: {args.name}")
    print(f"训练轮数: {args.epochs}")
    print(f"采样率:   {args.sr}")
    print(f"版本:     {args.version}")
    print(f"预训练G:  {args.pretrained_g or '无'}")
    print(f"预训练D:  {args.pretrained_d or '无'}")

    if not args.skip_preprocess:
        step1_slice(args.dataset, args.name, sr=int(args.sr.replace("k", "000")))
        step2_extract_f0(args.name, args.version)
        step3_extract_hubert(args.name, args.version)
        step4_gen_filelist(args.name, args.version)

    step5_train(args.name, args.epochs, args.save_every, args.sr, args.version,
                args.pretrained_g, args.pretrained_d)
    step6_export(args.name, args.epochs, args.sr, args.version)


if __name__ == "__main__":
    main()
