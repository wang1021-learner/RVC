<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
实时变声客户端（Qt）+ 推理服务 + 离线转换<br><br>

[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[**更新日志**](./docs/cn/Changelog_CN.md) | [**English**](./docs/en/README.en.md) | [**中文简体**](./README.md)

</div>

本仓库是 RVC 的**推理侧实现**，不是 Gradio WebUI，也不包含训练套件（没有 `webui.py` / `go-webui.bat` / `train/`）。

当前入口：

| 用途 | 命令 |
| --- | --- |
| 实时变声桌面客户端 | `python realtime_qt.py` |
| 独立 / 远程推理服务 | `python server/rvc_server.py` |
| 离线音频转换 | `python convert_audio.py input.wav output.wav -m <模型.pth> -i <索引.index>` |

底模来自开源 VCTK；说话人权重与 FAISS 索引由用户放到 `assets/weights/`、`assets/indices/`。

## 简介

+ **实时变声**：PySide6 客户端采集麦克风，经本机或远程 GPU 推理后播出
+ **检索混合**：用 FAISS **Top-K（k=4）** 从训练集特征里取近邻，再按「像角色」比例（`index_rate`）与源 HuBERT 特征混合，减轻音色泄漏。关闭检索或索引缺失时，特征就是源 HuBERT，泄漏会回来
+ **音高**：默认 [RMVPE](https://github.com/Dream-High/RMVPE)（InterSpeech 2023），可选 FCPE / PM
+ **设备**：NVIDIA CUDA；Windows 上 AMD/Intel 可走 DirectML；否则 CPU
+ 本树**没有**网页训练界面、ckpt-merge、pymss/UVR 人声分离

## 环境配置

只支持 **Python 3.11 x64**。Windows 便携 `runtime/` 和 `install_local.bat` 是 **3.11.9**。不要用 3.10 / 3.12。源码开发用 `.venv`。请在仓库根目录执行。

### Ubuntu 24.04

默认解释器是 3.12，需要单独装 3.11：

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev ffmpeg unzip libsndfile1 libportaudio2

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

打包/单机版用 `runtime\python.exe`（3.11.9）。`start_server.bat` 会跳过不是 3.11 的 venv。

### 按硬件选择依赖

先装匹配的 Torch，再装本仓库依赖。不要用旧文档里的 `go-webui.bat` 或 Gradio 依赖。

| 硬件 | 安装方式 |
| --- | --- |
| CPU、AMD、Intel | 先装 CPU 版 Torch，再 `requirments_cpu_py311.txt`；Windows 可另装 `torch-directml` |
| NVIDIA RTX 50 系 | 先装 CUDA 12.8 版 Torch，再 `requirments_cu128_py311.txt` |
| NVIDIA RTX 50 系以前 | 先装 CUDA 11.8 版 Torch，再 `requirments_cu118_py311.txt` |

#### CPU、AMD、Intel

```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirments_cpu_py311.txt
```

Windows DirectML（可选）：

```bash
python -m pip install torch-directml
```

#### NVIDIA RTX 50 系

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu128_py311.txt
```

#### NVIDIA RTX 50 系以前

```bash
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu118_py311.txt
```

只跑客户端、Torch 已在别处装好时，也可用精简清单 `requirements.txt`。

检查 Torch：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

`requirments_*.txt` 顶部已写镜像。大陆用户可保留默认；改官方源时只替换 `--index-url` / `--extra-index-url`。

## 模型与运行目录

从 [Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) 下载公共权重，说话人模型和索引按族放在一起：

```text
assets/
├── hubert_base/
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── model.safetensors    # Transformers HuBERT
│   ├── final_proj.pt        # RVC v1 的 768→256 投影，v2 不用
│   ├── hubert_v1.onnx       # 可选
│   └── hubert_v2.onnx       # 可选
├── rmvpe/rmvpe.pt
├── weights/                 # 用户 .pth
└── indices/                 # 与模型同族的 .index，如 thchs_v2.index
```

### 下载公共模型

```bash
python -m pip install --upgrade huggingface_hub

hf download lj1995/VoiceConversionWebUI --revision main \
  --include "hubert_base/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
  --local-dir assets/rmvpe
```

若 Hugging Face 给的是 `pytorch_model.bin`，放到 `assets/hubert_base/` 即可，和 `model.safetensors` 二选一。v1 模型还必须有 `final_proj.pt`。

Windows AMD/Intel DirectML 还需要：

```bash
hf download lj1995/VoiceConversionWebUI rmvpe.onnx --revision main \
  --local-dir assets/rmvpe
```

### FFmpeg

Ubuntu 已在系统依赖里安装。Windows 可把下面两个文件放到项目根目录：

- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe?download=true)
- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe?download=true)

## 开始使用

### 1. 实时变声桌面客户端

```bash
python realtime_qt.py
```

### 2. 独立 / 远程推理服务器

```bash
# 默认 0.0.0.0:8765（局域网可连）。仅本机请加 --host 127.0.0.1
python server/rvc_server.py

python server/rvc_server.py --host 0.0.0.0 --port 8765
python server/rvc_server.py --cpu          # 无独显
```

Windows 也可用 `start_server.bat`。

### 3. 批量音频离线转换

```bash
python convert_audio.py input.wav output.wav \
  -m assets/weights/your_model.pth \
  -i assets/indices/your_index.index
```

`.pth` 放 `assets/weights/`，对应 `.index` 放 `assets/indices/`（按说话人族命名，例如 `thchs_v2.index`）。检索 k=4，与实时路径相同；`index_rate` 为 0 或不提供索引则不做检索。

## 参考项目

+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Vocal pitch extraction: RMVPE](https://github.com/Dream-High/RMVPE)
  + 预训练由 [yxlllc](https://github.com/yxlllc/RMVPE) 与 [RVC-Boss](https://github.com/RVC-Boss) 训练测试

## 感谢所有贡献者作出的努力
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
