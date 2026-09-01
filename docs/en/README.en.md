<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
Realtime voice-changer client (Qt) + inference server + offline conversion<br><br>

[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[**Changelog**](./Changelog_EN.md) | [**English**](../en/README.en.md) | [**简体中文**](../../README.md)

</div>

This tree is the **inference** side of RVC. It is not a Gradio WebUI and it does not ship a training suite (no `webui.py`, `go-webui.bat`, or `train/`).

Current entry points:

| Role | Command |
| --- | --- |
| Realtime desktop client | `python realtime_qt.py` |
| Standalone / remote inference server | `python server/rvc_server.py` |
| Offline file conversion | `python convert_audio.py input.wav output.wav -m <model.pth> -i <index.index>` |

The pretrained base model uses the open VCTK set. Put speaker weights in `assets/weights/` and matching FAISS indexes in `assets/indices/`.

## Features

+ **Realtime VC**: PySide6 client captures the mic, runs local or remote GPU inference, and plays the result
+ **Retrieval mix**: FAISS **top-k (k=4)** neighbors from training features are blended with source HuBERT features using `index_rate` (“sound like the character”). With retrieval off or no index, the source HuBERT is used and timbre leakage returns
+ **Pitch**: [RMVPE](https://github.com/Dream-High/RMVPE) by default (InterSpeech 2023); FCPE / PM optional
+ **Devices**: NVIDIA CUDA; DirectML on Windows AMD/Intel; otherwise CPU
+ This tree has **no** training WebUI, ckpt-merge, or pymss/UVR vocal separation

## Environment setup

**Python 3.11 x64 only.** The portable `runtime/` and `install_local.bat` use **3.11.9**. Do not use 3.10 or 3.12. For source work use `.venv`. Run commands from the repository root.

### Ubuntu 24.04

The distro default is 3.12; install 3.11 separately:

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

Packaged/standalone builds use `runtime\python.exe` (3.11.9). `start_server.bat` skips any venv that is not 3.11.

### Choose dependencies by hardware

Install a matching Torch build first, then this repo’s requirements. Ignore older docs that mention `go-webui.bat` or Gradio.

| Hardware | Installation |
| --- | --- |
| CPU, AMD, Intel | CPU Torch, then `requirments_cpu_py311.txt`. On Windows you may also install `torch-directml` |
| NVIDIA RTX 50 series | CUDA 12.8 Torch, then `requirments_cu128_py311.txt` |
| NVIDIA GPUs before RTX 50 | CUDA 11.8 Torch, then `requirments_cu118_py311.txt` |

#### CPU, AMD, Intel

```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirments_cpu_py311.txt
```

Optional Windows DirectML:

```bash
python -m pip install torch-directml
```

#### NVIDIA RTX 50 series

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu128_py311.txt
```

#### NVIDIA GPUs before RTX 50

```bash
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu118_py311.txt
```

If Torch is already installed and you only need the client libraries, `requirements.txt` is enough.

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

The `requirments_*.txt` files already set package indexes. To use official indexes, replace only `--index-url` and `--extra-index-url`.

## Models and runtime directories

Download shared weights from [Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main). Keep speaker models and indexes grouped by family:

```text
assets/
├── hubert_base/
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── model.safetensors    # Transformers HuBERT
│   ├── final_proj.pt        # RVC v1 768→256 projector; unused for v2
│   ├── hubert_v1.onnx       # optional
│   └── hubert_v2.onnx       # optional
├── rmvpe/rmvpe.pt
├── weights/                 # user .pth
└── indices/                 # matching .index, e.g. thchs_v2.index
```

### Download shared models

```bash
python -m pip install --upgrade huggingface_hub

hf download lj1995/VoiceConversionWebUI --revision main \
  --include "hubert_base/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
  --local-dir assets/rmvpe
```

If Hugging Face gives `pytorch_model.bin`, place it under `assets/hubert_base/` (either that or `model.safetensors`). v1 models also need `final_proj.pt`.

Windows AMD/Intel DirectML additionally needs:

```bash
hf download lj1995/VoiceConversionWebUI rmvpe.onnx --revision main \
  --local-dir assets/rmvpe
```

### FFmpeg

The Ubuntu setup command installs FFmpeg. On Windows, put these in the repo root:

- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe?download=true)
- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe?download=true)

## Usage

### 1. Realtime desktop client

```bash
python realtime_qt.py
```

### 2. Standalone / remote inference server

```bash
# Default is 0.0.0.0:8765 (LAN-reachable). Use --host 127.0.0.1 for localhost only.
python server/rvc_server.py

python server/rvc_server.py --host 0.0.0.0 --port 8765
python server/rvc_server.py --cpu
```

On Windows you can also run `start_server.bat`.

### 3. Offline conversion

```bash
python convert_audio.py input.wav output.wav \
  -m assets/weights/your_model.pth \
  -i assets/indices/your_index.index
```

Put `.pth` files in `assets/weights/` and the matching `.index` in `assets/indices/` (family name, e.g. `thchs_v2.index`). Retrieval uses k=4, same as realtime. `index_rate=0` or a missing index skips retrieval.

## Credits

+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Vocal pitch extraction: RMVPE](https://github.com/Dream-High/RMVPE)
  + Pretrained model trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss)

## Thanks to all contributors
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
