import os
from typing import List, Optional, Tuple
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window

from tools.cuda_graph import run_cuda_graph

import logging

logger = logging.getLogger(__name__)


from tools.stft import STFT


from time import time as ttime


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        # self.shortcut:Optional[nn.Module] = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i, layer in enumerate(self.layers):
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class ResEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01
    ):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i, conv in enumerate(self.conv):
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Intermediate(nn.Module):  #
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for i in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i, conv2 in enumerate(self.conv2):
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x) :
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(E2E, self).__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * nn.N_MELS, nn.N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, mel):
        # print(mel.shape)
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        # print(x.shape)
        return x


from librosa.filters import mel


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        is_half,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                audio.device
            )
        if "privateuseone" in str(audio.device):
            if not hasattr(self, "stft"):
                self.stft = STFT(
                    filter_length=n_fft_new,
                    hop_length=hop_length_new,
                    win_length=win_length_new,
                    window="hann",
                ).to(audio.device)
            magnitude = self.stft.transform(audio)
        else:
            fft = torch.stft(
                audio,
                n_fft=n_fft_new,
                hop_length=hop_length_new,
                win_length=win_length_new,
                window=self.hann_window[keyshift_key],
                center=center,
                return_complex=True,
            )
            magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half == True:
            mel_output = mel_output.half()
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class RMVPE:
    def __init__(self, model_path, is_half, device=None):
        self.resample_kernel = {}
        self.resample_kernel = {}
        if isinstance(is_half, str):
            is_half = is_half.lower() == "true"
        if device is None:
            from configs.config import infer_device, infer_dtype

            device = str(infer_device)
            is_half = infer_dtype == torch.float16
        elif str(device).startswith("cuda"):
            from configs.config import get_device_dtype_sm

            parsed_device = torch.device(device)
            device_index = parsed_device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            selected_device, selected_dtype, _, _ = get_device_dtype_sm(device_index)
            device = str(selected_device)
            is_half = selected_dtype == torch.float16
        else:
            is_half = False
        self.is_half = is_half
        self.device = device
        self.mel_extractor = MelSpectrogram(
            is_half, 128, 16000, 1024, 160, None, 30, 8000
        ).to(device)
        self._ort_sess = None
        # NVIDIA 默认不走 ONNX（numpy IO 每块 CPU↔GPU 往返，实时反而加延迟）；
        # DirectML(私有设备) 只能走 ONNX，必须保留导出。RVC_ONNX=1 可强制 NVIDIA 用。
        need_onnx = "privateuseone" in str(device)
        if not need_onnx:
            from tools.ort_backend import onnx_enabled_for_realtime

            need_onnx = onnx_enabled_for_realtime()
        if need_onnx:
            onnx_path = os.path.splitext(model_path)[0] + ".onnx"
            if not os.path.isfile(onnx_path):
                try:
                    from infer.export_onnx import export_rmvpe

                    exported = export_rmvpe(model_path)
                    if exported is not None:
                        onnx_path = str(exported)
                except Exception:
                    logger.exception("RMVPE ONNX export skipped")
            if os.path.isfile(onnx_path):
                try:
                    from tools.ort_backend import create_session

                    sess = create_session(onnx_path)
                    if sess is not None:
                        self._ort_sess = sess
                        self.model = sess
                        logger.info("RMVPE via ONNX %s", onnx_path)
                except Exception:
                    logger.exception("RMVPE ONNX load failed")
        if self._ort_sess is None and "privateuseone" in str(device):
            import onnxruntime as ort

            ort_session = ort.InferenceSession(
                os.path.splitext(model_path)[0] + ".onnx",
                providers=["DmlExecutionProvider"],
            )
            self.model = ort_session
            self._ort_sess = ort_session
        elif self._ort_sess is None:
            if str(self.device) == "cuda":
                self.device = torch.device("cuda:0")

            def get_default_model():
                model = E2E(4, 1, (2, 2))
                ckpt = torch.load(model_path, map_location="cpu")
                model.load_state_dict(ckpt)
                model.eval()
                if is_half:
                    model = model.half()
                else:
                    model = model.float()
                return model

            self.model = get_default_model()

            self.model = self.model.to(device)
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368
        self.cents_mapping_torch = torch.from_numpy(self.cents_mapping).float().to(self.device)

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            if self._ort_sess is not None or "privateuseone" in str(self.device):
                import numpy as np

                from tools.ort_backend import run_iobinding

                hidden = run_iobinding(self.model, mel) if torch.is_tensor(mel) else None
                if hidden is None:
                    mel_np = mel.detach().float().cpu().numpy() if torch.is_tensor(mel) else mel
                    onnx_input_name = self.model.get_inputs()[0].name
                    hidden = self.model.run(None, {onnx_input_name: mel_np})[0]
                    hidden = torch.from_numpy(np.ascontiguousarray(hidden)).to(self.device)
            else:
                mel = mel.half() if self.is_half else mel.float()
                hidden = run_cuda_graph(
                    self.model,
                    "rmvpe-network",
                    lambda input_mel: self.model(input_mel),
                    mel,
                )
            return hidden[:, :n_frames]

    def extract_mel(self, audio, center=True):
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        audio = audio.float().to(self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if "privateuseone" in str(self.device):
            return self.mel_extractor(audio, center=center)
        return run_cuda_graph(
            self.mel_extractor,
            "rmvpe-mel-center-%s" % int(bool(center)),
            lambda input_audio: self.mel_extractor(input_audio, center=center),
            audio,
        )

    def to_local_average_cents_torch(self, salience, thred=0.05):
        """GPU 向量化解码：使用 torch.gather 与广播，消除 Python for 循环"""
        if salience.dim() == 3:
            salience = salience.squeeze(0)
        # salience: (n_frames, 360)
        center = salience.argmax(dim=-1)  # (n_frames,)
        salience_pad = F.pad(salience.float(), (4, 4), mode="constant", value=0.0)  # (n_frames, 368)
        
        # 构造每个 frame 对应的 9 个采样点索引 [center, center+1, ..., center+8] (对应 pad 后的 [center-4..center+4]+4)
        offsets = torch.arange(9, device=salience.device)  # (9,)
        indices = center.unsqueeze(-1) + offsets  # (n_frames, 9)
        
        # 提取局部 salience 和 cents
        todo_salience = torch.gather(salience_pad, 1, indices)  # (n_frames, 9)
        if self.cents_mapping_torch.device != salience.device:
            self.cents_mapping_torch = self.cents_mapping_torch.to(salience.device)
        todo_cents = self.cents_mapping_torch[indices]  # (n_frames, 9)
        
        product_sum = (todo_salience * todo_cents).sum(dim=-1)
        weight_sum = todo_salience.sum(dim=-1).clamp(min=1e-8)
        devided = product_sum / weight_sum
        
        maxx = salience.max(dim=-1).values
        devided = torch.where(maxx > thred, devided, torch.zeros_like(devided))
        return devided

    def decode_torch(self, hidden, thred=0.03):
        """GPU 向量化完整解码：直接返回设备上的 f0 Tensor"""
        cents_pred = self.to_local_average_cents_torch(hidden, thred=thred)
        f0 = torch.where(
            cents_pred > 0,
            10.0 * torch.pow(2.0, cents_pred / 1200.0),
            torch.zeros_like(cents_pred),
        )
        return f0

    def decode(self, hidden, thred=0.03):
        if torch.is_tensor(hidden):
            return self.decode_torch(hidden, thred=thred)
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer_hidden(self, audio):
        """GPU 部分：mel -> 网络 hidden，保持为设备上的 Tensor（不落 CPU）。"""
        mel = self.extract_mel(audio, center=True)
        return self.mel2hidden(mel)

    def decode_hidden(self, hidden, thred=0.03):
        """CPU 部分：hidden -> f0 numpy（含取回同步，供外部非 GPU 链路兼容使用）。"""
        if torch.is_tensor(hidden) and hidden.is_cuda:
            f0_t = self.decode_torch(hidden, thred=thred)
            return f0_t.cpu().numpy()
        if "privateuseone" not in str(self.device):
            hidden = hidden.squeeze(0).cpu().numpy() if torch.is_tensor(hidden) else hidden
        else:
            hidden = hidden[0]
        if hasattr(hidden, "astype") and self.is_half:
            hidden = hidden.astype("float32")
        return self.decode(hidden, thred=thred)

    def infer_from_audio(self, audio, thred=0.03):
        hidden = self.infer_hidden(audio)
        return self.decode_hidden(hidden, thred=thred)

    def to_local_average_cents(self, salience, thred=0.05):
        # t0 = ttime()
        center = np.argmax(salience, axis=1)  # 帧长#index
        salience = np.pad(salience, ((0, 0), (4, 4)))  # 帧长,368
        # t1 = ttime()
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        # t2 = ttime()
        todo_salience = np.array(todo_salience)  # 帧长，9
        todo_cents_mapping = np.array(todo_cents_mapping)  # 帧长，9
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)  # 帧长
        devided = product_sum / weight_sum  # 帧长
        # t3 = ttime()
        maxx = np.max(salience, axis=1)  # 帧长
        devided[maxx <= thred] = 0
        # t4 = ttime()
        # print("decode:%s\t%s\t%s\t%s" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return devided


if __name__ == "__main__":
    import librosa
    import soundfile as sf

    audio, sampling_rate = sf.read(r"C:\Users\liujing04\Desktop\Z\冬之花clip1.wav")
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    audio_bak = audio.copy()
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    model_path = r"D:\BaiduNetdiskDownload\RVC-beta-v2-0727AMD_realtime\rmvpe.pt"
    thred = 0.03  # 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rmvpe = RMVPE(model_path, is_half=False, device=device)
    t0 = ttime()
    f0 = rmvpe.infer_from_audio(audio, thred=thred)
    # f0 = rmvpe.infer_from_audio(audio, thred=thred)
    # f0 = rmvpe.infer_from_audio(audio, thred=thred)
    # f0 = rmvpe.infer_from_audio(audio, thred=thred)
    # f0 = rmvpe.infer_from_audio(audio, thred=thred)
    t1 = ttime()
    logger.info("%s %.2f", f0.shape, t1 - t0)
