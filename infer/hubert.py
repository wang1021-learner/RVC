import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoFeatureExtractor, HubertModel

from tools.cuda_graph import run_cuda_graph
from tools.ort_backend import create_session, onnx_available, onnx_disabled, provider_label, session_run


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

HUBERT_MODEL_PATH = (PROJECT_ROOT / "assets" / "hubert_base").resolve()


def _device_type(device):
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


class HubertOnnx:
    """ONNX HuBERT：v2 last_hidden / v1 layer9+proj。合成器仍走 PyTorch。"""

    def __init__(self, sess_v2, sess_v1, device, is_half):
        self.sess_v2 = sess_v2
        self.sess_v1 = sess_v1
        self.device = device
        self.is_half = is_half
        self._ort = True
        self.backend_label = provider_label(sess_v2 or sess_v1)

    def extract(self, source, version):
        sess = self.sess_v2 if version == "v2" else self.sess_v1
        if sess is None:
            raise RuntimeError("HuBERT ONNX session missing for %s" % version)
        arr = source.detach().float().contiguous().cpu().numpy()
        out = session_run(sess, arr)
        t = torch.from_numpy(np.ascontiguousarray(out)).to(
            device=source.device, dtype=source.dtype, non_blocking=False
        )
        return t


def _try_hubert_onnx(device, is_half):
    if onnx_disabled():
        return None
    from infer.export_onnx import HUBERT_ONNX_V1, HUBERT_ONNX_V2, export_hubert

    sess_v2 = create_session(HUBERT_ONNX_V2)
    sess_v1 = create_session(HUBERT_ONNX_V1)
    if sess_v2 is None and sess_v1 is None:
        # 先加载 PyTorch 再导出（仅一次）
        return None
    return HubertOnnx(sess_v2, sess_v1, device, is_half)


def load_hubert_model(device, is_half=False):
    """Load HuBERT：优先 ONNX/TensorRT，失败回退 Transformers。"""
    if not (HUBERT_MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(
            f"Transformers HuBERT model not found: {HUBERT_MODEL_PATH}"
        )

    onnx_model = _try_hubert_onnx(device, is_half)
    if onnx_model is not None and onnx_model.sess_v2 is not None:
        logger.info("HuBERT via %s", onnx_model.backend_label)
        return onnx_model

    dtype = torch.float16 if is_half else torch.float32
    load_options = {
        "local_files_only": True,
        "torch_dtype": dtype,
    }
    if _device_type(device) == "privateuseone":
        load_options["attn_implementation"] = "eager"

    logger.info(
        "Loading Transformers HuBERT from %s (%s on %s)",
        HUBERT_MODEL_PATH,
        dtype,
        device,
    )
    model = HubertModelWithFinalProj.from_pretrained(
        str(HUBERT_MODEL_PATH), **load_options
    )
    model = model.to(device).eval()
    if onnx_available():
        try:
            from infer.export_onnx import export_hubert

            orig_dev = next(model.parameters()).device
            cpu_model = model.float().cpu()
            export_hubert(cpu_model, "v2")
            export_hubert(cpu_model, "v1")
            del cpu_model
            model = model.to(device=orig_dev)
            if is_half and orig_dev.type == "cuda":
                model = model.half()
            again = _try_hubert_onnx(device, is_half)
            if again is not None and again.sess_v2 is not None:
                return again
        except Exception:
            logger.exception("HuBERT ONNX export skipped")
            try:
                model = model.to(device)
                if is_half and _device_type(device) == "cuda":
                    model = model.half()
            except Exception:
                pass
    return model


@lru_cache(maxsize=1)
def hubert_audio_requires_normalization():
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        str(HUBERT_MODEL_PATH), local_files_only=True
    )
    return bool(feature_extractor.do_normalize)


def _forward_v1_no_mask(model, input_values):
    outputs = model(
        input_values=input_values,
        attention_mask=None,
        output_hidden_states=True,
        return_dict=True,
    )
    return model.final_proj(outputs.hidden_states[9])


def _forward_v1_mask(model, input_values, mask):
    outputs = model(
        input_values=input_values,
        attention_mask=mask,
        output_hidden_states=True,
        return_dict=True,
    )
    return model.final_proj(outputs.hidden_states[9])


def _forward_v2_no_mask(model, input_values):
    return model(
        input_values=input_values,
        attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
    ).last_hidden_state


def _forward_v2_mask(model, input_values, mask):
    return model(
        input_values=input_values,
        attention_mask=mask,
        output_hidden_states=False,
        return_dict=True,
    ).last_hidden_state


def extract_hubert_features(model, source, version, padding_mask=None):
    """Return the RVC v1 (256-D) or v2 (768-D) HuBERT representation."""
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported RVC feature version: {version!r}")
    if getattr(model, "_ort", False):
        return model.extract(source, version)

    attention_mask = None
    if padding_mask is not None:
        # 全 False 的 mask 不要 .any()：那是一次 GPU 同步
        if padding_mask.device.type == "cpu":
            if bool(padding_mask.any()):
                attention_mask = (~padding_mask.bool()).long()
        elif bool(padding_mask.detach().any().item()):
            attention_mask = (~padding_mask.bool()).long()

    if version == "v1":
        if attention_mask is None:
            return run_cuda_graph(model, "hubert-v1-no-mask", lambda x: _forward_v1_no_mask(model, x), source)
        return run_cuda_graph(model, "hubert-v1-mask", lambda x, m: _forward_v1_mask(model, x, m), source, attention_mask)

    if attention_mask is None:
        return run_cuda_graph(model, "hubert-v2-no-mask", lambda x: _forward_v2_no_mask(model, x), source)
    return run_cuda_graph(model, "hubert-v2-mask", lambda x, m: _forward_v2_mask(model, x, m), source, attention_mask)
