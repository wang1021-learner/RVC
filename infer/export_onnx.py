"""把 HuBERT / RMVPE 导出为 ONNX，供实时路径加速。合成器 NSF 仍走 PyTorch。"""
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HUBERT_ONNX_V2 = PROJECT_ROOT / "assets" / "hubert_base" / "hubert_v2.onnx"
HUBERT_ONNX_V1 = PROJECT_ROOT / "assets" / "hubert_base" / "hubert_v1.onnx"
RMVPE_ONNX = PROJECT_ROOT / "assets" / "rmvpe" / "rmvpe.onnx"


class _HubertV2Wrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        return self.model(
            input_values=input_values,
            attention_mask=None,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state


class _HubertV1Wrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        outputs = self.model(
            input_values=input_values,
            attention_mask=None,
            output_hidden_states=True,
            return_dict=True,
        )
        return self.model.final_proj(outputs.hidden_states[9])


def _export(module, dummy, path, opset=17):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy,
            str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 1: "time"},
                "output": {0: "batch", 1: "frames"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    logger.info("exported ONNX %s", path)
    return path


def export_hubert(model, version="v2", samples=3200):
    dest = HUBERT_ONNX_V2 if version == "v2" else HUBERT_ONNX_V1
    if dest.is_file():
        return dest
    try:
        wrap = _HubertV2Wrap(model) if version == "v2" else _HubertV1Wrap(model)
        dummy = torch.zeros(1, samples, device="cpu", dtype=torch.float32)
        wrap = wrap.float().cpu()
        return _export(wrap, dummy, dest)
    except Exception:
        logger.exception("HuBERT ONNX export failed (%s)", version)
        return None


def export_rmvpe(pt_path, frames=32):
    if RMVPE_ONNX.is_file():
        return RMVPE_ONNX
    try:
        from infer.rmvpe import E2E

        model = E2E(4, 1, (2, 2))
        ckpt = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(ckpt)
        model.eval().float()
        dummy = torch.zeros(1, 128, frames, dtype=torch.float32)
        return _export(model, dummy, RMVPE_ONNX)
    except Exception:
        logger.exception("RMVPE ONNX export failed")
        return None
