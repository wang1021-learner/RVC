"""ONNX Runtime / 可选 TensorRT 会话。"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def onnx_disabled():
    return os.environ.get("RVC_NO_ONNX") == "1"


def onnx_available():
    """onnxruntime 是否真正可用（没装或显式禁用都算不可用）。"""
    if onnx_disabled():
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _providers(prefer_trt=True):
    try:
        import onnxruntime as ort
    except Exception:
        return []
    available = list(ort.get_available_providers())
    picked = []
    if prefer_trt and "TensorrtExecutionProvider" in available:
        cache = Path(__file__).resolve().parent.parent / "assets" / "trt_cache"
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        picked.append((
            "TensorrtExecutionProvider",
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache),
                "trt_fp16_enable": True,
            },
        ))
    if "CUDAExecutionProvider" in available:
        picked.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in available:
        picked.append("DmlExecutionProvider")
    picked.append("CPUExecutionProvider")
    return picked


def create_session(onnx_path, prefer_trt=True):
    if onnx_disabled() or not onnx_path or not os.path.isfile(onnx_path):
        return None
    try:
        import onnxruntime as ort
    except Exception:
        return None
    providers = _providers(prefer_trt=prefer_trt)
    if not providers:
        return None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(str(onnx_path), opts, providers=providers)
        logger.info("ONNX loaded %s via %s", onnx_path, sess.get_providers())
        return sess
    except Exception:
        logger.exception("ONNX session failed: %s", onnx_path)
        return None


def session_run(sess, feed_numpy):
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: feed_numpy})[0]
    return out


def provider_label(sess):
    if sess is None:
        return "PyTorch"
    try:
        p = sess.get_providers()[0]
    except Exception:
        return "ONNX"
    if "Tensorrt" in p:
        return "TensorRT"
    if "CUDA" in p:
        return "ONNX-CUDA"
    if "Dml" in p:
        return "ONNX-DML"
    return "ONNX-CPU"
