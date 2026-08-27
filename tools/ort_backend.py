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


def _has_accel_ep():
    try:
        import onnxruntime as ort
        names = set(ort.get_available_providers())
        return bool(names & {
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "DmlExecutionProvider",
        })
    except Exception:
        return False


def onnx_enabled_for_realtime():
    """实时 ONNX 默认关：和 CUDA Graph、独立 F0 流叠在一起容易把 GPU 上下文打崩。
    显式 RVC_ONNX=1 才开。"""
    if onnx_disabled():
        return False
    return os.environ.get("RVC_ONNX") == "1" and onnx_available()


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
        has_engine = False
        try:
            has_engine = any(cache.glob("*"))
        except Exception:
            has_engine = False
        # 实时默认走 CUDA EP；已有 engine 或 RVC_ORT_TRT=1 才上 TRT，避免首次编译卡死
        if has_engine or os.environ.get("RVC_ORT_TRT") == "1":
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
        print("[ORT] ONNX 已加载: %s via %s" % (os.path.basename(str(onnx_path)), sess.get_providers()))
        return sess
    except Exception as e:
        logger.exception("ONNX session failed: %s", onnx_path)
        print("[ORT] ONNX 加载失败: %s" % e)
        return None


_OUT_SHAPE_CACHE = {}
_IOB_HIT_LOGGED = False
_IOB_FALLBACK_LOGGED = False


def _log_iob_hit():
    global _IOB_HIT_LOGGED
    if not _IOB_HIT_LOGGED:
        _IOB_HIT_LOGGED = True
        msg = "ONNX IO Binding 生效（零拷贝，全程 GPU）"
        print("[ORT] " + msg)
        logger.info(msg)


def _log_iob_fallback(reason=""):
    global _IOB_FALLBACK_LOGGED
    if not _IOB_FALLBACK_LOGGED:
        _IOB_FALLBACK_LOGGED = True
        msg = "ONNX IO Binding 回退 numpy：%s" % (reason or "未知原因")
        print("[ORT] " + msg)
        logger.warning(msg)


def session_run(sess, feed_numpy):
    """numpy 路径；顺带缓存输出形状，供 IO Binding 复用。"""
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: feed_numpy})[0]
    try:
        _OUT_SHAPE_CACHE[(id(sess), tuple(feed_numpy.shape))] = tuple(out.shape)
    except Exception:
        pass
    return out


def run_iobinding(sess, torch_tensor):
    """零拷贝 GPU 推理：输入/输出经 DLPack 绑定，全程不落 CPU。

    返回 torch CUDA tensor；形状未知或任一步失败返回 None（调用方回退 numpy）。
    """
    try:
        import onnxruntime as ort
        import torch.utils.dlpack as tdl
        import numpy as np
    except Exception:
        return None

    x = torch_tensor.detach().float().contiguous()
    if not getattr(x, "is_cuda", False):
        return None
    device_id = x.get_device()

    out_shape = _OUT_SHAPE_CACHE.get((id(sess), tuple(x.shape)))
    if out_shape is None:
        return None  # 首次先走 numpy 学习形状

    try:
        inp_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        binding = sess.io_binding()
        ort_in = ort.OrtValue.from_dlpack(tdl.to_dlpack(x))
        binding.bind_ortvalue_input(inp_name, ort_in)
        out_ort = ort.OrtValue.ortvalue_from_shape_and_type(
            list(out_shape), np.float32, "cuda", device_id
        )
        binding.bind_ortvalue_output(out_name, out_ort)
        sess.run_with_iobinding(binding)
        out_t = tdl.from_dlpack(out_ort.to_dlpack())
        _log_iob_hit()
        return out_t.clone()
    except Exception as e:
        _log_iob_fallback(str(e))
        return None


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
