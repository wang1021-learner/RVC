#!/usr/bin/env python3
"""保守裁剪 torch：删除运行时用不到的静态库与惰性加载的 CUDA 组件。

删除范围（保守，零风险）：
- 所有 *.lib / *.a / *.pdb（链接器/调试用，运行时从不加载）
- cusolverMg64_11.dll（多卡求解器，单卡推理不用）
- cusparse64_11.dll（稀疏矩阵，本项目不用）
- nvrtc*（JIT 编译，本项目用 CUDA Graph，不用 JIT）

保留 cusolver64_11.dll（谨慎起见），cuDNN/cuBLAS/cuFFT/curand 全保留。

用法: python tools/trim_torch.py --torch-dir <torch目录> [--dry-run]
"""
import argparse
import os
import sys

DELETE_EXTS = {".lib", ".a", ".pdb"}
DELETE_NAME_PREFIXES = ("cusolvermg", "cusparse", "nvrtc")


def main():
    ap = argparse.ArgumentParser(description="保守裁剪 torch 安装目录")
    ap.add_argument("--torch-dir", required=True, help="torch 包目录")
    ap.add_argument("--dry-run", action="store_true", help="只报告不删除")
    args = ap.parse_args()
    root = os.path.abspath(args.torch_dir)
    if not os.path.isdir(root):
        print("torch 目录不存在:", root)
        return 1
    removed = 0
    saved = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            low = fn.lower()
            if os.path.splitext(low)[1] not in DELETE_EXTS and not low.startswith(
                DELETE_NAME_PREFIXES
            ):
                continue
            try:
                size = os.path.getsize(full)
                if not args.dry_run:
                    os.remove(full)
                removed += 1
                saved += size
                print("删除: %s (%.1f MB)" % (full, size / 1048576.0))
            except OSError as e:
                print("跳过: %s (%s)" % (full, e))
    print(
        "完成：%s%d 个文件，节省 %.1f MB"
        % ("（试运行）" if args.dry_run else "", removed, saved / 1048576.0)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
