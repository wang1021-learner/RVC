#!/usr/bin/env python3
"""保守裁剪 torch：仅删除运行时用不到的静态库与调试符号（*.lib, *.a, *.pdb）。

注意：
- 严禁删除任何 DLL（如 nvrtc*.dll、cusolver*.dll、cusparse*.dll 等），
  PyTorch 在 Windows 下初始化阶段 (_load_dll_libraries) 会强校验并动态链接这些库，
  误删会导致 [WinError 126] 找不到指定的模块错误。

用法: python tools/trim_torch.py --torch-dir <torch目录> [--dry-run]
"""
import argparse
import os
import sys

# 仅删除纯静态编译/调试文件（不影响任何运行时运行）
DELETE_EXTS = {".lib", ".a", ".pdb"}
DELETE_NAME_PREFIXES = ()


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
            ext = os.path.splitext(low)[1]
            
            # 判断是否需要删除（仅匹配指定后缀）
            should_delete = (ext in DELETE_EXTS) or (
                bool(DELETE_NAME_PREFIXES) and low.startswith(DELETE_NAME_PREFIXES)
            )
            if not should_delete:
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
