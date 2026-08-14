#!/usr/bin/env python3
"""
打包 RVC 实时语音转换部署包
将所有必要文件收集到 deploy/ 目录，拷贝到 Windows 后运行 build.bat 即可生成 exe
"""
import os
import shutil
import sys

PROJECT = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(PROJECT, "deploy")


def copy_tree(src, dst, name):
    src = os.path.join(PROJECT, src)
    dst = os.path.join(DEPLOY, dst)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  [目录] {name}: {src} -> {dst}")


def copy_file(src, dst, name):
    src = os.path.join(PROJECT, src)
    dst = os.path.join(DEPLOY, dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  [文件] {name}: {src} -> {dst}")


def main():
    print("=" * 60)
    print("RVC 实时语音转换 - 部署包打包工具")
    print("=" * 60)

    # 清理旧的部署目录
    if os.path.exists(DEPLOY):
        shutil.rmtree(DEPLOY)
    os.makedirs(DEPLOY)

    # 1. 主程序
    print("\n[1/7] 主程序")
    copy_file("realtime_qt.py", "realtime_qt.py", "PySide6 GUI 程序")

    # 2. Python 模块
    print("\n[2/7] Python 核心模块")
    for mod in ["configs", "infer", "tools", "i18n", "worker", "server"]:
        copy_tree(mod, mod, mod)

    # 3. 模型文件
    print("\n[3/7] 模型文件")
    # 训练好的模型
    model_file = "assets/weights/thchs_female_200e.pth"
    if os.path.exists(os.path.join(PROJECT, model_file)):
        copy_file(model_file, model_file, "说话人模型 (200轮)")
    else:
        print(f"  [警告] 未找到 {model_file}")

    # 4. 依赖模型
    print("\n[4/7] 依赖模型 (RMVPE + HuBERT)")
    copy_file("assets/rmvpe/rmvpe.pt", "assets/rmvpe/rmvpe.pt", "RMVPE F0 模型")
    copy_tree("assets/hubert_base", "assets/hubert_base", "HuBERT 模型")

    # 5. 索引文件
    print("\n[5/7] FAISS 索引文件")
    index_file = "logs/thchs_v2/added_IVF2716_Flat_nprobe_1_thchs_v2_v2.index"
    if os.path.exists(os.path.join(PROJECT, index_file)):
        copy_file(index_file, index_file, "FAISS 检索索引")
    else:
        print(f"  [警告] 未找到 {index_file}")

    # 6. 配置文件
    print("\n[6/7] 配置文件")
    copy_file("speakers.json", "speakers.json", "角色配置")

    # 7. 打包脚本和说明
    print("\n[7/7] Windows 打包脚本和说明")
    copy_file("app/build.bat", "build.bat", "Windows 打包脚本")
    copy_file("app/rvc_realtime.spec", "rvc_realtime.spec", "PyInstaller 配置")
    copy_file("app/requirements.txt", "requirements.txt", "Python 依赖清单")
    copy_file("app/README.txt", "README.txt", "使用说明")

    # 统计
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(DEPLOY):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
            file_count += 1

    print(f"\n{'=' * 60}")
    print(f"打包完成！")
    print(f"  输出目录: {DEPLOY}")
    print(f"  文件数量: {file_count}")
    print(f"  总大小:   {total_size / 1024 / 1024:.1f} MB")
    print(f"\n下一步:")
    print(f"  1. 将 deploy/ 目录拷贝到 Windows 电脑")
    print(f"  2. 安装 Python 3.10+ (勾选 Add to PATH)")
    print(f"  3. 双击运行 build.bat")
    print(f"  4. 生成 exe 在 deploy\\dist\\RVC实时语音转换\\")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
