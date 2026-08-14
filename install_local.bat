@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title RVC 本地推理环境安装
set "ROOT=%~dp0"
set "RT=%ROOT%runtime"

echo ============================================================
echo   RVC 本地推理环境安装
echo   - 需要 NVIDIA 显卡与网络连接
echo   - 需要下载约 3.5GB（torch 及依赖），耗时取决于网速
echo ============================================================
echo.

REM ── 1. 嵌入式 Python ──
if exist "%RT%\python.exe" goto :deps
echo [1/4] 下载并解压嵌入式 Python 3.11...
if not exist "%RT%" mkdir "%RT%"
set "PY_ZIP=%TEMP%\python-3.11.9-embed-amd64.zip"
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '%PY_ZIP%' -TimeoutSec 120 } catch { Invoke-WebRequest -Uri 'https://registry.npmmirror.com/-/binary/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '%PY_ZIP%' -TimeoutSec 120 }"
if not exist "%PY_ZIP%" (
    echo [错误] Python 下载失败，请检查网络后重试
    pause
    exit /b 1
)
powershell -NoProfile -Command "Expand-Archive -Path '%PY_ZIP%' -DestinationPath '%RT%' -Force"
if not exist "%RT%\python.exe" (
    echo [错误] Python 解压失败
    pause
    exit /b 1
)

REM 启用 site-packages（嵌入版默认关闭）
powershell -NoProfile -Command "$p='%RT%\python311._pth'; $c=Get-Content $p -Raw; $c=$c -replace '#import site','import site'; if ($c -notmatch 'Lib\site-packages') { $c=$c.TrimEnd()+[char]10+'Lib\site-packages' }; Set-Content -Path $p -Value $c -Encoding ASCII"

REM ── 2. pip 引导 ──
echo [2/4] 安装 pip...
set "GETPIP=%TEMP%\get-pip.py"
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GETPIP%' -TimeoutSec 60 } catch { Invoke-WebRequest -Uri 'https://mirrors.aliyun.com/pypi/get-pip.py' -OutFile '%GETPIP%' -TimeoutSec 60 }"
if not exist "%GETPIP%" (
    echo [错误] get-pip 下载失败，请检查网络后重试
    pause
    exit /b 1
)
"%RT%\python.exe" "%GETPIP%" -i https://mirrors.pku.edu.cn/pypi/simple
if errorlevel 1 (
    echo [错误] pip 安装失败
    pause
    exit /b 1
)

:deps
echo [3/4] 安装 torch 2.7.1 + cu118（约 3.5GB，请耐心等待）...
"%RT%\python.exe" -m pip install --no-input torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu118 --extra-index-url https://mirrors.pku.edu.cn/pypi/simple
if errorlevel 1 (
    echo [错误] torch 安装失败，可重新运行本脚本续装
    pause
    exit /b 1
)
echo 安装其余推理依赖...
"%RT%\python.exe" -m pip install --no-input -r "%ROOT%source\requirements_local_cu118.txt" -i https://mirrors.pku.edu.cn/pypi/simple
if errorlevel 1 (
    echo [错误] 依赖安装失败，可重新运行本脚本续装
    pause
    exit /b 1
)

REM ── 3. 裁剪 torch（约省 1.3GB 磁盘）──
echo [4/4] 裁剪 torch 无用文件...
"%RT%\python.exe" "%ROOT%source\tools\trim_torch.py" --torch-dir "%RT%\Lib\site-packages\torch"

REM ── 自检 ──
echo.
echo 自检 GPU 环境...
"%RT%\python.exe" -c "import torch; print('torch', torch.__version__, '| CUDA 可用:', torch.cuda.is_available())"
echo.
echo ============================================================
echo   安装完成！回到程序点「启动变声」即可使用本地推理。
echo   若显示 CUDA 可用: False，请安装 NVIDIA 显卡驱动后重试。
echo ============================================================
pause
