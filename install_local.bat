@echo off
setlocal enabledelayedexpansion
title RVC 本地推理环境一键安装
set "ROOT=%~dp0"
set "RT=%ROOT%runtime"

echo ============================================================
echo   RVC 本地推理环境一键安装
echo   - 需要 NVIDIA 显卡与网络连接
echo   - 需要下载约 3.5GB（torch 及依赖），耗时取决于网速
echo ============================================================
echo.

REM ── 1. 嵌入式 Python ──
if exist "%RT%\python.exe" goto :deps
echo [1/4] 下载并解压嵌入式 Python 3.11...
if not exist "%RT%" mkdir "%RT%"
set "PY_ZIP=%TEMP%\python-3.11.9-embed-amd64.zip"

powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri 'https://registry.npmmirror.com/-/binary/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '%PY_ZIP%' -TimeoutSec 120 -UseBasicParsing } catch { Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '%PY_ZIP%' -TimeoutSec 120 -UseBasicParsing }"

if not exist "%PY_ZIP%" (
    echo [错误] Python 下载失败，请检查网络后重新运行本脚本
    pause
    exit /b 1
)

powershell -NoProfile -Command "Expand-Archive -Path '%PY_ZIP%' -DestinationPath '%RT%' -Force"
if not exist "%RT%\python.exe" (
    echo [错误] Python 解压失败
    pause
    exit /b 1
)

REM 启用 site-packages
powershell -NoProfile -Command "$p='%RT%\python311._pth'; $c=Get-Content $p -Raw; $c=$c -replace '#import site','import site'; if ($c -notmatch 'Lib\\site-packages') { $c=$c.TrimEnd()+[char]10+'Lib\\site-packages'+[char]10+'.' }; Set-Content -Path $p -Value $c -Encoding ASCII"

REM ── 2. pip 引导 ──
echo [2/4] 安装 pip...
set "GETPIP=%TEMP%\get-pip.py"
powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri 'https://mirrors.aliyun.com/pypi/get-pip.py' -OutFile '%GETPIP%' -TimeoutSec 60 -UseBasicParsing } catch { Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GETPIP%' -TimeoutSec 60 -UseBasicParsing }"

if not exist "%GETPIP%" (
    echo [错误] get-pip 下载失败，请检查网络后重新运行本脚本
    pause
    exit /b 1
)

"%RT%\python.exe" "%GETPIP%" --no-warn-script-location -i https://mirrors.pku.edu.cn/pypi/simple
if errorlevel 1 (
    echo [错误] pip 安装失败
    pause
    exit /b 1
)

:deps
REM ── 3. 安装 torch ──
echo [3/4] 安装 torch 2.7.1 + cu118（约 3.5GB，请耐心等待）...
"%RT%\python.exe" -m pip install --no-input --default-timeout 100 --retries 5 torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu118 --extra-index-url https://mirrors.pku.edu.cn/pypi/simple
if errorlevel 1 (
    echo [错误] torch 安装失败，可重新运行本脚本续装
    pause
    exit /b 1
)

REM ── 4. 安装其余依赖（自动探测路径） ──
echo 安装其余推理依赖...
set "REQ_FILE=%ROOT%source\requirements_local_cu118.txt"
if not exist "!REQ_FILE!" set "REQ_FILE=%ROOT%requirements_local_cu118.txt"

if not exist "!REQ_FILE!" (
    echo [警告] 未找到 requirements_local_cu118.txt，跳过依赖文件安装
) else (
    "%RT%\python.exe" -m pip install --no-input --default-timeout 100 --retries 5 -r "!REQ_FILE!" -i https://mirrors.pku.edu.cn/pypi/simple
    if errorlevel 1 (
        echo [错误] 依赖安装失败，可重新运行本脚本续装
        pause
        exit /b 1
    )
)

REM ── 5. 裁剪 torch ──
echo [4/4] 裁剪 torch 无用文件...
set "TRIM_PY=%ROOT%source\tools\trim_torch.py"
if not exist "!TRIM_PY!" set "TRIM_PY=%ROOT%tools\trim_torch.py"

if exist "!TRIM_PY!" (
    "%RT%\python.exe" "!TRIM_PY!" --torch-dir "%RT%\Lib\site-packages\torch"
)

REM ── 自检 ──
echo.
echo 自检 GPU 环境...
"%RT%\python.exe" -c "import torch; print('torch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo.
echo ============================================================
echo   安装完成！回到程序点「启动变声」即可使用本地推理。
echo   若显示 CUDA available: False，请安装 NVIDIA 显卡驱动后重试。
echo ============================================================
pause
