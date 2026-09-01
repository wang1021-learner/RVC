@echo off
setlocal enabledelayedexpansion
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title RVC Local Inference Setup
set "ROOT=%~dp0"
set "RT=%ROOT%runtime"

echo ============================================================
echo   RVC Local Inference Setup
echo   - Requires NVIDIA GPU and internet connection
echo   - Downloads about 3.5GB (torch + deps)
echo ============================================================
echo.

REM 1. Embedded Python
if exist "%RT%\python.exe" goto :deps
echo [1/4] Downloading embedded Python 3.11.9 (required portable runtime)...
if not exist "%RT%" mkdir "%RT%"
set "PY_ZIP=%TEMP%\python-3.11.9-embed-amd64.zip"
powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri 'https://registry.npmmirror.com/-/binary/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '%PY_ZIP%' -TimeoutSec 120 -UseBasicParsing } catch { Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile '%PY_ZIP%' -TimeoutSec 120 -UseBasicParsing }"
if not exist "%PY_ZIP%" (
    echo [ERROR] Python download failed, check network and retry
    pause
    exit /b 1
)
powershell -NoProfile -Command "Expand-Archive -Path '%PY_ZIP%' -DestinationPath '%RT%' -Force"
if not exist "%RT%\python.exe" (
    echo [ERROR] Python extract failed
    pause
    exit /b 1
)

REM Enable site-packages (disabled by default in embeddable Python)
powershell -NoProfile -Command "$p='%RT%\python311._pth'; $c=Get-Content $p -Raw; $c=$c -replace '#import site','import site'; if ($c -notmatch 'Lib\site-packages') { $c=$c.TrimEnd()+[char]10+'Lib\site-packages'+[char]10+'.' }; Set-Content -Path $p -Value $c -Encoding ASCII"

REM 2. pip bootstrap
echo [2/4] Installing pip...
set "GETPIP=%TEMP%\get-pip.py"
powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri 'https://mirrors.aliyun.com/pypi/get-pip.py' -OutFile '%GETPIP%' -TimeoutSec 60 -UseBasicParsing } catch { Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GETPIP%' -TimeoutSec 60 -UseBasicParsing }"
if not exist "%GETPIP%" (
    echo [ERROR] get-pip download failed, check network and retry
    pause
    exit /b 1
)
"%RT%\python.exe" "%GETPIP%" --no-warn-script-location -i https://mirrors.pku.edu.cn/pypi/simple
if errorlevel 1 (
    echo [ERROR] pip install failed
    pause
    exit /b 1
)

:deps
REM 3. torch
echo [3/4] Installing torch 2.7.1 + cu118 (about 3.5GB, please wait)...
"%RT%\python.exe" -m pip install --no-input --default-timeout 100 --retries 5 torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu118 --extra-index-url https://mirrors.pku.edu.cn/pypi/simple
if errorlevel 1 (
    echo [ERROR] torch install failed, rerun this script to continue
    pause
    exit /b 1
)

REM 4. remaining deps
echo Installing remaining deps...
set "REQ_FILE=%ROOT%source\requirements_local_cu118.txt"
if not exist "!REQ_FILE!" set "REQ_FILE=%ROOT%requirements_local_cu118.txt"
if not exist "!REQ_FILE!" (
    echo [WARN] requirements_local_cu118.txt not found, skip deps
) else (
    "%RT%\python.exe" -m pip install --no-input --default-timeout 100 --retries 5 -r "!REQ_FILE!" -i https://mirrors.pku.edu.cn/pypi/simple
    if errorlevel 1 (
        echo [ERROR] deps install failed, rerun this script to continue
        pause
        exit /b 1
    )
)

REM 5. trim torch
echo [4/4] Trimming torch...
set "TRIM_PY=%ROOT%source\tools\trim_torch.py"
if not exist "!TRIM_PY!" set "TRIM_PY=%ROOT%tools\trim_torch.py"
if exist "!TRIM_PY!" (
    "%RT%\python.exe" "!TRIM_PY!" --torch-dir "%RT%\Lib\site-packages\torch"
)

REM check
echo.
echo Checking GPU...
"%RT%\python.exe" -c "import torch; print('torch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo.
echo ============================================================
echo   Install done! Open the app and click Start.
echo   If CUDA available: False, install NVIDIA driver and retry.
echo ============================================================
pause