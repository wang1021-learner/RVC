@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 启动 RVC 推理服务器（CPU 模式，端口 8765）
echo 客户端填写: ws://本机IP:8765
echo CPU 推理会比显卡慢，只适合先验证连通和加载。
echo.

set PY=
call :pick_py ".venv\Scripts\python.exe"
if defined PY goto :run
call :pick_py "venv\Scripts\python.exe"
if defined PY goto :run
call :pick_py "runtime\python.exe"
if defined PY goto :run
where py >nul 2>&1 && for /f "delims=" %%i in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PY=%%i"
if defined PY goto :run
echo [ERROR] 需要 Python 3.11 x64。请用 py -3.11 -m venv .venv，或运行 install_local.bat 生成 runtime\
pause
exit /b 1

:run
echo using: %PY%
set RVC_FORCE_CPU=1
"%PY%" -u server\rvc_server.py --host 0.0.0.0 --port 8765 --cpu %*
if errorlevel 1 pause
goto :eof

:pick_py
if not exist "%~1" exit /b 1
"%~1" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo skip %~1 ^(not Python 3.11^)
    exit /b 1
)
set "PY=%~1"
exit /b 0
