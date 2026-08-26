@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 启动 RVC 推理服务器（CPU 模式，端口 8765）
echo 客户端填写: ws://本机IP:8765
echo CPU 推理会比显卡慢，只适合先验证连通和加载。
echo.

set PY=
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if "%PY%"=="" if exist "venv\Scripts\python.exe" set PY=venv\Scripts\python.exe
if "%PY%"=="" set PY=python

set RVC_FORCE_CPU=1
"%PY%" -u server\rvc_server.py --host 0.0.0.0 --port 8765 --cpu
if errorlevel 1 pause
