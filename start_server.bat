@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 启动 RVC 推理服务器（局域网端口 8765）
echo 其它电脑客户端填写: ws://本机IP:8765
echo.

set PY=
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if "%PY%"=="" if exist "venv\Scripts\python.exe" set PY=venv\Scripts\python.exe
if "%PY%"=="" set PY=python

"%PY%" -u server\rvc_server.py --host 0.0.0.0 --port 8765
if errorlevel 1 pause
