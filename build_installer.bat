@echo off
chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
  echo 未找到 Inno Setup。请先安装：winget install JRSoftware.InnoSetup
  exit /b 1
)

if not exist "dist\RVC服务器客户端\RVC实时变声.exe" (
  echo 请先运行 build_client.bat 生成客户端文件
  exit /b 1
)

echo 正在编译服务器版安装包...
"%ISCC%" /Q "installer\rvc_server_client.iss"
if errorlevel 1 exit /b 1
echo 完成: dist\RVC实时变声-服务器版安装.exe
exit /b 0
