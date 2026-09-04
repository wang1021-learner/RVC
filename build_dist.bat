@echo off
chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PKG=%ROOT%dist\RVC单机版"
set "SRC=%PKG%\source"

echo ============================================
echo   RVC 单机版打包装配（构建机执行）
echo ============================================

python -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)"
if errorlevel 1 (
    echo [ERROR] 打包需要 Python 3.11 x64
    goto :err
)

echo [1/5] 构建客户端 exe...
cd /d "%ROOT%client_gui"
python -m PyInstaller rvc_realtime.spec --noconfirm
if errorlevel 1 (
    cd /d "%ROOT%"
    goto :err
)
cd /d "%ROOT%"

echo [2/5] 装配目录...
if exist "%PKG%" rmdir /s /q "%PKG%"
mkdir "%PKG%"
xcopy /e /i /q "client_gui\dist\RVC实时变声" "%PKG%" >nul

echo [3/5] 复制服务端源码与模型资产到 source\server ...
xcopy /e /i /q "server\worker" "%SRC%\server\worker" >nul
xcopy /e /i /q "server\infer" "%SRC%\server\infer" >nul
xcopy /e /i /q "server\configs" "%SRC%\server\configs" >nul
xcopy /e /i /q "server\i18n" "%SRC%\server\i18n" >nul
xcopy /e /i /q "server\tools" "%SRC%\server\tools" >nul
copy /y "server\rvc_server.py" "%SRC%\server\rvc_server.py" >nul
copy /y "server\requirements_local_cu118.txt" "%SRC%\server\requirements_local_cu118.txt" >nul
for /f "delims=" %%d in ('dir /s /b /ad "%SRC%\__pycache__" 2^>nul') do rd /s /q "%%d" 2>nul
mkdir "%SRC%\server\assets"
xcopy /e /i /q "server\assets\hubert_base" "%SRC%\server\assets\hubert_base" >nul
xcopy /e /i /q "server\assets\rmvpe" "%SRC%\server\assets\rmvpe" >nul
mkdir "%SRC%\server\assets\weights"
copy /y "server\assets\weights\myvoice.pth" "%SRC%\server\assets\weights\" >nul
copy /y "server\assets\weights\shanxi_e200_s11800.pth" "%SRC%\server\assets\weights\" >nul
mkdir "%SRC%\server\assets\indices"
copy /y "server\assets\indices\myvoice.index" "%SRC%\server\assets\indices\" >nul
copy /y "server\assets\indices\shanxi.index" "%SRC%\server\assets\indices\" >nul

echo [4/5] 复制安装脚本与使用说明...
copy /y "server\install_local.bat" "%PKG%\install_local.bat" >nul
copy /y "单机版使用说明.txt" "%PKG%\使用说明.txt" >nul
> "%PKG%\pack_mode.txt" echo standalone

echo [5/5] 生成 zip 并统计体积...
powershell -NoProfile -Command "Compress-Archive -Path '%PKG%' -DestinationPath '%ROOT%dist\RVC单机版.zip' -Force"
powershell -NoProfile -Command "$s=(Get-ChildItem '%PKG%' -Recurse -File | Measure-Object Length -Sum).Sum; Write-Host ('文件夹体积: {0:N2} GB' -f ($s/1GB))"

echo.
echo ============================================
echo   完成！
echo   分发文件: dist\RVC单机版.zip
echo   客户解压后：先点「安装本地推理」，再启动变声
echo ============================================
pause
exit /b 0

:err
echo [错误] 打包失败
pause
exit /b 1
