@echo off
chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PKG=%ROOT%dist\RVC单机版"
set "SRC=%PKG%\source"

echo ============================================
echo   RVC 单机版打包装配（构建机执行）
echo ============================================

echo [1/5] 构建客户端 exe...
python -m PyInstaller rvc_realtime.spec --noconfirm
if errorlevel 1 goto :err

echo [2/5] 装配目录...
if exist "%PKG%" rmdir /s /q "%PKG%"
mkdir "%PKG%"
xcopy /e /i /q "dist\RVC实时变声" "%PKG%" >nul

echo [3/5] 复制服务端源码与模型资产...
for %%d in (configs infer i18n tools worker server) do xcopy /e /i /q "%%d" "%SRC%\%%d" >nul
for /f "delims=" %%d in ('dir /s /b /ad "%SRC%\__pycache__" 2^>nul') do rd /s /q "%%d" 2>nul
mkdir "%SRC%\assets"
xcopy /e /i /q "assets\hubert_base" "%SRC%\assets\hubert_base" >nul
xcopy /e /i /q "assets\rmvpe" "%SRC%\assets\rmvpe" >nul
mkdir "%SRC%\assets\weights"
copy /y "assets\weights\thchs_female_200e.pth" "%SRC%\assets\weights\" >nul
copy /y "assets\weights\thchs_female_300e.pth" "%SRC%\assets\weights\" >nul
copy /y "assets\weights\thchs_v2_e200_s13200.pth" "%SRC%\assets\weights\" >nul
copy /y "assets\weights\myvoice.pth" "%SRC%\assets\weights\" >nul
copy /y "assets\weights\shanxi.pth" "%SRC%\assets\weights\" >nul
copy /y "assets\weights\shanxi_e200_s14800.pth" "%SRC%\assets\weights\" >nul
copy /y "assets\weights\shanxi_e200_s11800.pth" "%SRC%\assets\weights\" >nul
mkdir "%SRC%\logs\thchs_v2"
copy /y "logs\thchs_v2\added_IVF2716_Flat_nprobe_1_thchs_v2_v2.index" "%SRC%\logs\thchs_v2\" >nul
copy /y "logs\thchs_v2\added_IVF314_Flat_nprobe_1_myvoice_v2.index" "%SRC%\logs\thchs_v2\" >nul
copy /y "logs\thchs_v2\shanxi.index" "%SRC%\logs\thchs_v2\" >nul
copy /y "requirements_local_cu118.txt" "%SRC%\requirements_local_cu118.txt" >nul

echo [4/5] 复制安装脚本与使用说明...
copy /y "install_local.bat" "%PKG%\install_local.bat" >nul
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
