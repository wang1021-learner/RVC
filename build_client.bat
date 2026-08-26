@echo off
chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PKG=%ROOT%dist\RVC服务器客户端"

echo ============================================
echo   RVC 服务器客户端打包（不含模型、不含 torch）
echo ============================================

echo [1/3] 构建瘦客户端 exe...
python -m PyInstaller rvc_realtime.spec --noconfirm
if errorlevel 1 goto :err

echo [2/3] 装配目录（不复制 weights / logs / infer）...
if exist "%PKG%" rmdir /s /q "%PKG%"
mkdir "%PKG%"
xcopy /e /i /q "dist\RVC实时变声" "%PKG%" >nul
copy /y "服务器版使用说明.txt" "%PKG%\使用说明.txt" >nul
> "%PKG%\pack_mode.txt" echo server

echo [3/3] 生成 zip 并统计体积...
powershell -NoProfile -Command "Compress-Archive -Path '%PKG%' -DestinationPath '%ROOT%dist\RVC服务器客户端.zip' -Force"
powershell -NoProfile -Command "$s=(Get-ChildItem '%PKG%' -Recurse -File | Measure-Object Length -Sum).Sum; Write-Host ('文件夹体积: {0:N0} MB' -f ($s/1MB))"

echo.
echo ============================================
echo   完成！
echo   分发文件: dist\RVC服务器客户端.zip
echo   用户解压后双击 RVC实时变声.exe
echo   勾选「远程服务器」，填 ws://推理机IP:8765
echo ============================================
pause
exit /b 0

:err
echo [错误] 打包失败
pause
exit /b 1
