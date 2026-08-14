@echo off
chcp 65001 >nul
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"
echo ============================================
echo   RVC 实时语音转换 - 打包构建脚本
echo ============================================
echo.
echo 项目根目录: %CD%

REM ── 查找 conda ──
set "CONDA="
where conda >nul 2>&1
if not errorlevel 1 (
    set "CONDA=conda"
    goto :found_conda
)
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
    set "CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
    goto :found_conda
)
if exist "%USERPROFILE%\Miniconda3\Scripts\conda.exe" (
    set "CONDA=%USERPROFILE%\Miniconda3\Scripts\conda.exe"
    goto :found_conda
)
if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" (
    set "CONDA=C:\ProgramData\miniconda3\Scripts\conda.exe"
    goto :found_conda
)

echo [错误] 未找到 conda
echo 请确认 miniconda3 已正确安装
pause
exit /b 1

:found_conda
echo 找到 conda: %CONDA%
echo.

REM ── 创建/复用 conda 环境 ──
for /f "tokens=*" %%i in ('%CONDA% env list 2^>nul ^| findstr "rvc_build"') do set "ENV_LINE=%%i"

if defined ENV_LINE (
    echo 环境已存在，直接复用
) else (
    echo [1/4] 创建 conda 环境 rvc_build (Python 3.12)...
    call %CONDA% create -n rvc_build python=3.12 -y
    if errorlevel 1 (
        echo [错误] conda 环境创建失败
        pause
        exit /b 1
    )
)

REM ── 获取环境中的 Python 路径 ──
set "ENV_PY="
for /f "tokens=*" %%i in ('%CONDA% run -n rvc_build python -c "import sys; print(sys.executable)" 2^>nul') do set "ENV_PY=%%i"

if not defined ENV_PY (
    echo [错误] 无法获取 rvc_build 环境的 Python 路径
    pause
    exit /b 1
)

echo 环境 Python: %ENV_PY%
echo.

REM ── 安装依赖 ──
echo [2/4] 安装依赖...
"%ENV_PY%" -m pip install --upgrade pip
"%ENV_PY%" -m pip install -r deploy\requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)
"%ENV_PY%" -m pip install pyinstaller
if errorlevel 1 (
    echo [错误] PyInstaller 安装失败
    pause
    exit /b 1
)

REM ── 检查资源文件 ──
echo.
echo [3/4] 检查资源文件...
if not exist "assets\rmvpe\rmvpe.pt" echo [警告] 未找到 assets\rmvpe\rmvpe.pt
if not exist "assets\hubert_base" echo [警告] 未找到 assets\hubert_base 目录
if not exist "assets\weights\thchs_female_200e.pth" echo [警告] 未找到模型文件
if not exist "speakers.json" echo [警告] 未找到 speakers.json

REM ── 构建 exe ──
echo.
echo [4/4] 开始打包 (可能需要5-10分钟)...
"%ENV_PY%" -m PyInstaller rvc_realtime.spec --noconfirm
if errorlevel 1 (
    echo [错误] 打包失败
    pause
    exit /b 1
)

echo.
echo ============================================
echo   构建完成！
echo   输出目录: dist\RVC实时语音转换\
echo   可执行文件: dist\RVC实时语音转换\RVC实时语音转换.exe
echo ============================================
echo.
echo   将 dist\RVC实时语音转换\ 整个文件夹拷贝给用户
echo   用户双击 RVC实时语音转换.exe 即可运行
echo.
pause
