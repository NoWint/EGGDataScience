@echo off
chcp 65001 > nul
REM EEGDataScience 首次配置脚本 (Windows)
REM 作者: NoWint (https://github.com/NoWint)

cd /d "%~dp0"

echo ================================================
echo   EEGDataScience 首次配置
echo   作者: NoWint (https://github.com/NoWint)
echo ================================================
echo.

REM 检查 Python
where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 python，请先安装 Python 3.11+
    echo   下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i
echo [1/3] 检测到 Python %PYTHON_VERSION%

REM 检查是否已存在 venv
if exist "venv" (
    echo [2/3] 虚拟环境已存在，跳过创建
) else (
    echo [2/3] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
)

REM 激活并安装依赖
echo [3/3] 安装依赖...
call venv\Scripts\activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败，请检查网络连接或手动运行:
    echo   call venv\Scripts\activate ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ================================================
echo   配置完成！
echo   现在可以双击 start.bat 启动应用
echo ================================================
pause
