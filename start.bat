@echo off
chcp 65001 > nul
REM EEGDataScience 启动脚本 (Windows)
REM 作者: NoWint (https://github.com/NoWint)

cd /d "%~dp0"

echo ================================================
echo   EEGDataScience 启动中...
echo   作者: NoWint (https://github.com/NoWint)
echo ================================================
echo.

REM 检查 venv
if not exist "venv" (
    echo [错误] 未找到虚拟环境
    echo   首次使用请先双击 setup.bat 进行配置
    pause
    exit /b 1
)

call venv\Scripts\activate

REM 检查端口是否被占用
set PORT=18765
netstat -an | findstr ":%PORT% " | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo [提示] 端口 %PORT% 已被占用，可能服务已在运行
    echo   正在尝试打开浏览器...
    start http://localhost:%PORT%
    exit /b 0
)

REM 后台启动服务
echo [1/2] 启动分析服务 (端口 %PORT%)...
start /b pythonw -m app.server > "%TEMP%\eegdatascience.log" 2>&1

REM 等待服务就绪
echo [2/2] 等待服务就绪...
set COUNT=0
:waitloop
timeout /t 1 /nobreak > nul
set /a COUNT+=1

curl -s http://localhost:%PORT%/api/health >nul 2>&1
if not errorlevel 1 goto ready

if %COUNT% lss 15 goto waitloop

echo.
echo [错误] 服务启动失败，请查看日志:
echo   type "%TEMP%\eegdatascience.log"
pause
exit /b 1

:ready
echo.
echo ================================================
echo   服务已启动！
echo   地址: http://localhost:%PORT%
echo   日志: %TEMP%\eegdatascience.log
echo ================================================
echo.
echo   浏览器即将打开... (此窗口可关闭)
start http://localhost:%PORT%
timeout /t 2 /nobreak > nul
exit /b 0
