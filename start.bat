@echo off
chcp 65001 >nul
title AI语音政治风险监测系统

echo.
echo ========================================
echo    AI语音政治风险监测系统
echo ========================================
echo.

echo 正在检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    echo 请确保已安装Python 3.8+
    pause
    exit /b 1
)

echo 正在检查虚拟环境...
if not exist "audio_ai\Scripts\activate.bat" (
    echo 正在创建虚拟环境...
    python -m venv audio_ai
    if errorlevel 1 (
        echo 错误: 虚拟环境创建失败
        pause
        exit /b 1
    )
)

echo 正在激活虚拟环境...
call audio_ai\Scripts\activate.bat

echo 正在安装依赖包...
pip install -r requirements.txt
if errorlevel 1 (
    echo 错误: 依赖包安装失败
    pause
    exit /b 1
)

echo.
echo 正在启动系统...
echo 请稍等，模型加载需要一些时间...
echo.
echo 启动完成后，请在浏览器中访问: http://localhost:8000
echo.

python main.py

echo.
echo 系统已关闭
pause
