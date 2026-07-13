#!/bin/bash
# EEGDataScience 首次配置脚本 (macOS)
# 作者: NoWint (https://github.com/NoWint)

cd "$(dirname "$0")"

echo "================================================"
echo "  EEGDataScience 首次配置"
echo "  作者: NoWint (https://github.com/NoWint)"
echo "================================================"
echo ""

# 检查 Python 3
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 python3，请先安装 Python 3.11+"
    echo "  下载地址: https://www.python.org/downloads/"
    read -p "按回车键退出..."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[1/3] 检测到 Python $PYTHON_VERSION"

# 检查是否已存在 venv
if [ -d "venv" ]; then
    echo "[2/3] 虚拟环境已存在，跳过创建"
else
    echo "[2/3] 创建虚拟环境..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[错误] 创建虚拟环境失败"
        read -p "按回车键退出..."
        exit 1
    fi
fi

# 激活并安装依赖
echo "[3/3] 安装依赖..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] 依赖安装失败，请检查网络连接或手动运行:"
    echo "  source venv/bin/activate && pip install -r requirements.txt"
    read -p "按回车键退出..."
    exit 1
fi

echo ""
echo "================================================"
echo "  配置完成！"
echo "  现在可以双击 start.command 启动应用"
echo "================================================"
read -p "按回车键退出..."
