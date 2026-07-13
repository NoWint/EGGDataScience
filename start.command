#!/bin/bash
# EEGDataScience 启动脚本 (macOS)
# 作者: NoWint (https://github.com/NoWint)

cd "$(dirname "$0")"

echo "================================================"
echo "  EEGDataScience 启动中..."
echo "  作者: NoWint (https://github.com/NoWint)"
echo "================================================"
echo ""

# 检查 venv
if [ ! -d "venv" ]; then
    echo "[错误] 未找到虚拟环境"
    echo "  首次使用请先双击 setup.command 进行配置"
    read -p "按回车键退出..."
    exit 1
fi

source venv/bin/activate

# 检查端口是否被占用
PORT=18765
if lsof -i :$PORT &> /dev/null; then
    echo "[提示] 端口 $PORT 已被占用，可能服务已在运行"
    echo "  正在尝试打开浏览器..."
    open "http://localhost:$PORT"
    exit 0
fi

# 后台启动服务
echo "[1/2] 启动分析服务 (端口 $PORT)..."
nohup python -m app.server > /tmp/eegdatascience.log 2>&1 &
SERVER_PID=$!

# 等待服务就绪
echo "[2/2] 等待服务就绪..."
READY=0
for i in $(seq 1 15); do
    if curl -s "http://localhost:$PORT/api/health" > /dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done

if [ $READY -eq 1 ]; then
    echo ""
    echo "================================================"
    echo "  服务已启动！"
    echo "  地址: http://localhost:$PORT"
    echo "  日志: /tmp/eegdatascience.log"
    echo "  PID:  $SERVER_PID"
    echo "================================================"
    echo ""
    echo "  浏览器即将打开... (此窗口可关闭)"
    open "http://localhost:$PORT"
    sleep 2
    exit 0
else
    echo ""
    echo "[错误] 服务启动失败，请查看日志:"
    echo "  cat /tmp/eegdatascience.log"
    kill $SERVER_PID 2>/dev/null
    read -p "按回车键退出..."
    exit 1
fi
