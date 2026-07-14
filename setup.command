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

# 清除代理环境变量(避免 pip 走失效的本地代理)
# 某些工具(如 ICUBE)会注入 HTTP_PROXY,但代理服务可能未运行
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

# 激活 venv
source venv/bin/activate

echo "[3/3] 检查依赖..."
echo ""

# 逐个检查 requirements.txt 中的依赖,只对未安装的包执行 pip install
# 这样已安装的依赖不会触发网络请求,避免代理失效导致整个流程中断
MISSING_PKGS=()
while IFS= read -r line; do
    # 跳过空行和注释
    line=$(echo "$line" | sed 's/#.*//' | xargs)
    [ -z "$line" ] && continue

    # 提取包名(去掉版本约束 >=, ==, >, <, ~= 等)
    pkg_name=$(echo "$line" | sed -E 's/[><=!~].*//' | xargs)

    # 用 python import 检查是否已装(比 pip show 快且更准确)
    # 包名 → 检查命令映射(大多数用 import,少数需要特殊检查)
    case "$pkg_name" in
        python-multipart) check_cmd="import multipart" ;;
        # brainflow 5.x 依赖 pkg_resources,仅 import brainflow 不够,
        # 必须能加载 BoardShim 才算真正可用
        brainflow) check_cmd="from brainflow.board_shim import BoardShim" ;;
        setuptools) check_cmd="import pkg_resources" ;;
        *) check_cmd="import $pkg_name" ;;
    esac

    if python -c "$check_cmd" 2>/dev/null; then
        echo "  [✓] $pkg_name 已安装"
    else
        echo "  [✗] $pkg_name 未安装,需要安装"
        MISSING_PKGS+=("$line")
    fi
done < requirements.txt

echo ""

# 如果所有依赖都已安装,直接完成
if [ ${#MISSING_PKGS[@]} -eq 0 ]; then
    echo "所有依赖已满足,跳过安装"
else
    echo "需要安装 ${#MISSING_PKGS[@]} 个缺失依赖..."
    echo ""

    # 升级 pip(静默)
    pip install --upgrade pip -q 2>/dev/null

    # 分两批安装:
    # 1) 普通依赖(fastapi/numpy/pandas/scipy 等) - 用默认源
    # 2) brainflow - 国内镜像可能没有,用清华源兜底

    # 收集非 brainflow 的缺失包
    NORMAL_PKGS=()
    BRAINFLOW_MISSING=false
    for pkg in "${MISSING_PKGS[@]}"; do
        if echo "$pkg" | grep -q "^brainflow"; then
            BRAINFLOW_MISSING=true
        else
            NORMAL_PKGS+=("$pkg")
        fi
    done

    # 安装普通依赖
    if [ ${#NORMAL_PKGS[@]} -gt 0 ]; then
        echo "安装普通依赖: ${NORMAL_PKGS[*]}"
        pip install "${NORMAL_PKGS[@]}" -i https://pypi.tuna.tsinghua.edu.cn/simple/
        if [ $? -ne 0 ]; then
            echo "[警告] 部分普通依赖安装失败,尝试用默认源重试..."
            pip install "${NORMAL_PKGS[@]}"
        fi
    fi

    # 安装 brainflow(国内镜像可能没有,需特殊处理)
    if [ "$BRAINFLOW_MISSING" = true ]; then
        echo ""
        echo "安装 brainflow (EEG 采集核心依赖)..."
        # 先尝试清华源
        pip install brainflow -i https://pypi.tuna.tsinghua.edu.cn/simple/
        if [ $? -ne 0 ]; then
            echo "[警告] 清华源安装失败,尝试 PyPI 官方源..."
            # 清华源失败则用官方源(可能需要科学上网)
            pip install brainflow --index-url https://pypi.org/simple/
        fi

        if [ $? -ne 0 ]; then
            echo ""
            echo "[错误] brainflow 安装失败"
            echo "  brainflow 是 BrainFlow 官方发布的预编译 wheel,国内镜像可能未同步"
            echo "  请手动安装:"
            echo "    source venv/bin/activate"
            echo "    pip install brainflow --index-url https://pypi.org/simple/"
            echo "  或从 GitHub 下载 wheel:"
            echo "    https://github.com/brainflow-dev/brainflow/releases"
            read -p "按回车键退出..."
            exit 1
        fi
    fi
fi

echo ""
echo "================================================"
echo "  配置完成!"
echo "  现在可以双击 start.command 启动应用"
echo "================================================"
read -p "按回车键退出..."
