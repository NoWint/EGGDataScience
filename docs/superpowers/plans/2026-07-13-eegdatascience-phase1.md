# EEGDataScience 阶段 1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 EEG 心流恢复分析工具箱重组为标准 Python 项目结构，提供双击启动能力（macOS + Windows），并重构 UI 为侧边导航 + 左右分栏布局（紧凑浅色 TraeWork 风格）。

**Architecture:** 单体 FastAPI 应用，分析模块组织在 `app/analysis/` 包下。前端采用侧边导航（200px 分组式）+ 主内容区左右分栏（240px 配置面板 + 结果区）。延续 TraeWork 设计令牌（Brand #4B3FE3、Invert #262626、border-based layering）。

**Tech Stack:** Python 3.11+ / FastAPI / uvicorn / numpy / pandas / scipy / 原生 HTML + CSS + Chart.js 4.4.1

**Spec:** [docs/specs/2026-07-13-eegdatascience-phase1-design.md](file:///Users/xiatian/Desktop/EEG-Science/docs/specs/2026-07-13-eegdatascience-phase1-design.md)

---

## 文件结构

实施完成后的目标结构：

```
EEGDataScience/  (项目根目录保持 EEG-Science, 内部重组)
├── requirements.txt              # 新建
├── .gitignore                    # 新建
├── start.command                 # 新建 - macOS 双击启动
├── start.bat                     # 新建 - Windows 双击启动
├── setup.command                 # 新建 - macOS 首次配置
├── setup.bat                     # 新建 - Windows 首次配置
├── README.md                     # 新建 - 简短使用说明
├── app/                          # 新建 - 主应用包
│   ├── __init__.py               # 新建 (空)
│   ├── server.py                 # 从 toolbox/server.py 迁移, 更新导入路径
│   ├── analysis/
│   │   ├── __init__.py           # 新建 (导出 flow_recovery 的公共 API)
│   │   └── flow_recovery.py      # 从 toolbox/analysis.py 迁移, 无逻辑改动
│   └── static/
│       ├── index.html            # 重构 - 侧边导航 + 左右分栏
│       ├── css/
│       │   └── style.css         # 重构 - 匹配新 HTML 结构
│       └── js/
│           └── app.js            # 更新 - 适配新 HTML 结构
├── data/                         # 新建 - 用户数据目录
│   └── uploads/                  # 新建 - 上传文件存放
├── docs/                         # 已存在
│   ├── specs/
│   │   └── 2026-07-13-eegdatascience-phase1-design.md
│   └── superpowers/
│       └── plans/
│           └── 2026-07-13-eegdatascience-phase1.md  (本文件)
└── (旧 toolbox/ 目录在最终验证后删除)
```

**文件职责说明：**
- `app/server.py` — FastAPI 入口，路由定义，结果存储，报告生成。不包含分析逻辑。
- `app/analysis/flow_recovery.py` — 纯分析模块：预处理、特征提取、恢复时长、衰减幅度、统计检验、样例数据生成。无 FastAPI 依赖。
- `app/analysis/__init__.py` — 包初始化，导出公共 API 供 server.py 导入。
- `app/static/index.html` — 页面骨架，侧边栏 + 配置面板 + 结果区结构。
- `app/static/css/style.css` — 所有视觉样式，TraeWork 设计令牌。
- `app/static/js/app.js` — 前端交互逻辑，API 调用，图表渲染。
- `start.command` / `start.bat` — 双击启动，激活 venv → 启动服务 → 打开浏览器。
- `setup.command` / `setup.bat` — 首次配置，创建 venv → 安装依赖。

---

## Task 1: 创建项目骨架文件

**Files:**
- Create: `/Users/xiatian/Desktop/EEG-Science/requirements.txt`
- Create: `/Users/xiatian/Desktop/EEG-Science/.gitignore`
- Create: `/Users/xiatian/Desktop/EEG-Science/app/__init__.py`
- Create: `/Users/xiatian/Desktop/EEG-Science/app/analysis/__init__.py`
- Create: `/Users/xiatian/Desktop/EEG-Science/data/uploads/.gitkeep`
- Create: `/Users/xiatian/Desktop/EEG-Science/README.md`

- [ ] **Step 1: 创建目录结构**

```bash
cd /Users/xiatian/Desktop/EEG-Science
mkdir -p app/analysis app/static/css app/static/js data/uploads
```

- [ ] **Step 2: 创建 requirements.txt**

文件路径：`/Users/xiatian/Desktop/EEG-Science/requirements.txt`

```
fastapi>=0.135.0
uvicorn>=0.42.0
numpy>=2.0.0
pandas>=2.0.0
scipy>=1.17.0
```

- [ ] **Step 3: 创建 .gitignore**

文件路径：`/Users/xiatian/Desktop/EEG-Science/.gitignore`

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# 项目数据
data/uploads/*
!data/uploads/.gitkeep

# 日志
*.log
```

- [ ] **Step 4: 创建 app/__init__.py（空文件）**

文件路径：`/Users/xiatian/Desktop/EEG-Science/app/__init__.py`

```python
"""EEGDataScience 应用包"""
```

- [ ] **Step 5: 创建 app/analysis/__init__.py**

文件路径：`/Users/xiatian/Desktop/EEG-Science/app/analysis/__init__.py`

```python
"""EEG 分析模块包"""
from .flow_recovery import (
    load_eeg, load_events, preprocess,
    compute_band_powers, compute_entropy, extract_features,
    compute_recovery_time, compute_all_recovery, compute_attenuation,
    paired_t_test, repeated_measures_anova, pearson_correlation,
    generate_sample_eeg, events_to_df, run_full_pipeline,
    BANDS,
)

__all__ = [
    'load_eeg', 'load_events', 'preprocess',
    'compute_band_powers', 'compute_entropy', 'extract_features',
    'compute_recovery_time', 'compute_all_recovery', 'compute_attenuation',
    'paired_t_test', 'repeated_measures_anova', 'pearson_correlation',
    'generate_sample_eeg', 'events_to_df', 'run_full_pipeline',
    'BANDS',
]
```

- [ ] **Step 6: 创建 data/uploads/.gitkeep（空文件）**

文件路径：`/Users/xiatian/Desktop/EEG-Science/data/uploads/.gitkeep`

```
```
（空内容，仅用于保留目录）

- [ ] **Step 7: 创建 README.md**

文件路径：`/Users/xiatian/Desktop/EEG-Science/README.md`

```markdown
# EEGDataScience

跨学科任务切换对心流状态的影响及 EEG 恢复时间量化研究 — 便携式 EEG 头环实验数据分析平台。

作者: [NoWint](https://github.com/NoWint)

## 快速开始

### 首次配置
- **macOS**: 双击 `setup.command`
- **Windows**: 双击 `setup.bat`

### 启动应用
- **macOS**: 双击 `start.command`
- **Windows**: 双击 `start.bat`

启动后浏览器会自动打开 `http://localhost:18765`。

## 功能
- 心流恢复分析（4 种实验条件：A→A / A→B / A→C / B→C）
- 6 项 EEG 指标时序可视化（Theta/Alpha 比值、Alpha/Beta/Gamma 能量、谱熵、认知负载指数）
- 恢复时长量化（±5% 容差，30s 连续窗口）
- 跨条件统计比较（配对 t 检验、重复测量 ANOVA）
- 结构化分析报告生成

## 技术栈
Python 3.11+ / FastAPI / numpy / pandas / scipy / Chart.js
```

- [ ] **Step 8: 验证目录结构**

Run: `ls -la /Users/xiatian/Desktop/EEG-Science/app/ /Users/xiatian/Desktop/EEG-Science/data/uploads/`
Expected: 看到 `__init__.py`、`analysis/`、`static/` 子目录，`data/uploads/.gitkeep` 存在

- [ ] **Step 9: Commit**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git add requirements.txt .gitignore app/__init__.py app/analysis/__init__.py data/uploads/.gitkeep README.md
git commit -m "chore: 创建项目骨架 (requirements, gitignore, app 包结构)"
```

---

## Task 2: 迁移 analysis.py 到 app/analysis/flow_recovery.py

**Files:**
- Create: `/Users/xiatian/Desktop/EEG-Science/app/analysis/flow_recovery.py` (从 `toolbox/analysis.py` 复制，无逻辑改动)
- Source: `/Users/xiatian/Desktop/EEG-Science/toolbox/analysis.py`

- [ ] **Step 1: 复制 analysis.py 到新位置**

```bash
cd /Users/xiatian/Desktop/EEG-Science
cp toolbox/analysis.py app/analysis/flow_recovery.py
```

- [ ] **Step 2: 验证文件完整**

Run: `wc -l /Users/xiatian/Desktop/EEG-Science/app/analysis/flow_recovery.py`
Expected: 498 行（与原文件一致）

- [ ] **Step 3: 验证可导入（无语法错误）**

```bash
cd /Users/xiatian/Desktop/EEG-Science
python3 -c "from app.analysis.flow_recovery import run_full_pipeline, generate_sample_eeg; print('OK')"
```
Expected: 输出 `OK`，无 ImportError

- [ ] **Step 4: 验证包导入路径**

```bash
cd /Users/xiatian/Desktop/EEG-Science
python3 -c "from app.analysis import BANDS, run_full_pipeline; print(BANDS); print('Package import OK')"
```
Expected: 输出频段字典和 `Package import OK`

- [ ] **Step 5: Commit**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git add app/analysis/flow_recovery.py
git commit -m "refactor: 迁移 analysis.py → app/analysis/flow_recovery.py"
```

---

## Task 3: 迁移 server.py 到 app/server.py

**Files:**
- Create: `/Users/xiatian/Desktop/EEG-Science/app/server.py` (从 `toolbox/server.py` 迁移，更新导入路径和静态文件路径)
- Source: `/Users/xiatian/Desktop/EEG-Science/toolbox/server.py`

**关键修改点：**
1. 导入语句：`from analysis import (...)` → `from app.analysis import (...)`
2. `BASE_DIR`：`Path(__file__).parent` 现在指向 `app/`，静态文件目录为 `BASE_DIR / "static"`（不变）
3. `UPLOAD_DIR`：改为 `BASE_DIR.parent / "data" / "uploads"`（指向项目根/data/uploads）
4. 启动信息中的路径提示更新

- [ ] **Step 1: 复制 server.py 到新位置**

```bash
cd /Users/xiatian/Desktop/EEG-Science
cp toolbox/server.py app/server.py
```

- [ ] **Step 2: 修改导入语句**

在 `/Users/xiatian/Desktop/EEG-Science/app/server.py` 中，将第 19-24 行：

```python
from analysis import (
    generate_sample_eeg, events_to_df, run_full_pipeline,
    load_eeg, load_events, preprocess, extract_features,
    compute_all_recovery, compute_attenuation,
    paired_t_test, repeated_measures_anova, pearson_correlation,
)
```

改为：

```python
from app.analysis import (
    generate_sample_eeg, events_to_df, run_full_pipeline,
    load_eeg, load_events, preprocess, extract_features,
    compute_all_recovery, compute_attenuation,
    paired_t_test, repeated_measures_anova, pearson_correlation,
)
```

- [ ] **Step 3: 修改 UPLOAD_DIR 路径**

在 `/Users/xiatian/Desktop/EEG-Science/app/server.py` 中，将第 30 行：

```python
UPLOAD_DIR = BASE_DIR / "uploads"
```

改为：

```python
UPLOAD_DIR = BASE_DIR.parent / "data" / "uploads"
```

- [ ] **Step 4: 更新启动信息（可选，第 440-446 行）**

将 `if __name__ == "__main__":` 块中的 print 信息更新为项目新名称：

```python
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  EEGDataScience — 心流恢复分析")
    print("  跨学科任务切换 EEG 恢复时间量化研究")
    print("=" * 60)
    print(f"  服务地址: http://localhost:18765")
    print(f"  静态目录: {STATIC_DIR}")
    print(f"  上传目录: {UPLOAD_DIR}")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=18765)
```

- [ ] **Step 5: 验证可导入**

```bash
cd /Users/xiatian/Desktop/EEG-Science
python3 -c "from app.server import app; print('Server import OK')"
```
Expected: 输出 `Server import OK`，无 ImportError

- [ ] **Step 6: Commit**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git add app/server.py
git commit -m "refactor: 迁移 server.py → app/server.py, 更新导入路径"
```

---

## Task 4: 迁移静态文件到 app/static/

**Files:**
- Create: `/Users/xiatian/Desktop/EEG-Science/app/static/index.html` (临时复制旧版, Task 8 会重构)
- Create: `/Users/xiatian/Desktop/EEG-Science/app/static/css/style.css` (临时复制旧版, Task 9 会重构)
- Create: `/Users/xiatian/Desktop/EEG-Science/app/static/js/app.js` (临时复制旧版, Task 10 会更新)

- [ ] **Step 1: 复制静态文件**

```bash
cd /Users/xiatian/Desktop/EEG-Science
cp toolbox/static/index.html app/static/index.html
cp toolbox/static/css/style.css app/static/css/style.css
cp toolbox/static/js/app.js app/static/js/app.js
```

- [ ] **Step 2: 验证文件存在**

Run: `ls -la /Users/xiatian/Desktop/EEG-Science/app/static/ /Users/xiatian/Desktop/EEG-Science/app/static/css/ /Users/xiatian/Desktop/EEG-Science/app/static/js/`
Expected: 三个文件都存在

- [ ] **Step 3: 启动服务验证基础迁移**

```bash
cd /Users/xiatian/Desktop/EEG-Science
python3 app/server.py &
sleep 3
curl -s http://localhost:18765/api/health
curl -s -o /dev/null -w "%{http_code}" http://localhost:18765/
curl -s -o /dev/null -w "%{http_code}" http://localhost:18765/static/css/style.css
curl -s -o /dev/null -w "%{http_code}" http://localhost:18765/static/js/app.js
kill %1 2>/dev/null
```
Expected: health 返回 `{"status":"ok",...}`，页面和静态文件都返回 `200`

- [ ] **Step 4: Commit**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git add app/static/
git commit -m "refactor: 迁移静态文件 → app/static/ (临时保留旧版, 后续重构)"
```

---

## Task 5: 创建 macOS 启动脚本

**Files:**
- Create: `/Users/xiatian/Desktop/EEG-Science/setup.command`
- Create: `/Users/xiatian/Desktop/EEG-Science/start.command`

- [ ] **Step 1: 创建 setup.command**

文件路径：`/Users/xiatian/Desktop/EEG-Science/setup.command`

```bash
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
```

- [ ] **Step 2: 创建 start.command**

文件路径：`/Users/xiatian/Desktop/EEG-Science/start.command`

```bash
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
nohup python app/server.py > /tmp/eegdatascience.log 2>&1 &
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
```

- [ ] **Step 3: 赋予执行权限**

```bash
cd /Users/xiatian/Desktop/EEG-Science
chmod +x setup.command start.command
```

- [ ] **Step 4: 验证脚本可执行**

Run: `ls -la /Users/xiatian/Desktop/EEG-Science/*.command`
Expected: 两个文件都有 `x` 执行权限

- [ ] **Step 5: 验证脚本语法**

```bash
cd /Users/xiatian/Desktop/EEG-Science
bash -n setup.command && echo "setup.command syntax OK"
bash -n start.command && echo "start.command syntax OK"
```
Expected: 两个都输出 `syntax OK`

- [ ] **Step 6: Commit**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git add setup.command start.command
git commit -m "feat: 添加 macOS 启动脚本 (setup.command, start.command)"
```

---

## Task 6: 创建 Windows 启动脚本

**Files:**
- Create: `/Users/xiatian/Desktop/EEG-Science/setup.bat`
- Create: `/Users/xiatian/Desktop/EEG-Science/start.bat`

- [ ] **Step 1: 创建 setup.bat**

文件路径：`/Users/xiatian/Desktop/EEG-Science/setup.bat`

```bat
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
```

- [ ] **Step 2: 创建 start.bat**

文件路径：`/Users/xiatian/Desktop/EEG-Science/start.bat`

```bat
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
start /b pythonw app\server.py > "%TEMP%\eegdatascience.log" 2>&1

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
```

- [ ] **Step 3: 验证文件存在**

Run: `ls -la /Users/xiatian/Desktop/EEG-Science/*.bat`
Expected: 两个 .bat 文件存在

- [ ] **Step 4: Commit**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git add setup.bat start.bat
git commit -m "feat: 添加 Windows 启动脚本 (setup.bat, start.bat)"
```

---

## Task 7: 验证基础迁移完成

此任务不创建新文件，仅验证 Task 1-6 的迁移结果。**不涉及 UI 重构**，UI 重构在 Task 8-11 进行。

- [ ] **Step 1: 创建并激活 venv，安装依赖**

```bash
cd /Users/xiatian/Desktop/EEG-Science
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Expected: 所有依赖安装成功

- [ ] **Step 2: 启动服务**

```bash
cd /Users/xiatian/Desktop/EEG-Science
python app/server.py &
sleep 3
```
Expected: 服务启动，输出启动信息

- [ ] **Step 3: 验证 API 端点**

```bash
curl -s http://localhost:18765/api/health
curl -s -X POST http://localhost:18765/api/sample -H "Content-Type: application/json" -d '{"condition":"AtoB","fs":250}'
curl -s http://localhost:18765/api/stats
curl -s -o /dev/null -w "HTML: %{http_code}\n" http://localhost:18765/
curl -s -o /dev/null -w "CSS: %{http_code}\n" http://localhost:18765/static/css/style.css
curl -s -o /dev/null -w "JS: %{http_code}\n" http://localhost:18765/static/js/app.js
```
Expected: health 返回 ok，sample 返回分析结果 JSON，stats 返回统计结果，静态文件都 200

- [ ] **Step 4: 停止服务**

```bash
kill %1 2>/dev/null
```

- [ ] **Step 5: 验证上传目录创建**

Run: `ls -la /Users/xiatian/Desktop/EEG-Science/data/uploads/`
Expected: 目录存在（server.py 启动时会自动创建）

- [ ] **Step 6: 验证启动脚本语法**

```bash
cd /Users/xiatian/Desktop/EEG-Science
bash -n setup.command && echo "setup OK"
bash -n start.command && echo "start OK"
ls -la *.command *.bat
```
Expected: 所有脚本语法正确，有执行权限

- [ ] **Step 7: Commit（如有变更）**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git status
# 如果有 venv 相关的变更被意外跟踪，添加到 .gitignore
git add -A
git commit -m "chore: 验证基础迁移完成" || echo "无变更需提交"
```

---

## Task 8: 重构 HTML 为侧边导航 + 左右分栏布局

**Files:**
- Modify: `/Users/xiatian/Desktop/EEG-Science/app/static/index.html` (完全重写)

**设计要点：**
- 左侧 200px 侧边栏：分组式（分析模块 / 数据 / 其他）
- 主内容区左右分栏：240px 配置面板 + 结果区
- 当前仅"心流恢复分析"模块可用，其他模块为占位（点击无效果或提示"即将推出"）
- 作者 footer 在侧边栏底部
- 保留所有现有元素的 ID，以便 JS 适配最小化

- [ ] **Step 1: 重写 index.html**

文件路径：`/Users/xiatian/Desktop/EEG-Science/app/static/index.html`

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEGDataScience — 心流恢复分析</title>
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>

<div class="app-shell">

    <!-- ========== 侧边栏 ========== -->
    <aside class="sidebar">
        <div class="sidebar-brand">
            <span class="brand-mark"></span>
            <span class="brand-name">EEGDataScience</span>
        </div>

        <nav class="sidebar-nav">
            <div class="nav-group">
                <div class="nav-group-label">分析模块</div>
                <a class="nav-item active" data-module="flow-recovery">
                    <span class="nav-item-text">心流恢复分析</span>
                </a>
                <a class="nav-item nav-item-disabled" data-module="spectrum">
                    <span class="nav-item-text">频谱分析</span>
                    <span class="nav-item-tag">即将推出</span>
                </a>
                <a class="nav-item nav-item-disabled" data-module="erp">
                    <span class="nav-item-text">事件相关电位</span>
                    <span class="nav-item-tag">即将推出</span>
                </a>
                <a class="nav-item nav-item-disabled" data-module="connectivity">
                    <span class="nav-item-text">脑连接分析</span>
                    <span class="nav-item-tag">即将推出</span>
                </a>
            </div>

            <div class="nav-group">
                <div class="nav-group-label">数据</div>
                <a class="nav-item nav-item-disabled" data-module="subjects">
                    <span class="nav-item-text">被试管理</span>
                    <span class="nav-item-tag">即将推出</span>
                </a>
                <a class="nav-item nav-item-disabled" data-module="experiments">
                    <span class="nav-item-text">实验记录</span>
                    <span class="nav-item-tag">即将推出</span>
                </a>
                <a class="nav-item nav-item-disabled" data-module="archive">
                    <span class="nav-item-text">数据归档</span>
                    <span class="nav-item-tag">即将推出</span>
                </a>
            </div>

            <div class="nav-group">
                <div class="nav-group-label">其他</div>
                <a class="nav-item nav-item-disabled" data-module="settings">
                    <span class="nav-item-text">设置</span>
                </a>
            </div>
        </nav>

        <div class="sidebar-footer">
            <div class="footer-author">
                <span class="footer-label">作者</span>
                <a href="https://github.com/NoWint" target="_blank" class="footer-link">NoWint</a>
            </div>
        </div>
    </aside>

    <!-- ========== 主内容区 ========== -->
    <main class="main-area">

        <!-- 模块标题栏 -->
        <div class="module-header">
            <div class="module-title-group">
                <span class="module-breadcrumb">分析模块</span>
                <h1 class="module-title">心流恢复分析</h1>
            </div>
            <div class="module-actions">
                <button class="btn btn-secondary" id="btn-refresh-stats" title="刷新跨条件统计">刷新统计</button>
                <button class="btn btn-primary" id="btn-report-full" title="生成综合报告">生成报告</button>
            </div>
        </div>

        <!-- 左右分栏工作区 -->
        <div class="workspace">

            <!-- 左侧:配置面板 -->
            <aside class="config-panel">

                <!-- 数据来源 -->
                <div class="config-section">
                    <div class="config-section-label">数据来源</div>
                    <div class="source-tabs">
                        <button class="tab-btn active" data-tab="sample">模拟数据</button>
                        <button class="tab-btn" data-tab="upload">上传数据</button>
                    </div>

                    <div class="tab-panel active" id="panel-sample">
                        <div class="condition-list">
                            <button class="cond-card active" data-condition="AtoA">
                                <span class="cond-tag">对照</span>
                                <span class="cond-name">A → A</span>
                                <span class="cond-desc">同学科连续</span>
                            </button>
                            <button class="cond-card" data-condition="AtoB">
                                <span class="cond-tag">文理</span>
                                <span class="cond-name">A → B</span>
                                <span class="cond-desc">数理 → 语言</span>
                            </button>
                            <button class="cond-card" data-condition="AtoC">
                                <span class="cond-tag">理艺</span>
                                <span class="cond-name">A → C</span>
                                <span class="cond-desc">数理 → 艺术</span>
                            </button>
                            <button class="cond-card" data-condition="BtoC">
                                <span class="cond-tag">文艺</span>
                                <span class="cond-name">B → C</span>
                                <span class="cond-desc">语言 → 艺术</span>
                            </button>
                        </div>
                    </div>

                    <div class="tab-panel" id="panel-upload">
                        <div class="upload-area">
                            <div class="upload-slot">
                                <label class="upload-label">EEG 文件 (.csv)</label>
                                <input type="file" id="file-eeg" accept=".csv" class="upload-input">
                                <span class="upload-hint" id="hint-eeg">未选择</span>
                            </div>
                            <div class="upload-slot">
                                <label class="upload-label">事件标记 (.csv, 可选)</label>
                                <input type="file" id="file-events" accept=".csv" class="upload-input">
                                <span class="upload-hint" id="hint-events">未选择</span>
                            </div>
                            <input type="text" id="upload-condition" class="text-input" placeholder="条件名称" value="custom">
                        </div>
                    </div>
                </div>

                <!-- 分析参数 -->
                <div class="config-section">
                    <div class="config-section-label">分析参数</div>
                    <div class="param-list">
                        <div class="param-item">
                            <label>带通下限 (Hz)</label>
                            <input type="number" id="param-hp" value="1.0" step="0.5" min="0.1">
                        </div>
                        <div class="param-item">
                            <label>带通上限 (Hz)</label>
                            <input type="number" id="param-lp" value="45.0" step="1" min="10">
                        </div>
                        <div class="param-item">
                            <label>陷波频率 (Hz)</label>
                            <input type="number" id="param-notch" value="50.0" step="1">
                        </div>
                        <div class="param-item">
                            <label>伪迹阈值 (μV)</label>
                            <input type="number" id="param-artifact" value="100.0" step="10">
                        </div>
                        <div class="param-item">
                            <label>分析窗口 (s)</label>
                            <input type="number" id="param-window" value="2.0" step="0.5" min="0.5">
                        </div>
                        <div class="param-item">
                            <label>重叠比例</label>
                            <input type="number" id="param-overlap" value="0.5" step="0.1" min="0" max="0.9">
                        </div>
                        <div class="param-item">
                            <label>恢复容差 (%)</label>
                            <input type="number" id="param-tolerance" value="5" step="1" min="1" max="20">
                        </div>
                        <div class="param-item">
                            <label>达标窗口 (s)</label>
                            <input type="number" id="param-recovery-win" value="30" step="5" min="10">
                        </div>
                    </div>
                </div>

                <!-- 操作按钮 -->
                <div class="config-actions">
                    <button class="btn btn-primary btn-block" id="btn-run-sample">运行分析</button>
                    <button class="btn btn-secondary btn-block" id="btn-upload" style="display:none;">上传并分析</button>
                    <button class="btn btn-ghost btn-block" id="btn-report-single">生成当前条件报告</button>
                </div>

            </aside>

            <!-- 右侧:结果区 -->
            <section class="result-area">

                <!-- 结果状态提示(初始) -->
                <div class="result-empty" id="result-empty">
                    <div class="empty-icon"></div>
                    <p class="empty-text">选择实验条件并点击「运行分析」</p>
                    <p class="empty-hint">系统将生成模拟 EEG 数据并运行完整分析流水线</p>
                </div>

                <!-- 结果内容(分析后显示) -->
                <div class="result-content" id="result-content" style="display:none;">

                    <!-- 条件标识 -->
                    <div class="result-header">
                        <span class="result-badge" id="result-condition-badge">—</span>
                    </div>

                    <!-- 指标卡片 -->
                    <div class="metric-row">
                        <div class="metric-card">
                            <span class="metric-label">恢复时长</span>
                            <div class="metric-value-row">
                                <span class="metric-value" id="metric-recovery">—</span>
                                <span class="metric-unit">秒</span>
                            </div>
                        </div>
                        <div class="metric-card">
                            <span class="metric-label">伪迹占比</span>
                            <div class="metric-value-row">
                                <span class="metric-value" id="metric-artifact">—</span>
                                <span class="metric-unit">%</span>
                            </div>
                        </div>
                        <div class="metric-card">
                            <span class="metric-label">数据时长</span>
                            <div class="metric-value-row">
                                <span class="metric-value" id="metric-duration">—</span>
                                <span class="metric-unit">分钟</span>
                            </div>
                        </div>
                        <div class="metric-card">
                            <span class="metric-label">采样点数</span>
                            <div class="metric-value-row">
                                <span class="metric-value" id="metric-samples">—</span>
                                <span class="metric-unit">点</span>
                            </div>
                        </div>
                    </div>

                    <!-- 时序曲线图 -->
                    <div class="chart-block">
                        <div class="block-header">
                            <div>
                                <h3 class="block-title">心流稳态 — 跌落 — 恢复 时序演化</h3>
                                <p class="block-desc">各 EEG 指标归一化至心流稳态均值（=1.0），阴影带为±5%恢复阈值区间。</p>
                            </div>
                        </div>
                        <div class="chart-container">
                            <canvas id="chart-timeseries"></canvas>
                        </div>
                        <div class="legend-row" id="indicator-legend"></div>
                    </div>

                    <!-- 恢复时长明细 -->
                    <div class="chart-block">
                        <div class="block-header">
                            <div>
                                <h3 class="block-title">各指标恢复时长明细</h3>
                                <p class="block-desc">6 项 EEG 指标分别的恢复达标时间，综合恢复时长取最慢指标。</p>
                            </div>
                        </div>
                        <div class="chart-container chart-small">
                            <canvas id="chart-recovery-bar"></canvas>
                        </div>
                    </div>

                    <!-- 衰减幅度热图 -->
                    <div class="chart-block">
                        <div class="block-header">
                            <div>
                                <h3 class="block-title">切换期衰减幅度</h3>
                                <p class="block-desc">心流核心指标取跌落幅度（正值=衰减），认知损耗指标取升高幅度。</p>
                            </div>
                        </div>
                        <div class="heatmap" id="attenuation-heatmap"></div>
                    </div>

                    <!-- 跨条件统计 -->
                    <div class="chart-block" id="stats-block">
                        <div class="block-header">
                            <div>
                                <h3 class="block-title">跨条件统计比较</h3>
                                <p class="block-desc">配对 t 检验（对照 vs 实验组）与重复测量 ANOVA。需至少 2 个条件。</p>
                            </div>
                        </div>
                        <div class="stats-charts">
                            <div class="chart-container chart-small">
                                <canvas id="chart-stats-recovery"></canvas>
                            </div>
                            <div class="chart-container chart-small">
                                <canvas id="chart-stats-attenuation"></canvas>
                            </div>
                        </div>
                        <div class="stats-table-wrap" id="stats-table-wrap"></div>
                    </div>

                </div>

            </section>

        </div>

        <!-- 加载状态 -->
        <div class="loading-overlay" id="loading" style="display:none;">
            <div class="loading-card">
                <div class="loading-spinner"></div>
                <span class="loading-text">正在分析...</span>
            </div>
        </div>

    </main>

</div>

<script src="/static/js/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 验证 HTML 结构完整**

Run: `grep -c 'id="' /Users/xiatian/Desktop/EEG-Science/app/static/index.html`
Expected: 至少 20 个带 ID 的元素（包含所有 JS 需要的 ID）

- [ ] **Step 3: 验证关键 ID 存在**

```bash
cd /Users/xiatian/Desktop/EEG-Science
for id in btn-run-sample btn-upload btn-refresh-stats btn-report-full btn-report-single file-eeg file-events upload-condition param-hp param-lp param-notch param-artifact param-window param-overlap param-tolerance param-recovery-win metric-recovery metric-artifact metric-duration metric-samples chart-timeseries chart-recovery-bar attenuation-heatmap chart-stats-recovery chart-stats-attenuation stats-table-wrap indicator-legend loading result-empty result-content result-condition-badge; do
    grep -q "id=\"$id\"" app/static/index.html && echo "✓ $id" || echo "✗ MISSING: $id"
done
```
Expected: 所有 ID 都显示 ✓

- [ ] **Step 4: Commit**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git add app/static/index.html
git commit -m "feat: 重构 HTML 为侧边导航 + 左右分栏布局"
```

---

## Task 9: 重构 CSS 匹配新布局

**Files:**
- Modify: `/Users/xiatian/Desktop/EEG-Science/app/static/css/style.css` (完全重写)

**设计令牌（延续 TraeWork）：**
- Brand: #4B3FE3
- Invert: #262626 (主按钮)
- Surface: #FAFAFA (侧边栏背景), #FFFFFF (主区)
- Border: rgba(115,115,115,0.12/0.18/0.36)
- 文字: #262626 / #404040 / #737373 / #A1A1A1
- 圆角: 4/6/8/12px
- 字体: SF Pro Text / SF Pro / Inter / JetBrains Mono
- tabular-nums, letter-spacing -0.02em
- Viz 色板: #4B3FE3 #1DC981 #22A5F7 #F87454 #EDAA45 #B655FC

- [ ] **Step 1: 重写 style.css**

文件路径：`/Users/xiatian/Desktop/EEG-Science/app/static/css/style.css`

```css
/* ==========================================================
   EEGDataScience — 视觉样式
   设计系统: TraeWork (紧凑浅色)
   作者: NoWint (https://github.com/NoWint)
   ========================================================== */

/* ---------- 设计令牌 ---------- */
:root {
    /* 品牌色 */
    --brand: #4B3FE3;
    --brand-hover: #3A2FC9;
    --brand-bg: rgba(75, 63, 227, 0.08);
    --brand-bg-hover: rgba(75, 63, 227, 0.12);

    /* 反色 (主按钮) */
    --invert: #262626;
    --invert-hover: #000000;

    /* 表面色 */
    --surface-sidebar: #FAFAFA;
    --surface-main: #FFFFFF;
    --surface-elevated: #FFFFFF;
    --surface-sunken: #F5F5F5;

    /* 文字色 */
    --text-primary: #262626;
    --text-secondary: #404040;
    --text-tertiary: #737373;
    --text-quaternary: #A1A1A1;
    --text-on-brand: #FFFFFF;

    /* 边框色 */
    --border-l0: rgba(115, 115, 115, 0.08);
    --border-l1: rgba(115, 115, 115, 0.12);
    --border-l2: rgba(115, 115, 115, 0.18);
    --border-l3: rgba(115, 115, 115, 0.36);

    /* 状态色 */
    --success: #1DC981;
    --warning: #EDAA45;
    --error: #F87454;

    /* 数据可视化色板 */
    --viz-1: #4B3FE3;
    --viz-2: #1DC981;
    --viz-3: #22A5F7;
    --viz-4: #F87454;
    --viz-5: #EDAA45;
    --viz-6: #B655FC;

    /* 圆角 */
    --radius-sm: 4px;
    --radius-md: 6px;
    --radius-lg: 8px;
    --radius-xl: 12px;

    /* 间距 */
    --space-1: 4px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-5: 20px;
    --space-6: 24px;
    --space-8: 32px;
    --space-10: 40px;

    /* 字体 */
    --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro", "Inter", "PingFang SC", "Helvetica Neue", sans-serif;
    --font-mono: "JetBrains Mono", "SF Mono", Menlo, Monaco, Consolas, monospace;

    /* 浮动阴影 (仅用于浮动元素) */
    --shadow-float: 0 4px 16px rgba(0, 0, 0, 0.08), 0 1px 4px rgba(0, 0, 0, 0.06);
    --shadow-overlay: 0 8px 32px rgba(0, 0, 0, 0.12), 0 2px 8px rgba(0, 0, 0, 0.08);

    /* 布局 */
    --sidebar-width: 200px;
    --config-width: 260px;
    --header-height: 56px;
}

/* ---------- 全局重置 ---------- */
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html, body {
    height: 100%;
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.5;
    color: var(--text-primary);
    background: var(--surface-main);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
}

button, input, select, textarea {
    font-family: inherit;
    font-size: inherit;
    color: inherit;
}

a {
    color: var(--brand);
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* ---------- 应用骨架 ---------- */
.app-shell {
    display: flex;
    height: 100vh;
    overflow: hidden;
}

/* ==========================================================
   侧边栏
   ========================================================== */
.sidebar {
    width: var(--sidebar-width);
    background: var(--surface-sidebar);
    border-right: 1px solid var(--border-l2);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-4) var(--space-3);
    height: var(--header-height);
    border-bottom: 1px solid var(--border-l1);
}

.brand-mark {
    width: 8px;
    height: 8px;
    background: var(--brand);
    border-radius: 50%;
    flex-shrink: 0;
}

.brand-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.sidebar-nav {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-3);
}

.nav-group {
    margin-bottom: var(--space-5);
}

.nav-group-label {
    font-size: 10px;
    font-weight: 500;
    color: var(--text-quaternary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0 var(--space-2);
    margin-bottom: var(--space-2);
}

.nav-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-md);
    font-size: 13px;
    color: var(--text-secondary);
    cursor: pointer;
    margin-bottom: 2px;
    transition: background 0.15s ease, color 0.15s ease;
    user-select: none;
}

.nav-item:hover {
    background: var(--border-l0);
    color: var(--text-primary);
}

.nav-item.active {
    background: var(--brand-bg);
    color: var(--brand);
    font-weight: 500;
}

.nav-item-disabled {
    color: var(--text-quaternary);
    cursor: not-allowed;
}

.nav-item-disabled:hover {
    background: transparent;
    color: var(--text-quaternary);
}

.nav-item-text {
    flex: 1;
}

.nav-item-tag {
    font-size: 9px;
    color: var(--text-quaternary);
    background: var(--border-l1);
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    font-weight: 500;
    letter-spacing: 0.02em;
}

.sidebar-footer {
    padding: var(--space-3);
    border-top: 1px solid var(--border-l1);
}

.footer-author {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-1) var(--space-2);
}

.footer-label {
    font-size: 11px;
    color: var(--text-quaternary);
}

.footer-link {
    font-size: 12px;
    color: var(--text-secondary);
    font-weight: 500;
}

.footer-link:hover {
    color: var(--brand);
    text-decoration: none;
}

/* ==========================================================
   主内容区
   ========================================================== */
.main-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--surface-main);
}

/* ---------- 模块标题栏 ---------- */
.module-header {
    height: var(--header-height);
    padding: 0 var(--space-6);
    border-bottom: 1px solid var(--border-l1);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
}

.module-title-group {
    display: flex;
    flex-direction: column;
    gap: 1px;
}

.module-breadcrumb {
    font-size: 11px;
    color: var(--text-tertiary);
    letter-spacing: 0.02em;
}

.module-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.module-actions {
    display: flex;
    gap: var(--space-2);
}

/* ==========================================================
   工作区 (左右分栏)
   ========================================================== */
.workspace {
    flex: 1;
    display: flex;
    overflow: hidden;
}

/* ---------- 配置面板 ---------- */
.config-panel {
    width: var(--config-width);
    background: var(--surface-main);
    border-right: 1px solid var(--border-l2);
    overflow-y: auto;
    padding: var(--space-4);
    flex-shrink: 0;
}

.config-section {
    margin-bottom: var(--space-5);
}

.config-section-label {
    font-size: 10px;
    font-weight: 500;
    color: var(--text-quaternary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: var(--space-2);
}

/* 数据来源标签页 */
.source-tabs {
    display: flex;
    gap: 2px;
    margin-bottom: var(--space-3);
    background: var(--surface-sunken);
    padding: 2px;
    border-radius: var(--radius-md);
}

.tab-btn {
    flex: 1;
    padding: var(--space-2) var(--space-3);
    border: none;
    background: transparent;
    color: var(--text-tertiary);
    font-size: 12px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.15s ease;
}

.tab-btn:hover {
    color: var(--text-primary);
}

.tab-btn.active {
    background: var(--surface-main);
    color: var(--text-primary);
    font-weight: 500;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

.tab-panel {
    display: none;
}

.tab-panel.active {
    display: block;
}

/* 条件卡片列表 */
.condition-list {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
}

.cond-card {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--border-l2);
    background: var(--surface-main);
    border-radius: var(--radius-md);
    cursor: pointer;
    text-align: left;
    transition: all 0.15s ease;
}

.cond-card:hover {
    border-color: var(--border-l3);
}

.cond-card.active {
    border-color: var(--brand);
    background: var(--brand-bg);
}

.cond-tag {
    font-size: 10px;
    font-weight: 500;
    color: var(--text-tertiary);
    background: var(--border-l1);
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    flex-shrink: 0;
}

.cond-card.active .cond-tag {
    background: var(--brand);
    color: var(--text-on-brand);
}

.cond-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    flex-shrink: 0;
}

.cond-card.active .cond-name {
    color: var(--brand);
}

.cond-desc {
    font-size: 11px;
    color: var(--text-tertiary);
    margin-left: auto;
}

/* 上传区域 */
.upload-area {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
}

.upload-slot {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
}

.upload-label {
    font-size: 11px;
    color: var(--text-tertiary);
    font-weight: 500;
}

.upload-input {
    font-size: 12px;
    padding: var(--space-2);
    border: 1px solid var(--border-l2);
    border-radius: var(--radius-md);
    background: var(--surface-main);
    cursor: pointer;
}

.upload-input::-webkit-file-upload-button {
    background: var(--surface-sunken);
    border: none;
    padding: 4px 8px;
    border-radius: var(--radius-sm);
    margin-right: var(--space-2);
    cursor: pointer;
    font-size: 11px;
}

.upload-hint {
    font-size: 11px;
    color: var(--text-quaternary);
}

.text-input {
    font-size: 12px;
    padding: var(--space-2) var(--space-3);
    border: 1px solid var(--border-l2);
    border-radius: var(--radius-md);
    background: var(--surface-main);
    outline: none;
    transition: border-color 0.15s ease;
}

.text-input:focus {
    border-color: var(--brand);
}

/* 参数列表 */
.param-list {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
}

.param-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
}

.param-item label {
    font-size: 12px;
    color: var(--text-secondary);
    flex-shrink: 0;
}

.param-item input {
    width: 80px;
    font-size: 12px;
    padding: var(--space-1) var(--space-2);
    border: 1px solid var(--border-l2);
    border-radius: var(--radius-sm);
    background: var(--surface-main);
    text-align: right;
    font-variant-numeric: tabular-nums;
    outline: none;
    transition: border-color 0.15s ease;
}

.param-item input:focus {
    border-color: var(--brand);
}

/* 配置区操作按钮 */
.config-actions {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding-top: var(--space-3);
    border-top: 1px solid var(--border-l1);
}

/* ==========================================================
   按钮
   ========================================================== */
.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-4);
    border: none;
    border-radius: var(--radius-md);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
    letter-spacing: -0.01em;
}

.btn-primary {
    background: var(--invert);
    color: var(--text-on-brand);
}

.btn-primary:hover {
    background: var(--invert-hover);
}

.btn-secondary {
    background: var(--surface-main);
    color: var(--text-primary);
    border: 1px solid var(--border-l2);
}

.btn-secondary:hover {
    border-color: var(--border-l3);
    background: var(--surface-sunken);
}

.btn-ghost {
    background: transparent;
    color: var(--text-tertiary);
}

.btn-ghost:hover {
    color: var(--text-primary);
    background: var(--border-l0);
}

.btn-block {
    width: 100%;
}

/* ==========================================================
   结果区
   ========================================================== */
.result-area {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-5);
}

/* 空状态 */
.result-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    color: var(--text-quaternary);
}

.empty-icon {
    width: 48px;
    height: 48px;
    border: 2px solid var(--border-l2);
    border-radius: var(--radius-xl);
    margin-bottom: var(--space-4);
}

.empty-text {
    font-size: 14px;
    color: var(--text-tertiary);
    margin-bottom: var(--space-1);
}

.empty-hint {
    font-size: 12px;
    color: var(--text-quaternary);
}

/* 结果内容 */
.result-content {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
}

.result-header {
    display: flex;
    align-items: center;
    gap: var(--space-3);
}

.result-badge {
    font-size: 12px;
    font-weight: 500;
    color: var(--brand);
    background: var(--brand-bg);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius-md);
}

/* 指标卡片 */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-3);
}

.metric-card {
    border: 1px solid var(--border-l2);
    background: var(--surface-elevated);
    border-radius: var(--radius-lg);
    padding: var(--space-3) var(--space-4);
}

.metric-label {
    font-size: 11px;
    color: var(--text-tertiary);
    display: block;
    margin-bottom: var(--space-1);
}

.metric-value-row {
    display: flex;
    align-items: baseline;
    gap: var(--space-1);
}

.metric-value {
    font-size: 22px;
    font-weight: 600;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}

.metric-unit {
    font-size: 12px;
    color: var(--text-tertiary);
}

/* ==========================================================
   图表区块
   ========================================================== */
.chart-block {
    border: 1px solid var(--border-l1);
    background: var(--surface-elevated);
    border-radius: var(--radius-lg);
    padding: var(--space-4);
}

.block-header {
    margin-bottom: var(--space-3);
}

.block-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 2px;
    letter-spacing: -0.01em;
}

.block-desc {
    font-size: 12px;
    color: var(--text-tertiary);
}

.chart-container {
    position: relative;
    height: 320px;
}

.chart-container.chart-small {
    height: 220px;
}

.chart-container canvas {
    width: 100% !important;
    height: 100% !important;
}

/* 图例 */
.legend-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-3);
    margin-top: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--border-l1);
}

.legend-item {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: 12px;
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
}

.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
}

.legend-item.disabled {
    opacity: 0.4;
}

/* 衰减热图 */
.heatmap {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
}

.heatmap-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
}

.heatmap-label {
    width: 140px;
    font-size: 12px;
    color: var(--text-secondary);
    flex-shrink: 0;
}

.heatmap-bar-container {
    flex: 1;
    height: 24px;
    background: var(--surface-sunken);
    border-radius: var(--radius-sm);
    position: relative;
    overflow: hidden;
}

.heatmap-bar {
    height: 100%;
    border-radius: var(--radius-sm);
    transition: width 0.3s ease;
}

.heatmap-value {
    font-size: 11px;
    color: var(--text-tertiary);
    font-variant-numeric: tabular-nums;
    width: 56px;
    text-align: right;
    flex-shrink: 0;
}

/* 统计图表 */
.stats-charts {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-3);
    margin-bottom: var(--space-3);
}

/* 统计表 */
.stats-table-wrap {
    overflow-x: auto;
}

.stats-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}

.stats-table th {
    text-align: left;
    padding: var(--space-2) var(--space-3);
    color: var(--text-tertiary);
    font-weight: 500;
    border-bottom: 1px solid var(--border-l2);
}

.stats-table td {
    padding: var(--space-2) var(--space-3);
    border-bottom: 1px solid var(--border-l1);
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
}

.stats-table tr:last-child td {
    border-bottom: none;
}

/* ==========================================================
   加载状态
   ========================================================== */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

.loading-card {
    background: var(--surface-elevated);
    border: 1px solid var(--border-l2);
    border-radius: var(--radius-lg);
    padding: var(--space-6);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-3);
    box-shadow: var(--shadow-float);
}

.loading-spinner {
    width: 24px;
    height: 24px;
    border: 2px solid var(--border-l2);
    border-top-color: var(--brand);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.loading-text {
    font-size: 13px;
    color: var(--text-secondary);
}

/* ==========================================================
   滚动条
   ========================================================== */
.sidebar-nav::-webkit-scrollbar,
.config-panel::-webkit-scrollbar,
.result-area::-webkit-scrollbar {
    width: 6px;
}

.sidebar-nav::-webkit-scrollbar-track,
.config-panel::-webkit-scrollbar-track,
.result-area::-webkit-scrollbar-track {
    background: transparent;
}

.sidebar-nav::-webkit-scrollbar-thumb,
.config-panel::-webkit-scrollbar-thumb,
.result-area::-webkit-scrollbar-thumb {
    background: var(--border-l2);
    border-radius: 3px;
}

.sidebar-nav::-webkit-scrollbar-thumb:hover,
.config-panel::-webkit-scrollbar-thumb:hover,
.result-area::-webkit-scrollbar-thumb:hover {
    background: var(--border-l3);
}

/* ==========================================================
   响应式 (窄屏降级)
   ========================================================== */
@media (max-width: 1024px) {
    .metric-row {
        grid-template-columns: repeat(2, 1fr);
    }
    .stats-charts {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 768px) {
    .workspace {
        flex-direction: column;
    }
    .config-panel {
        width: 100%;
        border-right: none;
        border-bottom: 1px solid var(--border-l2);
        max-height: 40vh;
    }
}
```

- [ ] **Step 2: 验证 CSS 文件完整**

Run: `wc -l /Users/xiatian/Desktop/EEG-Science/app/static/css/style.css`
Expected: 约 700+ 行

- [ ] **Step 3: 验证关键类名存在**

```bash
cd /Users/xiatian/Desktop/EEG-Science
for cls in app-shell sidebar sidebar-brand sidebar-nav nav-item config-panel workspace result-area metric-card chart-block loading-overlay; do
    grep -q "\.$cls" app/static/css/style.css && echo "✓ .$cls" || echo "✗ MISSING: .$cls"
done
```
Expected: 所有类名都显示 ✓

- [ ] **Step 4: Commit**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git add app/static/css/style.css
git commit -m "feat: 重构 CSS 匹配侧边导航 + 左右分栏布局 (TraeWork 紧凑浅色)"
```

---

## Task 10: 更新 JS 适配新 HTML 结构

**Files:**
- Modify: `/Users/xiatian/Desktop/EEG-Science/app/static/js/app.js` (更新引用新 HTML 结构)

**关键修改点：**
1. 标签页切换：上传/模拟切换时显示对应按钮（btn-run-sample / btn-upload）
2. 结果显示：用 `result-empty` / `result-content` 的 display 切换替代原来的 `step-results` 显示
3. 跨条件统计区块：始终在 DOM 中（不再需要创建 `step-stats`）
4. 侧边栏模块切换：禁用的模块点击时提示"即将推出"
5. 图表颜色已使用 TraeWork 色板（无需改动）

- [ ] **Step 1: 更新初始化函数**

在 `/Users/xiatian/Desktop/EEG-Science/app/static/js/app.js` 中，将第 30-35 行的初始化函数：

```javascript
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initConditionCards();
    initButtons();
    initFileInputs();
});
```

改为：

```javascript
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initConditionCards();
    initButtons();
    initFileInputs();
    initSidebarNav();
});
```

- [ ] **Step 2: 更新 initTabs 函数,添加按钮切换逻辑**

将第 38-48 行的 `initTabs` 函数替换为：

```javascript
// ---------- 标签页切换 ----------
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`panel-${tab}`).classList.add('active');
            // 切换操作按钮显示
            document.getElementById('btn-run-sample').style.display = tab === 'sample' ? '' : 'none';
            document.getElementById('btn-upload').style.display = tab === 'upload' ? '' : 'none';
        });
    });
}
```

- [ ] **Step 3: 添加 initSidebarNav 函数**

在 `initFileInputs` 函数之后（约第 80 行后）添加：

```javascript
// ---------- 侧边栏导航 ----------
function initSidebarNav() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            if (item.classList.contains('nav-item-disabled')) {
                // 禁用模块提示
                const name = item.querySelector('.nav-item-text')?.textContent || '该模块';
                showToast(`${name}即将推出`);
                return;
            }
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            item.classList.add('active');
        });
    });
}

// ---------- 轻量提示 ----------
function showToast(msg) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = msg;
    toast.style.cssText = `
        position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
        background: var(--invert); color: #fff; padding: 8px 16px;
        border-radius: 6px; font-size: 13px; z-index: 2000;
        opacity: 0; transition: opacity 0.2s;
    `;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.style.opacity = '1');
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 200);
    }, 2000);
}
```

- [ ] **Step 4: 更新 runSample 函数,用结果区切换替代步骤显示**

将第 101-121 行的 `runSample` 函数中的结果显示部分：

```javascript
        analyzedConditions.add(selectedCondition);
        renderResults(data);
        document.getElementById('step-results').style.display = 'block';
        document.getElementById('step-stats').style.display = 'block';
        document.getElementById('step-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
```

改为：

```javascript
        analyzedConditions.add(selectedCondition);
        renderResults(data);
        document.getElementById('result-empty').style.display = 'none';
        document.getElementById('result-content').style.display = 'flex';
        document.getElementById('stats-block').style.display = 'block';
```

- [ ] **Step 5: 更新 uploadAndAnalyze 函数,同样的显示切换**

将第 155-159 行的 `uploadAndAnalyze` 函数中的对应部分：

```javascript
        analyzedConditions.add(condition);
        renderResults(data);
        document.getElementById('step-results').style.display = 'block';
        document.getElementById('step-stats').style.display = 'block';
        document.getElementById('step-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
```

改为：

```javascript
        analyzedConditions.add(condition);
        renderResults(data);
        document.getElementById('result-empty').style.display = 'none';
        document.getElementById('result-content').style.display = 'flex';
        document.getElementById('stats-block').style.display = 'block';
```

- [ ] **Step 6: 验证无残留的旧 ID 引用**

```bash
cd /Users/xiatian/Desktop/EEG-Science
grep -n "step-results\|step-stats\|step-data\|step-config" app/static/js/app.js || echo "✓ 无旧 ID 引用"
```
Expected: 输出 `✓ 无旧 ID 引用`

- [ ] **Step 7: 启动服务验证页面加载**

```bash
cd /Users/xiatian/Desktop/EEG-Science
source venv/bin/activate
python app/server.py &
sleep 3
curl -s -o /dev/null -w "HTML: %{http_code}\n" http://localhost:18765/
curl -s -o /dev/null -w "JS: %{http_code}\n" http://localhost:18765/static/js/app.js
kill %1 2>/dev/null
```
Expected: HTML 和 JS 都返回 200

- [ ] **Step 8: Commit**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git add app/static/js/app.js
git commit -m "feat: 更新 JS 适配侧边导航 + 左右分栏 HTML 结构"
```

---

## Task 11: 添加 Toast 样式到 CSS

**Files:**
- Modify: `/Users/xiatian/Desktop/EEG-Science/app/static/css/style.css` (追加 toast 样式)

- [ ] **Step 1: 在 CSS 文件末尾追加 toast 样式**

在 `/Users/xiatian/Desktop/EEG-Science/app/static/css/style.css` 末尾，响应式部分之前，插入：

```css

/* ==========================================================
   Toast 提示
   ========================================================== */
.toast {
    position: fixed;
    top: var(--space-5);
    left: 50%;
    transform: translateX(-50%);
    background: var(--invert);
    color: var(--text-on-brand);
    padding: var(--space-2) var(--space-4);
    border-radius: var(--radius-md);
    font-size: 13px;
    z-index: 2000;
    box-shadow: var(--shadow-float);
    opacity: 0;
    transition: opacity 0.2s ease;
}

.toast.show {
    opacity: 1;
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git add app/static/css/style.css
git commit -m "style: 添加 toast 提示样式"
```

---

## Task 12: 端到端验证 + 清理旧目录

**Files:**
- Delete: `/Users/xiatian/Desktop/EEG-Science/toolbox/` (整个旧目录)

- [ ] **Step 1: 启动服务**

```bash
cd /Users/xiatian/Desktop/EEG-Science
source venv/bin/activate
python app/server.py &
sleep 3
```
Expected: 服务启动成功

- [ ] **Step 2: 验证健康检查**

```bash
curl -s http://localhost:18765/api/health | python3 -m json.tool
```
Expected: 返回 `{"status": "ok", "service": "EEG Flow Recovery Analyzer"}`

- [ ] **Step 3: 验证样例数据分析**

```bash
curl -s -X POST http://localhost:18765/api/sample \
  -H "Content-Type: application/json" \
  -d '{"condition":"AtoB","fs":250}' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Condition: {data.get('condition')}\")
print(f\"Recovery time: {data.get('recovery_time')}\")
print(f\"Artifact ratio: {data.get('artifact_ratio')}\")
print(f\"Duration: {data.get('duration_sec', 0) / 60:.1f} min\")
print(f\"Indicators: {list(data.get('baseline_means', {}).keys())}\")
print('✓ Sample analysis OK')
"
```
Expected: 输出各项指标值，最后显示 `✓ Sample analysis OK`

- [ ] **Step 4: 验证跨条件统计**

```bash
# 运行多个条件
curl -s -X POST http://localhost:18765/api/sample -H "Content-Type: application/json" -d '{"condition":"AtoA","fs":250}' > /dev/null
curl -s -X POST http://localhost:18765/api/sample -H "Content-Type: application/json" -d '{"condition":"AtoC","fs":250}' > /dev/null
curl -s -X POST http://localhost:18765/api/sample -H "Content-Type: application/json" -d '{"condition":"BtoC","fs":250}' > /dev/null

# 获取统计
curl -s http://localhost:18765/api/stats | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Conditions: {data.get('conditions')}\")
print(f\"Recovery times: {data.get('recovery_times')}\")
print(f\"Paired t-tests: {list(data.get('paired_t_tests', {}).keys())}\")
print('✓ Stats OK')
"
```
Expected: 显示 4 个条件的统计结果

- [ ] **Step 5: 验证报告生成**

```bash
curl -s http://localhost:18765/api/report | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Title: {data.get('title')}\")
print(f\"Sections: {len(data.get('sections', []))}\")
print(f\"Analyzed conditions: {data.get('analyzed_conditions')}\")
print('✓ Report OK')
"
```
Expected: 报告包含多个 section

- [ ] **Step 6: 停止服务**

```bash
kill %1 2>/dev/null
```

- [ ] **Step 7: 删除旧 toolbox 目录**

```bash
cd /Users/xiatian/Desktop/EEG-Science
rm -rf toolbox/
```

- [ ] **Step 8: 验证旧目录已删除**

Run: `ls /Users/xiatian/Desktop/EEG-Science/toolbox/ 2>&1`
Expected: "No such file or directory"

- [ ] **Step 9: 最终启动验证（模拟用户双击）**

```bash
cd /Users/xiatian/Desktop/EEG-Science
source venv/bin/activate
python app/server.py &
sleep 3
curl -s http://localhost:18765/api/health && echo ""
curl -s -o /dev/null -w "Page: %{http_code}\n" http://localhost:18765/
kill %1 2>/dev/null
```
Expected: health 返回 ok，页面 200

- [ ] **Step 10: Commit**

```bash
cd /Users/xiatian/Desktop/EEG-Science
git add -A
git commit -m "chore: 删除旧 toolbox 目录, 完成项目重组"
```

---

## 自审记录

**Spec 覆盖检查：**
- ✓ 项目结构重组（app/ 包结构）— Task 1-4
- ✓ requirements.txt — Task 1
- ✓ venv 虚拟环境 — Task 5, 7
- ✓ macOS 启动脚本（setup.command, start.command）— Task 5
- ✓ Windows 启动脚本（setup.bat, start.bat）— Task 6
- ✓ 代码迁移（server.py, analysis.py）— Task 2-3
- ✓ 导入路径更新 — Task 3
- ✓ 静态文件路径更新 — Task 3-4
- ✓ UI 重构：侧边导航 + 左右分栏 — Task 8-11
- ✓ 视觉风格：紧凑浅色 TraeWork — Task 9
- ✓ 作者 GitHub footer — Task 8
- ✓ 端口策略 18765 — Task 3, 5, 6
- ✓ 删除旧 toolbox/ — Task 12

**Placeholder 检查：** 无 TBD/TODO/"implement later" 等占位符。

**类型一致性检查：** HTML 中的 ID 与 JS 中的 getElementById 引用一致（Task 8 Step 3 已验证）。
