# EEGDataScience 阶段 1 设计文档

## 项目概述

**项目名称**: EEGDataScience
**作者 GitHub**: [NoWint](https://github.com/NoWint)
**阶段**: 阶段 1 — 项目规范化 + 一键启动 + UI 重构
**目标**: 将现有 EEG 心流恢复分析工具箱重组为标准 Python 项目，提供双击启动能力（macOS + Windows），重构 UI 为侧边导航 + 左右分栏布局，为未来"完整科研平台"奠定结构基础。

## 背景

当前项目位于 `/Users/xiatian/Desktop/EEG-Science/toolbox/`，包含：
- `analysis.py` — EEG 心流恢复分析核心模块
- `server.py` — FastAPI 服务端（端口 18765）
- `static/` — 前端文件（HTML + CSS + JS，已对齐 TraeWork 设计系统）

现有问题：
- 无 requirements.txt，依赖全局安装
- 无虚拟环境，污染系统 Python
- 无启动脚本，需手动 `python server.py`
- 目录结构不利于多模块扩展
- 项目名未统一（当前为 EEG-Science，目标为 EEGDataScience）
- UI 为垂直步骤式，不利于未来多模块扩展

## 平台目标

- macOS（.command 启动脚本）
- Windows（.bat 启动脚本）
- Python 3.11+

## 架构决策

采用**单体 FastAPI 应用**架构，各分析模块作为 Python 包组织在 `app/analysis/` 下。未来扩展新分析模块时，只需在该目录下新增文件并注册路由。

## UI 布局决策

通过可视化伴侣探索后确定以下布局（见 `.superpowers/brainstorm/` 中的 mockup）：

### 整体布局：侧边导航 + 主内容区
- 左侧固定侧边栏切换模块
- 右侧主内容区展示当前模块内容
- 扩展性强，适合多模块科研平台

### 侧边栏结构：分组式
- 分三大组：「分析模块」「数据」「其他」
- 分析模块组：心流恢复分析、频谱分析、事件相关电位、脑连接分析（未来扩展）
- 数据组：被试管理、实验记录、数据归档（未来扩展）
- 其他组：设置
- 层级清晰，适合未来扩展到 5+ 模块

### 主内容区：左右分栏（配置 + 结果）
- 左侧配置面板固定（约 240px）：数据来源选择、分析参数、运行按钮
- 右侧结果区实时展示：指标卡片、时序图表、恢复明细、衰减热图
- 配置即时影响结果，适合反复调参探索

### 视觉风格：紧凑浅色（TraeWork 延续）
- 侧边栏 200px、浅灰背景 #FAFAFA
- 主区纯白 #FFFFFF
- 延续现有 TraeWork 设计令牌（Brand #4B3FE3、Invert #262626、border-based layering）
- 整体克制专业，信息密度高

## 项目结构

```
EEGDataScience/
├── requirements.txt
├── start.command              # macOS 双击启动
├── start.bat                  # Windows 双击启动
├── setup.command              # macOS 首次配置
├── setup.bat                  # Windows 首次配置
├── app/
│   ├── __init__.py
│   ├── server.py              # FastAPI 入口（从 toolbox/server.py 迁移）
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── flow_recovery.py   # 心流恢复分析（从 toolbox/analysis.py 迁移）
│   └── static/
│       ├── index.html
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
├── data/                      # 用户数据目录
│   └── uploads/
└── venv/                      # 首次配置自动创建（不纳入版本控制）
```

## 依赖管理

### requirements.txt
```
fastapi>=0.135.0
uvicorn>=0.42.0
numpy>=2.0.0
pandas>=2.0.0
scipy>=1.17.0
```

### 虚拟环境
- 首次配置脚本自动创建 `venv/` 目录
- 所有依赖安装在 venv 内，不污染系统 Python
- venv 目录不纳入版本控制

## 启动脚本设计

### 行为规格
双击启动脚本后：
1. 检查 `venv/` 是否存在，不存在则提示先运行 setup 脚本
2. 激活 venv
3. 后台启动 uvicorn 服务（端口 18765）
4. 轮询 `/api/health` 接口，确认服务就绪（最多等待 10 秒）
5. 打开默认浏览器访问 `http://localhost:18765`
6. 脚本退出，服务在后台持续运行

### macOS: `start.command`
```bash
#!/bin/bash
cd "$(dirname "$0")"
# 检查 venv
if [ ! -d "venv" ]; then
    echo "首次使用请先双击 setup.command 配置环境"
    read -p "按回车键退出..."
    exit 1
fi
source venv/bin/activate
# 后台启动服务
nohup python app/server.py > /dev/null 2>&1 &
# 等待服务就绪
for i in $(seq 1 10); do
    if curl -s http://localhost:18765/api/health > /dev/null 2>&1; then
        open http://localhost:18765
        exit 0
    fi
    sleep 1
done
echo "服务启动失败，请检查"
read -p "按回车键退出..."
```

### Windows: `start.bat`
```bat
@echo off
cd /d "%~dp0"
if not exist "venv" (
    echo 首次使用请先双击 setup.bat 配置环境
    pause
    exit /b 1
)
call venv\Scripts\activate
start /b pythonw app\server.py
:: 等待服务就绪
:waitloop
timeout /t 1 /nobreak > nul
curl -s http://localhost:18765/api/health > nul 2>&1
if errorlevel 1 (
    set /a count+=1
    if %count% lss 10 goto waitloop
    echo 服务启动失败，请检查
    pause
    exit /b 1
)
start http://localhost:18765
```

## 首次配置脚本

### macOS: `setup.command`
1. 检查 Python 3 是否安装
2. `python3 -m venv venv`
3. `source venv/bin/activate`
4. `pip install -r requirements.txt`
5. 提示完成

### Windows: `setup.bat`
1. 检查 Python 是否安装
2. `python -m venv venv`
3. `call venv\Scripts\activate`
4. `pip install -r requirements.txt`
5. 提示完成

## 代码迁移

### 目录重命名
将项目根目录从 `EEG-Science` 重命名为 `EEGDataScience`（或在目标位置创建新目录）。迁移完成后删除旧 `toolbox/` 目录。

### 导入路径更新
`server.py` 中的导入从：
```python
from analysis import (...)
```
改为：
```python
from app.analysis.flow_recovery import (...)
```

### 路径更新
`server.py` 中的 `BASE_DIR` 和 `STATIC_DIR` 路径需适配新目录结构：
```python
BASE_DIR = Path(__file__).parent  # app/
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR.parent / "data" / "uploads"  # 项目根/data/uploads
```

## 端口策略
- 固定端口 18765（已在现有版本验证无冲突）
- 启动时检测端口占用，若被占用则提示用户

## 不在阶段 1 范围内
- 频谱分析、事件相关电位、脑连接分析等新分析模块的实际实现（阶段 3，侧边栏仅占位）
- 被试管理、实验记录、数据归档的实际实现（阶段 2，侧边栏仅占位）
- 数据库存储（阶段 2）

## 成功标准
1. 双击 `start.command`（Mac）或 `start.bat`（Windows）可自动启动服务并打开浏览器
2. 首次使用双击 `setup.command` / `setup.bat` 可自动创建 venv 并安装依赖
3. UI 重构为侧边导航 + 左右分栏布局，视觉风格延续 TraeWork 紧凑浅色
4. 现有所有功能（分析、图表、报告生成）在新 UI 下正常工作
5. 项目结构清晰，`analysis/` 下可方便地新增分析模块
6. 文档 footer 显示作者 GitHub: NoWint
