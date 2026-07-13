# EEGDataScience 完整技术文档

> **作者**: NoWint ([https://github.com/NoWint](https://github.com/NoWint))
> **仓库**: [https://github.com/NoWint/EGGDataScience](https://github.com/NoWint/EGGDataScience)
> **版本**: 实时采集子项目交付版 (commit `4381bb8`)
> **更新日期**: 2026-07-13

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [目录结构](#3-目录结构)
4. [数据格式兼容](#4-数据格式兼容)
5. [分析模块详解](#5-分析模块详解)
6. [实时采集模块](#6-实时采集模块)
7. [REST API 参考](#7-rest-api-参考)
8. [WebSocket API](#8-websocket-api)
9. [前端架构](#9-前端架构)
10. [测试体系](#10-测试体系)
11. [部署与运行](#11-部署与运行)
12. [开发指南](#12-开发指南)
13. [故障排查](#13-故障排查)

---

## 1. 项目概述

### 1.1 定位

EEGDataScience 是一个面向 EEG（脑电图）心流恢复研究的端到端分析平台，集成：

- **离线分析**：心流恢复时序分析、频谱、ERP、ERSP 时频、头皮地形图、Focus 专注度
- **实时采集**：基于 BrainFlow 的多板卡实时数据采集与流式分析
- **数据兼容**：自动识别 OpenBCI GUI 导出（ODF）、BrainFlow CSV、普通 CSV 三种格式
- **可视化前端**：TraeWork 设计系统，侧边栏 + 工作区布局

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| 多格式自动识别 | 上传文件后自动判断 ODF / BrainFlow CSV / 普通 CSV，无需手动指定 |
| 心流恢复分析 | 基于 6 项指标（Theta/Alpha、Alpha/Beta/Gamma 能量、熵、认知负载）计算恢复时间 |
| 模块借鉴 | 从 OpenBCI GUI 借鉴频谱、地形图、Focus 模块，集成到统一管线 |
| 实时采集 | BrainFlow BoardShim 封装，支持 Synthetic / Cyton / Daisy / Ganglion |
| WebSocket 推送 | 后台轮询线程 + asyncio.Queue 桥接，低延迟数据流 |
| 滚动波形 | 前端 Canvas 5 秒滚动窗口，自动增益 |

### 1.3 技术栈

- **后端**: Python 3.11 + FastAPI + Uvicorn
- **实时**: BrainFlow 4.x + WebSocket + threading
- **分析**: NumPy + SciPy + scikit-learn
- **前端**: 原生 HTML/CSS/JS + Chart.js 4.4
- **测试**: pytest + FastAPI TestClient
- **设计**: TraeWork 紧凑浅色系统

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                       浏览器前端                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ 心流恢复分析 │  │ 频谱/地形图   │  │   实时采集界面      │ │
│  │  (Chart.js)  │  │  Focus 模块  │  │  (Canvas + WS)     │ │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬──────────┘ │
│         │                 │                    │            │
│         └──────── HTTP ───┴──── WebSocket ─────┘            │
└─────────┼─────────────────┼────────────────────┼────────────┘
          │                 │                    │
┌─────────▼─────────────────▼────────────────────▼────────────┐
│                    FastAPI Server (app/server.py)            │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ /api/    │  │ /api/analyze │  │ /ws/realtime           ││
│  │ upload   │  │ /api/sample  │  │ (WebSocket endpoint)   ││
│  └────┬─────┘  └──────┬───────┘  └───────────┬────────────┘│
│       │               │                      │             │
│  ┌────▼─────┐  ┌──────▼──────────┐  ┌────────▼──────────┐  │
│  │ load_eeg │  │ run_full_       │  │ AcquisitionManager│  │
│  │ _full()  │  │ pipeline()      │  │ (单例 + 后台线程)  │  │
│  └────┬─────┘  └──────┬──────────┘  └────────┬──────────┘  │
│       │               │                      │             │
│  ┌────▼─────┐  ┌──────▼──────────┐  ┌────────▼──────────┐  │
│  │openbci_  │  │ spectrum/erp/   │  │ BrainFlowAcquire  │  │
│  │import.py │  │ ersp/topomap/   │  │ (BoardShim 封装)  │  │
│  │          │  │ focus/stats     │  │                   │  │
│  └──────────┘  └─────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
          │                                          │
          ▼                                          ▼
   ┌──────────────┐                         ┌─────────────────┐
   │  CSV / ODF   │                         │  BrainFlow SDK  │
   │  本地文件    │                         │  (Synthetic/    │
   └──────────────┘                         │   Cyton/Ganglion)│
                                             └─────────────────┘
```

### 2.2 请求流转

**离线分析流程**:
1. 前端上传 EEG 文件 → `/api/upload`
2. `load_eeg_full()` 自动识别格式（ODF / BrainFlow CSV / 普通 CSV）
3. 前端调用 `/api/analyze` → `run_full_pipeline()`
4. 管线依次执行：滤波 → 窗口化 → 指标计算 → 恢复时间 → 频谱 → 地形图 → Focus
5. 返回 JSON 结果，前端 Chart.js 渲染

**实时采集流程**:
1. 前端 POST `/api/realtime/start` → `AcquisitionManager.start()`
2. Manager 创建 `BrainFlowAcquisition`，执行 prepare → start_stream
3. 启动 daemon 轮询线程（50ms 间隔）
4. 轮询线程调用 `get_board_data()` → 更新环形缓冲 → 推送回调
5. WebSocket 端点通过 `asyncio.Queue` 桥接同步回调到异步协程
6. 前端 WS 客户端接收数据帧 → Canvas 滚动绘制

---

## 3. 目录结构

```
EEG-Science/
├── app/
│   ├── server.py                  # FastAPI 入口 + WebSocket 挂载
│   ├── analysis/                  # 分析模块
│   │   ├── openbci_import.py      # OpenBCI/BrainFlow CSV 导入
│   │   ├── load_eeg_full.py       # 统一加载入口(自动识别格式)
│   │   ├── flow_recovery.py       # 心流恢复核心分析
│   │   ├── spectrum.py            # 频谱分析(FFT + 频带功率)
│   │   ├── erp.py                 # 事件相关电位
│   │   ├── ersp.py                # ERSP 时频分析
│   │   ├── topomap.py             # 头皮地形图(RBF 插值)
│   │   ├── focus.py               # Focus 专注度计算
│   │   ├── artifact.py            # 伪迹检测
│   │   └── stats_rigor.py         # 统计严谨性检验
│   ├── realtime/                  # 实时采集模块
│   │   ├── __init__.py
│   │   ├── acquisition.py         # BrainFlowAcquisition 封装
│   │   └── manager.py             # AcquisitionManager 单例
│   ├── routers/                   # FastAPI 路由
│   │   ├── spectrum.py            # 频谱相关端点
│   │   └── realtime.py            # 实时采集 REST + WS
│   └── static/                    # 前端静态资源
│       ├── index.html             # 主页面
│       ├── css/style.css          # TraeWork 样式
│       └── js/
│           ├── app.js             # 主交互逻辑
│           └── realtime.js        # 实时采集前端
├── tests/                         # 测试套件(43 个测试)
│   ├── test_realtime.py           # 实时模块测试(9 个)
│   ├── test_brainflow_csv.py      # BrainFlow CSV 测试(8 个)
│   ├── test_openbci_import.py     # OpenBCI 导入测试
│   ├── test_load_eeg_full.py      # 统一加载入口测试
│   ├── test_module_borrowing_*.py # 模块借鉴测试
│   ├── test_api_endpoints.py      # API 端点测试
│   ├── test_e2e.py                # 端到端测试
│   ├── test_focus.py              # Focus 模块测试
│   └── test_topomap.py            # 地形图测试
├── docs/                          # 文档
│   ├── EEGDataScience-完整技术文档.md  # 本文档
│   ├── superpowers/plans/         # 实施计划
│   └── specs/                     # 设计规范
└── data/                          # 示例数据
```

---

## 4. 数据格式兼容

### 4.1 支持的格式

| 格式 | 扩展名 | 特征 | 检测函数 |
|------|--------|------|----------|
| OpenBCI ODF | .txt / .csv | 首行以 `%OpenBCI` 开头 | `_detect_openbci()` |
| BrainFlow CSV（有表头） | .csv | 逗号分隔，首行为 `0,1,2,...` | `_detect_brainflow_csv()` |
| BrainFlow RAW（Tab 分隔） | .csv | Tab 分隔，无表头，列数 ≥ 10 | `_detect_brainflow_csv()` |
| 普通 CSV | .csv | 逗号分隔，列数 < 10 或含非数字 | 兜底 |

### 4.2 BrainFlow RAW CSV 格式详解

**文件示例**: `BrainFlow-RAW_2026-07-13_16-12-59_4.csv`（Ganglion 4ch）

```
0	-37.93	-29.30	-50.05	-32.96	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.00	1689238379.123	0
1	-38.12	-29.45	-50.20	-33.10	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.00	1689238379.127	0
2	-38.30	-29.60	-50.35	-33.25	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.00	0.00	1689238379.131	0
...
```

**列布局（15 列 Ganglion）**:

| 列索引 | 内容 | 说明 |
|--------|------|------|
| 0 | Sample Index | 0, 1, 2, ... |
| 1-4 | EXG Channel 0-3 | 4 通道 EEG 数据（μV） |
| 5-7 | Accel XYZ | 加速度计 |
| 8-12 | Other | 其他辅助通道 |
| 13 | Timestamp | Unix 时间戳（秒） |
| 14 | Marker | 事件标记 |

### 4.3 列数 → 板卡映射

[openbci_import.py](file:///Users/xiatian/Desktop/EEG-Science/app/analysis/openbci_import.py) 中的 `BRAINFLOW_COLUMN_MAP`:

```python
BRAINFLOW_COLUMN_MAP = {
    24: ("cyton", 8),     # Cyton 8ch (有表头)
    28: ("daisy", 16),    # Cyton 16ch (Daisy, 有表头)
    18: ("ganglion", 4),  # Ganglion 4ch (有表头)
    15: ("ganglion", 4),  # Ganglion 4ch (RAW Tab 分隔)
    22: ("cyton", 8),     # Cyton 8ch (RAW Tab 分隔)
    30: ("daisy", 16),    # Daisy 16ch (RAW Tab 分隔)
}
```

### 4.4 关键修复点（BrainFlow RAW CSV）

**问题 1：EXG 起始列错误**
- 原代码 `data = values[:, :n_exg]` 将 Sample Index（列 0）误识别为第一个 EXG 通道
- 修复：`exg_start = 1; data = values[:, exg_start:exg_start + n_exg]`

**问题 2：截断行含 `'-'`**
- BrainFlow 导出末尾可能存在不完整行，含 `'-'` 字符
- 修复：`pd.read_csv(..., na_values=['-', ''])` + `dropna()`

**问题 3：数字表头**
- 有表头格式首行为 `0,1,2,...`（纯数字索引），会被误读为数据
- 修复：`_is_brainflow_header_row()` 检测首行是否为 `0,1,2,...,n-1` 序列，是则跳过

### 4.5 统一加载入口

[load_eeg_full.py](file:///Users/xiatian/Desktop/EEG-Science/app/analysis/load_eeg_full.py) 提供统一接口：

```python
from app.analysis.openbci_import import load_eeg_full

result = load_eeg_full(filepath)
# 返回统一 dict:
# {
#     'data': np.ndarray,        # (n_samples, n_channels) μV
#     'fs': int,                 # 采样率
#     'channels': List[str],     # 通道名
#     'times': np.ndarray,       # 时间轴(秒)
#     'markers': Optional[List[Marker]],
#     'metadata': {
#         'format': 'openbci_odf' | 'brainflow_csv' | 'plain_csv',
#         'board': str,
#         'n_channels': int,
#         ...
#     }
# }
```

---

## 5. 分析模块详解

### 5.1 心流恢复分析（核心）

**文件**: [flow_recovery.py](file:///Users/xiatian/Desktop/EEG-Science/app/analysis/flow_recovery.py)

**输入**: EEG 数据 + 事件标记（flow_steady_start, switch_start, recovery_start）

**6 项指标**:

| 指标 | 含义 | 类型 |
|------|------|------|
| `theta_alpha_ratio` | Theta/Alpha 比值 | flow（越高越专注） |
| `alpha_rel` | Alpha 相对能量 | flow |
| `beta_rel` | Beta 相对能量 | flow |
| `gamma_rel` | Gamma 相对能量 | loss（越高越分神） |
| `eeg_entropy` | 脑电熵值 | loss |
| `cog_load` | 认知负载指数 | loss |

**恢复时间定义**: 切换事件后，指标首次回到 ±5% 阈值内并持续 N 秒的时间点

### 5.2 频谱分析

**文件**: [spectrum.py](file:///Users/xiatian/Desktop/EEG-Science/app/analysis/spectrum.py)

- **FFT**: scipy.signal.welch，窗长 4s，重叠 50%
- **频带**: δ(1-4Hz) / θ(4-8Hz) / α(8-13Hz) / β(13-30Hz) / γ(30-45Hz)
- **输出**: `band_powers` dict，含每通道的绝对/相对功率

### 5.3 头皮地形图

**文件**: [topomap.py](file:///Users/xiatian/Desktop/EEG-Science/app/analysis/topomap.py)

- **插值方法**: scipy.interpolate.Rbf，multiquadric 基函数
- **电极位置**: 8 通道标准 10-20 系统（Fp1, Fp2, C3, C4, Pz, O1, O2, Fz）
- **输出**: 32×32 网格的插值矩阵，前端 Canvas 渲染

### 5.4 Focus 专注度

**文件**: [focus.py](file:///Users/xiatian/Desktop/EEG-Science/app/analysis/focus.py)

- **算法**: 基于 Theta/Alpha 比值 + Beta 能量的复合指标
- **输出**: `{avg: float, stability: float}`（0-1 范围）

### 5.5 ERSP 时频分析

**文件**: [ersp.py](file:///Users/xiatian/Desktop/EEG-Science/app/analysis/ersp.py)

- **方法**: Morlet 小波连续小波变换（CWT）
- **频率范围**: 1-45Hz
- **输出**: 时频矩阵 + 显著性掩码（相对于基线）

### 5.6 事件相关电位（ERP）

**文件**: [erp.py](file:///Users/xiatian/Desktop/EEG-Science/app/analysis/erp.py)

- 基于 Marker 时间戳，提取 ±500ms 窗口
- 多次试验平均

---

## 6. 实时采集模块

### 6.1 模块组成

```
app/realtime/
├── __init__.py           # 导出 BrainFlowAcquisition, AcquisitionManager, get_manager
├── acquisition.py        # BrainFlow BoardShim 生命周期封装
└── manager.py            # 会话管理器 + 后台轮询 + WebSocket 桥接
```

### 6.2 BrainFlowAcquisition

**文件**: [acquisition.py](file:///Users/xiatian/Desktop/EEG-Science/app/realtime/acquisition.py)

**状态机**:
```
IDLE → CONNECTING → PREPARED → STREAMING → STOPPED → IDLE
                                    ↓
                                  ERROR → IDLE
```

**关键方法**:

| 方法 | 说明 |
|------|------|
| `prepare()` | 调用 `BoardShim.prepare_session()`，获取采样率/通道/板名 |
| `start_stream(buffer_size=450000)` | 调用 `BoardShim.start_stream()`，开始采集 |
| `get_latest_data()` | 调用 `BoardShim.get_board_data()`，返回最新数据帧 |
| `stop_stream()` | 停止采集，状态 → STOPPED |
| `release_session()` | 释放 BoardShim 会话，状态 → IDLE |
| `get_board_info()` | 静态获取板卡信息（无需 prepare） |

**数据帧结构**:
```python
{
    'data': List[List[float]],      # (n_channels, n_samples)
    'channels': List[str],          # ['Fp1', 'Fp2', 'C3', ...]
    'fs': int,                      # 采样率
    'timestamp': float,             # 相对开始时间(秒)
    'sample_indices': List[int],    # 样本索引(用于丢包检测)
}
```

**通道名映射**:
```python
CHANNEL_NAMES_8CH  = ['Fp1', 'Fp2', 'C3', 'C4', 'Pz', 'O1', 'O2', 'Fz']
CHANNEL_NAMES_16CH = CHANNEL_NAMES_8CH + ['F3', 'F4', 'P3', 'P4', 'T3', 'T4', 'Oz', 'FCz']
CHANNEL_NAMES_4CH  = ['Fp1', 'Fp2', 'C3', 'C4']
```

**丢包检测**: 通过 Sample Index 通道的连续性判断，gap > 1 时累加 `_packets_lost`。

### 6.3 AcquisitionManager

**文件**: [manager.py](file:///Users/xiatian/Desktop/EEG-Science/app/realtime/manager.py)

**单例模式**: 通过 `get_manager()` 获取全局唯一实例。

**核心字段**:
```python
self._buffer: deque(maxlen=5000)      # 环形缓冲(约 20 秒 @250Hz)
self._clients: List[Callable]         # WebSocket 推送回调列表
self._poll_thread: threading.Thread   # 后台轮询线程(daemon)
self._focus_cache: Dict               # Focus 缓存
self._band_powers_cache: Dict         # 频带功率缓存
```

**board_id 映射**:
```python
BOARD_ID_MAP = {
    'synthetic': BoardIds.SYNTHETIC_BOARD.value,  # -2
    'cyton': BoardIds.CYTON_BOARD.value,          # 0
    'daisy': BoardIds.CYTON_DAISY_BOARD.value,    # 2
    'ganglion': BoardIds.GANGLION_BOARD.value,    # 1
}
```

**轮询循环** (`_poll_loop`):
1. 每 50ms 调用 `acq.get_latest_data()`
2. 更新环形缓冲（前 8 通道）
3. 每 2 秒计算 Focus + 频带功率
4. 构建推送帧，调用所有注册的客户端回调

**推送帧结构**:
```python
{
    'type': 'data',
    'timestamp': float,
    'channels': List[str],
    'data': List[List[float]],
    'fs': int,
    'focus': {'avg': float, 'stability': float},
    'band_powers': {'delta': float, 'theta': float, ...}
}
```

### 6.4 同步→异步桥接（核心难点）

BrainFlow BoardShim 是**同步阻塞 API**，而 FastAPI WebSocket 是**异步协程**。直接在协程中调用 `get_board_data()` 会阻塞事件循环。

**解决方案**:

```
┌─────────────────┐     call_soon_threadsafe     ┌──────────────────┐
│ daemon 轮询线程  │ ───────────────────────────▶ │ asyncio.Queue    │
│ (同步 BrainFlow) │   push_callback(frame)       │ (异步安全)       │
└─────────────────┘                               └────────┬─────────┘
                                                           │ await queue.get()
                                                           ▼
                                                  ┌──────────────────┐
                                                  │ WebSocket 协程   │
                                                  │ websocket.send   │
                                                  └──────────────────┘
```

**关键代码** ([routers/realtime.py](file:///Users/xiatian/Desktop/EEG-Science/app/routers/realtime.py)):
```python
async def realtime_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager = get_manager()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def push_callback(frame):
        # 同步线程 → 异步队列(线程安全)
        try:
            loop.call_soon_threadsafe(queue.put_nowait, frame)
        except Exception:
            pass

    manager.add_client(push_callback)

    try:
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(frame)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})  # 心跳
    except WebSocketDisconnect:
        pass
    finally:
        manager.remove_client(push_callback)
```

### 6.5 BoardShim 生命周期

借鉴 OpenBCI GUI `BoardBrainflow.pde`:

```
prepare_session()  → 申请串口/蓝牙资源，初始化板卡
       ↓
start_stream(450000)  → 启动内部缓冲，开始接收数据
       ↓
get_board_data()  → 取出缓冲中的所有数据(取出后清空)
       ↓ (循环轮询)
stop_stream()  → 停止采集
       ↓
release_session()  → 释放串口/蓝牙资源
```

**注意**: `get_board_data()` 是**破坏性读取**（取出后缓冲清空），不是 `get_current_board_data()`。这样保证不会重复处理旧数据。

### 6.6 Synthetic Board 说明

- `BoardIds.SYNTHETIC_BOARD = -2`
- 采样率 250 Hz
- **返回 16 个 EXG 通道**（不是 8 个）
- 生成正弦波 + 噪声，用于开发测试
- 无需硬件，直接可用

---

## 7. REST API 参考

### 7.1 离线分析端点

#### `POST /api/upload`
上传 EEG 文件。

**请求**: `multipart/form-data`
- `eeg_file`: EEG 文件（.csv / .txt）
- `events_file`: 事件标记文件（可选）
- `condition`: 条件名

**响应**:
```json
{
    "ok": true,
    "filepath": "/tmp/uploads/xxx.csv",
    "format": "brainflow_csv",
    "n_channels": 4,
    "fs": 200
}
```

#### `POST /api/analyze`
执行完整分析管线。

**请求**: `application/json`
```json
{
    "hp": 1.0,
    "lp": 45.0,
    "notch": 50.0,
    "artifact_threshold": 100.0,
    "window_sec": 4.0,
    "overlap": 0.5,
    "tolerance": 0.05,
    "recovery_window": 10,
    "condition": "custom"
}
```

**响应**: 包含 `recovery_time`, `viz_data`, `band_powers`, `topomap_data`, `focus_scores` 等。

#### `POST /api/sample`
使用模拟数据运行分析（无需上传文件）。

#### `GET /api/openbci/detect?filepath=...`
检测文件格式。

#### `GET /api/openbci/info?filepath=...`
获取文件元信息（不加载全部数据）。

### 7.2 实时采集端点

**文件**: [routers/realtime.py](file:///Users/xiatian/Desktop/EEG-Science/app/routers/realtime.py)

#### `GET /api/realtime/status`
获取当前采集状态。

**响应**:
```json
{
    "state": "IDLE",           // IDLE | CONNECTING | STREAMING | STOPPED | ERROR
    "board_id": null,          // BrainFlow board_id
    "board_name": null,
    "fs": 0,                   // 采样率
    "channels": [],            // 通道名列表
    "n_clients": 0,            // WebSocket 客户端数
    "elapsed_sec": 0.0,        // 已采集时长(秒)
    "packets_lost": 0          // 丢包数
}
```

#### `POST /api/realtime/start`
启动采集。

**请求**:
```json
{
    "board_id": "synthetic",   // synthetic | cyton | daisy | ganglion
    "params": {
        "serial_port": "/dev/cu.usbserial-DM00...",
        "ip_address": "192.168.1.1",
        "ip_port": 6677,
        "mac_address": "..."
    }
}
```

**响应（成功）**:
```json
{
    "ok": true,
    "board_id": -2,
    "board_name": "Synthetic Board",
    "fs": 250,
    "channels": ["Fp1", "Fp2", "C3", ...],
    "n_exg": 16
}
```

**响应（失败）**:
```json
{
    "ok": false,
    "error": "Unknown board_id: invalid_board"
}
```

#### `POST /api/realtime/stop`
停止采集。

**响应**:
```json
{
    "ok": true,
    "elapsed_sec": 45.2
}
```

---

## 8. WebSocket API

### 8.1 端点

```
ws://localhost:8000/ws/realtime
```

### 8.2 消息类型

#### 数据帧（服务器 → 客户端）
```json
{
    "type": "data",
    "timestamp": 12.34,
    "channels": ["Fp1", "Fp2", "C3", "C4", "Pz", "O1", "O2", "Fz"],
    "data": [[...], [...], ...],   // (n_channels, n_samples)
    "fs": 250,
    "focus": {
        "avg": 0.65,
        "stability": 0.82
    },
    "band_powers": {
        "delta": 0.25,
        "theta": 0.18,
        "alpha": 0.22,
        "beta": 0.20,
        "gamma": 0.15
    }
}
```

#### 心跳（服务器 → 客户端）
```json
{
    "type": "ping"
}
```

### 8.3 客户端实现

前端 [realtime.js](file:///Users/xiatian/Desktop/EEG-Science/app/static/js/realtime.js) 特性：
- 自动重连（2 秒延迟）
- Canvas 滚动波形（5 秒窗口 / 1250 样本 @250Hz）
- 自动增益（根据数据范围动态调整 y 轴）
- Focus 实时显示（走神 / 一般 / 专注，颜色编码）
- 频带功率实时条形图（δ θ α β γ）

---

## 9. 前端架构

### 9.1 设计系统

**TraeWork 紧凑浅色系统**，定义于 [style.css](file:///Users/xiatian/Desktop/EEG-Science/app/static/css/style.css):

- **品牌色**: `#4B3FE3`（紫蓝）
- **反色**: `#262626`（主按钮）
- **表面色**: 侧边栏 `#FAFAFA` / 主区 `#FFFFFF`
- **数据色板**: 6 色可视化（紫蓝/绿/蓝/橙/黄/紫）
- **圆角**: 4/6/8/12px
- **字体**: SF Pro / Inter / PingFang SC

### 9.2 布局

```
┌──────────┬──────────────────────────────────────┐
│          │  模块标题栏 (breadcrumb + title)     │
│ 侧边栏    ├────────────┬─────────────────────────┤
│ 200px    │            │                         │
│          │ 配置面板    │      工作区             │
│ 导航分组  │ 260px      │  (图表/结果展示)        │
│          │            │                         │
│ 滤波设置  │            │                         │
│ (底部)   │            │                         │
└──────────┴────────────┴─────────────────────────┘
```

### 9.3 导航分组

| 分组 | 模块 |
|------|------|
| 分析模块 | 心流恢复 / 频谱 / ERP / ERSP / 地形图 / Focus / 脑连接(即将) |
| 工具 | 伪迹检测 / 统计可视化 |
| 实时 | 实时采集 |
| 数据 | OpenBCI 导入 / 被试管理 / 实验记录(即将) / 数据归档(即将) |
| 其他 | 设置(即将) |

### 9.4 实时采集界面

**视图 ID**: `#view-realtime`

**组件**:
- 板卡选择下拉框（Synthetic / Cyton / Daisy / Ganglion）
- 启动 / 停止按钮
- 状态栏（当前状态 + 采集时长 + 丢包数）
- Canvas 波形显示（8 通道叠加，5 秒滚动）
- Focus 指示卡（走神 / 一般 / 专注）
- 频带功率条形图（δ θ α β γ）

### 9.5 心流恢复界面

**配置面板**:
- 数据来源标签（模拟数据 / 上传数据）
- 条件卡片（A→A / A→B / A→C / B→C）
- 上传区域（EEG 文件 + 事件标记 + 条件名）
- 分析参数（带通 / 陷波 / 伪迹阈值 / 窗口 / 重叠 / 容差 / 恢复窗口）

**工作区**:
- 指标卡片（恢复时间 / 伪迹率 / 时长 / 样本数）
- 时序曲线图（6 指标 + 相位背景 + 阈值带）
- 恢复柱状图
- 衰减热力图
- 频谱图 / 地形图 / Focus 显示

---

## 10. 测试体系

### 10.1 测试概览

**总测试数**: 43 个，全部通过（30 秒）

```
tests/
├── test_realtime.py            # 9 个 - 实时采集
├── test_brainflow_csv.py       # 8 个 - BrainFlow CSV 导入
├── test_openbci_import.py      # 3 个 - OpenBCI ODF
├── test_load_eeg_full.py       # 4 个 - 统一加载入口
├── test_module_borrowing_api.py# 3 个 - 模块借鉴 API
├── test_module_borrowing_e2e.py# 2 个 - 模块借鉴 E2E
├── test_api_endpoints.py       # 7 个 - API 端点
├── test_e2e.py                 # 2 个 - 端到端
├── test_focus.py               # 2 个 - Focus 模块
└── test_topomap.py             # 3 个 - 地形图
```

### 10.2 实时模块测试详解

**文件**: [test_realtime.py](file:///Users/xiatian/Desktop/EEG-Science/tests/test_realtime.py)

| 测试名 | 说明 |
|--------|------|
| `test_acquisition_synthetic_start_stop` | BrainFlowAcquisition 完整生命周期 |
| `test_acquisition_board_info` | 板卡信息获取 |
| `test_manager_start_stop` | Manager 启动停止 |
| `test_manager_status_fields` | 状态字段完整性 |
| `test_realtime_status_endpoint` | GET /api/realtime/status |
| `test_realtime_start_stop_endpoints` | POST start + stop |
| `test_realtime_websocket` | WebSocket 数据帧接收 |
| `test_realtime_e2e_synthetic` | 端到端完整流程 |
| `test_realtime_invalid_board` | 无效 board_id 错误处理 |

### 10.3 运行测试

```bash
# 全量测试
python -m pytest tests/ -v

# 仅实时模块
python -m pytest tests/test_realtime.py -v

# 带覆盖率
python -m pytest tests/ --cov=app --cov-report=html
```

### 10.4 BrainFlow RAW CSV 测试

**文件**: [test_brainflow_csv.py](file:///Users/xiatian/Desktop/EEG-Science/tests/test_brainflow_csv.py)

| 测试名 | 说明 |
|--------|------|
| `test_detect_brainflow_csv_true` | 识别有表头 BrainFlow CSV |
| `test_detect_brainflow_raw_tab_true` | 识别 Tab 分隔 RAW 格式 |
| `test_detect_brainflow_csv_false_for_plain_csv` | 排除普通 CSV |
| `test_detect_brainflow_csv_false_for_odf` | 排除 ODF 格式 |
| `test_load_brainflow_csv_cyton8` | Cyton 8ch 加载 |
| `test_load_brainflow_raw_tab_ganglion` | Ganglion RAW 加载 |
| `test_load_brainflow_raw_exg_values_not_sample_index` | **关键**: EXG 值不等于 Sample Index |
| `test_load_brainflow_csv_header_skipped` | 数字表头跳过 |

---

## 11. 部署与运行

### 11.1 环境准备

```bash
# 克隆仓库
git clone https://github.com/NoWint/EGGDataScience.git
cd EGGDataScience

# 安装依赖
pip install fastapi uvicorn brainflow numpy scipy pandas scikit-learn
```

### 11.2 启动服务

```bash
# 开发模式(热重载)
python -m uvicorn app.server:app --reload --host 0.0.0.0 --port 8000

# 生产模式
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 1
```

**注意**: WebSocket 需要保持单 worker（`--workers 1`），否则多进程间共享状态会出问题。

### 11.3 访问

- **前端**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc

### 11.4 硬件连接

#### Cyton / Daisy（USB 串口）
```bash
# macOS 查看串口
ls /dev/cu.usbserial-*

# 启动参数
{
    "board_id": "cyton",
    "params": {"serial_port": "/dev/cu.usbserial-DM00xxxx"}
}
```

#### Ganglion（蓝牙）
```bash
# macOS 蓝牙
{
    "board_id": "ganglion",
    "params": {"mac_address": "Ganglion-xxxx"}
}
```

#### Synthetic（无需硬件，测试用）
```bash
{
    "board_id": "synthetic",
    "params": {}
}
```

---

## 12. 开发指南

### 12.1 添加新分析模块

1. 在 `app/analysis/` 创建 `new_module.py`
2. 实现核心函数，接收 `(data: np.ndarray, fs: int, ...)` 返回 `dict`
3. 在 `run_full_pipeline()` 中调用
4. 添加测试 `tests/test_new_module.py`
5. 前端在 `app.js` 添加渲染函数

### 12.2 添加新板卡支持

1. 在 `manager.py` 的 `BOARD_ID_MAP` 添加映射
2. 在 `acquisition.py` 添加通道名常量（如 `CHANNEL_NAMES_32CH`）
3. 测试

### 12.3 代码风格

- **Python**: PEP 8，类型注解（`typing`）
- **注释**: 中文（技术术语英文）
- **提交信息**: conventional commits（`feat:` / `fix:` / `test:` / `docs:`）
- **设计**: 遵循 TraeWork 设计令牌，不使用裸 CSS 值

### 12.4 Git 工作流

```bash
# 功能分支
git checkout -b feat/new-module

# 提交
git add .
git commit -m "feat: add new analysis module"

# 推送
git push origin feat/new-module

# 合并到 main
git checkout main
git merge feat/new-module
git push origin main
```

---

## 13. 故障排查

### 13.1 BrainFlow CSV 识别失败

**症状**: 上传 BrainFlow RAW CSV 返回 `format: plain_csv`

**排查**:
1. 检查首行是否以 `%` 开头（应为否）
2. 检查列数是否 ≥ 10
3. 检查首行是否全部为数字

**修复**: 确认文件未被编辑器修改（如 Excel 可能改变分隔符）。

### 13.2 EXG 数据全是 Sample Index

**症状**: 加载后第一通道数据为 `0, 1, 2, 3, ...`

**原因**: EXG 起始列错误（从列 0 而非列 1 开始）

**修复**: 已在 commit `d520ca3` 修复，确认 `exg_start = 1`。

### 13.3 实时采集启动失败

**症状**: `POST /api/realtime/start` 返回 `{"ok": false, "error": "..."}`

**排查**:
1. Synthetic Board 无需硬件，应总是成功
2. Cyton: 检查串口是否被其他程序占用（OpenBCI GUI）
3. Ganglion: 检查蓝牙是否配对

**修复**:
```bash
# macOS 释放串口
sudo kill $(lsof -t /dev/cu.usbserial-DM00xxxx)

# 重启蓝牙
sudo pkill bluetoothd
```

### 13.4 WebSocket 无数据

**症状**: 连接 WS 后只收到 `ping`，无 `data` 帧

**排查**:
1. 确认 `/api/realtime/status` 返回 `state: STREAMING`
2. 检查 `n_clients` 是否 > 0
3. 查看后端日志是否有异常

**修复**: 确保 `push_callback` 正确注册，`loop.call_soon_threadsafe` 未抛异常。

### 13.5 通道数为 16 而非 8

**说明**: Synthetic Board 默认返回 16 个 EXG 通道，这是正常行为。

**测试断言**: `assert len(status['channels']) == 16`

### 13.6 Git Push 认证失败

**症状**: `fatal: could not read Username for 'https://github.com'`

**修复**:
```bash
# 使用 gh CLI 配置 git 凭据
gh auth setup-git

# 或使用 token 推送
GH_TOKEN=$(gh auth token)
git push "https://x-access-token:${GH_TOKEN}@github.com/NoWint/EGGDataScience.git" main
```

---

## 附录

### A. 提交历史（本次交付）

| Commit | 类型 | 说明 |
|--------|------|------|
| `d520ca3` | fix | BrainFlow RAW CSV 兼容（Tab 分隔 / Ganglion 15 列） |
| `feac426` | feat | BrainFlowAcquisition BoardShim 封装 |
| `d2c79d6` | feat | AcquisitionManager 后台轮询 + 环形缓冲 |
| `5ca14d3` | feat | REST + WebSocket 端点 |
| `a7461b2` | feat | 实时采集前端界面 |
| `4381bb8` | test | 端到端测试 + 回归验证 |

### B. 关键设计决策

1. **同步→异步桥接**: 使用 `asyncio.Queue` + `loop.call_soon_threadsafe`，避免锁竞争
2. **环形缓冲**: `deque(maxlen=5000)` 自动丢弃旧数据，约 20 秒窗口 @250Hz
3. **Focus 计算频率**: 每 2 秒一次，平衡实时性与计算开销
4. **WebSocket 心跳**: 1 秒超时后发送 ping，保持连接活跃
5. **单 worker 部署**: WebSocket 状态在进程内，多 worker 需 Redis 消息总线
6. **EXG 起始列**: 列 0 是 Sample Index，EXG 从列 1 开始
7. **数字表头检测**: 首行为 `0,1,2,...,n-1` 序列时跳过

### C. 参考资源

- [BrainFlow 文档](https://brainflow.readthedocs.io/)
- [OpenBCI GUI 源码](https://github.com/OpenBCI/OpenBCI_GUI)
- [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/)
- [scipy.interpolate.Rbf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html)

---

*本文档由 EEGDataScience 团队维护，最后更新于 2026-07-13。*
