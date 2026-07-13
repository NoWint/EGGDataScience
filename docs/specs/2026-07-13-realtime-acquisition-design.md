# 实时脑电采集 设计文档

> 借鉴 OpenBCI GUI 实时采集能力,基于 BrainFlow BoardShim 实现 EEGDataScience 的实时设备连接与数据流。

## 目标

为 EEGDataScience 平台新增实时 EEG 采集能力:
1. 支持 OpenBCI Cyton/Daisy/Ganglion 设备实时连接
2. WebSocket 推送实时数据至前端,滚动波形渲染
3. 实时 Focus 专注度 + 频带功率显示
4. Synthetic Board 合成板支持无硬件开发测试

## 架构

```
浏览器 (realtime.js)
    ↕ WebSocket (/ws/realtime)
FastAPI 后端
    ├── REST: /api/realtime/{start,stop,status}
    └── 后台线程: 50ms 轮询 get_board_data() → 推送 WS 帧
         ↕ BrainFlow BoardShim
         OpenBCI 设备 或 Synthetic Board
```

**方案选择**:WebSocket (非 SSE),原因:
- 双向通信(前端可发控制命令)
- 低延迟,适合 250Hz 多通道 EEG
- FastAPI 原生支持,无需额外依赖

## 组件

### 后端 — app/realtime/ (新建模块)

**acquisition.py** — BoardShim 生命周期封装:
- `BrainFlowAcquisition` 类:封装 prepare_session / start_stream / poll / stop / release
- 后台线程:50ms 间隔调用 `get_board_data()`,推入缓冲队列
- 丢包检测:对比 sample index 列,记录间隙

**manager.py** — 会话管理器:
- 单例模式,全局唯一 BoardShim 实例
- WebSocket 客户端列表(支持多浏览器标签)
- 环形缓冲队列(最近 N 秒数据,供实时 Focus 计算)
- 状态机:IDLE → CONNECTING → STREAMING → STOPPED / ERROR

**routers/realtime.py** — REST + WebSocket 端点:
- `POST /api/realtime/start` — 请求体 `{board_id, params}` → 启动采集
- `POST /api/realtime/stop` — 停止采集
- `GET /api/realtime/status` — 返回 `{state, board_id, fs, channels, n_clients, elapsed_sec}`
- `WS /ws/realtime` — 接受连接,推送数据帧

### 前端 — app/static/js/realtime.js (新建)

- WebSocket 客户端:连接 /ws/realtime,自动重连
- Canvas 滚动波形:多通道实时绘制(每 50ms 刷新)
- 实时 Focus 条:复用后端推送的 focus_score
- 频带功率实时柱状图
- 连接状态指示器(IDLE/连接中/采集中/错误)
- 设备选择 UI:下拉框(Synthetic/Cyton/Daisy/Ganglion)+ 连接参数表单

### 前端 — app/static/index.html (修改)

- 侧边栏新增"实时采集"入口
- 主内容区新增实时采集视图(设备选择 + 状态 + Canvas + Focus 面板)

## API 设计

### POST /api/realtime/start

请求:
```json
{
    "board_id": "synthetic",  // "synthetic" | "cyton" | "daisy" | "ganglion"
    "params": {
        "serial_port": "/dev/ttyUSB0",  // Cyton/Daisy Serial
        "ip_address": "192.168.1.1",    // WiFi 模式
        "ip_port": 6677
    }
}
```

响应:
```json
{
    "ok": true,
    "board_id": -2,
    "board_name": "Synthetic Board",
    "fs": 250,
    "channels": ["Fp1", "Fp2", "C3", "C4", "Pz", "O1", "O2", "Fz"],
    "n_exg": 8,
    "n_accel": 3
}
```

### POST /api/realtime/stop

响应:`{"ok": true, "elapsed_sec": 120.5}`

### GET /api/realtime/status

响应:
```json
{
    "state": "STREAMING",  // IDLE | CONNECTING | STREAMING | STOPPED | ERROR
    "board_id": -2,
    "board_name": "Synthetic Board",
    "fs": 250,
    "channels": ["Fp1", ...],
    "n_clients": 1,
    "elapsed_sec": 45.2,
    "packets_lost": 0
}
```

### WS /ws/realtime

服务器推送数据帧(每 50ms):
```json
{
    "type": "data",
    "timestamp": 45.2,
    "channels": ["Fp1", "Fp2", ...],
    "data": [[1.2, 1.3, ...], [0.8, 0.9, ...]],  // (n_channels, n_samples_in_frame)
    "fs": 250,
    "focus": {"avg": 0.75, "stability": 0.12},  // 滚动缓冲计算
    "band_powers": {"delta": 0.15, "theta": 0.08, "alpha": 0.35, "beta": 0.25, "gamma": 0.17}
}
```

控制帧(客户端→服务器):
```json
{"action": "start", "board_id": "synthetic", "params": {}}
{"action": "stop"}
```

## 数据流

1. 用户在"实时采集"视图选择设备(Synthetic 默认),点击"开始采集"
2. 前端 `POST /api/realtime/start` 或通过 WebSocket 发送 start 帧
3. 后端 `manager.start_acquisition(board_id, params)`:
   - 创建 `BoardShim(board_id, BrainFlowInputParams(...))`
   - `prepare_session()` → `start_stream(450000)` (缓冲大小)
   - 启动后台轮询线程
4. 后台线程循环:
   - 每 50ms 调用 `board.get_board_data()` 获取新数据
   - 提取 EXG 通道(用 `BoardShim.get_exg_channels(board_id)`)
   - 更新环形缓冲(最近 20 秒)
   - 每 2 秒对缓冲计算 `compute_focus_scores` + `compute_band_powers`
   - JSON 编码数据帧,推送到所有 WebSocket 客户端
5. 前端接收帧:
   - Canvas 滚动绘制最新数据(保留最近 5 秒可见)
   - 更新 Focus 分数显示 + 频带功率柱状图
6. 用户点击"停止采集" → `POST /api/realtime/stop` → `stop_stream()` + `release_session()`

## 设备支持

| board_id 字符串 | BrainFlow BoardIds | 通道数 | 采样率 | 连接方式 |
|---|---|---|---|---|
| `synthetic` | `SYNTHETIC_BOARD` (-2) | 8 EXG + 3 Accel | 250 Hz | 无(合成数据) |
| `cyton` | `CYTON_BOARD` (0) | 8 EXG + 3 Accel | 250 Hz | Serial / WiFi |
| `daisy` | `CYTON_DAISY_BOARD` (2) | 16 EXG + 3 Accel | 125 Hz | Serial / WiFi |
| `ganglion` | `GANGLION_BOARD` (1) | 4 EXG + 3 Accel | 200 Hz | BLE / WiFi |

通道名映射(8 通道标准):`['Fp1', 'Fp2', 'C3', 'C4', 'Pz', 'O1', 'O2', 'Fz']`

## 错误处理

- `prepare_session()` 失败(设备未连接/端口占用):返回 400 + 错误详情
- `start_stream()` 失败:状态置为 ERROR,释放会话,推送错误帧
- WebSocket 断开:从客户端列表移除,不影响采集
- 后台线程异常:状态置为 ERROR,推送错误帧,尝试 release_session
- BrainFlow 未安装:`/api/realtime/start` 返回 503 + "brainflow not available"

## 测试策略

- **Synthetic Board 端到端**:无需硬件,测试完整 start → stream → stop 流程
- **WebSocket 连接测试**:TestClient 连接 WS,确认收到数据帧
- **状态机测试**:IDLE → STREAMING → STOPPED 转换
- **丢包检测测试**:Synthetic Board 无丢包(理想情况)
- **多客户端测试**:2 个 WS 连接同时接收

## 新手友好

- 默认 Synthetic Board,无需硬件即可体验完整流程
- 设备选择带说明文字:"合成板(演示用,无需硬件)" / "Cyton (8通道,专业级)"
- 连接状态实时颜色反馈:灰(空闲)/ 黄(连接中)/ 绿(采集中)/ 红(错误)
- 实时 Focus 分数带文字提示:专注(绿)/ 一般(黄)/ 走神(红)
- 波形自动调整增益,新手无需手动调参

## 范围边界

- 本期只做 Synthetic Board + 基础架构,真实设备连接(Serial/WiFi/BLE 参数)留后续
- 不做数据录制保存(已有离线 CSV 上传分析流程)
- 不做多设备并行(单例 BoardShim)
- 不做远程网络转发(单机本地使用)
