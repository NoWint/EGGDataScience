# 批量导入 + NeuroLink 实时对接 — 设计文档

> **作者**: NoWint ([https://github.com/NoWint](https://github.com/NoWint))
> **日期**: 2026-07-14
> **状态**: 待实现

---

## 1. 背景与目标

### 1.1 痛点

当前系统每次只能导入单个 EEG 文件分析。实验计划要求 4 名被试 × 4 种条件 = 16 次实验，`EEGdata/` 已有 16 个文件，逐一导入耗时且易错。此外，实验采用 NeuroLink 平台（`wss://eeg.yzjtiantian.cn/ws`）实时采集 EEG 数据，当前系统无法对接该平台做实时监测与分析。

### 1.2 目标

1. **批量导入**：一次选择多个 EEG 文件，通过表格分配到"被试+条件"，批量运行全部 5 个分析模块，导出合并报告 ZIP
2. **NeuroLink 实时对接**：作为 monitor 端连接 NeuroLink 平台，实时监测 EEG 波形与指标，实时心流分析，会话记录保存为可导入批量分析的 CSV，同步实验阶段

### 1.3 范围

本 spec 覆盖两个独立但共享分析内核的功能。后续将用 PyWebView + PyInstaller 将整个应用打包为桌面端 app（独立 spec）。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 UI                                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ 批量导入视图  │  │ 实时监测视图  │  │ 心流恢复分析视图   │ │
│  │ (表格分配+   │  │ (NeuroLink   │  │ (现有)            │ │
│  │  进度+报告)  │  │  连接+波形)  │  │                   │ │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬─────────┘ │
└─────────┼─────────────────┼────────────────────┼───────────┘
          │ HTTP             │ WebSocket          │ HTTP
┌─────────▼─────────────────▼────────────────────▼───────────┐
│                   FastAPI Server                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ /api/batch-  │  │ NeuroLink    │  │ 现有分析端点       │ │
│  │ analyze      │  │ Client(后台  │  │ /api/analyze-all  │ │
│  │ /api/batch-  │  │ 线程→WS桥接) │  │ /api/analyze      │ │
│  │ progress     │  │ /ws/neurolink│  │                   │ │
│  │ /api/export- │  │ 会话记录→CSV │  │                   │ │
│  │ batch-report │  └──────┬───────┘  └─────────┬─────────┘ │
│  └──────┬───────┘         │                     │           │
│         │                 │                     │           │
│  ┌──────▼─────────────────▼─────────────────────▼─────────┐ │
│  │           分析内核 (5 模块: 心流/频谱/ERP/ERSP/伪迹)    │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**关键设计点：**
- NeuroLink 客户端作为后台线程运行，通过 asyncio.Queue 桥接到 `/ws/neurolink` WebSocket 端点（复用现有 realtime 模块的桥接模式）
- 会话记录保存为 BrainFlow RAW 兼容 CSV，可无缝导入批量分析流程，形成"实时采集→离线深度分析"闭环
- 批量导入复用 `/api/analyze-all` 的分析内核，外层加循环+进度跟踪

---

## 3. 批量导入设计

### 3.1 数据流

```
用户选多文件 → 自动按时间戳分组 → 表格分配(被试+条件)
→ POST /api/batch-analyze (多文件+分配表)
→ 服务端循环: 每文件跑5模块, 存 BATCH_RESULTS_STORE
→ 前端轮询 GET /api/batch-progress (当前N/M, 模块名, 成功/失败)
→ 完成后 GET /api/export-batch-report → ZIP 下载
```

### 3.2 后端端点

#### `POST /api/batch-analyze`

- **接收**：多文件上传（`files: List[UploadFile]`）+ JSON 分配表（`assignments: str`，JSON 字符串解析为 `[{filename, subject, condition}, ...]`）
- **处理**：
  1. 保存文件到 `data/uploads/batch/`
  2. 生成 `batch_id`（时间戳）
  3. 启动后台线程串行分析，立即返回 `{batch_id, total}`
  4. 每文件调用分析内核（从 `/api/analyze-all` 提取共享函数 `_run_all_modules(data, fs, events_df)`，批量导入与单文件全分析共用），结果存 `BATCH_RESULTS_STORE[batch_id][f"{subject}_{condition}"]`
- **文件与分配表映射**：分配表中每项的 `filename` 字段与上传文件名匹配，服务端据此将文件内容与分析分配关联
- **进度状态**：`{batch_id, total, current, current_file, current_module, status: "running"|"done"|"failed", errors: [{file, error}]}`

#### `GET /api/batch-progress/{batch_id}`

- 返回当前进度，前端每 2s 轮询

#### `GET /api/export-batch-report?batch_id=xxx`

- 生成 ZIP：
  ```
  EEG_BatchReport_<timestamp>.zip
  ├── batch_summary.md          # 汇总表: 被试×条件×恢复时间×伪迹×各模块状态
  ├── per_file/
  │   ├── S01_AtoA_report.md    # 复用 _build_full_report_md
  │   ├── S01_AtoA_results.json
  │   ├── S01_AtoB_report.md
  │   └── ...
  └── original_data/            # 全部原始文件
  ```

### 3.3 前端视图

**批量导入视图**（新导航项"批量分析"）：
- 文件选择区（支持多选 .csv/.txt）
- 自动分组提示（按文件名时间戳前缀分组，同组标同色）
- 分配表格：每行 = 文件 | 时间戳 | 被试(输入框) | 条件(下拉 A→A/A→B/A→C/B→C)
- "开始批量分析"按钮 → 进度条（N/M 文件，当前模块名）→ 完成后"下载批量报告 ZIP"

### 3.4 约束

- 串行分析（不并行），避免内存峰值；16 文件 × ~40s ≈ 10 分钟
- 单文件失败不中断批次，记入 errors，最终报告标注失败项
- 分配表持久化到 SQLite experiments 表（新增 `eeg_path` 列关联文件）

---

## 4. NeuroLink 实时对接设计

### 4.1 连接流程

```
前端                      后端 NeuroLinkClient               NeuroLink 服务
  │                          │                                    │
  │─POST /api/neurolink/     │                                    │
  │  connect{room,nickname}─→│──── hello(device_info) ──────────→│
  │                          │←─── room_info ────────────────────│
  │                          │──── join_room(code) ─────────────→│
  │                          │←─── room_joined ──────────────────│
  │                          │──── claim_role(monitor) ─────────→│
  │  ← 200 connected         │←─── role_claimed ─────────────────│
  │                          │←─── eeg_frame (120Hz) ────────────│
  │← WS /ws/neurolink ←────  │←─── metrics_snapshot (1Hz) ───────│
  │  (波形+指标+阶段)         │←─── phase_sync / marker ──────────│
  │                          │                                    │
  │─POST /api/neurolink/     │                                    │
  │  start-recording ───────→│ 开始写 CSV                         │
  │─POST /api/neurolink/     │                                    │
  │  stop-recording ────────→│ 停止写 CSV → 存 data/recordings/   │
```

### 4.2 后端组件

#### `app/neurolink_client.py` — NeuroLink WebSocket 客户端

- `NeuroLinkClient` 类：管理连接、握手、数据接收
- 后台线程接收 `eeg_frame`/`metrics_snapshot`/`phase_sync`/`marker`
- 环形缓冲（60s 滑动窗口）供实时分析
- 可选：写入 CSV 记录（BrainFlow RAW 兼容格式，含 markers 列）
- 自动重连：连接断开后 5s 间隔重试，3 次后放弃

#### `app/routers/neurolink.py` — REST + WebSocket 路由

- `POST /api/neurolink/connect` — 连接（参数：room_code, nickname）
- `POST /api/neurolink/disconnect` — 断开
- `POST /api/neurolink/start-recording` — 开始记录（参数：subject, condition）
- `POST /api/neurolink/stop-recording` — 停止记录，返回保存的文件路径
- `GET /api/neurolink/status` — 连接状态、当前阶段、已记录时长
- `WS /ws/neurolink` — 推送实时数据帧到前端（复用 asyncio.Queue 桥接模式）

### 4.3 实时心流分析

- 在接收线程中对滑动窗口实时计算 6 指标（Theta/Alpha 比值、Alpha/Beta/Gamma 能量、谱熵、认知负载），窗口 60s、每 1s 更新一次（与 NeuroLink `metrics_snapshot` 频率对齐，可直接采用服务端计算结果作为交叉验证）
- 基于 `phase_sync` 的 `phase_id` 自动划分阶段（flow1=稳态, switch=切换, recovery=恢复）
- 检测心流进入/脱离（marker code 3/4）触发恢复计时
- 实时只做心流指标，深度分析（ERP/ERSP 等）留给离线批量

### 4.4 前端视图

**实时监测视图**（新导航项"实时监测"）：
- 连接面板：房间号输入 + 昵称 + 连接/断开按钮
- 波形显示：4 通道滚动 Canvas（复用现有 realtime.js 的滚动逻辑）
- 指标面板：θ/α、谱熵、认知负载、频带功率实时数值卡
- 阶段指示器：当前实验阶段（心流诱导/切换/恢复）+ 倒计时
- 记录控制：开始/停止记录按钮，被试+条件输入，停止后提示"可导入批量分析"

### 4.5 会话记录格式

保存为 BrainFlow RAW 兼容 CSV（与 EEGdata/ 中文件格式一致），列：
```
Index, EXG1, EXG2, EXG3, EXG4, Accel_X, Accel_Y, Accel_Z, Timestamp, Marker
```
停止记录后，文件可直接拖入批量导入流程分析，形成"实时采集→离线深度分析"闭环。

---

## 5. 共享基础设施

### 5.1 数据库扩展

`experiments` 表新增列关联文件与记录：
```sql
ALTER TABLE experiments ADD COLUMN eeg_path TEXT;        -- 关联的 EEG 文件路径
ALTER TABLE experiments ADD COLUMN source TEXT;          -- 'upload' | 'neurolink_recording'
ALTER TABLE experiments ADD COLUMN analysis_status TEXT; -- 'pending'|'running'|'done'|'failed'
```
批量分析完成后更新 `analysis_status`，实现"被试→实验→文件→分析结果"完整链路。

### 5.2 分析内核复用

批量导入和 NeuroLink 记录的离线分析都调用现有的 `run_full_pipeline`/`run_spectrum_analysis`/`run_erp_analysis`/`run_ersp_analysis`/`run_artifact_analysis` 五个函数，不修改分析内核本身。

---

## 6. 错误处理

| 场景 | 处理 |
|------|------|
| 批量分析单文件失败 | 记入 errors，继续下一文件，报告标注失败项 |
| NeuroLink 连接断开 | 自动重连（5s 间隔，3 次后放弃），前端提示 |
| NeuroLink 房间号错误 | 返回 `room_denied`，前端显示明确错误 |
| 记录中连接中断 | 保留已记录数据，标记为 incomplete |
| 批量进度轮询超时 | 前端 30s 无响应显示"分析可能仍在运行，可重新打开页面查看" |
| ZIP 导出时数据缺失 | 跳过缺失项，summary 标注 |

---

## 7. 测试策略

- **批量导入**：用 EEGdata/ 真实文件做端到端测试（4 文件子集），验证进度轮询 + ZIP 结构
- **NeuroLink**：用 mock WebSocket server 模拟 `eeg_frame`/`phase_sync`，验证握手、记录、断线重连
- **记录格式兼容**：记录的 CSV 用现有 `load_eeg_full()` 加载验证

---

## 8. 不做（YAGNI）

- 不做并发分析（串行足够，避免内存峰值）
- 不做 NeuroLink 数据转发给其他客户端（仅 monitor 角色）
- 不做实时 ERP/ERSP（实时只做心流指标，深度分析留给离线批量）
- 不做用户认证/多用户（单机桌面应用场景）

---

## 9. 文件变更清单

**新增：**
- `app/neurolink_client.py` — NeuroLink WebSocket 客户端
- `app/routers/neurolink.py` — NeuroLink REST+WS 路由
- `app/routers/batch.py` — 批量分析路由
- `app/static/js/batch.js` — 批量导入前端
- `app/static/js/neurolink.js` — 实时监测前端

**修改：**
- `app/server.py` — 注册新路由
- `app/database.py` — experiments 表加列
- `app/static/index.html` — 新增导航项+视图容器+脚本引用
- `app/static/js/app.js` — 侧边栏导航扩展
