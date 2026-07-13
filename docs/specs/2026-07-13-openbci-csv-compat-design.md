# OpenBCI CSV 兼容设计文档

## 项目概述

**项目名称**: EEGDataScience
**作者 GitHub**: [NoWint](https://github.com/NoWint)
**子项目**: OpenBCI CSV 兼容
**目标**: 让 EEGDataScience 平台完全透明地支持 OpenBCI GUI 导出的两种文本格式(ODF .txt + BrainFlow .csv),用户上传后系统自动检测格式并解析,无需手动选择数据源类型。

## 范围

### 包含
- 支持 OpenBCI GUI 默认导出的 ODF 格式(.txt,带 `%OpenBCI` 文件头)
- 支持 BrainFlow 直接导出的 CSV 格式(无文件头,数字索引列名)
- 提取 EXG(脑电,μV)+ Accel(加速度)+ Marker(事件标记)三类通道
- Marker 自动映射为 events_df
- 向后兼容现有 `load_eeg()` 4 元组接口

### 不包含
- BDF 二进制格式
- Digital/Analog IO 通道
- 实时设备连接(BrainFlow 实时流)
- 模块借鉴(FFT/频带能量等可视化模块)
- UI 改动(纯后端兼容)

## 现状诊断

### 阻断点(根因)
`/api/upload` [第 147 行](file:///Users/xiatian/Desktop/EEG-Science/app/server.py) `if not eeg_file.filename.endswith('.csv')` 拒绝 `.txt`,而 OpenBCI GUI ODF 默认导出 `.txt`。

### 次级问题
1. `load_openbci()` 只提取 EXG,不含 Accel/Marker
2. `_raw_to_uv()` 逻辑错误 —— OpenBCI 导出的 EXG **已经是 μV**(BrainFlow 内部转换过),再除一次会错误缩放
3. 不支持 BrainFlow CSV 格式
4. Marker Channel 未映射为 events_df
5. 时间轴用 Sample Index 推算,不如直接用 Timestamp 准确

## 整体架构

```
用户上传 .txt/.csv
       │
       ▼
/api/upload (放宽后缀: .txt + .csv)
       │
       ▼
load_eeg_full(filepath)  ← 新统一入口
       │
       ├─ _detect_openbci(fp) → True? → load_openbci(fp)  [ODF 格式]
       │                                        │
       │                                        ├─ 解析 % 头(板卡/通道/采样率)
       │                                        ├─ 提取 EXG(已是 μV,不转换)
       │                                        ├─ 提取 Accel
       │                                        ├─ 提取 Marker
       │                                        └─ 返回 dict
       │
       ├─ _detect_brainflow_csv(fp) → True? → load_brainflow_csv(fp)
       │                                        │
       │                                        ├─ 无 % 头,但列名为纯数字索引
       │                                        ├─ 按列数启发式判断板卡
       │                                        ├─ 同 ODF 提取逻辑
       │                                        └─ 返回 dict
       │
       └─ 否则 → 普通 CSV 解析(现有逻辑)
                  └─ 返回 dict (accel=None, markers=None)
       │
       ▼
统一 dict:
{
  data: np.ndarray (n_samples, n_exg),     # EXG, μV
  fs: int,                                  # 采样率
  channels: List[str],                      # EXG 通道名
  times: np.ndarray (n_samples,),           # 时间轴(秒)
  accel: np.ndarray | None (n_samples, 3),  # 加速度 XYZ (g)
  markers: List[Marker] | None,             # 事件标记
  metadata: dict,                           # 板卡/格式/通道数等元信息
}
       │
       ▼
/api/analyze
  ├─ data, fs, channels, times → preprocess → extract_features → ...
  └─ markers → events_df (无事件文件时自动使用)
```

## 组件清单

| 组件 | 文件 | 职责 |
|---|---|---|
| `load_eeg_full()` | `app/analysis/flow_recovery.py` | 新增统一入口,自动检测格式,返回完整 dict |
| `load_eeg()` | 同上 | 保持 4 元组接口,内部改调 `load_eeg_full()` 取前 4 字段 |
| `load_openbci()` | `app/analysis/openbci_import.py` | 扩展:提取 EXG+Accel+Marker,删除 `_raw_to_uv` |
| `load_brainflow_csv()` | 同上 | 新增:解析 BrainFlow CSV |
| `_detect_brainflow_csv()` | 同上 | 新增:检测 BrainFlow CSV |
| `Marker` dataclass | 同上 | 新增:`{timestamp, value, label}` |
| `/api/upload` | `app/server.py` | 放宽后缀限制 |
| `/api/analyze` | `app/server.py` | 改用 `load_eeg_full()`,markers → events_df |

## 格式检测与解析细节

### ODF 格式(OpenBCI GUI 默认)

**检测条件**:文件首行以 `%OpenBCI` 开头(现有 `_detect_openbci()` 已实现)

**样本格式**:
```
%OpenBCI Raw EXG Data
%Number of channels = 8
%Sample Rate = 250 Hz
%Board = OpenBCI_GUI$BoardCytonSerial
Sample Index, EXG Channel 0, ..., EXG Channel 7, Accel Channel 0-2,
... Other/Digital/Analog ..., Timestamp, Marker Channel, Timestamp (Formatted)
0, 45997.52, ..., 1557936889064, 0, 12:14:49.064
```

**解析逻辑扩展**:
- EXG 通道:已是 μV,**直接读取**(删除 `_raw_to_uv` 调用)
- Accel 通道:从列名 `Accel Channel 0/1/2` 提取,3 列
- Marker Channel:从列名 `Marker Channel` 提取,非 0 值为事件标记
- Timestamp:用毫秒列计算时间轴(秒),比 Sample Index 更准确

### BrainFlow CSV 格式

BrainFlow 直接通过 `board_shim.get_board_data()` 导出,无 `%` 头,列名为纯数字索引。

**检测条件**(启发式):
- 首行不以 `%` 开头(排除 ODF)
- 列名是纯数字索引(`0, 1, 2, ...`)且列数 ≥ 10(排除普通 CSV)

**板卡判断**(按列数启发式):

| 列数 | 板卡 | EXG 通道数 |
|---|---|---|
| 24 | Cyton 8ch | 8 |
| 28 | Cyton 16ch (Daisy) | 16 |
| 18 | Ganglion 4ch | 4 |

**列位置约定**(BrainFlow BoardShim 默认):
- 最后一列 = Marker Channel
- 倒数第二列 = Timestamp(秒,非毫秒)
- 前 N 列 = EXG(N = 通道数)
- EXG 后 3 列 = Accel

### Marker → events_df 映射

```python
@dataclass
class Marker:
    timestamp: float   # 秒
    value: int         # 原始 marker 值
    label: str         # "marker_{value}"

# 从 marker 列提取非 0 值
markers = [
    Marker(timestamp=times[i], value=int(val), label=f"marker_{int(val)}")
    for i, val in enumerate(marker_column) if val != 0
]

# 无事件文件时转 events_df
events_df = pd.DataFrame(
    [(m.label, m.timestamp) for m in markers],
    columns=['event_id', 'timestamp']
)
```

## 统一 dict 返回结构

```python
{
    'data': np.ndarray,           # (n_samples, n_exg) μV
    'fs': int,                    # 采样率
    'channels': List[str],        # ['EXG_0', 'EXG_1', ...]
    'times': np.ndarray,          # (n_samples,) 秒
    'accel': np.ndarray | None,   # (n_samples, 3) g,或 None
    'markers': List[Marker] | None,
    'metadata': {
        'format': 'openbci_odf' | 'brainflow_csv' | 'plain_csv',
        'board': str,             # 'cyton' | 'ganglion' | 'daisy' | 'unknown'
        'n_channels': int,
        'sample_rate': int,
        'has_accelerometer': bool,
        'has_markers': bool,
        'duration_sec': float,
        'n_samples': int,
    }
}
```

## API 变更

### `/api/upload`(修改)

```python
# 放宽后缀限制
ALLOWED_EXTS = ('.csv', '.txt')

if not eeg_file.filename.lower().endswith(ALLOWED_EXTS):
    raise HTTPException(400, "EEG文件需为 CSV 或 TXT 格式")
```

### `/api/analyze`(修改)

```python
# 改用 load_eeg_full()
result = load_eeg_full(eeg_path)
data, fs, channels, times = result['data'], result['fs'], result['channels'], result['times']

# 事件文件优先;无事件文件时用 markers
if events_path.exists():
    events_df = pd.read_csv(events_path)
elif result['markers']:
    events_df = pd.DataFrame(
        [(m.label, m.timestamp) for m in result['markers']],
        columns=['event_id', 'timestamp']
    )
else:
    # 现有默认时序(从 server.py 现有代码迁移)
    events_df = pd.DataFrame([
        ('S0', 0.0), ('B0', 5.0), ('B1', 65.0),
        ('F0', 65.0), ('F1', 305.0), ('F2', 545.0),
        ('X0', 545.0), ('X1', 665.0),
        ('R0', 665.0), ('R1', 1265.0), ('Q0', 1265.0),
    ], columns=['event_id', 'timestamp'])

# 返回结果增加元信息
result_out['metadata'] = result['metadata']
result_out['has_accel'] = result['accel'] is not None
result_out['has_markers'] = len(result['markers'] or []) > 0
```

### `/api/openbci/*`(保留)

现有 `/api/openbci/detect|info|convert|save` 保留,供需要独立获取元信息的场景使用。内部改调 `load_eeg_full()`。

## 错误处理

| 场景 | 处理 |
|---|---|
| 文件无法读取 | 400 + 错误信息 |
| 非 OpenBCI/BrainFlow/普通 CSV 格式 | 当普通 CSV 处理,失败则 400 |
| EXG 通道数为 0 | 400 "未找到 EXG 通道" |
| Timestamp 列缺失 | 退化为 Sample Index 推算 |
| Accel 通道缺失 | `accel = None` |
| Marker 列缺失 | `markers = None` |
| BrainFlow CSV 列数不在 18/24/28 | `board='unknown'`,按列数 ÷ 4 近似推断 EXG 通道数 |

## 测试策略

用 OpenBCI GUI 仓库自带的样本文件测试:
- [OpenBCI_GUI-v6-meditation.txt](file:///Users/xiatian/Desktop/EEG-Science/OpenBCI_GUI/OpenBCI_GUI/data/EEG_Sample_Data/OpenBCI_GUI-v6-meditation.txt) — Cyton 8ch ODF

测试用例:
1. ODF 检测 → 解析 8 通道 EXG + 3 通道 Accel + Marker
2. BrainFlow CSV 检测(构造测试文件)→ 解析正确通道
3. 普通 CSV → 现有逻辑不受影响
4. `/api/upload` 接受 `.txt`
5. `/api/analyze` 用 markers 自动生成 events_df
6. `load_eeg()` 向后兼容(4 元组)

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `app/analysis/openbci_import.py` | 修改 | 删除 `_raw_to_uv`,扩展 `load_openbci()` 提取 Accel+Marker,新增 `load_brainflow_csv()`、`_detect_brainflow_csv()`、`Marker` |
| `app/analysis/flow_recovery.py` | 修改 | 新增 `load_eeg_full()`,`load_eeg()` 改调它 |
| `app/analysis/__init__.py` | 修改 | 导出 `load_eeg_full`、`Marker`、`load_brainflow_csv` |
| `app/server.py` | 修改 | `/api/upload` 放宽后缀,`/api/analyze` 改用 `load_eeg_full()` |
| `app/routers/openbci.py` | 修改 | 内部改调 `load_eeg_full()` |

## YAGNI 检查

- ❌ 不做 BDF 二进制(用户未选)
- ❌ 不做 Digital/Analog 通道(用户未选)
- ❌ 不做实时采集(后续子项目)
- ❌ 不做 UI 改动(纯后端兼容)
- ✅ 保留 `/api/openbci/*` 独立端点(已有,改动最小)
- ✅ `load_eeg()` 保持 4 元组(向后兼容)
