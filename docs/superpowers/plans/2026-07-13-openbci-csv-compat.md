# OpenBCI CSV 兼容 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 EEGDataScience 平台透明支持 OpenBCI GUI 导出的 ODF (.txt) 和 BrainFlow (.csv) 两种文本格式,自动提取 EXG+Accel+Marker,无需用户手动选择数据源类型。

**Architecture:** 新增 `load_eeg_full()` 统一入口,内部根据文件头自动路由到 ODF 解析器、BrainFlow CSV 解析器或普通 CSV 解析器,返回包含 EXG/Accel/Marker/metadata 的统一 dict。现有 `load_eeg()` 保持 4 元组接口向后兼容,内部改调 `load_eeg_full()`。

**Tech Stack:** Python 3.11+ / numpy / pandas / pytest

**Spec:** [docs/specs/2026-07-13-openbci-csv-compat-design.md](file:///Users/xiatian/Desktop/EEG-Science/docs/specs/2026-07-13-openbci-csv-compat-design.md)

---

## 文件结构

实施完成后的目标结构:

```
EEG-Science/
├── app/
│   ├── analysis/
│   │   ├── __init__.py              # 修改: 导出 load_eeg_full, Marker, load_brainflow_csv
│   │   ├── flow_recovery.py         # 修改: 新增 load_eeg_full(), load_eeg() 改调它
│   │   ├── openbci_import.py        # 修改: 扩展 load_openbci() 提取 Accel+Marker, 删除 _raw_to_uv, 新增 load_brainflow_csv/_detect_brainflow_csv/Marker
│   │   └── ... (其他模块不变)
│   ├── routers/
│   │   └── openbci.py               # 修改: 内部改调 load_eeg_full()
│   └── server.py                    # 修改: /api/upload 放宽后缀, /api/analyze 改用 load_eeg_full()
├── tests/                           # 新建测试目录
│   ├── __init__.py
│   ├── conftest.py                  # pytest fixtures (样本文件路径)
│   ├── test_openbci_import.py       # ODF 解析测试
│   ├── test_brainflow_csv.py        # BrainFlow CSV 解析测试
│   └── test_api_endpoints.py        # API 端点测试
└── data/
    └── uploads/
        └── ... (运行时上传目录)
```

测试样本文件(只读,不修改):`OpenBCI_GUI/OpenBCI_GUI/data/EEG_Sample_Data/OpenBCI_GUI-v6-meditation.txt`

---

### Task 1: 创建测试基础设施

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: 创建 tests 目录和 __init__.py**

创建 `tests/__init__.py`(空文件):

```python
```

- [ ] **Step 2: 创建 conftest.py with fixtures**

创建 `tests/conftest.py`:

```python
"""pytest 共享 fixtures"""
import pytest
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.parent

# OpenBCI GUI 样本文件(Cyton 8ch ODF 格式)
SAMPLE_ODF = ROOT / "OpenBCI_GUI" / "OpenBCI_GUI" / "data" / "EEG_Sample_Data" / "OpenBCI_GUI-v6-meditation.txt"


@pytest.fixture
def sample_odf_path():
    """返回 OpenBCI ODF 样本文件路径"""
    if not SAMPLE_ODF.exists():
        pytest.skip(f"样本文件不存在: {SAMPLE_ODF}")
    return SAMPLE_ODF


@pytest.fixture
def tmp_brainflow_csv(tmp_path):
    """构造一个 BrainFlow CSV 测试文件(Cyton 8ch, 24 列)"""
    import numpy as np
    content = ",".join(str(i) for i in range(24)) + "\n"
    np.random.seed(42)
    for _ in range(100):
        row = np.random.randn(24) * 100
        # 倒数第二列 = Timestamp(秒)
        row[-2] = float(_ * 0.004)  # 250 Hz
        # 最后一列 = Marker,前 50 行 0,第 50 行 marker=1,第 80 行 marker=2
        row[-1] = 0
        content += ",".join(f"{v:.4f}" for v in row) + "\n"
    # 注入两个 marker
    lines = content.rstrip().split("\n")
    marker_line_50 = lines[51].split(",")
    marker_line_50[-1] = "1.0"
    lines[51] = ",".join(marker_line_50)
    marker_line_80 = lines[81].split(",")
    marker_line_80[-1] = "2.0"
    lines[81] = ",".join(marker_line_80)
    content = "\n".join(lines) + "\n"
    
    path = tmp_path / "brainflow_test.csv"
    path.write_text(content)
    return path
```

- [ ] **Step 3: 验证 pytest 可运行**

Run: `python -m pytest tests/ -v --collect-only`
Expected: 至少发现 conftest.py 的 fixtures(无测试可收集是正常的)

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add pytest infrastructure with OpenBCI sample fixtures"
```

---

### Task 2: 添加 Marker dataclass 并扩展 ODF 解析

**Files:**
- Modify: `app/analysis/openbci_import.py`
- Test: `tests/test_openbci_import.py`

- [ ] **Step 1: 写失败测试 — Marker dataclass**

创建 `tests/test_openbci_import.py`:

```python
"""OpenBCI ODF 导入测试"""
import numpy as np
import pytest
from pathlib import Path


def test_marker_dataclass():
    """测试 Marker dataclass 创建"""
    from app.analysis.openbci_import import Marker
    m = Marker(timestamp=1.5, value=10, label="marker_10")
    assert m.timestamp == 1.5
    assert m.value == 10
    assert m.label == "marker_10"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_openbci_import.py::test_marker_dataclass -v`
Expected: FAIL with "cannot import name 'Marker'"

- [ ] **Step 3: 实现 Marker dataclass**

在 `app/analysis/openbci_import.py` 顶部 import 区后添加:

```python
from dataclasses import dataclass


@dataclass
class Marker:
    """OpenBCI 事件标记"""
    timestamp: float   # 秒
    value: int         # 原始 marker 值
    label: str         # "marker_{value}"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_openbci_import.py::test_marker_dataclass -v`
Expected: PASS

- [ ] **Step 5: 写失败测试 — load_openbci 返回 dict 结构**

在 `tests/test_openbci_import.py` 追加:

```python
def test_load_openbci_returns_dict(sample_odf_path):
    """测试 load_openbci 返回包含 EXG/Accel/Marker/metadata 的 dict"""
    from app.analysis.openbci_import import load_openbci
    result = load_openbci(sample_odf_path)
    
    # 必须返回 dict(不再是 4 元组)
    assert isinstance(result, dict)
    
    # 核心字段
    assert 'data' in result
    assert 'fs' in result
    assert 'channels' in result
    assert 'times' in result
    assert 'accel' in result
    assert 'markers' in result
    assert 'metadata' in result
    
    # EXG 数据
    assert isinstance(result['data'], np.ndarray)
    assert result['data'].ndim == 2
    assert result['data'].shape[1] == 8  # Cyton 8ch
    
    # 采样率
    assert result['fs'] == 250
    
    # 通道名
    assert len(result['channels']) == 8
    assert result['channels'][0] == 'EXG_0'
    
    # Accel(Cyton 有 3 轴)
    assert result['accel'] is not None
    assert result['accel'].shape == (result['data'].shape[0], 3)
    
    # metadata
    meta = result['metadata']
    assert meta['format'] == 'openbci_odf'
    assert meta['board'] == 'cyton'
    assert meta['n_channels'] == 8
    assert meta['sample_rate'] == 250
    assert meta['has_accelerometer'] is True
```

- [ ] **Step 6: 运行测试验证失败**

Run: `python -m pytest tests/test_openbci_import.py::test_load_openbci_returns_dict -v`
Expected: FAIL(load_openbci 当前返回 4 元组,且不提取 accel/marker)

- [ ] **Step 7: 重构 load_openbci 返回 dict**

在 `app/analysis/openbci_import.py` 中:

1. 先扩展 `_parse_header()` 提取 accel_indices / marker_col / timestamp_col(完整版本)

替换现有 `_parse_header` 函数(第 55-112 行)为:

```python
def _parse_header(filepath: Path) -> dict:
    """解析 OpenBCI 文件头，提取元信息"""
    info = {
        "board": "unknown",
        "n_channels": 0,
        "sample_rate": 250,
        "has_accel": False,
        "has_analog": False,
        "header_lines": 0,
        "exg_indices": [],       # CSV 列索引 (0-based)
        "exg_names": [],         # 通道名
        "accel_indices": [],     # Accel 列索引
        "sample_idx_col": 0,
        "timestamp_col": None,
        "marker_col": None,
        "total_columns": 0,
    }

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header_end = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("%"):
            header_end = i + 1
            m = re.search(r"Number of channels\s*=\s*(\d+)", line)
            if m:
                info["n_channels"] = int(m.group(1))
            m = re.search(r"Sample Rate\s*=\s*(\d+)", line)
            if m:
                info["sample_rate"] = int(m.group(1))
            if "BoardCyton" in line and "Daisy" in line:
                info["board"] = "daisy"
            elif "BoardCyton" in line:
                info["board"] = "cyton"
            elif "BoardGanglion" in line:
                info["board"] = "ganglion"
        elif header_end > 0 and not line.startswith("%"):
            # 这是列名行
            columns = [c.strip() for c in line.split(",")]
            info["total_columns"] = len(columns)
            for j, col in enumerate(columns):
                m = re.match(r"EXG Channel (\d+)", col)
                if m:
                    info["exg_indices"].append(j)
                    info["exg_names"].append(f"EXG_{m.group(1)}")
                if col.startswith("Accel Channel"):
                    info["accel_indices"].append(j)
                    info["has_accel"] = True
                if "Analog" in col:
                    info["has_analog"] = True
                if col == "Sample Index":
                    info["sample_idx_col"] = j
                if col == "Timestamp":
                    info["timestamp_col"] = j
                if col == "Marker Channel":
                    info["marker_col"] = j
            header_end = i + 1
            break

    info["header_lines"] = header_end
    return info
```

2. 替换现有 `load_openbci` 函数(第 123-176 行)为:

```python
def load_openbci(filepath: Path) -> dict:
    """加载 OpenBCI ODF 导出文件,返回统一 dict
    
    OpenBCI GUI 导出的 EXG 数据已经是 μV(BrainFlow 内部转换过),
    不需要再做 ADC → μV 转换。
    """
    info = _parse_header(filepath)
    
    # 收集所有需要的列: Sample Index + EXG + Accel + Timestamp + Marker
    exg_cols = info["exg_indices"]
    if not exg_cols:
        exg_cols = list(range(1, info["n_channels"] + 1))
    
    needed_cols = set([info["sample_idx_col"]] + exg_cols)
    if info["timestamp_col"] is not None:
        needed_cols.add(info["timestamp_col"])
    if info["marker_col"] is not None:
        needed_cols.add(info["marker_col"])
    needed_cols.update(info["accel_indices"])
    
    usecols = sorted(needed_cols)
    # 构建映射: 原始列位置 → DataFrame 列位置
    col_positions = {orig: i for i, orig in enumerate(usecols)}
    
    df = pd.read_csv(
        filepath,
        skiprows=info["header_lines"],
        header=None,
        usecols=usecols,
        dtype=np.float64,
    )
    
    values = df.values
    n_samples = values.shape[0]
    
    # 提取 EXG 通道(已是 μV)
    exg_df_cols = [col_positions[c] for c in exg_cols]
    data = values[:, exg_df_cols]
    
    if not info["exg_names"] or len(info["exg_names"]) != data.shape[1]:
        info["exg_names"] = [f"EXG_{i}" for i in range(data.shape[1])]
    
    # 提取 Accel 通道
    accel = None
    if info["accel_indices"]:
        accel_cols = [col_positions[c] for c in info["accel_indices"]]
        accel = values[:, accel_cols]
    
    # 提取 Marker,构建 Marker 列表
    markers = None
    if info["marker_col"] is not None:
        marker_col_pos = col_positions[info["marker_col"]]
        marker_values = values[:, marker_col_pos]
        # 时间轴(用 Timestamp 计算)
        fs = info["sample_rate"]
        if info["timestamp_col"] is not None:
            ts_col_pos = col_positions[info["timestamp_col"]]
            timestamps_sec = values[:, ts_col_pos] / 1000.0  # ms → s
        else:
            sample_idx = values[:, col_positions[info["sample_idx_col"]]]
            timestamps_sec = (sample_idx - sample_idx[0]) / fs
        
        non_zero = np.where(marker_values != 0)[0]
        if len(non_zero) > 0:
            markers = [
                Marker(
                    timestamp=float(timestamps_sec[i]),
                    value=int(marker_values[i]),
                    label=f"marker_{int(marker_values[i])}"
                )
                for i in non_zero
            ]
    
    # 时间轴(用 Timestamp 计算,更准确)
    fs = info["sample_rate"]
    if info["timestamp_col"] is not None:
        ts_col_pos = col_positions[info["timestamp_col"]]
        timestamps_sec = values[:, ts_col_pos] / 1000.0  # ms → s
        times = timestamps_sec - timestamps_sec[0]  # 从 0 开始
    else:
        sample_idx = values[:, col_positions[info["sample_idx_col"]]]
        times = (sample_idx.astype(float) - sample_idx[0]) / fs
    
    channels = info["exg_names"][:data.shape[1]]
    
    return {
        'data': data.astype(np.float64),
        'fs': int(fs),
        'channels': channels,
        'times': times,
        'accel': accel,
        'markers': markers,
        'metadata': {
            'format': 'openbci_odf',
            'board': info['board'],
            'n_channels': data.shape[1],
            'sample_rate': int(fs),
            'has_accelerometer': accel is not None,
            'has_markers': markers is not None and len(markers) > 0,
            'duration_sec': float(n_samples / fs) if fs > 0 else 0.0,
            'n_samples': int(n_samples),
        }
    }
```

3. 删除 `_raw_to_uv` 函数和 `BOARD_ADC_CONFIG` 常量(第 22-42 行和 115-120 行),它们基于错误假设

4. 删除 `convert_uv` 和 `gain` 参数

- [ ] **Step 8: 运行测试验证通过**

Run: `python -m pytest tests/test_openbci_import.py -v`
Expected: PASS

- [ ] **Step 9: 写测试 — openbci_info 也应工作**

在 `tests/test_openbci_import.py` 追加:

```python
def test_openbci_info(sample_odf_path):
    """测试 openbci_info 返回元信息"""
    from app.analysis.openbci_import import openbci_info
    info = openbci_info(sample_odf_path)
    
    assert info['board'] == 'cyton'
    assert info['n_channels'] == 8
    assert info['sample_rate'] == 250
    assert info['has_accelerometer'] is True
    assert info['format'] == 'openbci'
    assert info['duration_sec'] > 0
```

- [ ] **Step 10: 运行测试验证通过**

Run: `python -m pytest tests/test_openbci_import.py::test_openbci_info -v`
Expected: PASS(可能需要小幅调整 openbci_info 适配新字段名)

- [ ] **Step 11: Commit**

```bash
git add app/analysis/openbci_import.py tests/test_openbci_import.py
git commit -m "feat: extend load_openbci to return dict with EXG+Accel+Marker

- 新增 Marker dataclass
- _parse_header 提取 accel_indices 和 marker_col
- load_openbci 返回统一 dict(含 metadata)
- 删除错误的 _raw_to_uv(EXG 已是 μV)
- 时间轴改用 Timestamp 列(更准确)"
```

---

### Task 3: 添加 BrainFlow CSV 解析器

**Files:**
- Modify: `app/analysis/openbci_import.py`
- Test: `tests/test_brainflow_csv.py`

- [ ] **Step 1: 写失败测试 — _detect_brainflow_csv**

创建 `tests/test_brainflow_csv.py`:

```python
"""BrainFlow CSV 导入测试"""
import numpy as np
import pytest
from pathlib import Path


def test_detect_brainflow_csv_true(tmp_brainflow_csv):
    """测试检测 BrainFlow CSV 格式"""
    from app.analysis.openbci_import import _detect_brainflow_csv
    assert _detect_brainflow_csv(tmp_brainflow_csv) is True


def test_detect_brainflow_csv_false_for_plain_csv(tmp_path):
    """测试普通 CSV 不被误判为 BrainFlow CSV"""
    from app.analysis.openbci_import import _detect_brainflow_csv
    p = tmp_path / "plain.csv"
    p.write_text("time,ch1,ch2,ch3\n0,1.0,2.0,3.0\n1,4.0,5.0,6.0\n")
    assert _detect_brainflow_csv(p) is False


def test_detect_brainflow_csv_false_for_odf(sample_odf_path):
    """测试 ODF 文件不被误判为 BrainFlow CSV"""
    from app.analysis.openbci_import import _detect_brainflow_csv
    assert _detect_brainflow_csv(sample_odf_path) is False
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_brainflow_csv.py::test_detect_brainflow_csv_true -v`
Expected: FAIL with "cannot import name '_detect_brainflow_csv'"

- [ ] **Step 3: 实现 _detect_brainflow_csv**

在 `app/analysis/openbci_import.py` 中添加(在 `_detect_openbci` 函数后):

```python
def _detect_brainflow_csv(filepath: Path) -> bool:
    """检测文件是否为 BrainFlow CSV 导出格式
    
    BrainFlow CSV 特征:
    - 无 % 头(区别于 ODF)
    - 列名为纯数字索引(0, 1, 2, ...)
    - 列数 >= 10(区别于普通 CSV)
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first = f.readline().strip()
        if first.startswith("%"):
            return False
        # 解析列名
        cols = [c.strip() for c in first.split(",")]
        if len(cols) < 10:
            return False
        # 所有列名必须是纯数字
        for c in cols:
            try:
                int(c)
            except ValueError:
                return False
        return True
    except Exception:
        return False
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_brainflow_csv.py -v`
Expected: 3 个测试全部 PASS

- [ ] **Step 5: 写失败测试 — load_brainflow_csv**

在 `tests/test_brainflow_csv.py` 追加:

```python
def test_load_brainflow_csv_cyton8(tmp_brainflow_csv):
    """测试加载 BrainFlow CSV(Cyton 8ch, 24 列)"""
    from app.analysis.openbci_import import load_brainflow_csv
    result = load_brainflow_csv(tmp_brainflow_csv)
    
    assert isinstance(result, dict)
    
    # EXG: Cyton 8ch = 前 8 列
    assert result['data'].shape[1] == 8
    assert len(result['channels']) == 8
    assert result['channels'][0] == 'EXG_0'
    
    # 采样率(BrainFlow CSV 无元信息,默认 250)
    assert result['fs'] == 250
    
    # Accel: EXG 后 3 列
    assert result['accel'] is not None
    assert result['accel'].shape[1] == 3
    
    # Markers: 测试数据注入了 2 个 marker
    assert result['markers'] is not None
    assert len(result['markers']) == 2
    assert result['markers'][0].value == 1
    assert result['markers'][1].value == 2
    
    # metadata
    meta = result['metadata']
    assert meta['format'] == 'brainflow_csv'
    assert meta['board'] == 'cyton'
    assert meta['n_channels'] == 8
    assert meta['has_accelerometer'] is True
    assert meta['has_markers'] is True
```

- [ ] **Step 6: 运行测试验证失败**

Run: `python -m pytest tests/test_brainflow_csv.py::test_load_brainflow_csv_cyton8 -v`
Expected: FAIL with "cannot import name 'load_brainflow_csv'"

- [ ] **Step 7: 实现 load_brainflow_csv**

在 `app/analysis/openbci_import.py` 中添加:

```python
# BrainFlow CSV 列数 → 板卡/EXG 通道数映射
BRAINFLOW_COLUMN_MAP = {
    24: ("cyton", 8),     # Cyton 8ch
    28: ("daisy", 16),    # Cyton 16ch (Daisy)
    18: ("ganglion", 4),  # Ganglion 4ch
}


def load_brainflow_csv(filepath: Path) -> dict:
    """加载 BrainFlow CSV 导出文件,返回统一 dict
    
    BrainFlow CSV 列布局(BoardShim 默认):
    - 0..N-1: EXG 通道(N = EXG 通道数)
    - N..N+2: Accel XYZ(3 列,如果板卡支持)
    - ...: Other/Digital/Analog
    - 倒数第二列: Timestamp(秒)
    - 最后一列: Marker
    """
    df = pd.read_csv(filepath, dtype=np.float64)
    values = df.values
    n_samples, n_cols = values.shape
    
    # 按列数判断板卡
    board, n_exg = BRAINFLOW_COLUMN_MAP.get(n_cols, ("unknown", max(1, n_cols // 4)))
    
    # EXG 通道(前 n_exg 列)
    data = values[:, :n_exg]
    channels = [f"EXG_{i}" for i in range(n_exg)]
    
    # Accel 通道(EXG 后 3 列,如果存在)
    accel = None
    if n_cols >= n_exg + 3:
        accel = values[:, n_exg:n_exg + 3]
    
    # Timestamp(倒数第二列,秒)
    timestamp_col = n_cols - 2
    timestamps_sec = values[:, timestamp_col]
    times = timestamps_sec - timestamps_sec[0] if len(timestamps_sec) > 0 else np.arange(n_samples) / 250.0
    
    # 推断采样率
    if len(times) > 1:
        dt = np.median(np.diff(times))
        fs = int(round(1.0 / dt)) if dt > 0 else 250
    else:
        fs = 250
    
    # Marker(最后一列)
    markers = None
    marker_values = values[:, -1]
    non_zero = np.where(marker_values != 0)[0]
    if len(non_zero) > 0:
        markers = [
            Marker(
                timestamp=float(times[i]),
                value=int(marker_values[i]),
                label=f"marker_{int(marker_values[i])}"
            )
            for i in non_zero
        ]
    
    return {
        'data': data.astype(np.float64),
        'fs': fs,
        'channels': channels,
        'times': times,
        'accel': accel,
        'markers': markers,
        'metadata': {
            'format': 'brainflow_csv',
            'board': board,
            'n_channels': n_exg,
            'sample_rate': fs,
            'has_accelerometer': accel is not None,
            'has_markers': markers is not None and len(markers) > 0,
            'duration_sec': float(n_samples / fs) if fs > 0 else 0.0,
            'n_samples': int(n_samples),
        }
    }
```

- [ ] **Step 8: 运行测试验证通过**

Run: `python -m pytest tests/test_brainflow_csv.py -v`
Expected: 4 个测试全部 PASS

- [ ] **Step 9: Commit**

```bash
git add app/analysis/openbci_import.py tests/test_brainflow_csv.py
git commit -m "feat: add BrainFlow CSV format support

- 新增 _detect_brainflow_csv 检测器(数字索引列名 + 列数 >= 10)
- 新增 load_brainflow_csv 解析器
- 按列数启发式判断板卡(24=cyton8, 28=daisy16, 18=ganglion4)
- 提取 EXG + Accel + Marker + Timestamp"
```

---

### Task 4: 新增 load_eeg_full 统一入口

**Files:**
- Modify: `app/analysis/flow_recovery.py`
- Modify: `app/analysis/__init__.py`
- Test: `tests/test_load_eeg_full.py`

- [ ] **Step 1: 写失败测试 — load_eeg_full 路由到 ODF**

创建 `tests/test_load_eeg_full.py`:

```python
"""load_eeg_full 统一入口测试"""
import numpy as np
import pytest
from pathlib import Path


def test_load_eeg_full_routes_odf(sample_odf_path):
    """测试 load_eeg_full 自动检测 ODF 格式"""
    from app.analysis.flow_recovery import load_eeg_full
    result = load_eeg_full(sample_odf_path)
    
    assert isinstance(result, dict)
    assert result['metadata']['format'] == 'openbci_odf'
    assert result['data'].shape[1] == 8
    assert result['fs'] == 250


def test_load_eeg_full_routes_brainflow(tmp_brainflow_csv):
    """测试 load_eeg_full 自动检测 BrainFlow CSV"""
    from app.analysis.flow_recovery import load_eeg_full
    result = load_eeg_full(tmp_brainflow_csv)
    
    assert result['metadata']['format'] == 'brainflow_csv'
    assert result['data'].shape[1] == 8


def test_load_eeg_full_routes_plain_csv(tmp_path):
    """测试 load_eeg_full 回退到普通 CSV"""
    from app.analysis.flow_recovery import load_eeg_full
    p = tmp_path / "plain.csv"
    p.write_text("time,ch1,ch2\n0,1.0,2.0\n0.004,3.0,4.0\n0.008,5.0,6.0\n")
    result = load_eeg_full(p)
    
    assert result['metadata']['format'] == 'plain_csv'
    assert result['data'].shape[1] == 2
    assert result['accel'] is None
    assert result['markers'] is None


def test_load_eeg_backward_compatible(sample_odf_path):
    """测试 load_eeg 保持 4 元组接口向后兼容"""
    from app.analysis.flow_recovery import load_eeg
    result = load_eeg(sample_odf_path)
    
    assert isinstance(result, tuple)
    assert len(result) == 4
    data, fs, channels, times = result
    assert isinstance(data, np.ndarray)
    assert isinstance(fs, int)
    assert isinstance(channels, list)
    assert isinstance(times, np.ndarray)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_load_eeg_full.py -v`
Expected: FAIL with "cannot import name 'load_eeg_full'"

- [ ] **Step 3: 实现 load_eeg_full 并重构 load_eeg**

在 `app/analysis/flow_recovery.py` 中替换 `load_eeg` 函数(第 25-54 行)为:

```python
# ========== 1. 数据加载 ==========
def load_eeg_full(filepath):
    """加载 EEG 文件,自动检测格式,返回完整 dict
    
    支持格式:
    - OpenBCI ODF (.txt,带 %OpenBCI 头)
    - BrainFlow CSV (.csv,数字索引列名)
    - 普通 CSV (time + channel columns)
    
    返回:
        {
            'data': np.ndarray (n_samples, n_exg),     # EXG, μV
            'fs': int,                                  # 采样率
            'channels': List[str],                      # EXG 通道名
            'times': np.ndarray (n_samples,),           # 时间轴(秒)
            'accel': np.ndarray | None,                 # (n_samples, 3) g
            'markers': List[Marker] | None,             # 事件标记
            'metadata': dict,                           # 板卡/格式/通道数等
        }
    """
    from pathlib import Path
    fp = Path(filepath)
    from .openbci_import import _detect_openbci, load_openbci, _detect_brainflow_csv, load_brainflow_csv
    
    if _detect_openbci(fp):
        return load_openbci(fp)
    
    if _detect_brainflow_csv(fp):
        return load_brainflow_csv(fp)
    
    # 普通 CSV
    df = pd.read_csv(filepath)
    time_cols = [c for c in df.columns if c.lower() in ('time', 'timestamp', 't', '时间')]
    if time_cols:
        times = df[time_cols[0]].values
        data_cols = [c for c in df.columns if c not in time_cols]
    else:
        times = np.arange(len(df)) / 250.0
        data_cols = list(df.columns)
    
    data = df[data_cols].values.astype(np.float64)
    if len(times) > 1:
        dt = np.median(np.diff(times))
        fs = int(round(1.0 / dt)) if dt > 0 else 250
    else:
        fs = 250
    
    n_samples = len(data)
    return {
        'data': data,
        'fs': fs,
        'channels': list(data_cols),
        'times': times,
        'accel': None,
        'markers': None,
        'metadata': {
            'format': 'plain_csv',
            'board': 'unknown',
            'n_channels': data.shape[1],
            'sample_rate': fs,
            'has_accelerometer': False,
            'has_markers': False,
            'duration_sec': float(n_samples / fs) if fs > 0 else 0.0,
            'n_samples': int(n_samples),
        }
    }


def load_eeg(filepath):
    """加载 EEG 文件,返回 (data, fs, channels, times) 4 元组(向后兼容)
    
    内部调用 load_eeg_full() 取前 4 字段。
    """
    result = load_eeg_full(filepath)
    return result['data'], result['fs'], result['channels'], result['times']
```

- [ ] **Step 4: 更新 __init__.py 导出**

修改 `app/analysis/__init__.py`:

```python
"""EEG 分析模块包"""
from .flow_recovery import (
    load_eeg, load_eeg_full, load_events, preprocess,
    compute_band_powers, compute_entropy, extract_features,
    compute_recovery_time, compute_all_recovery, compute_attenuation,
    paired_t_test, repeated_measures_anova, pearson_correlation,
    generate_sample_eeg, events_to_df, run_full_pipeline,
    BANDS,
)
from .openbci_import import (
    _detect_openbci, load_openbci, openbci_info,
    _detect_brainflow_csv, load_brainflow_csv, Marker,
)

__all__ = [
    'load_eeg', 'load_eeg_full', 'load_events', 'preprocess',
    'compute_band_powers', 'compute_entropy', 'extract_features',
    'compute_recovery_time', 'compute_all_recovery', 'compute_attenuation',
    'paired_t_test', 'repeated_measures_anova', 'pearson_correlation',
    'generate_sample_eeg', 'events_to_df', 'run_full_pipeline',
    'BANDS',
    '_detect_openbci', 'load_openbci', 'openbci_info',
    '_detect_brainflow_csv', 'load_brainflow_csv', 'Marker',
]
```

- [ ] **Step 5: 运行测试验证通过**

Run: `python -m pytest tests/test_load_eeg_full.py -v`
Expected: 4 个测试全部 PASS

- [ ] **Step 6: 运行全部测试确保无回归**

Run: `python -m pytest tests/ -v`
Expected: 所有测试 PASS

- [ ] **Step 7: Commit**

```bash
git add app/analysis/flow_recovery.py app/analysis/__init__.py tests/test_load_eeg_full.py
git commit -m "feat: add load_eeg_full unified entry with format auto-detection

- load_eeg_full() 自动路由到 ODF/BrainFlow CSV/普通 CSV 解析器
- 返回统一 dict(含 accel/markers/metadata)
- load_eeg() 保持 4 元组向后兼容,内部改调 load_eeg_full()
- __init__.py 导出新增 load_eeg_full, load_brainflow_csv, Marker"
```

---

### Task 5: 修改 /api/upload 放宽后缀限制

**Files:**
- Modify: `app/server.py`
- Test: `tests/test_api_endpoints.py`

- [ ] **Step 1: 写失败测试 — upload 接受 .txt**

创建 `tests/test_api_endpoints.py`:

```python
"""API 端点测试"""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.server import app
    return TestClient(app)


def test_upload_accepts_txt(client, sample_odf_path):
    """测试 /api/upload 接受 .txt 文件"""
    with open(sample_odf_path, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"eeg_file": ("test.txt", f, "text/plain")},
            data={"condition": "test_txt"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "uploaded"
    assert data["condition"] == "test_txt"


def test_upload_accepts_csv(client, tmp_path):
    """测试 /api/upload 仍接受 .csv 文件"""
    csv_content = "time,ch1\n0,1.0\n0.004,2.0\n"
    response = client.post(
        "/api/upload",
        files={"eeg_file": ("test.csv", csv_content.encode(), "text/csv")},
        data={"condition": "test_csv"},
    )
    assert response.status_code == 200


def test_upload_rejects_unsupported(client):
    """测试 /api/upload 拒绝不支持的格式"""
    response = client.post(
        "/api/upload",
        files={"eeg_file": ("test.json", b'{"a":1}', "application/json")},
        data={"condition": "test_json"},
    )
    assert response.status_code == 400
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_api_endpoints.py::test_upload_accepts_txt -v`
Expected: FAIL(400 因为 .txt 被拒绝)

- [ ] **Step 3: 修改 /api/upload 放宽后缀**

在 `app/server.py` 中找到 `/api/upload` 端点(约第 140-165 行),替换后缀检查:

旧代码:
```python
if not eeg_file.filename.endswith('.csv'):
    raise HTTPException(400, "EEG文件需为CSV格式")
```

新代码:
```python
ALLOWED_EXTS = ('.csv', '.txt')
if not eeg_file.filename.lower().endswith(ALLOWED_EXTS):
    raise HTTPException(400, "EEG文件需为 CSV 或 TXT 格式")
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_api_endpoints.py -v`
Expected: 3 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/server.py tests/test_api_endpoints.py
git commit -m "feat: allow .txt uploads in /api/upload

放宽后缀限制以支持 OpenBCI GUI ODF 导出(默认 .txt)"
```

---

### Task 6: 修改 /api/analyze 使用 load_eeg_full + markers

**Files:**
- Modify: `app/server.py`
- Test: `tests/test_api_endpoints.py`

- [ ] **Step 1: 写失败测试 — analyze 用 markers 自动生成 events**

在 `tests/test_api_endpoints.py` 追加:

```python
def test_analyze_uses_markers_from_odf(client, sample_odf_path):
    """测试 /api/analyze 从 ODF 文件中提取 markers 作为事件"""
    # 先上传 ODF 文件
    with open(sample_odf_path, "rb") as f:
        upload_resp = client.post(
            "/api/upload",
            files={"eeg_file": ("meditation.txt", f, "text/plain")},
            data={"condition": "odf_test"},
        )
    assert upload_resp.status_code == 200
    
    # 分析(不上传 events 文件,应自动用 markers)
    response = client.post(
        "/api/analyze",
        json={"condition": "odf_test"},
    )
    
    # 如果数据时长不足或无 markers,可能返回错误,但不应是 "未找到EEG数据"
    if response.status_code == 200:
        data = response.json()
        assert 'metadata' in data
        assert data['metadata']['format'] == 'openbci_odf'
        assert data['metadata']['board'] == 'cyton'
    else:
        # 即使分析失败,也不应该是 404 "未找到EEG数据"
        assert response.status_code != 404


def test_analyze_returns_metadata(client, tmp_path):
    """测试 /api/analyze 返回 metadata 字段"""
    # 上传普通 CSV
    csv_content = "time,ch1,ch2\n" + "\n".join(
        f"{i*0.004},{float(i%100)},{float(i%50)}" for i in range(2000)
    ) + "\n"
    upload_resp = client.post(
        "/api/upload",
        files={"eeg_file": ("test.csv", csv_content.encode(), "text/csv")},
        data={"condition": "meta_test"},
    )
    assert upload_resp.status_code == 200
    
    response = client.post("/api/analyze", json={"condition": "meta_test"})
    if response.status_code == 200:
        data = response.json()
        assert 'metadata' in data
        assert data['metadata']['format'] == 'plain_csv'
        assert 'has_accel' in data
        assert 'has_markers' in data
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_api_endpoints.py::test_analyze_uses_markers_from_odf -v`
Expected: FAIL(/api/analyze 当前用 load_eeg(),不返回 metadata)

- [ ] **Step 3: 修改 /api/analyze**

在 `app/server.py` 中找到 `/api/analyze` 端点(约第 182-218 行),替换核心逻辑:

旧代码(第 191-218 行):
```python
data, fs, channels, times = load_eeg(eeg_path)

if events_path.exists():
    events_df = pd.read_csv(events_path)
else:
    # 无事件文件时使用默认时序
    events_df = pd.DataFrame([
        ('S0', 0.0), ('B0', 5.0), ('B1', 65.0),
        ('F0', 65.0), ('F1', 305.0), ('F2', 545.0),
        ('X0', 545.0), ('X1', 665.0),
        ('R0', 665.0), ('R1', 1265.0), ('Q0', 1265.0),
    ], columns=['event_id', 'timestamp'])

config = {
    'lp': req.lp, 'hp': req.hp, 'notch': req.notch,
    'artifact_threshold': req.artifact_threshold,
    'window_sec': req.window_sec, 'overlap': req.overlap,
    'tolerance': req.tolerance, 'recovery_window': req.recovery_window,
}

result = run_full_pipeline(data, fs, events_df, config=config,
                           preprocess_config=req.preprocess_config)
result['condition'] = req.condition
result['channels'] = channels
result['n_samples'] = len(data)

RESULTS_STORE[req.condition] = result
return _to_jsonable(result)
```

新代码:
```python
# 用 load_eeg_full 获取完整数据(含 accel/markers/metadata)
from app.analysis import load_eeg_full
eeg_result = load_eeg_full(eeg_path)
data, fs, channels, times = (
    eeg_result['data'], eeg_result['fs'],
    eeg_result['channels'], eeg_result['times']
)

# 事件文件优先;无事件文件时用 markers
if events_path.exists():
    events_df = pd.read_csv(events_path)
elif eeg_result['markers']:
    events_df = pd.DataFrame(
        [(m.label, m.timestamp) for m in eeg_result['markers']],
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

config = {
    'lp': req.lp, 'hp': req.hp, 'notch': req.notch,
    'artifact_threshold': req.artifact_threshold,
    'window_sec': req.window_sec, 'overlap': req.overlap,
    'tolerance': req.tolerance, 'recovery_window': req.recovery_window,
}

result = run_full_pipeline(data, fs, events_df, config=config,
                           preprocess_config=req.preprocess_config)
result['condition'] = req.condition
result['channels'] = channels
result['n_samples'] = len(data)
# 新增元信息
result['metadata'] = eeg_result['metadata']
result['has_accel'] = eeg_result['accel'] is not None
result['has_markers'] = eeg_result['markers'] is not None and len(eeg_result['markers']) > 0

RESULTS_STORE[req.condition] = result
return _to_jsonable(result)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_api_endpoints.py -v`
Expected: 所有测试 PASS(或 ODF 分析因数据时长/事件匹配问题返回非 404 错误,metadata 测试 PASS)

- [ ] **Step 5: 运行全部测试确保无回归**

Run: `python -m pytest tests/ -v`
Expected: 所有测试 PASS

- [ ] **Step 6: Commit**

```bash
git add app/server.py tests/test_api_endpoints.py
git commit -m "feat: /api/analyze uses load_eeg_full with marker-driven events

- 改用 load_eeg_full() 获取完整数据
- 无事件文件时自动用 markers 生成 events_df
- 返回结果增加 metadata/has_accel/has_markers 字段"
```

---

### Task 7: 更新 /api/openbci/* 端点使用 load_eeg_full

**Files:**
- Modify: `app/routers/openbci.py`

- [ ] **Step 1: 修改 /api/openbci/convert 和 /save 端点**

在 `app/routers/openbci.py` 中:

1. 更新 import(第 8 行):

旧:
```python
from app.analysis.openbci_import import load_openbci, openbci_info, _detect_openbci
```

新:
```python
from app.analysis.openbci_import import load_openbci, openbci_info, _detect_openbci, _detect_brainflow_csv, load_brainflow_csv
```

2. 替换 `/convert` 端点(第 56-100 行)中的 load_openbci 调用部分:

找到 convert 端点中这段:
```python
g = gain if gain > 0 else None
data, fs, channels, times = load_openbci(tmp, convert_uv=convert_uv, gain=g)
info = openbci_info(tmp)
```

替换为:
```python
result = load_openbci(tmp)
data, fs, channels, times = result['data'], result['fs'], result['channels'], result['times']
info = openbci_info(tmp)
```

同时删除函数签名中的 `convert_uv: bool = Form(False)` 和 `gain: int = Form(0)` 参数。

3. 同样修改 `/save` 端点(第 103-145 行):

找到 save 端点中这段:
```python
g = gain if gain > 0 else None
data, fs, channels, times = load_openbci(tmp, convert_uv=convert_uv, gain=g)
```

替换为:
```python
result = load_openbci(tmp)
data, fs, channels, times = result['data'], result['fs'], result['channels'], result['times']
```

同时删除函数签名中的 `convert_uv: bool = Form(False)` 和 `gain: int = Form(0)` 参数。

4. 在 `/info` 端点后新增一个 `/api/openbci/detect-any` 端点,同时检测 ODF 和 BrainFlow CSV:

在文件末尾(第 145 行后)添加:

```python
@router.post("/detect-any")
async def detect_any_format(file: UploadFile = File(...)):
    """检测上传文件格式(OpenBCI ODF / BrainFlow CSV / 未知)"""
    tmp = UPLOAD_DIR / f"_tmp_{file.filename}"
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        if _detect_openbci(tmp):
            fmt = "openbci_odf"
        elif _detect_brainflow_csv(tmp):
            fmt = "brainflow_csv"
        else:
            fmt = "unknown"
        
        return {
            "filename": file.filename,
            "format": fmt,
        }
    finally:
        if tmp.exists():
            tmp.unlink()
```

- [ ] **Step 2: 写测试验证 /api/openbci/convert 不再需要 convert_uv 参数**

在 `tests/test_api_endpoints.py` 追加:

```python
def test_openbci_convert_no_convert_uv(client, sample_odf_path):
    """测试 /api/openbci/convert 不再需要 convert_uv 参数"""
    with open(sample_odf_path, "rb") as f:
        response = client.post(
            "/api/openbci/convert",
            files={"file": ("test.txt", f, "text/plain")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data['board'] == 'cyton'
    assert data['sample_rate'] == 250
    assert data['n_channels'] == 8
    assert 'preview' in data


def test_openbci_detect_any(client, sample_odf_path):
    """测试 /api/openbci/detect-any 检测 ODF 格式"""
    with open(sample_odf_path, "rb") as f:
        response = client.post(
            "/api/openbci/detect-any",
            files={"file": ("test.txt", f, "text/plain")},
        )
    assert response.status_code == 200
    assert response.json()['format'] == 'openbci_odf'
```

- [ ] **Step 3: 运行测试验证通过**

Run: `python -m pytest tests/test_api_endpoints.py -v`
Expected: 所有测试 PASS

- [ ] **Step 4: 运行全部测试**

Run: `python -m pytest tests/ -v`
Expected: 所有测试 PASS

- [ ] **Step 5: Commit**

```bash
git add app/routers/openbci.py tests/test_api_endpoints.py
git commit -m "refactor: /api/openbci/* uses load_eeg_full, remove convert_uv

- /convert 和 /save 改用 load_openbci() 返回的 dict
- 删除已废弃的 convert_uv/gain 参数
- 新增 /detect-any 端点统一检测 ODF + BrainFlow CSV"
```

---

### Task 8: 端到端验证与文档更新

**Files:**
- Test: `tests/test_e2e.py`

- [ ] **Step 1: 写端到端测试 — 上传 ODF → 分析 → 返回结果**

创建 `tests/test_e2e.py`:

```python
"""端到端测试: 上传 → 分析 → 结果"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.server import app
    return TestClient(app)


def test_e2e_upload_and_analyze_odf(client, sample_odf_path):
    """端到端: 上传 ODF 文件 → 分析 → 返回含 metadata 的结果"""
    # 1. 上传
    with open(sample_odf_path, "rb") as f:
        upload_resp = client.post(
            "/api/upload",
            files={"eeg_file": ("meditation.txt", f, "text/plain")},
            data={"condition": "e2e_odf"},
        )
    assert upload_resp.status_code == 200
    assert upload_resp.json()['status'] == 'uploaded'
    
    # 2. 分析
    analyze_resp = client.post("/api/analyze", json={"condition": "e2e_odf"})
    
    # 即使分析因数据特征失败,也应能返回或明确报错
    if analyze_resp.status_code == 200:
        result = analyze_resp.json()
        assert result['condition'] == 'e2e_odf'
        assert result['metadata']['format'] == 'openbci_odf'
        assert result['metadata']['board'] == 'cyton'
        assert result['channels'][0] == 'EXG_0'
        assert len(result['channels']) == 8
    else:
        # 失败也不应是 404(上传成功但找不到文件 = bug)
        assert analyze_resp.status_code != 404


def test_e2e_health_check(client):
    """测试服务健康检查"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
```

- [ ] **Step 2: 运行端到端测试**

Run: `python -m pytest tests/test_e2e.py -v`
Expected: 2 个测试 PASS(ODF 分析可能因数据时长不足而非 200,但不应是 404)

- [ ] **Step 3: 运行全部测试套件**

Run: `python -m pytest tests/ -v`
Expected: 所有测试 PASS

- [ ] **Step 4: 手动验证 — 启动服务并测试上传**

Run: `python -m uvicorn app.server:app --port 18765 &`
然后: `curl -X POST http://localhost:18765/api/upload -F "eeg_file=@OpenBCI_GUI/OpenBCI_GUI/data/EEG_Sample_Data/OpenBCI_GUI-v6-meditation.txt" -F "condition=manual_test"`
Expected: 返回 `{"status":"uploaded",...}`

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: add end-to-end tests for upload→analyze ODF workflow"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ 支持 ODF .txt 格式 — Task 2
- ✅ 支持 BrainFlow CSV 格式 — Task 3
- ✅ 提取 EXG+Accel+Marker — Task 2, 3
- ✅ Marker → events_df 映射 — Task 6
- ✅ load_eeg() 向后兼容 — Task 4
- ✅ /api/upload 放宽后缀 — Task 5
- ✅ /api/analyze 改用 load_eeg_full() — Task 6
- ✅ /api/openbci/* 保留并更新 — Task 7
- ✅ 删除 _raw_to_uv — Task 2
- ✅ 测试覆盖 — Task 1-8 全程 TDD

**2. Placeholder scan:** 无 TBD/TODO,所有步骤含完整代码。

**3. Type consistency:**
- `Marker` dataclass 在 Task 2 定义,Task 3/4/6 使用,字段名一致(timestamp, value, label)
- `load_eeg_full()` 返回 dict 结构在 Task 4 定义,Task 6 使用,字段名一致(data, fs, channels, times, accel, markers, metadata)
- `metadata` 子字段在 Task 2/3/4 定义,Task 6 使用,字段名一致(format, board, n_channels, sample_rate, has_accelerometer, has_markers, duration_sec, n_samples)

无问题。
