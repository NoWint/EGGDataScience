"""
OpenBCI CSV 导入兼容模块
支持 OpenBCI_GUI 导出的 .txt/.csv 文件自动解析

OpenBCI 导出格式 (Cyton 8ch):
  %OpenBCI Raw EEG Data
  %Number of channels = 8
  %Sample Rate = 250 Hz
  %Board = OpenBCI_GUI$BoardCytonSerial
  Sample Index, EXG Channel 0, ..., EXG Channel 7, Accel Channel 0-2,
  Other x7, Analog Channel 0-2, Timestamp, Other, Timestamp (Formatted)

Ganglion 4ch 同理，列数更少。
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Marker:
    """OpenBCI 事件标记"""
    timestamp: float   # 秒
    value: int         # 原始 marker 值
    label: str         # "marker_{value}"


def _detect_openbci(filepath: Path) -> bool:
    """检测文件是否为 OpenBCI 导出格式"""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first = f.readline()
        return first.startswith("%OpenBCI")
    except Exception:
        return False


def _detect_brainflow_csv(filepath: Path) -> bool:
    """检测文件是否为 BrainFlow CSV 导出格式

    支持两种 BrainFlow CSV 变体:
    1. 有表头(列名为纯数字索引 0,1,2,...),逗号分隔
    2. 无表头(Tab 分隔的 RAW 数据,如 BrainFlow-RAW_*.csv)

    识别条件:
    - 不以 % 开头(区别于 ODF)
    - 列数 >= 10(区别于普通 CSV)
    - 首行全部为数字(浮点或整数)
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first = f.readline().strip()
        if first.startswith("%"):
            return False

        # 尝试 Tab 和逗号两种分隔符
        for sep in ('\t', ','):
            cols = [c.strip() for c in first.split(sep)]
            if len(cols) < 10:
                continue
            # 检查是否全部为数字(支持浮点和整数)
            all_numeric = True
            for c in cols:
                try:
                    float(c)
                except ValueError:
                    all_numeric = False
                    break
            if all_numeric:
                return True

        return False
    except Exception:
        return False


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

    # 时间轴(用 Timestamp 计算,更准确)
    fs = info["sample_rate"]
    if info["timestamp_col"] is not None:
        ts_col_pos = col_positions[info["timestamp_col"]]
        timestamps_sec = values[:, ts_col_pos] / 1000.0  # ms → s
        times = timestamps_sec - timestamps_sec[0]  # 从 0 开始
    else:
        sample_idx = values[:, col_positions[info["sample_idx_col"]]]
        times = (sample_idx.astype(float) - sample_idx[0]) / fs

    # 提取 Marker,构建 Marker 列表
    markers = None
    if info["marker_col"] is not None:
        marker_col_pos = col_positions[info["marker_col"]]
        marker_values = values[:, marker_col_pos]
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


# BrainFlow CSV 列数 → 板卡/EXG 通道数映射
# 支持两种导出格式:
# - 有表头(逗号分隔): 24=Cyton, 28=Daisy, 18=Ganglion
# - 无表头(Tab 分隔 RAW): 15=Ganglion, 22=Cyton, 30=Daisy
BRAINFLOW_COLUMN_MAP = {
    24: ("cyton", 8),     # Cyton 8ch (有表头)
    28: ("daisy", 16),    # Cyton 16ch (Daisy, 有表头)
    18: ("ganglion", 4),  # Ganglion 4ch (有表头)
    15: ("ganglion", 4),  # Ganglion 4ch (RAW Tab 分隔: idx + 4 EXG + 3 accel + 5 other + ts + marker)
    22: ("cyton", 8),     # Cyton 8ch (RAW Tab 分隔)
    30: ("daisy", 16),    # Daisy 16ch (RAW Tab 分隔)
}


def _detect_separator(filepath: Path) -> str:
    """检测 BrainFlow CSV 文件的分隔符(Tab 或逗号)"""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first = f.readline()
        if '\t' in first:
            return '\t'
        return ','
    except Exception:
        return ','


def _is_brainflow_header_row(row_values: np.ndarray) -> bool:
    """检测首行是否为 BrainFlow 数字表头 (0, 1, 2, ..., n-1)"""
    if len(row_values) < 2:
        return False
    try:
        for i, v in enumerate(row_values):
            if float(v) != float(i):
                return False
        return True
    except (ValueError, TypeError):
        return False


def load_brainflow_csv(filepath: Path) -> dict:
    """加载 BrainFlow CSV 导出文件,返回统一 dict

    支持两种格式:
    1. 有表头(列名为纯数字 0,1,2,...),逗号分隔
    2. 无表头(Tab 分隔 RAW,如 BrainFlow-RAW_*.csv)

    BrainFlow CSV 列布局(两种格式相同):
    - 第 0 列: Sample Index
    - 第 1..N 列: EXG 通道(N = EXG 通道数)
    - 第 N+1..N+3 列: Accel XYZ
    - ...: Other
    - 倒数第二列: Timestamp(Unix 秒)
    - 最后一列: Marker
    """
    sep = _detect_separator(filepath)

    # 读取文件(先按无表头读取,后续检测是否需要跳过首行)
    # na_values: 处理可能存在的非数字值(如 '-', 空字符串, BrainFlow 导出末尾截断)
    # dropna: 丢弃含 NaN 的行(通常是文件末尾不完整行)
    df = pd.read_csv(filepath, sep=sep, header=None, na_values=['-', ''])
    df = df.dropna()
    values = df.values.astype(np.float64)
    n_samples, n_cols = values.shape

    # 检测并跳过数字表头行 (0, 1, 2, ...)
    if n_samples > 1 and _is_brainflow_header_row(values[0]):
        values = values[1:]
        n_samples = values.shape[0]

    # 按列数判断板卡
    board, n_exg = BRAINFLOW_COLUMN_MAP.get(n_cols, ("unknown", max(1, n_cols // 4)))

    # EXG 通道: 列 1 到 n_exg+1 (列 0 是 Sample Index)
    exg_start = 1
    data = values[:, exg_start:exg_start + n_exg]
    channels = [f"EXG_{i}" for i in range(n_exg)]

    # Accel 通道: EXG 后 3 列(跳过 Sample Index 偏移)
    accel = None
    accel_start = exg_start + n_exg
    if n_cols >= accel_start + 3:
        accel = values[:, accel_start:accel_start + 3]

    # Timestamp(倒数第二列,Unix 秒)
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


def openbci_info(filepath: Path) -> dict:
    """仅读取 OpenBCI 文件的元信息 (不加载全部数据)"""
    info = _parse_header(filepath)
    file_size = filepath.stat().st_size

    # 估算时长: 文件行数 × 数据行占比
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        total_lines = sum(1 for _ in f)
    data_lines = total_lines - info["header_lines"]
    duration_sec = data_lines / info["sample_rate"] if info["sample_rate"] > 0 else 0

    return {
        "board": info["board"],
        "n_channels": info["n_channels"],
        "sample_rate": info["sample_rate"],
        "exg_channels": info["exg_names"],
        "has_accelerometer": info["has_accel"],
        "has_analog": info["has_analog"],
        "file_size_kb": round(file_size / 1024, 1),
        "duration_sec": round(duration_sec, 1),
        "duration_min": round(duration_sec / 60, 1),
        "total_columns": info["total_columns"],
        "header_lines": info["header_lines"],
        "estimated_samples": data_lines,
        "format": "openbci",
    }
