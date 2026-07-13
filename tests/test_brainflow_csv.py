"""BrainFlow CSV 导入测试"""
import numpy as np
import pytest
from pathlib import Path

# Desktop 上的 BrainFlow RAW 导出文件(Ganglion 4ch, Tab 分隔, 无表头)
DESKTOP_RAW_4 = Path("/Users/xiatian/Desktop/BrainFlow-RAW_2026-07-13_16-12-59_4.csv")
DESKTOP_RAW_6 = Path("/Users/xiatian/Desktop/BrainFlow-RAW_2026-07-13_16-12-59_6.csv")


def test_detect_brainflow_csv_true(tmp_brainflow_csv):
    """测试检测 BrainFlow CSV 格式(有表头,逗号分隔)"""
    from app.analysis.openbci_import import _detect_brainflow_csv
    assert _detect_brainflow_csv(tmp_brainflow_csv) is True


def test_detect_brainflow_raw_tab_true():
    """测试检测 BrainFlow RAW CSV(Tab 分隔,无表头,Ganglion 15 列)"""
    from app.analysis.openbci_import import _detect_brainflow_csv
    for f in (DESKTOP_RAW_4, DESKTOP_RAW_6):
        if not f.exists():
            pytest.skip(f"样本文件不存在: {f}")
        assert _detect_brainflow_csv(f) is True, f"未能识别 RAW 文件: {f}"


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


def test_load_brainflow_raw_tab_ganglion():
    """测试加载 BrainFlow RAW CSV(Ganglion 4ch, Tab 分隔, 无表头, 15 列)

    验证关键修复:列 0 是 Sample Index,EXG 从列 1 开始(不是列 0)
    """
    from app.analysis.openbci_import import load_brainflow_csv

    for f in (DESKTOP_RAW_4, DESKTOP_RAW_6):
        if not f.exists():
            pytest.skip(f"样本文件不存在: {f}")

        result = load_brainflow_csv(f)

        # Ganglion 4ch
        assert result['data'].shape[1] == 4, f"{f}: EXG 通道数应为 4"
        assert len(result['channels']) == 4
        assert result['channels'] == ['EXG_0', 'EXG_1', 'EXG_2', 'EXG_3']

        # EXG 数据不应包含 Sample Index(列 0)
        # Sample Index 是递增整数(0,1,2,...),EXG 是 μV 信号(有正有负)
        data = result['data']
        # 第一列不应是 Sample Index(不应该是单调递增的非负整数序列)
        col0 = data[:, 0]
        assert not np.all(col0 >= 0) or not np.all(np.diff(col0) >= 0), \
            f"{f}: 第一列看起来是 Sample Index,EXG 提取偏移有误"

        # 板卡识别
        meta = result['metadata']
        assert meta['board'] == 'ganglion', f"{f}: 板卡应为 ganglion"
        assert meta['n_channels'] == 4

        # 采样率(Ganglion 默认 200 Hz)
        assert result['fs'] > 0
        assert meta['sample_rate'] == result['fs']

        # 有时间轴
        assert len(result['times']) == data.shape[0]
        assert result['times'][0] == 0.0  # 从 0 开始

        # Accel 存在(15 列: idx + 4 EXG + 3 accel + 5 other + ts + marker)
        assert result['accel'] is not None
        assert result['accel'].shape[1] == 3


def test_load_brainflow_raw_exg_values_not_sample_index():
    """专项验证:RAW 文件的 EXG 数据与文件中的 Sample Index 列不同"""
    from app.analysis.openbci_import import load_brainflow_csv, _detect_separator
    import pandas as pd

    f = DESKTOP_RAW_4
    if not f.exists():
        pytest.skip(f"样本文件不存在: {f}")

    # 直接读取原始文件,获取列 0 (Sample Index)
    sep = _detect_separator(f)
    raw = pd.read_csv(f, sep=sep, header=None, na_values=['-', '']).dropna().values.astype(np.float64)
    sample_index_col = raw[:, 0]

    # 通过 load_brainflow_csv 加载
    result = load_brainflow_csv(f)
    exg_col0 = result['data'][:, 0]

    # EXG 第一列不应等于 Sample Index 列
    assert not np.allclose(exg_col0, sample_index_col[:len(exg_col0)]), \
        "EXG 第一列与 Sample Index 相同,说明列偏移修复无效"


def test_load_brainflow_csv_header_skipped(tmp_brainflow_csv):
    """验证有表头的 BrainFlow CSV 首行(0,1,2,...)被正确跳过"""
    from app.analysis.openbci_import import load_brainflow_csv

    result = load_brainflow_csv(tmp_brainflow_csv)

    # 表头行被跳过,数据应为 100 行(不是 101 行)
    assert result['data'].shape[0] == 100, "表头行未被跳过"
    assert result['metadata']['n_samples'] == 100
