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
