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
