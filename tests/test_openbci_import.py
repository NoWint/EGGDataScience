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
