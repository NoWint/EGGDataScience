"""头皮地形图测试"""
import numpy as np
import pytest


def test_compute_topomap_data_8_channels():
    """测试 8 通道地形图数据生成"""
    from app.analysis.stats_viz import compute_topomap_data, CHANNEL_POSITIONS

    # 验证 8 通道位置已定义
    expected_channels = ['Fp1', 'Fp2', 'C3', 'C4', 'Pz', 'O1', 'O2', 'Fz']
    for ch in expected_channels:
        assert ch in CHANNEL_POSITIONS, f"通道 {ch} 未在 CHANNEL_POSITIONS 中定义"

    # 8 通道值
    values = [10.0, 12.0, 8.0, 9.0, 15.0, 7.0, 6.0, 11.0]
    result = compute_topomap_data(values, expected_channels)

    assert 'grid_x' in result
    assert 'grid_y' in result
    assert 'grid_z' in result
    assert 'channels' in result
    assert 'values' in result

    assert result['channels'] == expected_channels
    assert len(result['values']) == 8
    # grid 是 30x30
    assert len(result['grid_z']) == 30
    assert len(result['grid_z'][0]) == 30


def test_compute_topomap_data_3_channels_still_works():
    """测试原有 3 通道仍兼容"""
    from app.analysis.stats_viz import compute_topomap_data

    values = [10.0, 12.0, 11.0]
    channels = ['Fp1', 'Fp2', 'Fpz']
    result = compute_topomap_data(values, channels)

    assert result['channels'] == channels
    assert len(result['values']) == 3


def test_channel_positions_8ch():
    """测试 8 通道位置在单位圆内"""
    from app.analysis.stats_viz import CHANNEL_POSITIONS

    for name, (x, y) in CHANNEL_POSITIONS.items():
        # 位置应在 [-1, 1] 范围内
        assert -1.0 <= x <= 1.0, f"{name} x={x} 超出范围"
        assert -1.0 <= y <= 1.0, f"{name} y={y} 超出范围"
