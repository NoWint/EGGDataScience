"""模块借鉴 API 测试"""
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.server import app
    return TestClient(app)


def test_run_full_pipeline_returns_new_fields(sample_odf_path):
    """测试 run_full_pipeline 返回 topomap/band_powers/spectrogram/focus"""
    from app.analysis import load_eeg_full, run_full_pipeline
    import pandas as pd

    result = load_eeg_full(sample_odf_path)
    data, fs = result['data'], result['fs']

    # 取前 30 秒数据(加快测试)
    n_samples = min(30 * fs, len(data))
    data_short = data[:n_samples]

    events_df = pd.DataFrame([
        ('S0', 0.0), ('F0', 5.0), ('R0', 20.0),
    ], columns=['event_id', 'timestamp'])

    pipeline_result = run_full_pipeline(data_short, fs, events_df)

    # 新增字段
    assert 'topomap_data' in pipeline_result
    assert 'band_powers' in pipeline_result
    assert 'spectrogram_data' in pipeline_result
    assert 'focus_scores' in pipeline_result

    # topomap_data 结构
    topo = pipeline_result['topomap_data']
    assert 'grid_z' in topo
    assert 'channels' in topo

    # band_powers 结构
    bp = pipeline_result['band_powers']
    assert 'delta' in bp or 'alpha' in bp  # 至少有一个频带

    # spectrogram_data 结构
    spec = pipeline_result['spectrogram_data']
    assert 'freqs' in spec or 'sxx' in spec

    # focus_scores 结构
    focus = pipeline_result['focus_scores']
    assert 'scores' in focus
    assert 'avg' in focus
