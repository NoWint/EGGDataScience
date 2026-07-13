"""Focus 专注度检测测试"""
import numpy as np
import pytest


def test_compute_focus_scores_returns_dict(sample_odf_path):
    """测试 compute_focus_scores 返回正确格式"""
    from app.analysis.focus import compute_focus_scores
    from app.analysis import load_eeg_full
    
    result = load_eeg_full(sample_odf_path)
    data, fs = result['data'], result['fs']
    
    # 取前 20 秒数据(MLModel 需要足够长度)
    n_samples = min(20 * fs, len(data))
    focus_result = compute_focus_scores(data[:n_samples], fs)
    
    assert isinstance(focus_result, dict)
    assert 'scores' in focus_result
    assert 'avg' in focus_result
    assert 'stability' in focus_result
    
    # scores 是列表
    assert isinstance(focus_result['scores'], list)
    # avg 在 0-1 之间
    assert 0.0 <= focus_result['avg'] <= 1.0
    # stability >= 0
    assert focus_result['stability'] >= 0.0


def test_compute_focus_scores_short_data(sample_odf_path):
    """测试数据太短时返回空 scores"""
    from app.analysis.focus import compute_focus_scores
    from app.analysis import load_eeg_full
    
    result = load_eeg_full(sample_odf_path)
    data, fs = result['data'], result['fs']
    
    # 只给 1 秒数据(不足 4 秒窗口)
    focus_result = compute_focus_scores(data[:fs], fs)
    
    assert focus_result['scores'] == []
    assert focus_result['avg'] == 0.0
