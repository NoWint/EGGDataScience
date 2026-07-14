"""测试 _run_all_modules 共享分析函数"""
import numpy as np
from app.server import _run_all_modules


def test_run_all_modules_returns_5_modules():
    """_run_all_modules 应返回包含 5 个模块结果的字典"""
    fs = 250
    duration = 60  # 1 分钟
    n_samples = fs * duration
    t = np.linspace(0, duration, n_samples)
    data = np.sin(2 * np.pi * 10 * t).reshape(1, -1) * 50
    data = np.vstack([data] * 4).T  # 4 通道, (n_samples, n_channels)

    import pandas as pd
    events_df = pd.DataFrame([
        ('S0', 0.0), ('F0', 5.0), ('F1', 30.0),
        ('X0', 30.0), ('X1', 35.0),
        ('R0', 35.0), ('R1', 60.0),
    ], columns=['event_id', 'timestamp'])

    result = _run_all_modules(data, fs, events_df)

    assert 'flow_recovery' in result
    assert 'spectrum' in result
    assert 'erp' in result
    assert 'ersp' in result
    assert 'artifact' in result
    for mod in ['flow_recovery', 'spectrum', 'erp', 'ersp', 'artifact']:
        assert 'error' not in result[mod], f"{mod} 出错: {result[mod].get('error')}"
