"""Focus 专注度检测模块
使用 BrainFlow MLModel (ONNX) 进行专注度分类
借鉴 OpenBCI GUI W_Focus 模块
"""
import numpy as np
from typing import Dict, List

try:
    from brainflow import (
        BoardShim, BrainFlowModelParams, BrainFlowMetrics,
        BrainFlowClassifiers, MLModel, DataFilter, LogLevels
    )
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False


def compute_focus_scores(data: np.ndarray, fs: int,
                         window_sec: float = 4.0) -> Dict:
    """用 BrainFlow MLModel 计算专注度分数
    
    参数:
        data: (n_samples, n_channels) EEG 数据,μV
        fs: 采样率
        window_sec: 滑动窗口长度(秒)
    
    返回:
        {
            'scores': List[float],  # 各窗口专注度分数 0-1
            'avg': float,           # 平均专注度
            'stability': float,     # 稳定性(标准差,越小越稳定)
        }
    
    注意:
        - BrainFlow MLModel 要求特定通道数和采样率
        - 数据不足 window_sec 时返回空 scores
        - BrainFlow 未安装时返回空结果
    """
    if not BRAINFLOW_AVAILABLE:
        return {'scores': [], 'avg': 0.0, 'stability': 0.0}
    
    n_samples = len(data)
    window_samples = int(window_sec * fs)
    
    if n_samples < window_samples:
        return {'scores': [], 'avg': 0.0, 'stability': 0.0}
    
    # BrainFlow MLModel 需要 BoardShim.get_board_descr() 兼容的数据格式
    # 这里用 MINDFULNESS metric + DEFAULT_CLASSIFIER classifier
    # (brainflow 5.x 中 CONCENTRATION 已合并为 MINDFULNESS,专注度/正念同源)
    params = BrainFlowModelParams(
        BrainFlowMetrics.MINDFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    model = MLModel(params)
    
    try:
        model.prepare()
        
        scores = []
        # 滑动窗口,步长 = 窗口长度(无重叠)
        for i in range(0, n_samples - window_samples + 1, window_samples):
            segment = data[i:i + window_samples]
            
            # BrainFlow 要求 (n_channels, n_samples) 格式
            # 取前 8 通道(MLModel 训练用 8 通道)
            n_channels = min(8, segment.shape[1])
            segment_t = segment[:, :n_channels].T
            
            # 重采样到 250 Hz(MLModel 训练采样率)
            if fs != 250:
                from scipy import signal as scipy_signal
                new_length = int(len(segment_t[0]) * 250 / fs)
                segment_resampled = np.array([
                    scipy_signal.resample(ch, new_length) for ch in segment_t
                ])
            else:
                segment_resampled = segment_t
            
            try:
                score = model.predict(segment_resampled)
                scores.append(float(score[0]))
            except Exception:
                # 单个窗口预测失败,跳过
                continue
        
        avg = float(np.mean(scores)) if scores else 0.0
        stability = float(np.std(scores)) if scores else 0.0
        
        return {
            'scores': scores,
            'avg': avg,
            'stability': stability,
        }
    finally:
        try:
            model.release()
        except Exception:
            pass
