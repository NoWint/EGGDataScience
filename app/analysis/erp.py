"""
EEG 事件相关电位 (ERP) 分析模块
提供 epoch 提取、平均叠加、ERP 成分识别、峰值检测、差值波
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# ERP 成分定义（标准认知神经科学）
ERP_COMPONENTS = {
    'N100': {'window': (0.08, 0.14), 'polarity': 'negative', 'desc': '感觉门控, 早期注意'},
    'P200': {'window': (0.15, 0.25), 'polarity': 'positive', 'desc': '特征检测, 注意分配'},
    'N200': {'window': (0.20, 0.35), 'polarity': 'negative', 'desc': '认知控制, 冲突监测'},
    'P300': {'window': (0.30, 0.60), 'polarity': 'positive', 'desc': '工作记忆更新, 靶刺激识别'},
    'N400': {'window': (0.35, 0.55), 'polarity': 'negative', 'desc': '语义处理, 语境整合'},
}


def extract_epochs(data: np.ndarray, fs: int, events_df: pd.DataFrame,
                   event_id: str, pre_stim: float = 1.0,
                   post_stim: float = 2.0) -> Dict:
    """
    提取事件锁定的 epochs

    参数:
        data: (n_samples, n_channels) EEG 数据
        fs: 采样率
        events_df: DataFrame with ['event_id', 'timestamp']
        event_id: 锚定事件 ID (如 'X0')
        pre_stim: 刺激前时长 (秒)
        post_stim: 刺激后时长 (秒)

    返回:
        {'epochs': (n_epochs, n_samples_epoch, n_channels),
         'times': 相对时间轴,
         'n_epochs': int,
         'event_id': str}

    若事件只出现一次(如 X0)，则用滑窗法在事件周围生成多个 epoch（每 10s 一个伪 epoch）
    """
    n_samples, n_channels = data.shape
    pre_samples = int(pre_stim * fs)
    post_samples = int(post_stim * fs)
    epoch_len = pre_samples + post_samples
    times = np.arange(-pre_samples, post_samples) / fs  # 相对时间轴 (秒)

    # 筛选目标事件
    mask = events_df['event_id'] == event_id
    evt_times = events_df.loc[mask, 'timestamp'].values.astype(float)

    # 滑窗伪 epoch 策略：单次事件以事件前后 30s 范围内每 10s 滑窗生成多个 epoch
    if len(evt_times) == 1:
        anchor = evt_times[0]
        window_radius = 30.0  # 事件前后 30s
        step = 10.0
        # 在 [anchor - window_radius, anchor + window_radius] 内每 step 秒取一个锚点
        start_t = anchor - window_radius
        end_t = anchor + window_radius
        pseudo_anchors = np.arange(start_t, end_t + 1e-6, step)
        evt_times = pseudo_anchors
    elif len(evt_times) == 0:
        # 事件未找到，返回空结果
        return {
            'epochs': np.zeros((0, epoch_len, n_channels)),
            'times': times.tolist(),
            'n_epochs': 0,
            'event_id': event_id,
        }

    epochs = []
    for t_anchor in evt_times:
        center_idx = int(round(t_anchor * fs))
        start_idx = center_idx - pre_samples
        end_idx = center_idx + post_samples
        if start_idx < 0 or end_idx > n_samples:
            continue
        epochs.append(data[start_idx:end_idx, :])

    if not epochs:
        return {
            'epochs': np.zeros((0, epoch_len, n_channels)),
            'times': times.tolist(),
            'n_epochs': 0,
            'event_id': event_id,
        }

    epochs_arr = np.array(epochs, dtype=float)  # (n_epochs, epoch_len, n_channels)
    return {
        'epochs': epochs_arr,
        'times': times.tolist(),
        'n_epochs': epochs_arr.shape[0],
        'event_id': event_id,
    }


def average_epochs(epochs: np.ndarray) -> Dict:
    """
    平均叠加 epochs

    参数:
        epochs: (n_epochs, n_samples_epoch, n_channels)

    返回:
        {'waveform': (n_samples_epoch, n_channels), 'n_averaged': int}
    """
    if epochs is None or epochs.size == 0 or epochs.shape[0] == 0:
        n_channels = epochs.shape[2] if epochs is not None and epochs.ndim == 3 else 0
        return {
            'waveform': np.zeros((0, n_channels)),
            'n_averaged': 0,
        }
    waveform = epochs.mean(axis=0)  # (n_samples_epoch, n_channels)
    return {
        'waveform': waveform,
        'n_averaged': int(epochs.shape[0]),
    }


def compute_peak(waveform: np.ndarray, fs: int, window: Tuple[float, float],
                 polarity: str = 'positive') -> Dict:
    """
    在指定时间窗内检测峰值

    参数:
        waveform: (n_samples_epoch, n_channels) 或 (n_samples_epoch,)
        fs: 采样率
        window: (start_sec, end_sec) 相对于刺激的时间窗
        polarity: 'positive' 取最大值, 'negative' 取最小值

    返回:
        {'amplitude': float, 'latency': float (秒), 'index': int}
    """
    if waveform is None or waveform.size == 0:
        return {'amplitude': 0.0, 'latency': 0.0, 'index': 0}

    wf = np.asarray(waveform, dtype=float)
    if wf.ndim == 1:
        wf = wf[:, np.newaxis]

    n_samples, n_channels = wf.shape
    # 假定 waveform 对应时间窗为相对刺激 [-pre, post]
    # 这里依据窗口直接换算样本索引：相对 0 点 = 样本中点（假设传入相对时间窗）
    # 实际约定：window 是相对于刺激 onset 的时间 (秒)
    # waveform 的时间轴需由调用者保证包含 window 范围
    # 这里采用假设：waveform 时间轴 = np.arange(n_samples) / fs - pre_stim
    # 由于此函数无 pre_stim 参数，约定 window 索引按 0 点对齐 waveform 起点
    # 即 waveform[0] 对应 t=0
    start_sec, end_sec = window
    start_idx = max(0, int(round(start_sec * fs)))
    end_idx = min(n_samples, int(round(end_sec * fs)))
    if end_idx <= start_idx:
        return {'amplitude': 0.0, 'latency': 0.0, 'index': 0}

    seg = wf[start_idx:end_idx, :]
    # 多通道平均后做峰值检测
    seg_avg = seg.mean(axis=1)

    if polarity == 'negative':
        local_idx = int(np.argmin(seg_avg))
        amplitude = float(seg_avg[local_idx])
    else:
        local_idx = int(np.argmax(seg_avg))
        amplitude = float(seg_avg[local_idx])

    global_idx = start_idx + local_idx
    latency = global_idx / fs
    return {
        'amplitude': amplitude,
        'latency': float(latency),
        'index': int(global_idx),
    }


def detect_erp_components(waveform: np.ndarray, fs: int) -> List[Dict]:
    """
    自动识别标准 ERP 成分 (N100/P200/N200/P300/N400)

    参数:
        waveform: (n_samples_epoch, n_channels) 平均 ERP 波形
        fs: 采样率

    返回:
        [{'name': 'N100', 'amplitude': ..., 'latency': ..., 'desc': ...}, ...]
        对每个成分在其时间窗内检测峰值
    """
    if waveform is None or waveform.size == 0:
        return []

    results = []
    for name, info in ERP_COMPONENTS.items():
        peak = compute_peak(waveform, fs, info['window'], polarity=info['polarity'])
        results.append({
            'name': name,
            'amplitude': peak['amplitude'],
            'latency': peak['latency'],
            'window': list(info['window']),
            'polarity': info['polarity'],
            'desc': info['desc'],
        })
    return results


def compute_difference_wave(erp_a: np.ndarray, erp_b: np.ndarray) -> Dict:
    """
    计算两个 ERP 的差值波

    参数:
        erp_a: (n_samples, n_channels) ERP A 波形
        erp_b: (n_samples, n_channels) ERP B 波形

    返回:
        {'times': ..., 'diff': (n_samples, n_channels)}
    """
    a = np.asarray(erp_a, dtype=float)
    b = np.asarray(erp_b, dtype=float)
    if a.shape != b.shape:
        # 长度不一致时截断到较短
        n_min = min(a.shape[0], b.shape[0])
        a = a[:n_min, :]
        b = b[:n_min, :]
    diff = a - b
    times = np.arange(diff.shape[0]) / 1.0  # 单位由调用者上下文决定
    return {
        'times': times.tolist(),
        'diff': diff,
    }


def compute_peak_to_peak(waveform: np.ndarray) -> List[float]:
    """
    计算峰峰值 (max - min)

    参数:
        waveform: (n_samples, n_channels)

    返回:
        各通道的峰峰值列表
    """
    if waveform is None or waveform.size == 0:
        return []
    wf = np.asarray(waveform, dtype=float)
    if wf.ndim == 1:
        wf = wf[:, np.newaxis]
    ptp = (wf.max(axis=0) - wf.min(axis=0)).tolist()
    return [float(v) for v in ptp]


def compute_rmse(epochs: np.ndarray, avg: np.ndarray) -> float:
    """
    计算各 epoch 相对平均的 RMSE（残差噪声估计）

    参数:
        epochs: (n_epochs, n_samples_epoch, n_channels)
        avg: (n_samples_epoch, n_channels) 平均波形

    返回:
        float RMSE
    """
    if epochs is None or epochs.size == 0 or avg is None or avg.size == 0:
        return 0.0
    eps = np.asarray(epochs, dtype=float)
    av = np.asarray(avg, dtype=float)
    if eps.ndim == 2:
        eps = eps[np.newaxis, :, :]
    if av.ndim == 1:
        av = av[:, np.newaxis]
    residuals = eps - av[np.newaxis, :, :]
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return rmse


def run_erp_analysis(data: np.ndarray, fs: int, events_df: pd.DataFrame,
                     event_id: str = 'X0', pre_stim: float = 1.0,
                     post_stim: float = 2.0) -> Dict:
    """
    完整 ERP 分析流水线

    返回:
        {
            'event_id': ...,
            'epochs': {'n_epochs': ..., 'times': ...},  # 不返回完整 epochs 数据
            'averaged': {'waveform': [[...], ...], 'times': [...]},  # 转为 list
            'components': [...],  # ERP 成分识别结果
            'peak_to_peak': [...],
            'rmse': float,
            'n_channels': int,
            'fs': fs,
        }
    """
    n_samples, n_channels = data.shape

    # 1. 提取 epochs
    epochs_result = extract_epochs(data, fs, events_df, event_id, pre_stim, post_stim)
    epochs = epochs_result['epochs']
    times = epochs_result['times']
    n_epochs = epochs_result['n_epochs']

    # 2. 平均叠加
    avg_result = average_epochs(epochs)
    waveform = avg_result['waveform']
    n_averaged = avg_result['n_averaged']

    # 3. ERP 成分识别 (成分时间窗相对刺激 onset，取 post-stim 段)
    pre_samples = int(pre_stim * fs)
    waveform_post = waveform[pre_samples:, :] if waveform.size > 0 else waveform
    components = detect_erp_components(waveform_post, fs)

    # 4. 峰峰值
    ptp = compute_peak_to_peak(waveform)

    # 5. RMSE
    rmse = compute_rmse(epochs, waveform)

    return {
        'event_id': event_id,
        'epochs': {
            'n_epochs': n_epochs,
            'times': times,
            'n_averaged': n_averaged,
        },
        'averaged': {
            'waveform': waveform.tolist(),
            'times': times,
        },
        'components': components,
        'peak_to_peak': ptp,
        'rmse': rmse,
        'n_channels': int(n_channels),
        'fs': fs,
    }
