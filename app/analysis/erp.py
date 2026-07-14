"""
EEG 事件相关电位 (ERP) 分析模块
提供 epoch 提取、基线校正、平均叠加、ERP 成分识别、峰值检测、差值波
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, argrelextrema
from typing import Dict, List, Tuple, Optional


# ERP 成分定义（标准认知神经科学）
ERP_COMPONENTS = {
    'P50':  {'window': (0.04, 0.08), 'polarity': 'positive', 'desc': '感觉门控, 早期感觉处理'},
    'N100': {'window': (0.08, 0.14), 'polarity': 'negative', 'desc': '感觉门控, 早期注意'},
    'N170': {'window': (0.14, 0.20), 'polarity': 'negative', 'desc': '面部/视觉特征处理'},
    'P200': {'window': (0.15, 0.25), 'polarity': 'positive', 'desc': '特征检测, 注意分配'},
    'N200': {'window': (0.20, 0.35), 'polarity': 'negative', 'desc': '认知控制, 冲突监测'},
    'P3a':  {'window': (0.25, 0.40), 'polarity': 'positive', 'desc': '新异刺激定向反应'},
    'P300': {'window': (0.30, 0.60), 'polarity': 'positive', 'desc': '工作记忆更新, 靶刺激识别'},
    'P3b':  {'window': (0.35, 0.65), 'polarity': 'positive', 'desc': '记忆更新（替代原 P300）'},
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
    if fs <= 0:
        raise ValueError(f"采样率 fs 必须为正数, 收到 fs={fs}")
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


def baseline_correct(epochs: np.ndarray, fs: int,
                     baseline_start: float = -1.0,
                     baseline_end: float = -0.2,
                     pre_stim: Optional[float] = None) -> np.ndarray:
    """
    基线校正：减去 pre-stim 基线时段的均值

    参数:
        epochs: (n_epochs, n_samples_epoch, n_channels) 或 (n_samples_epoch, n_channels)
        fs: 采样率
        baseline_start: 基线窗口起始 (秒, 相对刺激 onset, 负值=刺激前)
        baseline_end: 基线窗口结束 (秒, 相对刺激 onset)
        pre_stim: 刺激前时长 (秒)；若为 None 则推断为 -baseline_start

    返回:
        校正后的 epochs (与输入同形状)
    """
    if epochs is None or epochs.size == 0:
        return epochs

    eps = np.asarray(epochs, dtype=float)
    if eps.ndim == 2:
        eps = eps[np.newaxis, :, :]  # 单 epoch → (1, n_samples, n_channels)
    n_epochs, n_samples, n_channels = eps.shape

    # 推断 pre_stim（epoch 起点对应 t = -pre_stim）
    if pre_stim is None:
        pre_stim = -baseline_start if baseline_start < 0 else 0.0

    # 基线窗口 → 样本索引 (epoch 时间轴: t[i] = i/fs - pre_stim)
    start_idx = max(0, int(round((baseline_start + pre_stim) * fs)))
    end_idx = min(n_samples, int(round((baseline_end + pre_stim) * fs)))
    if end_idx <= start_idx:
        return epochs  # 基线窗口无效，不校正

    # 各 epoch 各通道的基线均值 → 广播减去
    baseline_mean = eps[:, start_idx:end_idx, :].mean(axis=1)  # (n_epochs, n_channels)
    corrected = eps - baseline_mean[:, np.newaxis, :]
    return corrected


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
    在指定时间窗内检测峰值（含 Savitzky-Golay 平滑与局部极值检测）

    参数:
        waveform: (n_samples_epoch, n_channels) 或 (n_samples_epoch,)
            约定 waveform[0] 对应 t=0 (刺激 onset)
        fs: 采样率
        window: (start_sec, end_sec) 相对于刺激 onset 的时间窗
        polarity: 'positive' 取最大值, 'negative' 取最小值

    返回:
        {'amplitude': float, 'latency': float (秒), 'index': int,
         'significance': float (峰值 z-score 相对窗口内信号)}
    """
    if waveform is None or waveform.size == 0:
        return {'amplitude': 0.0, 'latency': 0.0, 'index': 0, 'significance': 0.0}

    wf = np.asarray(waveform, dtype=float)
    if wf.ndim == 1:
        wf = wf[:, np.newaxis]

    n_samples, n_channels = wf.shape
    start_sec, end_sec = window
    start_idx = max(0, int(round(start_sec * fs)))
    end_idx = min(n_samples, int(round(end_sec * fs)))
    if end_idx <= start_idx:
        return {'amplitude': 0.0, 'latency': 0.0, 'index': 0, 'significance': 0.0}

    seg = wf[start_idx:end_idx, :]
    # 多通道平均后做峰值检测
    seg_avg = seg.mean(axis=1)

    # Savitzky-Golay 平滑 (window=5, order=2)
    seg_len = len(seg_avg)
    if seg_len >= 5:
        seg_smooth = savgol_filter(seg_avg, window_length=5, polyorder=2)
    else:
        seg_smooth = seg_avg

    # 局部极值检测 (argrelextrema)，在时间窗内找最显著的峰
    if polarity == 'negative':
        extrema_idx = argrelextrema(seg_smooth, np.less)[0]
        if len(extrema_idx) > 0:
            local_idx = int(extrema_idx[np.argmin(seg_smooth[extrema_idx])])
        else:
            local_idx = int(np.argmin(seg_smooth))
        amplitude = float(seg_smooth[local_idx])
    else:
        extrema_idx = argrelextrema(seg_smooth, np.greater)[0]
        if len(extrema_idx) > 0:
            local_idx = int(extrema_idx[np.argmax(seg_smooth[extrema_idx])])
        else:
            local_idx = int(np.argmax(seg_smooth))
        amplitude = float(seg_smooth[local_idx])

    global_idx = start_idx + local_idx
    latency = global_idx / fs

    # 显著性: 峰值相对窗口内信号的 z-score
    seg_mean = float(np.mean(seg_smooth))
    seg_std = float(np.std(seg_smooth))
    significance = (amplitude - seg_mean) / (seg_std + 1e-12)

    return {
        'amplitude': amplitude,
        'latency': float(latency),
        'index': int(global_idx),
        'significance': float(significance),
    }


def detect_erp_components(waveform: np.ndarray, fs: int) -> List[Dict]:
    """
    自动识别标准 ERP 成分 (P50/N100/N170/P200/N200/P3a/P300/P3b/N400)

    参数:
        waveform: (n_samples_epoch, n_channels) 平均 ERP 波形
            约定 waveform[0] 对应 t=0 (刺激 onset)
        fs: 采样率

    返回:
        [{'name': 'N100', 'amplitude': ..., 'latency': ..., 'significance': ...,
          'window': [...], 'polarity': ..., 'desc': ...}, ...]
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
            'significance': peak['significance'],
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


def compute_channel_diff(waveform: np.ndarray,
                         channel_names: Tuple[str, ...] = ('Fp1', 'Fp2', 'Fpz')) -> Dict:
    """
    计算通道间差异波形与偏侧化指数

    参数:
        waveform: (n_samples, n_channels) ERP 波形
        channel_names: 通道名元组（长度应与 n_channels 一致）

    返回:
        {'pairs': [(label, diff_list), ...],
         'lateralization_index': float}
        lateralization_index 基于前两个通道 (通常 Fp1/Fp2) 的不对称性:
        mean(|L-R|) / (mean(|L|) + mean(|R|) + eps)
    """
    if waveform is None or waveform.size == 0:
        return {'pairs': [], 'lateralization_index': 0.0}

    wf = np.asarray(waveform, dtype=float)
    if wf.ndim == 1:
        wf = wf[:, np.newaxis]
    n_samples, n_channels = wf.shape

    names = list(channel_names)
    while len(names) < n_channels:
        names.append(f'Ch{len(names)}')
    names = names[:n_channels]

    # 所有通道对之间的差异波形
    pairs = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            label = f"{names[i]}-{names[j]}"
            diff = (wf[:, i] - wf[:, j]).tolist()
            pairs.append((label, diff))

    # 偏侧化指数: 基于前两个通道（通常 Fp1, Fp2）的归一化不对称性
    if n_channels >= 2:
        left = wf[:, 0]
        right = wf[:, 1]
        denom = np.mean(np.abs(left)) + np.mean(np.abs(right)) + 1e-12
        lateralization_index = float(np.mean(np.abs(left - right)) / denom)
    else:
        lateralization_index = 0.0

    return {
        'pairs': pairs,
        'lateralization_index': lateralization_index,
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
                     post_stim: float = 2.0,
                     baseline_correction: bool = True) -> Dict:
    """
    完整 ERP 分析流水线

    参数:
        baseline_correction: 是否启用基线校正（默认 True）

    返回:
        {
            'event_id': ...,
            'epochs': {'n_epochs': ..., 'times': ...},
            'averaged': {'waveform': [[...], ...], 'times': [...]},
            'components': [...],  # ERP 成分识别结果（含 significance）
            'peak_to_peak': [...],
            'rmse': float,
            'n_channels': int,
            'fs': fs,
            'baseline_corrected': bool,
        }
    """
    n_samples, n_channels = data.shape

    # 1. 提取 epochs
    epochs_result = extract_epochs(data, fs, events_df, event_id, pre_stim, post_stim)
    epochs = epochs_result['epochs']
    times = epochs_result['times']
    n_epochs = epochs_result['n_epochs']

    # 2. 基线校正 (在 extract_epochs 后、average_epochs 前)
    applied_baseline = False
    if baseline_correction and n_epochs > 0:
        epochs = baseline_correct(epochs, fs,
                                  baseline_start=-pre_stim,
                                  baseline_end=-0.2,
                                  pre_stim=pre_stim)
        applied_baseline = True

    # 3. 平均叠加
    avg_result = average_epochs(epochs)
    waveform = avg_result['waveform']
    n_averaged = avg_result['n_averaged']

    # 4. ERP 成分识别 (成分时间窗相对刺激 onset，取 post-stim 段)
    pre_samples = int(pre_stim * fs)
    waveform_post = waveform[pre_samples:, :] if waveform.size > 0 else waveform
    components = detect_erp_components(waveform_post, fs)

    # 5. 峰峰值
    ptp = compute_peak_to_peak(waveform)

    # 6. RMSE
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
        'baseline_corrected': applied_baseline,
    }
