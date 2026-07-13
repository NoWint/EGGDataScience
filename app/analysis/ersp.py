"""
EEG 事件相关谱扰动 (ERSP) 与时频进阶分析模块
提供连续小波变换、ERSP/ERD 计算、跨频率耦合 (PAC)、跨试验相位一致性 (ITPC)
"""
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.fft import next_fast_len
from typing import Dict, List, Tuple, Optional


# 频段定义（与 spectrum.py / flow_recovery.py 一致）
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}


# ========== 1. 连续小波变换 (CWT) ==========
def _morlet_wavelet(freq: float, fs: int, wavelet_width: float = 5.0,
                    n_samples: Optional[int] = None) -> np.ndarray:
    """
    构造复 Morlet 小波 (中心频率 freq, 采样率 fs)

    Morlet 小波: w(t) = exp(-t^2 / (2*sigma_t^2)) * exp(2j*pi*f*t)
    其中 sigma_t = wavelet_width / (2*pi*f), 保证时频分辨率权衡
    返回复数小波 (长度自适应, 支撑约 ±3*sigma_t)
    """
    # 时间域标准差
    sigma_t = wavelet_width / (2.0 * np.pi * freq)
    # 小波支撑范围 (±3 sigma, 但确保至少 5 个采样点)
    half_len = max(int(np.ceil(3 * sigma_t * fs)), 5)
    if n_samples is not None:
        # 与信号长度一致以便 FFT 卷积 (推荐)
        half_len = n_samples // 2
    t = np.arange(-half_len, half_len + 1) / fs
    # 高斯包络 × 复指数
    gaussian = np.exp(-t ** 2 / (2 * sigma_t ** 2))
    wavelet = gaussian * np.exp(2j * np.pi * freq * t)
    # 能量归一化, 使功率估计无偏
    wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet) ** 2) + 1e-12)
    return wavelet


def compute_cwt(data, fs, freqs=None, wavelet_width=5.0):
    """
    连续小波变换 (Morlet 小波, 复小波, 基于 FFT 卷积加速)

    参数:
        data: (n_samples,) 或 (n_samples, n_channels)
        fs: 采样率
        freqs: 频率数组, 默认 np.linspace(1, 45, 45)
        wavelet_width: 小波宽度参数 (默认 5.0 = 周期数)

    返回:
        {'power': (len(freqs), n_samples) 或 (n_channels, len(freqs), n_samples),
         'phase': 同形状相位矩阵,
         'freqs': [...]}
    """
    if freqs is None:
        freqs = np.linspace(1, 45, 45)
    freqs = np.asarray(freqs, dtype=float)

    data = np.asarray(data, dtype=float)
    single_channel = False
    if data.ndim == 1:
        data = data[:, np.newaxis]
        single_channel = True
    n_samples, n_channels = data.shape

    n_freqs = len(freqs)
    power = np.zeros((n_channels, n_freqs, n_samples))
    phase = np.zeros((n_channels, n_freqs, n_samples))

    # FFT 预计算 (复小波需用复 FFT, 信号虽实但统一用 fft 简化)
    n_fft = next_fast_len(n_samples)
    for ch in range(n_channels):
        sig_fft = np.fft.fft(data[:, ch], n=n_fft)
        for fi, freq in enumerate(freqs):
            wavelet = _morlet_wavelet(freq, fs, wavelet_width, n_samples=n_samples)
            w_fft = np.fft.fft(wavelet, n=n_fft)
            conv = np.fft.ifft(sig_fft * w_fft)[:n_samples]
            power[ch, fi, :] = np.abs(conv) ** 2
            phase[ch, fi, :] = np.angle(conv)

    if single_channel:
        power = power[0]
        phase = phase[0]

    return {
        'power': power,
        'phase': phase,
        'freqs': freqs.tolist(),
    }


# ========== 2. 事件相关谱扰动 (ERSP) ==========
def _extract_epochs(data: np.ndarray, fs: int, events_df: pd.DataFrame,
                    event_id: str, pre_stim: float, post_stim: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取事件锁定的 epochs

    返回:
        epochs: (n_epochs, n_samples_epoch, n_channels)
        times: 相对事件的时间轴 (秒)
    若事件只出现一次, 用滑窗法在事件周围生成伪 epochs
    """
    n_pre = int(pre_stim * fs)
    n_post = int(post_stim * fs)
    n_epoch = n_pre + n_post
    times = np.arange(-n_pre, n_post) / fs

    # 找事件索引
    mask = events_df['event_id'] == event_id
    event_times = events_df.loc[mask, 'timestamp'].values
    n_samples, n_channels = data.shape

    event_indices = (event_times * fs).astype(int)
    event_indices = event_indices[
        (event_indices - n_pre >= 0) & (event_indices + n_post <= n_samples)
    ]

    if len(event_indices) >= 2:
        # 多次事件: 直接抽取
        epochs = np.array([data[idx - n_pre:idx + n_post] for idx in event_indices])
    else:
        # 单次或无事件: 滑窗伪 epochs (步长 = 1/4 epoch 长度)
        if len(event_indices) == 0:
            # 找不到事件, 用数据中点作为锚点
            anchor = n_samples // 2
        else:
            anchor = event_indices[0]
        step = max(n_epoch // 4, 1)
        # 在锚点 ±2s 范围内滑动
        offsets = []
        max_offset = min(int(2.0 * fs), n_epoch)
        for off in range(-max_offset, max_offset + 1, step):
            start = anchor + off - n_pre
            end = start + n_epoch
            if start >= 0 and end <= n_samples:
                offsets.append(off)
        if not offsets:
            offsets = [0]
        epochs = np.array([
            data[anchor + off - n_pre:anchor + off + n_post] for off in offsets
        ])

    return epochs, times


def compute_ersp(data, fs, events_df, event_id='X0', pre_stim=2.0, post_stim=5.0,
                 freqs=None, baseline_start=-2.0, baseline_end=-0.2):
    """
    事件相关谱扰动 (ERSP)

    流程:
        1. 提取事件锁定的 epochs
        2. 对每个 epoch 做 CWT 得到时频功率
        3. 平均各 epoch 的功率
        4. 用基线期功率归一化: ERSP = 10*log10(power / baseline_mean)

    返回:
        {'times': [...], 'freqs': [...], 'ersp': (n_freqs, n_times) dB 矩阵,
         'n_epochs': int}

    若事件只出现一次, 用滑窗法在事件周围生成伪 epochs
    """
    if freqs is None:
        freqs = np.linspace(1, 45, 45)
    freqs = np.asarray(freqs, dtype=float)

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_samples, n_channels = data.shape

    # 1. 提取 epochs
    epochs, times = _extract_epochs(data, fs, events_df, event_id,
                                    pre_stim, post_stim)
    n_epochs, n_epoch_samples, _ = epochs.shape

    # 2. 对每个 epoch 做 CWT, 多通道平均
    power_avg = None
    for ep in range(n_epochs):
        # epochs[ep] shape: (n_epoch_samples, n_channels)
        cwt_result = compute_cwt(epochs[ep], fs, freqs=freqs)
        # 多通道平均功率: (n_freqs, n_times)
        power_ep = cwt_result['power'].mean(axis=0) if cwt_result['power'].ndim == 3 \
            else cwt_result['power']
        if power_avg is None:
            power_avg = power_ep
        else:
            power_avg += power_ep
    power_avg /= n_epochs

    # 3. 基线归一化
    bl_mask = (times >= baseline_start) & (times <= baseline_end)
    if bl_mask.sum() < 1:
        # 退路: 使用前 10% 时间窗作基线
        bl_mask = np.arange(n_epoch_samples) < max(n_epoch_samples // 10, 1)
    baseline_mean = power_avg[:, bl_mask].mean(axis=1, keepdims=True)
    baseline_mean = np.maximum(baseline_mean, 1e-12)

    ersp = 10 * np.log10(power_avg / baseline_mean)

    return {
        'times': times.tolist(),
        'freqs': freqs.tolist(),
        'ersp': ersp.tolist(),
        'n_epochs': int(n_epochs),
    }


# ========== 3. ERD / ERS 分类 ==========
def compute_erd_ers(ersp_matrix, threshold=0):
    """
    将 ERSP 分类为 ERD (去同步, 负值) 和 ERS (同步, 正值)

    参数:
        ersp_matrix: (n_freqs, n_times) dB 矩阵, 或嵌套 list
        threshold: 分类阈值 (dB), 默认 0

    返回:
        {'erd_mask': [[bool]], 'ers_mask': [[bool]],
         'erd_ratio': float, 'ers_ratio': float}
    """
    ersp = np.asarray(ersp_matrix, dtype=float)
    erd_mask = ersp < -abs(threshold)
    ers_mask = ersp > abs(threshold)
    total = ersp.size
    erd_ratio = float(erd_mask.sum()) / total if total > 0 else 0.0
    ers_ratio = float(ers_mask.sum()) / total if total > 0 else 0.0
    return {
        'erd_mask': erd_mask.tolist(),
        'ers_mask': ers_mask.tolist(),
        'erd_ratio': erd_ratio,
        'ers_ratio': ers_ratio,
    }


# ========== 4. 跨频率相位-振幅耦合 (PAC) ==========
def _bandpass_filter(sig: np.ndarray, fs: int, flo: float, fhi: float) -> np.ndarray:
    """Butterworth 带通滤波 (4 阶, 零相位)"""
    nyq = fs / 2.0
    wn = [flo / nyq, fhi / nyq]
    b, a = scipy_signal.butter(4, wn, btype='band')
    return scipy_signal.filtfilt(b, a, sig)


def compute_pac(signal_one_channel, fs, phase_band=(4, 8), amp_band=(30, 45)):
    """
    跨频率相位-振幅耦合 (PAC)

    流程:
        1. 用 Hilbert 变换提取低频相位 (phase_band)
        2. 用 Hilbert 变换提取高频振幅 (amp_band)
        3. 计算相位-振幅调制指数 (MI) — 基于相位 bin 分布的 Shannon 熵

    参数:
        signal_one_channel: (n_samples,) 单通道信号
        fs: 采样率
        phase_band: 相位提供频段 (默认 theta 4-8 Hz)
        amp_band: 振幅提供频段 (默认 gamma 30-45 Hz)

    返回:
        {'mi': float, 'phase_band': [lo, hi], 'amp_band': [lo, hi],
         'phase_amp_dist': 各相位 bin 的平均振幅列表 (用于可视化),
         'n_bins': int}
    """
    sig = np.asarray(signal_one_channel, dtype=float).ravel()
    n_bins = 18  # 相位分箱数

    # 1. 低频相位
    phase_sig = _bandpass_filter(sig, fs, phase_band[0], phase_band[1])
    phase_analytic = scipy_signal.hilbert(phase_sig)
    phase = np.angle(phase_analytic)

    # 2. 高频振幅
    amp_sig = _bandpass_filter(sig, fs, amp_band[0], amp_band[1])
    amp_analytic = scipy_signal.hilbert(amp_sig)
    amplitude = np.abs(amp_analytic)

    # 3. 调制指数 (MI) — Kullback-Leibler 散度法
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean_amp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.any():
            mean_amp[i] = amplitude[mask].mean()
    # 归一化为概率分布
    p = mean_amp / (mean_amp.sum() + 1e-12)
    # 与均匀分布比较, 得 MI
    uniform = 1.0 / n_bins
    kl_div = np.sum(p[p > 0] * np.log(p[p > 0] / uniform))
    mi = kl_div / np.log(n_bins)  # 归一化到 [0, 1]

    return {
        'mi': float(mi),
        'phase_band': [float(phase_band[0]), float(phase_band[1])],
        'amp_band': [float(amp_band[0]), float(amp_band[1])],
        'phase_amp_dist': mean_amp.tolist(),
        'phase_bins': bin_centers.tolist(),
        'n_bins': int(n_bins),
    }


# ========== 5. 跨试验相位一致性 (ITPC) ==========
def compute_itpc(data, fs, events_df, event_id='X0', pre_stim=1.0, post_stim=2.0,
                 freqs=None):
    """
    跨试验相位一致性 (ITPC / 相位锁定因子 PLF)

    流程:
        1. 提取 epochs
        2. 对每个 epoch 在每个频率做 CWT 提取相位
        3. 计算跨 epoch 的平均相位向量长度 |mean(e^{i*phase})|

    返回:
        {'times': [...], 'freqs': [...], 'itpc': (n_freqs, n_times) 矩阵, 值域 [0, 1],
         'n_epochs': int}
    """
    if freqs is None:
        freqs = np.linspace(1, 45, 45)
    freqs = np.asarray(freqs, dtype=float)

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_channels = data.shape[1]

    # 1. 提取 epochs
    epochs, times = _extract_epochs(data, fs, events_df, event_id,
                                    pre_stim, post_stim)
    n_epochs, n_epoch_samples, _ = epochs.shape
    n_freqs = len(freqs)

    # 2. 跨 epoch 相位向量累加 (多通道平均)
    phase_vector_sum = np.zeros((n_freqs, n_epoch_samples), dtype=complex)
    for ep in range(n_epochs):
        cwt_result = compute_cwt(epochs[ep], fs, freqs=freqs)
        # 多通道平均相位向量
        phase = cwt_result['phase']
        if phase.ndim == 3:  # (n_channels, n_freqs, n_times)
            phase = phase.mean(axis=0)
        phase_vector_sum += np.exp(1j * phase)

    # 3. 平均向量长度
    itpc = np.abs(phase_vector_sum) / n_epochs

    return {
        'times': times.tolist(),
        'freqs': freqs.tolist(),
        'itpc': itpc.tolist(),
        'n_epochs': int(n_epochs),
    }


# ========== 6. 完整分析流水线 ==========
def _downsample_matrix(matrix: np.ndarray, max_freqs: int = 30,
                       max_times: int = 200) -> np.ndarray:
    """对时频矩阵降采样 (频率轴、时间轴均匀抽取)"""
    n_freqs, n_times = matrix.shape
    # 频率轴抽样
    if n_freqs > max_freqs:
        f_idx = np.linspace(0, n_freqs - 1, max_freqs).astype(int)
    else:
        f_idx = np.arange(n_freqs)
    # 时间轴抽样
    if n_times > max_times:
        t_idx = np.linspace(0, n_times - 1, max_times).astype(int)
    else:
        t_idx = np.arange(n_times)
    return matrix[np.ix_(f_idx, t_idx)]


def _downsample_axis(axis: np.ndarray, target_len: int) -> np.ndarray:
    """对一维轴降采样"""
    n = len(axis)
    if n <= target_len:
        return axis
    idx = np.linspace(0, n - 1, target_len).astype(int)
    return axis[idx]


def run_ersp_analysis(data, fs, events_df, event_id='X0'):
    """
    完整 ERSP 分析流水线

    返回:
        {'ersp': {...}, 'erd_ers': {...}, 'pac': {...}, 'itpc': {...},
         'event_id': ..., 'fs': fs, 'n_channels': int}

    注意: 时频矩阵转 list 时降采样 (频率轴 30 点, 时间轴 200 点)
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_samples, n_channels = data.shape

    # 频率轴 (用于 ERSP/ITPC)
    freqs = np.linspace(1, 45, 45)

    # 1. ERSP
    ersp_result = compute_ersp(
        data, fs, events_df, event_id=event_id,
        pre_stim=2.0, post_stim=5.0,
        freqs=freqs, baseline_start=-2.0, baseline_end=-0.2,
    )

    # 降采样时频矩阵
    ersp_matrix = np.array(ersp_result['ersp'])
    ersp_ds = _downsample_matrix(ersp_matrix, max_freqs=30, max_times=200)
    ersp_freqs = _downsample_axis(np.array(ersp_result['freqs']), 30)
    ersp_times = _downsample_axis(np.array(ersp_result['times']), 200)
    ersp_result['ersp'] = ersp_ds.tolist()
    ersp_result['freqs'] = ersp_freqs.tolist()
    ersp_result['times'] = ersp_times.tolist()

    # 2. ERD/ERS 分类
    erd_ers = compute_erd_ers(ersp_matrix, threshold=0)
    # 对 mask 也降采样以匹配显示
    erd_mask_ds = _downsample_matrix(np.array(erd_ers['erd_mask']), 30, 200)
    ers_mask_ds = _downsample_matrix(np.array(erd_ers['ers_mask']), 30, 200)
    erd_ers['erd_mask'] = erd_mask_ds.tolist()
    erd_ers['ers_mask'] = ers_mask_ds.tolist()

    # 3. PAC — 取事件附近 ±10s 窗口, 多通道平均
    evt_mask = events_df['event_id'] == event_id
    if evt_mask.any():
        evt_time = float(events_df.loc[evt_mask, 'timestamp'].values[0])
    else:
        evt_time = n_samples / (2 * fs)
    win_start = max(int((evt_time - 10) * fs), 0)
    win_end = min(int((evt_time + 10) * fs), n_samples)
    # 多通道 PAC 取平均 MI
    pac_results = []
    for ch in range(n_channels):
        seg = data[win_start:win_end, ch]
        pac_results.append(compute_pac(seg, fs,
                                       phase_band=BANDS['theta'],
                                       amp_band=BANDS['gamma']))
    # 平均相位-振幅分布
    avg_dist = np.mean([p['phase_amp_dist'] for p in pac_results], axis=0)
    pac_summary = {
        'mi': float(np.mean([p['mi'] for p in pac_results])),
        'phase_band': pac_results[0]['phase_band'],
        'amp_band': pac_results[0]['amp_band'],
        'phase_amp_dist': avg_dist.tolist(),
        'phase_bins': pac_results[0]['phase_bins'],
        'n_bins': pac_results[0]['n_bins'],
        'per_channel_mi': [p['mi'] for p in pac_results],
    }

    # 4. ITPC
    itpc_result = compute_itpc(
        data, fs, events_df, event_id=event_id,
        pre_stim=1.0, post_stim=2.0, freqs=freqs,
    )
    itpc_matrix = np.array(itpc_result['itpc'])
    itpc_ds = _downsample_matrix(itpc_matrix, max_freqs=30, max_times=200)
    itpc_freqs = _downsample_axis(np.array(itpc_result['freqs']), 30)
    itpc_times = _downsample_axis(np.array(itpc_result['times']), 200)
    itpc_result['itpc'] = itpc_ds.tolist()
    itpc_result['freqs'] = itpc_freqs.tolist()
    itpc_result['times'] = itpc_times.tolist()

    return {
        'ersp': ersp_result,
        'erd_ers': erd_ers,
        'pac': pac_summary,
        'itpc': itpc_result,
        'event_id': event_id,
        'fs': int(fs),
        'n_channels': int(n_channels),
        'n_samples': int(n_samples),
        'duration_sec': float(n_samples / fs),
    }
