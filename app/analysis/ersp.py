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


def compute_freqs_logspace(fmin: float = 1.0, fmax: float = 45.0,
                           n_freqs: int = 50) -> np.ndarray:
    """
    生成对数间隔的频率数组 (低频分辨率更高)

    参数:
        fmin: 最低频率 (Hz)
        fmax: 最高频率 (Hz)
        n_freqs: 频率点数

    返回:
        (n_freqs,) 频率数组 (对数间隔)
    """
    if fmin <= 0:
        fmin = 1.0
    if fmax <= fmin:
        fmax = fmin + 1.0
    return np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)


def compute_cwt(data, fs, freqs=None, wavelet_width=5.0, n_freqs=50):
    """
    连续小波变换 (Morlet 小波, 复小波, 基于 FFT 卷积加速)

    参数:
        data: (n_samples,) 或 (n_samples, n_channels)
        fs: 采样率
        freqs: 频率数组, 默认对数间隔 (1-45Hz, n_freqs 点)
        wavelet_width: 小波宽度参数 (默认 5.0 = 周期数)
        n_freqs: freqs 为 None 时的频率点数 (默认 50, 对数间隔)

    返回:
        {'power': (len(freqs), n_samples) 或 (n_channels, len(freqs), n_samples),
         'phase': 同形状相位矩阵,
         'freqs': [...]}
    """
    if freqs is None:
        freqs = compute_freqs_logspace(1, 45, n_freqs)
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


def _compute_epochs_power(epochs: np.ndarray, fs: int,
                          freqs: np.ndarray) -> np.ndarray:
    """
    计算每个 epoch 的时频功率 (多通道平均)

    参数:
        epochs: (n_epochs, n_samples_epoch, n_channels)
        fs: 采样率
        freqs: 频率数组

    返回:
        (n_epochs, n_freqs, n_times) 功率矩阵
    """
    n_epochs = epochs.shape[0]
    n_freqs = len(freqs)
    n_times = epochs.shape[1]
    power = np.zeros((n_epochs, n_freqs, n_times))
    for ep in range(n_epochs):
        cwt_result = compute_cwt(epochs[ep], fs, freqs=freqs)
        p = cwt_result['power']
        if p.ndim == 3:  # (n_channels, n_freqs, n_times)
            p = p.mean(axis=0)
        power[ep] = p
    return power


def _compute_ersp_from_power(epochs_power: np.ndarray, times: np.ndarray,
                             freqs: np.ndarray,
                             baseline_start: float, baseline_end: float,
                             baseline_method: str = 'median'
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 per-epoch 功率矩阵计算 ERSP

    参数:
        epochs_power: (n_epochs, n_freqs, n_times)
        times: 时间轴
        freqs: 频率轴
        baseline_start: 基线起始时间 (秒)
        baseline_end: 基线结束时间 (秒)
        baseline_method: 'median' (中位数, 抗异常值) 或 'mean' (均值)

    返回:
        (ersp: (n_freqs, n_times) dB 矩阵,
         power_avg: (n_freqs, n_times) 平均功率,
         bl_mask: 基线时间掩码)
    """
    power_avg = epochs_power.mean(axis=0)
    n_times = len(times)
    bl_mask = (times >= baseline_start) & (times <= baseline_end)
    if bl_mask.sum() < 1:
        bl_mask = np.arange(n_times) < max(n_times // 10, 1)

    if baseline_method == 'median':
        baseline_val = np.median(power_avg[:, bl_mask], axis=1, keepdims=True)
    else:
        baseline_val = np.mean(power_avg[:, bl_mask], axis=1, keepdims=True)
    baseline_val = np.maximum(baseline_val, 1e-12)

    ersp = 10 * np.log10(power_avg / baseline_val)
    return ersp, power_avg, bl_mask


def compute_ersp(data, fs, events_df, event_id='X0', pre_stim=2.0, post_stim=5.0,
                 freqs=None, baseline_start=-2.0, baseline_end=-0.2,
                 baseline_method='median', n_freqs=50):
    """
    事件相关谱扰动 (ERSP)

    流程:
        1. 提取事件锁定的 epochs
        2. 对每个 epoch 做 CWT 得到时频功率
        3. 平均各 epoch 的功率
        4. 用基线期功率归一化: ERSP = 10*log10(power / baseline)
           baseline_method='median' 用中位数 (更抗异常值), 'mean' 用均值

    参数:
        baseline_method: 基线归一化方法 ('median' 或 'mean', 默认 'median')
        n_freqs: freqs 为 None 时的频率点数 (默认 50, 对数间隔)

    返回:
        {'times': [...], 'freqs': [...], 'ersp': (n_freqs, n_times) dB 矩阵,
         'n_epochs': int}

    若事件只出现一次, 用滑窗法在事件周围生成伪 epochs
    """
    if freqs is None:
        freqs = compute_freqs_logspace(1, 45, n_freqs)
    freqs = np.asarray(freqs, dtype=float)

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    # 1. 提取 epochs
    epochs, times = _extract_epochs(data, fs, events_df, event_id,
                                    pre_stim, post_stim)
    n_epochs = epochs.shape[0]

    # 2. per-epoch 功率
    epochs_power = _compute_epochs_power(epochs, fs, freqs)

    # 3. 基线归一化
    ersp, _, _ = _compute_ersp_from_power(
        epochs_power, times, freqs,
        baseline_start, baseline_end, baseline_method,
    )

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


# ========== 3b. 置换检验 & 跨条件对比 ==========
def permutation_test_ersp(epochs_power, baseline_power,
                          n_permutations=1000) -> Dict:
    """
    对每个时频点做置换检验, 评估 ERSP 的统计显著性

    参数:
        epochs_power: (n_epochs, n_freqs, n_times) 刺激后功率
        baseline_power: (n_epochs, n_freqs, n_baseline_times) 基线期功率
        n_permutations: 置换次数 (建议 ≤ 200 以控制时间)

    返回:
        {'p_values': (n_freqs, n_times) 矩阵,
         'significant_mask': bool 矩阵 (p < 0.05)}
    """
    epochs_power = np.asarray(epochs_power, dtype=float)
    baseline_power = np.asarray(baseline_power, dtype=float)

    n_epochs, n_freqs, n_times = epochs_power.shape

    # 基线功率: 每个 epoch 在基线时段取均值 -> (n_epochs, n_freqs)
    baseline_per_epoch = baseline_power.mean(axis=2)
    # 观测基线均值 (跨 epoch) -> (n_freqs,)
    baseline_observed = baseline_per_epoch.mean(axis=0)

    # 刺激后各 (freq, time) 的跨 epoch 均值
    post_mean = epochs_power.mean(axis=0)  # (n_freqs, n_times)

    # 实际差异: post - baseline (broadcast baseline 到每个 time)
    actual_diff = post_mean - baseline_observed[:, None]  # (n_freqs, n_times)

    # 池化: 将每个 epoch 的 post 值与 baseline 值合并
    # pooled shape: (2*n_epochs, n_freqs, n_times)
    # baseline 在 time 维度上广播 (每个 epoch 的 baseline 值对所有 time 相同)
    baseline_broadcast = np.broadcast_to(
        baseline_per_epoch[:, :, None], (n_epochs, n_freqs, n_times)
    )
    pooled = np.concatenate([epochs_power, baseline_broadcast], axis=0)
    n_pooled = 2 * n_epochs
    half = n_epochs

    rng = np.random.default_rng(0)
    abs_actual = np.abs(actual_diff)

    # 逐次置换, 累计计数 (避免存储全部零分布)
    count = np.zeros_like(actual_diff)
    for _ in range(n_permutations):
        idx = rng.permutation(n_pooled)
        group_a = pooled[idx[:half]]
        group_b = pooled[idx[half:]]
        null_diff = group_a.mean(axis=0) - group_b.mean(axis=0)
        count += (np.abs(null_diff) >= abs_actual).astype(int)

    # 双尾 p 值 (加 1 平滑避免 p=0)
    p_values = (count + 1.0) / (n_permutations + 1.0)
    significant_mask = p_values < 0.05

    return {
        'p_values': p_values,
        'significant_mask': significant_mask,
    }


def compare_ersp_conditions(ersp_a, ersp_b, n_permutations=200) -> Dict:
    """
    跨条件 ERSP 对比: 计算差值矩阵和统计显著性

    参数:
        ersp_a, ersp_b: ERSP 结果 dict (含 'ersp') 或 (n_freqs, n_times) dB 矩阵
                        若 dict 包含 'epochs_power' 字段, 则做置换检验;
                        否则基于效应量 (|diff| > 1 dB) 标记显著性
        n_permutations: 置换次数

    返回:
        {'diff': 差值矩阵, 'p_values': p 值矩阵,
         'significant_mask': 显著性掩码, 'n_permutations': int}
    """
    # 提取 ERSP 矩阵与可选的 per-epoch 功率
    if isinstance(ersp_a, dict):
        mat_a = np.asarray(ersp_a['ersp'], dtype=float)
        epa = ersp_a.get('epochs_power')
    else:
        mat_a = np.asarray(ersp_a, dtype=float)
        epa = None
    if isinstance(ersp_b, dict):
        mat_b = np.asarray(ersp_b['ersp'], dtype=float)
        epb = ersp_b.get('epochs_power')
    else:
        mat_b = np.asarray(ersp_b, dtype=float)
        epb = None

    diff = mat_b - mat_a
    abs_diff = np.abs(diff)

    if epa is not None and epb is not None:
        # 有 per-epoch 功率: 真正的置换检验 (打乱条件标签)
        epa = np.asarray(epa, dtype=float)
        epb = np.asarray(epb, dtype=float)
        pooled = np.concatenate([epa, epb], axis=0)
        n_a = epa.shape[0]
        n_total = pooled.shape[0]
        rng = np.random.default_rng(0)
        count = np.zeros_like(diff)
        for _ in range(n_permutations):
            idx = rng.permutation(n_total)
            perm_a = pooled[idx[:n_a]].mean(axis=0)
            perm_b = pooled[idx[n_a:]].mean(axis=0)
            perm_diff = perm_b - perm_a
            count += (np.abs(perm_diff) >= abs_diff).astype(int)
        p_values = (count + 1.0) / (n_permutations + 1.0)
        significant_mask = p_values < 0.05
    else:
        # 无 per-epoch 功率: 基于效应量阈值
        significant_mask = abs_diff > 1.0
        p_values = np.where(significant_mask, 0.01, 0.5)

    return {
        'diff': diff.tolist(),
        'p_values': p_values.tolist(),
        'significant_mask': significant_mask.tolist(),
        'n_permutations': int(n_permutations),
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
        freqs = compute_freqs_logspace(1, 45, 50)
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
    完整 ERSP 分析流水线 (含置换检验)

    返回:
        {'ersp': {...含 permutation_test...}, 'erd_ers': {...},
         'pac': {...}, 'itpc': {...},
         'event_id': ..., 'fs': fs, 'n_channels': int}

    注意: 时频矩阵转 list 时降采样 (频率轴 30 点, 时间轴 200 点)
          置换检验默认 n_permutations=200 以控制计算时间
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_samples, n_channels = data.shape

    # 频率轴 (对数间隔, 50 点)
    freqs = compute_freqs_logspace(1, 45, 50)

    # 1. ERSP + 置换检验 (共享 per-epoch 功率计算, 避免重复 CWT)
    epochs, times = _extract_epochs(
        data, fs, events_df, event_id, pre_stim=2.0, post_stim=5.0
    )
    n_epochs = epochs.shape[0]
    epochs_power = _compute_epochs_power(epochs, fs, freqs)
    ersp, _, bl_mask = _compute_ersp_from_power(
        epochs_power, times, freqs,
        baseline_start=-2.0, baseline_end=-0.2, baseline_method='median',
    )

    # 置换检验 (限制 200 次以控制时间)
    post_mask = ~bl_mask
    baseline_power = epochs_power[:, :, bl_mask]
    post_power = epochs_power[:, :, post_mask]
    perm_result = permutation_test_ersp(
        post_power, baseline_power, n_permutations=200
    )

    # 降采样时频矩阵
    ersp_matrix = ersp
    ersp_ds = _downsample_matrix(ersp_matrix, max_freqs=30, max_times=200)
    ersp_freqs = _downsample_axis(freqs, 30)
    ersp_times = _downsample_axis(times, 200)
    p_vals_ds = _downsample_matrix(perm_result['p_values'], 30, 200)
    sig_ds = _downsample_matrix(
        perm_result['significant_mask'].astype(int), 30, 200
    )

    ersp_result = {
        'times': ersp_times.tolist(),
        'freqs': ersp_freqs.tolist(),
        'ersp': ersp_ds.tolist(),
        'n_epochs': int(n_epochs),
        'permutation_test': {
            'p_values': p_vals_ds.tolist(),
            'significant_mask': sig_ds.tolist(),
            'n_permutations': 200,
        },
    }

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
