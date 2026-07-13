"""
EEG 频谱分析模块
提供 PSD、频段能量对比、时频分析、1/f 斜率分析
"""
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional


# 频段定义（与 flow_recovery.py 一致）
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}


def compute_psd(data: np.ndarray, fs: int, nperseg: int = 1024,
                overlap: float = 0.5) -> Dict[str, List[float]]:
    """
    计算多通道平均功率谱密度 (Welch 法)
    
    参数:
        data: (n_samples, n_channels) EEG 数据
        fs: 采样率
        nperseg: 每段样本数
        overlap: 重叠比例
    
    返回:
        {'freqs': [...], 'psd': [...]} — 频率轴与功率谱密度（多通道平均）
    """
    n_samples, n_channels = data.shape
    noverlap = int(nperseg * overlap)

    # 多通道平均 PSD
    psd_total = None
    freqs = None
    for ch in range(n_channels):
        f, p = signal.welch(data[:, ch], fs=fs, nperseg=min(nperseg, n_samples),
                           noverlap=noverlap, scaling='density')
        if psd_total is None:
            psd_total = p
            freqs = f
        else:
            psd_total += p
    psd_total /= n_channels

    return {
        'freqs': freqs.tolist(),
        'psd': psd_total.tolist(),
    }


def compute_aperiodic_signal(psd: np.ndarray, freqs: np.ndarray,
                             fit_range: Tuple[float, float] = (3, 40)) -> Dict:
    """
    1/f 斜率分析（非周期信号）

    用对数空间线性回归拟合: log(power) = -slope * log(freq) + intercept
    slope 反映大脑觉醒度（越陡 = 越困倦）

    参数:
        psd: 功率谱密度 (1D)
        freqs: 频率轴 (1D, Hz)
        fit_range: (lo, hi) 拟合频率范围 (Hz)

    返回:
        {'slope': float, 'intercept': float,
         'fit_freqs': [...], 'fit_line': [...],
         'r_squared': float}
    """
    psd = np.asarray(psd, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    flo, fhi = fit_range
    mask = (freqs >= flo) & (freqs <= fhi) & (psd > 0) & (freqs > 0)

    if np.sum(mask) < 2:
        return {
            'slope': 0.0,
            'intercept': 0.0,
            'fit_freqs': [],
            'fit_line': [],
            'r_squared': 0.0,
        }

    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])

    # 线性回归: log_p = a * log_f + b, 其中 a = -slope
    coeffs = np.polyfit(log_f, log_p, 1)
    a, b = coeffs
    slope = -float(a)
    intercept = float(b)

    # 拟合直线 (log 空间)
    fit_line = (a * log_f + b).tolist()

    # R²
    log_p_pred = a * log_f + b
    ss_res = float(np.sum((log_p - log_p_pred) ** 2))
    ss_tot = float(np.sum((log_p - np.mean(log_p)) ** 2))
    r_squared = float(1 - ss_res / (ss_tot + 1e-12))

    return {
        'slope': slope,
        'intercept': intercept,
        'fit_freqs': freqs[mask].tolist(),
        'fit_line': fit_line,
        'r_squared': r_squared,
    }


def compute_band_powers(data: np.ndarray, fs: int, nperseg: int = 1024,
                        overlap: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    计算各频段绝对功率、相对功率、峰值频率与谱边缘频率

    返回:
        {'delta': {'abs': float, 'rel': float, 'freq_range': [lo, hi],
                   'peak_frequency': float, 'spectral_edge_density': float}, ...}
        peak_frequency: 频段内功率最大值对应的频率
        spectral_edge_density: 频段内 95% 谱边缘频率 (累积功率达 95% 时的频率)
    """
    psd_result = compute_psd(data, fs, nperseg, overlap)
    freqs = np.array(psd_result['freqs'])
    psd = np.array(psd_result['psd'])

    # 频率分辨率
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total_power = np.sum(psd) * df

    band_powers = {}
    for bname, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs < fhi)
        band_psd = psd[mask]
        band_freqs = freqs[mask]
        abs_power = np.sum(band_psd) * df
        rel_power = abs_power / (total_power + 1e-12) * 100

        # 峰值频率 & 谱边缘频率 (95%)
        if len(band_psd) > 0:
            peak_frequency = float(band_freqs[int(np.argmax(band_psd))])
            # 累积功率达 95% 时的频率
            cumulative = np.cumsum(band_psd)
            total_band = cumulative[-1] if cumulative[-1] > 0 else 1e-12
            sed_idx = int(np.searchsorted(cumulative, 0.95 * total_band))
            sed_idx = min(sed_idx, len(band_freqs) - 1)
            spectral_edge_density = float(band_freqs[sed_idx])
        else:
            peak_frequency = 0.0
            spectral_edge_density = 0.0

        band_powers[bname] = {
            'abs': float(abs_power),
            'rel': float(rel_power),
            'freq_range': [flo, fhi],
            'peak_frequency': peak_frequency,
            'spectral_edge_density': spectral_edge_density,
        }

    return band_powers


def compute_spectrogram(data: np.ndarray, fs: int, window: str = 'hann',
                        nperseg: int = 256, overlap: float = 0.75,
                        max_freq: float = 45.0, log_scale: bool = True,
                        log_freq_axis: bool = False) -> Dict[str, List]:
    """
    计算时频图 (STFT)
    
    参数:
        data: (n_samples, n_channels) EEG 数据
        fs: 采样率
        nperseg: 每段样本数
        overlap: 重叠比例
        max_freq: 最大显示频率
        log_scale: 是否转为 dB (默认 True)
        log_freq_axis: 是否使用对数频率轴 (默认 False, 线性)
    
    返回:
        {'times': [...], 'freqs': [...], 'spectrogram': [[...]]} — 时间轴、频率轴、功率矩阵
    """
    n_samples, n_channels = data.shape
    noverlap = int(nperseg * overlap)

    # 多通道平均 STFT
    spec_total = None
    freqs = None
    times = None
    for ch in range(n_channels):
        f, t, Sxx = signal.spectrogram(
            data[:, ch], fs=fs, window=window,
            nperseg=min(nperseg, n_samples),
            noverlap=noverlap, scaling='density'
        )
        if spec_total is None:
            spec_total = Sxx
            freqs = f
            times = t
        else:
            spec_total += Sxx
    spec_total /= n_channels

    # 限制频率范围
    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask]
    spec_total = spec_total[freq_mask, :]

    # 对数频率轴 (可选): 将线性频率重采样到对数刻度
    if log_freq_axis and len(freqs) > 1:
        positive_mask = freqs > 0
        if np.sum(positive_mask) > 1:
            pos_freqs = freqs[positive_mask]
            fmin = pos_freqs[0]
            fmax = pos_freqs[-1]
            log_freqs = np.logspace(np.log10(fmin), np.log10(fmax),
                                    num=len(pos_freqs))
            spec_interp = np.zeros((len(log_freqs), spec_total.shape[1]))
            for t_idx in range(spec_total.shape[1]):
                spec_interp[:, t_idx] = np.interp(
                    log_freqs, pos_freqs, spec_total[positive_mask, t_idx]
                )
            freqs = log_freqs
            spec_total = spec_interp

    # 转 dB (可选)
    if log_scale:
        spec_out = 10 * np.log10(spec_total + 1e-12)
    else:
        spec_out = spec_total

    return {
        'times': times.tolist(),
        'freqs': freqs.tolist(),
        'spectrogram': spec_out.tolist(),
    }


def compare_conditions_psd(results: Dict[str, Dict]) -> Dict[str, List]:
    """
    对比多个条件的频段能量
    
    参数:
        results: {condition_name: {'band_powers': {...}}, ...}
    
    返回:
        {'conditions': [...], 'bands': [...], 'values': [[...]]}
    """
    conditions = list(results.keys())
    bands = list(BANDS.keys())

    values = []
    for band in bands:
        row = []
        for cond in conditions:
            bp = results[cond].get('band_powers', {})
            row.append(bp.get(band, {}).get('rel', 0))
        values.append(row)

    return {
        'conditions': conditions,
        'bands': bands,
        'values': values,
    }


def run_spectrum_analysis(data: np.ndarray, fs: int,
                          nperseg: int = 1024, overlap: float = 0.5) -> Dict:
    """
    运行完整频谱分析
    
    返回: PSD + 频段能量 + 时频图 + 1/f 斜率分析
    """
    psd = compute_psd(data, fs, nperseg, overlap)
    band_powers = compute_band_powers(data, fs, nperseg, overlap)
    spectrogram = compute_spectrogram(data, fs, nperseg=256, overlap=0.75)

    # 1/f 斜率分析
    aperiodic = compute_aperiodic_signal(
        np.array(psd['psd']), np.array(psd['freqs'])
    )

    return {
        'psd': psd,
        'band_powers': band_powers,
        'spectrogram': spectrogram,
        'aperiodic_signal': aperiodic,
        'fs': fs,
        'n_samples': len(data),
        'duration_sec': len(data) / fs,
    }
