"""
EEG 频谱分析模块
提供 PSD、频段能量对比、时频分析
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


def compute_band_powers(data: np.ndarray, fs: int, nperseg: int = 1024,
                        overlap: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    计算各频段绝对功率与相对功率
    
    返回:
        {'delta': {'abs': float, 'rel': float, 'freq_range': [lo, hi]}, ...}
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
        abs_power = np.sum(psd[mask]) * df
        rel_power = abs_power / (total_power + 1e-12) * 100
        band_powers[bname] = {
            'abs': float(abs_power),
            'rel': float(rel_power),
            'freq_range': [flo, fhi],
        }

    return band_powers


def compute_spectrogram(data: np.ndarray, fs: int, window: str = 'hann',
                        nperseg: int = 256, overlap: float = 0.75,
                        max_freq: float = 45.0) -> Dict[str, List]:
    """
    计算时频图 (STFT)
    
    参数:
        data: (n_samples, n_channels) EEG 数据
        fs: 采样率
        nperseg: 每段样本数
        overlap: 重叠比例
        max_freq: 最大显示频率
    
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

    # 转 dB
    spec_db = 10 * np.log10(spec_total + 1e-12)

    return {
        'times': times.tolist(),
        'freqs': freqs.tolist(),
        'spectrogram': spec_db.tolist(),
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
    
    返回: PSD + 频段能量 + 时频图
    """
    psd = compute_psd(data, fs, nperseg, overlap)
    band_powers = compute_band_powers(data, fs, nperseg, overlap)
    spectrogram = compute_spectrogram(data, fs, nperseg=256, overlap=0.75)

    return {
        'psd': psd,
        'band_powers': band_powers,
        'spectrogram': spectrogram,
        'fs': fs,
        'n_samples': len(data),
        'duration_sec': len(data) / fs,
    }
