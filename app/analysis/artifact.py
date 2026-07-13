"""
EEG 伪迹检测与清除模块
提供阈值检测、统计检测、ICA 独立成分分析、伪迹移除、数据质量评分
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy import stats
from typing import Dict, List, Tuple, Optional


# 频段定义（与 spectrum.py / flow_recovery.py 一致）
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _mask_to_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """将布尔掩码转为连续 (start, end) 段列表，end 为排他索引"""
    if not np.any(mask):
        return []
    m = mask.astype(np.int8)
    diff = np.diff(m, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _cap_segments(segments: List[Tuple[int, int]],
                  limit: int = 300) -> Tuple[List[Tuple[int, int]], int, bool]:
    """限制返回的段数量，避免响应过大
    返回: (截断后的段列表, 总段数, 是否被截断)
    """
    total = len(segments)
    if total <= limit:
        return segments, total, False
    return segments[:limit], total, True


# ---------------------------------------------------------------------------
# 伪迹检测
# ---------------------------------------------------------------------------
def detect_by_threshold(data: np.ndarray, fs: int,
                        threshold_uv: float = 100.0) -> Dict:
    """
    阈值法检测伪迹样本（幅值超阈值）

    参数:
        data: (n_samples, n_channels) EEG 数据
        fs: 采样率
        threshold_uv: 幅值阈值 (μV)，任一通道绝对幅值超过即标记为伪迹

    返回:
        {'artifact_mask': bool数组, 'artifact_ratio': float,
         'artifact_segments': [(start, end), ...]}
    """
    mask = np.any(np.abs(data) > threshold_uv, axis=1)
    segments = _mask_to_segments(mask)
    return {
        'artifact_mask': mask,
        'artifact_ratio': float(np.mean(mask)),
        'artifact_segments': segments,
    }


def detect_by_zscore(data: np.ndarray, z_threshold: float = 3.0) -> Dict:
    """
    Z-score 法检测伪迹（统计偏离）

    参数:
        data: (n_samples, n_channels) EEG 数据
        z_threshold: Z-score 阈值，任一通道 |z| 超过即标记为伪迹

    返回:
        与 detect_by_threshold 相同格式
    """
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True) + 1e-12
    z = (data - mean) / std
    mask = np.any(np.abs(z) > z_threshold, axis=1)
    segments = _mask_to_segments(mask)
    return {
        'artifact_mask': mask,
        'artifact_ratio': float(np.mean(mask)),
        'artifact_segments': segments,
    }


# ---------------------------------------------------------------------------
# 信号统计
# ---------------------------------------------------------------------------
def compute_signal_stats(data: np.ndarray, fs: int) -> Dict[str, List[float]]:
    """
    计算信号统计特征

    参数:
        data: (n_samples, n_channels) EEG 数据
        fs: 采样率

    返回:
        {'peak': [...], 'rms': [...], 'variance': [...],
         'kurtosis': [...], 'skewness': [...], 'snr_db': [...]}
        每个值为各通道的列表
    """
    n_samples, n_channels = data.shape
    nperseg = min(1024, n_samples)
    peak, rms, variance, kurtosis, skewness, snr_db = [], [], [], [], [], []

    for ch in range(n_channels):
        x = data[:, ch]
        peak.append(float(np.max(np.abs(x))))
        rms.append(float(np.sqrt(np.mean(x ** 2))))
        variance.append(float(np.var(x)))
        kurtosis.append(float(stats.kurtosis(x)))
        skewness.append(float(stats.skew(x)))

        # SNR: 1-45Hz 视为信号，>45Hz 视为噪声
        f, p = scipy_signal.welch(x, fs=fs, nperseg=nperseg)
        signal_power = float(np.sum(p[(f >= 1) & (f <= 45)]))
        noise_power = float(np.sum(p[f > 45])) + 1e-12
        snr_db.append(float(10.0 * np.log10(signal_power / noise_power + 1e-12)))

    return {
        'peak': peak,
        'rms': rms,
        'variance': variance,
        'kurtosis': kurtosis,
        'skewness': skewness,
        'snr_db': snr_db,
    }


# ---------------------------------------------------------------------------
# FastICA (numpy 实现，不依赖 sklearn)
# ---------------------------------------------------------------------------
def fast_ica(X: np.ndarray, n_components: Optional[int] = None,
             max_iter: int = 200, tol: float = 1e-4) -> Dict:
    """
    用 numpy 实现 FastICA 独立成分分析（不依赖 sklearn）

    参数:
        X: 形状 (n_samples, n_channels)
        n_components: 保留成分数，默认为 n_channels
        max_iter: 单成分最大迭代次数
        tol: 收敛阈值（基于方向变化的余弦）

    返回:
        {'components': (n_samples, n_components),  # 源成分矩阵
         'mixing': (n_components, n_channels),     # 混合矩阵
         'n_components': int}

    实现要点:
        1. 中心化 + 白化（协方差特征分解）
        2. 固定点迭代（tanh 非线性）求解独立成分
        3. Gram-Schmidt 去相关，返回源成分与混合矩阵
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_channels = X.shape
    if n_components is None:
        n_components = n_channels
    n_components = min(n_components, n_channels)

    # 1. 中心化
    mean = np.mean(X, axis=0, keepdims=True)
    Xc = X - mean

    # 2. 白化（协方差特征分解）
    cov = (Xc.T @ Xc) / max(n_samples, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # 升序
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    eigvals = np.maximum(eigvals, 1e-12)
    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:, :n_components]
    # 白化矩阵 (n_channels, n_components)
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
    Z = Xc @ whitening  # (n_samples, n_components)，白化后协方差≈I

    # 3. 固定点迭代（Hyvärinen FastICA，tanh 非线性）
    rng = np.random.default_rng(0)
    W = np.zeros((n_components, n_components))
    for i in range(n_components):
        w = rng.standard_normal(n_components)
        w = w / (np.linalg.norm(w) + 1e-12)
        for _ in range(max_iter):
            wZ = Z @ w  # (n_samples,)
            g = np.tanh(wZ)
            g_prime = 1.0 - g * g
            w_new = np.mean(Z * g[:, None], axis=0) - np.mean(g_prime) * w
            # Gram-Schmidt 去相关（对已求成分）
            if i > 0:
                w_new = w_new - W[:i].T @ (W[:i] @ w_new)
            norm = np.linalg.norm(w_new)
            if norm < 1e-12:
                w_new = rng.standard_normal(n_components)
                norm = np.linalg.norm(w_new)
            w_new = w_new / norm
            # 收敛判断（方向稳定）
            if abs(abs(float(w_new @ w)) - 1.0) < tol:
                w = w_new
                break
            w = w_new
        W[i] = w

    # 源成分: S = Z @ W.T = Xc @ (W @ whitening.T).T
    S = Z @ W.T  # (n_samples, n_components)
    # 完整解混矩阵 (n_components, n_channels)
    W_full = W @ whitening.T
    # 混合矩阵: Xc ≈ S @ mixing，故 mixing = pinv(W_full.T) = (n_components, n_channels)
    mixing = np.linalg.pinv(W_full.T)

    return {
        'components': S,
        'mixing': mixing,
        'n_components': int(n_components),
    }


# ---------------------------------------------------------------------------
# 成分分类
# ---------------------------------------------------------------------------
def classify_components(components: np.ndarray, mixing: np.ndarray,
                        fs: int) -> List[Dict]:
    """
    分类独立成分为: 眼动(EOG)/肌电(EMG)/心电(ECG)/脑电(EEG)

    基于统计特征:
        - 眼动: 低频主导(<8Hz)、高方差、前额通道强负载
        - 肌电: 高频主导(>30Hz)、高峭度
        - 心电: 周期性、低频峰值、极高峭度
        - 脑电: 1-30Hz 主导

    参数:
        components: (n_samples, n_components) 源成分
        mixing: (n_components, n_channels) 混合矩阵
        fs: 采样率

    返回:
        [{'index': i, 'type': 'EOG/EMG/ECG/EEG',
          'confidence': float, 'reason': str}, ...]
    """
    n_samples, n_comp = components.shape
    nperseg = min(1024, n_samples)

    info = []
    for i in range(n_comp):
        src = components[:, i]
        f, p = scipy_signal.welch(src, fs=fs, nperseg=nperseg)
        total = float(np.sum(p)) + 1e-12
        low = float(np.sum(p[f < 8]) / total)
        high = float(np.sum(p[f > 30]) / total)
        mid = float(np.sum(p[(f >= 1) & (f <= 30)]) / total)
        info.append({
            'var': float(np.var(src)),
            'kurt': float(stats.kurtosis(src)),
            'skew': float(stats.skew(src)),
            'low': low, 'high': high, 'mid': mid,
            'dom_freq': float(f[np.argmax(p)]),
            'loading': np.abs(mixing[i]),
        })

    var_arr = np.array([x['var'] for x in info])
    load_arr = np.array([float(np.mean(x['loading'])) for x in info])
    var_median = float(np.median(var_arr)) if n_comp > 0 else 0.0
    load_median = float(np.median(load_arr)) if n_comp > 0 else 0.0

    results = []
    for i in range(n_comp):
        x = info[i]
        comp_type = 'EEG'
        confidence = 0.5
        reason = f'1-30Hz 能量占比 {x["mid"]:.2f} 主导，统计特征平稳（脑电特征）'

        if (x['low'] > 0.5 and x['var'] >= var_median
                and load_arr[i] >= load_median):
            comp_type = 'EOG'
            confidence = min(0.95, 0.55 + x['low'] * 0.3)
            reason = (f'低频(<8Hz)能量占比 {x["low"]:.2f} 高，方差较大，'
                      f'前额通道负载强（眼动特征）')
        elif x['high'] > 0.4 and x['kurt'] > 3.0:
            comp_type = 'EMG'
            confidence = min(0.95, 0.55 + x['high'] * 0.3)
            reason = (f'高频(>30Hz)能量占比 {x["high"]:.2f} 高，'
                      f'峭度 {x["kurt"]:.2f} 大（肌电特征）')
        elif x['kurt'] > 5.0 and x['low'] > 0.3 and x['dom_freq'] < 5.0:
            comp_type = 'ECG'
            confidence = min(0.9, 0.55 + x['kurt'] / 30.0)
            reason = (f'峭度 {x["kurt"]:.2f} 极高且呈周期性低频峰值'
                      f'（主频 {x["dom_freq"]:.1f}Hz，心电特征）')
        else:
            comp_type = 'EEG'
            confidence = min(0.95, 0.5 + x['mid'] * 0.4)

        results.append({
            'index': i,
            'type': comp_type,
            'confidence': float(confidence),
            'reason': reason,
        })
    return results


# ---------------------------------------------------------------------------
# 伪迹移除
# ---------------------------------------------------------------------------
def remove_artifacts(data: np.ndarray, components: np.ndarray,
                     mixing: np.ndarray,
                     bad_indices: Optional[List[int]] = None) -> np.ndarray:
    """
    移除伪迹成分并重构信号

    参数:
        data: 原始数据 (n_samples, n_channels)
        components: 源成分 (n_samples, n_components)
        mixing: 混合矩阵 (n_components, n_channels)
        bad_indices: 需置零的伪迹成分索引列表

    返回:
        清洗后的数据 (n_samples, n_channels)
    """
    if bad_indices is None:
        bad_indices = []

    S = components.copy()
    if len(bad_indices) > 0:
        S[:, bad_indices] = 0.0

    # 重构：fast_ica 对数据做了中心化，需加回原始均值
    mean = np.mean(data, axis=0, keepdims=True)
    cleaned = S @ mixing + mean
    return cleaned


# ---------------------------------------------------------------------------
# 数据质量评分
# ---------------------------------------------------------------------------
def quality_score(data: np.ndarray, fs: int) -> Dict:
    """
    综合数据质量评分 (0-100)

    基于: 伪迹比例、信噪比、信号稳定性、频率分布合理性

    参数:
        data: (n_samples, n_channels) EEG 数据
        fs: 采样率

    返回:
        {'score': float, 'grade': 'A/B/C/D', 'factors': {...}}
    """
    n_samples, n_channels = data.shape

    # 1. 伪迹比例（5 倍惩罚）
    thr = detect_by_threshold(data, fs, threshold_uv=100.0)
    artifact_ratio = thr['artifact_ratio']
    artifact_factor = max(0.0, 1.0 - artifact_ratio * 5.0)

    # 2. 信噪比
    stats_res = compute_signal_stats(data, fs)
    snr_db = float(np.mean(stats_res['snr_db']))
    snr_factor = min(1.0, max(0.0, snr_db / 20.0))

    # 3. 信号稳定性（1 秒窗口 RMS 的变异系数，越小越稳定）
    win = max(fs, 1)
    n_win = n_samples // win
    if n_win > 1:
        rms_win = []
        for ch in range(n_channels):
            seg = data[:n_win * win, ch].reshape(n_win, win)
            rms_win.append(np.sqrt(np.mean(seg ** 2, axis=1)))
        rms_arr = np.concatenate(rms_win)
        cv = float(np.std(rms_arr) / (np.mean(np.abs(rms_arr)) + 1e-12))
        stability_factor = min(1.0, max(0.0, 1.0 - cv))
    else:
        cv = None
        stability_factor = 0.5

    # 4. 频率分布合理性（alpha+beta 相对功率应在合理区间）
    alpha_beta_ratio = None
    freq_factor = 0.5
    try:
        nperseg = min(1024, n_samples)
        psd_avg = None
        for ch in range(n_channels):
            f, p = scipy_signal.welch(data[:, ch], fs=fs, nperseg=nperseg)
            psd_avg = p if psd_avg is None else psd_avg + p
        psd_avg /= n_channels
        total_p = float(np.sum(psd_avg)) + 1e-12
        alpha_beta_ratio = float(np.sum(psd_avg[(f >= 8) & (f <= 30)]) / total_p)
        if 0.2 <= alpha_beta_ratio <= 0.6:
            freq_factor = 1.0
        else:
            freq_factor = max(0.0, 1.0 - abs(alpha_beta_ratio - 0.4) * 2.0)
    except Exception:
        freq_factor = 0.5

    # 加权综合
    score = (
        artifact_factor * 0.35 +
        snr_factor * 0.25 +
        stability_factor * 0.25 +
        freq_factor * 0.15
    ) * 100.0

    if score >= 85:
        grade = 'A'
    elif score >= 70:
        grade = 'B'
    elif score >= 50:
        grade = 'C'
    else:
        grade = 'D'

    factors = {
        'artifact_ratio': float(artifact_ratio),
        'artifact_factor': float(artifact_factor),
        'snr_db': snr_db,
        'snr_factor': float(snr_factor),
        'stability_cv': cv,
        'stability_factor': float(stability_factor),
        'alpha_beta_ratio': alpha_beta_ratio,
        'freq_factor': float(freq_factor),
    }

    return {
        'score': float(score),
        'grade': grade,
        'factors': factors,
    }


# ---------------------------------------------------------------------------
# 完整分析流水线
# ---------------------------------------------------------------------------
def run_artifact_analysis(data: np.ndarray, fs: int) -> Dict:
    """
    完整伪迹分析流水线

    参数:
        data: (n_samples, n_channels) EEG 数据
        fs: 采样率

    返回:
        {
            'threshold_detection': ...,
            'zscore_detection': ...,
            'signal_stats': ...,
            'ica_result': {'components': ..., 'mixing': ..., 'classification': ...},
            'quality': {'score': ..., 'grade': ..., 'factors': ...},
            'cleaned_data_shape': (n_samples, n_channels),
        }

    注意: components 数据降采样后转 list 返回（用于前端可视化），
          不返回完整 cleaned_data（太大），仅返回其形状。
    """
    # 检测
    thr_det = detect_by_threshold(data, fs)
    zsc_det = detect_by_zscore(data)
    sig_stats = compute_signal_stats(data, fs)

    # ICA
    ica_raw = fast_ica(data)
    components = ica_raw['components']  # (n_samples, n_comp)
    mixing = ica_raw['mixing']         # (n_comp, n_channels)
    classification = classify_components(components, mixing, fs)

    # 质量
    quality = quality_score(data, fs)

    # components 降采样为 list（前端可视化，最多 ~2000 点/成分）
    n_samples = components.shape[0]
    max_points = 2000
    if n_samples > max_points:
        step = max(1, n_samples // max_points)
        components_vis = components[::step].tolist()
    else:
        components_vis = components.tolist()

    # 段落截断（避免响应过大）
    thr_segs, thr_total, thr_trunc = _cap_segments(thr_det['artifact_segments'])
    zsc_segs, zsc_total, zsc_trunc = _cap_segments(zsc_det['artifact_segments'])

    return {
        'threshold_detection': {
            'artifact_ratio': thr_det['artifact_ratio'],
            'artifact_segments': thr_segs,
            'n_segments': thr_total,
            'segments_truncated': thr_trunc,
        },
        'zscore_detection': {
            'artifact_ratio': zsc_det['artifact_ratio'],
            'artifact_segments': zsc_segs,
            'n_segments': zsc_total,
            'segments_truncated': zsc_trunc,
        },
        'signal_stats': sig_stats,
        'ica_result': {
            'components': components_vis,
            'mixing': mixing.tolist(),
            'n_components': ica_raw['n_components'],
            'classification': classification,
        },
        'quality': quality,
        'cleaned_data_shape': tuple(int(s) for s in data.shape),
    }
