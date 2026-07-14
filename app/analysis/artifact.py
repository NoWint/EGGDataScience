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

# Daubechies 4 (db4) 分解滤波器系数 (8 抽头)
_DB4_LO_D = np.array([
    -0.010597401785069030, 0.032883011666885200, 0.030841381835560540,
    -0.187034811717913000, -0.027983769416859850, 0.630880767929480300,
    0.714846570552915400, 0.230377813308896500,
])
_DB4_HI_D = np.array([
    -0.230377813308896500, 0.714846570552915400, -0.630880767929480300,
    -0.027983769416859850, 0.187034811717913000, 0.030841381835560540,
    -0.032883011666885200, -0.010597401785069030,
])
# Haar 分解滤波器
_HAAR_LO_D = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
_HAAR_HI_D = np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2)])


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
# 小波伪迹检测
# ---------------------------------------------------------------------------
def _dwt_decompose(sig: np.ndarray, lo_d: np.ndarray, hi_d: np.ndarray,
                   level: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Mallat 算法多级离散小波分解

    参数:
        sig: (n_samples,) 输入信号
        lo_d: 低通分解滤波器
        hi_d: 高通分解滤波器
        level: 分解层数

    返回:
        (最终近似系数, [各级细节系数 level_1, level_2, ...])
        level_1 为最高频细节, 最适合检测瞬态伪迹
    """
    approx = sig
    details: List[np.ndarray] = []
    for _ in range(level):
        a_lo = scipy_signal.convolve(approx, lo_d, mode='full')
        a_hi = scipy_signal.convolve(approx, hi_d, mode='full')
        # 隔点下采样
        a_lo = a_lo[1::2]
        a_hi = a_hi[1::2]
        details.append(a_hi)
        approx = a_lo
    return approx, details


def detect_by_wavelet(data, fs, wavelet='db4', level=4, threshold=3.0) -> Dict:
    """
    小波伪迹检测

    用多级离散小波分解提取高频细节系数, 在细节系数上用稳健 z-score
    (中位数 + MAD) 检测异常值, |z| > threshold 标记为伪迹

    参数:
        data: (n_samples, n_channels) 或 (n_samples,) EEG 数据
        fs: 采样率
        wavelet: 小波类型 ('db4' 或 'haar')
        level: 分解层数 (第一层细节 = 最高频, 用于检测瞬态伪迹)
        threshold: 稳健 z-score 阈值

    返回:
        {'artifact_mask': bool数组, 'artifact_ratio': float,
         'artifact_segments': [(start, end), ...],
         'wavelet': str, 'level': int}
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_samples, n_channels = data.shape

    if wavelet == 'haar':
        lo_d, hi_d = _HAAR_LO_D, _HAAR_HI_D
    else:
        lo_d, hi_d = _DB4_LO_D, _DB4_HI_D

    # 限制层数 (信号长度需足够)
    max_level = max(1, int(np.floor(np.log2(max(n_samples, 2)))) - 1)
    level = min(level, max_level)

    mask = np.zeros(n_samples, dtype=bool)

    for ch in range(n_channels):
        sig = data[:, ch]
        _, details = _dwt_decompose(sig, lo_d, hi_d, level)
        # 第一级细节 = 最高频, 最适合检测瞬态伪迹
        d1 = details[0]
        if len(d1) < 2:
            continue
        # 稳健 z-score (中位数 + MAD, 抗异常值)
        med = np.median(d1)
        mad = np.median(np.abs(d1 - med)) * 1.4826 + 1e-12
        z = np.abs(d1 - med) / mad
        # 异常点索引
        anom_idx = np.where(z > threshold)[0]
        # 细节系数索引 -> 原信号索引 (每级下采样 2 倍, detail[i] ≈ 原信号 2*i+1)
        for di in anom_idx:
            orig = 2 * di + 1
            for off in (-1, 0, 1):
                idx = orig + off
                if 0 <= idx < n_samples:
                    mask[idx] = True

    segments = _mask_to_segments(mask)
    return {
        'artifact_mask': mask,
        'artifact_ratio': float(np.mean(mask)),
        'artifact_segments': segments,
        'wavelet': wavelet,
        'level': int(level),
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
    if n_samples == 0:
        return {'peak': [0.0]*n_channels, 'rms': [0.0]*n_channels,
                'variance': [0.0]*n_channels, 'kurtosis': [0.0]*n_channels,
                'skewness': [0.0]*n_channels, 'snr_db': [0.0]*n_channels}
    nperseg = min(1024, n_samples)
    if nperseg < 1:
        nperseg = 1
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
def _spectral_slope(freqs: np.ndarray, psd: np.ndarray) -> float:
    """1/f 频谱斜率估计 (log-log 线性回归, 1-30Hz 范围)

    典型 EEG 的斜率约为 -1 到 -2; ECG 低频斜率更陡 (< -1.5)
    """
    mask = (freqs >= 1) & (freqs <= 30) & (psd > 0)
    if mask.sum() < 3:
        return 0.0
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    slope = np.polyfit(log_f, log_p, 1)[0]
    return float(slope)


def _spatial_features(loading: np.ndarray) -> Dict:
    """从混合矩阵行向量计算空间分布特征

    参数:
        loading: (n_channels,) 某成分在各通道的负载

    返回:
        {'front_back_ratio': 前后通道负载比,
         'lr_asymmetry': 左右不对称性,
         'max_channel_index': 最大负载通道索引,
         'max_channel_relative': 相对位置 (0-1),
         'concentration': 负载集中度 (max/mean)}
    """
    loading = np.abs(np.asarray(loading, dtype=float))
    n_ch = len(loading)
    half = max(n_ch // 2, 1)
    front = float(np.mean(loading[:half]))
    back = float(np.mean(loading[half:])) if n_ch > half else 0.0
    front_back_ratio = front / (back + 1e-12)
    # 左右不对称 (用奇偶通道交替近似)
    if n_ch >= 4:
        left = float(np.mean(loading[::2]))
        right = float(np.mean(loading[1::2]))
        lr_asymmetry = abs(left - right) / (left + right + 1e-12)
    else:
        lr_asymmetry = 0.0
    max_ch = int(np.argmax(loading))
    concentration = float(np.max(loading)) / (float(np.mean(loading)) + 1e-12)
    return {
        'front_back_ratio': float(front_back_ratio),
        'lr_asymmetry': float(lr_asymmetry),
        'max_channel_index': max_ch,
        'max_channel_relative': float(max_ch / max(n_ch - 1, 1)),
        'concentration': float(concentration),
    }


def classify_components(components: np.ndarray, mixing: np.ndarray,
                        fs: int) -> List[Dict]:
    """
    分类独立成分为: 眼动(EOG)/肌电(EMG)/心电(ECG)/脑电(EEG)

    基于统计 + 频谱 + 空间特征:
        - 眼动(EOG): 低频(1-4Hz)能量占比 > 30% + 前通道负载高
        - 肌电(EMG): 高频(>30Hz)能量占比 > 40% + 高峭度(>5)
        - 心电(ECG): 1-3Hz 有明显周期性峰值 + 低频斜率陡 (< -1.5)
        - 脑电(EEG): 1-30Hz 主导 + 频谱斜率适中 (约 -1)

    参数:
        components: (n_samples, n_components) 源成分
        mixing: (n_components, n_channels) 混合矩阵
        fs: 采样率

    返回:
        [{'index': i, 'type': 'EOG/EMG/ECG/EEG',
          'confidence': float, 'reason': str,
          'spectral_slope': float, 'spatial_distribution': {...}}, ...]
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
        low_1_4 = float(np.sum(p[(f >= 1) & (f <= 4)]) / total)
        ecg_band = float(np.sum(p[(f >= 1) & (f <= 3)]) / total)
        slope = _spectral_slope(f, p)
        spatial = _spatial_features(mixing[i])
        info.append({
            'var': float(np.var(src)),
            'kurt': float(stats.kurtosis(src)),
            'skew': float(stats.skew(src)),
            'low': low, 'high': high, 'mid': mid,
            'low_1_4': low_1_4,
            'ecg_band': ecg_band,
            'dom_freq': float(f[np.argmax(p)]),
            'spectral_slope': slope,
            'spatial_distribution': spatial,
            'loading': np.abs(mixing[i]),
        })

    var_arr = np.array([x['var'] for x in info])
    load_arr = np.array([float(np.mean(x['loading'])) for x in info])
    var_median = float(np.median(var_arr)) if n_comp > 0 else 0.0
    load_median = float(np.median(load_arr)) if n_comp > 0 else 0.0

    results = []
    for i in range(n_comp):
        x = info[i]
        fb_ratio = x['spatial_distribution']['front_back_ratio']
        comp_type = 'EEG'
        confidence = 0.5
        reason = ''

        # EOG: 低频(1-4Hz)能量占比 > 30% + 前通道负载高 (front_back_ratio > 1.0)
        if x['low_1_4'] > 0.30 and fb_ratio > 1.0:
            comp_type = 'EOG'
            confidence = min(0.95, 0.55 + x['low_1_4'] * 0.3
                             + min(fb_ratio, 3.0) * 0.05)
            reason = (f'低频(1-4Hz)能量占比 {x["low_1_4"]:.2f} > 0.30，'
                      f'前后通道负载比 {fb_ratio:.2f}（前通道主导，眼动特征）')
        # EMG: 高频(>30Hz)能量占比 > 40% + 高峭度(>5)
        elif x['high'] > 0.40 and x['kurt'] > 5.0:
            comp_type = 'EMG'
            confidence = min(0.95, 0.55 + x['high'] * 0.3
                             + min(x['kurt'], 20.0) * 0.01)
            reason = (f'高频(>30Hz)能量占比 {x["high"]:.2f} > 0.40，'
                      f'峭度 {x["kurt"]:.2f} > 5（肌电特征）')
        # ECG: 1-3Hz 有明显周期性峰值 + 低频斜率陡 (< -1.5)
        elif (x['ecg_band'] > 0.20 and x['spectral_slope'] < -1.5
              and x['kurt'] > 3.0):
            comp_type = 'ECG'
            confidence = min(0.90, 0.55 + x['ecg_band'] * 0.3
                             + abs(x['spectral_slope']) * 0.05)
            reason = (f'1-3Hz 能量占比 {x["ecg_band"]:.2f}，'
                      f'频谱斜率 {x["spectral_slope"]:.2f} 陡，'
                      f'峭度 {x["kurt"]:.2f}（心电特征）')
        # EEG: 1-30Hz 主导 + 频谱斜率适中
        else:
            comp_type = 'EEG'
            # 斜率接近 -1 为典型 EEG
            slope_score = 1.0 - min(
                abs(x['spectral_slope'] - (-1.0)) / 1.5, 1.0)
            confidence = min(0.95, 0.5 + x['mid'] * 0.3
                             + slope_score * 0.15)
            reason = (f'1-30Hz 能量占比 {x["mid"]:.2f} 主导，'
                      f'频谱斜率 {x["spectral_slope"]:.2f} 适中（脑电特征）')

        results.append({
            'index': i,
            'type': comp_type,
            'confidence': float(confidence),
            'reason': reason,
            'spectral_slope': float(x['spectral_slope']),
            'spatial_distribution': x['spatial_distribution'],
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
def _method_consistency(*masks: np.ndarray) -> float:
    """计算多种检测方法的一致性 (平均 Jaccard 相似系数, 0-1)

    参数:
        *masks: 多个布尔掩码 (同长度)

    返回:
        一致性分数 (0-1, 越高表示各方法检测结果越一致)
    """
    masks = [np.asarray(m, dtype=bool) for m in masks]
    if len(masks) < 2:
        return 1.0
    jaccards = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a, b = masks[i], masks[j]
            if len(a) != len(b):
                continue
            inter = float(np.sum(a & b))
            union = float(np.sum(a | b))
            if union < 1e-12:
                jaccards.append(1.0)  # 两个都为空, 视为一致
            else:
                jaccards.append(inter / union)
    if not jaccards:
        return 1.0
    return float(np.mean(jaccards))


def quality_score(data: np.ndarray, fs: int) -> Dict:
    """
    综合数据质量评分 (0-100)

    基于: 伪迹比例(0.25) + SNR(0.20) + 稳定性(0.20) +
          频率分布(0.15) + 方法一致性(0.10) + 信号质量(0.10)

    参数:
        data: (n_samples, n_channels) EEG 数据
        fs: 采样率

    返回:
        {'score': float, 'grade': 'A/B/C/D', 'factors': {...}}
    """
    n_samples, n_channels = data.shape

    # 1. 伪迹比例（5 倍惩罚）
    thr = detect_by_threshold(data, fs, threshold_uv=100.0)
    zsc = detect_by_zscore(data)
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

    # 4. 频率分布合理性（各频段能量占比是否在合理范围）
    alpha_beta_ratio = None
    band_ratios: Dict[str, float] = {}
    freq_factor = 0.5
    try:
        nperseg = min(1024, n_samples)
        psd_avg = None
        f = None
        for ch in range(n_channels):
            f, p = scipy_signal.welch(data[:, ch], fs=fs, nperseg=nperseg)
            psd_avg = p if psd_avg is None else psd_avg + p
        psd_avg /= n_channels
        total_p = float(np.sum(psd_avg)) + 1e-12
        for name, (lo, hi) in BANDS.items():
            band_ratios[name] = float(
                np.sum(psd_avg[(f >= lo) & (f <= hi)]) / total_p
            )
        alpha_beta_ratio = band_ratios.get('alpha', 0) + band_ratios.get('beta', 0)
        if 0.2 <= alpha_beta_ratio <= 0.6:
            freq_factor = 1.0
        else:
            freq_factor = max(0.0, 1.0 - abs(alpha_beta_ratio - 0.4) * 2.0)
    except Exception:
        freq_factor = 0.5

    # 5. 方法一致性 (阈值法 vs Z-score vs 小波法)
    try:
        wav = detect_by_wavelet(data, fs)
        consistency_factor = _method_consistency(
            thr['artifact_mask'], zsc['artifact_mask'], wav['artifact_mask']
        )
    except Exception:
        consistency_factor = _method_consistency(
            thr['artifact_mask'], zsc['artifact_mask']
        )

    # 6. 信号质量 (峰值因子, 正常 EEG 约 3-6, 过高表示瞬态干扰)
    peak_arr = np.array(stats_res['peak'])
    rms_arr_stat = np.array(stats_res['rms'])
    crest_factor = peak_arr / (rms_arr_stat + 1e-12)
    signal_quality_factor = min(1.0, max(0.0, 1.0 - (np.mean(crest_factor) - 4.0) / 10.0))

    # 加权综合 (新权重)
    score = (
        artifact_factor * 0.25 +
        snr_factor * 0.20 +
        stability_factor * 0.20 +
        freq_factor * 0.15 +
        consistency_factor * 0.10 +
        signal_quality_factor * 0.10
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
        'band_ratios': band_ratios,
        'consistency_factor': float(consistency_factor),
        'signal_quality_factor': float(signal_quality_factor),
        'crest_factor': float(np.mean(crest_factor)),
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
    完整伪迹分析流水线 (含小波检测与改进的 ICA 分类)

    参数:
        data: (n_samples, n_channels) EEG 数据
        fs: 采样率

    返回:
        {
            'threshold_detection': ...,
            'zscore_detection': ...,
            'wavelet_detection': ...,   # 新增
            'signal_stats': ...,
            'ica_result': {'components': ..., 'mixing': ..., 'classification': ...},
            'quality': {'score': ..., 'grade': ..., 'factors': ...},
            'cleaned_data_shape': (n_samples, n_channels),
        }

    注意: components 数据降采样后转 list 返回（用于前端可视化），
          不返回完整 cleaned_data（太大），仅返回其形状。
    """
    # 检测 (阈值法 + Z-score + 小波法)
    thr_det = detect_by_threshold(data, fs)
    zsc_det = detect_by_zscore(data)
    wav_det = detect_by_wavelet(data, fs)
    sig_stats = compute_signal_stats(data, fs)

    # ICA (含改进的成分分类: 频谱斜率 + 空间分布)
    ica_raw = fast_ica(data)
    components = ica_raw['components']  # (n_samples, n_comp)
    mixing = ica_raw['mixing']         # (n_comp, n_channels)
    classification = classify_components(components, mixing, fs)

    # 质量 (含方法一致性 + 频率分布合理性 + 信号质量)
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
    wav_segs, wav_total, wav_trunc = _cap_segments(wav_det['artifact_segments'])

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
        'wavelet_detection': {
            'artifact_ratio': wav_det['artifact_ratio'],
            'artifact_segments': wav_segs,
            'n_segments': wav_total,
            'segments_truncated': wav_trunc,
            'wavelet': wav_det.get('wavelet', 'db4'),
            'level': wav_det.get('level', 4),
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
