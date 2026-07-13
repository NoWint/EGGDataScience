"""
EEG 心流恢复分析核心模块
跨学科任务切换对心流状态的影响及EEG恢复时间量化研究
"""
import numpy as np
import pandas as pd
from scipy import signal, stats
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ========== 频段定义 ==========
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}


# ========== 1. 数据加载 ==========
def load_eeg_full(filepath):
    """加载 EEG 文件,自动检测格式,返回完整 dict

    支持格式:
    - OpenBCI ODF (.txt,带 %OpenBCI 头)
    - BrainFlow CSV (.csv,数字索引列名)
    - 普通 CSV (time + channel columns)

    返回:
        {
            'data': np.ndarray (n_samples, n_exg),     # EXG, μV
            'fs': int,                                  # 采样率
            'channels': List[str],                      # EXG 通道名
            'times': np.ndarray (n_samples,),           # 时间轴(秒)
            'accel': np.ndarray | None,                 # (n_samples, 3) g
            'markers': List[Marker] | None,             # 事件标记
            'metadata': dict,                           # 板卡/格式/通道数等
        }
    """
    from pathlib import Path
    fp = Path(filepath)
    from .openbci_import import (_detect_openbci, load_openbci,
                                 _detect_brainflow_csv, load_brainflow_csv)

    if _detect_openbci(fp):
        return load_openbci(fp)

    if _detect_brainflow_csv(fp):
        return load_brainflow_csv(fp)

    # 普通 CSV
    df = pd.read_csv(filepath)
    time_cols = [c for c in df.columns if c.lower() in ('time', 'timestamp', 't', '时间')]
    if time_cols:
        times = df[time_cols[0]].values
        data_cols = [c for c in df.columns if c not in time_cols]
    else:
        times = np.arange(len(df)) / 250.0
        data_cols = list(df.columns)

    data = df[data_cols].values.astype(np.float64)
    if len(times) > 1:
        dt = np.median(np.diff(times))
        fs = int(round(1.0 / dt)) if dt > 0 else 250
    else:
        fs = 250

    n_samples = len(data)
    return {
        'data': data,
        'fs': fs,
        'channels': list(data_cols),
        'times': times,
        'accel': None,
        'markers': None,
        'metadata': {
            'format': 'plain_csv',
            'board': 'unknown',
            'n_channels': data.shape[1],
            'sample_rate': fs,
            'has_accelerometer': False,
            'has_markers': False,
            'duration_sec': float(n_samples / fs) if fs > 0 else 0.0,
            'n_samples': int(n_samples),
        }
    }


def load_eeg(filepath):
    """加载 EEG 文件,返回 (data, fs, channels, times) 4 元组(向后兼容)

    内部调用 load_eeg_full() 取前 4 字段。
    """
    result = load_eeg_full(filepath)
    return result['data'], result['fs'], result['channels'], result['times']


def load_events(filepath):
    """加载事件标记 CSV: columns=[event_id, timestamp]"""
    df = pd.read_csv(filepath)
    return df['event_id'].tolist(), df['timestamp'].values


# ========== 2. 预处理 ==========
def preprocess(data, fs, lp=45.0, hp=1.0, notch=50.0, artifact_threshold=100.0):
    """
    预处理流程: 带通 + 陷波 + 伪迹剔除
    data: (n_samples, n_channels) 单位 μV
    """
    n_samples, n_channels = data.shape
    processed = data.copy()

    # 带通滤波 (1-45 Hz)
    nyq = fs / 2.0
    b, a = signal.butter(4, [hp / nyq, lp / nyq], btype='band')
    for ch in range(n_channels):
        processed[:, ch] = signal.filtfilt(b, a, processed[:, ch])

    # 陷波滤波 (50 Hz 工频)
    if notch and notch < nyq - 1:
        Q = 30.0
        b, a = signal.iirnotch(notch, Q, fs)
        for ch in range(n_channels):
            processed[:, ch] = signal.filtfilt(b, a, processed[:, ch])

    # 伪迹剔除: 阈值法，超限窗口线性插值
    artifact_ratio = 0.0
    for ch in range(n_channels):
        ch_data = processed[:, ch]
        mask = np.abs(ch_data) > artifact_threshold
        artifact_ratio += mask.mean()
        if mask.any():
            idx_bad = np.where(mask)[0]
            idx_good = np.where(~mask)[0]
            if len(idx_good) > 0:
                processed[idx_bad, ch] = np.interp(idx_bad, idx_good, ch_data[idx_good])

    artifact_ratio /= n_channels
    return processed, artifact_ratio


def preprocess_advanced(data, fs, config=None):
    """高级预处理（参数化版本，复用 data_utils）
    config: {'lp': 45.0, 'hp': 1.0, 'notch': 50.0, 'resample_fs': None, ...}
    返回: {'data': ..., 'fs': ..., 'steps': [...]}
    """
    from app.analysis.data_utils import preprocess_pipeline
    return preprocess_pipeline(data, fs, config)


# ========== 3. 特征提取 ==========
def compute_band_powers(data, fs, window_sec=2.0, overlap=0.5):
    """
    用 Welch 法计算各频段相对功率
    返回: dict {band_name: 1D array}, 每秒一个值
    """
    n_samples, n_channels = data.shape
    win_samples = int(window_sec * fs)
    step = int(win_samples * (1 - overlap))
    n_windows = (n_samples - win_samples) // step + 1

    band_powers = {b: np.zeros(n_windows) for b in BANDS}

    for i in range(n_windows):
        start = i * step
        end = start + win_samples
        seg = data[start:end, :]  # (win_samples, n_channels)

        # 多通道平均功率谱
        psd_total = np.zeros(win_samples // 2 + 1)
        for ch in range(n_channels):
            freqs, psd = signal.welch(seg[:, ch], fs, nperseg=min(win_samples, len(seg[:, ch])))
            psd_total += psd
        psd_total /= n_channels
        total_power = psd_total.sum()

        for bname, (flo, fhi) in BANDS.items():
            idx = (freqs >= flo) & (freqs < fhi)
            band_powers[bname][i] = psd_total[idx].sum() / (total_power + 1e-12)

    # 特征时间轴 (秒)
    feat_times = np.arange(n_windows) * step / fs + window_sec / 2
    return band_powers, feat_times


def compute_entropy(data, fs, window_sec=2.0, overlap=0.5):
    """计算功率谱 Shannon 熵"""
    n_samples, n_channels = data.shape
    win_samples = int(window_sec * fs)
    step = int(win_samples * (1 - overlap))
    n_windows = (n_samples - win_samples) // step + 1

    entropy = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * step
        seg = data[start:start + win_samples, :]
        psd_total = np.zeros(win_samples // 2 + 1)
        for ch in range(n_channels):
            _, psd = signal.welch(seg[:, ch], fs, nperseg=min(win_samples, len(seg[:, ch])))
            psd_total += psd
        # 归一化为概率分布
        p = psd_total / (psd_total.sum() + 1e-12)
        p = p[p > 0]
        entropy[i] = -np.sum(p * np.log(p))
    return entropy


def extract_features(data, fs, window_sec=2.0, overlap=0.5, smooth_window=5):
    """
    提取全部6个指标的时间序列
    smooth_window: 滑动平均窗口大小(点数), 0=不平滑
    返回: DataFrame, 列=指标名, 含timestamp列
    """
    band_powers, feat_times = compute_band_powers(data, fs, window_sec, overlap)
    entropy = compute_entropy(data, fs, window_sec, overlap)

    theta = band_powers['theta']
    alpha = band_powers['alpha']
    beta = band_powers['beta']
    gamma = band_powers['gamma']

    # 滑动平均平滑 (减少 Welch 估计的逐窗抖动)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        theta = np.convolve(theta, kernel, mode='same')
        alpha = np.convolve(alpha, kernel, mode='same')
        beta = np.convolve(beta, kernel, mode='same')
        gamma = np.convolve(gamma, kernel, mode='same')
        entropy = np.convolve(entropy, kernel, mode='same')

    features = pd.DataFrame({
        'timestamp': feat_times,
        'theta_alpha_ratio': theta / (alpha + 1e-12),
        'alpha_rel': alpha,
        'beta_rel': beta,
        'gamma_rel': gamma,
        'eeg_entropy': entropy,
        'cog_load': (theta + beta) / (alpha + 1e-12),
    })
    return features


# ========== 4. 恢复时长计算 ==========
def compute_recovery_time(feature_series, baseline_mean, recovery_start_idx,
                          window_sec=30, fs_feat=1.0, tolerance=0.05):
    """
    恢复时长量化算法
    判定: 切换结束后，指标连续30s回归至稳态均值±5%区间
    返回: recovery_time_sec 或 None(未恢复)
    """
    lo = baseline_mean * (1 - tolerance)
    hi = baseline_mean * (1 + tolerance)
    win = int(window_sec * fs_feat)

    series = np.asarray(feature_series)[recovery_start_idx:]
    n = len(series)

    if n < win:
        return None

    in_range = (series >= lo) & (series <= hi)
    # 滑窗: 找第一个起点i，使 in_range[i:i+win] 全为True
    for i in range(n - win + 1):
        if np.all(in_range[i:i + win]):
            recovery_time = (i + win) / fs_feat
            return recovery_time
    return None


def compute_all_recovery(features_df, baseline_means, recovery_start_time,
                         feature_names=None, tolerance=0.05, window_sec=30):
    """
    对所有指标计算恢复时长
    features_df: 含 timestamp 列的特征 DataFrame
    baseline_means: dict {feature: mean}
    recovery_start_time: 恢复期开始的时间(秒)
    """
    if feature_names is None:
        feature_names = ['theta_alpha_ratio', 'alpha_rel', 'beta_rel',
                        'gamma_rel', 'eeg_entropy', 'cog_load']

    ts = features_df['timestamp'].values
    fs_feat = 1.0 / np.median(np.diff(ts)) if len(ts) > 1 else 1.0
    recovery_start_idx = np.searchsorted(ts, recovery_start_time)

    results = {}
    for name in feature_names:
        if name not in features_df.columns or name not in baseline_means:
            results[name] = None
            continue
        series = features_df[name].values
        rt = compute_recovery_time(series, baseline_means[name], recovery_start_idx,
                                    window_sec=window_sec, fs_feat=fs_feat,
                                    tolerance=tolerance)
        results[name] = rt

    # 综合恢复时长 = 最慢指标达标时间
    valid_rts = [v for v in results.values() if v is not None]
    overall = max(valid_rts) if valid_rts else None
    return overall, results


# ========== 5. 衰减幅度 ==========
def compute_attenuation(features_df, baseline_means, switch_start_time, switch_end_time):
    """
    衰减幅度 = (稳态均值 - 切换期极值) / 稳态均值 × 100%
    对心流核心指标取最小值(跌落)，对认知损耗指标取最大值(升高)
    """
    ts = features_df['timestamp'].values
    mask = (ts >= switch_start_time) & (ts <= switch_end_time)

    flow_indicators = ['theta_alpha_ratio', 'alpha_rel', 'beta_rel']
    loss_indicators = ['gamma_rel', 'eeg_entropy', 'cog_load']

    attenuation = {}
    for name in flow_indicators:
        if name in features_df.columns and mask.any():
            switch_min = features_df.loc[mask, name].min()
            attenuation[name] = (baseline_means[name] - switch_min) / (baseline_means[name] + 1e-12) * 100
        else:
            attenuation[name] = 0.0

    for name in loss_indicators:
        if name in features_df.columns and mask.any():
            switch_max = features_df.loc[mask, name].max()
            att = (switch_max - baseline_means[name]) / (baseline_means[name] + 1e-12) * 100
            attenuation[name] = min(att, 300.0)  # 限幅, 避免比率型指标极端值
        else:
            attenuation[name] = 0.0

    return attenuation


# ========== 6. 统计分析 ==========
def paired_t_test(group_a, group_b):
    """配对样本t检验 (被试内设计)"""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return {'t': 0, 'p': 1.0, 'd': 0, 'mean_diff': 0, 'n': n}
    t, p = stats.ttest_rel(a[:n], b[:n])
    diff = a[:n] - b[:n]
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)  # Cohen's d
    return {'t': float(t) if not np.isnan(t) else 0.0,
            'p': float(p) if not np.isnan(p) else 1.0,
            'd': float(d), 'mean_diff': float(diff.mean()), 'n': int(n)}


def repeated_measures_anova(groups):
    """
    重复测量单因素ANOVA
    groups: list of arrays, 每组同一被试的测量值
    """
    k = len(groups)
    n = min(len(g) for g in groups)
    if n < 2 or k < 2:
        return {'F': 0, 'p': 1.0, 'eta2': 0}

    data = np.array([g[:n] for g in groups]).T  # (n_subjects, k_conditions)
    subj_means = data.mean(axis=1, keepdims=True)
    cond_means = data.mean(axis=0)
    grand_mean = data.mean()

    ss_between = n * np.sum((cond_means - grand_mean) ** 2)
    ss_within = np.sum((data - cond_means) ** 2)
    ss_subject = k * np.sum((subj_means.flatten() - grand_mean) ** 2)
    ss_error = ss_within - ss_subject

    df_between = k - 1
    df_error = (n - 1) * (k - 1)
    ms_error = ss_error / (df_error + 1e-12)
    F = (ss_between / df_between) / (ms_error + 1e-12)
    p = 1 - stats.f.cdf(F, df_between, df_error)
    eta2 = ss_between / (ss_between + ss_within + 1e-12)
    return {'F': float(F), 'p': float(p), 'eta2': float(eta2),
            'df1': int(df_between), 'df2': int(df_error)}


def pearson_correlation(x, y):
    """Pearson相关分析"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return {'r': 0, 'p': 1.0, 'n': n}
    r, p = stats.pearsonr(x[:n], y[:n])
    return {'r': float(r) if not np.isnan(r) else 0.0,
            'p': float(p) if not np.isnan(p) else 1.0, 'n': int(n)}


# ========== 7. 样例数据生成 (用于工具箱演示) ==========
def generate_sample_eeg(fs=250, duration_sec=25 * 60, n_channels=3, seed=42, disruption=0.0):
    """
    生成模拟EEG数据, 模拟完整实验时序:
    - 0-1min: 静息基线
    - 1-9min: 心流诱发 (后4min为稳态)
    - 9-11min: 切换干扰
    - 11-21min: 恢复期

    disruption: 0.0=对照组(无切换扰动), 1.0=中等切换, 2.0=强切换
    返回: data, times, events
    """
    rng = np.random.default_rng(seed)
    n_samples = int(fs * duration_sec)
    t = np.arange(n_samples) / fs
    d = disruption  # 切换破坏强度系数

    data = np.zeros((n_samples, n_channels))

    # 构建各频段的时间包络 (向量化, 高效且平滑)
    def smooth_envelope(keypoints):
        kp_times = np.array([kp[0] for kp in keypoints])
        kp_vals = np.array([kp[1] for kp in keypoints])
        return np.interp(t, kp_times, kp_vals)

    # 心流稳态目标值
    steady = {'delta': 0.15, 'theta': 0.22, 'alpha': 0.35, 'beta': 0.28, 'gamma': 0.08}

    # 切换期目标值 (受 disruption 调制)
    # d=0 时与稳态一致, d越大偏离越严重
    switch_vals = {
        'delta': steady['delta'],
        'theta': steady['theta'] + 0.06 * d,
        'alpha': steady['alpha'] - 0.17 * d,   # Alpha 跌落
        'beta':  steady['beta']  + 0.10 * d,   # Beta 升高
        'gamma': steady['gamma'] + 0.14 * d,   # Gamma 飙升
    }

    # 恢复时间 (秒): disruption越大恢复越慢
    recovery_time_target = 60 + 120 * d  # 1min~5min

    envelopes = {}
    for bname in BANDS:
        s_val = steady[bname]
        x_val = switch_vals[bname]
        # 恢复期: 从切换值平滑回归到稳态值
        r_end = 665 + recovery_time_target
        envelopes[bname] = smooth_envelope([
            (0, 0.25), (60, 0.25),                    # 基线
            (300, s_val), (540, s_val),               # 心流稳态
            (545, x_val), (665, x_val),               # 切换期
            (665, x_val), (r_end, s_val), (1500, s_val),  # 恢复期
        ])

    # 生成各频段信号
    signal_amp = 50.0  # 高信噪比
    noise_std = 0.5

    for ch in range(n_channels):
        for bname, (flo, fhi) in BANDS.items():
            freqs = np.linspace(flo + 0.5, fhi - 0.5, 10)
            for f in freqs:
                phase = rng.uniform(0, 2 * np.pi)
                env = envelopes[bname]
                flutter = 1.0 + rng.normal(0, 0.015, n_samples)
                data[:, ch] += env * flutter * np.sin(2 * np.pi * f * t + phase) * signal_amp / len(freqs)

    data += rng.normal(0, noise_std, data.shape)

    events = [
        ('S0', 0.0), ('B0', 5.0), ('B1', 65.0),
        ('F0', 65.0), ('F1', 305.0), ('F2', 545.0),
        ('X0', 545.0), ('X1', 665.0),
        ('R0', 665.0), ('R1', 1265.0),
        ('Q0', 1265.0),
    ]
    return data, t, events


def events_to_df(events):
    """事件列表转 DataFrame"""
    return pd.DataFrame(events, columns=['event_id', 'timestamp'])


# ========== 8. 完整分析流水线 ==========
def run_full_pipeline(data, fs, events_df, config=None, preprocess_config=None):
    """
    运行完整分析流水线
    preprocess_config: 可选，若提供则用 preprocess_advanced 替代默认 preprocess
    返回: dict 含所有结果
    """
    if config is None:
        config = {
            'lp': 45.0, 'hp': 1.0, 'notch': 50.0,
            'artifact_threshold': 100.0,
            'window_sec': 2.0, 'overlap': 0.5, 'smooth_window': 7,
            'tolerance': 0.05, 'recovery_window': 30,
        }

    # 解析事件时间
    evt = dict(zip(events_df['event_id'], events_df['timestamp']))
    flow_steady_start = evt.get('F1', 300.0)
    flow_steady_end = evt.get('F2', evt.get('X0', 540.0))
    switch_start = evt.get('X0', 540.0)
    switch_end = evt.get('X1', evt.get('R0', 660.0))
    recovery_start = evt.get('R0', 660.0)

    # 1. 预处理
    if preprocess_config is not None:
        # 使用高级参数化预处理（复用 data_utils.preprocess_pipeline）
        prep_result = preprocess_advanced(data, fs, preprocess_config)
        processed = prep_result['data']
        fs = prep_result['fs']  # 采样率可能因重采样改变
        artifact_ratio = 0.0    # 高级预处理不做伪迹剔除
    else:
        processed, artifact_ratio = preprocess(
            data, fs,
            lp=config['lp'], hp=config['hp'], notch=config['notch'],
            artifact_threshold=config['artifact_threshold']
        )

    # 2. 特征提取
    features = extract_features(processed, fs,
                                window_sec=config['window_sec'],
                                overlap=config['overlap'],
                                smooth_window=config.get('smooth_window', 7))

    # 3. 计算心流稳态基准 (F1-F2 后4分钟)
    ts = features['timestamp'].values
    steady_mask = (ts >= flow_steady_start) & (ts <= flow_steady_end)
    feature_cols = ['theta_alpha_ratio', 'alpha_rel', 'beta_rel',
                    'gamma_rel', 'eeg_entropy', 'cog_load']
    baseline_means = {}
    for col in feature_cols:
        if steady_mask.any():
            baseline_means[col] = float(features.loc[steady_mask, col].mean())
        else:
            baseline_means[col] = 0.0

    # 4. 恢复时长
    overall_recovery, per_feature_recovery = compute_all_recovery(
        features, baseline_means, recovery_start,
        feature_names=feature_cols,
        tolerance=config['tolerance'],
        window_sec=config['recovery_window']
    )

    # 5. 衰减幅度
    attenuation = compute_attenuation(features, baseline_means,
                                      switch_start, switch_end)

    # 6. 准备可视化数据 (归一化到稳态均值=1.0)
    viz_data = {}
    for col in feature_cols:
        if baseline_means[col] > 0:
            viz_data[col] = (features[col].values / baseline_means[col]).tolist()
        else:
            viz_data[col] = features[col].values.tolist()
    viz_data['timestamp'] = ts.tolist()

    return {
        'features': features.to_dict('list'),
        'baseline_means': baseline_means,
        'recovery_time': overall_recovery,
        'recovery_per_feature': per_feature_recovery,
        'attenuation': attenuation,
        'artifact_ratio': float(artifact_ratio),
        'viz_data': viz_data,
        'event_times': {
            'flow_steady_start': float(flow_steady_start),
            'flow_steady_end': float(flow_steady_end),
            'switch_start': float(switch_start),
            'switch_end': float(switch_end),
            'recovery_start': float(recovery_start),
        },
        'config': config,
    }
