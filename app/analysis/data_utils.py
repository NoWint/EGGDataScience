"""
EEG 数据处理工具模块
提供数据验证、分块处理、降采样、预处理参数化
"""
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from typing import Tuple, Optional, Dict, List, Any, Callable
from math import gcd


def validate_eeg_data(data, fs=None, min_duration_sec=10, max_channels=32):
    """验证 EEG 数据格式和质量
    检查: 形状、采样率、NaN/Inf、幅值范围、通道数、最小时长
    返回: {'valid': bool, 'errors': [str], 'warnings': [str],
           'info': {'n_samples', 'n_channels', 'duration_sec', 'fs', 'channels'}}
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 转为 numpy 数组
    try:
        arr = np.asarray(data, dtype=np.float64)
    except Exception as e:
        return {
            'valid': False,
            'errors': [f'数据无法转为数值数组: {e}'],
            'warnings': [],
            'info': {'n_samples': 0, 'n_channels': 0, 'duration_sec': 0.0,
                     'fs': fs, 'channels': []},
        }

    # 形状检查
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        warnings.append('输入为 1D 数组，已视为单通道数据')
    elif arr.ndim != 2:
        errors.append(f'数据维度应为 1D 或 2D，实际为 {arr.ndim}D')
        return {
            'valid': False,
            'errors': errors,
            'warnings': warnings,
            'info': {'n_samples': 0, 'n_channels': 0, 'duration_sec': 0.0,
                     'fs': fs, 'channels': []},
        }

    n_samples, n_channels = arr.shape

    # 样本数检查
    if n_samples == 0:
        errors.append('数据样本数为 0')

    # 采样率检查
    fs_eff: Optional[float] = None
    if fs is not None:
        if not isinstance(fs, (int, float)) or fs <= 0:
            errors.append(f'采样率无效: {fs}')
        else:
            fs_eff = float(fs)
    else:
        warnings.append('未提供采样率 fs，无法计算时长')

    # 时长计算
    duration_sec = (n_samples / fs_eff) if fs_eff else 0.0

    # 通道数检查
    if n_channels > max_channels:
        warnings.append(f'通道数 {n_channels} 超过建议上限 {max_channels}')

    # 最小时长检查
    if fs_eff and duration_sec < min_duration_sec:
        warnings.append(f'数据时长 {duration_sec:.2f}s 低于建议最小 {min_duration_sec}s')

    # NaN/Inf 检查
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan > 0:
        errors.append(f'数据包含 {n_nan} 个 NaN 值')
    if n_inf > 0:
        errors.append(f'数据包含 {n_inf} 个 Inf 值')

    # 幅值范围检查
    if n_samples > 0 and n_nan == 0 and n_inf == 0:
        abs_max = float(np.abs(arr).max())
        if abs_max > 1e4:
            warnings.append(f'幅值最大 {abs_max:.2f}，可能存在单位错误或伪迹')
        if abs_max < 1e-3:
            warnings.append(f'幅值最大 {abs_max:.6f}，可能单位为 V 而非 μV')

    channels = [f'ch{i}' for i in range(n_channels)]
    valid = len(errors) == 0

    return {
        'valid': valid,
        'errors': errors,
        'warnings': warnings,
        'info': {
            'n_samples': int(n_samples),
            'n_channels': int(n_channels),
            'duration_sec': float(duration_sec),
            'fs': fs_eff,
            'channels': channels,
        },
    }


def resample_data(data, fs_orig, fs_target):
    """重采样 EEG 数据
    用 scipy.signal.resample_poly（抗混叠）
    返回: (resampled_data, fs_target)
    """
    arr = np.asarray(data, dtype=np.float64)
    was_1d = False
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        was_1d = True

    fs_orig = int(fs_orig)
    fs_target = int(fs_target)

    if fs_orig == fs_target:
        return arr, fs_target

    # 计算互质的 up / down
    g = gcd(fs_orig, fs_target)
    up = fs_target // g
    down = fs_orig // g

    # 沿时间轴（axis=0）重采样
    resampled = scipy_signal.resample_poly(arr, up, down, axis=0)

    if was_1d:
        resampled = resampled.flatten()
    return resampled, fs_target


def apply_filters(data, fs, lp=None, hp=None, notch=None, order=4):
    """应用滤波器（参数化）
    - lp: 低通截止 (Hz)
    - hp: 高通截止 (Hz)
    - notch: 陷波频率 (Hz)，默认 50Hz 工频
    用 scipy.signal.butter + filtfilt（零相位）
    返回: 滤波后数据
    """
    arr = np.asarray(data, dtype=np.float64)
    was_1d = False
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        was_1d = True

    n_samples, n_channels = arr.shape
    nyq = fs / 2.0
    processed = arr.copy()

    # 带通/低通/高通
    if lp is not None and hp is not None:
        if lp > nyq or hp > nyq or hp >= lp:
            raise ValueError(f'滤波参数无效: lp={lp}, hp={hp}, nyq={nyq}')
        b, a = scipy_signal.butter(order, [hp / nyq, lp / nyq], btype='band')
        for ch in range(n_channels):
            processed[:, ch] = scipy_signal.filtfilt(b, a, processed[:, ch])
    elif lp is not None:
        if lp > nyq:
            raise ValueError(f'低通截止 {lp} 超过奈奎斯特频率 {nyq}')
        b, a = scipy_signal.butter(order, lp / nyq, btype='low')
        for ch in range(n_channels):
            processed[:, ch] = scipy_signal.filtfilt(b, a, processed[:, ch])
    elif hp is not None:
        if hp > nyq:
            raise ValueError(f'高通截止 {hp} 超过奈奎斯特频率 {nyq}')
        b, a = scipy_signal.butter(order, hp / nyq, btype='high')
        for ch in range(n_channels):
            processed[:, ch] = scipy_signal.filtfilt(b, a, processed[:, ch])

    # 陷波（默认 50Hz 工频，仅当显式传入时应用）
    if notch is not None:
        if notch < nyq - 1:
            Q = 30.0
            b, a = scipy_signal.iirnotch(notch, Q, fs)
            for ch in range(n_channels):
                processed[:, ch] = scipy_signal.filtfilt(b, a, processed[:, ch])

    if was_1d:
        processed = processed.flatten()
    return processed


def preprocess_pipeline(data, fs, config=None):
    """参数化预处理流水线
    config: {
        'lp': 45.0, 'hp': 1.0, 'notch': 50.0,
        'resample_fs': None,  # None=不重采样
        'remove_dc': True,    # 去直流
        'normalize': False,   # z-score 归一化
    }
    返回: {'data': 处理后数据, 'fs': 实际采样率, 'steps': [执行的步骤描述]}
    """
    if config is None:
        config = {}

    lp = config.get('lp', 45.0)
    hp = config.get('hp', 1.0)
    notch = config.get('notch', 50.0)
    resample_fs = config.get('resample_fs', None)
    remove_dc = config.get('remove_dc', True)
    normalize = config.get('normalize', False)

    steps: List[str] = []
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # 1. 去直流
    if remove_dc:
        arr = arr - arr.mean(axis=0, keepdims=True)
        steps.append('去直流（去除均值）')

    # 2. 滤波
    if lp is not None or hp is not None or notch is not None:
        arr = apply_filters(arr, fs, lp=lp, hp=hp, notch=notch)
        filt_desc: List[str] = []
        if lp is not None and hp is not None:
            filt_desc.append(f'带通 {hp}-{lp} Hz')
        elif lp is not None:
            filt_desc.append(f'低通 {lp} Hz')
        elif hp is not None:
            filt_desc.append(f'高通 {hp} Hz')
        if notch is not None:
            filt_desc.append(f'陷波 {notch} Hz')
        steps.append('滤波: ' + ', '.join(filt_desc))

    # 3. 重采样
    fs_eff: Any = fs
    if resample_fs is not None and int(resample_fs) != int(fs):
        arr, fs_eff = resample_data(arr, fs, resample_fs)
        steps.append(f'重采样 {fs} -> {fs_eff} Hz')

    # 4. z-score 归一化
    if normalize:
        mu = arr.mean(axis=0, keepdims=True)
        sigma = arr.std(axis=0, keepdims=True)
        arr = (arr - mu) / (sigma + 1e-12)
        steps.append('z-score 归一化')

    return {
        'data': arr,
        'fs': fs_eff,
        'steps': steps,
    }


def downsample_for_display(data, target_points=1000, axis=0):
    """为显示降采样大数据
    用均匀抽取（不抗混叠，仅用于可视化）
    返回: 降采样后数据
    """
    arr = np.asarray(data)
    n = arr.shape[axis]
    if n <= target_points:
        return arr

    step = int(np.ceil(n / target_points))
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(0, n, step)
    return arr[tuple(slices)]


def chunk_process(data, fs, chunk_duration_sec=60, process_fn=None):
    """分块处理大数据
    将数据按 chunk_duration_sec 分块，逐块调用 process_fn
    用于内存友好的大文件处理
    返回: {'results': [各块结果], 'n_chunks': int, 'chunk_size': int}
    """
    arr = np.asarray(data)
    was_1d = False
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        was_1d = True

    n_samples = arr.shape[0]
    chunk_size = int(chunk_duration_sec * fs)
    if chunk_size <= 0:
        chunk_size = n_samples

    n_chunks = int(np.ceil(n_samples / chunk_size)) if chunk_size > 0 else 0
    results: List[Any] = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        chunk = arr[start:end, :]
        if was_1d:
            chunk = chunk.flatten()
        if process_fn is not None:
            results.append(process_fn(chunk, fs, i))
        else:
            results.append(chunk)

    return {
        'results': results,
        'n_chunks': n_chunks,
        'chunk_size': chunk_size,
    }


def estimate_memory_usage(n_samples, n_channels, dtype='float64'):
    """估算数据内存占用
    返回: {'bytes': int, 'mb': float, 'gb': float}
    """
    itemsize_map = {
        'float64': 8, 'float32': 4, 'float16': 2,
        'int64': 8, 'int32': 4, 'int16': 2, 'int8': 1,
        'complex128': 16, 'complex64': 8,
    }
    itemsize = itemsize_map.get(dtype, 8)
    total_bytes = int(n_samples) * int(n_channels) * itemsize
    return {
        'bytes': total_bytes,
        'mb': total_bytes / (1024.0 ** 2),
        'gb': total_bytes / (1024.0 ** 3),
    }


def auto_downsample_matrix(matrix, max_rows=50, max_cols=200):
    """自动降采样 2D 矩阵用于前端显示
    如果矩阵超过 max_rows x max_cols，均匀降采样
    返回: {'matrix': 降采样后, 'row_factor': int, 'col_factor': int, 'downsampled': bool}
    """
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        # 非 2D 矩阵，原样返回
        return {
            'matrix': arr,
            'row_factor': 1,
            'col_factor': 1,
            'downsampled': False,
        }

    n_rows, n_cols = arr.shape
    row_factor = 1
    col_factor = 1

    if n_rows > max_rows:
        row_factor = int(np.ceil(n_rows / max_rows))
    if n_cols > max_cols:
        col_factor = int(np.ceil(n_cols / max_cols))

    downsampled = (row_factor > 1) or (col_factor > 1)

    if downsampled:
        arr = arr[::row_factor, ::col_factor]

    return {
        'matrix': arr,
        'row_factor': row_factor,
        'col_factor': col_factor,
        'downsampled': downsampled,
    }
