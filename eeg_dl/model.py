"""EEG 深度学习模型 — 1D CNN 二分类

任务: 基于 4 通道 EEG 数据判断专注/放松状态
标签: Beta/Alpha 比值二值化(高于中位数 = 专注)
输入: (batch, 4, 1000) = 4 通道, 4 秒 @250Hz
输出: 2 类 (0=放松, 1=专注)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional
import json

from scipy import signal as scipy_signal
from app.analysis.openbci_import import load_brainflow_csv


# ========== 常量 ==========
TARGET_FS = 250              # 统一重采样到 250Hz
WINDOW_SEC = 4.0             # 4 秒窗口
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_FS)  # 1000
N_CHANNELS = 4               # Ganglion 4 通道


# ========== 数据集 ==========

class EEGDataset(Dataset):
    """EEG 窗口数据集

    每个 sample: (N_CHANNELS, WINDOW_SAMPLES) + label
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        """windows: (N, channels, samples), labels: (N,)"""
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def load_and_segment(filepath: str, target_fs: int = TARGET_FS) -> Tuple[np.ndarray, int]:
    """加载 CSV 并切分为 4 秒窗口

    返回:
        windows: (N, channels, WINDOW_SAMPLES)
        fs: 实际采样率
    """
    result = load_brainflow_csv(Path(filepath))
    data = result['data']  # (n_samples, n_channels)
    fs = result['fs']

    # 重采样到 target_fs
    if fs != target_fs:
        new_length = int(len(data) * target_fs / fs)
        data = scipy_signal.resample(data, new_length, axis=0)
        fs = target_fs

    # 只取前 N_CHANNELS 通道
    if data.shape[1] > N_CHANNELS:
        data = data[:, :N_CHANNELS]
    elif data.shape[1] < N_CHANNELS:
        raise ValueError(f"需要 {N_CHANNELS} 通道, 文件只有 {data.shape[1]} 通道")

    # 标准化(每通道 z-score)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - mean) / std

    # 切分窗口
    window_samples = int(WINDOW_SEC * fs)
    n_windows = len(data) // window_samples
    if n_windows == 0:
        raise ValueError(f"数据不足一个窗口({WINDOW_SEC}s)")

    data = data[:n_windows * window_samples]
    windows = data.reshape(n_windows, window_samples, N_CHANNELS)
    # 转为 (N, channels, samples)
    windows = windows.transpose(0, 2, 1)

    return windows, fs


def compute_band_power(segment: np.ndarray, fs: int, band: tuple) -> float:
    """计算某频带的平均功率

    segment: (channels, samples)
    """
    freqs, psd = scipy_signal.welch(segment, fs=fs, nperseg=min(256, segment.shape[1]))
    band_mask = (freqs >= band[0]) & (freqs < band[1])
    return float(np.mean(psd[:, band_mask]))


def generate_labels(windows: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """用 Beta/Alpha 比值生成二分类标签

    高于中位数 = 专注(1), 低于 = 放松(0)
    """
    alpha_band = (8, 13)
    beta_band = (13, 30)

    ratios = []
    for win in windows:
        alpha_power = compute_band_power(win, fs, alpha_band)
        beta_power = compute_band_power(win, fs, beta_band)
        ratio = beta_power / (alpha_power + 1e-8)
        ratios.append(ratio)

    ratios = np.array(ratios)
    median_ratio = np.median(ratios)
    labels = (ratios > median_ratio).astype(int)
    return labels


# ========== 模型 ==========

class EEGNet(nn.Module):
    """简单 1D CNN

    结构:
        Conv1d(4→16, k=25) → BN → ReLU → MaxPool(4)
        Conv1d(16→32, k=10) → BN → ReLU → MaxPool(4)
        Conv1d(32→64, k=5) → BN → ReLU → MaxPool(4)
        AdaptiveAvgPool(1) → Flatten
        Linear(64→32) → ReLU → Dropout(0.3)
        Linear(32→2)
    """

    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=25, padding=12),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=10, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ========== 训练 ==========

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = 'cpu',
    progress_callback=None,
) -> Tuple[EEGNet, dict]:
    """训练模型

    progress_callback(epoch, train_loss, val_loss, train_acc, val_acc)
    返回: (model, history)
    """
    model = EEGNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        train_loss = total_loss / total
        train_acc = correct / total

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item() * len(y)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += len(y)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if progress_callback:
            progress_callback(epoch + 1, train_loss, val_loss, train_acc, val_acc)

    return model, history


def predict_model(model: EEGNet, windows: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """推理

    windows: (N, channels, samples)
    返回: (N,) 预测标签
    """
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        X = torch.FloatTensor(windows).to(device)
        out = model(X)
        preds = out.argmax(1).cpu().numpy()
    return preds


def save_model(model: EEGNet, path: str, meta: dict = None):
    """保存模型"""
    save_dict = {
        'model_state': model.state_dict(),
        'meta': meta or {},
    }
    torch.save(save_dict, path)


def load_model(path: str, device: str = 'cpu') -> Tuple[EEGNet, dict]:
    """加载模型"""
    save_dict = torch.load(path, map_location=device, weights_only=False)
    model = EEGNet()
    model.load_state_dict(save_dict['model_state'])
    model = model.to(device)
    return model, save_dict.get('meta', {})
