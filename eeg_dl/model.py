"""EEG 深度学习模型 V2 — 增强版 1D CNN

改进:
  1. 多尺度卷积 (kernel 25/10/5 并行)
  2. 残差连接
  3. SE (Squeeze-and-Excitation) 注意力
  4. 频带功率 + 熵 特征融合
  5. ReduceLROnPlateau + Early Stopping

任务: 4 通道 EEG 专注/放松二分类
输入: (batch, 4, 1000) = 4 通道, 4 秒 @250Hz
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional
from scipy import signal as scipy_signal
from scipy.stats import entropy as scipy_entropy

from app.analysis.openbci_import import load_brainflow_csv


# ========== 常量 ==========
TARGET_FS = 250
WINDOW_SEC = 4.0
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_FS)  # 1000
N_CHANNELS = 4

# 频带定义
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}
N_BANDS = len(BANDS)  # 5
N_FEATURES = N_BANDS + N_CHANNELS  # 5 频带功率 + 4 通道熵 = 9


# ========== 特征工程 ==========

def extract_features(windows: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """提取手工特征: 5 频带相对功率 + 4 通道近似熵

    windows: (N, channels, samples)
    返回: (N, N_FEATURES)
    """
    n_windows = len(windows)
    features = np.zeros((n_windows, N_FEATURES))

    for i, win in enumerate(windows):
        # win: (channels, samples)
        all_band_powers = []
        total_power = 0

        # 计算各频带功率
        freqs, psd = scipy_signal.welch(win, fs=fs, nperseg=min(256, win.shape[1]))
        for band_name, (lo, hi) in BANDS.items():
            band_mask = (freqs >= lo) & (freqs < hi)
            band_power = float(np.mean(psd[:, band_mask]))
            all_band_powers.append(band_power)
            total_power += band_power

        # 相对功率
        for j, bp in enumerate(all_band_powers):
            features[i, j] = bp / (total_power + 1e-8)

        # 各通道近似熵 (用方差作为简化替代,计算更快)
        for ch in range(min(N_CHANNELS, win.shape[0])):
            features[i, N_BANDS + ch] = float(np.var(win[ch]))

    return features


# ========== 数据集 ==========

class EEGDataset(Dataset):
    """EEG 窗口数据集 (含手工特征)"""

    def __init__(self, windows: np.ndarray, labels: np.ndarray, features: np.ndarray = None):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)
        self.features = torch.FloatTensor(features) if features is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.features is not None:
            return self.windows[idx], self.features[idx], self.labels[idx]
        return self.windows[idx], self.labels[idx]


# ========== 数据加载 ==========

def load_and_segment(filepath: str, target_fs: int = TARGET_FS) -> Tuple[np.ndarray, int]:
    """加载 CSV 并切分为 4 秒窗口"""
    result = load_brainflow_csv(Path(filepath))
    data = result['data']
    fs = result['fs']

    if fs != target_fs:
        new_length = int(len(data) * target_fs / fs)
        data = scipy_signal.resample(data, new_length, axis=0)
        fs = target_fs

    if data.shape[1] > N_CHANNELS:
        data = data[:, :N_CHANNELS]
    elif data.shape[1] < N_CHANNELS:
        raise ValueError(f"需要 {N_CHANNELS} 通道, 文件只有 {data.shape[1]} 通道")

    # z-score 标准化
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - mean) / std

    window_samples = int(WINDOW_SEC * fs)
    n_windows = len(data) // window_samples
    if n_windows == 0:
        raise ValueError(f"数据不足一个窗口({WINDOW_SEC}s)")

    data = data[:n_windows * window_samples]
    windows = data.reshape(n_windows, window_samples, N_CHANNELS)
    windows = windows.transpose(0, 2, 1)  # (N, channels, samples)

    return windows, fs


def compute_band_power(segment: np.ndarray, fs: int, band: tuple) -> float:
    freqs, psd = scipy_signal.welch(segment, fs=fs, nperseg=min(256, segment.shape[1]))
    band_mask = (freqs >= band[0]) & (freqs < band[1])
    return float(np.mean(psd[:, band_mask]))


def generate_labels(windows: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Beta/Alpha 比值二值化"""
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


# ========== 模型组件 ==========

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, L)
        b, c, _ = x.shape
        s = self.squeeze(x).view(b, c)
        s = self.excitation(s).view(b, c, 1)
        return x * s


class MultiScaleConv(nn.Module):
    """多尺度并行卷积 (kernel 25/11/5,均奇数以保持长度对齐)"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert out_channels % 3 == 0
        per = out_channels // 3
        # kernel 必须为奇数,padding=(k-1)//2 才能保持长度不变
        self.conv_small = nn.Conv1d(in_channels, per, kernel_size=5, padding=2)
        self.conv_medium = nn.Conv1d(in_channels, per, kernel_size=11, padding=5)
        self.conv_large = nn.Conv1d(in_channels, per, kernel_size=25, padding=12)

    def forward(self, x):
        return torch.cat([
            self.conv_small(x),
            self.conv_medium(x),
            self.conv_large(x),
        ], dim=1)


class ResidualBlock(nn.Module):
    """残差块: Conv → BN → ReLU → SE → +残差"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + residual)


# ========== 完整模型 ==========

class EEGNetV2(nn.Module):
    """增强版 EEGNet

    结构:
        Input (4, 1000)
        → MultiScaleConv(4→48) → BN → ReLU → MaxPool(4)     # (48, 250)
        → ResidualBlock(48) → MaxPool(4)                      # (48, 62)
        → Conv(48→96, k=3) → BN → ReLU → MaxPool(4)          # (96, 15)
        → ResidualBlock(96)                                    # (96, 15)
        → AdaptiveAvgPool(1) → Flatten                         # (96,)
        ↓
    CNN features (96) + Handcrafted features (9) → Concatenate
        → Linear(105→64) → ReLU → Dropout(0.4)
        → Linear(64→32) → ReLU → Dropout(0.3)
        → Linear(32→2)
    """

    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = 2,
                 n_handcrafted: int = N_FEATURES):
        super().__init__()

        # CNN 特征提取
        self.features = nn.Sequential(
            MultiScaleConv(n_channels, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(4),        # 1000 → 250

            ResidualBlock(48),
            nn.MaxPool1d(4),        # 250 → 62

            nn.Conv1d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(4),        # 62 → 15

            ResidualBlock(96),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # 分类器 (CNN 特征 + 手工特征)
        cnn_dim = 96
        fused_dim = cnn_dim + n_handcrafted
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x, features=None):
        cnn_feat = self.features(x)
        if features is not None:
            combined = torch.cat([cnn_feat, features], dim=1)
        else:
            # 无手工特征时用零填充
            batch_size = x.shape[0]
            dummy = torch.zeros(batch_size, N_FEATURES, device=x.device)
            combined = torch.cat([cnn_feat, dummy], dim=1)
        return self.classifier(combined)


# ========== 训练 ==========

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 5e-4,
    device: str = 'cpu',
    patience: int = 7,
    progress_callback=None,
) -> Tuple[EEGNetV2, dict]:
    """训练模型 V2

    改进:
    - ReduceLROnPlateau (patience=3, factor=0.5)
    - Early stopping (patience=7)
    - 特征融合 (如有)
    """
    model = EEGNetV2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0
    best_model_state = None
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            if len(batch) == 3:
                X, feat, y = batch
                X, feat, y = X.to(device), feat.to(device), y.to(device)
            else:
                X, y = batch
                feat, X, y = None, X.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(X, feat) if feat is not None else model(X)
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
            for batch in val_loader:
                if len(batch) == 3:
                    X, feat, y = batch
                    X, feat, y = X.to(device), feat.to(device), y.to(device)
                else:
                    X, y = batch
                    feat, X, y = None, X.to(device), y.to(device)
                out = model(X, feat) if feat is not None else model(X)
                loss = criterion(out, y)
                val_loss += loss.item() * len(y)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += len(y)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        if progress_callback:
            progress_callback(epoch + 1, train_loss, val_loss, train_acc, val_acc, current_lr)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if progress_callback:
                    progress_callback(epoch + 1, train_loss, val_loss, train_acc, val_acc, current_lr,
                                     early_stopped=True)
                break

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def predict_model(model: EEGNetV2, windows: np.ndarray, features: np.ndarray = None,
                  device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """推理,返回 (预测标签, 概率)"""
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        X = torch.FloatTensor(windows).to(device)
        feat = torch.FloatTensor(features).to(device) if features is not None else None
        out = model(X, feat) if feat is not None else model(X)
        probs = torch.softmax(out, dim=1)
        preds = out.argmax(1).cpu().numpy()
        probs = probs.cpu().numpy()
    return preds, probs


def save_model(model: EEGNetV2, path: str, meta: dict = None):
    torch.save({
        'model_state': model.state_dict(),
        'meta': meta or {},
    }, path)


def load_model(path: str, device: str = 'cpu') -> Tuple[EEGNetV2, dict]:
    save_dict = torch.load(path, map_location=device, weights_only=False)
    model = EEGNetV2()
    model.load_state_dict(save_dict['model_state'])
    model = model.to(device)
    return model, save_dict.get('meta', {})
