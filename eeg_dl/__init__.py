"""EEG 深度学习模块"""
from .model import (
    EEGNetV2, EEGDataset, train_model, predict_model, save_model, load_model,
    load_and_segment, generate_labels, extract_features,
    TARGET_FS, WINDOW_SEC, WINDOW_SAMPLES, N_CHANNELS,
    BANDS, N_BANDS, N_FEATURES,
)
