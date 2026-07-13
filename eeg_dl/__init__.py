"""EEG 深度学习模块"""
from .model import EEGNet, EEGDataset, train_model, predict_model, save_model, load_model
from .model import load_and_segment, generate_labels, TARGET_FS, WINDOW_SEC, N_CHANNELS
