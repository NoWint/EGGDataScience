"""实时采集会话管理器

单例模式管理 BoardShim 生命周期 + WebSocket 客户端 + 后台轮询线程
"""
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Callable
from collections import deque

from .acquisition import BrainFlowAcquisition, BRAINFLOW_AVAILABLE

try:
    from brainflow.board_shim import BoardIds
except ImportError:
    pass


# board_id 字符串 → BrainFlow BoardIds.value
BOARD_ID_MAP = {}
if BRAINFLOW_AVAILABLE:
    BOARD_ID_MAP = {
        'synthetic': BoardIds.SYNTHETIC_BOARD.value,
        'cyton': BoardIds.CYTON_BOARD.value,
        'daisy': BoardIds.CYTON_DAISY_BOARD.value,
        'ganglion': BoardIds.GANGLION_BOARD.value,
    }


class AcquisitionManager:
    """采集会话管理器(非单例,但通常全局一个实例)"""

    def __init__(self):
        self._acq: Optional[BrainFlowAcquisition] = None
        self._state: str = 'IDLE'
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._clients: List[Callable] = []  # WebSocket 推送回调
        # 环形缓冲: 存 (ch_idx, sample) 元组, 8 通道交错
        # maxlen = 5000 samples/ch * 8 ch = 40000 (约 20 秒 @250Hz)
        self._buffer: deque = deque(maxlen=40000)
        self._last_focus_time: float = 0.0
        self._focus_cache: Dict = {'avg': 0.0, 'stability': 0.0}
        self._band_powers_cache: Dict = {}
        self._lock = threading.Lock()

    def start(self, board_id_str: str, params: Dict):
        """启动采集

        参数:
            board_id_str: 'synthetic' | 'cyton' | 'daisy' | 'ganglion'
            params: {serial_port, ip_address, ip_port, ...}
        """
        if not BRAINFLOW_AVAILABLE:
            raise RuntimeError("brainflow not installed")

        if self._state != 'IDLE':
            raise RuntimeError(f"Cannot start from state {self._state}")

        board_id = BOARD_ID_MAP.get(board_id_str)
        if board_id is None:
            raise ValueError(f"Unknown board_id: {board_id_str}")

        self._state = 'CONNECTING'
        try:
            self._acq = BrainFlowAcquisition(board_id, params)
            self._acq.prepare()
            self._acq.start_stream()
            self._state = 'STREAMING'

            # 启动轮询线程
            self._stop_event.clear()
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()
        except Exception as e:
            self._state = 'ERROR'
            if self._acq:
                self._acq.release_session()
                self._acq = None
            raise

    def stop(self):
        """停止采集"""
        if self._state not in ('STREAMING', 'ERROR'):
            return

        self._stop_event.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)

        if self._acq:
            self._acq.stop_stream()
            self._acq.release_session()
            self._acq = None

        self._state = 'IDLE'
        self._buffer.clear()
        self._focus_cache = {'avg': 0.0, 'stability': 0.0}
        self._band_powers_cache = {}

    def add_client(self, callback: Callable):
        """注册 WebSocket 推送回调"""
        with self._lock:
            self._clients.append(callback)

    def remove_client(self, callback: Callable):
        """移除 WebSocket 推送回调"""
        with self._lock:
            if callback in self._clients:
                self._clients.remove(callback)

    def get_status(self) -> Dict:
        """获取当前状态"""
        if self._acq is None:
            return {
                'state': self._state,
                'board_id': None,
                'board_name': None,
                'fs': 0,
                'channels': [],
                'n_clients': len(self._clients),
                'elapsed_sec': 0.0,
                'packets_lost': 0,
            }

        info = self._acq.get_board_info()
        return {
            'state': self._state,
            'board_id': info['board_id'],
            'board_name': info['board_name'],
            'fs': info['fs'],
            'channels': info['channels'],
            'n_clients': len(self._clients),
            'elapsed_sec': self._acq.elapsed_sec,
            'packets_lost': self._acq.packets_lost,
        }

    def _poll_loop(self):
        """后台轮询线程:50ms 间隔获取数据并推送"""
        while not self._stop_event.is_set():
            try:
                if self._acq is None or self._state != 'STREAMING':
                    break

                data = self._acq.get_latest_data()
                if data is None:
                    time.sleep(0.05)
                    continue

                # 更新环形缓冲(用前 8 通道)
                if data['data'] and len(data['data']) > 0:
                    for ch_idx, ch_data in enumerate(data['data']):
                        if ch_idx >= 8:
                            break
                        for sample in ch_data:
                            self._buffer.append((ch_idx, sample))

                # 每 2 秒计算 Focus + 频带功率
                now = time.time()
                if now - self._last_focus_time > 2.0:
                    self._compute_focus_and_bands(data['fs'])
                    self._last_focus_time = now

                # 构建推送帧
                frame = {
                    'type': 'data',
                    'timestamp': data['timestamp'],
                    'channels': data['channels'],
                    'data': data['data'],
                    'fs': data['fs'],
                    'focus': self._focus_cache,
                    'band_powers': self._band_powers_cache,
                }

                # 推送到所有客户端
                with self._lock:
                    clients = list(self._clients)
                for cb in clients:
                    try:
                        cb(frame)
                    except Exception:
                        pass

            except Exception:
                break

            time.sleep(0.05)

    def _compute_focus_and_bands(self, fs: int):
        """对环形缓冲计算 Focus 分数和频带功率"""
        if not self._buffer:
            return

        # 重建数据矩阵 (n_samples, n_channels)
        ch_data_dict = {}
        for ch_idx, sample in self._buffer:
            if ch_idx not in ch_data_dict:
                ch_data_dict[ch_idx] = []
            ch_data_dict[ch_idx].append(sample)

        if not ch_data_dict:
            return

        n_channels = max(ch_data_dict.keys()) + 1
        max_len = max(len(v) for v in ch_data_dict.values())

        data_matrix = np.zeros((max_len, n_channels))
        for ch_idx, samples in ch_data_dict.items():
            for i, s in enumerate(samples):
                if i < max_len:
                    data_matrix[i, ch_idx] = s

        # Focus
        try:
            from app.analysis.focus import compute_focus_scores
            result = compute_focus_scores(data_matrix, fs, window_sec=4.0)
            self._focus_cache = {
                'avg': result['avg'],
                'stability': result['stability'],
            }
        except Exception:
            pass

        # 频带功率
        try:
            from app.analysis.spectrum import compute_band_powers
            bp = compute_band_powers(data_matrix, fs)
            self._band_powers_cache = {
                k: float(v.get('rel', 0)) if isinstance(v, dict) else float(v)
                for k, v in bp.items()
            }
        except Exception:
            pass


# 全局单例
_manager_instance: Optional[AcquisitionManager] = None


def get_manager() -> AcquisitionManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = AcquisitionManager()
    return _manager_instance
