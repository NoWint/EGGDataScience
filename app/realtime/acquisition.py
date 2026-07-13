"""BrainFlow BoardShim 生命周期封装

借鉴 OpenBCI GUI BoardBrainflow.pde,封装实时采集流程:
prepare_session → start_stream → get_board_data → stop_stream → release_session
"""
import time
import threading
import numpy as np
from typing import Dict, List, Optional

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False


# 8 通道标准名(与 stats_viz.py CHANNEL_POSITIONS 对应)
CHANNEL_NAMES_8CH = ['Fp1', 'Fp2', 'C3', 'C4', 'Pz', 'O1', 'O2', 'Fz']
# 16 通道(Daisy)
CHANNEL_NAMES_16CH = CHANNEL_NAMES_8CH + ['F3', 'F4', 'P3', 'P4', 'T3', 'T4', 'Oz', 'FCz']
# 4 通道(Ganglion)
CHANNEL_NAMES_4CH = ['Fp1', 'Fp2', 'C3', 'C4']


class BrainFlowAcquisition:
    """BrainFlow 实时采集封装

    状态机: IDLE → PREPARED → STREAMING → STOPPED → IDLE
    """

    def __init__(self, board_id: int, params: Optional[Dict] = None):
        if not BRAINFLOW_AVAILABLE:
            raise RuntimeError("brainflow not installed")

        self.board_id = board_id
        self.params_dict = params or {}
        self.state = 'IDLE'
        self._board: Optional[BoardShim] = None
        self._fs: int = 0
        self._exg_channels: List[int] = []
        self._board_name: str = ''
        self._start_time: float = 0.0
        self._packets_lost: int = 0
        self._last_sample_idx: int = -1

    def _build_params(self) -> BrainFlowInputParams:
        """构建 BrainFlowInputParams"""
        params = BrainFlowInputParams()
        p = self.params_dict
        if 'serial_port' in p:
            params.serial_port = p['serial_port']
        if 'ip_address' in p:
            params.ip_address = p['ip_address']
        if 'ip_port' in p:
            params.ip_port = int(p['ip_port'])
        if 'mac_address' in p:
            params.mac_address = p['mac_address']
        return params

    def prepare(self):
        """准备采集会话"""
        if self.state != 'IDLE':
            raise RuntimeError(f"Cannot prepare from state {self.state}")

        self.state = 'CONNECTING'
        bf_params = self._build_params()
        self._board = BoardShim(self.board_id, bf_params)
        self._board.prepare_session()

        # 获取板卡信息
        self._fs = BoardShim.get_sampling_rate(self.board_id)
        self._exg_channels = BoardShim.get_exg_channels(self.board_id)
        try:
            self._board_name = BoardShim.get_board_descr(self.board_id).get('name', 'Unknown')
        except Exception:
            self._board_name = f'Board {self.board_id}'

        self.state = 'PREPARED'

    def start_stream(self, buffer_size: int = 450000):
        """开始采集数据流"""
        if self.state != 'PREPARED':
            raise RuntimeError(f"Cannot start_stream from state {self.state}")

        self._board.start_stream(buffer_size)
        self._start_time = time.time()
        self._packets_lost = 0
        self._last_sample_idx = -1
        self.state = 'STREAMING'

    def get_latest_data(self) -> Optional[Dict]:
        """获取最新数据帧

        返回:
            {
                'data': List[List[float]],  # (n_channels, n_samples)
                'channels': List[str],
                'fs': int,
                'timestamp': float,
                'sample_indices': List[int],
            }
        """
        if self.state != 'STREAMING' or self._board is None:
            return None

        # 获取板卡缓冲中的所有数据
        data = self._board.get_board_data()
        if data is None or data.shape[1] == 0:
            return None

        # 提取 EXG 通道
        exg_data = data[self._exg_channels, :]  # (n_exg, n_samples)

        # 通道名
        n_exg = len(self._exg_channels)
        if n_exg >= 8:
            channels = CHANNEL_NAMES_8CH + [f'EXG{i}' for i in range(n_exg - 8)]
        elif n_exg >= 4:
            channels = CHANNEL_NAMES_4CH + [f'EXG{i}' for i in range(n_exg - 4)]
        else:
            channels = [f'EXG{i}' for i in range(n_exg)]
        channels = channels[:n_exg]

        # 丢包检测(用 sample index 通道,如果有)
        sample_indices = []
        try:
            sample_idx_channel = BoardShim.get_package_num_channel(self.board_id)
            sample_indices = data[sample_idx_channel, :].tolist()
            if self._last_sample_idx >= 0 and len(sample_indices) > 0:
                expected = (self._last_sample_idx + 1) % 256
                actual = int(sample_indices[0])
                if actual != expected:
                    gap = (actual - expected) % 256
                    self._packets_lost += gap
            if len(sample_indices) > 0:
                self._last_sample_idx = int(sample_indices[-1])
        except Exception:
            pass

        return {
            'data': exg_data.tolist(),
            'channels': channels,
            'fs': self._fs,
            'timestamp': time.time() - self._start_time,
            'sample_indices': sample_indices,
        }

    def get_board_info(self) -> Dict:
        """获取板卡信息(无需 prepare)"""
        if not BRAINFLOW_AVAILABLE:
            return {'error': 'brainflow not available'}

        fs = BoardShim.get_sampling_rate(self.board_id)
        exg_channels = BoardShim.get_exg_channels(self.board_id)
        n_exg = len(exg_channels)

        # 通道名
        if n_exg >= 8:
            channels = CHANNEL_NAMES_8CH + [f'EXG{i}' for i in range(n_exg - 8)]
        elif n_exg >= 4:
            channels = CHANNEL_NAMES_4CH + [f'EXG{i}' for i in range(n_exg - 4)]
        else:
            channels = [f'EXG{i}' for i in range(n_exg)]
        channels = channels[:n_exg]

        try:
            board_name = BoardShim.get_board_descr(self.board_id).get('name', 'Unknown')
        except Exception:
            board_name = f'Board {self.board_id}'

        return {
            'board_id': self.board_id,
            'board_name': board_name,
            'fs': fs,
            'channels': channels,
            'n_exg': n_exg,
        }

    def stop_stream(self):
        """停止采集"""
        if self.state != 'STREAMING':
            return
        self._board.stop_stream()
        self.state = 'STOPPED'

    def release_session(self):
        """释放会话"""
        if self._board is not None:
            try:
                self._board.release_session()
            except Exception:
                pass
            self._board = None
        self.state = 'IDLE'

    @property
    def elapsed_sec(self) -> float:
        if self.state == 'STREAMING':
            return time.time() - self._start_time
        return 0.0

    @property
    def packets_lost(self) -> int:
        return self._packets_lost
