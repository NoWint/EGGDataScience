# 实时脑电采集 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 BrainFlow BoardShim 实现 EEGDataScience 实时 EEG 采集,WebSocket 推送,前端滚动波形渲染,支持 Synthetic Board 无硬件测试。

**Architecture:** 后端 `app/realtime/` 模块封装 BoardShim 生命周期,后台线程 50ms 轮询推数据,WebSocket 推送至前端。前端 `realtime.js` Canvas 滚动绘制 + 实时 Focus/频带功率。

**Tech Stack:** Python 3.11+ / FastAPI WebSocket / brainflow 5.22.0+ / 原生 Canvas

**Spec:** [docs/specs/2026-07-13-realtime-acquisition-design.md](file:///Users/xiatian/Desktop/EEG-Science/docs/specs/2026-07-13-realtime-acquisition-design.md)

---

## 文件结构

```
EEG-Science/
├── app/
│   ├── realtime/
│   │   ├── __init__.py              # 新建: 模块导出
│   │   ├── acquisition.py           # 新建: BoardShim 生命周期封装
│   │   └── manager.py               # 新建: 会话管理器 + 状态机
│   ├── routers/
│   │   └── realtime.py              # 新建: REST + WebSocket 端点
│   ├── server.py                    # 修改: 注册 realtime 路由
│   └── static/
│       ├── index.html               # 修改: 实时采集视图
│       └── js/
│           └── realtime.js          # 新建: WebSocket 客户端 + Canvas
└── tests/
    └── test_realtime.py             # 新建: Synthetic Board 端到端测试
```

---

### Task 1: 创建 acquisition.py — BoardShim 生命周期封装

**Files:**
- Create: `app/realtime/__init__.py`
- Create: `app/realtime/acquisition.py`
- Test: `tests/test_realtime.py`

- [ ] **Step 1: 写失败测试 — BrainFlowAcquisition 基本功能**

创建 `tests/test_realtime.py`:

```python
"""实时采集测试"""
import pytest
import time


def test_acquisition_synthetic_start_stop():
    """测试 Synthetic Board 启动停止"""
    from app.realtime.acquisition import BrainFlowAcquisition
    from brainflow.board_shim import BoardIds
    
    acq = BrainFlowAcquisition(BoardIds.SYNTHETIC_BOARD.value)
    assert acq.state == 'IDLE'
    
    acq.prepare()
    assert acq.state == 'PREPARED'
    
    acq.start_stream()
    assert acq.state == 'STREAMING'
    
    # 等待数据
    time.sleep(0.5)
    data = acq.get_latest_data()
    assert data is not None
    assert 'data' in data
    assert 'channels' in data
    assert len(data['data']) > 0  # 有通道数据
    
    acq.stop_stream()
    assert acq.state == 'STOPPED'
    
    acq.release_session()
    assert acq.state == 'IDLE'


def test_acquisition_board_info():
    """测试板卡信息获取"""
    from app.realtime.acquisition import BrainFlowAcquisition
    from brainflow.board_shim import BoardIds
    
    acq = BrainFlowAcquisition(BoardIds.SYNTHETIC_BOARD.value)
    info = acq.get_board_info()
    
    assert 'board_id' in info
    assert 'board_name' in info
    assert 'fs' in info
    assert 'channels' in info
    assert 'n_exg' in info
    assert info['n_exg'] > 0
    assert info['fs'] > 0
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_realtime.py::test_acquisition_synthetic_start_stop -v`
Expected: FAIL with "No module named 'app.realtime'"

- [ ] **Step 3: 创建 app/realtime/ 模块**

创建 `app/realtime/__init__.py`:
```python
"""实时采集模块"""
from .acquisition import BrainFlowAcquisition
from .manager import AcquisitionManager

__all__ = ['BrainFlowAcquisition', 'AcquisitionManager']
```

创建 `app/realtime/acquisition.py`:
```python
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
        self._board_name = BoardShim.get_board_descr(self.board_id).get('name', 'Unknown')
        
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
            sample_indices = []
        
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
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_realtime.py::test_acquisition_synthetic_start_stop tests/test_realtime.py::test_acquisition_board_info -v`
Expected: 2 个测试 PASS

- [ ] **Step 5: Commit**

```bash
git add app/realtime/__init__.py app/realtime/acquisition.py tests/test_realtime.py
git commit -m "feat: add BrainFlowAcquisition for BoardShim lifecycle

- 封装 prepare/start_stream/get_data/stop/release
- 支持 Synthetic Board 无硬件测试
- 丢包检测(sample index 间隙)
- 通道名映射(8ch/16ch/4ch)"
```

---

### Task 2: 创建 manager.py — 会话管理器

**Files:**
- Create: `app/realtime/manager.py`
- Modify: `app/realtime/__init__.py`
- Test: `tests/test_realtime.py`

- [ ] **Step 1: 写失败测试 — AcquisitionManager**

在 `tests/test_realtime.py` 末尾追加:

```python
def test_manager_start_stop():
    """测试 AcquisitionManager 启动停止"""
    from app.realtime.manager import AcquisitionManager
    
    manager = AcquisitionManager()
    
    # 初始状态
    status = manager.get_status()
    assert status['state'] == 'IDLE'
    
    # 启动 Synthetic Board
    manager.start('synthetic', {})
    assert manager.get_status()['state'] == 'STREAMING'
    
    # 等待数据
    time.sleep(0.5)
    
    # 停止
    manager.stop()
    assert manager.get_status()['state'] == 'IDLE'


def test_manager_status_fields():
    """测试状态返回字段"""
    from app.realtime.manager import AcquisitionManager
    
    manager = AcquisitionManager()
    manager.start('synthetic', {})
    time.sleep(0.3)
    
    status = manager.get_status()
    assert 'state' in status
    assert 'board_id' in status
    assert 'board_name' in status
    assert 'fs' in status
    assert 'channels' in status
    assert 'elapsed_sec' in status
    
    manager.stop()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_realtime.py::test_manager_start_stop -v`
Expected: FAIL with "cannot import name 'AcquisitionManager'"

- [ ] **Step 3: 实现 manager.py**

创建 `app/realtime/manager.py`:

```python
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
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False


# board_id 字符串 → BrainFlow BoardIds.value
BOARD_ID_MAP = {
    'synthetic': BoardIds.SYNTHETIC_BOARD.value if BRAINFLOW_AVAILABLE else -2,
    'cyton': BoardIds.CYTON_BOARD.value if BRAINFLOW_AVAILABLE else 0,
    'daisy': BoardIds.CYTON_DAISY_BOARD.value if BRAINFLOW_AVAILABLE else 2,
    'ganglion': BoardIds.GANGLION_BOARD.value if BRAINFLOW_AVAILABLE else 1,
}


class AcquisitionManager:
    """采集会话管理器(非单例,但通常全局一个实例)"""
    
    def __init__(self):
        self._acq: Optional[BrainFlowAcquisition] = None
        self._state: str = 'IDLE'
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._clients: List[Callable] = []  # WebSocket 推送回调
        self._buffer: deque = deque(maxlen=5000)  # 环形缓冲(约 20 秒 @250Hz)
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
                
                # 更新环形缓冲(用第一通道作为代表,或全部通道)
                if data['data'] and len(data['data']) > 0:
                    for ch_idx, ch_data in enumerate(data['data']):
                        # 只缓冲前 8 通道
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
        # buffer 存的是 (ch_idx, sample) 元组
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
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m pytest tests/test_realtime.py::test_manager_start_stop tests/test_realtime.py::test_manager_status_fields -v`
Expected: 2 个测试 PASS

- [ ] **Step 5: Commit**

```bash
git add app/realtime/manager.py app/realtime/__init__.py tests/test_realtime.py
git commit -m "feat: add AcquisitionManager with background polling thread

- 单例管理 BoardShim 生命周期
- 50ms 轮询 get_board_data 并推送 WebSocket 客户端
- 环形缓冲(20 秒)+ 每 2 秒计算 Focus/频带功率
- BOARD_ID_MAP: synthetic/cyton/daisy/ganglion"
```

---

### Task 3: 创建 routers/realtime.py — REST + WebSocket 端点

**Files:**
- Create: `app/routers/realtime.py`
- Modify: `app/server.py` (注册路由)
- Test: `tests/test_realtime.py`

- [ ] **Step 1: 写失败测试 — REST 端点**

在 `tests/test_realtime.py` 末尾追加:

```python
@pytest.fixture
def client():
    from app.server import app
    return TestClient(app)


def test_realtime_status_endpoint(client):
    """测试 GET /api/realtime/status"""
    resp = client.get("/api/realtime/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data['state'] == 'IDLE'


def test_realtime_start_stop_endpoints(client):
    """测试 POST /api/realtime/start + stop"""
    # 启动
    resp = client.post("/api/realtime/start", json={
        "board_id": "synthetic",
        "params": {},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data['ok'] is True
    assert data['board_name']  # 有板名
    assert data['fs'] > 0
    
    # 等待数据
    time.sleep(0.5)
    
    # 状态应为 STREAMING
    status = client.get("/api/realtime/status").json()
    assert status['state'] == 'STREAMING'
    
    # 停止
    resp = client.post("/api/realtime/stop")
    assert resp.status_code == 200
    assert resp.json()['ok'] is True
    
    # 状态应为 IDLE
    status = client.get("/api/realtime/status").json()
    assert status['state'] == 'IDLE'


def test_realtime_websocket(client):
    """测试 WebSocket 数据推送"""
    from fastapi.testclient import TestClient
    
    # 先启动采集
    client.post("/api/realtime/start", json={"board_id": "synthetic", "params": {}})
    time.sleep(0.3)
    
    # 连接 WebSocket
    try:
        with client.websocket_connect("/ws/realtime") as ws:
            # 接收数据帧
            data = ws.receive_json()
            assert data['type'] == 'data'
            assert 'data' in data
            assert 'channels' in data
            assert 'fs' in data
    finally:
        client.post("/api/realtime/stop")
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m pytest tests/test_realtime.py::test_realtime_status_endpoint -v`
Expected: FAIL (404 路由不存在)

- [ ] **Step 3: 创建 routers/realtime.py**

创建 `app/routers/realtime.py`:

```python
"""实时采集 REST + WebSocket 路由"""
import json
import asyncio
from typing import Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.realtime.manager import get_manager, BRAINFLOW_AVAILABLE


router = APIRouter(prefix="/api/realtime", tags=["realtime"])


class StartRequest(BaseModel):
    board_id: str = "synthetic"  # synthetic | cyton | daisy | ganglion
    params: Optional[Dict] = None


@router.get("/status")
def realtime_status():
    """获取采集状态"""
    manager = get_manager()
    return manager.get_status()


@router.post("/start")
def realtime_start(req: StartRequest):
    """启动采集"""
    if not BRAINFLOW_AVAILABLE:
        return {"ok": False, "error": "brainflow not installed"}
    
    manager = get_manager()
    try:
        manager.start(req.board_id, req.params or {})
        info = manager.get_status()
        return {
            "ok": True,
            "board_id": info['board_id'],
            "board_name": info['board_name'],
            "fs": info['fs'],
            "channels": info['channels'],
            "n_exg": len(info['channels']),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/stop")
def realtime_stop():
    """停止采集"""
    manager = get_manager()
    elapsed = manager.get_status().get('elapsed_sec', 0.0)
    manager.stop()
    return {"ok": True, "elapsed_sec": elapsed}


# WebSocket 端点(挂载在 /ws/realtime)
async def realtime_websocket(websocket: WebSocket):
    """WebSocket 实时数据推送"""
    await websocket.accept()
    manager = get_manager()
    
    # 创建异步队列作为推送中介
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    
    def push_callback(frame):
        """同步回调 → 推入异步队列"""
        try:
            loop.call_soon_threadsafe(queue.put_nowait, frame)
        except Exception:
            pass
    
    manager.add_client(push_callback)
    
    try:
        while True:
            try:
                # 等待数据帧(1 秒超时)
                frame = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(frame)
            except asyncio.TimeoutError:
                # 发送心跳
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        manager.remove_client(push_callback)
```

- [ ] **Step 4: 在 server.py 注册路由 + WebSocket**

读取 `app/server.py`,找到路由注册区域(通常有 `app.include_router(...)` 调用)。

添加:
```python
from app.routers import realtime as realtime_router
app.include_router(realtime_router.router)

# WebSocket 端点
from app.routers.realtime import realtime_websocket
from fastapi import WebSocket

@app.websocket("/ws/realtime")
async def ws_realtime(websocket: WebSocket):
    await realtime_websocket(websocket)
```

- [ ] **Step 5: 运行测试验证通过**

Run: `python -m pytest tests/test_realtime.py -v`
Expected: 所有测试 PASS

- [ ] **Step 6: Commit**

```bash
git add app/routers/realtime.py app/server.py tests/test_realtime.py
git commit -m "feat: add realtime REST + WebSocket endpoints

- POST /api/realtime/start | stop | GET /status
- WS /ws/realtime 推送数据帧
- 异步队列桥接同步轮询线程与 WebSocket"
```

---

### Task 4: 前端 — realtime.js + index.html

**Files:**
- Create: `app/static/js/realtime.js`
- Modify: `app/static/index.html`
- Modify: `app/static/css/style.css`

- [ ] **Step 1: 创建 realtime.js**

创建 `app/static/js/realtime.js`:

```javascript
/**
 * 实时采集模块
 * WebSocket 连接 + Canvas 滚动波形 + Focus 实时显示
 */

let rtWebSocket = null;
let rtCanvas = null;
let rtCtx = null;
let rtState = 'IDLE';
let rtDataBuffer = [];  // 最近 N 秒数据
let rtMaxSamples = 1250;  // 5 秒 @250Hz

/**
 * 初始化实时采集模块
 */
function initRealtime() {
    rtCanvas = document.getElementById('realtime-canvas');
    if (rtCanvas) {
        rtCtx = rtCanvas.getContext('2d');
        rtCanvas.width = rtCanvas.offsetWidth || 800;
        rtCanvas.height = 400;
    }
    
    // 绑定按钮
    const startBtn = document.getElementById('rt-start-btn');
    const stopBtn = document.getElementById('rt-stop-btn');
    
    if (startBtn) {
        startBtn.addEventListener('click', startRealtime);
    }
    if (stopBtn) {
        stopBtn.addEventListener('click', stopRealtime);
    }
    
    // 初始状态
    updateRealtimeStatus('IDLE');
}

/**
 * 启动采集
 */
async function startRealtime() {
    const boardSelect = document.getElementById('rt-board-select');
    const boardId = boardSelect ? boardSelect.value : 'synthetic';
    
    updateRealtimeStatus('CONNECTING');
    
    try {
        const resp = await fetch('/api/realtime/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({board_id: boardId, params: {}}),
        });
        const data = await resp.json();
        
        if (!data.ok) {
            updateRealtimeStatus('ERROR', data.error || '启动失败');
            return;
        }
        
        // 连接 WebSocket
        connectRealtimeWS();
        
        updateRealtimeStatus('STREAMING', `已连接 ${data.board_name}`);
    } catch (e) {
        updateRealtimeStatus('ERROR', e.message);
    }
}

/**
 * 停止采集
 */
async function stopRealtime() {
    try {
        await fetch('/api/realtime/stop', {method: 'POST'});
    } catch (e) {}
    
    if (rtWebSocket) {
        rtWebSocket.close();
        rtWebSocket = null;
    }
    
    updateRealtimeStatus('IDLE');
    rtDataBuffer = [];
}

/**
 * 连接 WebSocket
 */
function connectRealtimeWS() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/realtime`;
    
    rtWebSocket = new WebSocket(wsUrl);
    
    rtWebSocket.onmessage = (event) => {
        const frame = JSON.parse(event.data);
        if (frame.type === 'data') {
            handleRealtimeData(frame);
        }
    };
    
    rtWebSocket.onclose = () => {
        if (rtState === 'STREAMING') {
            // 意外断开,尝试重连
            setTimeout(connectRealtimeWS, 2000);
        }
    };
}

/**
 * 处理实时数据帧
 */
function handleRealtimeData(frame) {
    const {data, channels, fs, focus, band_powers} = frame;
    
    if (!data || data.length === 0) return;
    
    // 更新数据缓冲
    const nChannels = data.length;
    const nSamples = data[0].length;
    
    for (let i = 0; i < nSamples; i++) {
        const sample = [];
        for (let ch = 0; ch < nChannels; ch++) {
            sample.push(data[ch][i]);
        }
        rtDataBuffer.push(sample);
        
        // 保持缓冲长度
        if (rtDataBuffer.length > rtMaxSamples) {
            rtDataBuffer.shift();
        }
    }
    
    // 渲染波形
    renderRealtimeWaveform(channels);
    
    // 更新 Focus
    if (focus) {
        updateFocusDisplay(focus);
    }
    
    // 更新频带功率
    if (band_powers) {
        updateBandPowersDisplay(band_powers);
    }
}

/**
 * 渲染滚动波形
 */
function renderRealtimeWaveform(channels) {
    if (!rtCtx || rtDataBuffer.length === 0) return;
    
    const W = rtCanvas.width;
    const H = rtCanvas.height;
    const nChannels = channels.length;
    const channelHeight = H / nChannels;
    
    rtCtx.clearRect(0, 0, W, H);
    
    // 找最大绝对值(自动增益)
    let maxVal = 1.0;
    for (const sample of rtDataBuffer) {
        for (const v of sample) {
            if (Math.abs(v) > maxVal) maxVal = Math.abs(v);
        }
    }
    
    // 绘制每通道
    for (let ch = 0; ch < nChannels && ch < 8; ch++) {
        const yCenter = channelHeight * ch + channelHeight / 2;
        const amplitude = channelHeight * 0.4;
        
        // 通道名
        rtCtx.fillStyle = '#666';
        rtCtx.font = '11px sans-serif';
        rtCtx.textAlign = 'left';
        rtCtx.fillText(channels[ch] || `CH${ch}`, 4, yCenter - amplitude + 12);
        
        // 波形
        rtCtx.strokeStyle = '#4B3FE3';
        rtCtx.lineWidth = 1;
        rtCtx.beginPath();
        
        for (let i = 0; i < rtDataBuffer.length; i++) {
            const x = (i / rtMaxSamples) * W;
            const v = rtDataBuffer[i][ch] || 0;
            const y = yCenter - (v / maxVal) * amplitude;
            if (i === 0) rtCtx.moveTo(x, y);
            else rtCtx.lineTo(x, y);
        }
        rtCtx.stroke();
    }
}

/**
 * 更新 Focus 显示
 */
function updateFocusDisplay(focus) {
    const avgEl = document.getElementById('rt-focus-avg');
    const stabilityEl = document.getElementById('rt-focus-stability');
    const hintEl = document.getElementById('rt-focus-hint');
    
    if (avgEl) {
        avgEl.textContent = (focus.avg || 0).toFixed(2);
    }
    if (stabilityEl) {
        stabilityEl.textContent = (focus.stability || 0).toFixed(3);
    }
    if (hintEl) {
        const avg = focus.avg || 0;
        if (avg < 0.3) {
            hintEl.textContent = '走神';
            hintEl.style.color = '#ef4444';
        } else if (avg < 0.7) {
            hintEl.textContent = '一般';
            hintEl.style.color = '#f59e0b';
        } else {
            hintEl.textContent = '专注';
            hintEl.style.color = '#10b981';
        }
    }
}

/**
 * 更新频带功率
 */
function updateBandPowersDisplay(bp) {
    const bands = ['delta', 'theta', 'alpha', 'beta', 'gamma'];
    bands.forEach(b => {
        const el = document.getElementById(`rt-bp-${b}`);
        if (el) {
            el.textContent = (bp[b] || 0).toFixed(3);
        }
    });
}

/**
 * 更新状态显示
 */
function updateRealtimeStatus(state, message) {
    rtState = state;
    const statusEl = document.getElementById('rt-status');
    const stateEl = document.getElementById('rt-state');
    const startBtn = document.getElementById('rt-start-btn');
    const stopBtn = document.getElementById('rt-stop-btn');
    
    const stateText = {
        'IDLE': '空闲',
        'CONNECTING': '连接中...',
        'STREAMING': '采集中',
        'ERROR': '错误',
    }[state] || state;
    
    if (stateEl) stateEl.textContent = stateText;
    
    const colors = {
        'IDLE': '#999',
        'CONNECTING': '#f59e0b',
        'STREAMING': '#10b981',
        'ERROR': '#ef4444',
    };
    if (stateEl) stateEl.style.color = colors[state] || '#999';
    
    if (statusEl && message) {
        statusEl.textContent = message;
    }
    
    if (startBtn) startBtn.disabled = (state === 'STREAMING' || state === 'CONNECTING');
    if (stopBtn) stopBtn.disabled = (state === 'IDLE');
}

// 导出
window.initRealtime = initRealtime;
window.startRealtime = startRealtime;
window.stopRealtime = stopRealtime;
```

- [ ] **Step 2: 修改 index.html — 新增实时采集视图**

读取 `app/static/index.html`,在侧边栏新增入口 + 主内容区新增视图。

侧边栏入口(在现有模块之后):
```html
<a class="nav-item" data-module="realtime">
    <span class="nav-item-text">实时采集</span>
</a>
```

主内容区视图:
```html
<div class="module-view" id="view-realtime">
    <header class="module-header">
        <h1 class="module-title">实时脑电采集</h1>
        <div class="module-actions">
            <select id="rt-board-select" class="board-select">
                <option value="synthetic">合成板(演示用,无需硬件)</option>
                <option value="cyton">Cyton (8通道)</option>
                <option value="daisy">Daisy (16通道)</option>
                <option value="ganglion">Ganglion (4通道)</option>
            </select>
            <button id="rt-start-btn" class="btn btn-primary">开始采集</button>
            <button id="rt-stop-btn" class="btn btn-danger" disabled>停止采集</button>
        </div>
    </header>
    
    <div class="realtime-status-bar">
        <span>状态: <strong id="rt-state">空闲</strong></span>
        <span id="rt-status"></span>
    </div>
    
    <div class="realtime-canvas-container">
        <canvas id="realtime-canvas"></canvas>
    </div>
    
    <div class="realtime-stats">
        <div class="stat-card">
            <div class="stat-label">实时专注度</div>
            <div class="stat-value" id="rt-focus-avg">--</div>
            <div class="stat-hint" id="rt-focus-hint">等待数据</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">稳定性</div>
            <div class="stat-value" id="rt-focus-stability">--</div>
        </div>
        <div class="stat-card band-powers-card">
            <div class="stat-label">频带功率</div>
            <div class="bp-grid">
                <span>δ: <strong id="rt-bp-delta">--</strong></span>
                <span>θ: <strong id="rt-bp-theta">--</strong></span>
                <span>α: <strong id="rt-bp-alpha">--</strong></span>
                <span>β: <strong id="rt-bp-beta">--</strong></span>
                <span>γ: <strong id="rt-bp-gamma">--</strong></span>
            </div>
        </div>
    </div>
</div>
```

在 `</body>` 前引入:
```html
<script src="/static/js/realtime.js"></script>
```

在 DOMContentLoaded 中调用:
```javascript
if (window.initRealtime) initRealtime();
```

- [ ] **Step 3: 添加 CSS 样式**

在 `app/static/css/style.css` 末尾添加:

```css
/* === 实时采集模块 === */
.board-select { padding: 6px 12px; border: 1px solid #ddd; border-radius: 6px; }
.realtime-status-bar { padding: 12px; background: #f9fafb; border-radius: 8px; margin-bottom: 16px; display: flex; gap: 16px; }
.realtime-canvas-container { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 16px; }
#realtime-canvas { width: 100%; height: 400px; display: block; }
.realtime-stats { display: flex; gap: 16px; flex-wrap: wrap; }
.band-powers-card { min-width: 200px; }
.bp-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 14px; }
.bp-grid span { color: #666; }
.bp-grid strong { color: #333; }
.btn-primary { background: #4B3FE3; color: #fff; border: none; padding: 6px 16px; border-radius: 6px; cursor: pointer; }
.btn-primary:disabled { background: #ccc; cursor: not-allowed; }
.btn-danger { background: #ef4444; color: #fff; border: none; padding: 6px 16px; border-radius: 6px; cursor: pointer; }
.btn-danger:disabled { background: #ccc; cursor: not-allowed; }
```

- [ ] **Step 4: 启动服务验证**

```bash
cd /Users/xiatian/Desktop/EEG-Science
python -m uvicorn app.server:app --port 18765 &
sleep 2
curl -s http://localhost:18765/api/realtime/status | python -m json.tool
curl -s http://localhost:18765/ | grep "realtime" | head -5
pkill -f "uvicorn app.server:app --port 18765" || true
```

- [ ] **Step 5: Commit**

```bash
git add app/static/js/realtime.js app/static/index.html app/static/css/style.css
git commit -m "feat: add realtime acquisition frontend

- realtime.js: WebSocket 客户端 + Canvas 滚动波形
- 实时 Focus + 频带功率显示
- 设备选择 UI (Synthetic/Cyton/Daisy/Ganglion)
- 连接状态颜色反馈"
```

---

### Task 5: 端到端测试 + 回归验证

**Files:**
- Test: `tests/test_realtime.py`

- [ ] **Step 1: 补充端到端测试**

在 `tests/test_realtime.py` 末尾追加:

```python
def test_realtime_e2e_synthetic(client):
    """端到端:Synthetic Board 完整流程"""
    # 1. 初始状态
    assert client.get("/api/realtime/status").json()['state'] == 'IDLE'
    
    # 2. 启动
    resp = client.post("/api/realtime/start", json={"board_id": "synthetic", "params": {}})
    assert resp.status_code == 200
    assert resp.json()['ok'] is True
    
    # 3. 确认采集状态
    time.sleep(0.5)
    status = client.get("/api/realtime/status").json()
    assert status['state'] == 'STREAMING'
    assert status['fs'] == 250
    assert len(status['channels']) == 8
    
    # 4. WebSocket 接收数据
    with client.websocket_connect("/ws/realtime") as ws:
        frame = ws.receive_json()
        assert frame['type'] == 'data'
        assert len(frame['data']) > 0
    
    # 5. 停止
    resp = client.post("/api/realtime/stop")
    assert resp.json()['ok'] is True
    
    # 6. 确认回到 IDLE
    assert client.get("/api/realtime/status").json()['state'] == 'IDLE'


def test_realtime_invalid_board(client):
    """测试无效 board_id"""
    resp = client.post("/api/realtime/start", json={"board_id": "invalid_board", "params": {}})
    assert resp.status_code == 200
    assert resp.json()['ok'] is False
```

- [ ] **Step 2: 运行全部实时测试**

Run: `python -m pytest tests/test_realtime.py -v`
Expected: 所有测试 PASS

- [ ] **Step 3: 运行全部测试确保无回归**

Run: `python -m pytest tests/ -v --tb=short`
Expected: 所有测试 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_realtime.py
git commit -m "test: add realtime end-to-end and error case tests

- Synthetic Board 完整流程(start → WS → stop)
- 无效 board_id 错误处理"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ BrainFlow BoardShim 生命周期 — Task 1 (acquisition.py)
- ✅ 会话管理器 + 后台轮询 — Task 2 (manager.py)
- ✅ REST 端点 (start/stop/status) — Task 3 (routers/realtime.py)
- ✅ WebSocket 数据推送 — Task 3
- ✅ 前端 WebSocket 客户端 — Task 4 (realtime.js)
- ✅ Canvas 滚动波形 — Task 4
- ✅ 实时 Focus + 频带功率 — Task 2 后端计算 + Task 4 前端显示
- ✅ 设备选择 UI — Task 4
- ✅ Synthetic Board 无硬件测试 — Task 5
- ✅ 新手友好(默认 Synthetic + 状态颜色 + Focus 提示) — Task 4

**2. Placeholder scan:** 无 TBD/TODO。

**3. Type consistency:**
- `BrainFlowAcquisition(board_id, params)` — Task 1 定义,Task 2 使用,一致
- `manager.start(board_id_str, params)` — Task 2 定义,Task 3 使用,一致
- WebSocket 帧 `{type, timestamp, channels, data, fs, focus, band_powers}` — Task 2 定义,Task 4 使用,一致
- `get_status()` 返回 `{state, board_id, board_name, fs, channels, ...}` — Task 2 定义,Task 3 使用,一致

无问题。
