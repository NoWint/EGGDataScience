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


# ========== Task 3: REST + WebSocket 端点测试 ==========

@pytest.fixture
def client():
    """TestClient + 确保全局 manager 清洁"""
    from app.server import app
    from app.realtime.manager import get_manager
    from fastapi.testclient import TestClient

    # 测试前确保 manager 停止
    mgr = get_manager()
    mgr.stop()

    c = TestClient(app)
    yield c

    # 测试后清理
    mgr.stop()


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
    # 先启动采集
    client.post("/api/realtime/start", json={"board_id": "synthetic", "params": {}})
    time.sleep(0.3)

    # 连接 WebSocket
    try:
        with client.websocket_connect("/ws/realtime") as ws:
            # 接收数据帧(可能先收到 ping,循环直到收到 data)
            for _ in range(10):
                data = ws.receive_json()
                if data.get('type') == 'data':
                    assert 'data' in data
                    assert 'channels' in data
                    assert 'fs' in data
                    return
            pytest.fail("未收到 data 类型帧")
    finally:
        client.post("/api/realtime/stop")


# ========== Task 5: 端到端测试 ==========

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
    assert len(status['channels']) == 16  # Synthetic Board 返回 16 个 EXG 通道

    # 4. WebSocket 接收数据
    with client.websocket_connect("/ws/realtime") as ws:
        for _ in range(10):
            frame = ws.receive_json()
            if frame.get('type') == 'data':
                assert len(frame['data']) > 0
                break

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
