"""边界压力测试

测试实时采集模块在极端/异常场景下的行为:
- 并发 WebSocket 客户端
- 快速启停循环
- 状态机违反
- 断连重连
- 长时间采集(缓冲边界)
- 无效参数
- 单例一致性
"""
import pytest
import time
import threading
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """TestClient + 确保全局 manager 清洁"""
    from app.server import app
    from app.realtime.manager import get_manager

    mgr = get_manager()
    mgr.stop()
    c = TestClient(app)
    yield c
    mgr.stop()


# ========== 1. 状态机违反 ==========

def test_boundary_stop_when_idle(client):
    """IDLE 状态下 stop 应为无操作,不抛异常"""
    resp = client.post("/api/realtime/stop")
    assert resp.status_code == 200
    assert resp.json()['ok'] is True
    # 状态仍为 IDLE
    assert client.get("/api/realtime/status").json()['state'] == 'IDLE'


def test_boundary_start_when_already_streaming(client):
    """STREAMING 状态下再次 start 应失败"""
    # 第一次启动
    resp1 = client.post("/api/realtime/start", json={"board_id": "synthetic", "params": {}})
    assert resp1.json()['ok'] is True
    time.sleep(0.3)

    # 第二次启动(应失败)
    resp2 = client.post("/api/realtime/start", json={"board_id": "synthetic", "params": {}})
    assert resp2.json()['ok'] is False
    assert 'state' in str(resp2.json()).lower() or 'error' in str(resp2.json()).lower()

    client.post("/api/realtime/stop")


# ========== 2. 快速启停循环 ==========

def test_boundary_rapid_start_stop_cycles(client):
    """快速启停 5 次循环,验证状态机一致性"""
    for i in range(5):
        resp = client.post("/api/realtime/start", json={"board_id": "synthetic", "params": {}})
        assert resp.json()['ok'] is True, f"Cycle {i}: start failed"
        time.sleep(0.1)
        assert client.get("/api/realtime/status").json()['state'] == 'STREAMING'
        client.post("/api/realtime/stop")
        assert client.get("/api/realtime/status").json()['state'] == 'IDLE'


# ========== 3. 并发 WebSocket 客户端 ==========

def test_boundary_concurrent_websocket_clients(client):
    """2 个 WebSocket 客户端并发连接,都应收到数据"""
    client.post("/api/realtime/start", json={"board_id": "synthetic", "params": {}})
    time.sleep(0.3)

    received = {'client1': False, 'client2': False}

    def ws_client(name):
        try:
            with client.websocket_connect("/ws/realtime") as ws:
                for _ in range(15):
                    frame = ws.receive_json()
                    if frame.get('type') == 'data':
                        received[name] = True
                        return
        except Exception:
            pass

    t1 = threading.Thread(target=ws_client, args=('client1',))
    t2 = threading.Thread(target=ws_client, args=('client2',))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert received['client1'], "Client1 未收到数据"
    assert received['client2'], "Client2 未收到数据"

    client.post("/api/realtime/stop")


# ========== 4. WebSocket 断连重连 ==========

def test_boundary_websocket_disconnect_reconnect(client):
    """WebSocket 断连后重连,仍能接收数据"""
    client.post("/api/realtime/start", json={"board_id": "synthetic", "params": {}})
    time.sleep(0.3)

    # 第一次连接
    with client.websocket_connect("/ws/realtime") as ws1:
        for _ in range(10):
            frame = ws1.receive_json()
            if frame.get('type') == 'data':
                break
    # ws1 已关闭

    # 重连
    with client.websocket_connect("/ws/realtime") as ws2:
        got_data = False
        for _ in range(15):
            frame = ws2.receive_json()
            if frame.get('type') == 'data':
                got_data = True
                break
        assert got_data, "重连后未收到数据"

    client.post("/api/realtime/stop")


# ========== 5. 长时间采集(缓冲边界) ==========

def test_boundary_long_acquisition_buffer_cap(client):
    """采集 5 秒,验证缓冲不超过 maxlen"""
    from app.realtime.manager import get_manager

    manager = get_manager()
    manager.start('synthetic', {})
    time.sleep(5.0)

    # 缓冲应不超过 maxlen=40000 (5000 samples/ch * 8 ch)
    assert len(manager._buffer) <= 40000, f"Buffer overflow: {len(manager._buffer)}"
    # 5 秒 @250Hz * 8 通道 = 10000 元组
    assert len(manager._buffer) > 0

    manager.stop()


# ========== 6. 无效参数 ==========

def test_boundary_cyton_without_serial_port(client):
    """Cyton 板卡未提供 serial_port 应失败(但不崩溃)"""
    resp = client.post("/api/realtime/start", json={"board_id": "cyton", "params": {}})
    assert resp.json()['ok'] is False
    # 状态应为 ERROR 或 IDLE,不能卡在 CONNECTING
    status = client.get("/api/realtime/status").json()
    assert status['state'] in ('IDLE', 'ERROR'), f"Stuck in state: {status['state']}"

    # 清理: stop 应能恢复到 IDLE
    client.post("/api/realtime/stop")
    assert client.get("/api/realtime/status").json()['state'] == 'IDLE'


def test_boundary_empty_board_id(client):
    """空 board_id 应失败"""
    resp = client.post("/api/realtime/start", json={"board_id": "", "params": {}})
    assert resp.json()['ok'] is False


# ========== 7. 单例一致性 ==========

def test_boundary_manager_singleton():
    """get_manager 多次调用返回同一实例"""
    from app.realtime.manager import get_manager
    m1 = get_manager()
    m2 = get_manager()
    assert m1 is m2


# ========== 8. WebSocket 心跳 ==========

def test_boundary_websocket_heartbeat(client):
    """未启动采集时连接 WS,应在 1 秒内收到 ping"""
    with client.websocket_connect("/ws/realtime") as ws:
        frame = ws.receive_json()
        # 未采集时应收到 ping
        assert frame.get('type') == 'ping'


# ========== 9. 采集中的 Focus 更新 ==========

def test_boundary_focus_updates_after_5s(client):
    """采集 5 秒后,Focus 缓存应被更新(需 4s 窗口数据)"""
    from app.realtime.manager import get_manager

    manager = get_manager()
    manager.start('synthetic', {})
    time.sleep(0.5)

    # 初始 Focus 为默认值
    initial_focus = manager._focus_cache.copy()
    assert initial_focus == {'avg': 0.0, 'stability': 0.0}

    # 等待 5 秒(4s 窗口 + 1s 余量,让 Focus 计算有足够数据)
    time.sleep(5.0)

    # Focus 应已更新(Synthetic Board 数据丰富,应非零)
    updated_focus = manager._focus_cache
    assert updated_focus != initial_focus or updated_focus['avg'] != 0.0, \
        f"Focus 未更新: {updated_focus}"

    manager.stop()


# ========== 10. 丢包计数器 ==========

def test_boundary_packets_lost_counter(client):
    """采集期间丢包计数器应为 0(Synthetic Board 不会丢包)"""
    client.post("/api/realtime/start", json={"board_id": "synthetic", "params": {}})
    time.sleep(1.0)

    status = client.get("/api/realtime/status").json()
    assert status['packets_lost'] == 0, f"Synthetic Board 不应丢包,但 packets_lost={status['packets_lost']}"

    client.post("/api/realtime/stop")
