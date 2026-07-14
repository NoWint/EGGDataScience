"""测试 NeuroLink 客户端"""
import json
import time
from app.neurolink_client import NeuroLinkClient


def test_client_initialization():
    """NeuroLinkClient 初始化应有默认状态"""
    client = NeuroLinkClient()
    assert client.connected is False
    assert client.recording is False
    assert client.room_code is None


def test_client_record_frame_writes_csv(tmp_path):
    """记录 EEG 帧应写入 CSV 文件"""
    client = NeuroLinkClient()
    csv_path = tmp_path / "test_record.csv"
    client.start_recording(str(csv_path), subject="S01", condition="AtoA")
    client._write_frame({"type": "eeg_frame", "seq": 0,
                         "channels": [1.0, 2.0, 3.0, 4.0], "ts": 1710400000000})
    client.stop_recording()
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "1.0000" in content
    assert "Index" in content  # 表头


def test_client_metrics_buffer():
    """客户端应缓存最新 metrics_snapshot"""
    client = NeuroLinkClient()
    client._on_metrics({"type": "metrics_snapshot", "theta_alpha_ratio": 1.5,
                        "spectral_entropy": 2.0, "cognitive_load_index": 0.3,
                        "band_power": {"delta": 10, "theta": 8, "alpha": 5, "beta": 3, "gamma": 1}})
    metrics = client.get_latest_metrics()
    assert metrics["theta_alpha_ratio"] == 1.5
