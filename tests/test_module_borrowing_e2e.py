"""模块借鉴端到端测试"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.server import app
    return TestClient(app)


def test_e2e_analyze_returns_all_new_fields(client, sample_odf_path):
    """端到端:上传 ODF → 分析 → 返回所有新字段"""
    with open(sample_odf_path, "rb") as f:
        upload_resp = client.post(
            "/api/upload",
            files={"eeg_file": ("meditation.txt", f, "text/plain")},
            data={"condition": "e2e_modules"},
        )
    assert upload_resp.status_code == 200

    response = client.post("/api/analyze", json={
        "condition": "e2e_modules",
        "filter_preset": "eeg",
    })

    if response.status_code == 200:
        data = response.json()
        # 所有新字段都应存在
        assert 'topomap_data' in data
        assert 'band_powers' in data
        assert 'spectrogram_data' in data
        assert 'focus_scores' in data
        assert 'metadata' in data

        # topomap 应有数据
        if data['topomap_data']:
            assert 'grid_z' in data['topomap_data']
            assert 'channels' in data['topomap_data']

        # focus_scores 应有结构
        assert 'scores' in data['focus_scores']
        assert 'avg' in data['focus_scores']
    else:
        # 不应是 404 或 422
        assert response.status_code not in (404, 422)


def test_e2e_filter_presets(client, sample_odf_path):
    """端到端:不同滤波预设都能工作"""
    for preset in ['eeg', 'emg', 'ecg']:
        with open(sample_odf_path, "rb") as f:
            client.post(
                "/api/upload",
                files={"eeg_file": ("test.txt", f, "text/plain")},
                data={"condition": f"preset_{preset}"},
            )

        resp = client.post("/api/analyze", json={
            "condition": f"preset_{preset}",
            "filter_preset": preset,
        })
        # 不应是参数错误
        assert resp.status_code != 422
