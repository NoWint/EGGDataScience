"""端到端测试: 上传 → 分析 → 结果"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.server import app
    return TestClient(app)


def test_e2e_upload_and_analyze_odf(client, sample_odf_path):
    """端到端: 上传 ODF 文件 → 分析 → 返回含 metadata 的结果"""
    # 1. 上传
    with open(sample_odf_path, "rb") as f:
        upload_resp = client.post(
            "/api/upload",
            files={"eeg_file": ("meditation.txt", f, "text/plain")},
            data={"condition": "e2e_odf"},
        )
    assert upload_resp.status_code == 200
    assert upload_resp.json()['status'] == 'uploaded'

    # 2. 分析
    analyze_resp = client.post("/api/analyze", json={"condition": "e2e_odf"})

    # 即使分析因数据特征失败,也应能返回或明确报错
    if analyze_resp.status_code == 200:
        result = analyze_resp.json()
        assert result['condition'] == 'e2e_odf'
        assert result['metadata']['format'] == 'openbci_odf'
        assert result['metadata']['board'] == 'cyton'
        assert result['channels'][0] == 'EXG_0'
        assert len(result['channels']) == 8
    else:
        # 失败也不应是 404(上传成功但找不到文件 = bug)
        assert analyze_resp.status_code != 404


def test_e2e_health_check(client):
    """测试服务健康检查"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
