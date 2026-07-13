"""API 端点测试"""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.server import app
    return TestClient(app)


def test_upload_accepts_txt(client, sample_odf_path):
    """测试 /api/upload 接受 .txt 文件"""
    with open(sample_odf_path, "rb") as f:
        response = client.post(
            "/api/upload",
            files={"eeg_file": ("test.txt", f, "text/plain")},
            data={"condition": "test_txt"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "uploaded"
    assert data["condition"] == "test_txt"


def test_upload_accepts_csv(client, tmp_path):
    """测试 /api/upload 仍接受 .csv 文件"""
    csv_content = "time,ch1\n0,1.0\n0.004,2.0\n"
    response = client.post(
        "/api/upload",
        files={"eeg_file": ("test.csv", csv_content.encode(), "text/csv")},
        data={"condition": "test_csv"},
    )
    assert response.status_code == 200


def test_upload_rejects_unsupported(client):
    """测试 /api/upload 拒绝不支持的格式"""
    response = client.post(
        "/api/upload",
        files={"eeg_file": ("test.json", b'{"a":1}', "application/json")},
        data={"condition": "test_json"},
    )
    assert response.status_code == 400
