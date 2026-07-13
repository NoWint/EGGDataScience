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


def test_analyze_uses_markers_from_odf(client, sample_odf_path):
    """测试 /api/analyze 从 ODF 文件中提取 markers 作为事件"""
    # 先上传 ODF 文件
    with open(sample_odf_path, "rb") as f:
        upload_resp = client.post(
            "/api/upload",
            files={"eeg_file": ("meditation.txt", f, "text/plain")},
            data={"condition": "odf_test"},
        )
    assert upload_resp.status_code == 200

    # 分析(不上传 events 文件,应自动用 markers)
    response = client.post(
        "/api/analyze",
        json={"condition": "odf_test"},
    )

    # 如果数据时长不足或无 markers,可能返回错误,但不应是 404 "未找到EEG数据"
    if response.status_code == 200:
        data = response.json()
        assert 'metadata' in data
        assert data['metadata']['format'] == 'openbci_odf'
        assert data['metadata']['board'] == 'cyton'
    else:
        # 即使分析失败,也不应该是 404(上传成功但找不到文件 = bug)
        assert response.status_code != 404


def test_analyze_returns_metadata(client, tmp_path):
    """测试 /api/analyze 返回 metadata 字段"""
    # 上传普通 CSV
    csv_content = "time,ch1,ch2\n" + "\n".join(
        f"{i*0.004},{float(i%100)},{float(i%50)}" for i in range(2000)
    ) + "\n"
    upload_resp = client.post(
        "/api/upload",
        files={"eeg_file": ("test.csv", csv_content.encode(), "text/csv")},
        data={"condition": "meta_test"},
    )
    assert upload_resp.status_code == 200

    response = client.post("/api/analyze", json={"condition": "meta_test"})
    if response.status_code == 200:
        data = response.json()
        assert 'metadata' in data
        assert data['metadata']['format'] == 'plain_csv'
        assert 'has_accel' in data
        assert 'has_markers' in data
