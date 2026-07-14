"""测试批量分析端点"""
import json
import time
from fastapi.testclient import TestClient
from app.server import app


client = TestClient(app)


def test_batch_analyze_returns_batch_id():
    """POST /api/batch-analyze 应返回 batch_id"""
    # 用一个小的测试 CSV 文件
    import tempfile
    import numpy as np
    fs = 250
    n = fs * 10  # 10 秒
    t = np.linspace(0, 10, n)
    # BrainFlow RAW 格式: Index, EXG0-3, Accel, Timestamp
    rows = []
    for i in range(n):
        rows.append(f"{i}\t{10*np.sin(2*np.pi*10*t[i]):.4f}\t{5*np.sin(2*np.pi*10*t[i]+1):.4f}\t{8*np.sin(2*np.pi*10*t[i]+2):.4f}\t{6*np.sin(2*np.pi*10*t[i]+3):.4f}\t0\t0\t0\t{int(t[i]*1000)}\t0")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('\n'.join(rows))
        f.flush()
        filepath = f.name

    import os
    filename = os.path.basename(filepath)
    assignments = json.dumps([
        {"filename": filename, "subject": "S01", "condition": "AtoA"}
    ])

    with open(filepath, 'rb') as eeg_file:
        resp = client.post(
            "/api/batch-analyze",
            files={"files": (filename, eeg_file, "text/csv")},
            data={"assignments": assignments},
        )

    os.unlink(filepath)
    assert resp.status_code == 200
    data = resp.json()
    assert "batch_id" in data
    assert data["total"] == 1


def test_batch_progress_returns_status():
    """GET /api/batch-progress/{batch_id} 应返回进度状态"""
    # 先发起一个批量分析
    import tempfile, os, json
    import numpy as np
    fs = 250
    n = fs * 10
    rows = [f"{i}\t{10*np.sin(2*np.pi*10*i/fs):.4f}\t0\t0\t0\t0\t0\t0\t{int(i/fs*1000)}\t0" for i in range(n)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('\n'.join(rows)); f.flush()
        filepath = f.name
    filename = os.path.basename(filepath)
    assignments = json.dumps([{"filename": filename, "subject": "S01", "condition": "AtoA"}])
    with open(filepath, 'rb') as fh:
        resp = client.post("/api/batch-analyze", files={"files": (filename, fh, "text/csv")}, data={"assignments": assignments})
    os.unlink(filepath)
    batch_id = resp.json()["batch_id"]

    # 查询进度
    resp2 = client.get(f"/api/batch-progress/{batch_id}")
    assert resp2.status_code == 200
    progress = resp2.json()
    assert progress["status"] in ("running", "done", "failed")
    assert "total" in progress
    assert "current" in progress


def test_export_batch_report_returns_zip():
    """GET /api/export-batch-report 应返回 ZIP 文件"""
    import tempfile, os, json, time
    import numpy as np
    fs = 250
    n = fs * 10
    rows = [f"{i}\t{10*np.sin(2*np.pi*10*i/fs):.4f}\t0\t0\t0\t0\t0\t0\t{int(i/fs*1000)}\t0" for i in range(n)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('\n'.join(rows)); f.flush()
        filepath = f.name
    filename = os.path.basename(filepath)
    assignments = json.dumps([{"filename": filename, "subject": "S01", "condition": "AtoA"}])
    with open(filepath, 'rb') as fh:
        resp = client.post("/api/batch-analyze", files={"files": (filename, fh, "text/csv")}, data={"assignments": assignments})
    os.unlink(filepath)
    batch_id = resp.json()["batch_id"]

    # 等待分析完成
    for _ in range(60):
        prog = client.get(f"/api/batch-progress/{batch_id}").json()
        if prog["status"] != "running":
            break
        time.sleep(1)

    # 导出
    resp2 = client.get(f"/api/export-batch-report?batch_id={batch_id}")
    assert resp2.status_code == 200
    assert "zip" in resp2.headers.get("content-type", "").lower()
    # 验证是合法 ZIP
    import zipfile, io
    zf = zipfile.ZipFile(io.BytesIO(resp2.content))
    names = zf.namelist()
    assert "batch_summary.md" in names
    assert any(n.startswith("per_file/") for n in names)
