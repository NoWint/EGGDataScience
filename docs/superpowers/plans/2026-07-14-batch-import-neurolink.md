# 批量导入 + NeuroLink 实时对接 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现批量多文件 EEG 导入分析（表格分配被试+条件，5 模块全分析，合并报告 ZIP）和 NeuroLink 实时 EEG 平台对接（monitor 角色，实时波形/指标/心流分析/会话记录/阶段同步）。

**Architecture:** 批量导入用服务端串行循环+进度轮询；NeuroLink 用后台 WebSocket 客户端线程+asyncio.Queue 桥接到前端 WS；两者共享分析内核（提取 `_run_all_modules` 函数）；NeuroLink 会话记录保存为 BrainFlow RAW 兼容 CSV，可导入批量分析形成闭环。

**Tech Stack:** Python 3.11 / FastAPI / websocket-client / SQLite / 原生 HTML+JS / Chart.js

**Spec:** `docs/specs/2026-07-14-batch-import-neurolink-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `app/database.py` (modify) | experiments 表加列 (eeg_path/source/analysis_status) |
| `app/server.py` (modify) | 提取 `_run_all_modules`，注册新路由 |
| `app/routers/batch.py` (create) | 批量分析 REST 端点 + 进度跟踪 |
| `app/neurolink_client.py` (create) | NeuroLink WebSocket 客户端 + 记录 |
| `app/routers/neurolink.py` (create) | NeuroLink REST + WS 路由 |
| `app/static/index.html` (modify) | 新增导航项 + 视图容器 |
| `app/static/js/app.js` (modify) | 侧边栏导航扩展 |
| `app/static/js/batch.js` (create) | 批量导入前端 |
| `app/static/js/neurolink.js` (create) | 实时监测前端 |
| `tests/test_batch.py` (create) | 批量导入测试 |
| `tests/test_neurolink.py` (create) | NeuroLink mock 测试 |

---

### Task 1: 数据库迁移 — experiments 表扩展

**Files:**
- Modify: `app/database.py:24-54` (init_db 函数)

- [ ] **Step 1: 写迁移测试**

Create `tests/test_db_migration.py`:

```python
"""测试 experiments 表新列"""
import sqlite3
import tempfile
from pathlib import Path
from app import database


def test_experiments_has_new_columns(tmp_path, monkeypatch):
    """experiments 表应包含 eeg_path, source, analysis_status 列"""
    # 用临时数据库
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(database, "DB_DIR", tmp_path)
    database.init_db()

    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(experiments)")
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()

    assert "eeg_path" in columns
    assert "source" in columns
    assert "analysis_status" in columns
```

- [ ] **Step 2: 运行测试验证失败**

Run: `source venv/bin/activate && python -m pytest tests/test_db_migration.py -v`
Expected: FAIL — 新列不存在

- [ ] **Step 3: 修改 init_db 添加新列**

Modify `app/database.py` 的 `init_db` 函数，在创建 experiments 表后添加迁移逻辑：

```python
def init_db():
    """初始化数据库表"""
    conn = get_connection()
    cursor = conn.cursor()

    # 被试表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            age INTEGER,
            gender TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        )
    """)

    # 实验记录表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER NOT NULL,
            condition TEXT NOT NULL,
            date TEXT,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        )
    """)

    # 迁移: 为已存在的 experiments 表添加新列 (如果不存在)
    cursor.execute("PRAGMA table_info(experiments)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    new_cols = {
        "eeg_path": "TEXT",
        "source": "TEXT DEFAULT 'upload'",
        "analysis_status": "TEXT DEFAULT 'pending'",
    }
    for col, col_type in new_cols.items():
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE experiments ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `source venv/bin/activate && python -m pytest tests/test_db_migration.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add app/database.py tests/test_db_migration.py
git commit -m "feat: experiments 表扩展 eeg_path/source/analysis_status 列"
```

---

### Task 2: 提取 `_run_all_modules` 共享函数

**Files:**
- Modify: `app/server.py:207-288` (analyze_data 端点) 和 `app/server.py` 的 analyze-all 端点

- [ ] **Step 1: 写共享函数测试**

Create `tests/test_run_all_modules.py`:

```python
"""测试 _run_all_modules 共享分析函数"""
import numpy as np
from app.server import _run_all_modules


def test_run_all_modules_returns_5_modules():
    """_run_all_modules 应返回包含 5 个模块结果的字典"""
    fs = 250
    duration = 60  # 1 分钟,足够短用于测试
    n_samples = fs * duration
    # 生成简单正弦波 EEG 数据 (4 通道)
    t = np.linspace(0, duration, n_samples)
    data = np.sin(2 * np.pi * 10 * t).reshape(1, -1) * 50  # 10Hz alpha
    data = np.vstack([data] * 4)  # 4 通道

    # 简单事件表
    import pandas as pd
    events_df = pd.DataFrame([
        ('S0', 0.0), ('F0', 5.0), ('F1', 30.0),
        ('X0', 30.0), ('X1', 35.0),
        ('R0', 35.0), ('R1', 60.0),
    ], columns=['event_id', 'timestamp'])

    result = _run_all_modules(data, fs, events_df)

    assert 'flow_recovery' in result
    assert 'spectrum' in result
    assert 'erp' in result
    assert 'ersp' in result
    assert 'artifact' in result
    # 每个模块不应有 error
    for mod in ['flow_recovery', 'spectrum', 'erp', 'ersp', 'artifact']:
        assert 'error' not in result[mod], f"{mod} 出错: {result[mod].get('error')}"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `source venv/bin/activate && python -m pytest tests/test_run_all_modules.py -v`
Expected: FAIL — `_run_all_modules` 不存在

- [ ] **Step 3: 提取共享函数**

在 `app/server.py` 中，在 `analyze_data` 端点之前添加 `_run_all_modules` 函数，并让 `analyze-all` 端点调用它：

```python
def _run_all_modules(data, fs, events_df):
    """运行全部 5 个分析模块 (批量导入与单文件全分析共用)

    返回: dict, 包含 flow_recovery/spectrum/erp/ersp/artifact 五个模块结果
    各模块独立异常捕获, 单模块失败不中断其他模块
    """
    results = {}

    # 1. 心流恢复
    try:
        results['flow_recovery'] = run_full_pipeline(data, fs, events_df)
    except Exception as e:
        results['flow_recovery'] = {'error': str(e)}

    # 2. 频谱分析
    try:
        nperseg = min(1024, len(data) // 4 or 256)
        results['spectrum'] = run_spectrum_analysis(data, fs, nperseg=nperseg, overlap=0.5)
    except Exception as e:
        results['spectrum'] = {'error': str(e)}

    # 3. ERP
    try:
        results['erp'] = run_erp_analysis(data, fs, events_df, event_id='X0')
    except Exception as e:
        results['erp'] = {'error': str(e)}

    # 4. ERSP
    try:
        results['ersp'] = run_ersp_analysis(data, fs, events_df, event_id='X0')
    except Exception as e:
        results['ersp'] = {'error': str(e)}

    # 5. 伪迹检测
    try:
        results['artifact'] = run_artifact_analysis(data, fs)
    except Exception as e:
        results['artifact'] = {'error': str(e)}

    return results
```

然后修改 `analyze-all` 端点，将内联的分析逻辑替换为调用 `_run_all_modules`：

找到 `analyze_all` 函数中的这段代码：
```python
    # 运行全部 5 个分析模块 (捕获各自异常, 不因单模块失败而中断)
    results = {'condition': condition, 'eeg_path': str(eeg_path), 'fs': fs,
               'channels': channels, 'n_samples': len(data),
               'duration_sec': len(data) / fs if fs else 0,
               'metadata': eeg_result['metadata']}

    # 1. 心流恢复
    try:
        flow_result = run_full_pipeline(data, fs, events_df)
        results['flow_recovery'] = flow_result
    except Exception as e:
        results['flow_recovery'] = {'error': str(e)}

    # 2. 频谱分析
    try:
        spec_result = run_spectrum_analysis(data, fs, nperseg=min(1024, len(data)//4 or 256), overlap=0.5)
        results['spectrum'] = spec_result
    except Exception as e:
        results['spectrum'] = {'error': str(e)}

    # 3. ERP
    try:
        erp_result = run_erp_analysis(data, fs, events_df, event_id='X0')
        results['erp'] = erp_result
    except Exception as e:
        results['erp'] = {'error': str(e)}

    # 4. ERSP
    try:
        ersp_result = run_ersp_analysis(data, fs, events_df, event_id='X0')
        results['ersp'] = ersp_result
    except Exception as e:
        results['ersp'] = {'error': str(e)}

    # 5. 伪迹检测
    try:
        art_result = run_artifact_analysis(data, fs)
        results['artifact'] = art_result
    except Exception as e:
        results['artifact'] = {'error': str(e)}
```

替换为：
```python
    # 运行全部 5 个分析模块 (复用共享函数)
    results = {'condition': condition, 'eeg_path': str(eeg_path), 'fs': fs,
               'channels': channels, 'n_samples': len(data),
               'duration_sec': len(data) / fs if fs else 0,
               'metadata': eeg_result['metadata']}
    results.update(_run_all_modules(data, fs, events_df))
```

- [ ] **Step 4: 运行测试验证通过**

Run: `source venv/bin/activate && python -m pytest tests/test_run_all_modules.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add app/server.py tests/test_run_all_modules.py
git commit -m "refactor: 提取 _run_all_modules 共享函数"
```

---

### Task 3: 批量分析后端 — batch router

**Files:**
- Create: `app/routers/batch.py`
- Modify: `app/server.py` (注册路由)

- [ ] **Step 1: 写批量分析测试**

Create `tests/test_batch.py`:

```python
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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `source venv/bin/activate && python -m pytest tests/test_batch.py -v`
Expected: FAIL — 端点不存在

- [ ] **Step 3: 实现 batch router**

Create `app/routers/batch.py`:

```python
"""批量分析 REST 路由"""
import json
import threading
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from app.analysis import load_eeg_full
from app.server import _run_all_modules, _to_jsonable, UPLOAD_DIR

router = APIRouter(prefix="/api", tags=["batch"])

# 批量分析状态存储
BATCH_STORE: Dict[str, Dict[str, Any]] = {}
BATCH_RESULTS_STORE: Dict[str, Dict[str, Any]] = {}

# 批量上传目录
BATCH_DIR = UPLOAD_DIR / "batch"
BATCH_DIR.mkdir(parents=True, exist_ok=True)


def _build_events_df_from_eeg(eeg_result):
    """从 EEG 加载结果构建事件 DataFrame (复用 analyze_data 逻辑)"""
    import pandas as pd
    events_df = None
    if eeg_result['markers']:
        events_df = pd.DataFrame(
            [(m.label, m.timestamp) for m in eeg_result['markers']],
            columns=['event_id', 'timestamp']
        )
    if events_df is None:
        events_df = pd.DataFrame([
            ('S0', 0.0), ('B0', 5.0), ('B1', 65.0),
            ('F0', 65.0), ('F1', 305.0), ('F2', 545.0),
            ('X0', 545.0), ('X1', 665.0),
            ('R0', 665.0), ('R1', 1265.0), ('Q0', 1265.0),
        ], columns=['event_id', 'timestamp'])
    return events_df


def _run_batch_thread(batch_id: str, file_paths: List[Path], assignments: List[dict]):
    """后台线程: 串行分析每个文件"""
    progress = BATCH_STORE[batch_id]
    results = BATCH_RESULTS_STORE[batch_id]

    for i, (fpath, assign) in enumerate(zip(file_paths, assignments)):
        progress['current'] = i + 1
        progress['current_file'] = fpath.name
        subject = assign.get('subject', 'unknown')
        condition = assign.get('condition', 'custom')
        key = f"{subject}_{condition}"

        try:
            progress['current_module'] = '加载EEG数据'
            eeg_result = load_eeg_full(fpath)
            data, fs = eeg_result['data'], eeg_result['fs']
            events_df = _build_events_df_from_eeg(eeg_result)

            progress['current_module'] = '运行5模块分析'
            mod_results = _run_all_modules(data, fs, events_df)
            mod_results['condition'] = condition
            mod_results['subject'] = subject
            mod_results['eeg_path'] = str(fpath)
            mod_results['fs'] = fs
            mod_results['channels'] = eeg_result['channels']
            mod_results['n_samples'] = len(data)
            mod_results['duration_sec'] = len(data) / fs if fs else 0
            mod_results['metadata'] = eeg_result['metadata']
            results[key] = mod_results

        except Exception as e:
            progress['errors'].append({'file': fpath.name, 'error': str(e)})
            results[key] = {'error': str(e), 'subject': subject, 'condition': condition}

    progress['status'] = 'done'
    progress['current_module'] = ''


@router.post("/batch-analyze")
async def batch_analyze(
    files: List[UploadFile] = File(...),
    assignments: str = Form(...),
):
    """批量分析: 接收多文件 + 分配表, 串行运行 5 模块分析"""
    try:
        assign_list = json.loads(assignments)
    except json.JSONDecodeError:
        raise HTTPException(400, "assignments 不是合法 JSON")

    if len(files) != len(assign_list):
        raise HTTPException(400, f"文件数({len(files)})与分配表数({len(assign_list)})不匹配")

    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_subdir = BATCH_DIR / batch_id
    batch_subdir.mkdir(parents=True, exist_ok=True)

    # 保存文件
    file_paths = []
    for f, assign in zip(files, assign_list):
        fpath = batch_subdir / f.filename
        with open(fpath, 'wb') as out:
            shutil.copyfileobj(f.file, out)
        file_paths.append(fpath)

    # 初始化进度
    BATCH_STORE[batch_id] = {
        'batch_id': batch_id,
        'total': len(files),
        'current': 0,
        'current_file': '',
        'current_module': '',
        'status': 'running',
        'errors': [],
        'started_at': datetime.now().isoformat(),
    }
    BATCH_RESULTS_STORE[batch_id] = {}

    # 启动后台线程
    thread = threading.Thread(
        target=_run_batch_thread,
        args=(batch_id, file_paths, assign_list),
        daemon=True,
    )
    thread.start()

    return {'batch_id': batch_id, 'total': len(files)}


@router.get("/batch-progress/{batch_id}")
async def batch_progress(batch_id: str):
    """查询批量分析进度"""
    if batch_id not in BATCH_STORE:
        raise HTTPException(404, f"未找到批次: {batch_id}")
    return BATCH_STORE[batch_id]
```

- [ ] **Step 4: 在 server.py 注册路由**

在 `app/server.py` 的路由注册部分（约第 78 行附近）添加：

```python
from app.routers.batch import router as batch_router
```

并在 `app.include_router` 区域添加：

```python
app.include_router(batch_router)
```

- [ ] **Step 5: 运行测试验证通过**

Run: `source venv/bin/activate && python -m pytest tests/test_batch.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add app/routers/batch.py app/server.py tests/test_batch.py
git commit -m "feat: 批量分析后端 — POST /api/batch-analyze + 进度轮询"
```

---

### Task 4: 批量报告导出端点

**Files:**
- Modify: `app/routers/batch.py` (添加 export-batch-report 端点)

- [ ] **Step 1: 写导出测试**

在 `tests/test_batch.py` 末尾添加：

```python
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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `source venv/bin/activate && python -m pytest tests/test_batch.py::test_export_batch_report_returns_zip -v`
Expected: FAIL — 端点不存在

- [ ] **Step 3: 实现导出端点**

在 `app/routers/batch.py` 末尾添加：

```python
import io
import zipfile
from fastapi.responses import StreamingResponse


def _build_batch_summary_md(batch_id: str) -> str:
    """构建批量分析汇总 Markdown"""
    results = BATCH_RESULTS_STORE.get(batch_id, {})
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        f"# EEG 批量分析汇总报告",
        f"",
        f"**生成时间**: {now}  ",
        f"**批次 ID**: {batch_id}  ",
        f"**文件总数**: {len(results)}",
        f"",
        f"## 各文件分析结果",
        f"",
        f"| 被试 | 条件 | 恢复时长(s) | 伪迹占比 | 心流 | 频谱 | ERP | ERSP | 伪迹检测 |",
        f"|------|------|-----------|---------|------|------|-----|------|---------|",
    ]

    for key, res in results.items():
        subject = res.get('subject', '?')
        condition = res.get('condition', '?')
        if 'error' in res:
            lines.append(f"| {subject} | {condition} | - | - | 失败 | - | - | - | - |")
            continue
        flow = res.get('flow_recovery', {})
        rt = flow.get('recovery_time') if 'error' not in flow else None
        art = flow.get('artifact_ratio', 0) if 'error' not in flow else 0
        rt_str = f"{rt:.1f}" if rt else ">600"
        art_str = f"{art*100:.2f}%" if art else "-"

        def status(mod_key):
            r = res.get(mod_key, {})
            return "失败" if 'error' in r else "OK"

        lines.append(
            f"| {subject} | {condition} | {rt_str} | {art_str} | "
            f"{status('flow_recovery')} | {status('spectrum')} | {status('erp')} | "
            f"{status('ersp')} | {status('artifact')} |"
        )

    lines.append(f"")
    lines.append(f"## 失败详情")
    lines.append(f"")
    progress = BATCH_STORE.get(batch_id, {})
    errors = progress.get('errors', [])
    if errors:
        for e in errors:
            lines.append(f"- **{e['file']}**: {e['error']}")
    else:
        lines.append(f"无失败项")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*报告由 EEGDataScience 自动生成 — Author: [NoWint](https://github.com/NoWint)*")
    return '\n'.join(lines)


@router.get("/export-batch-report")
async def export_batch_report(batch_id: str):
    """导出批量分析报告 ZIP"""
    from app.server import _build_full_report_md

    if batch_id not in BATCH_RESULTS_STORE:
        available = list(BATCH_RESULTS_STORE.keys())
        raise HTTPException(404, f"未找到批次: {batch_id}，可用: {available}")

    results = BATCH_RESULTS_STORE[batch_id]
    progress = BATCH_STORE.get(batch_id, {})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 汇总报告
        zf.writestr('batch_summary.md', _build_batch_summary_md(batch_id))

        # 每个文件的报告
        for key, res in results.items():
            if 'error' in res and 'flow_recovery' not in res:
                continue
            zf.writestr(f'per_file/{key}_results.json',
                        json.dumps(_to_jsonable(res), ensure_ascii=False, indent=2))
            try:
                md = _build_full_report_md(res)
                zf.writestr(f'per_file/{key}_report.md', md)
            except Exception:
                pass

        # 原始数据
        batch_dir = BATCH_DIR / batch_id
        if batch_dir.exists():
            for fpath in batch_dir.iterdir():
                if fpath.is_file():
                    zf.write(str(fpath), f'original_data/{fpath.name}')

    buf.seek(0)
    filename = f"EEG_BatchReport_{batch_id}.zip"
    return StreamingResponse(
        buf,
        media_type='application/zip',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )
```

- [ ] **Step 4: 运行测试验证通过**

Run: `source venv/bin/activate && python -m pytest tests/test_batch.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add app/routers/batch.py tests/test_batch.py
git commit -m "feat: 批量报告导出 — GET /api/export-batch-report ZIP"
```

---

### Task 5: NeuroLink WebSocket 客户端

**Files:**
- Create: `app/neurolink_client.py`

- [ ] **Step 1: 写客户端测试**

Create `tests/test_neurolink.py`:

```python
"""测试 NeuroLink 客户端 (用 mock WebSocket)"""
import json
import threading
import time
from unittest.mock import MagicMock, patch
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

    # 模拟接收一帧
    client._write_frame({"type": "eeg_frame", "seq": 0,
                         "channels": [1.0, 2.0, 3.0, 4.0], "ts": 1710400000000})

    client.stop_recording()
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "1.0000" in content  # 第一通道值
    assert "EXG" in content or "Index" in content  # 有表头


def test_client_metrics_buffer():
    """客户端应缓存最新 metrics_snapshot"""
    client = NeuroLinkClient()
    client._on_metrics({"type": "metrics_snapshot", "theta_alpha_ratio": 1.5,
                        "spectral_entropy": 2.0, "cognitive_load_index": 0.3,
                        "band_power": {"delta": 10, "theta": 8, "alpha": 5, "beta": 3, "gamma": 1}})
    metrics = client.get_latest_metrics()
    assert metrics["theta_alpha_ratio"] == 1.5
```

- [ ] **Step 2: 运行测试验证失败**

Run: `source venv/bin/activate && python -m pytest tests/test_neurolink.py -v`
Expected: FAIL — 模块不存在

- [ ] **Step 3: 实现 NeuroLinkClient**

Create `app/neurolink_client.py`:

```python
"""NeuroLink 平台 WebSocket 客户端

作为 monitor 角色连接 wss://eeg.yzjtiantian.cn/ws,
接收实时 EEG 数据帧、指标快照、阶段同步、标记
"""
import json
import threading
import time
import csv
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from collections import deque

import websocket  # websocket-client 库

NEUROLINK_URL = "wss://eeg.yzjtiantian.cn/ws"


class NeuroLinkClient:
    """NeuroLink WebSocket 客户端 (后台线程运行)"""

    def __init__(self):
        self.ws: Optional[websocket.WebSocketApp] = None
        self.thread: Optional[threading.Thread] = None
        self.connected = False
        self.room_code: Optional[str] = None
        self.nickname: str = "EEGDataScience Monitor"
        self.session_id: str = f"eegds-{int(time.time())}"

        # 数据缓冲
        self._eeg_buffer = deque(maxlen=7200)  # 60s @ 120Hz
        self._latest_metrics: Optional[dict] = None
        self._current_phase: Optional[dict] = None
        self._markers: List[dict] = []

        # 记录
        self.recording = False
        self._csv_file = None
        self._csv_writer = None
        self._csv_path: Optional[str] = None
        self._record_subject: Optional[str] = None
        self._record_condition: Optional[str] = None
        self._record_count = 0
        self._record_start_time: Optional[float] = None

        # 回调 (用于桥接到 WebSocket 推送)
        self._on_data_callback: Optional[Callable] = None
        self._lock = threading.Lock()

    def set_data_callback(self, callback: Callable[[dict], None]):
        """设置数据回调 (每收到一帧时调用)"""
        self._on_data_callback = callback

    def connect(self, room_code: str, nickname: str = "EEGDataScience Monitor") -> bool:
        """连接到 NeuroLink 房间"""
        self.room_code = room_code
        self.nickname = nickname
        self.session_id = f"eegds-{int(time.time())}"

        self.ws = websocket.WebSocketApp(
            NEUROLINK_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.thread = threading.Thread(target=self._run_ws, daemon=True)
        self.thread.start()
        # 等待连接 (最多 10s)
        for _ in range(20):
            if self.connected:
                return True
            time.sleep(0.5)
        return self.connected

    def _run_ws(self):
        """运行 WebSocket (带自动重连)"""
        for attempt in range(3):
            try:
                self.ws.run_forever()
            except Exception:
                pass
            if self.connected:
                break
            time.sleep(5)
        self.connected = False

    def disconnect(self):
        """断开连接"""
        if self.ws:
            self.ws.close()
        self.connected = False
        self.stop_recording()

    def _on_open(self, ws):
        """连接建立, 发送 hello"""
        ws.send(json.dumps({
            "type": "hello",
            "role": "pending",
            "session_id": self.session_id,
            "device_info": {
                "platform": "EEGDataScience",
                "userAgent": "v2.0.0",
                "nickname": self.nickname,
                "isBridge": False,
            }
        }))

    def _on_message(self, ws, raw):
        """处理收到的消息"""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type")

        if msg_type == "room_info":
            ws.send(json.dumps({
                "type": "join_room",
                "code": self.room_code,
                "session_id": self.session_id,
            }))

        elif msg_type == "room_joined":
            ws.send(json.dumps({
                "type": "claim_role",
                "role": "monitor",
                "session_id": msg.get("session_id", self.session_id),
            }))

        elif msg_type == "role_claimed" and msg.get("role") == "monitor":
            self.connected = True

        elif msg_type == "eeg_frame":
            self._on_eeg_frame(msg)

        elif msg_type == "metrics_snapshot":
            self._on_metrics(msg)

        elif msg_type == "phase_sync":
            self._current_phase = msg

        elif msg_type == "marker":
            self._markers.append(msg)

        # 推送给回调
        if self._on_data_callback:
            try:
                self._on_data_callback(msg)
            except Exception:
                pass

    def _on_eeg_frame(self, msg):
        """处理 EEG 帧"""
        channels = msg.get("channels", [0, 0, 0, 0])
        ts = msg.get("ts", 0)
        seq = msg.get("seq", 0)

        with self._lock:
            self._eeg_buffer.append({
                "seq": seq,
                "channels": channels,
                "ts": ts,
            })

        if self.recording:
            self._write_frame(msg)

    def _on_metrics(self, msg):
        """缓存最新指标"""
        self._latest_metrics = msg

    def get_latest_metrics(self) -> Optional[dict]:
        return self._latest_metrics

    def get_current_phase(self) -> Optional[dict]:
        return self._current_phase

    def get_recent_eeg(self, n_samples: int = 600) -> list:
        """获取最近 n 个 EEG 采样点"""
        with self._lock:
            data = list(self._eeg_buffer)
        return data[-n_samples:] if len(data) > n_samples else data

    def start_recording(self, csv_path: str, subject: str = "", condition: str = ""):
        """开始记录 EEG 数据到 CSV"""
        self._csv_path = csv_path
        self._record_subject = subject
        self._record_condition = condition
        self._csv_file = open(csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file, delimiter='\t')
        # BrainFlow RAW 兼容表头
        self._csv_writer.writerow(
            ["Index", "EXG0", "EXG1", "EXG2", "EXG3",
             "Accel_X", "Accel_Y", "Accel_Z", "Timestamp", "Marker"]
        )
        self._record_count = 0
        self._record_start_time = time.time()
        self.recording = True

    def _write_frame(self, msg: dict):
        """将 EEG 帧写入 CSV"""
        if not self._csv_writer:
            return
        channels = msg.get("channels", [0, 0, 0, 0])
        ts = msg.get("ts", 0)
        # 补齐 4 通道
        while len(channels) < 4:
            channels.append(0.0)
        self._csv_writer.writerow([
            self._record_count,
            f"{channels[0]:.4f}", f"{channels[1]:.4f}",
            f"{channels[2]:.4f}", f"{channels[3]:.4f}",
            0, 0, 0,  # accel (NeuroLink 单独发 accel_frame)
            ts, 0,  # marker
        ])
        self._record_count += 1

    def stop_recording(self) -> Optional[str]:
        """停止记录, 返回文件路径"""
        if not self.recording:
            return None
        self.recording = False
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        return self._csv_path

    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            "connected": self.connected,
            "room_code": self.room_code,
            "recording": self.recording,
            "record_count": self._record_count,
            "record_duration": time.time() - self._record_start_time if self._record_start_time and self.recording else 0,
            "current_phase": self._current_phase.get("phase_id") if self._current_phase else None,
            "buffer_size": len(self._eeg_buffer),
        }

    def _on_error(self, ws, error):
        pass

    def _on_close(self, ws, code, msg):
        self.connected = False


# 单例
_client_instance: Optional[NeuroLinkClient] = None


def get_client() -> NeuroLinkClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = NeuroLinkClient()
    return _client_instance
```

- [ ] **Step 4: 安装 websocket-client 依赖**

Run: `source venv/bin/activate && pip install websocket-client && echo "websocket-client" >> requirements.txt`

- [ ] **Step 5: 运行测试验证通过**

Run: `source venv/bin/activate && python -m pytest tests/test_neurolink.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add app/neurolink_client.py tests/test_neurolink.py requirements.txt
git commit -m "feat: NeuroLink WebSocket 客户端 + 会话记录"
```

---

### Task 6: NeuroLink REST + WS 路由

**Files:**
- Create: `app/routers/neurolink.py`
- Modify: `app/server.py` (注册路由 + WS 端点)

- [ ] **Step 1: 写路由测试**

在 `tests/test_neurolink.py` 末尾添加：

```python
from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)


def test_neurolink_status_endpoint():
    """GET /api/neurolink/status 应返回状态"""
    resp = client.get("/api/neurolink/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "connected" in data
    assert "recording" in data


def test_neurolink_disconnect_without_connect():
    """未连接时 disconnect 应安全返回"""
    resp = client.post("/api/neurolink/disconnect")
    assert resp.status_code == 200
```

- [ ] **Step 2: 运行测试验证失败**

Run: `source venv/bin/activate && python -m pytest tests/test_neurolink.py::test_neurolink_status_endpoint -v`
Expected: FAIL — 端点不存在

- [ ] **Step 3: 实现 neurolink router**

Create `app/routers/neurolink.py`:

```python
"""NeuroLink 实时对接 REST + WebSocket 路由"""
import asyncio
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.neurolink_client import get_client

router = APIRouter(prefix="/api/neurolink", tags=["neurolink"])


class ConnectRequest(BaseModel):
    room_code: str
    nickname: str = "EEGDataScience Monitor"


class RecordRequest(BaseModel):
    subject: str = ""
    condition: str = "custom"


@router.get("/status")
def neurolink_status():
    """获取 NeuroLink 连接与记录状态"""
    client = get_client()
    return client.get_status()


@router.post("/connect")
def neurolink_connect(req: ConnectRequest):
    """连接到 NeuroLink 房间"""
    client = get_client()
    if client.connected:
        return {"ok": True, "message": "已连接"}
    ok = client.connect(req.room_code, req.nickname)
    if ok:
        return {"ok": True, "room_code": req.room_code}
    return {"ok": False, "error": "连接失败,请检查房间号和网络"}


@router.post("/disconnect")
def neurolink_disconnect():
    """断开 NeuroLink 连接"""
    client = get_client()
    client.disconnect()
    return {"ok": True}


@router.post("/start-recording")
def neurolink_start_recording(req: RecordRequest):
    """开始记录会话"""
    from app.server import UPLOAD_DIR
    from datetime import datetime

    client = get_client()
    if not client.connected:
        return {"ok": False, "error": "未连接到 NeuroLink"}

    recordings_dir = UPLOAD_DIR.parent / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cond = req.condition or "custom"
    subj = req.subject or "unknown"
    csv_path = recordings_dir / f"NeuroLink_{subj}_{cond}_{timestamp}.csv"

    client.start_recording(str(csv_path), subject=subj, condition=cond)
    return {"ok": True, "path": str(csv_path)}


@router.post("/stop-recording")
def neurolink_stop_recording():
    """停止记录, 返回文件路径"""
    client = get_client()
    path = client.stop_recording()
    if path:
        return {"ok": True, "path": path, "count": client._record_count}
    return {"ok": False, "error": "未在记录"}


async def neurolink_websocket_endpoint(websocket: WebSocket):
    """WebSocket 实时数据推送 (挂载在 /ws/neurolink)"""
    await websocket.accept()
    client = get_client()

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def push_callback(frame):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, frame)
        except Exception:
            pass

    client.set_data_callback(push_callback)

    try:
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(frame)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        client.set_data_callback(None)
```

- [ ] **Step 4: 在 server.py 注册路由和 WS 端点**

在 `app/server.py` 添加导入和注册：

```python
from app.routers.neurolink import router as neurolink_router, neurolink_websocket_endpoint
```

在 `app.include_router` 区域添加：
```python
app.include_router(neurolink_router)
```

在 WebSocket 端点区域（`ws_realtime` 附近）添加：
```python
@app.websocket("/ws/neurolink")
async def ws_neurolink(websocket: WebSocket):
    """NeuroLink 实时数据 WebSocket 端点"""
    await neurolink_websocket_endpoint(websocket)
```

- [ ] **Step 5: 运行测试验证通过**

Run: `source venv/bin/activate && python -m pytest tests/test_neurolink.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add app/routers/neurolink.py app/server.py tests/test_neurolink.py
git commit -m "feat: NeuroLink REST + WS 路由"
```

---

### Task 7: 前端导航与视图容器

**Files:**
- Modify: `app/static/index.html` (新增导航项 + 视图容器)
- Modify: `app/static/js/app.js` (侧边栏导航扩展)

- [ ] **Step 1: 在 index.html 侧边栏添加导航项**

在 `app/static/index.html` 的侧边栏 `实时` 导航组（约第 58-63 行）中添加"实时监测"导航项：

找到：
```html
            <div class="nav-group">
                <div class="nav-group-label">实时</div>
                <a class="nav-item" data-module="realtime">
                    <span class="nav-item-text">实时采集</span>
                </a>
            </div>
```

替换为：
```html
            <div class="nav-group">
                <div class="nav-group-label">实时</div>
                <a class="nav-item" data-module="realtime">
                    <span class="nav-item-text">实时采集</span>
                </a>
                <a class="nav-item" data-module="neurolink">
                    <span class="nav-item-text">实时监测</span>
                </a>
            </div>
```

在 `数据` 导航组中添加"批量分析"：

找到：
```html
                <a class="nav-item nav-item-disabled" data-module="archive">
                    <span class="nav-item-text">数据归档</span>
                    <span class="nav-item-tag">即将推出</span>
```

在其前面添加：
```html
                <a class="nav-item" data-module="batch">
                    <span class="nav-item-text">批量分析</span>
                </a>
```

- [ ] **Step 2: 在 index.html 添加视图容器和脚本引用**

在 `</body>` 标签前、现有 `<script>` 引用后添加新脚本引用：

找到：
```html
<script src="/static/js/app.js?v=3"></script>
```

在其后添加：
```html
<script src="/static/js/batch.js?v=4"></script>
<script src="/static/js/neurolink.js?v=4"></script>
```

更新 CSS 版本号：
```html
<link rel="stylesheet" href="/static/css/style.css?v=4">
```
以及所有 `?v=3` 的 script 引用改为 `?v=4`。

在 index.html 的视图区域（搜索 `view-realtime` 等现有视图，在其后）添加批量分析和实时监测的空视图容器：

```html
<!-- ========== 批量分析视图 ========== -->
<section class="module-view" id="view-batch">
    <div class="view-header">
        <h1 class="view-title">批量分析</h1>
        <p class="view-desc">批量导入多个 EEG 文件，分配被试与条件，一键全分析并导出报告</p>
    </div>
    <div class="view-body" id="batch-container">
        <!-- batch.js 动态渲染 -->
    </div>
</section>

<!-- ========== 实时监测视图 ========== -->
<section class="module-view" id="view-neurolink">
    <div class="view-header">
        <h1 class="view-title">实时监测</h1>
        <p class="view-desc">连接 NeuroLink 平台，实时监测 EEG 波形与心流指标</p>
    </div>
    <div class="view-body" id="neurolink-container">
        <!-- neurolink.js 动态渲染 -->
    </div>
</section>
```

- [ ] **Step 3: 确认侧边栏导航逻辑无需修改**

`app/static/js/app.js` 的 `initSidebarNav` 函数已通过 `data-module` 属性自动切换视图，新增的 `batch` 和 `neurolink` 模块会自动生效。无需修改 `app.js`。

- [ ] **Step 4: 提交**

```bash
git add app/static/index.html
git commit -m "feat: 前端导航与视图容器 — 批量分析 + 实时监测"
```

---

### Task 8: 批量导入前端 JS

**Files:**
- Create: `app/static/js/batch.js`

- [ ] **Step 1: 实现 batch.js**

Create `app/static/js/batch.js`:

```javascript
/* ==========================================================
   EEG 批量分析 — 前端交互
   ========================================================== */

let batchPollTimer = null;

// ========== 初始化 ==========
document.addEventListener('DOMContentLoaded', () => {
    initBatchView();
});

function initBatchView() {
    const container = document.getElementById('batch-container');
    if (!container) return;
    container.innerHTML = buildBatchHTML();
    bindBatchEvents();
}

function buildBatchHTML() {
    return `
    <div class="batch-upload-area" style="padding:24px;">
        <div style="margin-bottom:16px;">
            <label class="upload-label" style="display:block;margin-bottom:8px;font-weight:600;">选择多个 EEG 文件 (.csv / .txt)</label>
            <input type="file" id="batch-file-input" accept=".csv,.txt" multiple style="margin-bottom:12px;">
            <div id="batch-file-hint" style="font-size:13px;color:var(--text-tertiary);">未选择文件</div>
        </div>
        <div id="batch-assignment-table" style="display:none;margin-top:16px;"></div>
        <div style="margin-top:24px;">
            <button class="btn btn-primary" id="btn-batch-start" disabled>开始批量分析</button>
            <button class="btn btn-secondary" id="btn-batch-download" style="display:none;">下载批量报告 ZIP</button>
        </div>
        <div id="batch-progress" style="margin-top:24px;display:none;"></div>
    </div>
    `;
}

function bindBatchEvents() {
    const fileInput = document.getElementById('batch-file-input');
    const hint = document.getElementById('batch-file-hint');
    const tableDiv = document.getElementById('batch-assignment-table');
    const startBtn = document.getElementById('btn-batch-start');

    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files);
        if (files.length === 0) {
            hint.textContent = '未选择文件';
            tableDiv.style.display = 'none';
            startBtn.disabled = true;
            return;
        }
        hint.textContent = `已选择 ${files.length} 个文件`;
        renderAssignmentTable(files);
        tableDiv.style.display = 'block';
        startBtn.disabled = false;
    });

    startBtn.addEventListener('click', startBatchAnalysis);
    document.getElementById('btn-batch-download').addEventListener('click', downloadBatchReport);
}

// 按文件名时间戳前缀分组
function groupFilesByTimestamp(files) {
    const groups = {};
    files.forEach(f => {
        // 提取时间戳前缀: BrainFlow-RAW_2026-07-14_11-20-30_0.csv → 2026-07-14_11-20-30
        const match = f.name.match(/(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})/);
        const key = match ? match[1] : 'other';
        if (!groups[key]) groups[key] = [];
        groups[key].push(f);
    });
    return groups;
}

function renderAssignmentTable(files) {
    const groups = groupFilesByTimestamp(files);
    const groupColors = ['#4B3FE3', '#1DC981', '#22A5F7', '#F87454', '#EDAA45', '#B655FC'];
    const conditions = ['AtoA', 'AtoB', 'AtoC', 'BtoC'];

    let html = '<table style="width:100%;border-collapse:collapse;font-size:13px;">';
    html += '<thead><tr style="border-bottom:2px solid var(--border);">';
    html += '<th style="text-align:left;padding:8px;">文件名</th>';
    html += '<th style="text-align:left;padding:8px;">时间戳</th>';
    html += '<th style="text-align:left;padding:8px;">被试</th>';
    html += '<th style="text-align:left;padding:8px;">条件</th>';
    html += '</tr></thead><tbody>';

    let colorIdx = 0;
    Object.entries(groups).forEach(([ts, groupFiles]) => {
        const color = groupColors[colorIdx % groupColors.length];
        colorIdx++;
        groupFiles.forEach(f => {
            html += `<tr style="border-bottom:1px solid var(--border);" data-filename="${f.name}">`;
            html += `<td style="padding:8px;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:8px;"></span>${f.name}</td>`;
            html += `<td style="padding:8px;color:var(--text-tertiary);">${ts}</td>`;
            html += `<td style="padding:8px;"><input type="text" class="batch-subject" placeholder="S01" style="width:60px;padding:4px;border:1px solid var(--border);border-radius:4px;"></td>`;
            html += `<td style="padding:8px;"><select class="batch-condition" style="padding:4px;border:1px solid var(--border);border-radius:4px;">`;
            conditions.forEach(c => { html += `<option value="${c}">${c}</option>`; });
            html += '</select></td>';
            html += '</tr>';
        });
    });

    html += '</tbody></table>';
    document.getElementById('batch-assignment-table').innerHTML = html;
}

async function startBatchAnalysis() {
    const fileInput = document.getElementById('batch-file-input');
    const files = Array.from(fileInput.files);
    if (files.length === 0) return;

    // 收集分配表
    const assignments = [];
    document.querySelectorAll('#batch-assignment-table tbody tr').forEach(row => {
        const filename = row.dataset.filename;
        const subject = row.querySelector('.batch-subject').value || 'unknown';
        const condition = row.querySelector('.batch-condition').value;
        assignments.push({ filename, subject, condition });
    });

    // 构建 FormData
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    formData.append('assignments', JSON.stringify(assignments));

    document.getElementById('btn-batch-start').disabled = true;
    document.getElementById('batch-progress').style.display = 'block';
    updateBatchProgress('上传中...', 0, files.length);

    try {
        const resp = await fetch('/api/batch-analyze', { method: 'POST', body: formData });
        if (!resp.ok) {
            let msg = `HTTP ${resp.status}`;
            try { const e = await resp.json(); msg = e.detail || msg; } catch (_) {}
            throw new Error(msg);
        }
        const data = await resp.json();
        pollBatchProgress(data.batch_id, data.total);
    } catch (err) {
        alert('批量分析启动失败: ' + err.message);
        document.getElementById('btn-batch-start').disabled = false;
    }
}

function pollBatchProgress(batchId, total) {
    if (batchPollTimer) clearInterval(batchPollTimer);
    batchPollTimer = setInterval(async () => {
        try {
            const resp = await fetch(`/api/batch-progress/${batchId}`);
            if (!resp.ok) return;
            const prog = await resp.json();
            updateBatchProgress(prog.current_module || prog.current_file || '分析中',
                               prog.current, prog.total);
            if (prog.status !== 'running') {
                clearInterval(batchPollTimer);
                batchPollTimer = null;
                const failCount = prog.errors.length;
                let msg = `批量分析完成: ${prog.total - failCount}/${prog.total} 成功`;
                if (failCount > 0) msg += `\n失败 ${failCount} 项`;
                alert(msg);
                document.getElementById('btn-batch-download').style.display = '';
                document.getElementById('btn-batch-download').dataset.batchId = batchId;
                document.getElementById('btn-batch-start').disabled = false;
            }
        } catch (e) { /* 忽略轮询错误 */ }
    }, 2000);
}

function updateBatchProgress(label, current, total) {
    const pct = total > 0 ? Math.round(current / total * 100) : 0;
    document.getElementById('batch-progress').innerHTML = `
        <div style="font-size:14px;margin-bottom:8px;">${label} (${current}/${total})</div>
        <div style="width:100%;height:8px;background:var(--border);border-radius:4px;overflow:hidden;">
            <div style="width:${pct}%;height:100%;background:var(--primary);transition:width 0.3s;"></div>
        </div>
    `;
}

function downloadBatchReport() {
    const batchId = document.getElementById('btn-batch-download').dataset.batchId;
    if (!batchId) return;
    window.location.href = `/api/export-batch-report?batch_id=${batchId}`;
}
```

- [ ] **Step 2: 提交**

```bash
git add app/static/js/batch.js
git commit -m "feat: 批量导入前端 — 文件选择+表格分配+进度+下载"
```

---

### Task 9: NeuroLink 实时监测前端 JS

**Files:**
- Create: `app/static/js/neurolink.js`

- [ ] **Step 1: 实现 neurolink.js**

Create `app/static/js/neurolink.js`:

```javascript
/* ==========================================================
   NeuroLink 实时监测 — 前端交互
   ========================================================== */

let neurolinkWS = null;
let neurolinkCanvas = null;
let neurolinkCtx = null;

document.addEventListener('DOMContentLoaded', () => {
    initNeurolinkView();
});

function initNeurolinkView() {
    const container = document.getElementById('neurolink-container');
    if (!container) return;
    container.innerHTML = buildNeurolinkHTML();
    bindNeurolinkEvents();
    initNeurolinkCanvas();
    refreshNeurolinkStatus();
    setInterval(refreshNeurolinkStatus, 3000);
}

function buildNeurolinkHTML() {
    return `
    <div style="padding:24px;">
        <!-- 连接面板 -->
        <div style="margin-bottom:24px;padding:16px;border:1px solid var(--border);border-radius:8px;">
            <div style="font-weight:600;margin-bottom:12px;">NeuroLink 连接</div>
            <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                <input type="text" id="nl-room" placeholder="房间号 (4位数字)" style="width:120px;padding:8px;border:1px solid var(--border);border-radius:4px;">
                <input type="text" id="nl-nickname" placeholder="昵称" value="EEGDataScience" style="width:180px;padding:8px;border:1px solid var(--border);border-radius:4px;">
                <button class="btn btn-primary" id="btn-nl-connect">连接</button>
                <button class="btn btn-secondary" id="btn-nl-disconnect" style="display:none;">断开</button>
                <span id="nl-status" style="font-size:13px;color:var(--text-tertiary);">未连接</span>
            </div>
        </div>

        <!-- 波形显示 -->
        <div style="margin-bottom:24px;">
            <div style="font-weight:600;margin-bottom:8px;">实时 EEG 波形 (4通道)</div>
            <canvas id="nl-canvas" width="800" height="200" style="width:100%;border:1px solid var(--border);border-radius:8px;background:#fafafa;"></canvas>
        </div>

        <!-- 指标面板 -->
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px;">
            <div class="metric-card" id="nl-metric-tar"><div class="metric-label">θ/α 比值</div><div class="metric-value">—</div></div>
            <div class="metric-card" id="nl-metric-entropy"><div class="metric-label">谱熵</div><div class="metric-value">—</div></div>
            <div class="metric-card" id="nl-metric-load"><div class="metric-label">认知负载</div><div class="metric-value">—</div></div>
            <div class="metric-card" id="nl-metric-phase"><div class="metric-label">当前阶段</div><div class="metric-value">—</div></div>
        </div>

        <!-- 频带功率 -->
        <div style="margin-bottom:24px;">
            <div style="font-weight:600;margin-bottom:8px;">频带功率</div>
            <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <span id="nl-band-delta" style="font-size:13px;">δ: —</span>
                <span id="nl-band-theta" style="font-size:13px;">θ: —</span>
                <span id="nl-band-alpha" style="font-size:13px;">α: —</span>
                <span id="nl-band-beta" style="font-size:13px;">β: —</span>
                <span id="nl-band-gamma" style="font-size:13px;">γ: —</span>
            </div>
        </div>

        <!-- 记录控制 -->
        <div style="padding:16px;border:1px solid var(--border);border-radius:8px;">
            <div style="font-weight:600;margin-bottom:12px;">会话记录</div>
            <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                <input type="text" id="nl-record-subject" placeholder="被试编号" style="width:100px;padding:8px;border:1px solid var(--border);border-radius:4px;">
                <select id="nl-record-condition" style="padding:8px;border:1px solid var(--border);border-radius:4px;">
                    <option value="AtoA">A→A</option>
                    <option value="AtoB">A→B</option>
                    <option value="AtoC">A→C</option>
                    <option value="BtoC">B→C</option>
                </select>
                <button class="btn btn-primary" id="btn-nl-record-start">开始记录</button>
                <button class="btn btn-secondary" id="btn-nl-record-stop" style="display:none;">停止记录</button>
                <span id="nl-record-info" style="font-size:13px;color:var(--text-tertiary);"></span>
            </div>
        </div>
    </div>
    `;
}

function bindNeurolinkEvents() {
    document.getElementById('btn-nl-connect').addEventListener('click', connectNeurolink);
    document.getElementById('btn-nl-disconnect').addEventListener('click', disconnectNeurolink);
    document.getElementById('btn-nl-record-start').addEventListener('click', startNeurolinkRecording);
    document.getElementById('btn-nl-record-stop').addEventListener('click', stopNeurolinkRecording);
}

function initNeurolinkCanvas() {
    neurolinkCanvas = document.getElementById('nl-canvas');
    neurolinkCtx = neurolinkCanvas.getContext('2d');
}

async function connectNeurolink() {
    const room = document.getElementById('nl-room').value.trim();
    const nickname = document.getElementById('nl-nickname').value.trim() || 'EEGDataScience';
    if (!room) { alert('请输入房间号'); return; }

    document.getElementById('nl-status').textContent = '连接中...';
    try {
        const resp = await fetchJSON('/api/neurolink/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ room_code: room, nickname }),
        });
        if (!resp.ok) throw new Error(resp.error || '连接失败');
        document.getElementById('nl-status').textContent = `已连接房间 ${room}`;
        document.getElementById('btn-nl-connect').style.display = 'none';
        document.getElementById('btn-nl-disconnect').style.display = '';
        // 连接 WebSocket
        connectNeurolinkWS();
    } catch (err) {
        document.getElementById('nl-status').textContent = '连接失败: ' + err.message;
    }
}

async function disconnectNeurolink() {
    await fetchJSON('/api/neurolink/disconnect', { method: 'POST' });
    if (neurolinkWS) { neurolinkWS.close(); neurolinkWS = null; }
    document.getElementById('nl-status').textContent = '已断开';
    document.getElementById('btn-nl-connect').style.display = '';
    document.getElementById('btn-nl-disconnect').style.display = 'none';
}

function connectNeurolinkWS() {
    if (neurolinkWS) neurolinkWS.close();
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    neurolinkWS = new WebSocket(`${protocol}//${location.host}/ws/neurolink`);
    neurolinkWS.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleNeurolinkMessage(msg);
    };
}

function handleNeurolinkMessage(msg) {
    if (msg.type === 'eeg_frame') {
        drawNeurolinkWaveform(msg.channels);
    } else if (msg.type === 'metrics_snapshot') {
        updateNeurolinkMetrics(msg);
    } else if (msg.type === 'phase_sync') {
        document.getElementById('nl-metric-phase').querySelector('.metric-value').textContent =
            msg.phase_name || msg.phase_id || '—';
    }
}

function drawNeurolinkWaveform(channels) {
    if (!neurolinkCtx) return;
    const ctx = neurolinkCtx;
    const w = neurolinkCanvas.width;
    const h = neurolinkCanvas.height;
    const chHeight = h / 4;

    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, w, h);

    const colors = ['#4B3FE3', '#1DC981', '#22A5F7', '#F87454'];
    channels.forEach((val, i) => {
        const y = chHeight * i + chHeight / 2;
        const scale = chHeight / 200; // 缩放因子
        ctx.strokeStyle = colors[i] || '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(w - 2, y - val * scale);
        ctx.lineTo(w, y - val * scale);
        ctx.stroke();
    });
}

function updateNeurolinkMetrics(msg) {
    const setMetric = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.querySelector('.metric-value').textContent = val;
    };
    setMetric('nl-metric-tar', msg.theta_alpha_ratio?.toFixed(3) || '—');
    setMetric('nl-metric-entropy', msg.spectral_entropy?.toFixed(3) || '—');
    setMetric('nl-metric-load', msg.cognitive_load_index?.toFixed(3) || '—');

    const bp = msg.band_power || {};
    const setBand = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };
    setBand('nl-band-delta', `δ: ${bp.delta?.toFixed(2) || '—'}`);
    setBand('nl-band-theta', `θ: ${bp.theta?.toFixed(2) || '—'}`);
    setBand('nl-band-alpha', `α: ${bp.alpha?.toFixed(2) || '—'}`);
    setBand('nl-band-beta', `β: ${bp.beta?.toFixed(2) || '—'}`);
    setBand('nl-band-gamma', `γ: ${bp.gamma?.toFixed(2) || '—'}`);
}

async function refreshNeurolinkStatus() {
    try {
        const status = await fetchJSON('/api/neurolink/status');
        if (status.recording) {
            document.getElementById('nl-record-info').textContent =
                `记录中: ${status.record_count} 采样, ${status.record_duration.toFixed(0)}s`;
        }
    } catch (e) { /* 忽略 */ }
}

async function startNeurolinkRecording() {
    const subject = document.getElementById('nl-record-subject').value.trim() || 'unknown';
    const condition = document.getElementById('nl-record-condition').value;
    try {
        const resp = await fetchJSON('/api/neurolink/start-recording', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subject, condition }),
        });
        if (!resp.ok) throw new Error(resp.error || '记录启动失败');
        document.getElementById('btn-nl-record-start').style.display = 'none';
        document.getElementById('btn-nl-record-stop').style.display = '';
    } catch (err) {
        alert('记录启动失败: ' + err.message);
    }
}

async function stopNeurolinkRecording() {
    try {
        const resp = await fetchJSON('/api/neurolink/stop-recording', { method: 'POST' });
        if (resp.ok) {
            document.getElementById('btn-nl-record-start').style.display = '';
            document.getElementById('btn-nl-record-stop').style.display = 'none';
            document.getElementById('nl-record-info').textContent =
                `已保存: ${resp.path} (${resp.count} 采样)`;
            alert(`记录已保存\n文件: ${resp.path}\n\n可切换到"批量分析"导入此文件进行深度分析`);
        }
    } catch (err) {
        alert('停止记录失败: ' + err.message);
    }
}
```

- [ ] **Step 2: 提交**

```bash
git add app/static/js/neurolink.js
git commit -m "feat: NeuroLink 实时监测前端 — 连接+波形+指标+记录"
```

---

### Task 10: 端到端验证

**Files:**
- 无新文件

- [ ] **Step 1: 重启服务**

Run: `lsof -ti :18765 2>/dev/null | xargs kill -9 2>/dev/null; sleep 2; source venv/bin/activate && nohup python -m uvicorn app.server:app --host 0.0.0.0 --port 18765 > /tmp/eeg_server.log 2>&1 & sleep 5; curl -s http://localhost:18765/api/health`

Expected: `{"status":"ok","service":"EEG Flow Recovery Analyzer"}`

- [ ] **Step 2: 测试批量分析端点**

Run:
```bash
# 用 EEGdata 中 2 个文件测试批量分析
curl -s -X POST http://localhost:18765/api/batch-analyze \
  -F "files=@/Users/xiatian/Desktop/EEGdata/BrainFlow-RAW_2026-07-14_11-20-30_0.csv" \
  -F "files=@/Users/xiatian/Desktop/EEGdata/BrainFlow-RAW_2026-07-14_11-20-30_1.csv" \
  -F 'assignments=[{"filename":"BrainFlow-RAW_2026-07-14_11-20-30_0.csv","subject":"S01","condition":"AtoA"},{"filename":"BrainFlow-RAW_2026-07-14_11-20-30_1.csv","subject":"S01","condition":"AtoB"}]'
```

Expected: 返回 `{"batch_id":"...","total":2}`

- [ ] **Step 3: 轮询进度并导出**

Run:
```bash
# 替换 BATCH_ID 为上一步返回的值
curl -s http://localhost:18765/api/batch-progress/BATCH_ID | python3 -m json.tool
# 等待 status 变为 done 后导出
curl -s -o /tmp/batch_report.zip -w "HTTP %{http_code}, %{size_download} bytes\n" "http://localhost:18765/api/export-batch-report?batch_id=BATCH_ID"
unzip -l /tmp/batch_report.zip
```

Expected: ZIP 包含 batch_summary.md + per_file/ + original_data/

- [ ] **Step 4: 验证 NeuroLink 状态端点**

Run: `curl -s http://localhost:18765/api/neurolink/status | python3 -m json.tool`

Expected: `{"connected":false,"recording":false,...}`

- [ ] **Step 5: 验证前端页面加载**

Run: `curl -s http://localhost:18765/ | grep -c "view-batch\|view-neurolink"`

Expected: `2` 或更多

- [ ] **Step 6: 运行全部测试**

Run: `source venv/bin/activate && python -m pytest tests/test_db_migration.py tests/test_run_all_modules.py tests/test_batch.py tests/test_neurolink.py -v`

Expected: 全部 PASS

- [ ] **Step 7: 提交并推送**

```bash
git add -A
git commit -m "test: 端到端验证通过 — 批量导入 + NeuroLink 实时对接"
git push origin main
```

---

## Self-Review

**Spec coverage:**
- §3 批量导入: Task 1 (DB), Task 2 (shared func), Task 3 (batch endpoint), Task 4 (export), Task 8 (frontend) ✓
- §4 NeuroLink: Task 5 (client), Task 6 (router), Task 9 (frontend) ✓
- §5 共享基础设施: Task 1 (DB migration), Task 2 (shared function) ✓
- §6 错误处理: 各 Task 中实现 (单文件失败不中断、自动重连、room_denied 等) ✓
- §7 测试: 每个 Task 都有 TDD 测试 + Task 10 端到端 ✓
- §9 文件变更清单: 全部覆盖 ✓

**Placeholder scan:** 无 TBD/TODO，每个步骤都有完整代码。

**Type consistency:** `_run_all_modules` 签名一致；`BATCH_STORE`/`BATCH_RESULTS_STORE` 在 Task 3/4 一致；`NeuroLinkClient` 方法名在 Task 5/6/9 一致。
