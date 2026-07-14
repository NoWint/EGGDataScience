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

router = APIRouter(prefix="/api", tags=["batch"])

# 批量分析状态存储
BATCH_STORE: Dict[str, Dict[str, Any]] = {}
BATCH_RESULTS_STORE: Dict[str, Dict[str, Any]] = {}


def _get_upload_dir() -> Path:
    """延迟获取 UPLOAD_DIR, 避免 server.py 模块加载循环"""
    import sys
    main_mod = sys.modules.get('__main__')
    if main_mod and hasattr(main_mod, 'UPLOAD_DIR'):
        return main_mod.UPLOAD_DIR
    from app.server import UPLOAD_DIR
    return UPLOAD_DIR


def _get_run_all_modules():
    """延迟获取 _run_all_modules, 避免 server.py 模块加载循环"""
    import sys
    main_mod = sys.modules.get('__main__')
    if main_mod and hasattr(main_mod, '_run_all_modules'):
        return main_mod._run_all_modules
    from app.server import _run_all_modules
    return _run_all_modules


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
    run_all_modules = _get_run_all_modules()

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
            mod_results = run_all_modules(data, fs, events_df)
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

    upload_dir = _get_upload_dir()
    batch_dir = upload_dir / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_subdir = batch_dir / batch_id
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
    # 延迟导入以避免循环导入
    from app.server import _to_jsonable

    if batch_id not in BATCH_RESULTS_STORE:
        available = list(BATCH_RESULTS_STORE.keys())
        raise HTTPException(404, f"未找到批次: {batch_id}，可用: {available}")

    results = BATCH_RESULTS_STORE[batch_id]

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
            # 尝试生成单文件报告 MD (如果 _build_full_report_md 存在)
            try:
                from app.server import _build_full_report_md
                md = _build_full_report_md(res)
                zf.writestr(f'per_file/{key}_report.md', md)
            except (ImportError, Exception):
                pass  # 函数不存在或生成失败时跳过

        # 原始数据
        batch_dir = _get_upload_dir() / "batch" / batch_id
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
