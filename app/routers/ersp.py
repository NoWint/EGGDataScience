"""
ERSP 时频进阶分析 API 路由
"""
import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.analysis.ersp import (
    run_ersp_analysis, compute_cwt, compute_ersp,
    compute_erd_ers, compute_pac, compute_itpc,
    compute_freqs_logspace, compare_ersp_conditions,
    permutation_test_ersp,
    _extract_epochs, _compute_epochs_power, _compute_ersp_from_power,
    _downsample_matrix, _downsample_axis,
)
from app.analysis import generate_sample_eeg, events_to_df, load_eeg
from pathlib import Path

router = APIRouter(prefix="/api/ersp", tags=["ersp"])

# 复用主服务的上传目录
UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"

# 各条件对应的 disruption 参数与随机种子（与 spectrum.py / erp.py 一致）
disruption_map = {"AtoA": 0.0, "AtoB": 1.0, "AtoC": 1.6, "BtoC": 1.3}
seed_map = {"AtoA": 42, "AtoB": 100, "AtoC": 200, "BtoC": 300}


class SampleErspRequest(BaseModel):
    condition: str = "AtoA"
    fs: int = 250
    event_id: str = "X0"


class CompareErspRequest(BaseModel):
    condition_a: str = "AtoA"
    condition_b: str = "AtoB"
    fs: int = 250
    event_id: str = "X0"


def _default_events_df() -> pd.DataFrame:
    """默认事件时序"""
    return pd.DataFrame([
        ('S0', 0.0), ('B0', 5.0), ('B1', 65.0),
        ('F0', 65.0), ('F1', 305.0), ('F2', 545.0),
        ('X0', 545.0), ('X1', 665.0),
        ('R0', 665.0), ('R1', 1265.0), ('Q0', 1265.0),
    ], columns=['event_id', 'timestamp'])


def _compute_ersp_with_power(data, fs, events_df, event_id, freqs):
    """计算 ERSP 并附带 per-epoch 功率 (用于置换检验)

    返回 dict 含 'ersp', 'freqs', 'times', 'epochs_power', 'n_epochs'
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    epochs, times = _extract_epochs(data, fs, events_df, event_id, 2.0, 5.0)
    n_epochs = epochs.shape[0]
    epochs_power = _compute_epochs_power(epochs, fs, freqs)
    ersp, _, _ = _compute_ersp_from_power(
        epochs_power, times, freqs,
        baseline_start=-2.0, baseline_end=-0.2, baseline_method='median',
    )
    return {
        'ersp': ersp,
        'freqs': freqs,
        'times': times,
        'epochs_power': epochs_power,
        'n_epochs': int(n_epochs),
    }


@router.get("/health")
async def ersp_health():
    """ERSP 分析模块健康检查"""
    return {"status": "ok", "module": "ersp_analysis"}


@router.post("/sample")
async def analyze_sample(req: SampleErspRequest):
    """生成模拟 EEG 数据并进行 ERSP 时频进阶分析 (含置换检验)"""
    if req.condition not in disruption_map:
        raise HTTPException(400, f"未知条件: {req.condition}，可选: {list(disruption_map.keys())}")

    disruption = disruption_map[req.condition]
    seed = seed_map[req.condition]

    data, times, events = generate_sample_eeg(
        fs=req.fs, duration_sec=25 * 60, n_channels=3,
        seed=seed, disruption=disruption,
    )
    events_df = events_to_df(events)

    result = run_ersp_analysis(data, req.fs, events_df, event_id=req.event_id)
    result['condition'] = req.condition
    result['disruption'] = disruption
    return result


@router.post("/analyze")
async def analyze_uploaded(
    eeg_file: UploadFile = File(...),
    events_file: Optional[UploadFile] = File(None),
    fs: int = Form(250),
    event_id: str = Form("X0"),
):
    """上传 EEG 与事件 CSV 并进行 ERSP 时频进阶分析 (含置换检验)"""
    if not eeg_file.filename.endswith('.csv'):
        raise HTTPException(400, "EEG 文件需为 CSV 格式")

    import shutil

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    eeg_tmp = UPLOAD_DIR / "ersp_eeg_tmp.csv"
    with open(eeg_tmp, "wb") as f:
        shutil.copyfileobj(eeg_file.file, f)

    events_tmp = None
    if events_file is not None and events_file.filename:
        if not events_file.filename.endswith('.csv'):
            raise HTTPException(400, "事件文件需为 CSV 格式")
        events_tmp = UPLOAD_DIR / "ersp_events_tmp.csv"
        with open(events_tmp, "wb") as f:
            shutil.copyfileobj(events_file.file, f)

    try:
        data, detected_fs, channels, _ = load_eeg(eeg_tmp)
        if fs == 250 and detected_fs != 250:
            fs = detected_fs

        if events_tmp is not None:
            events_df = pd.read_csv(events_tmp)
            if 'event_id' not in events_df.columns or 'timestamp' not in events_df.columns:
                raise HTTPException(400, "事件 CSV 需包含 event_id 与 timestamp 列")
        else:
            events_df = _default_events_df()

        result = run_ersp_analysis(data, fs, events_df, event_id=event_id)
        result['channels'] = channels
        result['n_samples'] = len(data)
        result['duration_sec'] = len(data) / fs
        return result
    finally:
        eeg_tmp.unlink(missing_ok=True)
        if events_tmp is not None:
            events_tmp.unlink(missing_ok=True)


@router.post("/compare")
async def compare_ersp(req: CompareErspRequest):
    """跨条件 ERSP 对比 (含置换检验统计显著性)

    生成两个条件的模拟 EEG 数据, 计算各自的 ERSP, 并做跨条件差异的置换检验
    """
    if req.condition_a not in disruption_map:
        raise HTTPException(400, f"未知条件 A: {req.condition_a}")
    if req.condition_b not in disruption_map:
        raise HTTPException(400, f"未知条件 B: {req.condition_b}")

    freqs = compute_freqs_logspace(1, 45, 50)
    results = {}

    for cond_key, cond in [('a', req.condition_a), ('b', req.condition_b)]:
        disruption = disruption_map[cond]
        seed = seed_map[cond]
        data, _, events = generate_sample_eeg(
            fs=req.fs, duration_sec=25 * 60, n_channels=3,
            seed=seed, disruption=disruption,
        )
        events_df = events_to_df(events)
        results[cond_key] = _compute_ersp_with_power(
            data, req.fs, events_df, req.event_id, freqs
        )

    # 跨条件对比 (带 per-epoch 功率, 做真正的置换检验)
    comparison = compare_ersp_conditions(
        results['a'], results['b'], n_permutations=200
    )

    # 降采样对比结果矩阵用于响应
    diff_ds = _downsample_matrix(np.array(comparison['diff']), 30, 200)
    p_ds = _downsample_matrix(np.array(comparison['p_values']), 30, 200)
    sig_ds = _downsample_matrix(
        np.array(comparison['significant_mask'], dtype=int), 30, 200
    )
    ersp_a_ds = _downsample_matrix(results['a']['ersp'], 30, 200)
    ersp_b_ds = _downsample_matrix(results['b']['ersp'], 30, 200)
    freqs_ds = _downsample_axis(freqs, 30)
    times_ds = _downsample_axis(results['a']['times'], 200)

    return {
        'condition_a': req.condition_a,
        'condition_b': req.condition_b,
        'event_id': req.event_id,
        'fs': req.fs,
        'freqs': freqs_ds.tolist(),
        'times': times_ds.tolist(),
        'ersp_a': ersp_a_ds.tolist(),
        'ersp_b': ersp_b_ds.tolist(),
        'n_epochs_a': results['a']['n_epochs'],
        'n_epochs_b': results['b']['n_epochs'],
        'comparison': {
            'diff': diff_ds.tolist(),
            'p_values': p_ds.tolist(),
            'significant_mask': sig_ds.tolist(),
            'n_permutations': comparison['n_permutations'],
        },
    }
