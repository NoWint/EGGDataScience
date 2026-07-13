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


def _default_events_df() -> pd.DataFrame:
    """默认事件时序"""
    return pd.DataFrame([
        ('S0', 0.0), ('B0', 5.0), ('B1', 65.0),
        ('F0', 65.0), ('F1', 305.0), ('F2', 545.0),
        ('X0', 545.0), ('X1', 665.0),
        ('R0', 665.0), ('R1', 1265.0), ('Q0', 1265.0),
    ], columns=['event_id', 'timestamp'])


@router.get("/health")
async def ersp_health():
    """ERSP 分析模块健康检查"""
    return {"status": "ok", "module": "ersp_analysis"}


@router.post("/sample")
async def analyze_sample(req: SampleErspRequest):
    """生成模拟 EEG 数据并进行 ERSP 时频进阶分析"""
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
    """上传 EEG 与事件 CSV 并进行 ERSP 时频进阶分析"""
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
