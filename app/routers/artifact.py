"""
伪迹检测 API 路由
"""
import json
import shutil
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.analysis.artifact import (
    run_artifact_analysis, detect_by_threshold, detect_by_zscore,
    detect_by_wavelet, compute_signal_stats, fast_ica, classify_components,
    remove_artifacts, quality_score, _cap_segments,
)
from app.analysis import generate_sample_eeg, load_eeg

router = APIRouter(prefix="/api/artifact", tags=["artifact"])

# 复用主服务的上传目录
UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"

# 各条件对应的 disruption / seed 参数（与 spectrum 路由一致）
disruption_map = {"AtoA": 0.0, "AtoB": 1.0, "AtoC": 1.6, "BtoC": 1.3}
seed_map = {"AtoA": 42, "AtoB": 100, "AtoC": 200, "BtoC": 300}


class SampleArtifactRequest(BaseModel):
    condition: str = "AtoA"
    fs: int = 250


class WaveletDetectRequest(BaseModel):
    condition: str = "AtoA"
    fs: int = 250
    wavelet: str = "db4"
    level: int = 4
    threshold: float = 3.0


@router.get("/health")
async def artifact_health():
    """伪迹检测模块健康检查"""
    return {"status": "ok", "module": "artifact_detection"}


@router.post("/sample")
async def analyze_sample(req: SampleArtifactRequest):
    """生成模拟 EEG 数据并分析伪迹"""
    disruption = disruption_map.get(req.condition, 0.0)
    seed = seed_map.get(req.condition, 42)

    data, times, events = generate_sample_eeg(
        fs=req.fs, duration_sec=25 * 60, n_channels=3,
        seed=seed, disruption=disruption,
    )

    result = run_artifact_analysis(data, req.fs)
    result['condition'] = req.condition
    result['disruption'] = disruption
    return result


@router.post("/analyze")
async def analyze_uploaded(
    eeg_file: UploadFile = File(...),
    fs: int = Form(250),
):
    """上传 EEG 文件并分析伪迹"""
    if not eeg_file.filename.endswith('.csv'):
        raise HTTPException(400, "EEG 文件需为 CSV 格式")

    # 保存临时文件
    tmp_path = UPLOAD_DIR / "artifact_tmp.csv"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(eeg_file.file, f)

    try:
        data, detected_fs, channels, times = load_eeg(tmp_path)
        if fs == 250 and detected_fs != 250:
            fs = detected_fs

        result = run_artifact_analysis(data, fs)
        result['channels'] = channels
        result['n_samples'] = len(data)
        result['duration_sec'] = len(data) / fs
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/remove")
async def remove_artifacts_api(
    eeg_file: UploadFile = File(...),
    fs: int = Form(250),
    bad_indices: str = Form("[]"),
):
    """
    上传数据并返回清洗后的数据统计摘要

    参数:
        eeg_file: EEG CSV 文件
        fs: 采样率
        bad_indices: 需移除的成分索引，JSON 数组字符串，如 "[0, 2]"

    返回: 清洗前后统计摘要（不返回完整清洗数据）
    """
    if not eeg_file.filename.endswith('.csv'):
        raise HTTPException(400, "EEG 文件需为 CSV 格式")

    # 解析 bad_indices
    try:
        bad_idx_list = json.loads(bad_indices)
    except json.JSONDecodeError:
        raise HTTPException(400, "bad_indices 需为合法 JSON 数组字符串")
    if not isinstance(bad_idx_list, list):
        raise HTTPException(400, "bad_indices 需为列表")
    # 校验元素为整数
    bad_idx_list = [int(i) for i in bad_idx_list]

    # 保存临时文件
    tmp_path = UPLOAD_DIR / "artifact_remove_tmp.csv"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(eeg_file.file, f)

    try:
        data, detected_fs, channels, times = load_eeg(tmp_path)
        if fs == 250 and detected_fs != 250:
            fs = detected_fs

        # ICA 分解 + 伪迹移除
        ica = fast_ica(data)
        components = ica['components']
        mixing = ica['mixing']
        cleaned = remove_artifacts(data, components, mixing, bad_idx_list)

        # 统计摘要（不返回完整数据）
        orig_stats = compute_signal_stats(data, fs)
        clean_stats = compute_signal_stats(cleaned, fs)
        orig_thr = detect_by_threshold(data, fs)
        clean_thr = detect_by_threshold(cleaned, fs)
        orig_quality = quality_score(data, fs)
        clean_quality = quality_score(cleaned, fs)

        return {
            'bad_indices': bad_idx_list,
            'n_components': ica['n_components'],
            'channels': channels,
            'fs': fs,
            'n_samples': len(data),
            'original': {
                'signal_stats': orig_stats,
                'artifact_ratio': orig_thr['artifact_ratio'],
                'quality': orig_quality,
            },
            'cleaned': {
                'signal_stats': clean_stats,
                'artifact_ratio': clean_thr['artifact_ratio'],
                'quality': clean_quality,
                'shape': list(cleaned.shape),
            },
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/wavelet-detect")
async def wavelet_detect_sample(req: WaveletDetectRequest):
    """生成模拟 EEG 数据并用小波法检测伪迹

    参数:
        condition: 模拟数据条件 (AtoA/AtoB/AtoC/BtoC)
        fs: 采样率
        wavelet: 小波类型 ('db4' 或 'haar')
        level: 分解层数
        threshold: 稳健 z-score 阈值
    """
    if req.condition not in disruption_map:
        raise HTTPException(400, f"未知条件: {req.condition}，可选: {list(disruption_map.keys())}")

    disruption = disruption_map[req.condition]
    seed = seed_map[req.condition]

    data, times, events = generate_sample_eeg(
        fs=req.fs, duration_sec=25 * 60, n_channels=3,
        seed=seed, disruption=disruption,
    )

    result = detect_by_wavelet(
        data, req.fs, wavelet=req.wavelet,
        level=req.level, threshold=req.threshold,
    )
    segs, total, trunc = _cap_segments(result['artifact_segments'])

    return {
        'condition': req.condition,
        'fs': req.fs,
        'wavelet': req.wavelet,
        'level': result.get('level', req.level),
        'threshold': req.threshold,
        'artifact_ratio': result['artifact_ratio'],
        'artifact_segments': segs,
        'n_segments': total,
        'segments_truncated': trunc,
    }
