"""
频谱分析 API 路由
"""
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.analysis.spectrum import (
    run_spectrum_analysis, compute_psd, compute_band_powers,
    compute_spectrogram, compare_conditions_psd, compute_aperiodic_signal,
)
from app.analysis import generate_sample_eeg, load_eeg
from pathlib import Path

router = APIRouter(prefix="/api/spectrum", tags=["spectrum"])

# 复用主服务的上传目录
UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"


def _get_results_store() -> dict:
    """获取主服务的分析结果存储
    python -m app.server 启动时 __main__ 与 app.server 是不同模块，
    需优先从 __main__ 获取（实际运行并修改 RESULTS_STORE 的模块）
    """
    import sys
    main_mod = sys.modules.get('__main__')
    if main_mod and hasattr(main_mod, 'RESULTS_STORE'):
        return main_mod.RESULTS_STORE
    from app.server import RESULTS_STORE
    return RESULTS_STORE


class SampleSpectrumRequest(BaseModel):
    condition: str = "AtoA"
    fs: int = 250
    nperseg: int = 1024
    overlap: float = 0.5


@router.get("/health")
async def spectrum_health():
    """频谱分析模块健康检查"""
    return {"status": "ok", "module": "spectrum_analysis"}


@router.post("/sample")
async def analyze_sample(req: SampleSpectrumRequest):
    """生成模拟 EEG 数据并进行频谱分析"""
    # 各条件对应的 disruption 参数
    disruption_map = {"AtoA": 0.0, "AtoB": 1.0, "AtoC": 1.6, "BtoC": 1.3}
    seed_map = {"AtoA": 42, "AtoB": 100, "AtoC": 200, "BtoC": 300}

    disruption = disruption_map.get(req.condition, 0.0)
    seed = seed_map.get(req.condition, 42)

    data, times, events = generate_sample_eeg(
        fs=req.fs, duration_sec=25 * 60, n_channels=3,
        seed=seed, disruption=disruption,
    )

    result = run_spectrum_analysis(data, req.fs, req.nperseg, req.overlap)
    result['condition'] = req.condition
    result['disruption'] = disruption

    # 存储到 RESULTS_STORE 供后续 /aperiodic 查询
    store = _get_results_store()
    store[req.condition] = result

    return result


@router.post("/analyze")
async def analyze_uploaded(
    eeg_file: UploadFile = File(...),
    fs: int = Form(250),
    nperseg: int = Form(1024),
    overlap: float = Form(0.5),
):
    """上传 EEG 文件并进行频谱分析"""
    if not eeg_file.filename.endswith('.csv'):
        raise HTTPException(400, "EEG 文件需为 CSV 格式")

    # 保存临时文件
    import tempfile
    import shutil

    tmp_path = UPLOAD_DIR / f"spectrum_tmp.csv"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(eeg_file.file, f)

    try:
        data, detected_fs, channels, times = load_eeg(tmp_path)
        if fs == 250 and detected_fs != 250:
            fs = detected_fs

        result = run_spectrum_analysis(data, fs, nperseg, overlap)
        result['channels'] = channels
        result['n_samples'] = len(data)
        result['duration_sec'] = len(data) / fs
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/aperiodic/{condition}")
async def get_aperiodic(condition: str):
    """
    获取指定条件的 1/f 斜率分析（从 RESULTS_STORE 获取数据）

    返回:
        {'slope': float, 'intercept': float,
         'fit_freqs': [...], 'fit_line': [...],
         'r_squared': float, 'condition': str}
    """
    store = _get_results_store()
    if condition not in store:
        available = list(store.keys())
        raise HTTPException(404, f"未找到条件: {condition}，可用: {available}")

    result = store[condition]

    # 优先使用已计算的 aperiodic_signal
    if 'aperiodic_signal' in result:
        ap = dict(result['aperiodic_signal'])
        ap['condition'] = condition
        return ap

    # 否则从 psd 数据实时计算
    psd_data = result.get('psd')
    if not psd_data:
        raise HTTPException(400, f"条件 {condition} 无 PSD 数据，无法计算 1/f 斜率")

    freqs = np.array(psd_data.get('freqs', []))
    psd = np.array(psd_data.get('psd', []))
    if len(freqs) == 0 or len(psd) == 0:
        raise HTTPException(400, f"条件 {condition} PSD 数据不完整")

    aperiodic = compute_aperiodic_signal(psd, freqs)
    aperiodic['condition'] = condition
    return aperiodic


@router.post("/compare")
async def compare_conditions(conditions: list[str]):
    """
    对比多个条件的频段能量
    接收条件列表，自动生成各条件的模拟数据并对比
    """
    if len(conditions) < 2:
        raise HTTPException(400, "至少需要 2 个条件进行对比")

    disruption_map = {"AtoA": 0.0, "AtoB": 1.0, "AtoC": 1.6, "BtoC": 1.3}
    seed_map = {"AtoA": 42, "AtoB": 100, "AtoC": 200, "BtoC": 300}

    results = {}
    for cond in conditions:
        disruption = disruption_map.get(cond, 0.0)
        seed = seed_map.get(cond, 42)

        data, _, _ = generate_sample_eeg(
            fs=250, duration_sec=25 * 60, n_channels=3,
            seed=seed, disruption=disruption,
        )
        band_powers = compute_band_powers(data, 250)
        results[cond] = {'band_powers': band_powers}

    comparison = compare_conditions_psd(results)
    comparison['raw_results'] = results
    return comparison
