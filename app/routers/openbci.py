"""
OpenBCI CSV 导入 API
提供文件格式检测、元信息读取、数据转换端点
"""
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.analysis.openbci_import import (
    load_openbci,
    openbci_info,
    _detect_openbci,
    _detect_brainflow_csv,
    load_brainflow_csv,
)

router = APIRouter(prefix="/api/openbci", tags=["openbci"])

UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/detect")
async def detect_format(file: UploadFile = File(...)):
    """检测上传文件是否为 OpenBCI 格式"""
    tmp = UPLOAD_DIR / f"_tmp_{file.filename}"
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)
        is_openbci = _detect_openbci(tmp)
        return {
            "filename": file.filename,
            "is_openbci": is_openbci,
            "format": "openbci" if is_openbci else "unknown",
        }
    finally:
        if tmp.exists():
            tmp.unlink()


@router.post("/info")
async def get_info(file: UploadFile = File(...)):
    """读取 OpenBCI 文件元信息（板卡/通道/采样率/时长）"""
    if not file.filename:
        raise HTTPException(400, "未提供文件")

    tmp = UPLOAD_DIR / f"_tmp_{file.filename}"
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if not _detect_openbci(tmp):
            raise HTTPException(400, "文件不是 OpenBCI 导出格式（缺少 %OpenBCI 头）")

        info = openbci_info(tmp)
        info["filename"] = file.filename
        return info
    finally:
        if tmp.exists():
            tmp.unlink()


@router.post("/convert")
async def convert_file(
    file: UploadFile = File(...),
):
    """上传 OpenBCI CSV → 解析并返回预览数据（前 500 点）"""
    if not file.filename:
        raise HTTPException(400, "未提供文件")

    tmp = UPLOAD_DIR / f"_tmp_{file.filename}"
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if not _detect_openbci(tmp):
            raise HTTPException(400, "文件不是 OpenBCI 导出格式")

        result = load_openbci(tmp)
        data, fs, channels, times = result['data'], result['fs'], result['channels'], result['times']
        info = openbci_info(tmp)

        # 预览前 500 点
        preview_n = min(500, len(data))
        preview = {
            "times": times[:preview_n].tolist(),
            "channels": {ch: data[:preview_n, i].tolist()
                         for i, ch in enumerate(channels)},
        }

        return {
            "filename": file.filename,
            "board": info["board"],
            "sample_rate": fs,
            "n_samples": len(data),
            "n_channels": len(channels),
            "channels": channels,
            "duration_sec": round(len(data) / fs, 1),
            "has_accelerometer": info["has_accelerometer"],
            "has_analog": info["has_analog"],
            "preview": preview,
        }
    finally:
        if tmp.exists():
            tmp.unlink()


@router.post("/save")
async def save_and_analyze(
    file: UploadFile = File(...),
    condition: str = Form("openbci"),
):
    """上传 OpenBCI CSV → 保存为标准格式 → 返回分析就绪路径"""
    if not file.filename:
        raise HTTPException(400, "未提供文件")

    tmp = UPLOAD_DIR / f"_tmp_{file.filename}"
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if not _detect_openbci(tmp):
            raise HTTPException(400, "文件不是 OpenBCI 导出格式")

        result = load_openbci(tmp)
        data, fs, channels, times = result['data'], result['fs'], result['channels'], result['times']

        # 存为标准 CSV
        import pandas as pd
        df_out = pd.DataFrame(data, columns=channels)
        df_out.insert(0, "time", times)

        out_path = UPLOAD_DIR / f"openbci_{condition}_{file.filename.replace('.txt', '.csv')}"
        df_out.to_csv(out_path, index=False)

        return {
            "status": "saved",
            "filename": out_path.name,
            "path": str(out_path),
            "n_samples": len(data),
            "n_channels": len(channels),
            "channels": channels,
            "sample_rate": fs,
            "duration_sec": round(len(data) / fs, 1),
            "condition": condition,
        }
    finally:
        if tmp.exists():
            tmp.unlink()


@router.post("/detect-any")
async def detect_any_format(file: UploadFile = File(...)):
    """检测上传文件格式(OpenBCI ODF / BrainFlow CSV / 未知)"""
    tmp = UPLOAD_DIR / f"_tmp_{file.filename}"
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if _detect_openbci(tmp):
            fmt = "openbci_odf"
        elif _detect_brainflow_csv(tmp):
            fmt = "brainflow_csv"
        else:
            fmt = "unknown"

        return {
            "filename": file.filename,
            "format": fmt,
        }
    finally:
        if tmp.exists():
            tmp.unlink()
