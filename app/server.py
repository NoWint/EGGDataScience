"""
EEGDataScience — FastAPI 服务端
跨学科任务切换对心流状态的影响及EEG恢复时间量化研究
"""
import os
import json
import math
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from app.analysis import (
    generate_sample_eeg, events_to_df, run_full_pipeline,
    load_eeg, load_eeg_full, load_events, preprocess, extract_features,
    compute_all_recovery, compute_attenuation,
    paired_t_test, repeated_measures_anova, pearson_correlation,
)
from app.routers.subjects import router as subjects_router
from app.routers.spectrum import router as spectrum_router
from app.routers.artifact import router as artifact_router
from app.routers.erp import router as erp_router
from app.routers.ersp import router as ersp_router
from app.routers.stats_viz import router as stats_viz_router
from app.routers.openbci import router as openbci_router
from app.routers.realtime import router as realtime_router, realtime_websocket_endpoint


# ========== 安全 JSON 序列化 ==========
def _to_jsonable(obj):
    """递归转换 numpy 类型为 JSON 可序列化, 并过滤 NaN/Inf 为 None"""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if not (math.isnan(v) or math.isinf(v)) else None
    elif isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj]
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, float):
        return obj if not (math.isnan(obj) or math.isinf(obj)) else None
    elif obj is None or isinstance(obj, str):
        return obj
    else:
        return str(obj)


class SafeJSONResponse(JSONResponse):
    """自定义 JSON 响应: 过滤 NaN/Inf, 保证返回有效 JSON"""
    def render(self, content) -> bytes:
        cleaned = _to_jsonable(content)
        return json.dumps(
            cleaned,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


app = FastAPI(title="EEGDataScience", version="2.0.0",
              default_response_class=SafeJSONResponse)

# 注册模块路由
app.include_router(subjects_router)
app.include_router(spectrum_router)
app.include_router(artifact_router)
app.include_router(erp_router)
app.include_router(ersp_router)
app.include_router(stats_viz_router)
app.include_router(openbci_router)
app.include_router(realtime_router)


# ========== WebSocket 端点 ==========
@app.websocket("/ws/realtime")
async def ws_realtime(websocket: WebSocket):
    """实时采集 WebSocket 端点"""
    await realtime_websocket_endpoint(websocket)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# 存储各条件的分析结果 (用于跨条件统计)
RESULTS_STORE = {}


# ========== 页面路由 ==========
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ========== API: 健康检查 ==========
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "EEG Flow Recovery Analyzer"}


# ========== API: 生成样例数据并分析 ==========
class SampleRequest(BaseModel):
    condition: str = "AtoA"  # AtoA | AtoB | AtoC | BtoC
    fs: int = 250
    seed: int = 42


@app.post("/api/sample")
async def generate_sample(req: SampleRequest):
    """生成模拟EEG数据并运行完整分析流水线"""
    # 各条件的切换破坏强度: 0=对照, 越大越严重
    # 理艺切换(A→C)破坏最大, 文理(A→B)次之, 文艺(B→C)中等
    disruption_map = {"AtoA": 0.0, "AtoB": 1.0, "AtoC": 1.6, "BtoC": 1.3}
    disruption = disruption_map.get(req.condition, 0.0)

    seed_map = {"AtoA": 42, "AtoB": 100, "AtoC": 200, "BtoC": 300}
    seed = req.seed or seed_map.get(req.condition, 42)

    data, times, events = generate_sample_eeg(
        fs=req.fs, duration_sec=25 * 60, n_channels=3, seed=seed,
        disruption=disruption,
    )

    events_df = events_to_df(events)
    result = run_full_pipeline(data, req.fs, events_df)
    result['condition'] = req.condition
    result['n_samples'] = len(data)
    result['duration_sec'] = len(data) / req.fs

    RESULTS_STORE[req.condition] = result
    return _to_jsonable(result)


# ========== API: 上传数据 ==========
# 允许上传的 EEG 文件后缀: CSV 与 OpenBCI GUI ODF 默认导出的 TXT
ALLOWED_EXTS = ('.csv', '.txt')


@app.post("/api/upload")
async def upload_data(
    eeg_file: UploadFile = File(...),
    events_file: Optional[UploadFile] = File(None),
    condition: str = Form("custom"),
):
    """上传 EEG CSV + 事件标记 CSV"""
    if not eeg_file.filename.lower().endswith(ALLOWED_EXTS):
        raise HTTPException(400, "EEG文件需为 CSV 或 TXT 格式")

    eeg_path = UPLOAD_DIR / f"eeg_{condition}.csv"
    with open(eeg_path, "wb") as f:
        shutil.copyfileobj(eeg_file.file, f)

    events_path = None
    if events_file and events_file.filename:
        events_path = UPLOAD_DIR / f"events_{condition}.csv"
        with open(events_path, "wb") as f:
            shutil.copyfileobj(events_file.file, f)

    return {
        "status": "uploaded",
        "eeg_path": str(eeg_path),
        "events_path": str(events_path) if events_path else None,
        "condition": condition,
    }


# ========== API: 分析上传的数据 ==========
# 滤波预设: 不同信号类型推荐不同通带
FILTER_PRESETS = {
    "eeg": {"hp": 1.0, "lp": 45.0, "notch": 50.0},
    "emg": {"hp": 20.0, "lp": 250.0, "notch": 50.0},
    "ecg": {"hp": 0.5, "lp": 40.0, "notch": 50.0},
}


class AnalyzeRequest(BaseModel):
    condition: str = "custom"
    lp: float = 45.0
    hp: float = 1.0
    notch: float = 50.0
    artifact_threshold: float = 100.0
    window_sec: float = 2.0
    overlap: float = 0.5
    tolerance: float = 0.05
    recovery_window: int = 30
    preprocess_config: Optional[Dict] = None  # 可选：高级预处理配置
    # 滤波预设: "eeg" | "emg" | "ecg" | "custom"
    filter_preset: str = "eeg"
    # 仅 custom 模式下生效: {"hp": float, "lp": float, "notch": float}
    filter_params: Optional[Dict] = None


@app.post("/api/analyze")
async def analyze_data(req: AnalyzeRequest):
    """分析已上传的EEG数据"""
    eeg_path = UPLOAD_DIR / f"eeg_{req.condition}.csv"
    events_path = UPLOAD_DIR / f"events_{req.condition}.csv"

    if not eeg_path.exists():
        raise HTTPException(404, f"未找到EEG数据: {req.condition}")

    # 用 load_eeg_full 获取完整数据(含 accel/markers/metadata)
    eeg_result = load_eeg_full(eeg_path)
    data, fs, channels, times = (
        eeg_result['data'], eeg_result['fs'],
        eeg_result['channels'], eeg_result['times']
    )

    # 事件文件优先;无事件文件时用 markers 自动生成 events_df
    if events_path.exists():
        events_df = pd.read_csv(events_path)
    elif eeg_result['markers']:
        events_df = pd.DataFrame(
            [(m.label, m.timestamp) for m in eeg_result['markers']],
            columns=['event_id', 'timestamp']
        )
    else:
        # 无事件文件时使用默认时序
        events_df = pd.DataFrame([
            ('S0', 0.0), ('B0', 5.0), ('B1', 65.0),
            ('F0', 65.0), ('F1', 305.0), ('F2', 545.0),
            ('X0', 545.0), ('X1', 665.0),
            ('R0', 665.0), ('R1', 1265.0), ('Q0', 1265.0),
        ], columns=['event_id', 'timestamp'])

    # 根据 filter_preset 设置滤波参数
    # - "custom": 使用 filter_params (回退到 req 字段)
    # - "eeg"/"emg"/"ecg": 使用对应预设覆盖 req 中的 hp/lp/notch
    if req.filter_preset == "custom" and req.filter_params:
        hp = req.filter_params.get("hp", req.hp)
        lp = req.filter_params.get("lp", req.lp)
        notch = req.filter_params.get("notch", req.notch)
    else:
        preset = FILTER_PRESETS.get(req.filter_preset, FILTER_PRESETS["eeg"])
        hp, lp, notch = preset["hp"], preset["lp"], preset["notch"]

    config = {
        'lp': lp, 'hp': hp, 'notch': notch,
        'artifact_threshold': req.artifact_threshold,
        'window_sec': req.window_sec, 'overlap': req.overlap,
        'tolerance': req.tolerance, 'recovery_window': req.recovery_window,
    }

    result = run_full_pipeline(data, fs, events_df, config=config,
                               preprocess_config=req.preprocess_config)
    result['condition'] = req.condition
    result['channels'] = channels
    result['n_samples'] = len(data)
    # 新增元信息
    result['metadata'] = eeg_result['metadata']
    result['has_accel'] = eeg_result['accel'] is not None
    result['has_markers'] = eeg_result['markers'] is not None and len(eeg_result['markers']) > 0

    RESULTS_STORE[req.condition] = result
    return _to_jsonable(result)


# ========== API: 跨条件统计 ==========
@app.get("/api/stats")
async def get_stats():
    """对已分析的所有条件进行跨条件统计"""
    conditions = list(RESULTS_STORE.keys())
    if len(conditions) < 2:
        return {"error": "至少需要2个条件才能进行统计比较", "conditions": conditions}

    # 收集各条件的恢复时长与衰减幅度
    recovery_times = {}
    attenuations = {}
    for cond, res in RESULTS_STORE.items():
        recovery_times[cond] = res.get('recovery_time')
        attenuations[cond] = res.get('attenuation', {})

    # 配对t检验: 对照组(AtoA) vs 各实验组
    t_test_results = {}
    control = "AtoA"
    if control in RESULTS_STORE:
        control_rt = [RESULTS_STORE[control].get('recovery_time') or 600.0]
        for cond in conditions:
            if cond == control:
                continue
            exp_rt = [RESULTS_STORE[cond].get('recovery_time') or 600.0]
            t_test_results[f"{control}_vs_{cond}"] = paired_t_test(control_rt, exp_rt)

    # 重复测量ANOVA: 三类跨学科切换
    cross_conds = [c for c in conditions if c != control]
    anova_results = {}
    if len(cross_conds) >= 2:
        for indicator in ['recovery_time']:
            groups = []
            for c in cross_conds:
                rt = RESULTS_STORE[c].get('recovery_time')
                groups.append([rt or 600.0])
            if len(groups) >= 2:
                anova_results[indicator] = repeated_measures_anova(groups)

    return {
        "conditions": conditions,
        "recovery_times": recovery_times,
        "attenuations": attenuations,
        "paired_t_tests": t_test_results,
        "anova": anova_results,
    }


# ========== API: 获取已存结果 ==========
@app.get("/api/results/{condition}")
async def get_result(condition: str):
    if condition not in RESULTS_STORE:
        raise HTTPException(404, f"未找到条件: {condition}")
    return _to_jsonable(RESULTS_STORE[condition])


@app.get("/api/results")
async def list_results():
    return {"conditions": list(RESULTS_STORE.keys())}


# ========== API: 一键生成分析报告 ==========
CONDITION_INFO = {
    "AtoA": {"label": "A→A 同学科连续 (对照组)", "desc": "单一数理逻辑任务持续进行，无思维范式切换，作为心流稳态基准。"},
    "AtoB": {"label": "A→B 文理切换", "desc": "数理逻辑任务切换至语言人文任务，从理性分析转为语义理解与情感感知。"},
    "AtoC": {"label": "A→C 理艺切换", "desc": "数理逻辑任务切换至艺术创想任务，从理性运算转为直觉思维与形象创作。"},
    "BtoC": {"label": "B→C 文艺切换", "desc": "语言人文任务切换至艺术创想任务，从语义表达转为自由发散创作。"},
}


def _interpret_recovery(rt):
    """根据恢复时长生成解读文本"""
    if rt is None:
        return "在观测窗口内未完全恢复至稳态水平，提示心流状态受到严重且持久的破坏。"
    if rt < 60:
        return f"恢复时长 {rt:.0f}s，心流状态在切换后迅速回归，破坏程度较轻。"
    if rt < 180:
        return f"恢复时长 {rt:.0f}s，心流状态在切换后经历中等时长恢复，存在明显认知损耗。"
    if rt < 360:
        return f"恢复时长 {rt:.0f}s，心流状态恢复缓慢，切换造成显著且持久的注意力破坏。"
    return f"恢复时长 {rt:.0f}s，心流状态恢复极慢，切换对沉浸式专注造成严重破坏。"


def _interpret_attenuation(att, indicator_type):
    """根据衰减幅度生成解读"""
    avg = sum(att.values()) / max(len(att), 1)
    abs_avg = abs(avg)
    if indicator_type == 'flow':
        if avg >= 0:
            # 心流指标跌落（Alpha 下降等）
            if abs_avg > 40:
                return f"心流核心指标平均跌落 {abs_avg:.1f}%，稳态特征遭到显著破坏。"
            if abs_avg > 20:
                return f"心流核心指标平均跌落 {abs_avg:.1f}%，存在中等程度的状态衰减。"
            return f"心流核心指标平均跌落 {abs_avg:.1f}%，状态保持相对稳定。"
        else:
            # 心流指标升高（Beta 升高、Theta/Alpha 比值上升 = 偏离稳态最优区间）
            if abs_avg > 40:
                return f"心流核心指标平均偏离稳态 {abs_avg:.1f}%，节律特征显著偏离心流最优区间。"
            if abs_avg > 20:
                return f"心流核心指标平均偏离稳态 {abs_avg:.1f}%，存在可察觉的节律偏移。"
            return f"心流核心指标平均偏离稳态 {abs_avg:.1f}%，节律特征基本维持。"
    else:
        if abs_avg > 40:
            return f"认知损耗指标平均升高 {abs_avg:.1f}%，认知负荷显著增加。"
        if abs_avg > 20:
            return f"认知损耗指标平均升高 {abs_avg:.1f}%，存在可察觉的认知负荷上升。"
        return f"认知损耗指标平均升高 {abs_avg:.1f}%，认知负荷变化有限。"


def _generate_conclusions(condition, result, stats=None):
    """生成结论与建议"""
    rt = result.get('recovery_time')
    att = result.get('attenuation', {})
    conclusions = []

    flow_att = {k: att.get(k, 0) for k in ['theta_alpha_ratio', 'alpha_rel', 'beta_rel']}
    loss_att = {k: att.get(k, 0) for k in ['gamma_rel', 'eeg_entropy', 'cog_load']}

    # 结论1: 恢复特征
    conclusions.append(_interpret_recovery(rt))

    # 结论2: 心流衰减
    conclusions.append(_interpret_attenuation(flow_att, 'flow'))

    # 结论3: 认知损耗
    conclusions.append(_interpret_attenuation(loss_att, 'loss'))

    # 结论4: 跨条件对比 (如有)
    if stats and len(stats.get('recovery_times', {})) > 1:
        rts = {k: (v if v else 600) for k, v in stats['recovery_times'].items()}
        if condition in rts and 'AtoA' in rts:
            ctrl_rt = rts['AtoA']
            cond_rt = rts.get(condition, 600)
            diff = cond_rt - ctrl_rt
            if diff > 60:
                conclusions.append(f"相较对照组(A→A)，本条件恢复时间延长 {diff:.0f}s，跨学科切换显著增加了心流恢复成本。")
            elif diff > 0:
                conclusions.append(f"相较对照组(A→A)，本条件恢复时间略增 {diff:.0f}s，切换带来一定恢复延迟。")
            else:
                conclusions.append("本条件恢复时间与对照组接近，切换未造成额外恢复负担。")

    # 建议
    suggestions = []
    if rt and rt > 180:
        suggestions.append("建议在跨学科任务切换间预留至少 3 分钟缓冲，避免心流状态未恢复即进入新任务。")
    if any(abs(v) > 40 for v in flow_att.values()):
        suggestions.append("心流核心指标衰减显著，建议通过番茄工作法等节奏管理减少频繁切换。")
    if any(abs(v) > 40 for v in loss_att.values()):
        suggestions.append("认知负荷升高明显，建议切换后进行简短的正念呼吸或放松练习以加速认知重置。")
    if not suggestions:
        suggestions.append("当前条件下心流状态保持稳定，可维持现有任务节奏。")

    return conclusions, suggestions


@app.get("/api/report")
@app.get("/api/report/{condition}")
async def generate_report(condition: Optional[str] = None):
    """生成结构化分析报告"""
    from datetime import datetime

    conditions = list(RESULTS_STORE.keys())
    if not conditions:
        raise HTTPException(400, "尚未有分析结果，请先运行至少一个条件的分析")

    # 如果未指定条件，生成综合报告（取最后一个分析的或全部）
    target_conds = [condition] if condition and condition in RESULTS_STORE else conditions

    stats = None
    if len(conditions) >= 2:
        try:
            stats = await get_stats()
        except Exception:
            stats = None

    report_sections = []

    for cond in target_conds:
        result = RESULTS_STORE[cond]
        info = CONDITION_INFO.get(cond, {"label": cond, "desc": "自定义实验条件"})
        rt = result.get('recovery_time')
        att = result.get('attenuation', {})
        per_feat = result.get('recovery_per_feature', {})
        baseline = result.get('baseline_means', {})
        evt = result.get('event_times', {})
        cfg = result.get('config', {})

        conclusions, suggestions = _generate_conclusions(cond, result, stats)

        # 各指标详情
        indicator_details = []
        indicator_names = {
            'theta_alpha_ratio': ('Theta/Alpha 比值', '心流稳态核心特征'),
            'alpha_rel': ('Alpha 能量', '沉浸式放松专注'),
            'beta_rel': ('Beta 能量', '主动专注投入'),
            'gamma_rel': ('Gamma 能量', '认知负荷与信息整合'),
            'eeg_entropy': ('脑电谱熵', '脑电信号复杂度'),
            'cog_load': ('认知负载指数', '(θ+β)/α 综合认知负荷'),
        }
        for key, (name, desc) in indicator_names.items():
            indicator_details.append({
                'name': name,
                'desc': desc,
                'baseline': baseline.get(key, 0),
                'recovery_time': per_feat.get(key),
                'attenuation': att.get(key, 0),
            })

        section = {
            'condition': cond,
            'condition_label': info['label'],
            'condition_desc': info['desc'],
            'data_summary': {
                'duration_min': result.get('duration_sec', 0) / 60 if result.get('duration_sec') else 25,
                'n_samples': result.get('n_samples', 0),
                'artifact_ratio': result.get('artifact_ratio', 0),
                'channels': result.get('channels', ['Fp1', 'Fp2', 'Fpz']),
            },
            'key_findings': {
                'recovery_time': rt,
                'recovery_interpretation': _interpret_recovery(rt),
                'event_times': evt,
                'config': cfg,
            },
            'indicator_details': indicator_details,
            'attenuation': att,
            'conclusions': conclusions,
            'suggestions': suggestions,
        }
        report_sections.append(section)

    # 统计摘要 (多条件时)
    stats_summary = None
    if stats and len(conditions) >= 2:
        stats_summary = {
            'conditions': stats.get('conditions', []),
            'recovery_times': stats.get('recovery_times', {}),
            'paired_t_tests': stats.get('paired_t_tests', {}),
            'anova': stats.get('anova', {}),
        }

    return _to_jsonable({
        'title': 'EEG 心流恢复分析报告',
        'subtitle': '跨学科任务切换对心流状态的影响及EEG恢复时间量化研究',
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sections': report_sections,
        'stats_summary': stats_summary,
        'analyzed_conditions': conditions,
    })


# ========== 工具函数 ==========
# _to_jsonable 已在文件顶部定义, 供 SafeJSONResponse 使用


# ========== 静态文件 ==========
if STATIC_DIR.exists():
    from starlette.middleware.base import BaseHTTPMiddleware

    class NoCacheStaticMiddleware(BaseHTTPMiddleware):
        """禁止浏览器缓存静态文件,确保每次加载最新版本"""
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

    app.add_middleware(NoCacheStaticMiddleware)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  EEGDataScience — 心流恢复分析")
    print("  跨学科任务切换 EEG 恢复时间量化研究")
    print("=" * 60)
    print(f"  服务地址: http://localhost:18765")
    print(f"  静态目录: {STATIC_DIR}")
    print(f"  上传目录: {UPLOAD_DIR}")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=18765)
