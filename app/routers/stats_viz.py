"""
统计可视化与数据导出 API 路由
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import numpy as np

from app.analysis.stats_viz import (
    run_stats_viz_analysis, prepare_summary_table, generate_indicator_profile,
    compute_topomap_data, cross_subject_stats, compute_effect_size,
    compute_paired_stats, prepare_csv_export,
)
from app.analysis.stats_rigor import (
    run_rigorous_stats, permutation_test, benjamini_hochberg,
    bonferroni_correction, bootstrap_ci, cross_condition_stats,
)

router = APIRouter(prefix="/api/stats-viz", tags=["stats-viz"])


def _get_results_store() -> Dict[str, Dict]:
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


# ========== 请求模型 ==========

class TopomapRequest(BaseModel):
    values: List[float]
    channel_names: Optional[List[str]] = None


class CrossSubjectRequest(BaseModel):
    results: List[Dict[str, Any]]
    metric_keys: Optional[List[str]] = None


class EffectSizeRequest(BaseModel):
    group_a: List[float]
    group_b: List[float]
    metric: str = 'recovery_time'


class ExportRequest(BaseModel):
    results: Any  # dict (单结果 / results_store) 或 list
    export_type: str = 'summary'


# ========== 统计严谨性请求模型 ==========

class PairedTestRequest(BaseModel):
    group_a: List[float]
    group_b: List[float]
    metric: str = 'recovery_time'


class PermutationRequest(BaseModel):
    group_a: List[float]
    group_b: List[float]
    n_permutations: int = 1000
    alternative: str = 'two-sided'


class FdrRequest(BaseModel):
    p_values: List[float]


class BootstrapRequest(BaseModel):
    data: List[float]
    statistic: str = 'mean'
    n_bootstrap: int = 1000


# ========== 健康检查 ==========

@router.get("/health")
async def stats_viz_health():
    """统计可视化模块健康检查"""
    return {"status": "ok", "module": "stats_visualization"}


# ========== 汇总表 ==========

@router.get("/summary")
async def get_summary_table():
    """从 RESULTS_STORE 生成汇总表"""
    store = _get_results_store()
    if not store:
        raise HTTPException(400, "尚未有分析结果，请先运行至少一个条件的分析")
    return prepare_summary_table(store)


# ========== 雷达图数据 ==========

@router.get("/indicator-profile")
async def get_indicator_profile():
    """生成各条件的指标雷达图数据"""
    store = _get_results_store()
    if not store:
        raise HTTPException(400, "尚未有分析结果，请先运行至少一个条件的分析")
    return generate_indicator_profile(store)


# ========== 地形图数据 ==========

@router.post("/topomap")
async def topomap(req: TopomapRequest):
    """生成 3 通道地形图插值数据"""
    if not req.values:
        raise HTTPException(400, "values 不能为空")
    channel_names = tuple(req.channel_names) if req.channel_names else ('Fp1', 'Fp2', 'Fpz')
    if len(channel_names) != len(req.values):
        raise HTTPException(400, "values 与 channel_names 长度不一致")
    try:
        return compute_topomap_data(req.values, channel_names)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ========== 跨被试统计 ==========

@router.post("/cross-subject")
async def cross_subject(req: CrossSubjectRequest):
    """跨被试统计"""
    if not req.results:
        raise HTTPException(400, "results 不能为空")
    return cross_subject_stats(req.results, metric_keys=req.metric_keys)


# ========== 效应量与配对统计 ==========

@router.post("/effect-size")
async def effect_size(req: EffectSizeRequest):
    """效应量计算 + 配对统计检验"""
    if len(req.group_a) < 2 or len(req.group_b) < 2:
        raise HTTPException(400, "每组至少需要 2 个观测值")
    es = compute_effect_size(req.group_a, req.group_b, metric_name=req.metric)
    ps = compute_paired_stats(req.group_a, req.group_b, metric_name=req.metric)
    return {
        'effect_size': es,
        'paired_stats': ps,
    }


# ========== CSV 导出 ==========

@router.post("/export")
async def export_csv(req: ExportRequest):
    """导出 CSV (summary / detail / timeseries)"""
    if req.export_type not in ('summary', 'detail', 'timeseries'):
        raise HTTPException(400, "export_type 必须为 summary/detail/timeseries")
    csv_str = prepare_csv_export(req.results, export_type=req.export_type)
    date_str = datetime.now().strftime('%Y%m%d')
    return {
        'csv': csv_str,
        'filename': f"eeg_{req.export_type}_{date_str}.csv",
    }


# ========== 统计严谨性端点 ==========

@router.post("/paired-test")
async def paired_test(req: PairedTestRequest):
    """配对 t 检验 + 置换检验 + 效应量 (完整统计流水线)"""
    if len(req.group_a) < 2 or len(req.group_b) < 2:
        raise HTTPException(400, "每组至少需要 2 个观测值")
    return run_rigorous_stats(req.group_a, req.group_b, metric_name=req.metric)


@router.post("/permutation")
async def permutation(req: PermutationRequest):
    """置换检验（非参数）"""
    if len(req.group_a) < 1 or len(req.group_b) < 1:
        raise HTTPException(400, "每组至少需要 1 个观测值")
    if req.n_permutations < 1:
        raise HTTPException(400, "n_permutations 必须 >= 1")
    return permutation_test(
        req.group_a, req.group_b,
        n_permutations=req.n_permutations,
        alternative=req.alternative,
    )


@router.post("/fdr")
async def fdr(req: FdrRequest):
    """Benjamini-Hochberg FDR 多重比较校正"""
    if not req.p_values:
        raise HTTPException(400, "p_values 不能为空")
    return benjamini_hochberg(req.p_values)


@router.get("/cross-condition")
async def cross_condition():
    """跨条件统计比较 (从 RESULTS_STORE 获取数据)"""
    store = _get_results_store()
    if not store:
        raise HTTPException(400, "尚未有分析结果，请先运行至少一个条件的分析")
    return cross_condition_stats(store)


@router.post("/bootstrap")
async def bootstrap(req: BootstrapRequest):
    """Bootstrap 置信区间"""
    if len(req.data) < 2:
        raise HTTPException(400, "data 至少需要 2 个观测值")
    if req.n_bootstrap < 1:
        raise HTTPException(400, "n_bootstrap 必须 >= 1")
    stat_map = {'mean': np.mean, 'median': np.median, 'std': np.std}
    stat_func = stat_map.get(req.statistic, np.mean)
    return bootstrap_ci(req.data, statistic=stat_func, n_bootstrap=req.n_bootstrap)
