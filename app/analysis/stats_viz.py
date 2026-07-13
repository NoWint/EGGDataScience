"""
EEG 统计可视化与数据导出模块
提供跨被试统计、地形图数据、效应量计算、CSV 导出、汇总表生成
"""
import csv
import io
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy import stats as scipy_stats
from scipy.interpolate import Rbf


# 10-20 标准位置(8 通道 Cyton 标准布局 + 原有 3 通道)
CHANNEL_POSITIONS = {
    'Fp1': (-0.3, 0.8),    # 左前额
    'Fp2': (0.3, 0.8),     # 右前额
    'Fpz': (0.0, 0.85),    # 中前额
    # 8 通道扩展(基于 10-20 标准位置,归一化到单位圆)
    'C3': (-0.7, 0.0),     # 左中央
    'C4': (0.7, 0.0),      # 右中央
    'Pz': (0.0, -0.6),     # 中顶枕
    'O1': (-0.4, -0.8),    # 左枕
    'O2': (0.4, -0.8),     # 右枕
    'Fz': (0.0, 0.5),      # 中额
    'F3': (-0.35, 0.45),   # 左额
    'F4': (0.35, 0.45),    # 右额
    'P3': (-0.5, -0.35),   # 左顶
    'P4': (0.5, -0.35),    # 右顶
    'T3': (-0.85, -0.1),   # 左颞
    'T4': (0.85, -0.1),    # 右颞
}

# 默认统计指标
DEFAULT_METRIC_KEYS = ['recovery_time']


def cross_subject_stats(results_list, metric_keys=None):
    """跨被试统计

    参数:
        results_list: 多个被试的分析结果列表 [{condition, recovery_time, attenuation, ...}, ...]
        metric_keys: 要统计的指标键，默认 ['recovery_time']

    返回:
        {
            'n_subjects': int,
            'metrics': {
                'recovery_time': {'mean', 'std', 'sem', 'ci95', 'min', 'max', 'values'},
                ...
            }
        }
    """
    if metric_keys is None:
        metric_keys = list(DEFAULT_METRIC_KEYS)

    n = len(results_list)
    metrics_out = {}

    for key in metric_keys:
        values = []
        for r in results_list:
            if not isinstance(r, dict):
                continue
            v = r.get(key)
            if v is None:
                continue
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                continue

        if not values:
            metrics_out[key] = {
                'mean': None, 'std': None, 'sem': None,
                'ci95': [None, None], 'min': None, 'max': None,
                'values': [],
            }
            continue

        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        sem = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        # 95% 置信区间: t 分布
        if len(arr) > 1:
            t_crit = float(scipy_stats.t.ppf(0.975, df=len(arr) - 1))
            ci_low = mean - t_crit * sem
            ci_high = mean + t_crit * sem
        else:
            ci_low = ci_high = mean

        metrics_out[key] = {
            'mean': mean,
            'std': std,
            'sem': float(sem),
            'ci95': [float(ci_low), float(ci_high)],
            'min': float(arr.min()),
            'max': float(arr.max()),
            'values': values,
        }

    return {
        'n_subjects': n,
        'metrics': metrics_out,
    }


def compute_topomap_data(values, channel_names=('Fp1', 'Fp2', 'Fpz')):
    """生成地形图插值数据

    使用径向基函数 (RBF) 在 30×30 网格上插值。

    参数:
        values: 各通道的值列表 [v_fp1, v_fp2, v_fpz]
        channel_names: 通道名元组

    返回:
        {'grid_x', 'grid_y', 'grid_z', 'channels', 'values'}
    """
    if len(values) != len(channel_names):
        raise ValueError("values 与 channel_names 长度不一致")

    # 获取通道位置
    positions = []
    for name in channel_names:
        if name not in CHANNEL_POSITIONS:
            raise ValueError(f"未知通道: {name}")
        positions.append(CHANNEL_POSITIONS[name])
    positions = np.array(positions, dtype=float)

    xs = positions[:, 0]
    ys = positions[:, 1]
    zs = np.array(values, dtype=float)

    # RBF 插值 (multiquadric 在 EEG 地形图中常用)
    rbf = Rbf(xs, ys, zs, function='multiquadric', smooth=0.1)

    # 30x30 网格, 范围 [-1, 1] x [-1, 1]
    grid_axis = np.linspace(-1, 1, 30)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    grid_z = rbf(grid_x, grid_y)

    return {
        'grid_x': grid_x.tolist(),
        'grid_y': grid_y.tolist(),
        'grid_z': grid_z.tolist(),
        'channels': list(channel_names),
        'values': [float(v) for v in values],
    }


def _effect_size_label(d: float) -> str:
    """根据 Cohen's d 给出效应量标签"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    else:
        return 'large'


def compute_effect_size(group_a, group_b, metric_name='metric'):
    """计算效应量 (Cohen's d)

    返回:
        {
            'metric', 'cohen_d', 'effect_size_label',
            'mean_diff', 'pooled_std', 'n_a', 'n_b',
        }
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return {
            'metric': metric_name,
            'cohen_d': 0.0,
            'effect_size_label': 'negligible',
            'mean_diff': 0.0,
            'pooled_std': 0.0,
            'n_a': n_a,
            'n_b': n_b,
        }

    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a = float(a.var(ddof=1)) if n_a > 1 else 0.0
    var_b = float(b.var(ddof=1)) if n_b > 1 else 0.0

    # 合并标准差
    denom = max(n_a + n_b - 2, 1)
    pooled_std = float(np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / denom))
    mean_diff = mean_a - mean_b
    cohen_d = float(mean_diff / pooled_std) if pooled_std > 1e-12 else 0.0

    return {
        'metric': metric_name,
        'cohen_d': cohen_d,
        'effect_size_label': _effect_size_label(cohen_d),
        'mean_diff': float(mean_diff),
        'pooled_std': pooled_std,
        'n_a': n_a,
        'n_b': n_b,
    }


def compute_paired_stats(group_a, group_b, metric_name='metric'):
    """配对统计检验 (t 检验 + Wilcoxon)

    返回:
        {
            'metric', 't_stat', 't_p',
            'wilcoxon_stat', 'wilcoxon_p',
            'mean_diff', 'effect_size_label', 'n_pairs',
        }
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n = min(len(a), len(b))

    if n < 2:
        return {
            'metric': metric_name,
            't_stat': 0.0, 't_p': 1.0,
            'wilcoxon_stat': None, 'wilcoxon_p': None,
            'mean_diff': 0.0, 'effect_size_label': 'negligible',
            'n_pairs': n,
        }

    a_paired = a[:n]
    b_paired = b[:n]
    diff = a_paired - b_paired

    # 配对 t 检验
    try:
        t_stat, t_p = scipy_stats.ttest_rel(a_paired, b_paired)
        t_stat, t_p = float(t_stat), float(t_p)
    except Exception:
        t_stat, t_p = 0.0, 1.0

    # Wilcoxon 符号秩检验 (差值全为 0 时跳过)
    wilcoxon_stat, wilcoxon_p = None, None
    if np.any(diff != 0):
        try:
            w_stat, w_p = scipy_stats.wilcoxon(a_paired, b_paired)
            wilcoxon_stat, wilcoxon_p = float(w_stat), float(w_p)
        except Exception:
            wilcoxon_stat, wilcoxon_p = None, None

    # 配对效应量 (Cohen's d, 基于差值)
    d_std = float(diff.std(ddof=1)) if n > 1 else 0.0
    cohen_d = float(diff.mean() / d_std) if d_std > 1e-12 else 0.0

    return {
        'metric': metric_name,
        't_stat': t_stat,
        't_p': t_p,
        'wilcoxon_stat': wilcoxon_stat,
        'wilcoxon_p': wilcoxon_p,
        'mean_diff': float(diff.mean()),
        'effect_size_label': _effect_size_label(cohen_d),
        'n_pairs': n,
    }


def _normalize_results_store(results: Any) -> Dict[str, Dict]:
    """将多种输入形式统一为 {condition: result_dict} 形式

    支持:
        - results_store dict (值为 dict): 直接返回
        - 单个 result dict (含标量字段): 包装为 {'result': results}
        - result dict 列表: 转为 {索引: result}
    """
    if isinstance(results, dict):
        # 区分 results_store 与单个 result: 前者所有值均为 dict
        if results and all(isinstance(v, dict) for v in results.values()):
            return results
        else:
            return {'result': results}
    elif isinstance(results, list):
        return {str(i): r for i, r in enumerate(results) if isinstance(r, dict)}
    else:
        return {}


def prepare_csv_export(results, export_type='summary'):
    """准备 CSV 导出数据

    参数:
        results: 分析结果 dict 或 results list
        export_type: 'summary' | 'detail' | 'timeseries'

    返回: CSV 字符串
    """
    output = io.StringIO()
    writer = csv.writer(output)

    store = _normalize_results_store(results)

    if export_type == 'summary':
        table = prepare_summary_table(store)
        writer.writerow(table['headers'])
        for row in table['rows']:
            writer.writerow(row)

    elif export_type == 'detail':
        # 每个条件 × 每个指标 一行
        writer.writerow(['condition', 'indicator', 'baseline',
                         'recovery_time', 'attenuation'])
        indicators = ['theta_alpha_ratio', 'alpha_rel', 'beta_rel',
                      'gamma_rel', 'eeg_entropy', 'cog_load']
        for cond, res in store.items():
            if not isinstance(res, dict):
                continue
            baseline = res.get('baseline_means', {}) or {}
            per_feat = res.get('recovery_per_feature', {}) or {}
            att = res.get('attenuation', {}) or {}
            for ind in indicators:
                writer.writerow([
                    cond, ind,
                    baseline.get(ind, ''),
                    per_feat.get(ind, ''),
                    att.get(ind, ''),
                ])

    elif export_type == 'timeseries':
        # 时间序列: 取第一个含时序数据的条件导出
        indicators = ['theta_alpha_ratio', 'alpha_rel', 'beta_rel',
                      'gamma_rel', 'eeg_entropy', 'cog_load']
        written = False
        for cond, res in store.items():
            if not isinstance(res, dict):
                continue
            viz = res.get('viz_data', {}) or {}
            ts = viz.get('timestamp', [])
            if not ts:
                features = res.get('features', {})
                if isinstance(features, dict):
                    ts = features.get('timestamp', [])
            if not ts:
                continue

            # 收集各指标序列
            series = {}
            for ind in indicators:
                if ind in viz:
                    series[ind] = viz[ind]
                elif isinstance(res.get('features'), dict):
                    series[ind] = res['features'].get(ind, [])
                else:
                    series[ind] = []

            writer.writerow(['condition', 'timestamp'] + indicators)
            for i, t in enumerate(ts):
                row = [cond, t]
                for ind in indicators:
                    s = series.get(ind, [])
                    row.append(s[i] if i < len(s) else '')
                writer.writerow(row)
            written = True
            break  # 仅导出第一个有时序的条件, 避免列错位

        if not written:
            writer.writerow(['condition', 'timestamp', 'theta_alpha_ratio'])
            writer.writerow(['(no timeseries data)', '', ''])

    else:
        writer.writerow(['error', f'未知 export_type: {export_type}'])

    return output.getvalue()


def prepare_summary_table(results_store):
    """生成汇总表

    参数:
        results_store: {condition: result_dict, ...}

    返回:
        {
            'headers': [...],
            'rows': [[...], ...],
            'conditions': [...],
        }
    """
    headers = [
        'condition', 'recovery_time',
        'theta_alpha_ratio_att', 'alpha_rel_att', 'beta_rel_att',
        'gamma_rel_att', 'cog_load_att', 'artifact_ratio',
    ]
    rows = []
    conditions = list(results_store.keys())

    for cond in conditions:
        res = results_store[cond] or {}
        if not isinstance(res, dict):
            res = {}
        att = res.get('attenuation', {}) or {}
        rt = res.get('recovery_time')
        rows.append([
            cond,
            rt if rt is not None else '',
            att.get('theta_alpha_ratio', ''),
            att.get('alpha_rel', ''),
            att.get('beta_rel', ''),
            att.get('gamma_rel', ''),
            att.get('cog_load', ''),
            res.get('artifact_ratio', ''),
        ])

    return {
        'headers': headers,
        'rows': rows,
        'conditions': conditions,
    }


def generate_indicator_profile(results_store):
    """生成各条件的指标雷达图数据

    将各指标归一化为 0-1 的综合评分:
        - recovery_speed: 恢复速度 (越快越高)
        - flow_stability: 心流稳定性 (衰减越小越高)
        - attention_recovery: 注意力恢复 (越快越高)
        - cog_load_control: 认知负荷控制 (cog_load 升高越小越高)
        - signal_quality: 信号质量 (伪迹越少越高)

    返回:
        {
            'axes': [...],
            'conditions': [...],
            'values': {condition: [v1, v2, ...]},  # 归一化到 [0, 1]
        }
    """
    axes = ['recovery_speed', 'flow_stability', 'attention_recovery',
            'cog_load_control', 'signal_quality']
    conditions = list(results_store.keys())

    # 收集各条件各轴的原始值 (均为"越小越好"的方向)
    raw = {c: {} for c in conditions}
    for cond in conditions:
        res = results_store[cond] or {}
        if not isinstance(res, dict):
            res = {}
        att = res.get('attenuation', {}) or {}
        per_feat = res.get('recovery_per_feature', {}) or {}
        artifact_ratio = float(res.get('artifact_ratio', 0.0) or 0.0)
        rt = res.get('recovery_time')

        # 1. 恢复速度: rt 越小越快; None 视为最差 (取 600s 上限)
        raw[cond]['recovery_speed'] = float(rt) if rt is not None else 600.0

        # 2. 心流稳定性: flow 指标衰减绝对值均值 (越小越稳定)
        flow_atts = [abs(float(att.get(k, 0) or 0))
                     for k in ('theta_alpha_ratio', 'alpha_rel', 'beta_rel')]
        raw[cond]['flow_stability'] = sum(flow_atts) / max(len(flow_atts), 1)

        # 3. 注意力恢复: flow 指标的恢复时长均值 (越短越好)
        flow_recs = []
        for k in ('theta_alpha_ratio', 'alpha_rel', 'beta_rel'):
            v = per_feat.get(k)
            if v is not None:
                try:
                    flow_recs.append(float(v))
                except (TypeError, ValueError):
                    pass
        raw[cond]['attention_recovery'] = (
            sum(flow_recs) / len(flow_recs) if flow_recs else 600.0
        )

        # 4. 认知负荷控制: cog_load 衰减越小越好 (正衰减 = 升高 = 不利)
        raw[cond]['cog_load_control'] = abs(float(att.get('cog_load', 0) or 0))

        # 5. 信号质量: artifact_ratio 越小越好
        raw[cond]['signal_quality'] = artifact_ratio

    # 跨条件 min-max 归一化, 反转使"越小越好" → "越大越好"
    values = {}
    for cond in conditions:
        scores = []
        for axis in axes:
            v = raw[cond][axis]
            axis_vals = [raw[c][axis] for c in conditions]
            vmin = min(axis_vals)
            vmax = max(axis_vals)
            if vmax - vmin < 1e-12:
                norm = 1.0  # 所有条件该轴相同
            else:
                norm = 1.0 - (v - vmin) / (vmax - vmin)
            scores.append(float(max(0.0, min(1.0, norm))))
        values[cond] = scores

    return {
        'axes': axes,
        'conditions': conditions,
        'values': values,
    }


def run_stats_viz_analysis(results_store):
    """完整统计可视化分析

    返回:
        {
            'summary_table': {...},
            'indicator_profile': {...},
            'cross_subject': {...} (若有多被试数据),
            'export_ready': True,
        }
    """
    summary_table = prepare_summary_table(results_store)
    indicator_profile = generate_indicator_profile(results_store)

    # 跨被试统计: 将每个条件视为一个观测, 提供可用时的多条件汇总
    cross_subject = None
    results_list = []
    for cond, res in results_store.items():
        if not isinstance(res, dict):
            continue
        item = dict(res)
        item.setdefault('condition', cond)
        results_list.append(item)

    if len(results_list) >= 2:
        cross_subject = cross_subject_stats(
            results_list, metric_keys=['recovery_time']
        )

    return {
        'summary_table': summary_table,
        'indicator_profile': indicator_profile,
        'cross_subject': cross_subject,
        'export_ready': True,
    }
