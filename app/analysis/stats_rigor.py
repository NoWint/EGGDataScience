"""
EEG 统计严谨性分析模块
提供配对t检验、置换检验、FDR多重比较校正、效应量计算、置信区间
"""
import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple, Optional, Union, Callable


# ========== 配对 t 检验 ==========

def paired_t_test(group_a, group_b, alternative='two-sided'):
    """配对 t 检验

    参数:
        group_a: A 组数据 (list 或 array)
        group_b: B 组数据 (list 或 array)
        alternative: 'two-sided' | 'less' | 'greater'

    返回:
        {'t_stat', 'p_value', 'df', 'mean_diff', 'std_diff', 'ci95', 'effect_size_d'}
        CI95 用 t 分布计算
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n = min(len(a), len(b))

    if n < 2:
        return {
            't_stat': 0.0,
            'p_value': 1.0,
            'df': max(n - 1, 0),
            'mean_diff': 0.0,
            'std_diff': 0.0,
            'ci95': [0.0, 0.0],
            'effect_size_d': 0.0,
        }

    a = a[:n]
    b = b[:n]
    diff = a - b
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    sem_diff = std_diff / np.sqrt(n)
    df = n - 1

    t_stat, p_value = scipy_stats.ttest_rel(a, b, alternative=alternative)
    t_stat = float(t_stat)
    p_value = float(p_value)

    # CI95 (t 分布)
    t_crit = float(scipy_stats.t.ppf(0.975, df=df))
    ci_lower = float(mean_diff - t_crit * sem_diff)
    ci_upper = float(mean_diff + t_crit * sem_diff)

    # 配对 Cohen's d
    effect_size_d = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'df': int(df),
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'ci95': [ci_lower, ci_upper],
        'effect_size_d': effect_size_d,
    }


# ========== 置换检验 ==========

def permutation_test(group_a, group_b, n_permutations=1000, alternative='two-sided'):
    """置换检验（非参数）

    流程: 将两组数据合并打乱，随机分组计算差值，重复 n_permutations 次得到零分布

    参数:
        group_a: A 组数据
        group_b: B 组数据
        n_permutations: 置换次数 (默认 1000)
        alternative: 'two-sided' | 'less' | 'greater'

    返回:
        {'p_value', 'observed_diff', 'null_distribution', 'significant', 'n_permutations'}
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n_a = len(a)
    n_b = len(b)

    if n_a == 0 or n_b == 0:
        return {
            'p_value': 1.0,
            'observed_diff': 0.0,
            'null_distribution': [],
            'significant': False,
            'n_permutations': int(n_permutations),
        }

    observed_diff = float(a.mean() - b.mean())
    combined = np.concatenate([a, b])

    null_distribution = []
    for _ in range(n_permutations):
        perm = np.random.permutation(combined)
        perm_a = perm[:n_a]
        perm_b = perm[n_a:]
        null_diff = float(perm_a.mean() - perm_b.mean())
        null_distribution.append(null_diff)

    null_arr = np.array(null_distribution)

    # 计算 p 值 (加 1 避免 p=0)
    if alternative == 'greater':
        count = int(np.sum(null_arr >= observed_diff))
    elif alternative == 'less':
        count = int(np.sum(null_arr <= observed_diff))
    else:  # two-sided
        count = int(np.sum(np.abs(null_arr) >= abs(observed_diff)))

    p_value = float((count + 1) / (n_permutations + 1))

    return {
        'p_value': p_value,
        'observed_diff': observed_diff,
        'null_distribution': null_distribution,
        'significant': bool(p_value < 0.05),
        'n_permutations': int(n_permutations),
    }


# ========== Bootstrap 置信区间 ==========

def bootstrap_ci(data, statistic=np.mean, n_bootstrap=1000, ci=95):
    """Bootstrap 置信区间

    参数:
        data: 数据数组
        statistic: 统计量函数 (默认 np.mean)
        n_bootstrap: Bootstrap 采样次数
        ci: 置信水平 (默认 95)

    返回:
        {'mean', 'ci_lower', 'ci_upper', 'bias'}
    """
    arr = np.asarray(data, dtype=float)
    n = len(arr)

    if n == 0:
        return {
            'mean': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'bias': 0.0,
        }

    observed_stat = float(statistic(arr))

    if n < 2:
        return {
            'mean': observed_stat,
            'ci_lower': observed_stat,
            'ci_upper': observed_stat,
            'bias': 0.0,
        }

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = arr[np.random.randint(0, n, size=n)]
        boot_stats.append(float(statistic(sample)))

    boot_arr = np.array(boot_stats)
    boot_mean = float(boot_arr.mean())

    alpha = (100 - ci) / 100.0
    ci_lower = float(np.percentile(boot_arr, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
    bias = float(boot_mean - observed_stat)

    return {
        'mean': observed_stat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bias': bias,
    }


# ========== 多重比较校正 ==========

def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR 多重比较校正

    参数:
        p_values: p 值列表

    返回:
        {'adjusted_p', 'significant', 'threshold'}
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    alpha = 0.05

    if n == 0:
        return {
            'adjusted_p': [],
            'significant': [],
            'threshold': float(alpha),
        }

    # 排序 (升序)
    order = np.argsort(p)
    ranked = p[order]

    # BH 调整: p_adj = p * n / rank
    adjusted = ranked * n / np.arange(1, n + 1)

    # 单调性: 从大到小取 min
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    # 恢复原始顺序
    final = np.empty(n)
    final[order] = adjusted
    final = np.minimum(final, 1.0)

    return {
        'adjusted_p': [float(v) for v in final],
        'significant': [bool(v < alpha) for v in final],
        'threshold': float(alpha),
    }


def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni 多重比较校正

    参数:
        p_values: p 值列表
        alpha: 显著性水平 (默认 0.05)

    返回:
        {'adjusted_p', 'significant', 'threshold'}
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)

    if n == 0:
        return {
            'adjusted_p': [],
            'significant': [],
            'threshold': float(alpha),
        }

    threshold = alpha / n
    adjusted = np.minimum(p * n, 1.0)

    return {
        'adjusted_p': [float(v) for v in adjusted],
        'significant': [bool(v < threshold) for v in p],
        'threshold': float(threshold),
    }


# ========== 效应量 ==========

def _effect_size_label_d(d: float) -> str:
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


def cohens_d(group_a, group_b, paired=True):
    """Cohen's d 效应量

    参数:
        group_a: A 组数据
        group_b: B 组数据
        paired: 是否为配对设计 (默认 True)

    返回:
        {'d', 'label', 'interpretation', 'ci95'}
        d < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    if paired:
        n = min(len(a), len(b))
        if n < 2:
            return {
                'd': 0.0,
                'label': 'negligible',
                'interpretation': '观测值不足，无法计算效应量',
                'ci95': [0.0, 0.0],
            }
        a = a[:n]
        b = b[:n]
        diff = a - b
        mean_diff = float(diff.mean())
        std_diff = float(diff.std(ddof=1))
        d = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0

        # CI 近似 (Hedges & Olkin)
        se_d = np.sqrt(1.0 / n + d ** 2 / (2 * n))
        t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
        ci_lower = float(d - t_crit * se_d)
        ci_upper = float(d + t_crit * se_d)
    else:
        n_a = len(a)
        n_b = len(b)
        if n_a < 2 or n_b < 2:
            return {
                'd': 0.0,
                'label': 'negligible',
                'interpretation': '观测值不足，无法计算效应量',
                'ci95': [0.0, 0.0],
            }
        var_a = float(a.var(ddof=1))
        var_b = float(b.var(ddof=1))
        pooled_std = float(np.sqrt(
            ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        ))
        mean_diff = float(a.mean() - b.mean())
        d = float(mean_diff / pooled_std) if pooled_std > 1e-12 else 0.0

        # CI 近似
        se_d = np.sqrt((n_a + n_b) / (n_a * n_b) + d ** 2 / (2 * (n_a + n_b)))
        t_crit = float(scipy_stats.t.ppf(0.975, df=n_a + n_b - 2))
        ci_lower = float(d - t_crit * se_d)
        ci_upper = float(d + t_crit * se_d)

    label = _effect_size_label_d(d)
    interpretation = (
        f"Cohen's d = {d:.3f}, 属于{label}效应量。"
        f"两组差异约为 {abs(d):.3f} 个标准差。"
    )

    return {
        'd': d,
        'label': label,
        'interpretation': interpretation,
        'ci95': [ci_lower, ci_upper],
    }


def eta_squared(groups):
    """η² (eta squared) 效应量（ANOVA）

    参数:
        groups: 多组数据列表 [array1, array2, ...]

    返回:
        {'eta_squared', 'partial_eta_squared', 'label'}
    """
    k = len(groups)
    if k < 2:
        return {
            'eta_squared': 0.0,
            'partial_eta_squared': 0.0,
            'label': 'negligible',
        }

    n = min(len(g) for g in groups)
    if n < 2:
        return {
            'eta_squared': 0.0,
            'partial_eta_squared': 0.0,
            'label': 'negligible',
        }

    data = np.array([np.asarray(g, dtype=float)[:n] for g in groups]).T  # (n, k)
    grand_mean = float(data.mean())
    cond_means = data.mean(axis=0)

    ss_between = float(n * np.sum((cond_means - grand_mean) ** 2))
    ss_within = float(np.sum((data - cond_means) ** 2))
    ss_total = float(np.sum((data - grand_mean) ** 2))

    # 重复测量: 扣除被试间变异
    subj_means = data.mean(axis=1)
    ss_subject = float(k * np.sum((subj_means - grand_mean) ** 2))
    ss_error = ss_within - ss_subject

    eta_sq = float(ss_between / (ss_total + 1e-12))
    partial_eta = float(ss_between / (ss_between + ss_error + 1e-12))

    # 标签 (Cohen 标准: 0.01/0.06/0.14)
    if partial_eta < 0.01:
        label = 'negligible'
    elif partial_eta < 0.06:
        label = 'small'
    elif partial_eta < 0.14:
        label = 'medium'
    else:
        label = 'large'

    return {
        'eta_squared': eta_sq,
        'partial_eta_squared': partial_eta,
        'label': label,
    }


# ========== 标准误与置信区间 ==========

def compute_sem(data, axis=0):
    """标准误 (Standard Error of Mean)

    参数:
        data: 数据数组
        axis: 计算轴

    返回: SEM 值 (float 或 array)
    """
    arr = np.asarray(data, dtype=float)
    n = arr.shape[axis] if arr.ndim > 0 else 1
    if n < 2:
        return 0.0
    sem = arr.std(axis=axis, ddof=1) / np.sqrt(n)
    if np.ndim(sem) == 0:
        return float(sem)
    return sem


def compute_ci95(data, axis=0):
    """95% 置信区间（t分布）

    参数:
        data: 数据数组
        axis: 计算轴

    返回: [lower, upper]
    """
    arr = np.asarray(data, dtype=float)
    n = arr.shape[axis] if arr.ndim > 0 else 1
    mean = arr.mean(axis=axis)

    if n < 2:
        if np.ndim(mean) == 0:
            return [float(mean), float(mean)]
        return [mean, mean]

    sem = compute_sem(arr, axis=axis)
    t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
    lower = mean - t_crit * sem
    upper = mean + t_crit * sem

    if np.ndim(lower) == 0:
        return [float(lower), float(upper)]
    return [lower, upper]


# ========== 完整统计检验流水线 ==========

def run_rigorous_stats(group_a, group_b, metric_name='metric', paired=True):
    """完整统计检验流水线

    参数:
        group_a: A 组数据
        group_b: B 组数据
        metric_name: 指标名称
        paired: 是否为配对设计 (默认 True)

    返回:
        {
            'metric': metric_name,
            'descriptive': {'a': {'mean', 'std', 'sem', 'ci95', 'n'}, 'b': {...}},
            'paired_t': {...},
            'permutation': {...},
            'effect_size': {...},
            'bootstrap_ci': {...},
            'summary': '...text...'  # 人类可读的统计结论
        }
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    # 描述性统计
    def _desc(arr):
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        if n == 0:
            return {'mean': 0.0, 'std': 0.0, 'sem': 0.0, 'ci95': [0.0, 0.0], 'n': 0}
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        if n > 1:
            t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
            ci95 = [float(mean - t_crit * sem), float(mean + t_crit * sem)]
        else:
            ci95 = [mean, mean]
        return {'mean': mean, 'std': std, 'sem': sem, 'ci95': ci95, 'n': n}

    desc_a = _desc(a)
    desc_b = _desc(b)

    # 配对 t 检验
    tt = paired_t_test(a, b)

    # 置换检验
    perm = permutation_test(a, b, n_permutations=1000)

    # 效应量
    es = cohens_d(a, b, paired=paired)

    # Bootstrap CI (对差值)
    n = min(len(a), len(b))
    diff = a[:n] - b[:n] if n > 0 else np.array([0.0])
    boot = bootstrap_ci(diff, statistic=np.mean, n_bootstrap=1000)

    # 生成人类可读摘要
    sig_t = '显著' if tt['p_value'] < 0.05 else '不显著'
    sig_p = '显著' if perm['p_value'] < 0.05 else '不显著'
    summary = (
        f"指标 {metric_name}: A组均值={desc_a['mean']:.2f}±{desc_a['std']:.2f} "
        f"(n={desc_a['n']}), B组均值={desc_b['mean']:.2f}±{desc_b['std']:.2f} "
        f"(n={desc_b['n']}). "
        f"配对t检验: t={tt['t_stat']:.3f}, p={tt['p_value']:.4f} ({sig_t}), "
        f"df={tt['df']}. "
        f"均值差={tt['mean_diff']:.2f}, 95%CI=[{tt['ci95'][0]:.2f}, "
        f"{tt['ci95'][1]:.2f}]. "
        f"Cohen's d={es['d']:.3f} ({es['label']}). "
        f"置换检验p={perm['p_value']:.4f} ({sig_p}, {perm['n_permutations']}次). "
        f"Bootstrap均值差95%CI=[{boot['ci_lower']:.2f}, {boot['ci_upper']:.2f}]."
    )

    return {
        'metric': metric_name,
        'descriptive': {'a': desc_a, 'b': desc_b},
        'paired_t': tt,
        'permutation': perm,
        'effect_size': es,
        'bootstrap_ci': boot,
        'summary': summary,
    }


# ========== 跨条件统计比较 ==========

def cross_condition_stats(results_store, metric='recovery_time'):
    """跨条件统计比较

    对 RESULTS_STORE 中所有条件两两比较

    参数:
        results_store: {condition: result_dict, ...}
        metric: 要比较的指标键 (默认 'recovery_time')

    返回:
        {
            'metric': ...,
            'conditions': [...],
            'condition_values': {condition: [values]},
            'pairwise': [
                {'pair': 'A_vs_B', 't_test': {...}, 'permutation': {...}, 'effect_size': {...}},
                ...
            ],
            'fdr_corrected': {...},  # 所有 p 值的 FDR 校正
        }
    """
    conditions = list(results_store.keys())

    # 提取各条件的指标值 (支持标量和列表)
    condition_values = {}
    for cond in conditions:
        res = results_store.get(cond, {})
        if not isinstance(res, dict):
            continue
        val = res.get(metric)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            vals = [float(v) for v in val if v is not None]
            if vals:
                condition_values[cond] = vals
        else:
            try:
                condition_values[cond] = [float(val)]
            except (TypeError, ValueError):
                continue

    valid_conditions = list(condition_values.keys())

    # 两两比较
    pairwise = []
    all_p_values = []
    pair_names = []

    for i in range(len(valid_conditions)):
        for j in range(i + 1, len(valid_conditions)):
            c_a = valid_conditions[i]
            c_b = valid_conditions[j]
            vals_a = condition_values[c_a]
            vals_b = condition_values[c_b]
            pair_name = f"{c_a}_vs_{c_b}"

            if len(vals_a) >= 2 and len(vals_b) >= 2:
                tt = paired_t_test(vals_a, vals_b)
                perm = permutation_test(vals_a, vals_b, n_permutations=1000)
                es = cohens_d(vals_a, vals_b, paired=True)
                pairwise.append({
                    'pair': pair_name,
                    't_test': tt,
                    'permutation': perm,
                    'effect_size': es,
                })
                all_p_values.append(tt['p_value'])
                pair_names.append(pair_name)
            else:
                pairwise.append({
                    'pair': pair_name,
                    't_test': None,
                    'permutation': None,
                    'effect_size': None,
                    'note': '观测值不足 (每组至少需要 2 个)',
                })

    # FDR 校正
    if all_p_values:
        fdr = benjamini_hochberg(all_p_values)
        fdr_corrected = {
            'pairs': pair_names,
            'adjusted_p': fdr['adjusted_p'],
            'significant': fdr['significant'],
            'threshold': fdr['threshold'],
        }
    else:
        fdr_corrected = {
            'pairs': [],
            'adjusted_p': [],
            'significant': [],
            'threshold': 0.05,
        }

    return {
        'metric': metric,
        'conditions': valid_conditions,
        'condition_values': condition_values,
        'pairwise': pairwise,
        'fdr_corrected': fdr_corrected,
    }
