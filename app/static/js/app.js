/* ==========================================================
   EEG 心流恢复分析工具箱 — 前端交互
   ========================================================== */

// ---------- 全局状态 ----------
let selectedCondition = 'AtoA';
let charts = {};
let analyzedConditions = new Set();

// 指标中文名与颜色 (TraeWork 数据可视化色板)
const INDICATORS = [
    { key: 'theta_alpha_ratio', name: 'Theta/Alpha 比值', color: '#4B3FE3', type: 'flow' },
    { key: 'alpha_rel',         name: 'Alpha 能量',      color: '#1DC981', type: 'flow' },
    { key: 'beta_rel',          name: 'Beta 能量',       color: '#22A5F7', type: 'flow' },
    { key: 'gamma_rel',         name: 'Gamma 能量',      color: '#F87454', type: 'loss' },
    { key: 'eeg_entropy',       name: '脑电熵值',        color: '#EDAA45', type: 'loss' },
    { key: 'cog_load',          name: '认知负载指数',    color: '#B655FC', type: 'loss' },
];

const CONDITION_LABELS = {
    'AtoA': 'A→A (对照)',
    'AtoB': 'A→B (文理)',
    'AtoC': 'A→C (理艺)',
    'BtoC': 'B→C (文艺)',
};

// ========== 安全 fetch helper ==========
async function fetchJSON(url, options) {
    const resp = await fetch(url, options);
    if (!resp.ok) {
        let msg = `HTTP ${resp.status}`;
        try {
            const errData = await resp.json();
            msg = errData.detail || errData.error || msg;
        } catch (e) { /* 非 JSON 响应 */ }
        throw new Error(msg);
    }
    return resp.json();
}

// ==========================================================
// 初始化
// ==========================================================
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initConditionCards();
    initButtons();
    initFileInputs();
    initSidebarNav();
    if (window.initRealtime) initRealtime();
});

// ---------- 标签页切换 ----------
// 仅处理心流恢复视图的源数据标签 (data-tab)，其他模块各自绑定避免互相干扰
function initTabs() {
    document.querySelectorAll('.tab-btn[data-tab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll('.tab-btn[data-tab]').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('#view-flow-recovery .tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`panel-${tab}`).classList.add('active');
            // 切换操作按钮显示
            document.getElementById('btn-run-sample').style.display = tab === 'sample' ? '' : 'none';
            document.getElementById('btn-upload').style.display = tab === 'upload' ? '' : 'none';
        });
    });
}

// ---------- 条件卡片选择 ----------
function initConditionCards() {
    document.querySelectorAll('.cond-card').forEach(card => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.cond-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            selectedCondition = card.dataset.condition;
        });
    });
}

// ---------- 按钮绑定 ----------
function initButtons() {
    document.getElementById('btn-run-sample').addEventListener('click', runSample);
    document.getElementById('btn-upload').addEventListener('click', uploadAndAnalyze);
    document.getElementById('btn-refresh-stats').addEventListener('click', refreshStats);
    document.getElementById('btn-report-single')?.addEventListener('click', () => generateReport(selectedCondition));
    document.getElementById('btn-report-full')?.addEventListener('click', () => generateReport(null));
}

// ---------- 文件输入提示 ----------
function initFileInputs() {
    const eegInput = document.getElementById('file-eeg');
    const evtInput = document.getElementById('file-events');
    eegInput.addEventListener('change', () => {
        document.getElementById('hint-eeg').textContent = eegInput.files[0]?.name || '未选择文件';
    });
    evtInput.addEventListener('change', () => {
        document.getElementById('hint-events').textContent = evtInput.files[0]?.name || '未选择文件';
    });
}

// ---------- 侧边栏导航 ----------
function initSidebarNav() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            if (item.classList.contains('nav-item-disabled')) {
                const name = item.querySelector('.nav-item-text')?.textContent || '该模块';
                showToast(`${name}即将推出`);
                return;
            }
            // 切换导航高亮
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            // 切换视图
            const module = item.dataset.module;
            document.querySelectorAll('.module-view').forEach(v => v.classList.remove('active'));
            const view = document.getElementById(`view-${module}`);
            if (view) view.classList.add('active');
            // 被试管理：加载数据
            if (module === 'subjects') loadSubjects();
        });
    });
}

// ---------- 轻量提示 ----------
function showToast(msg) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = msg;
    toast.style.cssText = `
        position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
        background: var(--invert); color: #fff; padding: 8px 16px;
        border-radius: 6px; font-size: 13px; z-index: 2000;
        opacity: 0; transition: opacity 0.2s;
    `;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.style.opacity = '1');
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 200);
    }, 2000);
}

// ==========================================================
// 读取参数
// ==========================================================
function getParams() {
    return {
        hp: parseFloat(document.getElementById('param-hp').value),
        lp: parseFloat(document.getElementById('param-lp').value),
        notch: parseFloat(document.getElementById('param-notch').value),
        artifact_threshold: parseFloat(document.getElementById('param-artifact').value),
        window_sec: parseFloat(document.getElementById('param-window').value),
        overlap: parseFloat(document.getElementById('param-overlap').value),
        tolerance: parseFloat(document.getElementById('param-tolerance').value) / 100,
        recovery_window: parseInt(document.getElementById('param-recovery-win').value),
    };
}

// ==========================================================
// 运行模拟数据分析
// ==========================================================
async function runSample() {
    showLoading(true);
    try {
        const data = await fetchJSON('/api/sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ condition: selectedCondition, fs: 250 }),
        });
        if (data.error) throw new Error(data.error);
        analyzedConditions.add(selectedCondition);
        renderResults(data);
        document.getElementById('result-empty').style.display = 'none';
        document.getElementById('result-content').style.display = 'flex';
        document.getElementById('stats-block').style.display = 'block';
    } catch (err) {
        alert('分析失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

// ==========================================================
// 上传并分析
// ==========================================================
async function uploadAndAnalyze() {
    const eegFile = document.getElementById('file-eeg').files[0];
    if (!eegFile) { alert('请选择EEG数据文件'); return; }

    const condition = document.getElementById('upload-condition').value || 'custom';
    const evtFile = document.getElementById('file-events').files[0];
    const params = getParams();

    showLoading(true);
    try {
        // 上传
        const formData = new FormData();
        formData.append('eeg_file', eegFile);
        if (evtFile) formData.append('events_file', evtFile);
        formData.append('condition', condition);

        const uploadResp = await fetch('/api/upload', { method: 'POST', body: formData });
        const uploadData = await uploadResp.json();
        if (uploadData.detail) throw new Error(uploadData.detail);

        // 分析
        const analyzeResp = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...params, condition }),
        });
        const data = await analyzeResp.json();
        if (data.error) throw new Error(data.error);

        analyzedConditions.add(condition);
        renderResults(data);
        document.getElementById('result-empty').style.display = 'none';
        document.getElementById('result-content').style.display = 'flex';
        document.getElementById('stats-block').style.display = 'block';
    } catch (err) {
        alert('分析失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

// ==========================================================
// 渲染结果
// ==========================================================
function renderResults(data) {
    const cond = data.condition || selectedCondition;
    document.getElementById('result-condition-badge').textContent = CONDITION_LABELS[cond] || cond;

    // 指标卡片
    const rt = data.recovery_time;
    document.getElementById('metric-recovery').textContent = rt !== null && rt !== undefined ? rt.toFixed(1) : '>600';
    document.getElementById('metric-artifact').textContent = (data.artifact_ratio * 100).toFixed(2);
    document.getElementById('metric-duration').textContent = (data.duration_sec / 60).toFixed(1);
    document.getElementById('metric-samples').textContent = data.n_samples?.toLocaleString() || '—';

    // 图表
    renderTimeSeriesChart(data);
    renderRecoveryBar(data);
    renderAttenuationHeatmap(data);
    renderLegend();

    // 新增:模块借鉴渲染
    if (data.band_powers) renderSpectrum(data);
    if (data.topomap_data) renderTopomapModule(data.topomap_data);
    if (data.focus_scores) renderFocus(data.focus_scores);
}

// ==========================================================
// 时序曲线图 (核心图)
// ==========================================================
function renderTimeSeriesChart(data) {
    const canvas = document.getElementById('chart-timeseries');
    const ctx = canvas.getContext('2d');

    if (charts.timeseries) charts.timeseries.destroy();

    const viz = data.viz_data;
    const ts = viz.timestamp.map(t => t / 60); // 转为分钟
    const evt = data.event_times;

    // 构建数据集
    const datasets = INDICATORS.map(ind => ({
        label: ind.name,
        data: viz[ind.key].map(v => v || 0),
        borderColor: ind.color,
        backgroundColor: ind.color + '15',
        borderWidth: 1.8,
        pointRadius: 0,
        tension: 0.35,
        fill: false,
    }));

    // ±5% 阈值带
    datasets.push({
        label: '±5% 阈值',
        data: ts.map(() => 1.0),
        borderColor: 'transparent',
        backgroundColor: 'rgba(115,115,115,0.06)',
        pointRadius: 0,
        fill: { target: { value: 1.05 }, above: 'rgba(115,115,115,0.06)' },
    });

    // 相位背景插件 (TraeWork 中性叠加色)
    const phasePlugin = {
        id: 'phaseBackground',
        beforeDraw(chart) {
            const { ctx, chartArea, scales } = chart;
            if (!chartArea) return;
            const xScale = scales.x;

            // 心流稳态期 (中性浅灰)
            const f1 = evt.flow_steady_start / 60;
            const f2 = evt.flow_steady_end / 60;
            const x1 = xScale.getPixelForValue(f1);
            const x2 = xScale.getPixelForValue(f2);
            ctx.fillStyle = 'rgba(115,115,115,0.05)';
            ctx.fillRect(x1, chartArea.top, x2 - x1, chartArea.bottom - chartArea.top);

            // 切换期 (error surface 浅红)
            const x0 = evt.switch_start / 60;
            const x1s = evt.switch_end / 60;
            const sx1 = xScale.getPixelForValue(x0);
            const sx2 = xScale.getPixelForValue(x1s);
            ctx.fillStyle = 'rgba(232,70,58,0.06)';
            ctx.fillRect(sx1, chartArea.top, sx2 - sx1, chartArea.bottom - chartArea.top);

            // 恢复期 (primary surface 浅蓝)
            const r0 = evt.recovery_start / 60;
            const rx1 = xScale.getPixelForValue(r0);
            ctx.fillStyle = 'rgba(47,116,255,0.04)';
            ctx.fillRect(rx1, chartArea.top, chartArea.right - rx1, chartArea.bottom - chartArea.top);

            // 竖线标注
            ctx.strokeStyle = 'rgba(232,70,58,0.36)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(sx1, chartArea.top);
            ctx.lineTo(sx1, chartArea.bottom);
            ctx.stroke();

            ctx.strokeStyle = 'rgba(47,116,255,0.36)';
            ctx.beginPath();
            ctx.moveTo(rx1, chartArea.top);
            ctx.lineTo(rx1, chartArea.bottom);
            ctx.stroke();
            ctx.setLineDash([]);

            // 文字标注
            ctx.fillStyle = '#737373';
            ctx.font = '11px ' + getComputedStyle(document.body).fontFamily;
            ctx.textAlign = 'center';
            ctx.fillText('稳态', (x1 + x2) / 2, chartArea.top + 14);
            ctx.fillText('切换', (sx1 + sx2) / 2, chartArea.top + 14);
            ctx.fillText('恢复', (rx1 + chartArea.right) / 2, chartArea.top + 14);
        }
    };

    charts.timeseries = new Chart(ctx, {
        type: 'line',
        data: { labels: ts, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#FFFFFF',
                    titleColor: '#171717',
                    bodyColor: '#404040',
                    borderColor: 'rgba(115,115,115,0.18)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    titleFont: { family: 'SF Pro Text, Inter, sans-serif', size: 13 },
                    bodyFont: { size: 12 },
                    callbacks: {
                        title: (items) => `时间: ${parseFloat(items[0].label).toFixed(2)} min`,
                        label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(3)}`,
                    },
                },
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: '时间 (分钟)', color: '#737373', font: { size: 12 } },
                    grid: { color: 'rgba(115,115,115,0.08)' },
                    ticks: { color: '#737373', font: { size: 11 } },
                },
                y: {
                    title: { display: true, text: '归一化值 (稳态=1.0)', color: '#737373', font: { size: 12 } },
                    grid: { color: 'rgba(115,115,115,0.08)' },
                    ticks: { color: '#737373', font: { size: 11 } },
                    suggestedMin: 0.4,
                    suggestedMax: 1.6,
                },
            },
        },
        plugins: [phasePlugin],
    });
}

// ==========================================================
// 恢复时长柱状图
// ==========================================================
function renderRecoveryBar(data) {
    const canvas = document.getElementById('chart-recovery-bar');
    const ctx = canvas.getContext('2d');
    if (charts.recoveryBar) charts.recoveryBar.destroy();

    const perFeature = data.recovery_per_feature;
    const labels = INDICATORS.map(i => i.name);
    const values = INDICATORS.map(i => {
        const v = perFeature[i.key];
        return v !== null && v !== undefined ? v : 600;
    });
    const colors = INDICATORS.map(i => i.color);

    charts.recoveryBar = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors.map(c => c + 'CC'),
                borderColor: colors,
                borderWidth: 1.5,
                borderRadius: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#FFFFFF',
                    titleColor: '#171717',
                    bodyColor: '#404040',
                    borderColor: 'rgba(115,115,115,0.18)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: (item) => {
                            const v = item.parsed.y;
                            return v >= 600 ? '未恢复 (>600s)' : `恢复时长: ${v.toFixed(1)}s`;
                        },
                    },
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#404040', font: { size: 11 } } },
                y: {
                    title: { display: true, text: '恢复时长 (秒)', color: '#737373', font: { size: 12 } },
                    grid: { color: 'rgba(115,115,115,0.08)' },
                    ticks: { color: '#737373', font: { size: 11 } },
                },
            },
        },
    });
}

// ==========================================================
// 衰减幅度热图
// ==========================================================
function renderAttenuationHeatmap(data) {
    const container = document.getElementById('attenuation-heatmap');
    const att = data.attenuation;

    const items = [
        ...INDICATORS.filter(i => i.type === 'flow').map(i => ({ ...i, val: att[i.key] || 0, label: '跌落' })),
        ...INDICATORS.filter(i => i.type === 'loss').map(i => ({ ...i, val: att[i.key] || 0, label: '升高' })),
    ];

    const maxVal = Math.max(...items.map(i => Math.abs(i.val)), 1);

    container.innerHTML = items.map(item => {
        const widthPct = Math.min(Math.abs(item.val) / maxVal * 100, 100);
        const isPositive = item.val >= 0;
        const color = isPositive ? item.color : '#737373';
        return `
            <div class="heatmap-row">
                <div class="heatmap-label">${item.name} <span style="color:#737373;font-size:11px">(${item.label})</span></div>
                <div class="heatmap-bar-wrap">
                    <div class="heatmap-bar" style="width:${widthPct}%;background:${color}">
                        ${item.val.toFixed(1)}%
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ==========================================================
// 图例
// ==========================================================
function renderLegend() {
    const container = document.getElementById('indicator-legend');
    container.innerHTML = INDICATORS.map(ind => `
        <div class="legend-item">
            <span class="legend-swatch" style="background:${ind.color}"></span>
            <span>${ind.name}</span>
            <span style="color:#737373;font-size:11px">${ind.type === 'flow' ? '心流核心' : '认知损耗'}</span>
        </div>
    `).join('');
}

// ==========================================================
// 跨条件统计
// ==========================================================
async function refreshStats() {
    showLoading(true);
    try {
        const resp = await fetch('/api/stats');
        const data = await resp.json();
        if (data.error) {
            alert(data.error);
            return;
        }
        renderStatsRecoveryChart(data);
        renderStatsAttenuationChart(data);
        renderStatsTable(data);
    } catch (err) {
        alert('统计失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

function renderStatsRecoveryChart(data) {
    const canvas = document.getElementById('chart-stats-recovery');
    const ctx = canvas.getContext('2d');
    if (charts.statsRecovery) charts.statsRecovery.destroy();

    const conds = data.conditions;
    const labels = conds.map(c => CONDITION_LABELS[c] || c);
    const values = conds.map(c => {
        const v = data.recovery_times[c];
        return v !== null && v !== undefined ? v : 600;
    });
    const colors = conds.map(c => c === 'AtoA' ? '#15A877' : '#4B3FE3');

    charts.statsRecovery = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 6 }] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#FFFFFF', titleColor: '#171717', bodyColor: '#404040',
                    borderColor: 'rgba(115,115,115,0.18)', borderWidth: 1, cornerRadius: 8,
                    callbacks: { label: (item) => item.parsed.y >= 600 ? '未恢复' : `${item.parsed.y.toFixed(1)}s` },
                },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#404040', font: { size: 11 } } },
                y: { title: { display: true, text: '恢复时长 (秒)', color: '#737373' }, grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373' } },
            },
        },
    });
}

function renderStatsAttenuationChart(data) {
    const canvas = document.getElementById('chart-stats-attenuation');
    const ctx = canvas.getContext('2d');
    if (charts.statsAttenuation) charts.statsAttenuation.destroy();

    const conds = data.conditions;
    const labels = conds.map(c => CONDITION_LABELS[c] || c);
    const indicators = ['theta_alpha_ratio', 'alpha_rel', 'gamma_rel', 'cog_load'];
    const indColors = { theta_alpha_ratio: '#4B3FE3', alpha_rel: '#1DC981', gamma_rel: '#F87454', cog_load: '#B655FC' };

    const datasets = indicators.map(key => ({
        label: INDICATORS.find(i => i.key === key).name,
        data: conds.map(c => data.attenuations[c]?.[key] || 0),
        backgroundColor: indColors[key] + 'CC',
        borderColor: indColors[key],
        borderWidth: 1.5,
        borderRadius: 4,
    }));

    charts.statsAttenuation = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#404040', font: { size: 11 }, boxWidth: 12, padding: 12 } },
                tooltip: { backgroundColor: '#FFFFFF', titleColor: '#171717', bodyColor: '#404040', borderColor: 'rgba(115,115,115,0.18)', borderWidth: 1, cornerRadius: 8 },
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#404040', font: { size: 11 } } },
                y: { title: { display: true, text: '衰减幅度 (%)', color: '#737373' }, grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373' } },
            },
        },
    });
}

function renderStatsTable(data) {
    const wrap = document.getElementById('stats-table-wrap');
    let html = '<table class="stats-table"><thead><tr><th>检验类型</th><th>比较</th><th>统计量</th><th>p值</th><th>效应量</th><th>显著性</th></tr></thead><tbody>';

    // 配对t检验
    if (data.paired_t_tests) {
        for (const [key, res] of Object.entries(data.paired_t_tests)) {
            const sig = res.p < 0.05;
            html += `<tr>
                <td>配对t检验</td>
                <td>${key.replace('_vs_', ' vs ')}</td>
                <td class="mono">t = ${res.t.toFixed(3)}</td>
                <td class="mono ${sig ? 'p-sig' : 'p-ns'}">${res.p.toFixed(4)}</td>
                <td class="mono">d = ${res.d.toFixed(3)}</td>
                <td>${sig ? '显著 *' : '不显著'}</td>
            </tr>`;
        }
    }

    // ANOVA
    if (data.anova) {
        for (const [key, res] of Object.entries(data.anova)) {
            if (!res || res.F === 0) continue;
            const sig = res.p < 0.05;
            html += `<tr>
                <td>重复测量ANOVA</td>
                <td>${key} (三类切换)</td>
                <td class="mono">F(${res.df1},${res.df2}) = ${res.F.toFixed(3)}</td>
                <td class="mono ${sig ? 'p-sig' : 'p-ns'}">${res.p.toFixed(4)}</td>
                <td class="mono">η² = ${res.eta2.toFixed(3)}</td>
                <td>${sig ? '显著 *' : '不显著'}</td>
            </tr>`;
        }
    }

    html += '</tbody></table>';
    html += '<p style="margin-top:12px;font-size:12px;color:#737373">* p&lt;0.05 显著；效应量: Cohen\'s d (t检验), η² (ANOVA)。被试内设计采用配对检验。</p>';
    wrap.innerHTML = html;
}

// ==========================================================
// 工具函数
// ==========================================================
function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'flex' : 'none';
}

// ==========================================================
// 一键生成报告
// ==========================================================
async function generateReport(condition) {
    showLoading(true);
    try {
        const url = condition ? `/api/report/${condition}` : '/api/report';
        const resp = await fetch(url);
        const data = await resp.json();
        if (data.detail) throw new Error(data.detail);

        // 捕获当前图表图像
        const images = captureChartImages();

        // 构建报告 HTML 并在新窗口打开
        const html = buildReportHTML(data, images);
        const reportWin = window.open('', '_blank');
        if (!reportWin) {
            alert('请允许弹出窗口以查看报告');
            return;
        }
        reportWin.document.write(html);
        reportWin.document.close();
    } catch (err) {
        alert('报告生成失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

// ---------- 捕获图表图像 ----------
function captureChartImages() {
    const images = {};
    const canvasIds = ['chart-timeseries', 'chart-recovery-bar', 'chart-stats-recovery', 'chart-stats-attenuation'];
    for (const id of canvasIds) {
        const canvas = document.getElementById(id);
        if (canvas && canvas.width > 0) {
            try {
                images[id] = canvas.toDataURL('image/png');
            } catch (e) {
                images[id] = null;
            }
        }
    }
    return images;
}

// ---------- 构建报告 HTML ----------
function buildReportHTML(data, images) {
    const sections = data.sections || [];
    const stats = data.stats_summary;

    let bodyHTML = '';

    // 各条件分析章节
    sections.forEach((sec, idx) => {
        const ds = sec.data_summary;
        const kf = sec.key_findings;
        const rt = kf.recovery_time;
        const rtDisplay = rt !== null && rt !== undefined ? `${rt.toFixed(1)} 秒` : '未恢复 (>600s)';

        bodyHTML += `
        <section class="report-section">
            <h2>${idx + 1}. ${sec.condition_label}</h2>
            <p class="section-desc">${sec.condition_desc}</p>

            <h3>数据概况</h3>
            <table class="data-table">
                <tr><td>数据时长</td><td>${ds.duration_min.toFixed(1)} 分钟</td></tr>
                <tr><td>采样点数</td><td>${ds.n_samples.toLocaleString()}</td></tr>
                <tr><td>采集通道</td><td>${(ds.channels || ['Fp1','Fp2','Fpz']).join(', ')}</td></tr>
                <tr><td>伪迹占比</td><td>${(ds.artifact_ratio * 100).toFixed(2)}%</td></tr>
            </table>

            <h3>核心发现</h3>
            <div class="key-metric">
                <span class="metric-label">综合恢复时长</span>
                <span class="metric-value">${rtDisplay}</span>
            </div>
            <p class="finding-text">${kf.recovery_interpretation}</p>
        `;

        // 时序图 (仅单条件报告或第一个条件时显示)
        if (idx === 0 && images['chart-timeseries']) {
            bodyHTML += `
            <h3>心流稳态 — 跌落 — 恢复 时序演化</h3>
            <img src="${images['chart-timeseries']}" class="chart-img" alt="时序曲线图">
            `;
        }

        // 各指标明细表
        bodyHTML += `
            <h3>各指标恢复详情</h3>
            <table class="indicator-table">
                <thead>
                    <tr><th>指标</th><th>说明</th><th>稳态基准</th><th>恢复时长</th><th>衰减幅度</th></tr>
                </thead>
                <tbody>
        `;
        for (const ind of sec.indicator_details) {
            const indRt = ind.recovery_time;
            const indRtDisplay = indRt !== null && indRt !== undefined ? `${indRt.toFixed(1)}s` : '未恢复';
            bodyHTML += `<tr>
                <td>${ind.name}</td>
                <td class="desc">${ind.desc}</td>
                <td class="mono">${ind.baseline.toFixed(4)}</td>
                <td class="mono">${indRtDisplay}</td>
                <td class="mono">${ind.attenuation.toFixed(1)}%</td>
            </tr>`;
        }
        bodyHTML += `</tbody></table>`;

        // 恢复时长柱状图
        if (images['chart-recovery-bar']) {
            bodyHTML += `<img src="${images['chart-recovery-bar']}" class="chart-img" alt="恢复时长明细图">`;
        }

        // 结论
        bodyHTML += `<h3>分析结论</h3><ol class="conclusion-list">`;
        for (const c of sec.conclusions) {
            bodyHTML += `<li>${c}</li>`;
        }
        bodyHTML += `</ol>`;

        // 建议
        bodyHTML += `<h3>实践建议</h3><ul class="suggestion-list">`;
        for (const s of sec.suggestions) {
            bodyHTML += `<li>${s}</li>`;
        }
        bodyHTML += `</ul></section>`;
    });

    // 跨条件统计章节
    if (stats && stats.conditions && stats.conditions.length >= 2) {
        bodyHTML += `
        <section class="report-section">
            <h2>${sections.length + 1}. 跨条件统计比较</h2>
            <h3>各条件恢复时长对比</h3>
        `;
        if (images['chart-stats-recovery']) {
            bodyHTML += `<img src="${images['chart-stats-recovery']}" class="chart-img" alt="恢复时长对比图">`;
        }
        if (images['chart-stats-attenuation']) {
            bodyHTML += `<h3>各条件衰减幅度对比</h3><img src="${images['chart-stats-attenuation']}" class="chart-img" alt="衰减幅度对比图">`;
        }

        // 统计检验表
        bodyHTML += `<h3>统计检验结果</h3><table class="stats-table"><thead><tr><th>检验类型</th><th>比较</th><th>统计量</th><th>p值</th><th>效应量</th><th>显著性</th></tr></thead><tbody>`;

        if (stats.paired_t_tests) {
            for (const [key, res] of Object.entries(stats.paired_t_tests)) {
                const sig = res.p < 0.05;
                bodyHTML += `<tr>
                    <td>配对t检验</td>
                    <td>${key.replace('_vs_', ' vs ')}</td>
                    <td class="mono">t = ${res.t.toFixed(3)}</td>
                    <td class="mono ${sig ? 'sig' : ''}">${res.p.toFixed(4)}</td>
                    <td class="mono">d = ${res.d.toFixed(3)}</td>
                    <td>${sig ? '显著 *' : '不显著'}</td>
                </tr>`;
            }
        }
        if (stats.anova) {
            for (const [key, res] of Object.entries(stats.anova)) {
                if (!res || res.F === 0) continue;
                const sig = res.p < 0.05;
                bodyHTML += `<tr>
                    <td>重复测量ANOVA</td>
                    <td>${key} (三类切换)</td>
                    <td class="mono">F(${res.df1},${res.df2}) = ${res.F.toFixed(3)}</td>
                    <td class="mono ${sig ? 'sig' : ''}">${res.p.toFixed(4)}</td>
                    <td class="mono">η² = ${res.eta2.toFixed(3)}</td>
                    <td>${sig ? '显著 *' : '不显著'}</td>
                </tr>`;
            }
        }
        bodyHTML += `</tbody></table>`;
        bodyHTML += `<p class="note">* p&lt;0.05 显著；效应量: Cohen's d (t检验), η² (ANOVA)。被试内设计采用配对检验。</p>`;
        bodyHTML += `</section>`;
    }

    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${data.title}</title>
<style>
    :root {
        --brand: #4B3FE3;
        --text-default: #171717;
        --text-secondary: #404040;
        --text-tertiary: #737373;
        --border: rgba(115,115,115,0.18);
        --border-light: rgba(115,115,115,0.12);
        --bg-secondary: #F5F5F5;
        --radius: 8px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, 'SF Pro Text', 'Inter', 'PingFang SC', sans-serif;
        color: var(--text-default);
        background: #FFFFFF;
        line-height: 1.6;
        font-size: 14px;
        -webkit-font-smoothing: antialiased;
    }
    .report {
        max-width: 800px;
        margin: 0 auto;
        padding: 48px 56px;
    }
    .report-header {
        border-bottom: 1px solid var(--border);
        padding-bottom: 32px;
        margin-bottom: 32px;
    }
    .report-header h1 {
        font-size: 24px;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }
    .report-header .subtitle {
        font-size: 13px;
        color: var(--text-tertiary);
        margin-bottom: 16px;
    }
    .report-meta {
        display: flex;
        gap: 24px;
        font-size: 12px;
        color: var(--text-tertiary);
    }
    .report-meta span { font-variant-numeric: tabular-nums; }
    .toolbar {
        position: fixed;
        top: 16px;
        right: 16px;
        z-index: 100;
    }
    .toolbar button {
        padding: 8px 20px;
        background: var(--brand);
        color: #FFFFFF;
        border: none;
        border-radius: var(--radius);
        font-size: 13px;
        cursor: pointer;
        font-family: inherit;
    }
    .toolbar button:hover { opacity: 0.9; }
    .report-section {
        margin-bottom: 40px;
    }
    .report-section h2 {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
        letter-spacing: -0.01em;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-light);
    }
    .report-section h3 {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-secondary);
        margin-top: 24px;
        margin-bottom: 12px;
    }
    .section-desc {
        font-size: 13px;
        color: var(--text-tertiary);
        margin-bottom: 20px;
    }
    .data-table, .indicator-table, .stats-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-bottom: 16px;
    }
    .data-table td {
        padding: 8px 12px;
        border-bottom: 1px solid var(--border-light);
    }
    .data-table td:first-child {
        color: var(--text-tertiary);
        width: 40%;
    }
    .data-table td:last-child {
        font-variant-numeric: tabular-nums;
        font-weight: 500;
    }
    .indicator-table th, .stats-table th {
        text-align: left;
        padding: 10px 12px;
        font-size: 12px;
        font-weight: 500;
        color: var(--text-tertiary);
        border-bottom: 1px solid var(--border);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .indicator-table td, .stats-table td {
        padding: 10px 12px;
        border-bottom: 1px solid var(--border-light);
        font-size: 13px;
    }
    .indicator-table td.desc { color: var(--text-tertiary); font-size: 12px; }
    .mono { font-variant-numeric: tabular-nums; font-family: 'SF Mono', 'JetBrains Mono', monospace; font-size: 12px; }
    .sig { color: var(--brand); font-weight: 600; }
    .key-metric {
        display: inline-flex;
        align-items: baseline;
        gap: 12px;
        padding: 16px 24px;
        background: var(--bg-secondary);
        border-radius: var(--radius);
        margin-bottom: 12px;
    }
    .key-metric .metric-label {
        font-size: 12px;
        color: var(--text-tertiary);
    }
    .key-metric .metric-value {
        font-size: 28px;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
    }
    .finding-text {
        font-size: 13px;
        color: var(--text-secondary);
        margin-bottom: 16px;
    }
    .chart-img {
        width: 100%;
        max-width: 720px;
        height: auto;
        margin: 16px 0;
        border: 1px solid var(--border-light);
        border-radius: var(--radius);
    }
    .conclusion-list, .suggestion-list {
        padding-left: 20px;
        margin-bottom: 16px;
    }
    .conclusion-list li, .suggestion-list li {
        font-size: 13px;
        color: var(--text-secondary);
        margin-bottom: 8px;
        line-height: 1.7;
    }
    .note {
        font-size: 11px;
        color: var(--text-tertiary);
        margin-top: 8px;
    }
    .report-footer {
        margin-top: 48px;
        padding-top: 24px;
        border-top: 1px solid var(--border-light);
        font-size: 11px;
        color: var(--text-tertiary);
        text-align: center;
    }
    @media print {
        .toolbar { display: none; }
        .report { padding: 0; max-width: none; }
        .chart-img { page-break-inside: avoid; }
        .report-section { page-break-inside: avoid; }
    }
</style>
</head>
<body>
<div class="toolbar">
    <button onclick="window.print()">打印 / 导出PDF</button>
</div>
<div class="report">
    <div class="report-header">
        <h1>${data.title}</h1>
        <p class="subtitle">${data.subtitle}</p>
        <div class="report-meta">
            <span>生成时间: ${data.generated_at}</span>
            <span>已分析条件: ${data.analyzed_conditions.length} 个</span>
            <span>条件列表: ${data.analyzed_conditions.join(', ')}</span>
        </div>
    </div>
    ${bodyHTML}
    <div class="report-footer">
        EEG 心流恢复分析工具箱 · 跨学科任务切换 EEG 恢复时间量化研究<br>
        本报告由工具箱自动生成，数据与分析结论供研究参考
    </div>
</div>
</body>
</html>`;
}

// ==========================================================
// 被试管理
// ==========================================================
async function loadSubjects() {
    try {
        const resp = await fetch('/api/subjects');
        const subjects = await resp.json();
        renderSubjectsTable(subjects);
    } catch (err) {
        document.getElementById('subjects-table-wrap').innerHTML = '<p class="block-desc">加载失败</p>';
    }
}

function renderSubjectsTable(subjects) {
    if (!subjects.length) {
        document.getElementById('subjects-table-wrap').innerHTML = '<p class="block-desc">暂无被试记录，点击右上角「添加被试」</p>';
        return;
    }
    const html = `
        <table class="stats-table">
            <thead><tr><th>编号</th><th>年龄</th><th>性别</th><th>创建时间</th><th>操作</th></tr></thead>
            <tbody>
                ${subjects.map(s => `
                    <tr>
                        <td><a href="#" onclick="loadSubjectExperiments(${s.id}, '${s.code}'); return false;">${s.code}</a></td>
                        <td>${s.age || '—'}</td>
                        <td>${s.gender || '—'}</td>
                        <td>${s.created_at}</td>
                        <td><button class="btn btn-ghost" style="padding:2px 8px;font-size:11px;" onclick="deleteSubject(${s.id})">删除</button></td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    document.getElementById('subjects-table-wrap').innerHTML = html;
}

async function loadSubjectExperiments(subjectId, subjectCode) {
    document.getElementById('subject-experiments-block').style.display = 'block';
    document.getElementById('subject-exp-title').textContent = `${subjectCode} 的实验记录`;
    try {
        const resp = await fetch(`/api/subjects/${subjectId}/experiments`);
        const exps = await resp.json();
        if (!exps.length) {
            document.getElementById('experiments-table-wrap').innerHTML = '<p class="block-desc">暂无实验记录</p>';
            return;
        }
        const html = `
            <table class="stats-table">
                <thead><tr><th>条件</th><th>日期</th><th>备注</th><th>创建时间</th></tr></thead>
                <tbody>
                    ${exps.map(e => `
                        <tr>
                            <td>${e.condition}</td>
                            <td>${e.date || '—'}</td>
                            <td>${e.notes || '—'}</td>
                            <td>${e.created_at}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        document.getElementById('experiments-table-wrap').innerHTML = html;
    } catch (err) {
        document.getElementById('experiments-table-wrap').innerHTML = '<p class="block-desc">加载失败</p>';
    }
}

async function deleteSubject(id) {
    if (!confirm('确认删除该被试？相关实验记录也将被删除。')) return;
    try {
        await fetchJSON(`/api/subjects/${id}`, { method: 'DELETE' });
        showToast('已删除');
        loadSubjects();
        document.getElementById('subject-experiments-block').style.display = 'none';
    } catch (err) {
        alert('删除失败: ' + err.message);
    }
}

// 添加被试按钮事件
document.addEventListener('DOMContentLoaded', () => {
    const addBtn = document.getElementById('subject-add-btn');
    if (addBtn) {
        addBtn.addEventListener('click', () => {
            const code = prompt('被试编号 (如 S001):');
            if (!code) return;
            const age = prompt('年龄 (可留空):') || null;
            const gender = prompt('性别 M/F (可留空):') || null;
            fetchJSON('/api/subjects', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code, age: age ? parseInt(age) : null, gender })
            }).then(() => {
                showToast('已添加');
                loadSubjects();
            }).catch(err => alert('添加失败: ' + err.message));
        });
    }
});

// ==========================================================
// 频谱分析
// ==========================================================
let spectrumCondition = 'AtoA';
let spectrumCharts = {};

document.addEventListener('DOMContentLoaded', () => {
    // 条件选择
    document.querySelectorAll('[data-scondition]').forEach(card => {
        card.addEventListener('click', () => {
            document.querySelectorAll('[data-scondition]').forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            spectrumCondition = card.dataset.scondition;
        });
    });
    // 标签页切换
    document.querySelectorAll('[data-stab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.stab;
            document.querySelectorAll('[data-stab]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('[id^="spectrum-panel-"]').forEach(p => p.classList.remove('active'));
            document.getElementById(`spectrum-panel-${tab}`).classList.add('active');
            document.getElementById('spectrum-run-sample').style.display = tab === 'sample' ? '' : 'none';
            document.getElementById('spectrum-run-upload').style.display = tab === 'upload' ? '' : 'none';
        });
    });
    // 运行分析
    document.getElementById('spectrum-run-sample')?.addEventListener('click', runSpectrumSample);
    document.getElementById('spectrum-run-upload')?.addEventListener('click', runSpectrumUpload);
    // 文件选择提示
    document.getElementById('spectrum-file')?.addEventListener('change', (e) => {
        document.getElementById('spectrum-file-hint').textContent = e.target.files[0]?.name || '未选择';
    });
});

async function runSpectrumSample() {
    showLoading(true);
    try {
        const nperseg = parseInt(document.getElementById('spectrum-nperseg').value);
        const overlap = parseFloat(document.getElementById('spectrum-overlap').value);
        const data = await fetchJSON('/api/spectrum/sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ condition: spectrumCondition, fs: 250, nperseg, overlap })
        });
        if (data.error) throw new Error(data.error);
        document.getElementById('spectrum-empty').style.display = 'none';
        document.getElementById('spectrum-content').style.display = 'flex';
        renderSpectrumCharts(data);
    } catch (err) {
        alert('分析失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

async function runSpectrumUpload() {
    const file = document.getElementById('spectrum-file').files[0];
    if (!file) { alert('请选择 EEG 文件'); return; }
    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('eeg_file', file);
        formData.append('fs', 250);
        formData.append('nperseg', document.getElementById('spectrum-nperseg').value);
        formData.append('overlap', document.getElementById('spectrum-overlap').value);
        const data = await fetchJSON('/api/spectrum/analyze', { method: 'POST', body: formData });
        if (data.error) throw new Error(data.error);
        document.getElementById('spectrum-empty').style.display = 'none';
        document.getElementById('spectrum-content').style.display = 'flex';
        renderSpectrumCharts(data);
    } catch (err) {
        alert('分析失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

function renderSpectrumCharts(data) {
    renderPSDChart(data.psd);
    renderBandsChart(data.band_powers);
    renderSpectrogramChart(data.spectrogram);
}

function renderPSDChart(psd) {
    const ctx = document.getElementById('chart-psd').getContext('2d');
    if (spectrumCharts.psd) spectrumCharts.psd.destroy();
    spectrumCharts.psd = new Chart(ctx, {
        type: 'line',
        data: {
            labels: psd.freqs,
            datasets: [{
                label: 'PSD',
                data: psd.psd,
                borderColor: '#4B3FE3',
                backgroundColor: 'rgba(75,63,227,0.08)',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: '频率 (Hz)' }, type: 'linear' },
                y: { title: { display: true, text: '功率谱密度 (μV²/Hz)' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function renderBandsChart(bandPowers) {
    const ctx = document.getElementById('chart-bands').getContext('2d');
    if (spectrumCharts.bands) spectrumCharts.bands.destroy();
    const labels = Object.keys(bandPowers);
    const values = labels.map(b => bandPowers[b].rel);
    const colors = ['#A1A1A1', '#4B3FE3', '#1DC981', '#22A5F7', '#F87454'];
    spectrumCharts.bands = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [{
                label: '相对功率 (%)',
                data: values,
                backgroundColor: colors,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { title: { display: true, text: '相对功率 (%)' } } },
            plugins: { legend: { display: false } }
        }
    });
}

function renderSpectrogramChart(spec) {
    const ctx = document.getElementById('chart-spectrogram').getContext('2d');
    if (spectrumCharts.spectrogram) spectrumCharts.spectrogram.destroy();
    // 时频图用矩阵热图方式渲染
    // 降采样：如果时间点太多，取间隔采样
    const maxTimePoints = 500;
    const step = Math.max(1, Math.floor(spec.times.length / maxTimePoints));
    const times = spec.times.filter((_, i) => i % step === 0);
    const data = spec.spectrogram.map(row => row.filter((_, i) => i % step === 0));

    // Chart.js 没有原生热图，用 contour 或自定义渲染
    // 这里用简化的方式：每个频段一行，用颜色矩阵
    const datasets = data.map((row, idx) => ({
        label: `${spec.freqs[idx].toFixed(1)}Hz`,
        data: row,
        borderColor: `hsl(${240 - idx * 5}, 70%, 50%)`,
        backgroundColor: 'transparent',
        borderWidth: 0.5,
        pointRadius: 0,
    }));
    spectrumCharts.spectrogram = new Chart(ctx, {
        type: 'line',
        data: { labels: times.map(t => t.toFixed(1)), datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: '时间 (s)' } },
                y: { title: { display: true, text: '功率 (dB)' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// ==========================================================
// 事件相关电位 (ERP)
// ==========================================================
let erpCondition = 'AtoA';
let erpEventId = 'X0';
let erpCharts = {};
let erpLastData = null;

document.addEventListener('DOMContentLoaded', () => {
    // 源数据标签切换
    document.querySelectorAll('[data-etab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.etab;
            document.querySelectorAll('[data-etab]').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('#view-erp .source-tabs + .tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`erp-panel-${tab}`).classList.add('active');
            document.getElementById('erp-run-sample').style.display = tab === 'sample' ? '' : 'none';
            document.getElementById('erp-run-upload').style.display = tab === 'upload' ? '' : 'none';
        });
    });
    // 结果区标签切换
    document.querySelectorAll('[data-ertab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.ertab;
            document.querySelectorAll('[data-ertab]').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('#view-erp .tab-panel-block').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.querySelector(`[data-ertab-panel="${tab}"]`).classList.add('active');
            // 切换后 resize 图表 (Chart.js 在 display:none 时尺寸为 0)
            if (tab === 'waveform' && erpCharts.waveform) erpCharts.waveform.resize();
            if (tab === 'diff' && erpCharts.diff) erpCharts.diff.resize();
        });
    });
    // 条件/事件选择
    document.getElementById('erp-condition')?.addEventListener('change', e => erpCondition = e.target.value);
    document.getElementById('erp-event-id')?.addEventListener('change', e => erpEventId = e.target.value);
    // 运行按钮
    document.getElementById('erp-run-sample')?.addEventListener('click', runERPSample);
    document.getElementById('erp-run-header')?.addEventListener('click', runERPSample);
    document.getElementById('erp-run-upload')?.addEventListener('click', runERPUpload);
    document.getElementById('erp-run-diff')?.addEventListener('click', runERPCompare);
    // 文件选择提示
    document.getElementById('erp-file-eeg')?.addEventListener('change', e => {
        document.getElementById('erp-hint-eeg').textContent = e.target.files[0]?.name || '未选择';
    });
    document.getElementById('erp-file-events')?.addEventListener('change', e => {
        document.getElementById('erp-hint-events').textContent = e.target.files[0]?.name || '未选择';
    });
});

async function runERPSample() {
    showLoading(true);
    try {
        const data = await fetchJSON('/api/erp/sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ condition: erpCondition, fs: 250, event_id: erpEventId })
        });
        if (data.error) throw new Error(data.error);
        erpLastData = data;
        document.getElementById('erp-empty').style.display = 'none';
        document.getElementById('erp-content').style.display = 'flex';
        renderERPChart(data);
        renderERPComponentsTable(data);
    } catch (err) {
        alert('ERP 分析失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

async function runERPUpload() {
    const eegFile = document.getElementById('erp-file-eeg').files[0];
    if (!eegFile) { alert('请选择 EEG 文件'); return; }
    const evtFile = document.getElementById('erp-file-events').files[0];
    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('eeg_file', eegFile);
        if (evtFile) formData.append('events_file', evtFile);
        formData.append('condition', 'custom');
        formData.append('event_id', erpEventId);
        formData.append('fs', 250);
        const resp = await fetch('/api/erp/analyze', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        erpLastData = data;
        document.getElementById('erp-empty').style.display = 'none';
        document.getElementById('erp-content').style.display = 'flex';
        renderERPChart(data);
        renderERPComponentsTable(data);
    } catch (err) {
        alert('ERP 分析失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

async function runERPCompare() {
    const c1 = document.getElementById('erp-diff-c1').value;
    const c2 = document.getElementById('erp-diff-c2').value;
    if (c1 === c2) { showToast('请选择两个不同条件'); return; }
    showLoading(true);
    try {
        const resp = await fetch('/api/erp/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conditions: [c1, c2], fs: 250, event_id: erpEventId })
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        renderERPDiffChart(data, c1, c2);
    } catch (err) {
        alert('差值波计算失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

function renderERPChart(data) {
    const ctx = document.getElementById('chart-erp-waveform').getContext('2d');
    if (erpCharts.waveform) erpCharts.waveform.destroy();
    const avg = data.averaged;
    const channels = avg.channels || ['Fp1', 'Fp2', 'Fpz'];
    const colors = { Fp1: '#4B3FE3', Fp2: '#1DC981', Fpz: '#F87454' };

    // 峰值标注插件
    const peakPlugin = {
        id: 'erpPeaks',
        afterDraw(chart) {
            const { ctx, scales } = chart;
            const comps = data.components || [];
            // 找出 N100 与 P300 的潜伏期
            const n100 = comps.find(c => /N100|N1/i.test(c.name || c.component || ''));
            const p300 = comps.find(c => /P300|P3/i.test(c.name || c.component || ''));
            ctx.font = '11px ' + getComputedStyle(document.body).fontFamily;
            if (n100 && n100.latency !== undefined) {
                const x = scales.x.getPixelForValue(n100.latency);
                ctx.strokeStyle = 'rgba(75,63,227,0.5)';
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(x, chart.chartArea.top);
                ctx.lineTo(x, chart.chartArea.bottom);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = '#4B3FE3';
                ctx.textAlign = 'center';
                ctx.fillText('N100', x, chart.chartArea.top + 12);
            }
            if (p300 && p300.latency !== undefined) {
                const x = scales.x.getPixelForValue(p300.latency);
                ctx.strokeStyle = 'rgba(248,116,84,0.5)';
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(x, chart.chartArea.top);
                ctx.lineTo(x, chart.chartArea.bottom);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = '#F87454';
                ctx.textAlign = 'center';
                ctx.fillText('P300', x, chart.chartArea.top + 12);
            }
        }
    };

    const datasets = channels.map(ch => ({
        label: ch,
        data: avg.waveform[ch] || [],
        borderColor: colors[ch] || '#737373',
        backgroundColor: 'transparent',
        borderWidth: 1.8,
        pointRadius: 0,
        tension: 0.2,
    }));

    erpCharts.waveform = new Chart(ctx, {
        type: 'line',
        data: { labels: avg.times, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'bottom', labels: { color: '#404040', font: { size: 11 }, boxWidth: 12, padding: 12 } },
                tooltip: {
                    backgroundColor: '#FFFFFF', titleColor: '#171717', bodyColor: '#404040',
                    borderColor: 'rgba(115,115,115,0.18)', borderWidth: 1, cornerRadius: 8,
                    callbacks: {
                        title: (items) => `时间: ${parseFloat(items[0].label).toFixed(0)} ms`,
                        label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(2)} μV`,
                    },
                },
            },
            scales: {
                x: { type: 'linear', title: { display: true, text: '时间 (ms)', color: '#737373', font: { size: 12 } },
                    grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373', font: { size: 11 } } },
                y: { title: { display: true, text: '幅度 (μV)', color: '#737373', font: { size: 12 } },
                    grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373', font: { size: 11 } } },
            },
        },
        plugins: [peakPlugin],
    });
}

function renderERPComponentsTable(data) {
    const wrap = document.getElementById('erp-components-table');
    const comps = data.components || [];
    if (!comps.length) {
        wrap.innerHTML = '<p class="block-desc">未识别到典型成分</p>';
        document.getElementById('erp-components-note').textContent = '';
        return;
    }
    let html = '<table class="stats-table"><thead><tr><th>成分</th><th>潜伏期 (ms)</th><th>幅度 (μV)</th><th>描述</th></tr></thead><tbody>';
    comps.forEach(c => {
        const name = c.name || c.component || '—';
        const lat = c.latency !== undefined ? c.latency.toFixed(0) : '—';
        const amp = c.amplitude !== undefined ? c.amplitude.toFixed(2) : '—';
        const desc = c.description || c.desc || '—';
        html += `<tr><td>${name}</td><td class="mono">${lat}</td><td class="mono">${amp}</td><td class="desc" style="color:#737373;font-size:12px;">${desc}</td></tr>`;
    });
    html += '</tbody></table>';
    wrap.innerHTML = html;
    const note = [];
    if (data.peak_to_peak !== undefined) note.push(`峰峰值: ${data.peak_to_peak.toFixed(2)} μV`);
    if (data.rmse !== undefined) note.push(`RMSE: ${data.rmse.toFixed(3)}`);
    document.getElementById('erp-components-note').textContent = note.join(' · ');
}

function renderERPDiffChart(data, c1, c2) {
    const ctx = document.getElementById('chart-erp-diff').getContext('2d');
    if (erpCharts.diff) erpCharts.diff.destroy();
    const diff = data.difference || data.diff || data;
    const times = diff.times || data.times || [];
    const waveform = diff.waveform || diff.values || [];
    erpCharts.diff = new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [{
                label: `${CONDITION_LABELS[c1] || c1} − ${CONDITION_LABELS[c2] || c2}`,
                data: waveform,
                borderColor: '#4B3FE3',
                backgroundColor: 'rgba(75,63,227,0.08)',
                borderWidth: 1.8,
                pointRadius: 0,
                fill: { target: 'origin', above: 'rgba(75,63,227,0.10)', below: 'rgba(248,116,84,0.10)' },
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: '#FFFFFF', titleColor: '#171717', bodyColor: '#404040',
                    borderColor: 'rgba(115,115,115,0.18)', borderWidth: 1, cornerRadius: 8,
                    callbacks: { title: (items) => `时间: ${parseFloat(items[0].label).toFixed(0)} ms`,
                        label: (item) => `差值: ${item.parsed.y.toFixed(2)} μV` } },
            },
            scales: {
                x: { type: 'linear', title: { display: true, text: '时间 (ms)', color: '#737373' },
                    grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373' } },
                y: { title: { display: true, text: '差值 (μV)', color: '#737373' },
                    grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373' } },
            },
        }
    });
}

// ==========================================================
// ERSP 时频分析
// ==========================================================
let erspCondition = 'AtoA';
let erspEventId = 'X0';
let erspCharts = {};

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-rtab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.rtab;
            document.querySelectorAll('[data-rtab]').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('#view-ersp .source-tabs + .tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`ersp-panel-${tab}`).classList.add('active');
            document.getElementById('ersp-run-sample').style.display = tab === 'sample' ? '' : 'none';
            document.getElementById('ersp-run-upload').style.display = tab === 'upload' ? '' : 'none';
        });
    });
    document.querySelectorAll('[data-rrtab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.rrtab;
            document.querySelectorAll('[data-rrtab]').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('#view-ersp .tab-panel-block').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.querySelector(`[data-rrtab-panel="${tab}"]`).classList.add('active');
            if (tab === 'erd' && erspCharts.erd) erspCharts.erd.resize();
            if (tab === 'pac' && erspCharts.pacMi) erspCharts.pacMi.resize();
            if (tab === 'pac' && erspCharts.pacDist) erspCharts.pacDist.resize();
        });
    });
    document.getElementById('ersp-condition')?.addEventListener('change', e => erspCondition = e.target.value);
    document.getElementById('ersp-event-id')?.addEventListener('change', e => erspEventId = e.target.value);
    document.getElementById('ersp-run-sample')?.addEventListener('click', runERSPSample);
    document.getElementById('ersp-run-header')?.addEventListener('click', runERSPSample);
    document.getElementById('ersp-run-upload')?.addEventListener('click', runERSPUpload);
    document.getElementById('ersp-file-eeg')?.addEventListener('change', e => {
        document.getElementById('ersp-hint-eeg').textContent = e.target.files[0]?.name || '未选择';
    });
    document.getElementById('ersp-file-events')?.addEventListener('change', e => {
        document.getElementById('ersp-hint-events').textContent = e.target.files[0]?.name || '未选择';
    });
});

async function runERSPSample() {
    showLoading(true);
    try {
        const data = await fetchJSON('/api/ersp/sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ condition: erspCondition, fs: 250, event_id: erspEventId })
        });
        if (data.error) throw new Error(data.error);
        document.getElementById('ersp-empty').style.display = 'none';
        document.getElementById('ersp-content').style.display = 'flex';
        renderERSPResults(data);
    } catch (err) {
        alert('ERSP 分析失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

async function runERSPUpload() {
    const eegFile = document.getElementById('ersp-file-eeg').files[0];
    if (!eegFile) { alert('请选择 EEG 文件'); return; }
    const evtFile = document.getElementById('ersp-file-events').files[0];
    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('eeg_file', eegFile);
        if (evtFile) formData.append('events_file', evtFile);
        formData.append('condition', 'custom');
        formData.append('event_id', erspEventId);
        formData.append('fs', 250);
        const resp = await fetch('/api/ersp/analyze', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        document.getElementById('ersp-empty').style.display = 'none';
        document.getElementById('ersp-content').style.display = 'flex';
        renderERSPResults(data);
    } catch (err) {
        alert('ERSP 分析失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

function renderERSPResults(data) {
    const ersp = data.ersp || {};
    renderHeatmap('canvas-ersp', ersp.times || [], ersp.freqs || [], ersp.ersp || [], 'diverging', '功率变化 (dB)');
    if (data.erd_ers) renderERDERSChart(data.erd_ers);
    if (data.pac) renderPACCharts(data.pac);
    const itpc = data.itpc || {};
    renderHeatmap('canvas-itpc', itpc.times || ersp.times || [], itpc.freqs || ersp.freqs || [], itpc.itpc || [], 'sequential', 'ITPC (0-1)');
}

function renderERDERSChart(erdErs) {
    const ctx = document.getElementById('chart-erd-ers').getContext('2d');
    if (erspCharts.erd) erspCharts.erd.destroy();
    const labels = Object.keys(erdErs);
    const erdVals = labels.map(b => erdErs[b].erd || 0);
    const ersVals = labels.map(b => erdErs[b].ers || 0);
    erspCharts.erd = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [
                { label: 'ERD (%)', data: erdVals, backgroundColor: '#22A5F7CC', borderColor: '#22A5F7', borderWidth: 1, borderRadius: 4 },
                { label: 'ERS (%)', data: ersVals, backgroundColor: '#F87454CC', borderColor: '#F87454', borderWidth: 1, borderRadius: 4 },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom', labels: { color: '#404040', font: { size: 11 }, boxWidth: 12, padding: 12 } } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#404040', font: { size: 11 } } },
                y: { title: { display: true, text: '占比 (%)', color: '#737373' }, grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373' } },
            },
        }
    });
}

function renderPACCharts(pac) {
    // 调制指数柱状图
    const ctxMi = document.getElementById('chart-pac-mi').getContext('2d');
    if (erspCharts.pacMi) erspCharts.pacMi.destroy();
    const mi = pac.mi || {};
    const miLabels = Object.keys(mi);
    erspCharts.pacMi = new Chart(ctxMi, {
        type: 'bar',
        data: {
            labels: miLabels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [{
                label: '调制指数 MI',
                data: miLabels.map(l => mi[l]),
                backgroundColor: '#B655FCCC', borderColor: '#B655FC', borderWidth: 1, borderRadius: 4,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#404040', font: { size: 11 } } },
                y: { title: { display: true, text: 'MI', color: '#737373' }, grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373' } },
            },
        }
    });
    // 相位-振幅分布线图
    const ctxDist = document.getElementById('chart-pac-dist').getContext('2d');
    if (erspCharts.pacDist) erspCharts.pacDist.destroy();
    const dist = pac.phase_amp_dist || [];
    const phaseBins = dist.map((_, i) => ((i / dist.length) * 360).toFixed(0));
    erspCharts.pacDist = new Chart(ctxDist, {
        type: 'line',
        data: {
            labels: phaseBins,
            datasets: [{
                label: '振幅均值',
                data: dist,
                borderColor: '#4B3FE3',
                backgroundColor: 'rgba(75,63,227,0.08)',
                borderWidth: 1.8, pointRadius: 0, fill: true, tension: 0.3,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: '相位 (°)', color: '#737373' }, grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373', font: { size: 11 } } },
                y: { title: { display: true, text: '高频振幅', color: '#737373' }, grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373' } },
            },
        }
    });
}

// ==========================================================
// 伪迹检测
// ==========================================================
let artifactCondition = 'AtoA';
let artifactCharts = {};

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-atab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.atab;
            document.querySelectorAll('[data-atab]').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('#view-artifact .source-tabs + .tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`artifact-panel-${tab}`).classList.add('active');
            document.getElementById('artifact-run-sample').style.display = tab === 'sample' ? '' : 'none';
            document.getElementById('artifact-run-upload').style.display = tab === 'upload' ? '' : 'none';
        });
    });
    document.querySelectorAll('[data-arttab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.arttab;
            document.querySelectorAll('[data-arttab]').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('#view-artifact .tab-panel-block').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.querySelector(`[data-arttab-panel="${tab}"]`).classList.add('active');
            if (tab === 'stats' && artifactCharts.methods) artifactCharts.methods.resize();
            if (tab === 'ica' && artifactCharts.icaPower) artifactCharts.icaPower.resize();
        });
    });
    document.getElementById('artifact-condition')?.addEventListener('change', e => artifactCondition = e.target.value);
    document.getElementById('artifact-run-sample')?.addEventListener('click', runArtifactSample);
    document.getElementById('artifact-run-header')?.addEventListener('click', runArtifactSample);
    document.getElementById('artifact-run-upload')?.addEventListener('click', runArtifactUpload);
    document.getElementById('artifact-file-eeg')?.addEventListener('change', e => {
        document.getElementById('artifact-hint-eeg').textContent = e.target.files[0]?.name || '未选择';
    });
});

async function runArtifactSample() {
    showLoading(true);
    try {
        const data = await fetchJSON('/api/artifact/sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ condition: artifactCondition, fs: 250 })
        });
        if (data.error) throw new Error(data.error);
        document.getElementById('artifact-empty').style.display = 'none';
        document.getElementById('artifact-content').style.display = 'flex';
        renderArtifactResults(data);
    } catch (err) {
        alert('伪迹检测失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

async function runArtifactUpload() {
    const eegFile = document.getElementById('artifact-file-eeg').files[0];
    if (!eegFile) { alert('请选择 EEG 文件'); return; }
    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('eeg_file', eegFile);
        formData.append('condition', 'custom');
        formData.append('fs', 250);
        const resp = await fetch('/api/artifact/analyze', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        document.getElementById('artifact-empty').style.display = 'none';
        document.getElementById('artifact-content').style.display = 'flex';
        renderArtifactResults(data);
    } catch (err) {
        alert('伪迹检测失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

function renderArtifactResults(data) {
    // 质量评分卡片
    const q = data.quality || {};
    renderQualityScore(q.score || 0, q.grade || '—', q.factors || {});
    // 检测统计
    const th = data.threshold_detection || {};
    const zs = data.zscore_detection || {};
    renderArtifactMethodsChart(th, zs);
    renderArtifactStatsTable(th, zs);
    // ICA
    const ica = data.ica_result || {};
    renderICATable(ica.classification || []);
    renderICAPowerChart(ica.classification || []);
    // 信号特征
    renderSignalStatsTable(data.signal_stats || {});
}

function renderQualityScore(score, grade, factors) {
    document.getElementById('quality-score').textContent = score.toFixed(1);
    const gradeEl = document.getElementById('quality-grade');
    gradeEl.textContent = grade;
    const gradeColors = { A: '#1DC981', B: '#22A5F7', C: '#EDAA45', D: '#F87454' };
    gradeEl.style.color = gradeColors[grade] || '#737373';
    const wrap = document.getElementById('quality-factors');
    const keys = Object.keys(factors);
    if (!keys.length) {
        wrap.innerHTML = '<p class="block-desc">无评分因素</p>';
        return;
    }
    wrap.innerHTML = keys.map(k => {
        const v = factors[k];
        const val = typeof v === 'number' ? v.toFixed(2) : v;
        return `<div class="quality-factor-item"><span class="quality-factor-label">${k}</span><span class="quality-factor-value">${val}</span></div>`;
    }).join('');
}

function renderArtifactMethodsChart(th, zs) {
    const ctx = document.getElementById('chart-artifact-methods').getContext('2d');
    if (artifactCharts.methods) artifactCharts.methods.destroy();
    artifactCharts.methods = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['伪迹比例 (%)', '伪迹段数'],
            datasets: [
                { label: '阈值法', data: [(th.ratio || 0) * 100, th.segments || 0], backgroundColor: '#4B3FE3CC', borderColor: '#4B3FE3', borderWidth: 1, borderRadius: 4 },
                { label: 'Z-score 法', data: [(zs.ratio || 0) * 100, zs.segments || 0], backgroundColor: '#EDAA45CC', borderColor: '#EDAA45', borderWidth: 1, borderRadius: 4 },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom', labels: { color: '#404040', font: { size: 11 }, boxWidth: 12, padding: 12 } } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#404040', font: { size: 11 } } },
                y: { grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373' } },
            },
        }
    });
}

function renderArtifactStatsTable(th, zs) {
    const wrap = document.getElementById('artifact-stats-table');
    const rows = [
        ['伪迹比例', ((th.ratio || 0) * 100).toFixed(2) + ' %', ((zs.ratio || 0) * 100).toFixed(2) + ' %'],
        ['伪迹段数', (th.segments || 0).toString(), (zs.segments || 0).toString()],
        ['阈值/参数', (th.threshold !== undefined ? th.threshold.toFixed(2) : '—') + ' μV', 'z = ' + (zs.z_threshold !== undefined ? zs.z_threshold.toFixed(2) : '3.0')],
    ];
    let html = '<table class="stats-table"><thead><tr><th>指标</th><th>阈值法</th><th>Z-score 法</th></tr></thead><tbody>';
    rows.forEach(r => { html += `<tr><td>${r[0]}</td><td class="mono">${r[1]}</td><td class="mono">${r[2]}</td></tr>`; });
    html += '</tbody></table>';
    wrap.innerHTML = html;
}

function renderICATable(classification) {
    const wrap = document.getElementById('ica-table-wrap');
    if (!Array.isArray(classification) || !classification.length) {
        wrap.innerHTML = '<p class="block-desc">未提供 ICA 分类结果</p>';
        return;
    }
    let html = '<table class="stats-table"><thead><tr><th>成分</th><th>类型</th><th>置信度</th><th>原因</th></tr></thead><tbody>';
    classification.forEach(c => {
        const idx = c.index !== undefined ? c.index : (c.component !== undefined ? c.component : '—');
        const type = c.type || c.label || '—';
        const conf = c.confidence !== undefined ? (c.confidence * 100).toFixed(1) + '%' : '—';
        const reason = c.reason || c.description || '—';
        const typeColor = { eye: '#22A5F7', muscle: '#F87454', heart: '#EDAA45', line: '#A1A1A1', brain: '#1DC981' }[type.toLowerCase()] || '#737373';
        html += `<tr><td class="mono">#${idx}</td><td><span style="color:${typeColor};font-weight:500">${type}</span></td><td class="mono">${conf}</td><td style="color:#737373;font-size:12px">${reason}</td></tr>`;
    });
    html += '</tbody></table>';
    wrap.innerHTML = html;
}

function renderICAPowerChart(classification) {
    const ctx = document.getElementById('chart-ica-power').getContext('2d');
    if (artifactCharts.icaPower) artifactCharts.icaPower.destroy();
    if (!Array.isArray(classification) || !classification.length) {
        artifactCharts.icaPower = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
        return;
    }
    const labels = classification.map(c => '#' + (c.index !== undefined ? c.index : c.component || '?'));
    const values = classification.map(c => (c.variance_explained || c.power_ratio || 0) * 100);
    artifactCharts.icaPower = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{ label: '相对功率 (%)', data: values, backgroundColor: '#4B3FE3CC', borderColor: '#4B3FE3', borderWidth: 1, borderRadius: 4 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#404040', font: { size: 11 } } },
                y: { title: { display: true, text: '相对功率 (%)', color: '#737373' }, grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373' } },
            },
        }
    });
}

function renderSignalStatsTable(signalStats) {
    const wrap = document.getElementById('signal-stats-table');
    const channels = Object.keys(signalStats);
    if (!channels.length) {
        wrap.innerHTML = '<p class="block-desc">无信号特征数据</p>';
        return;
    }
    const metrics = ['peak', 'rms', 'variance', 'kurtosis', 'skewness', 'snr_db'];
    const metricNames = { peak: 'Peak (μV)', rms: 'RMS (μV)', variance: 'Variance', kurtosis: 'Kurtosis', skewness: 'Skewness', snr_db: 'SNR (dB)' };
    let html = '<table class="stats-table"><thead><tr><th>通道</th>' + metrics.map(m => `<th>${metricNames[m]}</th>`).join('') + '</tr></thead><tbody>';
    channels.forEach(ch => {
        const s = signalStats[ch] || {};
        html += `<tr><td>${ch}</td>` + metrics.map(m => {
            const v = s[m];
            return `<td class="mono">${v !== undefined ? (typeof v === 'number' ? v.toFixed(3) : v) : '—'}</td>`;
        }).join('') + '</tr>';
    });
    html += '</tbody></table>';
    wrap.innerHTML = html;
}

// ==========================================================
// 统计可视化
// ==========================================================
let statsVizCharts = {};
let statsVizSummary = null;
let statsVizProfile = null;

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-svtab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.svtab;
            document.querySelectorAll('[data-svtab]').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('#view-stats-viz .tab-panel-block').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.querySelector(`[data-svtab-panel="${tab}"]`).classList.add('active');
            if (tab === 'radar' && statsVizCharts.radar) statsVizCharts.radar.resize();
            if (tab === 'cross' && statsVizCharts.cross) statsVizCharts.cross.resize();
        });
    });
    document.getElementById('stats-viz-refresh')?.addEventListener('click', refreshStatsViz);
    document.getElementById('stats-viz-refresh-header')?.addEventListener('click', refreshStatsViz);
    document.getElementById('stats-viz-export')?.addEventListener('click', exportStatsViz);
    document.getElementById('stats-viz-export-header')?.addEventListener('click', exportStatsViz);
    document.getElementById('topomap-render')?.addEventListener('click', renderTopomapFromConfig);
});

async function refreshStatsViz() {
    showLoading(true);
    try {
        const [sumResp, profResp] = await Promise.all([
            fetch('/api/stats-viz/summary'),
            fetch('/api/stats-viz/indicator-profile'),
        ]);
        const sum = await sumResp.json();
        const prof = await profResp.json();
        if (sum.error) throw new Error(sum.error);
        if (prof.error) throw new Error(prof.error);
        statsVizSummary = sum;
        statsVizProfile = prof;
        document.getElementById('stats-viz-empty').style.display = 'none';
        document.getElementById('stats-viz-content').style.display = 'flex';
        renderSummaryTable(sum);
        renderRadarChart(prof);
        renderCrossSubjectChart(sum);
        showToast('数据已刷新');
    } catch (err) {
        alert('刷新失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

function renderSummaryTable(sum) {
    const wrap = document.getElementById('summary-table-wrap');
    const headers = sum.headers || [];
    const rows = sum.rows || [];
    if (!headers.length || !rows.length) {
        wrap.innerHTML = '<p class="block-desc">暂无已分析结果，请先在「心流恢复分析」中运行至少一个条件</p>';
        return;
    }
    let html = '<table class="stats-table"><thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>';
    rows.forEach(r => {
        html += '<tr>' + r.map(v => {
            const isNum = typeof v === 'number';
            return `<td class="${isNum ? 'mono' : ''}">${isNum ? v.toFixed(2) : (v === null || v === undefined ? '—' : v)}</td>`;
        }).join('') + '</tr>';
    });
    html += '</tbody></table>';
    wrap.innerHTML = html;
}

function renderRadarChart(prof) {
    const ctx = document.getElementById('chart-radar').getContext('2d');
    if (statsVizCharts.radar) statsVizCharts.radar.destroy();
    const axes = prof.axes || [];
    const conds = prof.conditions || [];
    const values = prof.values || {};
    const palette = ['#4B3FE3', '#1DC981', '#22A5F7', '#F87454'];
    const datasets = conds.map((c, i) => ({
        label: CONDITION_LABELS[c] || c,
        data: axes.map(a => values[c]?.[a] ?? 0),
        borderColor: palette[i % palette.length],
        backgroundColor: palette[i % palette.length] + '20',
        borderWidth: 1.8,
        pointBackgroundColor: palette[i % palette.length],
        pointRadius: 3,
    }));
    statsVizCharts.radar = new Chart(ctx, {
        type: 'radar',
        data: { labels: axes, datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom', labels: { color: '#404040', font: { size: 11 }, boxWidth: 12, padding: 12 } } },
            scales: {
                r: {
                    min: 0, max: 1,
                    grid: { color: 'rgba(115,115,115,0.18)' },
                    angleLines: { color: 'rgba(115,115,115,0.18)' },
                    pointLabels: { color: '#404040', font: { size: 11 } },
                    ticks: { color: '#737373', backdropColor: 'transparent', font: { size: 10 } },
                },
            },
        }
    });
}

async function renderTopomapFromConfig() {
    const indicator = document.getElementById('topomap-indicator').value;
    const condition = document.getElementById('topomap-condition').value;
    // 从汇总数据中尝试取出 3 通道的指标值
    let values = [0.5, 0.5, 0.5];
    if (statsVizSummary) {
        const headers = statsVizSummary.headers || [];
        const rows = statsVizSummary.rows || [];
        const condRow = rows.find(r => r[0] === condition);
        if (condRow) {
            const idx = headers.indexOf(indicator);
            if (idx > 0) {
                const v = condRow[idx];
                if (typeof v === 'number') values = [v, v * 0.95, v * 1.05];
            }
        }
    }
    showLoading(true);
    try {
        const resp = await fetch('/api/stats-viz/topomap', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ values, channel_names: ['Fp1', 'Fp2', 'Fpz'] })
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        renderTopomapOnCanvas('canvas-topomap', data, values, `${CONDITION_LABELS[condition] || condition} · ${indicator}`);
    } catch (err) {
        alert('地形图渲染失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

function renderTopomapOnCanvas(canvasId, gridData, values, title) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    const radius = Math.min(w, h) / 2 - 30;

    // 通道位置 (Fp1 左, Fp2 右, Fpz 中下前)
    const channels = [
        { name: 'Fp1', x: cx - radius * 0.45, y: cy - radius * 0.55, v: values[0] },
        { name: 'Fp2', x: cx + radius * 0.45, y: cy - radius * 0.55, v: values[1] },
        { name: 'Fpz', x: cx, y: cy - radius * 0.78, v: values[2] },
    ];

    // 找最大最小用于色映射
    const vMin = Math.min(...values);
    const vMax = Math.max(...values);
    const vRange = Math.max(Math.abs(vMax - vMin), 0.001);

    // 在网格上插值绘制
    const gx = gridData.grid_x || [];
    const gy = gridData.grid_y || [];
    const gz = gridData.grid_z || [];
    if (gx.length && gz.length) {
        const gMin = Math.min(...gz.flat ? gz.flat() : gz);
        const gMax = Math.max(...gz.flat ? gz.flat() : gz);
        const gRange = Math.max(Math.abs(gMax - gMin), 0.001);
        // 网格已归一化到 [-1, 1]，需映射到画布
        for (let i = 0; i < gx.length; i++) {
            for (let j = 0; j < (Array.isArray(gz[i]) ? gz[i].length : 0); j++) {
                const nx = gx[i];
                const ny = Array.isArray(gy[i]) ? gy[i][j] : gy[i];
                const val = Array.isArray(gz[i]) ? gz[i][j] : gz[j * gx.length + i];
                const px = cx + nx * radius;
                const py = cy - ny * radius;  // Y 翻转 (画布 Y 向下)
                if (Math.sqrt((px - cx) ** 2 + (py - cy) ** 2) > radius) continue;
                const t = (val - gMin) / gRange;
                ctx.fillStyle = interpolateColor(t, 'sequential');
                ctx.fillRect(px - 3, py - 3, 6, 6);
            }
        }
    } else {
        // 后备：用通道值径向插值
        for (let dx = -radius; dx <= radius; dx += 3) {
            for (let dy = -radius; dy <= radius; dy += 3) {
                if (dx * dx + dy * dy > radius * radius) continue;
                let num = 0, den = 0;
                channels.forEach(ch => {
                    const d = Math.sqrt((dx + cx - ch.x) ** 2 + (dy + cy - ch.y) ** 2) + 1;
                    num += ch.v / d;
                    den += 1 / d;
                });
                const v = num / den;
                const t = (v - vMin) / vRange;
                ctx.fillStyle = interpolateColor(t, 'sequential');
                ctx.fillRect(cx + dx - 1, cy + dy - 1, 3, 3);
            }
        }
    }

    // 头圆
    ctx.strokeStyle = '#404040';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();
    // 鼻子
    ctx.beginPath();
    ctx.moveTo(cx - 12, cy - radius);
    ctx.lineTo(cx, cy - radius - 16);
    ctx.lineTo(cx + 12, cy - radius);
    ctx.stroke();
    // 耳朵
    ctx.beginPath();
    ctx.arc(cx - radius, cy, 8, -Math.PI / 2, Math.PI / 2);
    ctx.arc(cx + radius, cy, 8, Math.PI / 2, -Math.PI / 2);
    ctx.stroke();
    // 通道点
    channels.forEach(ch => {
        ctx.fillStyle = '#262626';
        ctx.beginPath();
        ctx.arc(ch.x, ch.y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#404040';
        ctx.font = '11px ' + getComputedStyle(document.body).fontFamily;
        ctx.textAlign = 'center';
        ctx.fillText(ch.name, ch.x, ch.y - 8);
    });
    // 标题
    ctx.fillStyle = '#262626';
    ctx.font = '13px ' + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = 'center';
    ctx.fillText(title, cx, h - 8);
}

function renderCrossSubjectChart(sum) {
    const ctx = document.getElementById('chart-cross-subject').getContext('2d');
    if (statsVizCharts.cross) statsVizCharts.cross.destroy();
    const wrap = document.getElementById('cross-stats-table');
    const conds = sum.conditions || [];
    const headers = sum.headers || [];
    const rows = sum.rows || [];
    if (!conds.length || conds.length < 2) {
        wrap.innerHTML = '<p class="block-desc">单被试模式，跨被试统计需要多被试数据</p>';
        statsVizCharts.cross = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
        return;
    }
    const labels = conds.map(c => CONDITION_LABELS[c] || c);
    const recoveryIdx = headers.indexOf('recovery_time');
    const values = conds.map(c => {
        const row = rows.find(r => r[0] === c);
        if (row && recoveryIdx > 0) {
            const v = row[recoveryIdx];
            return typeof v === 'number' ? v : 600;
        }
        return 600;
    });
    statsVizCharts.cross = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{ label: '恢复时长 (s)', data: values, backgroundColor: '#4B3FE3CC', borderColor: '#4B3FE3', borderWidth: 1, borderRadius: 4 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#404040', font: { size: 11 } } },
                y: { title: { display: true, text: '恢复时长 (秒)', color: '#737373' }, grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373' } },
            },
        }
    });
    wrap.innerHTML = `<p class="block-desc">当前为汇总数据；接入多被试数据后将显示均值 ± 95% CI</p>`;
}

async function exportStatsViz() {
    if (!statsVizSummary) {
        showToast('请先刷新数据');
        return;
    }
    try {
        const resp = await fetch('/api/stats-viz/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: statsVizSummary, export_type: 'summary' })
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        // 触发下载
        const blob = new Blob([data.csv || ''], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = data.filename || 'stats_summary.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast('已导出 ' + (data.filename || 'CSV'));
    } catch (err) {
        alert('导出失败: ' + err.message);
    }
}

// ==========================================================
// Canvas 热力图渲染 (ERSP / ITPC)
// ==========================================================
function renderHeatmap(canvasId, times, freqs, matrix, colorScale, colorbarLabel) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (!times.length || !freqs.length || !matrix.length) {
        ctx.fillStyle = '#A1A1A1';
        ctx.font = '12px ' + getComputedStyle(document.body).fontFamily;
        ctx.textAlign = 'center';
        ctx.fillText('无数据', w / 2, h / 2);
        return;
    }

    const nFreq = freqs.length;
    const nTime = times.length;
    // 留出边距绘制坐标轴与颜色条
    const padL = 56, padR = 70, padT = 16, padB = 32;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const cellW = plotW / nTime;
    const cellH = plotH / nFreq;

    // 找 min/max
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < nFreq; i++) {
        const row = matrix[i] || [];
        for (let j = 0; j < nTime; j++) {
            const v = row[j];
            if (v !== undefined && v !== null && !isNaN(v)) {
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        }
    }
    if (!isFinite(mn) || !isFinite(mx)) { mn = 0; mx = 1; }
    // diverging 使用对称范围
    let normMin = mn, normMax = mx;
    if (colorScale === 'diverging') {
        const absMax = Math.max(Math.abs(mn), Math.abs(mx), 0.001);
        normMin = -absMax;
        normMax = absMax;
    }

    // 绘制单元格
    for (let i = 0; i < nFreq; i++) {
        const row = matrix[i] || [];
        for (let j = 0; j < nTime; j++) {
            const v = row[j];
            if (v === undefined || v === null || isNaN(v)) continue;
            const t = (v - normMin) / (normMax - normMin);
            ctx.fillStyle = interpolateColor(t, colorScale);
            ctx.fillRect(padL + j * cellW, padT + (nFreq - 1 - i) * cellH, cellW + 0.5, cellH + 0.5);
        }
    }

    // 边框
    ctx.strokeStyle = 'rgba(115,115,115,0.36)';
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);

    // X 轴刻度 (时间)
    ctx.fillStyle = '#737373';
    ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = 'center';
    const xTicks = 6;
    for (let k = 0; k <= xTicks; k++) {
        const tIdx = Math.round((nTime - 1) * k / xTicks);
        const tVal = times[tIdx];
        const x = padL + cellW * tIdx + cellW / 2;
        ctx.fillText(tVal !== undefined ? tVal.toFixed(0) : '', x, h - padB + 14);
    }
    ctx.textAlign = 'right';
    ctx.fillText('时间 (ms)', w - padR, h - 4);

    // Y 轴刻度 (频率)
    ctx.textAlign = 'right';
    const yTicks = Math.min(6, nFreq);
    for (let k = 0; k < yTicks; k++) {
        const fIdx = Math.round((nFreq - 1) * k / (yTicks - 1 || 1));
        const fVal = freqs[fIdx];
        const y = padT + cellH * (nFreq - 1 - fIdx) + cellH / 2;
        ctx.fillText(fVal !== undefined ? fVal.toFixed(1) : '', padL - 6, y + 3);
    }
    ctx.save();
    ctx.translate(12, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('频率 (Hz)', 0, 0);
    ctx.restore();

    // 颜色条
    const cbX = w - padR + 14;
    const cbY = padT;
    const cbW = 12;
    const cbH = plotH;
    for (let i = 0; i < cbH; i++) {
        const t = 1 - i / cbH;
        ctx.fillStyle = interpolateColor(t, colorScale);
        ctx.fillRect(cbX, cbY + i, cbW, 1);
    }
    ctx.strokeStyle = 'rgba(115,115,115,0.36)';
    ctx.strokeRect(cbX, cbY, cbW, cbH);
    // 颜色条刻度
    ctx.fillStyle = '#737373';
    ctx.textAlign = 'left';
    ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
    ctx.fillText(normMax.toFixed(2), cbX + cbW + 4, cbY + 8);
    ctx.fillText(((normMax + normMin) / 2).toFixed(2), cbX + cbW + 4, cbY + cbH / 2 + 3);
    ctx.fillText(normMin.toFixed(2), cbX + cbW + 4, cbY + cbH);
    ctx.save();
    ctx.translate(cbX + cbW + 30, cbY + cbH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(colorbarLabel || '', 0, 0);
    ctx.restore();
}

// 颜色插值
// scale='diverging': 蓝(0) - 白(0.5) - 红(1) 适用于 ERSP
// scale='sequential': 白(0) - 红(1) 适用于 ITPC
function interpolateColor(t, scale) {
    t = Math.max(0, Math.min(1, t));
    if (scale === 'diverging') {
        // 蓝 #22A5F7 -> 白 #FFFFFF -> 红 #F87454
        if (t < 0.5) {
            const k = t / 0.5;
            const r = Math.round(34 + (255 - 34) * k);
            const g = Math.round(165 + (255 - 165) * k);
            const b = Math.round(247 + (255 - 247) * k);
            return `rgb(${r},${g},${b})`;
        } else {
            const k = (t - 0.5) / 0.5;
            const r = Math.round(255 + (248 - 255) * k);
            const g = Math.round(255 + (116 - 255) * k);
            const b = Math.round(255 + (84 - 255) * k);
            return `rgb(${r},${g},${b})`;
        }
    } else {
        // 白 -> 红
        const r = Math.round(255 + (248 - 255) * t);
        const g = Math.round(255 + (116 - 255) * t);
        const b = Math.round(255 + (84 - 255) * t);
        return `rgb(${r},${g},${b})`;
    }
}

// ==========================================================
// OpenBCI 导入视图
// ==========================================================
let obciData = null;
let obciChart = null;

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('obci-file');
    const detectBtn = document.getElementById('obci-detect');
    const saveBtn = document.getElementById('obci-save');
    const analyzeBtn = document.getElementById('obci-analyze');

    fileInput?.addEventListener('change', (e) => {
        const f = e.target.files[0];
        document.getElementById('obci-file-hint').textContent = f ? f.name : '未选择';
        detectBtn.disabled = !f;
        analyzeBtn.disabled = true;
        obciData = null;
    });

    detectBtn?.addEventListener('click', detectOpenBCI);
    saveBtn?.addEventListener('click', saveOpenBCI);
    analyzeBtn?.addEventListener('click', analyzeOpenBCI);
});

async function detectOpenBCI() {
    const file = document.getElementById('obci-file').files[0];
    if (!file) { showToast('请先选择文件'); return; }

    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('convert_uv', document.getElementById('obci-convert-uv').value === '1');
        formData.append('gain', document.getElementById('obci-gain').value || '0');

        const resp = await fetch('/api/openbci/convert', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.detail) throw new Error(data.detail);

        obciData = data;
        document.getElementById('obci-empty').style.display = 'none';
        document.getElementById('obci-content').style.display = 'flex';
        document.getElementById('obci-analyze').disabled = false;

        showOpenBCIInfo(data);
        renderOpenBCIPreview(data);
        renderOpenBCIStats(data);

        showToast(`${data.board.toUpperCase()} · ${data.n_channels}ch · ${data.sample_rate}Hz · ${data.duration_sec}s`);
    } catch (err) {
        showToast('检测失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

function showOpenBCIInfo(data) {
    const section = document.getElementById('obci-info-section');
    const list = document.getElementById('obci-info-list');
    section.style.display = 'block';

    const items = [
        ['板卡', data.board?.toUpperCase() || '—'],
        ['通道数', data.n_channels],
        ['采样率', data.sample_rate + ' Hz'],
        ['样本数', data.n_samples?.toLocaleString()],
        ['时长', data.duration_sec + ' 秒'],
        ['加速计', data.has_accelerometer ? '有' : '无'],
        ['模拟输入', data.has_analog ? '有' : '无'],
    ];

    list.innerHTML = items.map(([label, value]) =>
        `<div class="param-item">
            <label>${label}</label>
            <span style="color:#737373;font-size:13px;padding-top:4px;display:block;">${value}</span>
        </div>`
    ).join('');
}

function renderOpenBCIPreview(data) {
    const canvas = document.getElementById('chart-obci-preview');
    const ctx = canvas.getContext('2d');
    if (obciChart) obciChart.destroy();

    const p = data.preview;
    const times = p.times || [];
    const channels = p.channels || {};
    const chNames = Object.keys(channels);

    const displayChs = chNames.slice(0, 4);
    const step = Math.max(1, Math.floor(times.length / 800));
    const labels = times.filter((_, i) => i % step === 0);

    const palette = ['#4B3FE3', '#1DC981', '#22A5F7', '#F87454'];
    const datasets = displayChs.map((ch, i) => ({
        label: ch,
        data: channels[ch].filter((_, j) => j % step === 0),
        borderColor: palette[i % palette.length],
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
    }));

    obciChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'bottom', labels: { color: '#404040', font: { size: 11 }, boxWidth: 12, padding: 12 } },
                tooltip: {
                    backgroundColor: '#FFFFFF', titleColor: '#171717', bodyColor: '#404040',
                    borderColor: 'rgba(115,115,115,0.18)', borderWidth: 1, cornerRadius: 8,
                },
            },
            scales: {
                x: { title: { display: true, text: '时间 (s)', color: '#737373' },
                    grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373', font: { size: 11 } } },
                y: { title: { display: true, text: 'Raw ADC', color: '#737373' },
                    grid: { color: 'rgba(115,115,115,0.08)' }, ticks: { color: '#737373', font: { size: 11 } } },
            },
        },
    });

    document.getElementById('obci-preview-info').textContent =
        `显示 ${displayChs.length}/${chNames.length} 个通道的前 ${times.length} 个样本`;
}

function renderOpenBCIStats(data) {
    const wrap = document.getElementById('obci-stats-table');
    const p = data.preview;
    const channels = p.channels || {};
    let html = '<table class="stats-table"><thead><tr><th>通道</th><th>均值</th><th>范围</th><th>标准差</th></tr></thead><tbody>';
    for (const [name, values] of Object.entries(channels)) {
        const arr = values || [];
        if (!arr.length) continue;
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const min = Math.min(...arr);
        const max = Math.max(...arr);
        const std = Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length);
        html += `<tr>
            <td>${name}</td>
            <td class="mono">${mean.toFixed(1)}</td>
            <td class="mono">[${min.toFixed(0)}, ${max.toFixed(0)}]</td>
            <td class="mono">${std.toFixed(1)}</td>
        </tr>`;
    }
    html += '</tbody></table>';
    wrap.innerHTML = html;
}

async function saveOpenBCI() {
    const file = document.getElementById('obci-file').files[0];
    if (!file) { showToast('请先选择文件'); return; }
    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('condition', document.getElementById('obci-condition').value || 'openbci');
        formData.append('convert_uv', document.getElementById('obci-convert-uv').value === '1');
        formData.append('gain', document.getElementById('obci-gain').value || '0');
        const resp = await fetch('/api/openbci/save', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.detail) throw new Error(data.detail);
        showToast(`已保存: ${data.filename} (${data.n_samples} 样本, ${data.n_channels}ch)`);
    } catch (err) {
        showToast('保存失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

async function analyzeOpenBCI() {
    const file = document.getElementById('obci-file').files[0];
    if (!file) { showToast('请先选择文件'); return; }
    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('condition', document.getElementById('obci-condition').value || 'openbci');
        formData.append('convert_uv', document.getElementById('obci-convert-uv').value === '1');
        formData.append('gain', document.getElementById('obci-gain').value || '0');

        const saveResp = await fetch('/api/openbci/save', { method: 'POST', body: formData });
        const saveData = await saveResp.json();
        if (saveData.detail) throw new Error(saveData.detail);

        const params = getParams();
        const analyzeResp = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...params, condition: saveData.condition }),
        });
        const analyzeData = await analyzeResp.json();
        if (analyzeData.error) throw new Error(analyzeData.error);

        // Switch to flow recovery view
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        const flowNav = document.querySelector('[data-module="flow-recovery"]');
        if (flowNav) flowNav.classList.add('active');
        document.querySelectorAll('.module-view').forEach(v => v.classList.remove('active'));
        const flowView = document.getElementById('view-flow-recovery');
        if (flowView) flowView.classList.add('active');

        analyzedConditions.add(saveData.condition);
        renderResults(analyzeData);
        document.getElementById('result-empty').style.display = 'none';
        document.getElementById('result-content').style.display = 'flex';
        document.getElementById('stats-block').style.display = 'block';

        showToast(`分析完成 · ${saveData.filename} · 恢复时长: ${analyzeData.recovery_time?.toFixed(1) || 'N/A'}s`);
    } catch (err) {
        showToast('分析失败: ' + err.message);
    } finally {
        showLoading(false);
    }
}

// ==========================================================
// 模块借鉴新增渲染 (频谱 / 地形图 / Focus)
// ==========================================================
function renderSpectrum(data) {
    if (data.spectrogram_data && data.spectrogram_data.freqs) {
        renderFFTChart(data.spectrogram_data);
    }
    if (data.band_powers) {
        renderBandPowerChart(data.band_powers);
    }
    if (data.spectrogram_data && data.spectrogram_data.sxx) {
        renderSpectrogramCanvas(data.spectrogram_data);
    }
}

function renderBandPowerChart(bandPowers) {
    const canvas = document.getElementById('chart-bandpower');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (charts.bandpower) charts.bandpower.destroy();

    const bands = ['delta', 'theta', 'alpha', 'beta', 'gamma'];
    const labels = ['Delta (1-4Hz)', 'Theta (4-8Hz)', 'Alpha (8-13Hz)', 'Beta (13-30Hz)', 'Gamma (30-45Hz)'];
    const values = bands.map(b => {
        const v = bandPowers[b];
        if (v && typeof v === 'object' && !Array.isArray(v)) {
            return v.rel || v.abs || 0;
        }
        if (Array.isArray(v)) return v[v.length - 1] || 0;
        return v || 0;
    });

    charts.bandpower = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '频带相对功率',
                data: values,
                backgroundColor: ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'],
            }],
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true } },
        },
    });
}

function renderFFTChart(specData) {
    const canvas = document.getElementById('chart-fft');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (charts.fft) charts.fft.destroy();

    const freqs = specData.freqs || [];
    const sxx = specData.sxx;
    let psdData = [];
    if (Array.isArray(sxx) && sxx.length > 0) {
        psdData = Array.isArray(sxx[0]) ? sxx[0] : sxx;
    }

    charts.fft = new Chart(ctx, {
        type: 'line',
        data: {
            labels: freqs.map(f => f.toFixed(1)),
            datasets: [{
                label: 'PSD',
                data: psdData,
                borderColor: '#4B3FE3',
                borderWidth: 1,
                pointRadius: 0,
            }],
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: '频率 (Hz)' } },
                y: { type: 'logarithmic', title: { display: true, text: '功率' } },
            },
        },
    });
}

function renderSpectrogramCanvas(specData) {
    const canvas = document.getElementById('chart-spectrogram');
    if (!canvas || !specData.sxx) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width || 600;
    const H = canvas.height || 300;
    canvas.width = W;
    canvas.height = H;

    ctx.clearRect(0, 0, W, H);

    const sxx = specData.sxx;
    const rows = sxx.length;
    const cols = sxx[0] ? sxx[0].length : 0;
    if (cols === 0) return;

    let zMin = Infinity, zMax = -Infinity;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (sxx[i][j] < zMin) zMin = sxx[i][j];
            if (sxx[i][j] > zMax) zMax = sxx[i][j];
        }
    }
    const zRange = zMax - zMin || 1;

    const cellW = W / cols;
    const cellH = H / rows;

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const t = (sxx[i][j] - zMin) / zRange;
            ctx.fillStyle = jetColor(t);
            ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
        }
    }
}

function jetColor(t) {
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.25) { r = 0; g = 4 * t * 255; b = 255; }
    else if (t < 0.5) { r = 0; g = 255; b = (1 - 4 * (t - 0.25)) * 255; }
    else if (t < 0.75) { r = 4 * (t - 0.5) * 255; g = 255; b = 0; }
    else { r = 255; g = (1 - 4 * (t - 0.75)) * 255; b = 0; }
    return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

function renderTopomapModule(topomapData) {
    const canvas = document.getElementById('topomap-canvas');
    if (!canvas || !window.renderTopomap) return;
    window.renderTopomap(canvas, topomapData);
}

function renderFocus(focusScores) {
    const avgEl = document.getElementById('focus-avg');
    const stabilityEl = document.getElementById('focus-stability');
    const hintEl = document.getElementById('focus-hint');

    if (avgEl) {
        const avg = focusScores.avg || 0;
        avgEl.textContent = avg.toFixed(2);

        if (hintEl) {
            if (avg < 0.3) {
                hintEl.textContent = '走神';
                hintEl.style.color = '#ef4444';
            } else if (avg < 0.7) {
                hintEl.textContent = '一般';
                hintEl.style.color = '#f59e0b';
            } else {
                hintEl.textContent = '专注';
                hintEl.style.color = '#10b981';
            }
        }
    }

    if (stabilityEl) {
        stabilityEl.textContent = (focusScores.stability || 0).toFixed(3);
    }

    const canvas = document.getElementById('chart-focus');
    if (!canvas || !focusScores.scores || focusScores.scores.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (charts.focus) charts.focus.destroy();

    charts.focus = new Chart(ctx, {
        type: 'line',
        data: {
            labels: focusScores.scores.map((_, i) => `窗口${i + 1}`),
            datasets: [{
                label: '专注度',
                data: focusScores.scores,
                borderColor: '#4B3FE3',
                backgroundColor: 'rgba(75, 63, 227, 0.1)',
                fill: true,
                tension: 0.3,
            }],
        },
        options: {
            responsive: true,
            scales: {
                y: { min: 0, max: 1, title: { display: true, text: '专注度分数' } },
            },
        },
    });
}

// ---------- 模块借鉴交互事件 ----------
document.addEventListener('DOMContentLoaded', () => {
    // 频谱 Tab 切换(如果存在 tab 按钮)
    document.querySelectorAll('[data-spectrum-tab]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.spectrumTab;
            document.querySelectorAll('[data-spectrum-tab]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('[data-spectrum-panel]').forEach(p => p.hidden = true);
            const panel = document.querySelector(`[data-spectrum-panel="${tabName}"]`);
            if (panel) panel.hidden = false;
        });
    });

    // 地形图频带切换
    document.querySelectorAll('[data-band]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('[data-band]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // 滤波预设切换
    document.querySelectorAll('[data-preset]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('[data-preset]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const preset = btn.dataset.preset;
            const advanced = document.querySelector('.filter-advanced');
            if (advanced) advanced.hidden = (preset !== 'custom');
        });
    });
});
