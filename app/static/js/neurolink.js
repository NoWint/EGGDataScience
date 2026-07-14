/* ==========================================================
   NeuroLink 实时监测 — 前端交互
   ========================================================== */

let neurolinkWS = null;
let neurolinkCanvas = null;
let neurolinkCtx = null;
// 波形滚动缓冲: 每通道一个数组
const WAVE_BUFFER_SIZE = 600; // ~5 秒 @ 120Hz
let waveBuffers = [[], [], [], []];
let waveAnimId = null;

document.addEventListener('DOMContentLoaded', () => {
    initNeurolinkView();
});

function initNeurolinkView() {
    const container = document.getElementById('neurolink-container');
    if (!container) return;
    container.innerHTML = buildNeurolinkHTML();
    bindNeurolinkEvents();
    initNeurolinkCanvas();
    refreshNeurolinkStatus();
    setInterval(refreshNeurolinkStatus, 2000);
}

function buildNeurolinkHTML() {
    return `
    <div style="padding:24px;">
        <!-- 连接面板 -->
        <div class="nl-panel">
            <div class="nl-panel-title">NeuroLink 连接</div>
            <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                <input type="text" id="nl-room" placeholder="房间号 (4位数字)" style="width:120px;padding:8px;border:1px solid var(--border-l2);border-radius:var(--radius-md);background:var(--surface-main);color:var(--text-primary);">
                <input type="text" id="nl-nickname" placeholder="昵称" value="EEGDataScience" style="width:180px;padding:8px;border:1px solid var(--border-l2);border-radius:var(--radius-md);background:var(--surface-main);color:var(--text-primary);">
                <button class="btn btn-primary" id="btn-nl-connect">连接</button>
                <button class="btn btn-secondary" id="btn-nl-disconnect" style="display:none;">断开</button>
                <span id="nl-status" style="font-size:13px;color:var(--text-tertiary);">未连接</span>
            </div>
        </div>

        <!-- 心流状态 -->
        <div class="nl-panel" id="nl-flow-panel">
            <div class="nl-panel-title">心流状态检测</div>
            <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;">
                <span class="nl-flow-badge idle" id="nl-flow-badge">等待数据</span>
                <span style="font-size:13px;color:var(--text-tertiary);">心流指数 (θ/α): <strong id="nl-flow-index" style="color:var(--text-primary);">—</strong></span>
            </div>
        </div>

        <!-- 波形显示 -->
        <div class="nl-panel">
            <div class="nl-panel-title">实时 EEG 波形 (4通道)</div>
            <canvas id="nl-canvas" width="800" height="200" class="nl-canvas"></canvas>
            <div style="display:flex;gap:16px;margin-top:8px;font-size:12px;color:var(--text-tertiary);">
                <span><span style="display:inline-block;width:10px;height:2px;background:#4B3FE3;vertical-align:middle;"></span> CH1 (Fp1)</span>
                <span><span style="display:inline-block;width:10px;height:2px;background:#1DC981;vertical-align:middle;"></span> CH2 (Fp2)</span>
                <span><span style="display:inline-block;width:10px;height:2px;background:#22A5F7;vertical-align:middle;"></span> CH3 (C3)</span>
                <span><span style="display:inline-block;width:10px;height:2px;background:#F87454;vertical-align:middle;"></span> CH4 (C4)</span>
            </div>
        </div>

        <!-- 指标面板 -->
        <div class="nl-metric-grid">
            <div class="metric-card" id="nl-metric-tar"><div class="metric-label">θ/α 比值</div><div class="metric-value">—</div></div>
            <div class="metric-card" id="nl-metric-entropy"><div class="metric-label">谱熵</div><div class="metric-value">—</div></div>
            <div class="metric-card" id="nl-metric-load"><div class="metric-label">认知负载</div><div class="metric-value">—</div></div>
            <div class="metric-card" id="nl-metric-phase"><div class="metric-label">当前阶段</div><div class="metric-value">—</div></div>
        </div>

        <!-- 频带功率 -->
        <div class="nl-panel">
            <div class="nl-panel-title">频带功率</div>
            <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--text-secondary);">
                <span id="nl-band-delta">δ: —</span>
                <span id="nl-band-theta">θ: —</span>
                <span id="nl-band-alpha">α: —</span>
                <span id="nl-band-beta">β: —</span>
                <span id="nl-band-gamma">γ: —</span>
            </div>
        </div>

        <!-- 记录控制 -->
        <div class="nl-panel">
            <div class="nl-panel-title">会话记录 (CSV 含心流状态标记)</div>
            <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                <input type="text" id="nl-record-subject" placeholder="被试编号" style="width:100px;padding:8px;border:1px solid var(--border-l2);border-radius:var(--radius-md);background:var(--surface-main);color:var(--text-primary);">
                <select id="nl-record-condition" style="padding:8px;border:1px solid var(--border-l2);border-radius:var(--radius-md);background:var(--surface-main);color:var(--text-primary);">
                    <option value="AtoA">A→A</option>
                    <option value="AtoB">A→B</option>
                    <option value="AtoC">A→C</option>
                    <option value="BtoC">B→C</option>
                </select>
                <button class="btn btn-primary" id="btn-nl-record-start">开始记录</button>
                <button class="btn btn-secondary" id="btn-nl-record-stop" style="display:none;">停止记录</button>
                <span id="nl-record-info" style="font-size:13px;color:var(--text-tertiary);"></span>
            </div>
        </div>
    </div>
    `;
}

function bindNeurolinkEvents() {
    document.getElementById('btn-nl-connect').addEventListener('click', connectNeurolink);
    document.getElementById('btn-nl-disconnect').addEventListener('click', disconnectNeurolink);
    document.getElementById('btn-nl-record-start').addEventListener('click', startNeurolinkRecording);
    document.getElementById('btn-nl-record-stop').addEventListener('click', stopNeurolinkRecording);
}

function initNeurolinkCanvas() {
    neurolinkCanvas = document.getElementById('nl-canvas');
    neurolinkCtx = neurolinkCanvas.getContext('2d');
    // 启动渲染循环
    if (waveAnimId) cancelAnimationFrame(waveAnimId);
    waveAnimId = requestAnimationFrame(renderWaveform);
}

async function connectNeurolink() {
    const room = document.getElementById('nl-room').value.trim();
    const nickname = document.getElementById('nl-nickname').value.trim() || 'EEGDataScience';
    if (!room) { alert('请输入房间号'); return; }

    document.getElementById('nl-status').textContent = '连接中...';
    document.getElementById('btn-nl-connect').disabled = true;
    try {
        const resp = await fetchJSON('/api/neurolink/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ room_code: room, nickname }),
        });
        if (!resp.ok) throw new Error(resp.error || '连接失败');
        document.getElementById('nl-status').textContent = `已连接房间 ${room}`;
        document.getElementById('btn-nl-connect').style.display = 'none';
        document.getElementById('btn-nl-disconnect').style.display = '';
        document.getElementById('btn-nl-connect').disabled = false;
        // 连接 WebSocket
        connectNeurolinkWS();
    } catch (err) {
        document.getElementById('nl-status').textContent = '连接失败: ' + err.message;
        document.getElementById('btn-nl-connect').disabled = false;
    }
}

async function disconnectNeurolink() {
    try {
        await fetchJSON('/api/neurolink/disconnect', { method: 'POST' });
    } catch (e) { /* 忽略 */ }
    if (neurolinkWS) { neurolinkWS.close(); neurolinkWS = null; }
    document.getElementById('nl-status').textContent = '已断开';
    document.getElementById('btn-nl-connect').style.display = '';
    document.getElementById('btn-nl-disconnect').style.display = 'none';
    // 清空波形缓冲
    waveBuffers = [[], [], [], []];
}

function connectNeurolinkWS() {
    if (neurolinkWS) neurolinkWS.close();
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    neurolinkWS = new WebSocket(`${protocol}//${location.host}/ws/neurolink`);
    neurolinkWS.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleNeurolinkMessage(msg);
        } catch (e) { /* 忽略解析错误 */ }
    };
    neurolinkWS.onerror = () => {
        document.getElementById('nl-status').textContent = 'WebSocket 连接异常';
    };
}

function handleNeurolinkMessage(msg) {
    if (msg.type === 'eeg_frame') {
        // 将新采样点推入滚动缓冲
        const ch = msg.channels || [0, 0, 0, 0];
        for (let i = 0; i < 4; i++) {
            waveBuffers[i].push(ch[i] || 0);
            if (waveBuffers[i].length > WAVE_BUFFER_SIZE) {
                waveBuffers[i].shift();
            }
        }
    } else if (msg.type === 'metrics_snapshot') {
        updateNeurolinkMetrics(msg);
    } else if (msg.type === 'phase_sync') {
        document.getElementById('nl-metric-phase').querySelector('.metric-value').textContent =
            msg.phase_name || msg.phase_id || '—';
    }
}

// 渲染循环: 持续重绘波形
function renderWaveform() {
    if (!neurolinkCtx) {
        waveAnimId = requestAnimationFrame(renderWaveform);
        return;
    }
    const ctx = neurolinkCtx;
    const w = neurolinkCanvas.width;
    const h = neurolinkCanvas.height;
    const chHeight = h / 4;
    const colors = ['#4B3FE3', '#1DC981', '#22A5F7', '#F87454'];

    // 背景
    const bgColor = getComputedStyle(document.documentElement).getPropertyValue('--surface-sunken').trim() || '#f5f5f5';
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, w, h);

    // 中线
    const lineColor = getComputedStyle(document.documentElement).getPropertyValue('--border-l1').trim() || 'rgba(0,0,0,0.1)';
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1;
    for (let i = 0; i < 4; i++) {
        const y = chHeight * i + chHeight / 2;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // 绘制每通道波形
    for (let ch = 0; ch < 4; ch++) {
        const buf = waveBuffers[ch];
        if (buf.length === 0) continue;
        const yBase = chHeight * ch + chHeight / 2;
        const scale = chHeight / 200; // 缩放因子 (±100μV)
        ctx.strokeStyle = colors[ch];
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < buf.length; i++) {
            const x = (i / WAVE_BUFFER_SIZE) * w;
            const y = yBase - buf[i] * scale;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    waveAnimId = requestAnimationFrame(renderWaveform);
}

function updateNeurolinkMetrics(msg) {
    const setMetric = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.querySelector('.metric-value').textContent = val;
    };
    setMetric('nl-metric-tar', msg.theta_alpha_ratio?.toFixed(3) || '—');
    setMetric('nl-metric-entropy', msg.spectral_entropy?.toFixed(3) || '—');
    setMetric('nl-metric-load', msg.cognitive_load_index?.toFixed(3) || '—');

    const bp = msg.band_power || {};
    const setBand = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };
    setBand('nl-band-delta', `δ: ${bp.delta?.toFixed(2) || '—'}`);
    setBand('nl-band-theta', `θ: ${bp.theta?.toFixed(2) || '—'}`);
    setBand('nl-band-alpha', `α: ${bp.alpha?.toFixed(2) || '—'}`);
    setBand('nl-band-beta', `β: ${bp.beta?.toFixed(2) || '—'}`);
    setBand('nl-band-gamma', `γ: ${bp.gamma?.toFixed(2) || '—'}`);

    // 更新心流状态显示
    updateFlowDisplay(msg.theta_alpha_ratio, msg.cognitive_load_index);
}

function updateFlowDisplay(tar, load) {
    const badge = document.getElementById('nl-flow-badge');
    const indexEl = document.getElementById('nl-flow-index');
    if (!badge || !indexEl) return;

    if (tar !== undefined && tar !== null) {
        indexEl.textContent = tar.toFixed(3);
    }

    // 心流判断逻辑 (与后端一致)
    if (tar >= 1.0 && tar <= 2.0 && (load === undefined || load < 0.5)) {
        badge.className = 'nl-flow-badge entered';
        badge.textContent = '心流状态';
    } else if (tar > 2.5 || (load !== undefined && load > 0.7)) {
        badge.className = 'nl-flow-badge exited';
        badge.textContent = '心流脱离';
    } else {
        badge.className = 'nl-flow-badge idle';
        badge.textContent = '过渡状态';
    }
}

async function refreshNeurolinkStatus() {
    try {
        const status = await fetchJSON('/api/neurolink/status');
        if (status.recording) {
            document.getElementById('nl-record-info').textContent =
                `记录中: ${status.record_count} 采样, ${status.record_duration.toFixed(0)}s`;
        }
        // 从状态更新心流显示 (WebSocket 未连接时)
        if (status.flow_index !== undefined && status.flow_index > 0) {
            updateFlowDisplay(status.flow_index, undefined);
        }
    } catch (e) { /* 忽略 */ }
}

async function startNeurolinkRecording() {
    const subject = document.getElementById('nl-record-subject').value.trim() || 'unknown';
    const condition = document.getElementById('nl-record-condition').value;
    try {
        const resp = await fetchJSON('/api/neurolink/start-recording', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subject, condition }),
        });
        if (!resp.ok) throw new Error(resp.error || '记录启动失败');
        document.getElementById('btn-nl-record-start').style.display = 'none';
        document.getElementById('btn-nl-record-stop').style.display = '';
    } catch (err) {
        alert('记录启动失败: ' + err.message);
    }
}

async function stopNeurolinkRecording() {
    try {
        const resp = await fetchJSON('/api/neurolink/stop-recording', { method: 'POST' });
        if (resp.ok) {
            document.getElementById('btn-nl-record-start').style.display = '';
            document.getElementById('btn-nl-record-stop').style.display = 'none';
            document.getElementById('nl-record-info').textContent =
                `已保存: ${resp.path} (${resp.count} 采样)`;
            alert(`记录已保存\n文件: ${resp.path}\n\nCSV 的 Marker 列记录心流状态 (0=待机, 3=心流进入, 4=心流脱离)\n\n可切换到"批量分析"导入此文件进行深度分析`);
        }
    } catch (err) {
        alert('停止记录失败: ' + err.message);
    }
}
