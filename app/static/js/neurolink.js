/* ==========================================================
   NeuroLink 实时监测 — 前端交互
   ========================================================== */

let neurolinkWS = null;
let neurolinkCanvas = null;
let neurolinkCtx = null;

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
    setInterval(refreshNeurolinkStatus, 3000);
}

function buildNeurolinkHTML() {
    return `
    <div style="padding:24px;">
        <!-- 连接面板 -->
        <div style="margin-bottom:24px;padding:16px;border:1px solid var(--border);border-radius:8px;">
            <div style="font-weight:600;margin-bottom:12px;">NeuroLink 连接</div>
            <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                <input type="text" id="nl-room" placeholder="房间号 (4位数字)" style="width:120px;padding:8px;border:1px solid var(--border);border-radius:4px;">
                <input type="text" id="nl-nickname" placeholder="昵称" value="EEGDataScience" style="width:180px;padding:8px;border:1px solid var(--border);border-radius:4px;">
                <button class="btn btn-primary" id="btn-nl-connect">连接</button>
                <button class="btn btn-secondary" id="btn-nl-disconnect" style="display:none;">断开</button>
                <span id="nl-status" style="font-size:13px;color:var(--text-tertiary);">未连接</span>
            </div>
        </div>

        <!-- 波形显示 -->
        <div style="margin-bottom:24px;">
            <div style="font-weight:600;margin-bottom:8px;">实时 EEG 波形 (4通道)</div>
            <canvas id="nl-canvas" width="800" height="200" style="width:100%;border:1px solid var(--border);border-radius:8px;background:#fafafa;"></canvas>
        </div>

        <!-- 指标面板 -->
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px;">
            <div class="metric-card" id="nl-metric-tar"><div class="metric-label">θ/α 比值</div><div class="metric-value">—</div></div>
            <div class="metric-card" id="nl-metric-entropy"><div class="metric-label">谱熵</div><div class="metric-value">—</div></div>
            <div class="metric-card" id="nl-metric-load"><div class="metric-label">认知负载</div><div class="metric-value">—</div></div>
            <div class="metric-card" id="nl-metric-phase"><div class="metric-label">当前阶段</div><div class="metric-value">—</div></div>
        </div>

        <!-- 频带功率 -->
        <div style="margin-bottom:24px;">
            <div style="font-weight:600;margin-bottom:8px;">频带功率</div>
            <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <span id="nl-band-delta" style="font-size:13px;">δ: —</span>
                <span id="nl-band-theta" style="font-size:13px;">θ: —</span>
                <span id="nl-band-alpha" style="font-size:13px;">α: —</span>
                <span id="nl-band-beta" style="font-size:13px;">β: —</span>
                <span id="nl-band-gamma" style="font-size:13px;">γ: —</span>
            </div>
        </div>

        <!-- 记录控制 -->
        <div style="padding:16px;border:1px solid var(--border);border-radius:8px;">
            <div style="font-weight:600;margin-bottom:12px;">会话记录</div>
            <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                <input type="text" id="nl-record-subject" placeholder="被试编号" style="width:100px;padding:8px;border:1px solid var(--border);border-radius:4px;">
                <select id="nl-record-condition" style="padding:8px;border:1px solid var(--border);border-radius:4px;">
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
}

async function connectNeurolink() {
    const room = document.getElementById('nl-room').value.trim();
    const nickname = document.getElementById('nl-nickname').value.trim() || 'EEGDataScience';
    if (!room) { alert('请输入房间号'); return; }

    document.getElementById('nl-status').textContent = '连接中...';
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
        // 连接 WebSocket
        connectNeurolinkWS();
    } catch (err) {
        document.getElementById('nl-status').textContent = '连接失败: ' + err.message;
    }
}

async function disconnectNeurolink() {
    await fetchJSON('/api/neurolink/disconnect', { method: 'POST' });
    if (neurolinkWS) { neurolinkWS.close(); neurolinkWS = null; }
    document.getElementById('nl-status').textContent = '已断开';
    document.getElementById('btn-nl-connect').style.display = '';
    document.getElementById('btn-nl-disconnect').style.display = 'none';
}

function connectNeurolinkWS() {
    if (neurolinkWS) neurolinkWS.close();
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    neurolinkWS = new WebSocket(`${protocol}//${location.host}/ws/neurolink`);
    neurolinkWS.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleNeurolinkMessage(msg);
    };
}

function handleNeurolinkMessage(msg) {
    if (msg.type === 'eeg_frame') {
        drawNeurolinkWaveform(msg.channels);
    } else if (msg.type === 'metrics_snapshot') {
        updateNeurolinkMetrics(msg);
    } else if (msg.type === 'phase_sync') {
        document.getElementById('nl-metric-phase').querySelector('.metric-value').textContent =
            msg.phase_name || msg.phase_id || '—';
    }
}

function drawNeurolinkWaveform(channels) {
    if (!neurolinkCtx) return;
    const ctx = neurolinkCtx;
    const w = neurolinkCanvas.width;
    const h = neurolinkCanvas.height;
    const chHeight = h / 4;

    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, w, h);

    const colors = ['#4B3FE3', '#1DC981', '#22A5F7', '#F87454'];
    channels.forEach((val, i) => {
        const y = chHeight * i + chHeight / 2;
        const scale = chHeight / 200; // 缩放因子
        ctx.strokeStyle = colors[i] || '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(w - 2, y - val * scale);
        ctx.lineTo(w, y - val * scale);
        ctx.stroke();
    });
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
}

async function refreshNeurolinkStatus() {
    try {
        const status = await fetchJSON('/api/neurolink/status');
        if (status.recording) {
            document.getElementById('nl-record-info').textContent =
                `记录中: ${status.record_count} 采样, ${status.record_duration.toFixed(0)}s`;
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
            alert(`记录已保存\n文件: ${resp.path}\n\n可切换到"批量分析"导入此文件进行深度分析`);
        }
    } catch (err) {
        alert('停止记录失败: ' + err.message);
    }
}
