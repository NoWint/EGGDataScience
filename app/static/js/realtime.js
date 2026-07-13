/**
 * 实时采集模块
 * WebSocket 连接 + Canvas 滚动波形 + Focus 实时显示
 */

let rtWebSocket = null;
let rtCanvas = null;
let rtCtx = null;
let rtState = 'IDLE';
let rtDataBuffer = [];  // 最近 N 秒数据
let rtMaxSamples = 1250;  // 5 秒 @250Hz

/**
 * 初始化实时采集模块
 */
function initRealtime() {
    rtCanvas = document.getElementById('realtime-canvas');
    if (rtCanvas) {
        rtCtx = rtCanvas.getContext('2d');
        rtCanvas.width = rtCanvas.offsetWidth || 800;
        rtCanvas.height = 400;
    }

    const startBtn = document.getElementById('rt-start-btn');
    const stopBtn = document.getElementById('rt-stop-btn');

    if (startBtn) {
        startBtn.addEventListener('click', startRealtime);
    }
    if (stopBtn) {
        stopBtn.addEventListener('click', stopRealtime);
    }

    updateRealtimeStatus('IDLE');
}

/**
 * 启动采集
 */
async function startRealtime() {
    const boardSelect = document.getElementById('rt-board-select');
    const boardId = boardSelect ? boardSelect.value : 'synthetic';

    updateRealtimeStatus('CONNECTING');

    try {
        const resp = await fetch('/api/realtime/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({board_id: boardId, params: {}}),
        });
        const data = await resp.json();

        if (!data.ok) {
            updateRealtimeStatus('ERROR', data.error || '启动失败');
            return;
        }

        connectRealtimeWS();
        updateRealtimeStatus('STREAMING', `已连接 ${data.board_name}`);
    } catch (e) {
        updateRealtimeStatus('ERROR', e.message);
    }
}

/**
 * 停止采集
 */
async function stopRealtime() {
    try {
        await fetch('/api/realtime/stop', {method: 'POST'});
    } catch (e) {}

    if (rtWebSocket) {
        rtWebSocket.close();
        rtWebSocket = null;
    }

    updateRealtimeStatus('IDLE');
    rtDataBuffer = [];
}

/**
 * 连接 WebSocket
 */
function connectRealtimeWS() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/realtime`;

    rtWebSocket = new WebSocket(wsUrl);

    rtWebSocket.onmessage = (event) => {
        const frame = JSON.parse(event.data);
        if (frame.type === 'data') {
            handleRealtimeData(frame);
        }
    };

    rtWebSocket.onclose = () => {
        if (rtState === 'STREAMING') {
            setTimeout(connectRealtimeWS, 2000);
        }
    };
}

/**
 * 处理实时数据帧
 */
function handleRealtimeData(frame) {
    const {data, channels, fs, focus, band_powers} = frame;

    if (!data || data.length === 0) return;

    const nChannels = data.length;
    const nSamples = data[0].length;

    for (let i = 0; i < nSamples; i++) {
        const sample = [];
        for (let ch = 0; ch < nChannels; ch++) {
            sample.push(data[ch][i]);
        }
        rtDataBuffer.push(sample);

        if (rtDataBuffer.length > rtMaxSamples) {
            rtDataBuffer.shift();
        }
    }

    renderRealtimeWaveform(channels);

    if (focus) {
        updateFocusDisplay(focus);
    }

    if (band_powers) {
        updateBandPowersDisplay(band_powers);
    }
}

/**
 * 渲染滚动波形
 */
function renderRealtimeWaveform(channels) {
    if (!rtCtx || rtDataBuffer.length === 0) return;

    const W = rtCanvas.width;
    const H = rtCanvas.height;
    const nChannels = channels.length;
    const channelHeight = H / nChannels;

    rtCtx.clearRect(0, 0, W, H);

    // 找最大绝对值(自动增益)
    let maxVal = 1.0;
    for (const sample of rtDataBuffer) {
        for (const v of sample) {
            if (Math.abs(v) > maxVal) maxVal = Math.abs(v);
        }
    }

    // 绘制每通道
    for (let ch = 0; ch < nChannels && ch < 8; ch++) {
        const yCenter = channelHeight * ch + channelHeight / 2;
        const amplitude = channelHeight * 0.4;

        // 通道名
        rtCtx.fillStyle = '#666';
        rtCtx.font = '11px sans-serif';
        rtCtx.textAlign = 'left';
        rtCtx.fillText(channels[ch] || `CH${ch}`, 4, yCenter - amplitude + 12);

        // 波形
        rtCtx.strokeStyle = '#4B3FE3';
        rtCtx.lineWidth = 1;
        rtCtx.beginPath();

        for (let i = 0; i < rtDataBuffer.length; i++) {
            const x = (i / rtMaxSamples) * W;
            const v = rtDataBuffer[i][ch] || 0;
            const y = yCenter - (v / maxVal) * amplitude;
            if (i === 0) rtCtx.moveTo(x, y);
            else rtCtx.lineTo(x, y);
        }
        rtCtx.stroke();
    }
}

/**
 * 更新 Focus 显示
 */
function updateFocusDisplay(focus) {
    const avgEl = document.getElementById('rt-focus-avg');
    const stabilityEl = document.getElementById('rt-focus-stability');
    const hintEl = document.getElementById('rt-focus-hint');

    if (avgEl) {
        avgEl.textContent = (focus.avg || 0).toFixed(2);
    }
    if (stabilityEl) {
        stabilityEl.textContent = (focus.stability || 0).toFixed(3);
    }
    if (hintEl) {
        const avg = focus.avg || 0;
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

/**
 * 更新频带功率
 */
function updateBandPowersDisplay(bp) {
    const bands = ['delta', 'theta', 'alpha', 'beta', 'gamma'];
    bands.forEach(b => {
        const el = document.getElementById(`rt-bp-${b}`);
        if (el) {
            el.textContent = (bp[b] || 0).toFixed(3);
        }
    });
}

/**
 * 更新状态显示
 */
function updateRealtimeStatus(state, message) {
    rtState = state;
    const statusEl = document.getElementById('rt-status');
    const stateEl = document.getElementById('rt-state');
    const startBtn = document.getElementById('rt-start-btn');
    const stopBtn = document.getElementById('rt-stop-btn');

    const stateText = {
        'IDLE': '空闲',
        'CONNECTING': '连接中...',
        'STREAMING': '采集中',
        'ERROR': '错误',
    }[state] || state;

    if (stateEl) stateEl.textContent = stateText;

    const colors = {
        'IDLE': '#999',
        'CONNECTING': '#f59e0b',
        'STREAMING': '#10b981',
        'ERROR': '#ef4444',
    };
    if (stateEl) stateEl.style.color = colors[state] || '#999';

    if (statusEl && message) {
        statusEl.textContent = message;
    }

    if (startBtn) startBtn.disabled = (state === 'STREAMING' || state === 'CONNECTING');
    if (stopBtn) stopBtn.disabled = (state === 'IDLE');
}

window.initRealtime = initRealtime;
window.startRealtime = startRealtime;
window.stopRealtime = stopRealtime;
