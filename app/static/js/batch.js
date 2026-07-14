/* ==========================================================
   EEG 批量分析 — 前端交互
   ========================================================== */

let batchPollTimer = null;

// ========== 初始化 ==========
document.addEventListener('DOMContentLoaded', () => {
    initBatchView();
});

function initBatchView() {
    const container = document.getElementById('batch-container');
    if (!container) return;
    container.innerHTML = buildBatchHTML();
    bindBatchEvents();
}

function buildBatchHTML() {
    return `
    <div class="batch-upload-area" style="padding:24px;">
        <div style="margin-bottom:16px;">
            <label class="upload-label" style="display:block;margin-bottom:8px;font-weight:600;">选择多个 EEG 文件 (.csv / .txt)</label>
            <input type="file" id="batch-file-input" accept=".csv,.txt" multiple style="margin-bottom:12px;">
            <div id="batch-file-hint" style="font-size:13px;color:var(--text-tertiary);">未选择文件</div>
        </div>
        <div id="batch-assignment-table" style="display:none;margin-top:16px;"></div>
        <div style="margin-top:24px;">
            <button class="btn btn-primary" id="btn-batch-start" disabled>开始批量分析</button>
            <button class="btn btn-secondary" id="btn-batch-download" style="display:none;">下载批量报告 ZIP</button>
        </div>
        <div id="batch-progress" style="margin-top:24px;display:none;"></div>
    </div>
    `;
}

function bindBatchEvents() {
    const fileInput = document.getElementById('batch-file-input');
    const hint = document.getElementById('batch-file-hint');
    const tableDiv = document.getElementById('batch-assignment-table');
    const startBtn = document.getElementById('btn-batch-start');

    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files);
        if (files.length === 0) {
            hint.textContent = '未选择文件';
            tableDiv.style.display = 'none';
            startBtn.disabled = true;
            return;
        }
        hint.textContent = `已选择 ${files.length} 个文件`;
        renderAssignmentTable(files);
        tableDiv.style.display = 'block';
        startBtn.disabled = false;
    });

    startBtn.addEventListener('click', startBatchAnalysis);
    document.getElementById('btn-batch-download').addEventListener('click', downloadBatchReport);
}

// 按文件名时间戳前缀分组
function groupFilesByTimestamp(files) {
    const groups = {};
    files.forEach(f => {
        // 提取时间戳前缀: BrainFlow-RAW_2026-07-14_11-20-30_0.csv → 2026-07-14_11-20-30
        const match = f.name.match(/(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})/);
        const key = match ? match[1] : 'other';
        if (!groups[key]) groups[key] = [];
        groups[key].push(f);
    });
    return groups;
}

function renderAssignmentTable(files) {
    const groups = groupFilesByTimestamp(files);
    const groupColors = ['#4B3FE3', '#1DC981', '#22A5F7', '#F87454', '#EDAA45', '#B655FC'];
    const conditions = ['AtoA', 'AtoB', 'AtoC', 'BtoC'];

    let html = '<table style="width:100%;border-collapse:collapse;font-size:13px;">';
    html += '<thead><tr style="border-bottom:2px solid var(--border);">';
    html += '<th style="text-align:left;padding:8px;">文件名</th>';
    html += '<th style="text-align:left;padding:8px;">时间戳</th>';
    html += '<th style="text-align:left;padding:8px;">被试</th>';
    html += '<th style="text-align:left;padding:8px;">条件</th>';
    html += '</tr></thead><tbody>';

    let colorIdx = 0;
    Object.entries(groups).forEach(([ts, groupFiles]) => {
        const color = groupColors[colorIdx % groupColors.length];
        colorIdx++;
        groupFiles.forEach(f => {
            html += `<tr style="border-bottom:1px solid var(--border);" data-filename="${f.name}">`;
            html += `<td style="padding:8px;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:8px;"></span>${f.name}</td>`;
            html += `<td style="padding:8px;color:var(--text-tertiary);">${ts}</td>`;
            html += `<td style="padding:8px;"><input type="text" class="batch-subject" placeholder="S01" style="width:60px;padding:4px;border:1px solid var(--border);border-radius:4px;"></td>`;
            html += `<td style="padding:8px;"><select class="batch-condition" style="padding:4px;border:1px solid var(--border);border-radius:4px;">`;
            conditions.forEach(c => { html += `<option value="${c}">${c}</option>`; });
            html += '</select></td>';
            html += '</tr>';
        });
    });

    html += '</tbody></table>';
    document.getElementById('batch-assignment-table').innerHTML = html;
}

async function startBatchAnalysis() {
    const fileInput = document.getElementById('batch-file-input');
    const files = Array.from(fileInput.files);
    if (files.length === 0) return;

    // 收集分配表
    const assignments = [];
    document.querySelectorAll('#batch-assignment-table tbody tr').forEach(row => {
        const filename = row.dataset.filename;
        const subject = row.querySelector('.batch-subject').value || 'unknown';
        const condition = row.querySelector('.batch-condition').value;
        assignments.push({ filename, subject, condition });
    });

    // 构建 FormData
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    formData.append('assignments', JSON.stringify(assignments));

    document.getElementById('btn-batch-start').disabled = true;
    document.getElementById('batch-progress').style.display = 'block';
    updateBatchProgress('上传中...', 0, files.length);

    try {
        const resp = await fetch('/api/batch-analyze', { method: 'POST', body: formData });
        if (!resp.ok) {
            let msg = `HTTP ${resp.status}`;
            try { const e = await resp.json(); msg = e.detail || msg; } catch (_) {}
            throw new Error(msg);
        }
        const data = await resp.json();
        pollBatchProgress(data.batch_id, data.total);
    } catch (err) {
        alert('批量分析启动失败: ' + err.message);
        document.getElementById('btn-batch-start').disabled = false;
    }
}

function pollBatchProgress(batchId, total) {
    if (batchPollTimer) clearInterval(batchPollTimer);
    batchPollTimer = setInterval(async () => {
        try {
            const resp = await fetch(`/api/batch-progress/${batchId}`);
            if (!resp.ok) return;
            const prog = await resp.json();
            updateBatchProgress(prog.current_module || prog.current_file || '分析中',
                               prog.current, prog.total);
            if (prog.status !== 'running') {
                clearInterval(batchPollTimer);
                batchPollTimer = null;
                const failCount = prog.errors.length;
                let msg = `批量分析完成: ${prog.total - failCount}/${prog.total} 成功`;
                if (failCount > 0) msg += `\n失败 ${failCount} 项`;
                alert(msg);
                document.getElementById('btn-batch-download').style.display = '';
                document.getElementById('btn-batch-download').dataset.batchId = batchId;
                document.getElementById('btn-batch-start').disabled = false;
            }
        } catch (e) { /* 忽略轮询错误 */ }
    }, 2000);
}

function updateBatchProgress(label, current, total) {
    const pct = total > 0 ? Math.round(current / total * 100) : 0;
    document.getElementById('batch-progress').innerHTML = `
        <div style="font-size:14px;margin-bottom:8px;">${label} (${current}/${total})</div>
        <div style="width:100%;height:8px;background:var(--border);border-radius:4px;overflow:hidden;">
            <div style="width:${pct}%;height:100%;background:var(--primary);transition:width 0.3s;"></div>
        </div>
    `;
}

function downloadBatchReport() {
    const batchId = document.getElementById('btn-batch-download').dataset.batchId;
    if (!batchId) return;
    window.location.href = `/api/export-batch-report?batch_id=${batchId}`;
}
