/**
 * 头皮地形图渲染器
 * 借鉴 OpenBCI GUI W_HeadPlot,用 2D canvas 渲染
 */

// 8 通道标准位置(与后端 CHANNEL_POSITIONS 一致)
const CHANNEL_POS = {
    'Fp1': {x: -0.3, y: 0.8},
    'Fp2': {x: 0.3, y: 0.8},
    'C3': {x: -0.7, y: 0.0},
    'C4': {x: 0.7, y: 0.0},
    'Pz': {x: 0.0, y: -0.6},
    'O1': {x: -0.4, y: -0.8},
    'O2': {x: 0.4, y: -0.8},
    'Fz': {x: 0.0, y: 0.5},
};

/**
 * 渲染头皮地形图
 * @param {HTMLCanvasElement} canvas
 * @param {Object} topomapData - 后端返回的 {grid_x, grid_y, grid_z, channels, values}
 */
function renderTopomap(canvas, topomapData) {
    if (!canvas || !topomapData || !topomapData.grid_z) return;
    
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;
    const radius = Math.min(W, H) * 0.4;
    
    // 清空
    ctx.clearRect(0, 0, W, H);
    
    const gridZ = topomapData.grid_z;
    const rows = gridZ.length;
    const cols = gridZ[0].length;
    
    // 找最大最小值(用于颜色映射)
    let zMin = Infinity, zMax = -Infinity;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (gridZ[i][j] < zMin) zMin = gridZ[i][j];
            if (gridZ[i][j] > zMax) zMax = gridZ[i][j];
        }
    }
    const zRange = zMax - zMin || 1;
    
    // 绘制热力图(裁剪到圆形)
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.clip();
    
    const cellW = (radius * 2) / cols;
    const cellH = (radius * 2) / rows;
    
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const normalized = (gridZ[i][j] - zMin) / zRange;
            const color = jetColor(normalized);
            ctx.fillStyle = color;
            const x = cx - radius + j * cellW;
            const y = cy - radius + i * cellH;
            ctx.fillRect(x, y, cellW + 1, cellH + 1);
        }
    }
    ctx.restore();
    
    // 绘制头部轮廓
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.stroke();
    
    // 绘制鼻子(顶部三角)
    ctx.beginPath();
    ctx.moveTo(cx - 15, cy - radius);
    ctx.lineTo(cx, cy - radius - 20);
    ctx.lineTo(cx + 15, cy - radius);
    ctx.stroke();
    
    // 绘制电极位置点
    const channels = topomapData.channels || [];
    ctx.fillStyle = '#000';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    
    channels.forEach(ch => {
        const pos = CHANNEL_POS[ch];
        if (!pos) return;
        const x = cx + pos.x * radius;
        const y = cy - pos.y * radius;  // canvas y 轴向下
        
        // 电极点
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
        
        // 标签
        ctx.fillText(ch, x, y - 8);
    });
}

/**
 * jet 色图(蓝→青→绿→黄→红)
 */
function jetColor(t) {
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.25) {
        r = 0; g = 4 * t * 255; b = 255;
    } else if (t < 0.5) {
        r = 0; g = 255; b = (1 - 4 * (t - 0.25)) * 255;
    } else if (t < 0.75) {
        r = 4 * (t - 0.5) * 255; g = 255; b = 0;
    } else {
        r = 255; g = (1 - 4 * (t - 0.75)) * 255; b = 0;
    }
    return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

// 导出供 app.js 调用
window.renderTopomap = renderTopomap;
