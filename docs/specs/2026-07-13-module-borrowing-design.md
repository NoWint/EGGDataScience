# 模块借鉴设计文档

## 项目概述

**项目名称**: EEGDataScience
**作者 GitHub**: [NoWint](https://github.com/NoWint)
**子项目**: 模块借鉴(借鉴 OpenBCI GUI 可视化模块)
**目标**: 借鉴 OpenBCI GUI 的可视化模块设计,增强 EEGDataScience 的离线分析可视化能力,覆盖头皮地形图渲染、Focus 专注度检测、频谱模块前端完整化、独立滤波 UI 面板四个模块。面向新手用户,强调开箱即用与直观可解释。

## 范围

### 包含
- **频谱模块前端完整化**:FFT 频谱图、频带能量柱状图、时频谱图三个子模块的前端 UI
- **头皮地形图渲染**:2D canvas 渲染,8 通道标准位置,RBF 插值热力图
- **Focus 专注度检测**:BrainFlow MLModel (ONNX) 推理,专注度时序图
- **独立滤波 UI 面板**:预设(脑电/肌电/心电)+ 高级手动调节
- **后端增强**:`/api/analyze` 返回 topomap/band_powers/spectrogram/focus 字段

### 不包含
- 实时波形滚动显示(属于实时采集子项目)
- EMG/PulseSensor/Accelerometer 模块(非 EEG 核心场景)
- 多布局 Widget 切换(单页按模块切换足够)
- 阻抗测量(硬件相关,属于实时采集)
- 网络流输出(属于实时采集)
- 引入新前端库(用 Chart.js + canvas)

## 现状诊断

### 后端分析能力(已完整)
- `spectrum.py` — compute_psd / compute_band_powers / compute_spectrogram / compute_aperiodic_signal
- `erp.py` — ERP 提取、基线校正、峰值检测
- `ersp.py` — 事件相关谱扰动
- `stats_viz.py` — compute_topomap_data(但默认只支持 3 通道)
- `flow_recovery.py` — run_full_pipeline(心流恢复主流程)
- `artifact.py` — 伪迹剔除

### 前端现状(不完整)
- 只有心流恢复分析模块有完整 UI
- 频谱/ERP/ERSP 侧边栏有入口但 UI 未完整实现
- 头皮地形图后端有数据但前端未渲染
- Focus 专注度完全缺失
- 滤波参数散在分析请求里,无独立 UI

### OpenBCI GUI 对比
| 模块 | EEGDataScience 后端 | EEGDataScience 前端 | 本次借鉴 |
|---|---|---|---|
| W_FFT | ✅ spectrum.py | ❌ 未完整 | 前端完整化 |
| W_BandPower | ✅ spectrum.py | ❌ 未完整 | 前端完整化 |
| W_Spectrogram | ✅ spectrum.py | ❌ 未完整 | 前端完整化 |
| W_HeadPlot | ✅ stats_viz.py(3ch) | ❌ 未渲染 | 8ch + 前端渲染 |
| W_Focus | ❌ 无 | ❌ 无 | BrainFlow MLModel |
| FilterUI | ✅ preprocess 参数 | ❌ 无独立 UI | 新增 UI 面板 |

## 整体架构

**方案 A:统一 /api/analyze 增强返回**(用户认可)

```
用户上传 .txt/.csv → /api/analyze(增强返回)
                        │
                        ├─ 现有: data, fs, channels, times, recovery_metrics
                        │
                        ├─ 新增: topomap_data     ← stats_viz.compute_topomap_data(8ch)
                        ├─ 新增: band_powers      ← spectrum.compute_band_powers
                        ├─ 新增: spectrogram_data ← spectrum.compute_spectrogram
                        ├─ 新增: focus_scores     ← focus.py(brainflow.MLModel)
                        └─ 新增: metadata         ← load_eeg_full 已有
                        │
                        ▼
前端按模块切换渲染:
  心流恢复 | 频谱分析 | 头皮地形图 | Focus 专注度 | 滤波设置
```

**滤波参数**作为 `/api/analyze` 请求字段:
- `filter_preset`: "eeg" | "emg" | "ecg" | "custom"
- `filter_params`: {hp, lp, notch}(仅 preset=custom 时生效)

## 组件清单

### 后端

| 组件 | 文件 | 职责 |
|---|---|---|
| `compute_focus_scores()` | `app/analysis/focus.py`(新建) | BrainFlow MLModel 专注度计算 |
| `compute_topomap_data()` | `app/analysis/stats_viz.py`(修改) | 扩展支持 8 通道 + CHANNEL_POSITIONS |
| `run_full_pipeline()` | `app/analysis/flow_recovery.py`(修改) | 增加返回 topomap/band_powers/spectrogram/focus |
| `/api/analyze` | `app/server.py`(修改) | 请求增加 filter_preset/filter_params 字段 |
| `requirements.txt`(修改) | 新增 brainflow 包 |

### 前端

| 组件 | 文件 | 职责 |
|---|---|---|
| 侧边栏模块入口 | `app/static/index.html`(修改) | 新增 Focus/地形图模块入口 + 内容区容器 |
| 模块渲染函数 | `app/static/js/app.js`(修改) | 新增 renderTopomap / renderFocus / renderSpectrum |
| 地形图渲染器 | `app/static/js/topomap.js`(新建) | canvas 地形图渲染 |
| 模块样式 | `app/static/css/style.css`(修改) | 新增模块样式 |

## 模块详细设计

### 1. 频谱模块前端完整化

**后端**(已有,无需改动):
- `spectrum.compute_psd(data, fs)` → {freqs, psd}
- `spectrum.compute_band_powers(data, fs)` → {band_name: array}
- `spectrum.compute_spectrogram(data, fs)` → {freqs, times, sxx}

**前端**:三个 Tab 切换
- **FFT 频谱图**:对数坐标,各通道叠加,频带区域标注(Delta/Theta/Alpha/Beta/Gamma 用不同背景色)
- **频带能量柱状图**:Delta(1-4)/Theta(4-8)/Alpha(8-13)/Beta(13-30)/Gamma(30-55) 分组柱状
- **时频谱图**:瀑布图,颜色映射功率(viridis 色图)

**新手友好**:预设频带范围,无需调节;图例清晰标注各频带含义。

### 2. 头皮地形图渲染

**后端**(修改 `stats_viz.py`):
- 扩展 `CHANNEL_POSITIONS` 支持 8 通道标准位置:
  - Fp1(-0.3, 0.8), Fp2(0.3, 0.8)
  - C3(-0.7, 0.0), C4(0.7, 0.0)
  - Pz(0.0, -0.6)
  - O1(-0.3, -0.8), O2(0.3, -0.8)
  - Fz(0.0, 0.5)
- `compute_topomap_data(values, channel_names)` 支持任意通道数

**前端**(`topomap.js` 新建):
- 2D canvas 渲染
- 圆形头部轮廓 + 鼻子标记(顶部三角)
- RBF 插值热力图(红=高功率,蓝=低,jet 色图)
- 8 电极位置点标注(Fp1/Fp2/C3/C4/Pz/O1/O2/Fz)
- 可切换显示频带(Alpha/Beta/Theta 按钮)
- 颜色图例(右侧色条)

### 3. Focus 专注度检测

**后端**(`focus.py` 新建):
```python
from brainflow import MLModel, BrainFlowModelParams, BrainFlowMetrics, BrainFlowClassifiers

def compute_focus_scores(data, fs):
    """用 BrainFlow MLModel 计算专注度分数
    
    返回: {scores: List[float], avg: float, stability: float}
    """
    params = BrainFlowModelParams(BrainFlowMetrics.CONCENTRATION.value,
                                  BrainFlowClassifiers.DEFAULT.value)
    model = MLModel(params)
    model.prepare()
    
    scores = []
    window_sec = 4.0  # 4 秒窗口
    step = int(window_sec * fs)
    for i in range(0, len(data) - step, step):
        segment = data[i:i+step]
        score = model.predict(segment)
        scores.append(float(score[0]))
    
    model.release()
    
    return {
        'scores': scores,
        'avg': float(np.mean(scores)) if scores else 0.0,
        'stability': float(np.std(scores)) if scores else 0.0,
    }
```

**前端**:
- 专注度时序折线图(0-1 分数)
- 平滑曲线叠加
- 平均专注度 + 稳定性(标准差)统计卡片
- 新手提示:0-0.3 走神(红),0.3-0.7 一般(黄),0.7-1 专注(绿)

### 4. 独立滤波 UI 面板

**后端**(`server.py` 修改):
- `/api/analyze` 请求增加字段:
  - `filter_preset`: "eeg" | "emg" | "ecg" | "custom"(默认 "eeg")
  - `filter_params`: {hp: float, lp: float, notch: float}(仅 preset="custom" 时生效)
- 预设映射:
  - eeg: hp=1, lp=45, notch=50
  - emg: hp=20, lp=250, notch=50
  - ecg: hp=0.5, lp=40, notch=50

**前端**:
- 侧边栏底部"滤波设置"卡片
- 预设按钮组:脑电 / 肌电 / 心电 / 自定义
- 高级参数(自定义时展开):HP / LP / 陷波频率 slider
- 参数变更后点"重新分析"生效

## API 变更

### `/api/analyze` 请求(修改)

新增字段:
```python
class AnalyzeRequest(BaseModel):
    # 现有字段...
    condition: str
    lp: float = 45.0
    hp: float = 1.0
    # ...
    
    # 新增字段
    filter_preset: str = "eeg"  # "eeg" | "emg" | "ecg" | "custom"
    filter_params: Optional[dict] = None  # {hp, lp, notch} 仅 custom 时生效
```

### `/api/analyze` 返回(增强)

新增字段:
```python
result = {
    # 现有字段...
    'data': ...,
    'fs': ...,
    'channels': ...,
    'recovery_metrics': ...,
    'metadata': ...,  # Task 4 已有
    
    # 新增字段
    'topomap_data': {
        'grid_x': ..., 'grid_y': ..., 'grid_z': ...,
        'channels': ['Fp1', 'Fp2', 'C3', 'C4', 'Pz', 'O1', 'O2', 'Fz'],
        'values': [...],  # 各通道 alpha 功率
    },
    'band_powers': {
        'delta': [...], 'theta': [...], 'alpha': [...],
        'beta': [...], 'gamma': [...],
    },
    'spectrogram_data': {
        'freqs': [...], 'times': [...], 'sxx': [[...]],
    },
    'focus_scores': {
        'scores': [...], 'avg': 0.65, 'stability': 0.12,
    },
}
```

## 错误处理

| 场景 | 处理 |
|---|---|
| BrainFlow 包未安装 | focus_scores 返回 None,前端隐藏 Focus 模块 |
| MLModel 推理失败 | focus_scores 返回 None,日志记录错误 |
| 通道数 < 8 | topomap_data 用实际通道数,前端标注缺失位置 |
| 频谱计算数据太短 | 返回空数组,前端显示"数据不足" |
| filter_preset 无效 | 默认 "eeg" |

## 测试策略

1. **后端单元测试**:
   - `focus.py`:compute_focus_scores 返回正确格式
   - `stats_viz.py`:compute_topomap_data 支持 8 通道
   - `flow_recovery.py`:run_full_pipeline 返回新字段

2. **API 测试**:
   - `/api/analyze` 返回 topomap_data/band_powers/spectrogram_data/focus_scores
   - filter_preset 参数生效

3. **前端测试**:
   - 手动验证:上传 ODF 文件 → 切换模块 → 渲染正确

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `app/analysis/focus.py` | 新建 | BrainFlow MLModel 专注度计算 |
| `app/analysis/stats_viz.py` | 修改 | 扩展 compute_topomap_data 支持 8 通道 + CHANNEL_POSITIONS |
| `app/analysis/flow_recovery.py` | 修改 | run_full_pipeline 增加返回新字段 |
| `app/analysis/__init__.py` | 修改 | 导出 compute_focus_scores |
| `app/server.py` | 修改 | /api/analyze 请求增加 filter_preset/filter_params |
| `requirements.txt` | 修改 | 新增 brainflow 包 |
| `app/static/index.html` | 修改 | 侧边栏新增模块入口 + 内容区容器 |
| `app/static/js/app.js` | 修改 | 新增渲染函数 |
| `app/static/js/topomap.js` | 新建 | canvas 地形图渲染 |
| `app/static/css/style.css` | 修改 | 新增模块样式 |

## 新手友好设计

- ✅ 滤波预设一键应用(脑电/肌电/心电),无需懂参数
- ✅ Focus 分数有文字解释(走神/一般/专注)
- ✅ 地形图有频带切换 + 颜色图例
- ✅ 频谱图有频带区域标注(Delta/Theta/Alpha/Beta/Gamma)
- ✅ 默认参数合理,上传后直接出结果

## YAGNI 检查

- ❌ 不做实时波形滚动(属于实时采集子项目)
- ❌ 不做 EMG/PulseSensor/Accelerometer(非 EEG 核心场景)
- ❌ 不做多布局 Widget 切换(单页按模块切换足够)
- ❌ 不做阻抗测量(硬件相关,属于实时采集)
- ❌ 不做网络流输出(属于实时采集)
- ✅ 用 Chart.js(现有) + canvas(地形图),不引入新前端库
