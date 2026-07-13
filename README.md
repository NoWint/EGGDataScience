# EEGDataScience

跨学科任务切换对心流状态的影响及 EEG 恢复时间量化研究 — 便携式 EEG 头环实验数据分析平台。

作者: [NoWint](https://github.com/NoWint)

## 快速开始

### 首次配置
- **macOS**: 双击 `setup.command`
- **Windows**: 双击 `setup.bat`

### 启动应用
- **macOS**: 双击 `start.command`
- **Windows**: 双击 `start.bat`

启动后浏览器会自动打开 `http://localhost:18765`。

## 功能
- 心流恢复分析（4 种实验条件：A→A / A→B / A→C / B→C）
- 6 项 EEG 指标时序可视化（Theta/Alpha 比值、Alpha/Beta/Gamma 能量、谱熵、认知负载指数）
- 恢复时长量化（±5% 容差，30s 连续窗口）
- 跨条件统计比较（配对 t 检验、重复测量 ANOVA）
- 结构化分析报告生成

## 技术栈
Python 3.11+ / FastAPI / numpy / pandas / scipy / Chart.js
