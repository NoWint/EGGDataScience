"""EEG DL V2 终端 TUI — Rich 实现

V2 改进:
  1. 训练流程: 特征工程融合 + LR 实时显示 + Early Stopping 提示
  2. 训练曲线: ASCII 折线图显示 train/val 准确率走势
  3. 混淆矩阵: 训练完成后在验证集上计算 TP/FP/TN/FN
  4. 预测流程: 显示概率(专注概率百分比)而非仅 0/1

运行: python -m eeg_dl.tui
"""
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout
from rich.text import Text
from rich import box

import torch
from torch.utils.data import DataLoader, random_split

from .model import (
    EEGDataset, EEGNetV2, train_model, predict_model, save_model, load_model,
    load_and_segment, generate_labels, extract_features,
    TARGET_FS, WINDOW_SEC, N_CHANNELS, WINDOW_SAMPLES, N_FEATURES,
)

console = Console()

# 默认数据文件
DEFAULT_FILES = [
    '/Users/xiatian/Desktop/BrainFlow-RAW_2026-07-13_16-12-59_4.csv',
    '/Users/xiatian/Desktop/BrainFlow-RAW_2026-07-13_16-12-59_6.csv',
]
MODEL_PATH = 'eeg_dl_model.pth'


def print_header():
    console.print(Panel.fit(
        "[bold cyan]EEG 深度学习终端 V2[/bold cyan]\n"
        "[dim]MultiScale CNN + SE Attention + Feature Fusion · 4 通道 · 专注/放松二分类[/dim]",
        border_style="cyan",
    ))


def show_main_menu() -> str:
    """显示主菜单,返回选择"""
    console.print()
    table = Table(show_header=False, box=box.SIMPLE, border_style="dim")
    table.add_column("选项", style="cyan", width=4)
    table.add_column("功能", style="white")
    table.add_row("1", "训练模型 (Train)")
    table.add_row("2", "预测推理 (Predict)")
    table.add_row("3", "查看数据集 (View Data)")
    table.add_row("q", "退出")
    console.print(table)
    console.print()
    return Prompt.ask("[bold]选择[/bold]", choices=["1", "2", "3", "q"], default="1")


# ========== 训练 ==========

def _ascii_curve(series: List[float], width: int = 50, height: int = 8,
                 lo: float = 0.0, hi: float = 1.0, label: str = "") -> Text:
    """将数值序列渲染为 ASCII 折线(用于训练曲线)"""
    if not series:
        return Text("(无数据)", style="dim")

    span = hi - lo if hi > lo else 1.0
    n = len(series)
    # 每个 x 占用 floor(width / n) 个字符,至少 1
    grid = [[' '] * width for _ in range(height)]

    def to_grid(v):
        v = max(lo, min(hi, v))
        y = int((1 - (v - lo) / span) * (height - 1))
        return y

    # 画连线
    for i in range(n):
        x = int(i * (width - 1) / max(1, n - 1)) if n > 1 else 0
        y = to_grid(series[i])
        grid[y][x] = '●'
        if i > 0:
            prev_x = int((i - 1) * (width - 1) / max(1, n - 1)) if n > 1 else 0
            prev_y = to_grid(series[i - 1])
            # 线性插值连线
            steps = max(abs(x - prev_x), abs(y - prev_y), 1)
            for s in range(1, steps + 1):
                ix = prev_x + (x - prev_x) * s // steps
                iy = prev_y + (y - prev_y) * s // steps
                if grid[iy][ix] == ' ':
                    grid[iy][ix] = '·'

    lines = []
    for y in range(height):
        v_label = f"{hi - (y / (height - 1)) * span:.2f}"
        lines.append(f"{v_label:>5} │" + ''.join(grid[y]))
    # x 轴
    lines.append("      " + "└" + "─" * (width - 1))
    x_axis = "      " + f"1".ljust(width // 2) + f"{n}"
    lines.append(x_axis)

    text = Text()
    if label:
        text.append(f"{label}\n", style="cyan")
    text.append("\n".join(lines))
    return text


def _confusion_matrix_panel(y_true: np.ndarray, y_pred: np.ndarray) -> Panel:
    """构建混淆矩阵面板(2x2)"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    table = Table(box=box.SIMPLE_HEAVY, show_header=True)
    table.add_column("", style="dim", width=12)
    table.add_column("真实: 放松", style="yellow", justify="right")
    table.add_column("真实: 专注", style="green", justify="right")
    table.add_row("[bold]预测: 放松[/bold]", str(tn), str(fn))
    table.add_row("[bold]预测: 专注[/bold]", str(fp), str(tp))

    return Panel(
        table,
        title="[bold]混淆矩阵 (验证集)[/bold]",
        border_style="magenta",
        subtitle=(
            f"准确率={acc*100:.1f}%  精确率={precision*100:.1f}%  "
            f"召回率={recall*100:.1f}%  F1={f1*100:.1f}%"
        ),
    )


def do_train():
    """训练流程 V2"""
    console.print("\n[bold cyan]═══ 训练模型 V2 ═══[/bold cyan]\n")

    # 1. 选择数据文件
    files = select_data_files()
    if not files:
        console.print("[red]未选择文件[/red]")
        return

    # 2. 加载 + 切分 + 特征提取
    all_windows = []
    all_labels = []
    all_features = []
    for f in files:
        console.print(f"[dim]加载[/dim] {Path(f).name} ...")
        try:
            windows, fs = load_and_segment(f)
            labels = generate_labels(windows, fs)
            features = extract_features(windows, fs)
            all_windows.append(windows)
            all_labels.append(labels)
            all_features.append(features)
            console.print(
                f"  → {len(windows)} 窗口, {windows.shape[1]} 通道, "
                f"专注占比 {labels.mean()*100:.1f}%, 特征维度 {features.shape[1]}"
            )
        except Exception as e:
            console.print(f"  [red]失败: {e}[/red]")
            continue

    if not all_windows:
        console.print("[red]无有效数据[/red]")
        return

    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    features = np.concatenate(all_features, axis=0)
    console.print(f"\n[green]总计: {len(windows)} 窗口, 特征维度 {features.shape[1]}[/green]")

    # 3. 数据集分割
    dataset = EEGDataset(windows, labels, features=features)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    console.print(f"训练集: {n_train} · 验证集: {n_val}")

    # 4. 超参数
    epochs = IntPrompt.ask("训练轮数", default=30)
    batch_size = IntPrompt.ask("批大小", default=32)
    lr = float(Prompt.ask("学习率", default="0.0005"))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"设备: [yellow]{device}[/yellow]")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 5. 训练
    console.print("\n[bold]开始训练 V2...[/bold]\n")
    live_table = Table(title="训练进度", box=box.ROUNDED)
    live_table.add_column("Epoch", justify="right", style="cyan", width=6)
    live_table.add_column("Train Loss", justify="right", style="yellow")
    live_table.add_column("Val Loss", justify="right", style="yellow")
    live_table.add_column("Train Acc", justify="right", style="green")
    live_table.add_column("Val Acc", justify="right", style="green")
    live_table.add_column("LR", justify="right", style="magenta")
    live_table.add_column("状态", justify="center")

    rows = []

    def callback(epoch, train_loss, val_loss, train_acc, val_acc, current_lr,
                 early_stopped=False):
        if val_acc > 0.7:
            status = "[green]良好[/green]"
        elif val_acc > 0.55:
            status = "[yellow]一般[/yellow]"
        else:
            status = "[red]欠佳[/red]"
        if early_stopped:
            status = "[red bold]早停[/red bold]"
        rows.append([
            str(epoch), f"{train_loss:.4f}", f"{val_loss:.4f}",
            f"{train_acc*100:.1f}%", f"{val_acc*100:.1f}%",
            f"{current_lr:.2e}", status
        ])
        # 实时刷新表格(只显示最近 10 行)
        live_table.rows = []
        for r in rows[-10:]:
            live_table.add_row(*r)
        console.print(live_table, end='\r')

    model, history = train_model(
        train_loader, val_loader,
        epochs=epochs, lr=lr, device=device,
        progress_callback=callback,
    )
    console.print()  # 换行

    # 6. 训练曲线
    console.print("\n[bold cyan]训练曲线[/bold cyan]")
    console.print(_ascii_curve(history['train_acc'], label="训练准确率",
                                lo=0.0, hi=1.0, width=60, height=8))
    console.print()
    console.print(_ascii_curve(history['val_acc'], label="验证准确率",
                                lo=0.0, hi=1.0, width=60, height=8))
    console.print()
    console.print(_ascii_curve(history['train_loss'], label="训练损失",
                                lo=0.0, hi=max(history['train_loss']) * 1.1,
                                width=60, height=6))

    # 7. 混淆矩阵(在验证集上)
    console.print("\n[bold cyan]混淆矩阵 (验证集)[/bold cyan]")
    try:
        val_windows_np = np.array([val_ds[i][0].numpy() for i in range(len(val_ds))])
        val_features_np = np.array([val_ds[i][1].numpy() for i in range(len(val_ds))]) \
            if features is not None else None
        val_labels_np = np.array([val_ds[i][-1].item() for i in range(len(val_ds))])
        preds, probs = predict_model(model, val_windows_np,
                                     features=val_features_np, device=device)
        console.print(_confusion_matrix_panel(val_labels_np, preds))
    except Exception as e:
        console.print(f"[red]混淆矩阵计算失败: {e}[/red]")

    # 8. 最终结果
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0.0
    actual_epochs = len(history['train_acc'])
    console.print(Panel(
        f"[bold green]训练完成[/bold green]\n\n"
        f"  最终训练准确率: [green]{final_train_acc*100:.1f}%[/green]\n"
        f"  最终验证准确率: [green]{final_val_acc*100:.1f}%[/green]\n"
        f"  最佳验证准确率: [bold green]{best_val_acc*100:.1f}%[/bold green]\n"
        f"  实际训练轮数: {actual_epochs} (上限 {epochs})\n"
        f"  参数量: {sum(p.numel() for p in model.parameters()):,}",
        title="结果", border_style="green",
    ))

    # 9. 保存模型
    save_model(model, MODEL_PATH, meta={
        'version': 'v2',
        'epochs_requested': epochs,
        'epochs_actual': actual_epochs,
        'lr': lr,
        'batch_size': batch_size,
        'train_acc': final_train_acc,
        'val_acc': final_val_acc,
        'best_val_acc': best_val_acc,
        'n_train': n_train,
        'n_val': n_val,
        'n_features': N_FEATURES,
        'files': [Path(f).name for f in files],
    })
    console.print(f"\n[bold]模型已保存:[/bold] [cyan]{MODEL_PATH}[/cyan]")


# ========== 预测 ==========

def do_predict():
    """预测流程 V2 — 显示概率"""
    console.print("\n[bold cyan]═══ 预测推理 V2 ═══[/bold cyan]\n")

    if not Path(MODEL_PATH).exists():
        console.print(f"[red]模型文件不存在: {MODEL_PATH}[/red]")
        console.print("[dim]请先训练模型[/dim]")
        return

    # 加载模型
    model, meta = load_model(MODEL_PATH)
    console.print(Panel(
        f"模型加载成功 ({meta.get('version', 'v1')})\n\n"
        f"  最佳验证准确率: {meta.get('best_val_acc', meta.get('val_acc', 0))*100:.1f}%\n"
        f"  训练数据: {', '.join(meta.get('files', []))}",
        title="模型信息", border_style="cyan",
    ))

    # 选择文件
    files = select_data_files()
    if not files:
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 预测
    all_results = []
    for f in files:
        console.print(f"\n[dim]预测[/dim] {Path(f).name} ...")
        try:
            windows, fs = load_and_segment(f)
            features = extract_features(windows, fs)
            preds, probs = predict_model(model, windows, features=features, device=device)

            # probs[:, 1] = 专注概率
            focus_probs = probs[:, 1]
            focused = int(preds.sum())
            relaxed = len(preds) - focused
            focus_pct = focused / len(preds) * 100
            avg_focus_prob = float(focus_probs.mean())

            console.print(f"  窗口数: {len(preds)}")
            console.print(f"  专注: [green]{focused}[/green] ({focus_pct:.1f}%)")
            console.print(f"  放松: [yellow]{relaxed}[/yellow] ({100-focus_pct:.1f}%)")
            console.print(f"  平均专注概率: [magenta]{avg_focus_prob*100:.1f}%[/magenta]")

            all_results.append({
                'file': Path(f).name,
                'n_windows': len(preds),
                'focused': focused,
                'relaxed': relaxed,
                'focus_pct': focus_pct,
                'avg_prob': avg_focus_prob,
            })

            # 显示前 10 个窗口的预测(带概率)
            sample_table = Table(
                title=f"前 10 窗口预测 — {Path(f).name}",
                box=box.SIMPLE,
            )
            sample_table.add_column("窗口", style="cyan", width=6)
            sample_table.add_column("预测", width=8)
            sample_table.add_column("专注概率", justify="right", style="magenta")
            sample_table.add_column("状态")
            for i in range(min(10, len(preds))):
                p = preds[i]
                prob = focus_probs[i]
                # 概率条
                bar_len = int(prob * 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)
                state = "[green]专注[/green]" if p == 1 else "[yellow]放松[/yellow]"
                sample_table.add_row(
                    str(i + 1), str(p),
                    f"{prob*100:5.1f}% {bar}",
                    state,
                )
            console.print(sample_table)

        except Exception as e:
            console.print(f"  [red]失败: {e}[/red]")

    # 汇总
    if len(all_results) > 1:
        summary = Table(title="汇总", box=box.ROUNDED, border_style="green")
        summary.add_column("文件", style="white")
        summary.add_column("窗口", justify="right", style="cyan")
        summary.add_column("专注", justify="right", style="green")
        summary.add_column("放松", justify="right", style="yellow")
        summary.add_column("专注率", justify="right")
        summary.add_column("平均概率", justify="right", style="magenta")
        for r in all_results:
            summary.add_row(
                r['file'], str(r['n_windows']),
                str(r['focused']), str(r['relaxed']),
                f"{r['focus_pct']:.1f}%",
                f"{r['avg_prob']*100:.1f}%",
            )
        console.print(summary)


# ========== 查看数据 ==========

def do_view_data():
    """显示数据集统计"""
    console.print("\n[bold cyan]═══ 数据集统计 ═══[/bold cyan]\n")

    files = select_data_files()
    if not files:
        return

    table = Table(box=box.ROUNDED, border_style="cyan")
    table.add_column("文件", style="white")
    table.add_column("窗口数", justify="right", style="cyan")
    table.add_column("通道", justify="right")
    table.add_column("采样率", justify="right")
    table.add_column("时长", justify="right")
    table.add_column("专注占比", justify="right", style="green")
    table.add_column("特征维度", justify="right", style="magenta")

    for f in files:
        try:
            from app.analysis.openbci_import import load_brainflow_csv
            raw = load_brainflow_csv(Path(f))
            windows, fs = load_and_segment(f)
            labels = generate_labels(windows, fs)
            features = extract_features(windows, fs)
            duration = len(windows) * WINDOW_SEC

            table.add_row(
                Path(f).name,
                str(len(windows)),
                str(N_CHANNELS),
                f"{fs} Hz",
                f"{duration/60:.1f} min",
                f"{labels.mean()*100:.1f}%",
                str(features.shape[1]),
            )
        except Exception as e:
            table.add_row(Path(f).name, "[red]失败[/red]", "-", "-", "-", "-", "-")

    console.print(table)


# ========== 辅助 ==========

def select_data_files() -> list:
    """选择数据文件"""
    console.print("\n[bold]数据文件:[/bold]")
    for i, f in enumerate(DEFAULT_FILES, 1):
        exists = "[green]✓[/green]" if Path(f).exists() else "[red]✗[/red]"
        console.print(f"  {exists} [{i}] {Path(f).name}")
    console.print(f"  [dim][a] 全部[/dim]")
    console.print(f"  [dim][c] 自定义路径[/dim]")

    choice = Prompt.ask("选择", default="a")

    if choice == 'a':
        return [f for f in DEFAULT_FILES if Path(f).exists()]
    elif choice == 'c':
        path = Prompt.ask("输入文件路径")
        if Path(path).exists():
            return [path]
        console.print(f"[red]文件不存在: {path}[/red]")
        return []
    else:
        idx = int(choice) - 1
        if 0 <= idx < len(DEFAULT_FILES) and Path(DEFAULT_FILES[idx]).exists():
            return [DEFAULT_FILES[idx]]
        console.print("[red]无效选择[/red]")
        return []


# ========== 主循环 ==========

def main():
    print_header()

    while True:
        choice = show_main_menu()

        if choice == '1':
            do_train()
        elif choice == '2':
            do_predict()
        elif choice == '3':
            do_view_data()
        elif choice == 'q':
            console.print("[bold cyan]再见[/bold cyan]")
            break

        console.print()
        Prompt.ask("[dim]按回车继续[/dim]", default="")


if __name__ == '__main__':
    main()
