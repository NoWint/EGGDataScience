"""pytest 共享 fixtures"""
import pytest
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.parent

# OpenBCI GUI 样本文件(Cyton 8ch ODF 格式)
SAMPLE_ODF = ROOT / "OpenBCI_GUI" / "OpenBCI_GUI" / "data" / "EEG_Sample_Data" / "OpenBCI_GUI-v6-meditation.txt"


@pytest.fixture
def sample_odf_path():
    """返回 OpenBCI ODF 样本文件路径"""
    if not SAMPLE_ODF.exists():
        pytest.skip(f"样本文件不存在: {SAMPLE_ODF}")
    return SAMPLE_ODF


@pytest.fixture
def tmp_brainflow_csv(tmp_path):
    """构造一个 BrainFlow CSV 测试文件(Cyton 8ch, 24 列)"""
    import numpy as np
    content = ",".join(str(i) for i in range(24)) + "\n"
    np.random.seed(42)
    for _ in range(100):
        row = np.random.randn(24) * 100
        # 倒数第二列 = Timestamp(秒)
        row[-2] = float(_ * 0.004)  # 250 Hz
        # 最后一列 = Marker,前 50 行 0,第 50 行 marker=1,第 80 行 marker=2
        row[-1] = 0
        content += ",".join(f"{v:.4f}" for v in row) + "\n"
    # 注入两个 marker
    lines = content.rstrip().split("\n")
    marker_line_50 = lines[51].split(",")
    marker_line_50[-1] = "1.0"
    lines[51] = ",".join(marker_line_50)
    marker_line_80 = lines[81].split(",")
    marker_line_80[-1] = "2.0"
    lines[81] = ",".join(marker_line_80)
    content = "\n".join(lines) + "\n"
    
    path = tmp_path / "brainflow_test.csv"
    path.write_text(content)
    return path
