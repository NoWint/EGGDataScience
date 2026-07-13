"""实时采集测试"""
import pytest
import time


def test_acquisition_synthetic_start_stop():
    """测试 Synthetic Board 启动停止"""
    from app.realtime.acquisition import BrainFlowAcquisition
    from brainflow.board_shim import BoardIds

    acq = BrainFlowAcquisition(BoardIds.SYNTHETIC_BOARD.value)
    assert acq.state == 'IDLE'

    acq.prepare()
    assert acq.state == 'PREPARED'

    acq.start_stream()
    assert acq.state == 'STREAMING'

    # 等待数据
    time.sleep(0.5)
    data = acq.get_latest_data()
    assert data is not None
    assert 'data' in data
    assert 'channels' in data
    assert len(data['data']) > 0  # 有通道数据

    acq.stop_stream()
    assert acq.state == 'STOPPED'

    acq.release_session()
    assert acq.state == 'IDLE'


def test_acquisition_board_info():
    """测试板卡信息获取"""
    from app.realtime.acquisition import BrainFlowAcquisition
    from brainflow.board_shim import BoardIds

    acq = BrainFlowAcquisition(BoardIds.SYNTHETIC_BOARD.value)
    info = acq.get_board_info()

    assert 'board_id' in info
    assert 'board_name' in info
    assert 'fs' in info
    assert 'channels' in info
    assert 'n_exg' in info
    assert info['n_exg'] > 0
    assert info['fs'] > 0
