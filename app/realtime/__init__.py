"""实时采集模块"""
from .acquisition import BrainFlowAcquisition
from .manager import AcquisitionManager, get_manager

__all__ = ['BrainFlowAcquisition', 'AcquisitionManager', 'get_manager']
