"""NeuroLink 平台 WebSocket 客户端

作为 monitor 角色连接 wss://eeg.yzjtiantian.cn/ws,
接收实时 EEG 数据帧、指标快照、阶段同步、标记
"""
import json
import threading
import time
import csv
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from collections import deque

import websocket  # websocket-client 库

logger = logging.getLogger("neurolink")

NEUROLINK_URL = "wss://eeg.yzjtiantian.cn/ws"


class NeuroLinkClient:
    """NeuroLink WebSocket 客户端 (后台线程运行)"""

    def __init__(self):
        self.ws: Optional[websocket.WebSocketApp] = None
        self.thread: Optional[threading.Thread] = None
        self.connected = False
        self.room_code: Optional[str] = None
        self.nickname: str = "EEGDataScience Monitor"
        self.session_id: str = f"eegds-{int(time.time())}"

        # 数据缓冲
        self._eeg_buffer = deque(maxlen=7200)  # 60s @ 120Hz
        self._latest_metrics: Optional[dict] = None
        self._current_phase: Optional[dict] = None
        self._markers: List[dict] = []

        # 记录
        self.recording = False
        self._csv_file = None
        self._csv_writer = None
        self._csv_path: Optional[str] = None
        self._record_subject: Optional[str] = None
        self._record_condition: Optional[str] = None
        self._record_count = 0
        self._record_start_time: Optional[float] = None

        # 回调 (用于桥接到 WebSocket 推送)
        self._on_data_callback: Optional[Callable] = None
        self._lock = threading.Lock()

        # 错误信息 (供前端显示)
        self._last_error: Optional[str] = None

        # 心流状态检测
        self._flow_state: str = "idle"  # idle | entered | exited
        self._flow_index: float = 0.0

    def set_data_callback(self, callback: Callable[[dict], None]):
        """设置数据回调 (每收到一帧时调用)"""
        self._on_data_callback = callback

    def connect(self, room_code: str, nickname: str = "EEGDataScience Monitor") -> bool:
        """连接到 NeuroLink 房间"""
        self.room_code = room_code
        self.nickname = nickname
        self.session_id = f"eegds-{int(time.time())}"
        self._last_error = None
        self.connected = False

        self.ws = websocket.WebSocketApp(
            NEUROLINK_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.thread = threading.Thread(target=self._run_ws, daemon=True)
        self.thread.start()
        # 等待连接 (最多 10s)
        for _ in range(20):
            if self.connected:
                return True
            time.sleep(0.5)
        return self.connected

    def _run_ws(self):
        """运行 WebSocket (带自动重连)"""
        for attempt in range(3):
            try:
                self.ws.run_forever()
            except Exception:
                pass
            if self.connected:
                break
            time.sleep(5)
        self.connected = False

    def disconnect(self):
        """断开连接"""
        if self.ws:
            self.ws.close()
        self.connected = False
        self.stop_recording()

    def _on_open(self, ws):
        """连接建立, 发送 hello"""
        ws.send(json.dumps({
            "type": "hello",
            "role": "pending",
            "session_id": self.session_id,
            "device_info": {
                "platform": "EEGDataScience",
                "userAgent": "v2.0.0",
                "nickname": self.nickname,
                "isBridge": False,
            }
        }))

    def _on_message(self, ws, raw):
        """处理收到的消息"""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type")

        if msg_type == "room_info":
            logger.info("收到 room_info, 加入房间 %s", self.room_code)
            ws.send(json.dumps({
                "type": "join_room",
                "code": self.room_code,
                "session_id": self.session_id,
            }))

        elif msg_type == "room_joined":
            logger.info("已加入房间, 认领 monitor 角色")
            ws.send(json.dumps({
                "type": "claim_role",
                "role": "monitor",
                "session_id": msg.get("session_id", self.session_id),
            }))

        elif msg_type == "role_claimed" and msg.get("role") == "monitor":
            logger.info("monitor 角色已确认, 开始接收数据")
            self.connected = True

        elif msg_type == "role_denied":
            reason = msg.get("reason", "")
            logger.warning("角色被拒绝: %s", reason)
            if "reconnect" in reason:
                logger.info("5 秒后发送 reconnect 请求")
                time.sleep(5)
                ws.send(json.dumps({
                    "type": "reconnect",
                    "session_id": self.session_id,
                }))
            else:
                self._last_error = f"角色被拒绝: {reason}"

        elif msg_type == "room_denied":
            reason = msg.get("reason", "房间不存在或已关闭")
            logger.error("房间加入失败: %s", reason)
            self._last_error = f"房间加入失败: {reason}"
            self.connected = False

        elif msg_type == "error":
            logger.error("服务端错误: %s", msg.get("message", msg))
            self._last_error = msg.get("message", "未知错误")

        elif msg_type == "eeg_frame":
            self._on_eeg_frame(msg)

        elif msg_type == "metrics_snapshot":
            self._on_metrics(msg)

        elif msg_type == "phase_sync":
            self._current_phase = msg

        elif msg_type == "marker":
            self._markers.append(msg)

        # 推送给回调
        if self._on_data_callback:
            try:
                self._on_data_callback(msg)
            except Exception:
                pass

    def _on_eeg_frame(self, msg):
        """处理 EEG 帧"""
        channels = msg.get("channels", [0, 0, 0, 0])
        ts = msg.get("ts", 0)
        seq = msg.get("seq", 0)

        with self._lock:
            self._eeg_buffer.append({
                "seq": seq,
                "channels": channels,
                "ts": ts,
            })

        if self.recording:
            self._write_frame(msg)

    def _on_metrics(self, msg):
        """缓存最新指标, 计算心流状态"""
        self._latest_metrics = msg
        # 心流状态检测: θ/α 比值稳定在 1.0-2.0 且认知负载低 → 心流进入
        tar = msg.get("theta_alpha_ratio", 0)
        load = msg.get("cognitive_load_index", 1)
        self._flow_index = tar
        if tar >= 1.0 and tar <= 2.0 and load < 0.5:
            self._flow_state = "entered"
        elif tar > 2.5 or load > 0.7:
            self._flow_state = "exited"
        # 其他情况保持当前状态

    def get_latest_metrics(self) -> Optional[dict]:
        return self._latest_metrics

    def get_current_phase(self) -> Optional[dict]:
        return self._current_phase

    def get_recent_eeg(self, n_samples: int = 600) -> list:
        """获取最近 n 个 EEG 采样点"""
        with self._lock:
            data = list(self._eeg_buffer)
        return data[-n_samples:] if len(data) > n_samples else data

    def start_recording(self, csv_path: str, subject: str = "", condition: str = ""):
        """开始记录 EEG 数据到 CSV"""
        self._csv_path = csv_path
        self._record_subject = subject
        self._record_condition = condition
        self._csv_file = open(csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file, delimiter='\t')
        # BrainFlow RAW 兼容表头
        self._csv_writer.writerow(
            ["Index", "EXG0", "EXG1", "EXG2", "EXG3",
             "Accel_X", "Accel_Y", "Accel_Z", "Timestamp", "Marker"]
        )
        self._record_count = 0
        self._record_start_time = time.time()
        self.recording = True

    def _write_frame(self, msg: dict):
        """将 EEG 帧写入 CSV (Marker 列记录心流状态: 0=idle, 3=entered, 4=exited)"""
        if not self._csv_writer:
            return
        channels = msg.get("channels", [0, 0, 0, 0])
        ts = msg.get("ts", 0)
        # 补齐 4 通道
        while len(channels) < 4:
            channels.append(0.0)
        # 心流状态编码为 Marker: 0=idle, 3=心流进入, 4=心流脱离
        flow_marker = {"idle": 0, "entered": 3, "exited": 4}.get(self._flow_state, 0)
        self._csv_writer.writerow([
            self._record_count,
            f"{channels[0]:.4f}", f"{channels[1]:.4f}",
            f"{channels[2]:.4f}", f"{channels[3]:.4f}",
            0, 0, 0,  # accel (NeuroLink 单独发 accel_frame)
            ts, flow_marker,
        ])
        self._record_count += 1

    def stop_recording(self) -> Optional[str]:
        """停止记录, 返回文件路径"""
        if not self.recording:
            return None
        self.recording = False
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        return self._csv_path

    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            "connected": self.connected,
            "room_code": self.room_code,
            "recording": self.recording,
            "record_count": self._record_count,
            "record_duration": time.time() - self._record_start_time if self._record_start_time and self.recording else 0,
            "current_phase": self._current_phase.get("phase_id") if self._current_phase else None,
            "buffer_size": len(self._eeg_buffer),
            "flow_state": self._flow_state,
            "flow_index": round(self._flow_index, 3),
            "last_error": self._last_error,
        }

    def _on_error(self, ws, error):
        logger.error("WebSocket 错误: %s", error)

    def _on_close(self, ws, code, msg):
        logger.info("WebSocket 关闭: code=%s, msg=%s", code, msg)
        self.connected = False


# 单例
_client_instance: Optional[NeuroLinkClient] = None


def get_client() -> NeuroLinkClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = NeuroLinkClient()
    return _client_instance
