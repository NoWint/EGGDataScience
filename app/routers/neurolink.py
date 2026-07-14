"""NeuroLink 实时对接 REST + WebSocket 路由"""
import asyncio
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.neurolink_client import get_client

router = APIRouter(prefix="/api/neurolink", tags=["neurolink"])


class ConnectRequest(BaseModel):
    room_code: str
    nickname: str = "EEGDataScience Monitor"


class RecordRequest(BaseModel):
    subject: str = ""
    condition: str = "custom"


@router.get("/status")
def neurolink_status():
    """获取 NeuroLink 连接与记录状态"""
    client = get_client()
    return client.get_status()


@router.post("/connect")
def neurolink_connect(req: ConnectRequest):
    """连接到 NeuroLink 房间"""
    client = get_client()
    if client.connected:
        return {"ok": True, "message": "已连接"}
    ok = client.connect(req.room_code, req.nickname)
    if ok:
        return {"ok": True, "room_code": req.room_code}
    return {"ok": False, "error": "连接失败,请检查房间号和网络"}


@router.post("/disconnect")
def neurolink_disconnect():
    """断开 NeuroLink 连接"""
    client = get_client()
    client.disconnect()
    return {"ok": True}


@router.post("/start-recording")
def neurolink_start_recording(req: RecordRequest):
    """开始记录会话"""
    from app.server import UPLOAD_DIR
    from datetime import datetime

    client = get_client()
    if not client.connected:
        return {"ok": False, "error": "未连接到 NeuroLink"}

    recordings_dir = UPLOAD_DIR.parent / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cond = req.condition or "custom"
    subj = req.subject or "unknown"
    csv_path = recordings_dir / f"NeuroLink_{subj}_{cond}_{timestamp}.csv"

    client.start_recording(str(csv_path), subject=subj, condition=cond)
    return {"ok": True, "path": str(csv_path)}


@router.post("/stop-recording")
def neurolink_stop_recording():
    """停止记录, 返回文件路径"""
    client = get_client()
    path = client.stop_recording()
    if path:
        return {"ok": True, "path": path, "count": client._record_count}
    return {"ok": False, "error": "未在记录"}


async def neurolink_websocket_endpoint(websocket: WebSocket):
    """WebSocket 实时数据推送 (挂载在 /ws/neurolink)"""
    await websocket.accept()
    client = get_client()

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def push_callback(frame):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, frame)
        except Exception:
            pass

    client.set_data_callback(push_callback)

    try:
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(frame)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        client.set_data_callback(None)
