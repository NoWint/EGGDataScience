"""实时采集 REST + WebSocket 路由"""
import asyncio
from typing import Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.realtime.manager import get_manager, BRAINFLOW_AVAILABLE


router = APIRouter(prefix="/api/realtime", tags=["realtime"])


class StartRequest(BaseModel):
    board_id: str = "synthetic"  # synthetic | cyton | daisy | ganglion
    params: Optional[Dict] = None


@router.get("/status")
def realtime_status():
    """获取采集状态"""
    manager = get_manager()
    return manager.get_status()


@router.post("/start")
def realtime_start(req: StartRequest):
    """启动采集"""
    if not BRAINFLOW_AVAILABLE:
        return {"ok": False, "error": "brainflow not installed"}

    manager = get_manager()
    try:
        manager.start(req.board_id, req.params or {})
        info = manager.get_status()
        return {
            "ok": True,
            "board_id": info['board_id'],
            "board_name": info['board_name'],
            "fs": info['fs'],
            "channels": info['channels'],
            "n_exg": len(info['channels']),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/stop")
def realtime_stop():
    """停止采集"""
    manager = get_manager()
    elapsed = manager.get_status().get('elapsed_sec', 0.0)
    manager.stop()
    return {"ok": True, "elapsed_sec": elapsed}


async def realtime_websocket_endpoint(websocket: WebSocket):
    """WebSocket 实时数据推送

    挂载在 /ws/realtime,由 server.py 的 @app.websocket 调用
    """
    await websocket.accept()
    manager = get_manager()

    # 创建异步队列作为推送中介(桥接同步轮询线程与异步 WebSocket)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def push_callback(frame):
        """同步回调 → 推入异步队列"""
        try:
            loop.call_soon_threadsafe(queue.put_nowait, frame)
        except Exception:
            pass

    manager.add_client(push_callback)

    try:
        while True:
            try:
                # 等待数据帧(1 秒超时)
                frame = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(frame)
            except asyncio.TimeoutError:
                # 发送心跳
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        manager.remove_client(push_callback)
