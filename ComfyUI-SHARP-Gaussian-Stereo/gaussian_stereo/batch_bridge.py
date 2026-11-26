"""Browser-render bridge for the original gsplat viewer."""
import asyncio
import base64
import threading
import time
import uuid
from aiohttp import web
from server import PromptServer

_pending = {}
_lock = threading.Lock()
_registered = False
_dedicated_seen = None


def register_routes():
    global _registered
    if _registered:
        return
    routes = PromptServer.instance.routes
    routes.get("/sharp_stereo_batch/next")(_next)
    routes.post("/sharp_stereo_batch/complete")(_complete)
    routes.get("/sharp_stereo_batch/ply/{job_id}.ply")(_ply)
    _registered = True


async def _next(request):
    global _dedicated_seen
    node_id = request.query.get("node_id")
    shared_worker = request.query.get("worker") == "1"
    dedicated = shared_worker and request.query.get("dedicated") == "1"
    # Long polling keeps browser timers out of the normal frame dispatch path.
    # Short requests from old viewers remain compatible.
    deadline = time.monotonic() + (20.0 if shared_worker and request.query.get("wait") == "1" else 0.0)
    while True:
        with _lock:
            now = time.monotonic()
            if dedicated:
                _dedicated_seen = now
            prefer_dedicated = (not dedicated and shared_worker and _dedicated_seen is not None
                                and now - _dedicated_seen < 10)
            if not prefer_dedicated:
                for item in _pending.values():
                    if (shared_worker or item["job"]["node_id"] == node_id) and not item["claimed"] and not item["event"].is_set():
                        item["claimed"] = True
                        return web.json_response({"job": item["job"]}, headers={"Cache-Control": "no-store"})
        if now >= deadline:
            return web.json_response({"job": None}, headers={"Cache-Control": "no-store"})
        transport = request.transport
        if transport is None or transport.is_closing():
            return web.json_response({"job": None})
        await asyncio.sleep(min(0.05, deadline - now))


async def _ply(request):
    # Only active, claimed jobs can be fetched; no arbitrary filesystem access.
    jid = request.match_info["job_id"]
    with _lock:
        item = _pending.get(jid)
        if item is None or not item["claimed"] or item["event"].is_set():
            raise web.HTTPNotFound(headers={"Cache-Control": "no-store"})
        payload = item["ply"]
    # The bytes are immutable. Keep a local reference while aiohttp sends the body.
    return web.Response(body=payload, content_type="application/octet-stream",
                        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})


async def _complete(request):
    data = await request.json()
    jid = str(data.get("id", ""))
    with _lock:
        item = _pending.get(jid)
        if item is None or item["event"].is_set():
            return web.json_response({"ok": False, "error": "Unknown/expired job"}, status=404)
        try:
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            raw = base64.b64decode(data["png"], validate=True)
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Invalid PNG")
            item["result"] = raw
        except Exception as exc:
            item["error"] = str(exc)
        finally:
            item["event"].set()
    return web.json_response({"ok": True})


def capture(ply_bytes, extrinsics, intrinsics, node_id, timeout=180, stereo=None):
    import comfy.model_management
    jid = uuid.uuid4().hex
    if not isinstance(ply_bytes, bytes) or not ply_bytes.startswith(b"ply\n"):
        raise ValueError("Expected binary PLY bytes")
    url = f"/sharp_stereo_batch/ply/{jid}.ply"
    job = {"id": jid, "node_id": str(node_id), "url": url, "filename": f"{jid}.ply",
           "extrinsics": extrinsics, "intrinsics": intrinsics, "timeout": timeout,
           "stereo": stereo}
    item = {"event": threading.Event(), "result": None, "error": None,
            "job": job, "claimed": False, "ply": ply_bytes}
    with _lock:
        _pending[jid] = item
    try:
        deadline = time.monotonic() + timeout
        while not item["event"].is_set():
            comfy.model_management.throw_exception_if_processing_interrupted()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for SHARP shared renderer. Keep a ComfyUI page or the dedicated SHARP renderer page open with WebGL enabled. Background browser throttling may require keeping the renderer window visible.")
            item["event"].wait(min(remaining, 0.25))
        if item["error"]:
            raise RuntimeError(item["error"])
        return item["result"]
    finally:
        with _lock:
            _pending.pop(jid, None)
