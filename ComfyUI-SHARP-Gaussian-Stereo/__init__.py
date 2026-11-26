"""ComfyUI SHARP Gaussian Stereo custom nodes."""

from .gaussian_stereo.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .gaussian_stereo.batch_bridge import register_routes
try:
    register_routes()
except Exception as exc:
    print(f"[SHARP Gaussian Stereo] Could not register batch bridge routes: {exc}")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
