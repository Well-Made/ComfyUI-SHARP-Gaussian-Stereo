"""Modified ComfyUI node integration based on ComfyUI-Sharp (GPL-3.0).

ComfyUI node backend for the standalone Gaussian Stereo Preview extension.

The camera inputs retain the established custom socket type names
EXTRINSICS and INTRINSICS for workflow compatibility. They are not STRING inputs.
"""

import base64
import math
from pathlib import Path
import threading
import uuid

import numpy as np
import torch
from PIL import Image
from io import BytesIO

from aiohttp import web
from server import PromptServer

import folder_paths


_TEMP_PLY_BY_NODE = {}
_TEMP_PLY_LOCK = threading.Lock()
_CAPTURE_LOCK = threading.Lock()
_PENDING_CAPTURES = {}
_CAPTURE_TIMEOUT_SECONDS = 120


async def _receive_capture(request):
    run_id = request.query.get("run_id", "")
    if not run_id:
        return web.json_response({"error": "Missing run_id"}, status=400)

    with _CAPTURE_LOCK:
        pending = _PENDING_CAPTURES.get(run_id)
    if pending is None:
        return web.json_response({"error": "Unknown or expired capture request"}, status=404)

    try:
        payload = await request.json()
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        data_url = payload.get("image", "")
        if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
            raise ValueError("Expected a PNG data URL")
        encoded = data_url.split(",", 1)[1]
        image_bytes = base64.b64decode(encoded, validate=True)
        if not image_bytes or len(image_bytes) > 64 * 1024 * 1024:
            raise ValueError("PNG payload is empty or exceeds the 64 MB limit")
        if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Invalid PNG data")

        with _CAPTURE_LOCK:
            current = _PENDING_CAPTURES.get(run_id)
            if current is None:
                return web.json_response({"error": "Capture request expired"}, status=410)
            current["image_bytes"] = image_bytes
            current["event"].set()
        return web.json_response({"received": True})
    except Exception as exc:
        with _CAPTURE_LOCK:
            current = _PENDING_CAPTURES.get(run_id)
            if current is not None:
                current["error"] = str(exc)
                current["event"].set()
        return web.json_response({"error": str(exc)}, status=400)


# Register the upload endpoint once when this custom-node module is imported.
try:
    PromptServer.instance.routes.post("/sharp_gaussian_stereo/capture")(_receive_capture)
except Exception as _route_error:
    # ComfyUI may reload custom-node modules during development. Avoid making a
    # duplicate-route warning fatal; the existing route remains registered.
    print(f"[Gaussian Stereo] Capture route registration note: {_route_error}")


class GaussianStereoPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to a Gaussian Splatting PLY file inside ComfyUI input, output, or temp.",
                }),
            },
            "optional": {
                # Keep the established camera socket identifiers for compatibility.
                "extrinsics": ("EXTRINSICS",),
                "intrinsics": ("INTRINSICS",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "utils/preview"
    OUTPUT_NODE = True

    @staticmethod
    def _allowed_roots():
        roots = []
        for getter in (
            folder_paths.get_input_directory,
            folder_paths.get_output_directory,
            folder_paths.get_temp_directory,
        ):
            try:
                roots.append(Path(getter()).resolve())
            except Exception:
                pass
        return roots

    @classmethod
    def _resolve_ply_path(cls, raw_path):
        value = str(raw_path or "").strip().strip('"')
        if not value:
            raise ValueError("PLY path is empty")
        candidate = Path(value).expanduser()
        roots = cls._allowed_roots()
        candidates = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
        for item in candidates:
            try:
                path = item.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if (path.is_file() and path.suffix.lower() == ".ply"
                    and any(path == root or root in path.parents for root in roots)):
                return path
        raise ValueError(
            "PLY file was not found inside ComfyUI input, output, or temp directories: "
            f"{value}"
        )

    @staticmethod
    def _to_builtin(value):
        """Convert common matrix containers (numpy arrays/tuples) to JSON-like lists."""
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [GaussianStereoPreview._to_builtin(item) for item in value]
        # Convert numpy and other numeric scalar types without importing numpy.
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Camera matrices must contain only finite numbers")
            return number
        raise ValueError(f"Unsupported camera matrix value: {type(value).__name__}")

    @classmethod
    def _validate_matrix(cls, value, name, rows, cols):
        if value is None:
            return None
        matrix = cls._to_builtin(value)
        if (not isinstance(matrix, list) or len(matrix) != rows or
                any(not isinstance(row, list) or len(row) != cols for row in matrix)):
            raise ValueError(f"{name} must be a {rows}x{cols} matrix")
        return matrix

    @classmethod
    def IS_CHANGED(cls, ply_path, extrinsics=None, intrinsics=None):
        try:
            path = cls._resolve_ply_path(ply_path)
            stat = path.stat()
            # Include camera values so changes to connected upstream nodes trigger refresh.
            return repr((path, stat.st_mtime_ns, stat.st_size, extrinsics, intrinsics))
        except Exception:
            return float("nan")

    def preview(self, ply_path, extrinsics=None, intrinsics=None):
        try:
            path = self._resolve_ply_path(ply_path)
            extrinsics_data = self._validate_matrix(extrinsics, "extrinsics", 4, 4)
            intrinsics_data = self._validate_matrix(intrinsics, "intrinsics", 3, 3)
            size_mb = path.stat().st_size / (1024 * 1024)

            ui_data = {
                "ply_file": [str(path)],
                "filename": [path.name],
                "file_size_mb": [round(size_mb, 2)],
            }
            if extrinsics_data is not None:
                ui_data["extrinsics"] = [extrinsics_data]
            if intrinsics_data is not None:
                ui_data["intrinsics"] = [intrinsics_data]
            return {"ui": ui_data, "result": ()}
        except Exception as exc:
            return {"ui": {"error": [str(exc)]}, "result": ()}


class SharpGaussianStereoPreview(GaussianStereoPreview):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SHARP_MODEL_CONFIG",),
                "image": ("IMAGE",),
                "focal_length_mm": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 500.0, "step": 0.1,
                    "tooltip": (
                        "SHARP camera focal length in millimeters. Changes the "
                        "reconstructed geometry and requires a new inference. "
                        "Perceived Depth Scale provides a real-time viewer-side alternative."
                    ),
                }),
                "save_to_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "extrinsics": ("EXTRINSICS",),
                "intrinsics": ("INTRINSICS",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "utils/preview"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, model, image, focal_length_mm=30.0,
                   save_to_output=True, extrinsics=None, intrinsics=None, unique_id=None):
        return float("nan")

    @staticmethod
    def _send_capture_request(run_id, node_id, path, extrinsics_data, intrinsics_data):
        PromptServer.instance.send_sync("sharp_gaussian_capture_request", {
            "run_id": run_id,
            "node_id": str(node_id),
            "ply_file": str(path),
            "filename": path.name,
            "extrinsics": extrinsics_data,
            "intrinsics": intrinsics_data,
        })

    @staticmethod
    def _decode_capture(image_bytes):
        with Image.open(BytesIO(image_bytes)) as pil_image:
            rgb = pil_image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).to(dtype=torch.float32).div_(255.0)
        return tensor.unsqueeze(0)

    def run(self, model, image, focal_length_mm=30.0,
            save_to_output=True, extrinsics=None, intrinsics=None, unique_id=None):
        if getattr(image, "ndim", None) != 4:
            raise ValueError("Expected IMAGE tensor with shape [B,H,W,C]")
        if image.shape[0] != 1:
            raise ValueError("SHARP Gaussian Stereo Preview currently supports one image per run")
        if unique_id is None:
            raise RuntimeError("ComfyUI did not provide the preview node ID; cannot request the browser capture")

        run_id = uuid.uuid4().hex
        capture_event = threading.Event()
        pending = {"event": capture_event, "image_bytes": None, "error": None}
        with _CAPTURE_LOCK:
            _PENDING_CAPTURES[run_id] = pending

        try:
            output_dir = (folder_paths.get_output_directory() if save_to_output
                          else folder_paths.get_temp_directory())
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            from .sharp_inference import run_sharp_inference
            ply_value, sharp_extrinsics, sharp_intrinsics = run_sharp_inference(
                model_config=model, image=image, output_dir=output_dir,
                focal_length_mm=focal_length_mm, extrinsics=extrinsics, intrinsics=intrinsics,
            )
            path = self._resolve_ply_path(ply_value)
            if not save_to_output:
                node_key = str(unique_id)
                temp_root = Path(folder_paths.get_temp_directory()).resolve()
                with _TEMP_PLY_LOCK:
                    previous = _TEMP_PLY_BY_NODE.get(node_key)
                    if previous is not None and previous != path:
                        try:
                            previous_resolved = Path(previous).resolve()
                            if (previous_resolved.parent == temp_root
                                    and previous_resolved.name.startswith("gaussian_stereo_")
                                    and previous_resolved.suffix.lower() == ".ply"):
                                previous_resolved.unlink(missing_ok=True)
                        except OSError as cleanup_error:
                            print(f"[Gaussian Stereo] Could not remove previous temp PLY {previous}: {cleanup_error}")
                    _TEMP_PLY_BY_NODE[node_key] = path

            extrinsics_data = self._validate_matrix(sharp_extrinsics, "extrinsics", 4, 4)
            intrinsics_data = self._validate_matrix(sharp_intrinsics, "intrinsics", 3, 3)

            # Ask the browser viewer to load this run's PLY and capture its initial
            # composed canvas. The node waits on a worker thread, not the aiohttp loop.
            self._send_capture_request(run_id, unique_id, path, extrinsics_data, intrinsics_data)
            if not capture_event.wait(_CAPTURE_TIMEOUT_SECONDS):
                raise TimeoutError(
                    "Timed out waiting for the browser viewer to produce the initial stereo image. "
                    "Make sure the ComfyUI browser tab is open and the preview node is visible."
                )
            if pending["error"]:
                raise RuntimeError(f"Browser capture failed: {pending['error']}")
            if not pending["image_bytes"]:
                raise RuntimeError("Browser capture completed without returning an image")

            output_image = self._decode_capture(pending["image_bytes"])
            ui_data = {
                "ply_file": [str(path)],
                "filename": [path.name],
                "file_size_mb": [round(path.stat().st_size / (1024 * 1024), 2)],
                "save_to_output": [bool(save_to_output)],
            }
            if extrinsics_data is not None:
                ui_data["extrinsics"] = [extrinsics_data]
            if intrinsics_data is not None:
                ui_data["intrinsics"] = [intrinsics_data]
            return {"ui": ui_data, "result": (output_image,)}
        finally:
            with _CAPTURE_LOCK:
                _PENDING_CAPTURES.pop(run_id, None)



from .sharp_model import LoadSharpModel

NODE_CLASS_MAPPINGS = {
    "LoadSharpModel": LoadSharpModel,
    "SharpGaussianStereoPreview": SharpGaussianStereoPreview,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSharpModel": "(Down)Load SHARP Model",
    "SharpGaussianStereoPreview": "SHARP Gaussian Stereo Preview",
}
