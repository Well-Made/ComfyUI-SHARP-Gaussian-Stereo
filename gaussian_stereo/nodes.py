"""Modified ComfyUI node integration based on ComfyUI-Sharp (GPL-3.0).

ComfyUI node backend for the standalone Gaussian Stereo Preview extension.

The camera inputs retain the established custom socket type names
EXTRINSICS and INTRINSICS for workflow compatibility. They are not STRING inputs.
"""

import math
from pathlib import Path
import threading

import folder_paths


_TEMP_PLY_BY_NODE = {}
_TEMP_PLY_LOCK = threading.Lock()


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

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "utils/preview"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, model, image, focal_length_mm=30.0,
                   save_to_output=True, extrinsics=None, intrinsics=None, unique_id=None):
        return float("nan")

    def run(self, model, image, focal_length_mm=30.0,
            save_to_output=True, extrinsics=None, intrinsics=None, unique_id=None):
        try:
            if getattr(image, "ndim", None) != 4:
                raise ValueError("Expected IMAGE tensor with shape [B,H,W,C]")
            if image.shape[0] != 1:
                raise ValueError("SHARP Gaussian Stereo Preview currently supports one image per run")
            output_dir = (folder_paths.get_output_directory() if save_to_output
                          else folder_paths.get_temp_directory())
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            from .sharp_inference import run_sharp_inference
            ply_value, sharp_extrinsics, sharp_intrinsics = run_sharp_inference(
                model_config=model, image=image, output_dir=output_dir,
                focal_length_mm=focal_length_mm, extrinsics=extrinsics, intrinsics=intrinsics,
            )
            path = self._resolve_ply_path(ply_value)
            if not save_to_output and unique_id is not None:
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
            return {"ui": ui_data, "result": ()}
        except Exception as exc:
            return {"ui": {"error": [str(exc)]}, "result": ()}


from .sharp_model import LoadSharpModel

NODE_CLASS_MAPPINGS = {
    "LoadSharpModel": LoadSharpModel,
    "SharpGaussianStereoPreview": SharpGaussianStereoPreview,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSharpModel": "(Down)Load SHARP Model",
    "SharpGaussianStereoPreview": "SHARP Gaussian Stereo Preview",
}
