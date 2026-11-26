"""Single-image SHARP inference and PLY export.

Adapted from ComfyUI-Sharp (GPL-3.0), nodes/predict.py. The comfy-env
registration/proxy layer and unrelated batch/depth outputs are intentionally
not used here.
"""

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .sharp.gaussians import save_ply, unproject_gaussians
from .sharp.model import MonodepthOutput
from .sharp_model import _load_sharp_model
from .utils.image import convert_focallength

log = logging.getLogger("sharp")
_encode_cache = {
    "image_hash": None,
    "monodepth_output": None,
    "image_resized": None,
    "patcher_id": None,
}


def _monodepth_to(output, device):
    return MonodepthOutput(
        disparity=output.disparity.to(device),
        encoder_features=[value.to(device) for value in output.encoder_features],
        decoder_features=output.decoder_features.to(device),
        output_features=[value.to(device) for value in output.output_features],
        intermediate_features=[value.to(device) for value in output.intermediate_features],
    )


def _predict_image_cached(patcher, image_np, f_px, device, extrinsics=None, intrinsics=None, *, quiet=False):
    import comfy.model_management

    report = log.debug if quiet else log.info
    internal_shape = (1536, 1536)
    memory_required = patcher.memory_required([1, 3, internal_shape[0], internal_shape[1]])
    comfy.model_management.load_models_gpu([patcher], memory_required=memory_required)
    predictor = patcher.model
    height, width = image_np.shape[:2]
    image_hash = hashlib.sha256(image_np.tobytes()).hexdigest()[:16]
    patcher_id = id(patcher)

    if (_encode_cache["image_hash"] == image_hash
            and _encode_cache["patcher_id"] == patcher_id):
        monodepth_output = _monodepth_to(_encode_cache["monodepth_output"], device)
        image_resized = _encode_cache["image_resized"].to(device)
    else:
        _encode_cache.update(image_hash=None, monodepth_output=None, image_resized=None, patcher_id=None)
        # Match the neural-network input to the actual loaded parameter dtype.
        # Keep image normalization and resizing in FP32, then cast once at the
        # model boundary to avoid manual_cast promoting the ViT layers back to FP32.
        model_dtype = next(
            (parameter.dtype for parameter in predictor.parameters()
             if parameter.is_floating_point()),
            torch.float32,
        )
        image_tensor = torch.from_numpy(image_np.copy()).float().to(device).permute(2, 0, 1) / 255.0
        image_resized = F.interpolate(
            image_tensor[None], size=internal_shape, mode="bilinear", align_corners=True
        ).to(dtype=model_dtype)
        started = time.time()
        monodepth_output, _ = predictor.encode(image_resized)
        report("SHARP encode completed in %.2fs", time.time() - started)
        _encode_cache["image_hash"] = image_hash
        _encode_cache["patcher_id"] = patcher_id
        _encode_cache["monodepth_output"] = _monodepth_to(monodepth_output, "cpu")
        _encode_cache["image_resized"] = image_resized.cpu()
        comfy.model_management.soft_empty_cache()

    disparity_factor = torch.tensor(
        [f_px / width], dtype=image_resized.dtype, device=device
    )
    started = time.time()
    gaussians_ndc = predictor.decode(monodepth_output, image_resized, disparity_factor)
    report("SHARP decode completed in %.2fs", time.time() - started)

    if intrinsics is not None:
        unproj_intrinsics = intrinsics.clone()
    else:
        unproj_intrinsics = torch.tensor(
            [[f_px, 0, width / 2, 0], [0, f_px, height / 2, 0],
             [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32, device=device,
        )
    intrinsics_resized = unproj_intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height
    unproj_extrinsics = extrinsics if extrinsics is not None else torch.eye(4, device=device)
    return unproject_gaussians(
        gaussians_ndc, unproj_extrinsics, intrinsics_resized, internal_shape
    )


@torch.no_grad()
def run_sharp_inference(model_config, image, output_dir, focal_length_mm=30.0,
                        extrinsics=None, intrinsics=None, *, quiet=False):
    import comfy.model_management

    if not isinstance(model_config, dict):
        raise TypeError("SHARP model input is invalid. Connect (Down)Load SHARP Model.")
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError("SHARP Gaussian Stereo Preview expects one IMAGE with shape [1,H,W,C].")

    patcher = _load_sharp_model(model_config)
    device = patcher.load_device
    image_np = (image[0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    height, width = image_np.shape[:2]
    use_camera = extrinsics is not None and intrinsics is not None

    if use_camera:
        if extrinsics.dim() == 3:
            if extrinsics.shape[0] != 1:
                raise ValueError("SHARP preview accepts one extrinsics matrix.")
            extrinsics = extrinsics[0]
        if intrinsics.dim() == 3:
            if intrinsics.shape[0] != 1:
                raise ValueError("SHARP preview accepts one intrinsics matrix.")
            intrinsics = intrinsics[0]
        img_extrinsics = extrinsics.to(device)
        img_intrinsics = intrinsics.to(device)
        if img_intrinsics.dim() == 2 and img_intrinsics.shape == (3, 3):
            if float(img_intrinsics[0, 0]) < 2.0:
                img_intrinsics = img_intrinsics.clone()
                img_intrinsics[0, :] *= float(width)
                img_intrinsics[1, :] *= float(height)
            K4 = torch.zeros((4, 4), dtype=img_intrinsics.dtype, device=device)
            K4[:3, :3] = img_intrinsics
            K4[3, 3] = 1.0
            img_intrinsics = K4
        elif img_intrinsics.shape != (4, 4):
            raise ValueError("Intrinsics must be a 3x3 or 4x4 matrix.")
        f_px = float(img_intrinsics[0, 0].item())
    else:
        f_px = convert_focallength(width, height, focal_length_mm if focal_length_mm > 0 else 30.0)
        img_extrinsics = None
        img_intrinsics = None

    report = log.debug if quiet else log.info
    report("Running SHARP inference for %dx%d image", width, height)
    gaussians = _predict_image_cached(
        patcher, image_np, f_px, device,
        extrinsics=img_extrinsics, intrinsics=img_intrinsics, quiet=quiet,
    )

    # Batch rendering requests an in-memory PLY; single-image previews retain file export.
    output_path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / f"gaussian_stereo_{int(time.time() * 1000)}.ply"
    payload, metadata = save_ply(gaussians, f_px, (height, width), output_path)
    if output_path is None:
        report("SHARP PLY in RAM: %.1f MiB (%s gaussians)", len(payload) / 1048576,
                 f"{metadata['num_gaussians']:,}")
    else:
        report("Saved SHARP PLY: %s (%s gaussians)", output_path, f"{metadata['num_gaussians']:,}")
    comfy.model_management.soft_empty_cache()
    return (payload if output_path is None else str(output_path)), metadata["extrinsic"], metadata["intrinsic"]
