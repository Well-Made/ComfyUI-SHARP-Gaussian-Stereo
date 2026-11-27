"""Dependency-light SHARP batch stereo node.

The renderer is a PyTorch forward splatter. It deliberately avoids gsplat,
CUDA extensions, browser automation, and per-frame PLY/PNG files. It is an
approximate renderer, not a pixel-identical port of gsplat.js.
"""

import logging
import math

import torch

from .sharp_inference import _predict_image_cached
from .sharp_model import _load_sharp_model
from .utils.image import convert_focallength

log = logging.getLogger("sharp")


def _camera_inputs(image, focal_length_mm, extrinsics, intrinsics, device):
    h, w = image.shape[:2]
    if extrinsics is not None and intrinsics is not None:
        ex = extrinsics
        intr = intrinsics
        if ex.dim() == 3:
            if ex.shape[0] != 1:
                raise ValueError("Connect one EXTRINSICS matrix or one matrix per frame.")
            ex = ex[0]
        if intr.dim() == 3:
            if intr.shape[0] != 1:
                raise ValueError("Connect one INTRINSICS matrix or one matrix per frame.")
            intr = intr[0]
        ex = ex.to(device)
        intr = intr.to(device)
        if intr.dim() == 2 and intr.shape == (3, 3):
            if float(intr[0, 0]) < 2.0:
                intr = intr.clone()
                intr[0, :] *= w
                intr[1, :] *= h
            k4 = torch.zeros((4, 4), dtype=intr.dtype, device=device)
            k4[:3, :3] = intr
            k4[3, 3] = 1.0
            intr = k4
        elif intr.shape != (4, 4):
            raise ValueError("Intrinsics must be a 3x3 or 4x4 matrix.")
        return float(intr[0, 0].item()), ex, intr
    fpx = convert_focallength(w, h, focal_length_mm if focal_length_mm > 0 else 30.0)
    return fpx, None, None


@torch.no_grad()
def infer_gaussians(model_config, image, focal_length_mm=30.0,
                    extrinsics=None, intrinsics=None):
    """Run SHARP for one frame and return world-space Gaussian tensors + f_px."""
    import comfy.model_management

    if not isinstance(model_config, dict):
        raise TypeError("SHARP model input is invalid. Connect (Down)Load SHARP Model.")
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError("Expected one frame with IMAGE shape [1,H,W,C].")
    patcher = _load_sharp_model(model_config)
    device = patcher.load_device
    frame_np = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    f_px, ex, intr = _camera_inputs(frame_np, focal_length_mm, extrinsics, intrinsics, device)
    gaussians = _predict_image_cached(patcher, frame_np, f_px, device, extrinsics=ex, intrinsics=intr)
    return gaussians, f_px


def _render_eye(gaussians, height, width, f_px, eye_x, target_z,
                toe_in, depth_scale, splat_size_scale, background, source):
    """Fast bilinear forward splat of SHARP Gaussian centers into one eye."""
    device = gaussians.mean_vectors.device
    means = gaussians.mean_vectors.reshape(-1, 3).float()
    colors = gaussians.colors.reshape(-1, 3).float().clamp(0, 1)
    opacity = gaussians.opacities.reshape(-1).float().clamp(0, 1)

    # Perceived-depth adjustment is radial about the original camera origin,
    # matching the viewer's depth-scale concept for the default SHARP camera.
    xyz = means.clone()
    if depth_scale != 1.0:
        xyz[:, 2] = target_z + (xyz[:, 2] - target_z) * depth_scale
        xyz[:, 0] *= depth_scale
        xyz[:, 1] *= depth_scale

    # Eye cameras are displaced along camera-right. Toe-in rotates toward the
    # viewer's convergence target; parallel keeps the optical axes unchanged.
    eye = torch.tensor([eye_x, 0.0, 0.0], device=device, dtype=torch.float32)
    rel = xyz - eye
    if toe_in:
        forward = torch.tensor([-eye_x, 0.0, target_z], device=device, dtype=torch.float32)
        forward = torch.nn.functional.normalize(forward, dim=0)
        right = torch.stack((forward[2], forward.new_tensor(0.0), -forward[0]))
        cam_x = (rel * right).sum(dim=1)
        cam_z = (rel * forward).sum(dim=1)
    else:
        cam_x = rel[:, 0]
        cam_z = rel[:, 2]

    valid = (torch.isfinite(cam_x) & torch.isfinite(cam_z) &
             torch.isfinite(rel[:, 1]) & torch.isfinite(opacity) &
             (cam_z > 0.05) & (opacity > 0.005))
    if not valid.any():
        return torch.full((height, width, 3), background, device=device, dtype=torch.float32)
    cam_x, cam_y, cam_z = cam_x[valid], rel[valid, 1], cam_z[valid]
    col, opa = colors[valid], opacity[valid]

    # Pinhole projection; SHARP's camera convention is +Z forward, +Y down.
    u = f_px * cam_x / cam_z + width * 0.5
    v = f_px * cam_y / cam_z + height * 0.5
    # A compact 3x3 Gaussian footprint approximates the viewer's splat-size
    # control without requiring a native rasterizer. Clamp the footprint to
    # keep the per-frame scatter workload bounded.
    sigma = max(0.35, min(1.75, 0.5 * float(splat_size_scale)))
    cx = torch.round(u).long()
    cy = torch.round(v).long()
    accum = torch.zeros((height * width, 3), device=device, dtype=torch.float32)
    weights = torch.zeros((height * width,), device=device, dtype=torch.float32)
    inv_two_sigma2 = 0.5 / (sigma * sigma)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            xx, yy = cx + ox, cy + oy
            ok = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            if not ok.any():
                continue
            distance2 = (xx.to(u.dtype) - u).square() + (yy.to(v.dtype) - v).square()
            footprint = torch.exp(-distance2 * inv_two_sigma2)
            idx = yy[ok] * width + xx[ok]
            weight = (footprint[ok] * opa[ok]).clamp_min(0)
            accum.index_add_(0, idx, col[ok] * weight[:, None])
            weights.index_add_(0, idx, weight)

    coverage = weights.clamp(0, 1)
    rendered = accum / weights.clamp_min(1e-6)[:, None]
    rendered = rendered * coverage[:, None]
    # Fill disocclusion holes with the corresponding source pixel rather than
    # leaving black gaps. This is a simple fallback, not true view synthesis.
    src = source.to(device=device, dtype=torch.float32).reshape(-1, 3)
    if src.shape[0] == height * width:
        rendered += src * (1.0 - coverage[:, None])
    else:
        rendered += torch.tensor(background, device=device)[None] * (1.0 - coverage[:, None])
    return rendered.reshape(height, width, 3).clamp(0, 1)


@torch.no_grad()
def render_stereo_batch_frame(gaussians, source, f_px, stereo_mode="toe-in",
                              stereo_intensity=30, perceived_depth_scale=1.0,
                              splat_size_scale=1.0, background_color="black"):
    h, w = source.shape[:2]
    means = gaussians.mean_vectors.reshape(-1, 3).float()
    z = means[:, 2]
    z = z[torch.isfinite(z) & (z > 0.05)]
    if z.numel() == 0:
        raise RuntimeError("SHARP returned no valid positive-depth Gaussians.")
    zmin = torch.quantile(z, 0.01)
    target_z = max(0.1, float(zmin.item()) * 1.5)
    intensity = max(-100.0, min(100.0, float(stereo_intensity))) / 100.0
    baseline = target_z * 0.12 * intensity
    bg = (0.0, 0.0, 0.0) if background_color == "black" else (1.0, 1.0, 1.0)
    # Positive slider values retain normal left/right order; negative values
    # reverse the baseline, corresponding to cross-eyed viewing.
    left = _render_eye(gaussians, h, w, f_px, -baseline * 0.5, target_z,
                       stereo_mode == "toe-in", perceived_depth_scale,
                       splat_size_scale, bg, source)
    right = _render_eye(gaussians, h, w, f_px, baseline * 0.5, target_z,
                        stereo_mode == "toe-in", perceived_depth_scale,
                        splat_size_scale, bg, source)
    return torch.cat((left, right), dim=1)
