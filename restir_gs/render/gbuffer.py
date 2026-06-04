from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from restir_gs.render.synthetic_scene import PinholeCamera

if TYPE_CHECKING:
    from restir_gs.render.gsplat_renderer import RenderBuffers


@dataclass(frozen=True)
class GBuffer:
    rgb: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    position_cam: torch.Tensor
    normal_cam: torch.Tensor
    valid_mask: torch.Tensor
    normal_mask: torch.Tensor


def unproject_depth_to_camera(
    depth: torch.Tensor,
    intrinsics_3x3: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Unproject z-depth into camera-space positions."""
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape [H,W], got {tuple(depth.shape)}")
    if intrinsics_3x3.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape [3,3], got {tuple(intrinsics_3x3.shape)}")
    if valid_mask.shape != depth.shape:
        raise ValueError(f"Expected valid mask shape {tuple(depth.shape)}, got {tuple(valid_mask.shape)}")

    height, width = depth.shape
    dtype = depth.dtype
    device = depth.device
    valid = valid_mask.to(torch.bool)

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )

    fx = intrinsics_3x3[0, 0].to(dtype=dtype, device=device)
    fy = intrinsics_3x3[1, 1].to(dtype=dtype, device=device)
    cx = intrinsics_3x3[0, 2].to(dtype=dtype, device=device)
    cy = intrinsics_3x3[1, 2].to(dtype=dtype, device=device)

    z = torch.where(valid, depth, torch.zeros_like(depth))
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return torch.stack((x, y, z), dim=-1)


def estimate_normals_from_position(
    position_cam: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate camera-space normals from screen-space position gradients."""
    if position_cam.ndim != 3 or position_cam.shape[-1] != 3:
        raise ValueError(f"Expected position shape [H,W,3], got {tuple(position_cam.shape)}")
    if valid_mask.shape != position_cam.shape[:2]:
        raise ValueError(f"Expected valid mask shape {tuple(position_cam.shape[:2])}, got {tuple(valid_mask.shape)}")

    height, width, _ = position_cam.shape
    normal_cam = torch.zeros_like(position_cam)
    normal_mask = torch.zeros((height, width), dtype=torch.bool, device=position_cam.device)
    if height < 3 or width < 3:
        return normal_cam, normal_mask

    valid = valid_mask.to(torch.bool)
    neighborhood_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
    )

    dx = position_cam[1:-1, 2:] - position_cam[1:-1, :-2]
    dy = position_cam[2:, 1:-1] - position_cam[:-2, 1:-1]
    normals = torch.cross(dx, dy, dim=-1)
    lengths = torch.linalg.norm(normals, dim=-1)
    usable = neighborhood_valid & torch.isfinite(lengths) & (lengths > 1e-8)

    unit_normals = torch.zeros_like(normals)
    unit_normals[usable] = normals[usable] / lengths[usable].unsqueeze(-1)
    flip = unit_normals[..., 2] < 0.0
    unit_normals[flip] = -unit_normals[flip]

    normal_cam[1:-1, 1:-1] = torch.where(usable[..., None], unit_normals, torch.zeros_like(unit_normals))
    normal_mask[1:-1, 1:-1] = usable
    return normal_cam, normal_mask


def make_pseudo_gbuffer(
    render_buffers: "RenderBuffers",
    camera: PinholeCamera,
    alpha_threshold: float = 1e-4,
) -> GBuffer:
    """Build a pseudo G-buffer from RGB, expected depth, and alpha buffers."""
    if camera.intrinsics.shape == (1, 3, 3):
        intrinsics = camera.intrinsics[0]
    elif camera.intrinsics.shape == (3, 3):
        intrinsics = camera.intrinsics
    else:
        raise ValueError(f"Expected camera intrinsics shape [1,3,3] or [3,3], got {tuple(camera.intrinsics.shape)}")

    finite_depth = torch.isfinite(render_buffers.depth) & (render_buffers.depth > 0.0)
    valid_mask = (render_buffers.alpha > alpha_threshold) & finite_depth
    position_cam = unproject_depth_to_camera(render_buffers.depth, intrinsics, valid_mask)
    normal_cam, normal_mask = estimate_normals_from_position(position_cam, valid_mask)

    return GBuffer(
        rgb=render_buffers.rgb,
        depth=render_buffers.depth,
        alpha=render_buffers.alpha,
        position_cam=position_cam,
        normal_cam=normal_cam,
        valid_mask=valid_mask,
        normal_mask=normal_mask,
    )
