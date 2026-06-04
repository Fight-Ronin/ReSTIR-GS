from __future__ import annotations

from dataclasses import dataclass

import torch
from gsplat import rasterization

from restir_gs.render.synthetic_scene import PinholeCamera, SyntheticGaussians


@dataclass(frozen=True)
class RenderBuffers:
    rgb: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor


@torch.no_grad()
def render_rgbd(scene: SyntheticGaussians, camera: PinholeCamera) -> RenderBuffers:
    """Render RGB, expected depth, and alpha from synthetic Gaussians."""
    renders, alphas, _ = rasterization(
        scene.means,
        scene.quats,
        scene.scales,
        scene.opacities,
        scene.colors,
        camera.viewmats,
        camera.intrinsics,
        width=camera.width,
        height=camera.height,
        render_mode="RGB+ED",
    )

    if renders.shape != (1, camera.height, camera.width, 4):
        raise RuntimeError(f"Expected RGB+ED render shape (1,H,W,4), got {tuple(renders.shape)}")
    if alphas.shape != (1, camera.height, camera.width, 1):
        raise RuntimeError(f"Expected alpha shape (1,H,W,1), got {tuple(alphas.shape)}")

    return RenderBuffers(
        rgb=renders[0, ..., :3].contiguous(),
        depth=renders[0, ..., 3].contiguous(),
        alpha=alphas[0, ..., 0].contiguous(),
    )
