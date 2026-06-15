from __future__ import annotations

from dataclasses import dataclass

import torch

from restir_gs.render.camera_sequence import look_at_viewmat
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.synthetic_scene import PinholeCamera, SyntheticGaussians


@dataclass(frozen=True)
class ShadowMapBundle:
    light_indices: torch.Tensor
    light_cameras: list[PinholeCamera]
    depth_maps: torch.Tensor
    alpha_maps: torch.Tensor
    scene_radius: float
    depth_bias: float



def make_shadow_map_bundle(
    scene: SyntheticGaussians,
    world_light_positions: torch.Tensor,
    light_indices: torch.Tensor,
    target_world: torch.Tensor,
    scene_radius: float,
    resolution: int = 128,
    shadow_bias_scale: float = 0.02,
    focal_scale: float = 1.5,
) -> ShadowMapBundle:
    """Render expected-depth shadow maps from selected world-space point lights."""
    if resolution <= 0:
        raise ValueError(f"Expected positive shadow resolution, got {resolution}")
    if scene_radius <= 0.0:
        raise ValueError(f"Expected positive scene_radius, got {scene_radius}")
    if shadow_bias_scale < 0.0:
        raise ValueError(f"Expected non-negative shadow_bias_scale, got {shadow_bias_scale}")
    if focal_scale <= 0.0:
        raise ValueError(f"Expected positive focal_scale, got {focal_scale}")
    if light_indices.ndim != 1:
        raise ValueError(f"Expected light_indices shape [L], got {tuple(light_indices.shape)}")
    if world_light_positions.ndim != 2 or world_light_positions.shape[-1] != 3:
        raise ValueError(f"Expected world light positions shape [N,3], got {tuple(world_light_positions.shape)}")
    if target_world.shape != (3,):
        raise ValueError(f"Expected target_world shape [3], got {tuple(target_world.shape)}")
    if light_indices.numel() > 0:
        if int(light_indices.min().detach().cpu()) < 0 or int(light_indices.max().detach().cpu()) >= world_light_positions.shape[0]:
            raise ValueError("Shadow light indices must be in [0, light_count).")

    device = world_light_positions.device
    dtype = world_light_positions.dtype
    target = target_world.to(device=device, dtype=dtype)
    focal = float(resolution) * float(focal_scale)
    intrinsics = torch.tensor(
        [[[focal, 0.0, resolution * 0.5], [0.0, focal, resolution * 0.5], [0.0, 0.0, 1.0]]],
        dtype=dtype,
        device=device,
    )

    light_cameras: list[PinholeCamera] = []
    depth_maps: list[torch.Tensor] = []
    alpha_maps: list[torch.Tensor] = []
    for light_index in light_indices.to(device=device, dtype=torch.long):
        eye = world_light_positions[light_index]
        light_camera = PinholeCamera(
            viewmats=look_at_viewmat(eye, target)[None],
            intrinsics=intrinsics.clone(),
            width=resolution,
            height=resolution,
        )
        render = render_rgbd(scene, light_camera)
        light_cameras.append(light_camera)
        depth_maps.append(render.depth)
        alpha_maps.append(render.alpha)

    empty_depth = torch.empty((0, resolution, resolution), dtype=dtype, device=device)
    empty_alpha = torch.empty((0, resolution, resolution), dtype=dtype, device=device)
    return ShadowMapBundle(
        light_indices=light_indices.to(device=device, dtype=torch.long),
        light_cameras=light_cameras,
        depth_maps=torch.stack(depth_maps, dim=0) if depth_maps else empty_depth,
        alpha_maps=torch.stack(alpha_maps, dim=0) if alpha_maps else empty_alpha,
        scene_radius=float(scene_radius),
        depth_bias=float(shadow_bias_scale) * float(scene_radius),
    )



def make_light_camera(
    light_position_world: torch.Tensor,
    target_world: torch.Tensor,
    resolution: int = 128,
    focal_scale: float = 1.5,
) -> PinholeCamera:
    """Create one shadow-map camera with camera +Z looking from the light to the target."""
    if resolution <= 0:
        raise ValueError(f"Expected positive shadow resolution, got {resolution}")
    if focal_scale <= 0.0:
        raise ValueError(f"Expected positive focal_scale, got {focal_scale}")
    if light_position_world.shape != (3,) or target_world.shape != (3,):
        raise ValueError(
            f"Expected light_position_world and target_world shape [3], got {tuple(light_position_world.shape)} "
            f"and {tuple(target_world.shape)}"
        )
    dtype = light_position_world.dtype
    device = light_position_world.device
    focal = float(resolution) * float(focal_scale)
    intrinsics = torch.tensor(
        [[[focal, 0.0, resolution * 0.5], [0.0, focal, resolution * 0.5], [0.0, 0.0, 1.0]]],
        dtype=dtype,
        device=device,
    )
    return PinholeCamera(
        viewmats=look_at_viewmat(light_position_world, target_world)[None],
        intrinsics=intrinsics,
        width=resolution,
        height=resolution,
    )

