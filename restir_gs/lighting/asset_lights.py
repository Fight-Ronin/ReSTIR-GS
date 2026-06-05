from __future__ import annotations

from dataclasses import dataclass

import torch

from restir_gs.lighting.deferred import PointLights
from restir_gs.render.gbuffer import GBuffer
from restir_gs.render.synthetic_scene import PinholeCamera


@dataclass(frozen=True)
class WorldPointLights:
    positions_world: torch.Tensor
    colors: torch.Tensor
    intensities: torch.Tensor


def make_asset_scaled_point_lights(
    gbuffer: GBuffer,
    count: int,
    seed: int = 2027,
    device: torch.device | str | None = None,
) -> tuple[PointLights, dict[str, object]]:
    """Create deterministic point lights scaled to a real asset G-buffer."""
    if count <= 0:
        raise ValueError(f"Expected positive light count, got {count}")

    valid = gbuffer.valid_mask & gbuffer.normal_mask
    positions = gbuffer.position_cam[valid].detach()
    if positions.numel() == 0:
        raise RuntimeError("Cannot create asset-scaled lights without valid G-buffer positions.")

    pos_min = positions.min(dim=0).values
    pos_max = positions.max(dim=0).values
    center = (pos_min + pos_max) * 0.5
    extent = (pos_max - pos_min).clamp_min(1e-3)
    xy_radius = torch.max(extent[:2]).clamp_min(1.0) * 0.75
    depth_span = extent[2].clamp_min(1.0)
    light_scale = torch.max(xy_radius, depth_span)
    z_min = (pos_min[2] - depth_span * 0.25).clamp_min(0.05)
    z_max = pos_max[2] + depth_span * 0.75

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    xy = (torch.rand((count, 2), generator=generator, dtype=torch.float32) * 2.0 - 1.0) * float(xy_radius.detach().cpu())
    z = torch.rand((count, 1), generator=generator, dtype=torch.float32) * float((z_max - z_min).detach().cpu())
    z = z + float(z_min.detach().cpu())
    light_positions = torch.cat((xy, z), dim=-1)
    light_positions[:, 0] += float(center[0].detach().cpu())
    light_positions[:, 1] += float(center[1].detach().cpu())

    colors = torch.rand((count, 3), generator=generator, dtype=torch.float32) * 0.6 + 0.4
    intensity = 3.0 * float((light_scale * light_scale).detach().cpu()) / float(count)
    intensities = torch.full((count,), intensity, dtype=torch.float32)

    info = {
        "mode": "asset_scaled_camera_space",
        "seed": seed,
        "position_min": _tuple3(light_positions.min(dim=0).values),
        "position_max": _tuple3(light_positions.max(dim=0).values),
        "xy_radius": float(xy_radius.detach().cpu()),
        "z_min": float(z_min.detach().cpu()),
        "z_max": float(z_max.detach().cpu()),
        "intensity_per_light": intensity,
    }
    target_device = gbuffer.rgb.device if device is None else torch.device(device)
    return (
        PointLights(
            positions_cam=light_positions.to(target_device),
            colors=colors.to(target_device),
            intensities=intensities.to(target_device),
        ),
        info,
    )


def make_asset_scaled_world_lights(
    means: torch.Tensor,
    count: int,
    seed: int = 2027,
    bbox_percentile: float = 0.98,
    radius_scale: float = 1.25,
    device: torch.device | str | None = None,
) -> tuple[WorldPointLights, dict[str, object]]:
    """Create deterministic scene-stable point lights around Gaussian means."""
    if count <= 0:
        raise ValueError(f"Expected positive light count, got {count}")
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"Expected means shape [N,3], got {tuple(means.shape)}")
    if means.shape[0] <= 0:
        raise ValueError("Expected at least one Gaussian mean.")
    if not 0.0 < bbox_percentile <= 1.0:
        raise ValueError(f"Expected bbox_percentile in (0,1], got {bbox_percentile}")
    if radius_scale <= 0.0:
        raise ValueError(f"Expected positive radius_scale, got {radius_scale}")

    means_cpu = means.detach().cpu().float()
    if not bool(torch.isfinite(means_cpu).all()):
        raise ValueError("Expected finite Gaussian means.")

    bbox_min, bbox_max = _robust_bbox(means_cpu, bbox_percentile)
    center = (bbox_min + bbox_max) * 0.5
    half_extent = (bbox_max - bbox_min) * 0.5
    radius = torch.linalg.norm(half_extent).clamp_min(1e-3) * float(radius_scale)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    directions = torch.randn((count, 3), generator=generator, dtype=torch.float32)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(1e-8)
    light_positions = center[None, :] + directions * radius
    colors = torch.rand((count, 3), generator=generator, dtype=torch.float32) * 0.6 + 0.4
    intensity = 3.0 * float((radius * radius).detach().cpu()) / float(count)
    intensities = torch.full((count,), intensity, dtype=torch.float32)

    info = {
        "mode": "asset_scaled_world_space",
        "light_space": "world",
        "light_policy": "asset_scaled_spherical_shell",
        "seed": seed,
        "bbox_percentile": float(bbox_percentile),
        "radius_scale": float(radius_scale),
        "bbox_min": _tuple3(bbox_min),
        "bbox_max": _tuple3(bbox_max),
        "center": _tuple3(center),
        "radius": float(radius.detach().cpu()),
        "position_min": _tuple3(light_positions.min(dim=0).values),
        "position_max": _tuple3(light_positions.max(dim=0).values),
        "intensity_per_light": intensity,
    }
    target_device = means.device if device is None else torch.device(device)
    return (
        WorldPointLights(
            positions_world=light_positions.to(target_device),
            colors=colors.to(target_device),
            intensities=intensities.to(target_device),
        ),
        info,
    )


def world_lights_to_camera_lights(world_lights: WorldPointLights, camera: PinholeCamera) -> PointLights:
    """Transform scene-stable world-space lights into the camera-space shader API."""
    if world_lights.positions_world.ndim != 2 or world_lights.positions_world.shape[-1] != 3:
        raise ValueError(f"Expected world light positions shape [N,3], got {tuple(world_lights.positions_world.shape)}")
    if world_lights.colors.shape != world_lights.positions_world.shape:
        raise ValueError(f"Expected world light colors shape {tuple(world_lights.positions_world.shape)}, got {tuple(world_lights.colors.shape)}")
    if world_lights.intensities.shape != (world_lights.positions_world.shape[0],):
        raise ValueError(
            f"Expected world light intensities shape [{world_lights.positions_world.shape[0]}], "
            f"got {tuple(world_lights.intensities.shape)}"
        )
    if camera.viewmats.shape != (1, 4, 4):
        raise ValueError(f"Expected camera.viewmats shape [1,4,4], got {tuple(camera.viewmats.shape)}")

    viewmat = camera.viewmats[0]
    dtype = viewmat.dtype
    device = viewmat.device
    positions_world = world_lights.positions_world.to(device=device, dtype=dtype)
    ones = torch.ones((positions_world.shape[0], 1), dtype=dtype, device=device)
    positions_h = torch.cat((positions_world, ones), dim=-1)
    positions_cam = torch.einsum("ij,nj->ni", viewmat, positions_h)[..., :3]
    return PointLights(
        positions_cam=positions_cam,
        colors=world_lights.colors.to(device=device, dtype=dtype),
        intensities=world_lights.intensities.to(device=device, dtype=dtype),
    )


def _robust_bbox(means: torch.Tensor, bbox_percentile: float) -> tuple[torch.Tensor, torch.Tensor]:
    if bbox_percentile >= 1.0:
        return means.min(dim=0).values, means.max(dim=0).values
    tail = (1.0 - float(bbox_percentile)) * 0.5
    quantiles = torch.tensor([tail, 1.0 - tail], dtype=means.dtype)
    bbox = torch.quantile(means, quantiles, dim=0)
    return bbox[0], bbox[1]


def _tuple3(values: torch.Tensor) -> tuple[float, float, float]:
    data = values.detach().cpu().tolist()
    return (float(data[0]), float(data[1]), float(data[2]))
