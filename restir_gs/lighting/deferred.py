from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from restir_gs.render.gbuffer import GBuffer


@dataclass(frozen=True)
class PointLights:
    positions_cam: torch.Tensor
    colors: torch.Tensor
    intensities: torch.Tensor


@dataclass(frozen=True)
class LightingBuffers:
    irradiance_rgb: torch.Tensor
    diffuse_rgb: torch.Tensor
    shade_rgb: torch.Tensor
    composite_rgb: torch.Tensor
    valid_mask: torch.Tensor


def make_deterministic_point_lights(
    count: int = 128,
    seed: int = 2027,
    device: torch.device | str = "cuda",
) -> PointLights:
    """Create a reproducible camera-space point-light set."""
    if count <= 0:
        raise ValueError(f"Expected positive light count, got {count}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    xy = torch.rand((count, 2), generator=generator, dtype=torch.float32) * 2.4 - 1.2
    z = torch.rand((count, 1), generator=generator, dtype=torch.float32) * 3.0 + 0.8
    colors = torch.rand((count, 3), generator=generator, dtype=torch.float32) * 0.6 + 0.4
    intensities = torch.full((count,), 3.0 / float(count), dtype=torch.float32)

    return PointLights(
        positions_cam=torch.cat((xy, z), dim=-1).to(device),
        colors=colors.to(device),
        intensities=intensities.to(device),
    )


def _check_lights(lights: PointLights) -> None:
    if lights.positions_cam.ndim != 2 or lights.positions_cam.shape[-1] != 3:
        raise ValueError(f"Expected light positions shape [N,3], got {tuple(lights.positions_cam.shape)}")
    if lights.colors.shape != lights.positions_cam.shape:
        raise ValueError(f"Expected light colors shape {tuple(lights.positions_cam.shape)}, got {tuple(lights.colors.shape)}")
    if lights.intensities.shape != (lights.positions_cam.shape[0],):
        raise ValueError(
            f"Expected light intensities shape [{lights.positions_cam.shape[0]}], got {tuple(lights.intensities.shape)}"
        )


def _evaluate_selected_light_irradiance(
    gbuffer: GBuffer,
    lights: PointLights,
    light_indices: torch.Tensor,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
) -> torch.Tensor:
    if light_indices.ndim != 3:
        raise ValueError(f"Expected light indices shape [H,W,K], got {tuple(light_indices.shape)}")
    if light_indices.shape[:2] != gbuffer.rgb.shape[:2]:
        raise ValueError(f"Expected light indices image shape {tuple(gbuffer.rgb.shape[:2])}, got {tuple(light_indices.shape[:2])}")
    _check_lights(lights)
    if light_indices.numel() > 0:
        if int(light_indices.min().detach().cpu()) < 0 or int(light_indices.max().detach().cpu()) >= lights.positions_cam.shape[0]:
            raise ValueError("Light indices must be in [0, light_count).")

    dtype = gbuffer.rgb.dtype
    device = gbuffer.rgb.device
    indices = light_indices.to(device=device, dtype=torch.long)
    positions_cam = lights.positions_cam.to(device=device, dtype=dtype)
    colors = lights.colors.to(device=device, dtype=dtype)
    intensities = lights.intensities.to(device=device, dtype=dtype)

    selected_positions = positions_cam[indices]
    selected_colors = colors[indices]
    selected_intensities = intensities[indices]

    light_vec = selected_positions - gbuffer.position_cam[..., None, :]
    dist2_raw = torch.sum(light_vec * light_vec, dim=-1)
    dist2 = dist2_raw + distance_epsilon
    wi = light_vec * torch.rsqrt(dist2_raw.clamp_min(distance_epsilon)[..., None])
    cos_theta = torch.sum(gbuffer.normal_cam[..., None, :] * wi, dim=-1)
    if two_sided:
        cos_theta = cos_theta.abs()
    else:
        cos_theta = cos_theta.clamp_min(0.0)

    irradiance = selected_colors * selected_intensities[..., None] * cos_theta[..., None] / dist2[..., None]
    valid = (gbuffer.valid_mask & gbuffer.normal_mask)[..., None, None]
    return torch.where(valid, irradiance, torch.zeros_like(irradiance))


def evaluate_selected_light_diffuse(
    gbuffer: GBuffer,
    lights: PointLights,
    light_indices: torch.Tensor,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
) -> torch.Tensor:
    """Evaluate selected point-light diffuse RGB contributions."""
    irradiance = _evaluate_selected_light_irradiance(
        gbuffer,
        lights,
        light_indices,
        two_sided=two_sided,
        distance_epsilon=distance_epsilon,
    )
    return gbuffer.rgb[..., None, :] * irradiance / math.pi


def shade_deferred_lambertian(
    gbuffer: GBuffer,
    lights: PointLights,
    ambient: float = 0.2,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
    chunk_size: int = 64,
) -> LightingBuffers:
    """Evaluate all point lights against the pseudo G-buffer."""
    if chunk_size <= 0:
        raise ValueError(f"Expected positive chunk size, got {chunk_size}")
    _check_lights(lights)

    device = gbuffer.rgb.device
    valid_flat = (gbuffer.valid_mask & gbuffer.normal_mask).reshape(-1).to(torch.bool)

    irradiance_flat = torch.zeros_like(gbuffer.rgb.reshape(-1, 3))
    for start in range(0, lights.positions_cam.shape[0], chunk_size):
        end = min(start + chunk_size, lights.positions_cam.shape[0])
        light_indices = torch.arange(start, end, dtype=torch.long, device=device)
        light_indices = light_indices.expand(gbuffer.rgb.shape[0], gbuffer.rgb.shape[1], end - start)
        irradiance = _evaluate_selected_light_irradiance(
            gbuffer,
            lights,
            light_indices,
            two_sided=two_sided,
            distance_epsilon=distance_epsilon,
        )
        irradiance_flat += irradiance.sum(dim=2).reshape(-1, 3)

    irradiance_flat = torch.where(valid_flat[:, None], irradiance_flat, torch.zeros_like(irradiance_flat))

    irradiance_rgb = irradiance_flat.reshape_as(gbuffer.rgb)
    valid_mask = valid_flat.reshape(gbuffer.valid_mask.shape)
    dynamic_shade = irradiance_rgb / math.pi
    diffuse_rgb = torch.where(valid_mask[..., None], gbuffer.rgb * dynamic_shade, torch.zeros_like(gbuffer.rgb))
    shade_rgb = torch.where(
        valid_mask[..., None],
        torch.full_like(gbuffer.rgb, float(ambient)) + dynamic_shade,
        torch.zeros_like(gbuffer.rgb),
    )
    composite_lit = gbuffer.rgb * shade_rgb
    composite_rgb = torch.where(valid_mask[..., None], composite_lit, gbuffer.rgb)

    return LightingBuffers(
        irradiance_rgb=irradiance_rgb,
        diffuse_rgb=diffuse_rgb,
        shade_rgb=shade_rgb,
        composite_rgb=composite_rgb,
        valid_mask=valid_mask,
    )
