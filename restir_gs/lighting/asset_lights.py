from __future__ import annotations

import torch

from restir_gs.lighting.deferred import PointLights
from restir_gs.render.gbuffer import GBuffer


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


def _tuple3(values: torch.Tensor) -> tuple[float, float, float]:
    data = values.detach().cpu().tolist()
    return (float(data[0]), float(data[1]), float(data[2]))
