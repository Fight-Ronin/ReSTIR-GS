from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from restir_gs.lighting.deferred import LightingBuffers, PointLights, evaluate_selected_light_diffuse, shade_deferred_lambertian
from restir_gs.render.camera_sequence import look_at_viewmat
from restir_gs.render.gbuffer import GBuffer
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


def evaluate_shadow_visibility(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    shadow_bundle: ShadowMapBundle,
    light_indices: torch.Tensor,
    alpha_threshold: float = 1e-4,
    pcf_radius: int = 0,
) -> torch.Tensor:
    """Evaluate shadow-map visibility for selected world-light indices shaped [H,W,K].

    ``pcf_radius=0`` uses the legacy single nearest shadow texel. Positive radii
    average hard shadow comparisons over a square PCF kernel and return soft
    visibility in ``[0, 1]``.
    """
    if alpha_threshold < 0.0:
        raise ValueError(f"Expected non-negative alpha_threshold, got {alpha_threshold}")
    if pcf_radius < 0:
        raise ValueError(f"Expected non-negative pcf_radius, got {pcf_radius}")
    if light_indices.ndim != 3:
        raise ValueError(f"Expected light_indices shape [H,W,K], got {tuple(light_indices.shape)}")
    if light_indices.shape[:2] != gbuffer.depth.shape:
        raise ValueError(f"Expected light index image shape {tuple(gbuffer.depth.shape)}, got {tuple(light_indices.shape[:2])}")
    if len(shadow_bundle.light_cameras) != int(shadow_bundle.light_indices.numel()):
        raise ValueError("Shadow bundle light camera count must match light_indices length.")

    height, width, candidate_count = light_indices.shape
    device = gbuffer.rgb.device
    dtype = gbuffer.rgb.dtype
    visibility = torch.zeros((height, width, candidate_count), dtype=dtype, device=device)
    valid = (gbuffer.valid_mask & gbuffer.normal_mask).to(device=device)
    world_positions = _gbuffer_world_positions(gbuffer, camera, dtype=dtype, device=device)

    bundle_lookup = {int(light_id.detach().cpu()): slot for slot, light_id in enumerate(shadow_bundle.light_indices)}
    indices = light_indices.to(device=device, dtype=torch.long)
    for light_id, slot in bundle_lookup.items():
        candidate_mask = indices == int(light_id)
        if not bool(candidate_mask.any()):
            continue
        light_visibility = _visibility_for_shadow_slot(
            world_positions,
            valid,
            shadow_bundle.light_cameras[slot],
            shadow_bundle.depth_maps[slot].to(device=device, dtype=dtype),
            shadow_bundle.alpha_maps[slot].to(device=device, dtype=dtype),
            depth_bias=shadow_bundle.depth_bias,
            alpha_threshold=alpha_threshold,
            pcf_radius=pcf_radius,
        )
        visibility = torch.where(candidate_mask, light_visibility[..., None], visibility)
    return visibility


def evaluate_selected_light_visible_diffuse(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    shadow_bundle: ShadowMapBundle,
    light_indices: torch.Tensor,
    alpha_threshold: float = 1e-4,
    pcf_radius: int = 0,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
) -> torch.Tensor:
    """Evaluate Lambertian selected-light contributions multiplied by shadow visibility."""
    diffuse = evaluate_selected_light_diffuse(
        gbuffer,
        lights,
        light_indices,
        two_sided=two_sided,
        distance_epsilon=distance_epsilon,
    )
    visibility = evaluate_shadow_visibility(
        gbuffer,
        camera,
        shadow_bundle,
        light_indices,
        alpha_threshold=alpha_threshold,
        pcf_radius=pcf_radius,
    )
    return diffuse * visibility[..., None]


def shade_deferred_lambertian_visible(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    shadow_bundle: ShadowMapBundle,
    ambient: float = 0.2,
    alpha_threshold: float = 1e-4,
    pcf_radius: int = 0,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
    chunk_size: int = 64,
) -> LightingBuffers:
    """Evaluate all-lights Lambertian lighting with shadow-map visibility."""
    if chunk_size <= 0:
        raise ValueError(f"Expected positive chunk size, got {chunk_size}")
    expected_light_ids = torch.arange(lights.positions_cam.shape[0], dtype=torch.long, device=shadow_bundle.light_indices.device)
    if shadow_bundle.light_indices.numel() != lights.positions_cam.shape[0] or not torch.equal(
        shadow_bundle.light_indices.to(dtype=torch.long),
        expected_light_ids,
    ):
        raise ValueError("All-lights visible Lambertian requires one shadow map for every light in index order.")
    device = gbuffer.rgb.device
    visible_flat = torch.zeros_like(gbuffer.rgb.reshape(-1, 3))
    for start in range(0, lights.positions_cam.shape[0], chunk_size):
        end = min(start + chunk_size, lights.positions_cam.shape[0])
        light_indices = torch.arange(start, end, dtype=torch.long, device=device)
        light_indices = light_indices.expand(gbuffer.rgb.shape[0], gbuffer.rgb.shape[1], end - start)
        visible_flat += evaluate_selected_light_visible_diffuse(
            gbuffer,
            camera,
            lights,
            shadow_bundle,
            light_indices,
            alpha_threshold=alpha_threshold,
            pcf_radius=pcf_radius,
            two_sided=two_sided,
            distance_epsilon=distance_epsilon,
        ).sum(dim=2).reshape(-1, 3)

    valid_mask = gbuffer.valid_mask & gbuffer.normal_mask
    diffuse_rgb = torch.where(valid_mask[..., None], visible_flat.reshape_as(gbuffer.rgb), torch.zeros_like(gbuffer.rgb))
    dynamic_shade = torch.where(
        gbuffer.rgb.abs() > 1e-8,
        diffuse_rgb / torch.clamp(gbuffer.rgb, min=1e-8),
        torch.zeros_like(gbuffer.rgb),
    )
    composite_lit = gbuffer.rgb * float(ambient) + diffuse_rgb
    composite_rgb = torch.where(valid_mask[..., None], composite_lit, gbuffer.rgb)
    shade_rgb = torch.where(
        valid_mask[..., None],
        torch.full_like(gbuffer.rgb, float(ambient)) + dynamic_shade,
        torch.zeros_like(gbuffer.rgb),
    )
    return LightingBuffers(
        irradiance_rgb=dynamic_shade * math.pi,
        diffuse_rgb=diffuse_rgb,
        specular_rgb=torch.zeros_like(diffuse_rgb),
        shade_rgb=shade_rgb,
        composite_rgb=composite_rgb,
        valid_mask=valid_mask,
    )


def shade_deferred_lambertian_unshadowed(
    gbuffer: GBuffer,
    lights: PointLights,
    ambient: float = 0.2,
) -> LightingBuffers:
    """Compatibility wrapper used by tests and smoke outputs."""
    return shade_deferred_lambertian(gbuffer, lights, ambient=ambient)


def _gbuffer_world_positions(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    positions = gbuffer.position_cam.to(device=device, dtype=dtype)
    ones = torch.ones((*positions.shape[:2], 1), dtype=dtype, device=device)
    position_h = torch.cat((positions, ones), dim=-1)
    inv_view = torch.linalg.inv(camera.viewmats[0].to(device=device, dtype=dtype))
    world_h = torch.einsum("ij,hwj->hwi", inv_view, position_h)
    return world_h[..., :3]


def _visibility_for_shadow_slot(
    world_positions: torch.Tensor,
    valid_mask: torch.Tensor,
    light_camera: PinholeCamera,
    shadow_depth: torch.Tensor,
    shadow_alpha: torch.Tensor,
    depth_bias: float,
    alpha_threshold: float,
    pcf_radius: int,
) -> torch.Tensor:
    device = world_positions.device
    dtype = world_positions.dtype
    height, width = shadow_depth.shape
    ones = torch.ones((*world_positions.shape[:2], 1), dtype=dtype, device=device)
    world_h = torch.cat((world_positions, ones), dim=-1)
    light_h = torch.einsum("ij,hwj->hwi", light_camera.viewmats[0].to(device=device, dtype=dtype), world_h)
    light_z = light_h[..., 2]
    intrinsics = light_camera.intrinsics[0].to(device=device, dtype=dtype)
    u = light_h[..., 0] * intrinsics[0, 0] / light_z.clamp_min(torch.finfo(dtype).tiny) + intrinsics[0, 2]
    v = light_h[..., 1] * intrinsics[1, 1] / light_z.clamp_min(torch.finfo(dtype).tiny) + intrinsics[1, 2]
    x = torch.round(u).to(torch.long)
    y = torch.round(v).to(torch.long)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if pcf_radius == 0:
        return _hard_shadow_compare(
            x,
            y,
            light_z,
            valid_mask,
            in_bounds,
            shadow_depth,
            shadow_alpha,
            depth_bias,
            alpha_threshold,
        ).to(dtype=dtype)

    visibility_sum = torch.zeros_like(light_z, dtype=dtype)
    sample_count = 0
    for dy in range(-pcf_radius, pcf_radius + 1):
        for dx in range(-pcf_radius, pcf_radius + 1):
            sample_x = x + dx
            sample_y = y + dy
            sample_in_bounds = (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
            visibility_sum = visibility_sum + _hard_shadow_compare(
                sample_x,
                sample_y,
                light_z,
                valid_mask,
                sample_in_bounds,
                shadow_depth,
                shadow_alpha,
                depth_bias,
                alpha_threshold,
            ).to(dtype=dtype)
            sample_count += 1
    return visibility_sum / float(sample_count)


def _hard_shadow_compare(
    x: torch.Tensor,
    y: torch.Tensor,
    light_z: torch.Tensor,
    valid_mask: torch.Tensor,
    in_bounds: torch.Tensor,
    shadow_depth: torch.Tensor,
    shadow_alpha: torch.Tensor,
    depth_bias: float,
    alpha_threshold: float,
) -> torch.Tensor:
    height, width = shadow_depth.shape
    safe_x = x.clamp(0, max(width - 1, 0))
    safe_y = y.clamp(0, max(height - 1, 0))
    flat = safe_y * width + safe_x
    sampled_depth = shadow_depth.reshape(-1)[flat.reshape(-1)].reshape(light_z.shape)
    sampled_alpha = shadow_alpha.reshape(-1)[flat.reshape(-1)].reshape(light_z.shape)
    no_blocker = sampled_alpha <= float(alpha_threshold)
    depth_pass = light_z <= sampled_depth + float(depth_bias)
    return valid_mask & in_bounds & (light_z > 0.0) & (no_blocker | depth_pass)
