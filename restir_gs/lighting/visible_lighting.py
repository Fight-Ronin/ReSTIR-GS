from __future__ import annotations

import math

import torch

from restir_gs.lighting.deferred import LightingBuffers, PointLights, evaluate_selected_light_diffuse
from restir_gs.lighting.shadow_maps import ShadowMapBundle
from restir_gs.lighting.shadow_visibility import (
    ShadowVisibilityCache,
    evaluate_shadow_visibility,
    evaluate_shadow_visibility_selected_dense,
    evaluate_shadow_visibility_selected_dense_fast,
    gather_shadow_visibility,
)
from restir_gs.render.gbuffer import GBuffer
from restir_gs.render.synthetic_scene import PinholeCamera


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


def evaluate_selected_light_visible_diffuse_selected_dense(
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
    """Evaluate selected visible diffuse using selected-only dense visibility."""
    diffuse = evaluate_selected_light_diffuse(
        gbuffer,
        lights,
        light_indices,
        two_sided=two_sided,
        distance_epsilon=distance_epsilon,
    )
    visibility = evaluate_shadow_visibility_selected_dense(
        gbuffer,
        camera,
        shadow_bundle,
        light_indices,
        alpha_threshold=alpha_threshold,
        pcf_radius=pcf_radius,
    )
    return diffuse * visibility[..., None]


def evaluate_selected_light_visible_diffuse_selected_dense_fast(
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
    """Evaluate selected visible diffuse using candidate-flat selected visibility."""
    diffuse = evaluate_selected_light_diffuse(
        gbuffer,
        lights,
        light_indices,
        two_sided=two_sided,
        distance_epsilon=distance_epsilon,
    )
    visibility = evaluate_shadow_visibility_selected_dense_fast(
        gbuffer,
        camera,
        shadow_bundle,
        light_indices,
        alpha_threshold=alpha_threshold,
        pcf_radius=pcf_radius,
    )
    return diffuse * visibility[..., None]


def evaluate_selected_light_visible_diffuse_cached(
    gbuffer: GBuffer,
    lights: PointLights,
    visibility_cache: ShadowVisibilityCache,
    light_indices: torch.Tensor,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
) -> torch.Tensor:
    """Evaluate selected-light visible diffuse using a frame-local visibility cache."""
    diffuse = evaluate_selected_light_diffuse(
        gbuffer,
        lights,
        light_indices,
        two_sided=two_sided,
        distance_epsilon=distance_epsilon,
    )
    visibility = gather_shadow_visibility(visibility_cache, light_indices)
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


def shade_deferred_lambertian_visible_cached(
    gbuffer: GBuffer,
    lights: PointLights,
    visibility_cache: ShadowVisibilityCache,
    ambient: float = 0.2,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
    chunk_size: int = 64,
) -> LightingBuffers:
    """Evaluate all-lights Lambertian lighting using a frame-local visibility cache."""
    if chunk_size <= 0:
        raise ValueError(f"Expected positive chunk size, got {chunk_size}")
    light_count = lights.positions_cam.shape[0]
    expected = torch.arange(light_count, dtype=torch.long, device=visibility_cache.light_indices.device)
    if visibility_cache.light_indices.shape != (light_count,) or not torch.equal(
        visibility_cache.light_indices.to(dtype=torch.long),
        expected,
    ):
        raise ValueError("Cached all-lights visible Lambertian requires one cached visibility layer per light in index order.")
    device = gbuffer.rgb.device
    visible_flat = torch.zeros_like(gbuffer.rgb.reshape(-1, 3))
    for start in range(0, light_count, chunk_size):
        end = min(start + chunk_size, light_count)
        light_indices = torch.arange(start, end, dtype=torch.long, device=device)
        light_indices = light_indices.expand(gbuffer.rgb.shape[0], gbuffer.rgb.shape[1], end - start)
        visible_flat += evaluate_selected_light_visible_diffuse_cached(
            gbuffer,
            lights,
            visibility_cache,
            light_indices,
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

