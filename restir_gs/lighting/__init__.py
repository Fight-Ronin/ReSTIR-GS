"""Deferred lighting helpers for the ReSTIR-GS prototype."""

from restir_gs.lighting.asset_lights import (
    WorldPointLights,
    make_asset_scaled_point_lights,
    make_asset_scaled_world_lights,
    world_lights_to_camera_lights,
)
from restir_gs.lighting.deferred import (
    LightingBuffers,
    PointLights,
    evaluate_selected_light_blinn_phong,
    evaluate_selected_light_diffuse,
    make_deterministic_point_lights,
    shade_deferred_blinn_phong,
    shade_deferred_lambertian,
)
from restir_gs.lighting.shadow_maps import (
    ShadowMapBundle,
    make_light_camera,
    make_shadow_map_bundle,
)
from restir_gs.lighting.shadow_visibility import (
    ShadowVisibilityCache,
    evaluate_shadow_visibility,
    evaluate_shadow_visibility_selected_dense,
    evaluate_shadow_visibility_selected_dense_fast,
    gather_shadow_visibility,
    make_shadow_visibility_cache,
)
from restir_gs.lighting.visible_lighting import (
    evaluate_selected_light_visible_diffuse,
    evaluate_selected_light_visible_diffuse_cached,
    evaluate_selected_light_visible_diffuse_selected_dense,
    evaluate_selected_light_visible_diffuse_selected_dense_fast,
    shade_deferred_lambertian_visible,
    shade_deferred_lambertian_visible_cached,
)

__all__ = [
    "LightingBuffers",
    "PointLights",
    "ShadowMapBundle",
    "ShadowVisibilityCache",
    "WorldPointLights",
    "evaluate_selected_light_blinn_phong",
    "evaluate_selected_light_diffuse",
    "evaluate_selected_light_visible_diffuse",
    "evaluate_selected_light_visible_diffuse_cached",
    "evaluate_selected_light_visible_diffuse_selected_dense",
    "evaluate_selected_light_visible_diffuse_selected_dense_fast",
    "evaluate_shadow_visibility",
    "evaluate_shadow_visibility_selected_dense",
    "evaluate_shadow_visibility_selected_dense_fast",
    "gather_shadow_visibility",
    "make_asset_scaled_point_lights",
    "make_asset_scaled_world_lights",
    "make_deterministic_point_lights",
    "make_light_camera",
    "make_shadow_map_bundle",
    "make_shadow_visibility_cache",
    "shade_deferred_blinn_phong",
    "shade_deferred_lambertian",
    "shade_deferred_lambertian_visible",
    "shade_deferred_lambertian_visible_cached",
    "world_lights_to_camera_lights",
]
