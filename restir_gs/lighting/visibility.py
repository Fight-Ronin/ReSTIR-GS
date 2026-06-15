from __future__ import annotations

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
    "ShadowMapBundle",
    "ShadowVisibilityCache",
    "evaluate_selected_light_visible_diffuse",
    "evaluate_selected_light_visible_diffuse_cached",
    "evaluate_selected_light_visible_diffuse_selected_dense",
    "evaluate_selected_light_visible_diffuse_selected_dense_fast",
    "evaluate_shadow_visibility",
    "evaluate_shadow_visibility_selected_dense",
    "evaluate_shadow_visibility_selected_dense_fast",
    "gather_shadow_visibility",
    "make_light_camera",
    "make_shadow_map_bundle",
    "make_shadow_visibility_cache",
    "shade_deferred_lambertian_visible",
    "shade_deferred_lambertian_visible_cached",
]
