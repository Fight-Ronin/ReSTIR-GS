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

__all__ = [
    "LightingBuffers",
    "PointLights",
    "WorldPointLights",
    "evaluate_selected_light_blinn_phong",
    "evaluate_selected_light_diffuse",
    "make_asset_scaled_point_lights",
    "make_asset_scaled_world_lights",
    "make_deterministic_point_lights",
    "shade_deferred_blinn_phong",
    "shade_deferred_lambertian",
    "world_lights_to_camera_lights",
]
