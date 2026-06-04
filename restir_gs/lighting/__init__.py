"""Deferred lighting helpers for the ReSTIR-GS prototype."""

from restir_gs.lighting.asset_lights import make_asset_scaled_point_lights
from restir_gs.lighting.deferred import (
    LightingBuffers,
    PointLights,
    evaluate_selected_light_diffuse,
    make_deterministic_point_lights,
    shade_deferred_lambertian,
)

__all__ = [
    "LightingBuffers",
    "PointLights",
    "evaluate_selected_light_diffuse",
    "make_asset_scaled_point_lights",
    "make_deterministic_point_lights",
    "shade_deferred_lambertian",
]
