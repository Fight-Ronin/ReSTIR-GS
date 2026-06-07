from __future__ import annotations

import torch
import pytest

from restir_gs.lighting.deferred import PointLights, shade_deferred_lambertian
from restir_gs.lighting.visibility import (
    ShadowMapBundle,
    evaluate_selected_light_visible_diffuse,
    evaluate_shadow_visibility,
    make_light_camera,
    shade_deferred_lambertian_visible,
)
from restir_gs.render.gbuffer import GBuffer
from restir_gs.render.synthetic_scene import PinholeCamera


def test_light_camera_maps_target_to_positive_center_depth() -> None:
    light = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32)
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    camera = make_light_camera(light, target, resolution=8, focal_scale=1.0)
    target_h = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    light_h = camera.viewmats[0] @ target_h
    u = light_h[0] * camera.intrinsics[0, 0, 0] / light_h[2] + camera.intrinsics[0, 0, 2]
    v = light_h[1] * camera.intrinsics[0, 1, 1] / light_h[2] + camera.intrinsics[0, 1, 2]

    assert light_h[2] > 0.0
    assert torch.allclose(u, torch.tensor(4.0))
    assert torch.allclose(v, torch.tensor(4.0))


def test_visibility_passes_when_depth_is_within_bias() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 1.0))
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=2.0, alpha=1.0, depth_bias=0.1)

    visibility = evaluate_shadow_visibility(gbuffer, camera, shadow, torch.zeros((1, 1, 1), dtype=torch.long))

    assert torch.allclose(visibility, torch.ones((1, 1, 1)))


def test_visibility_blocks_when_behind_opaque_shadow_depth() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=1.0, alpha=1.0, depth_bias=0.01)

    visibility = evaluate_shadow_visibility(gbuffer, camera, shadow, torch.zeros((1, 1, 1), dtype=torch.long))

    assert torch.allclose(visibility, torch.zeros((1, 1, 1)))


def test_pcf_radius_zero_matches_hard_shadow_behavior() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=1.0, alpha=1.0, depth_bias=0.01)

    implicit_hard = evaluate_shadow_visibility(gbuffer, camera, shadow, torch.zeros((1, 1, 1), dtype=torch.long))
    explicit_hard = evaluate_shadow_visibility(
        gbuffer,
        camera,
        shadow,
        torch.zeros((1, 1, 1), dtype=torch.long),
        pcf_radius=0,
    )

    assert torch.allclose(explicit_hard, implicit_hard)


def test_pcf_radius_one_averages_hard_shadow_samples() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=1.0, alpha=1.0, depth_bias=0.0)
    shadow.depth_maps[0, 0, 0] = 3.0
    shadow.depth_maps[0, 0, 1] = 3.0
    shadow.depth_maps[0, 1, 0] = 3.0

    visibility = evaluate_shadow_visibility(
        gbuffer,
        camera,
        shadow,
        torch.zeros((1, 1, 1), dtype=torch.long),
        pcf_radius=1,
    )

    assert torch.allclose(visibility, torch.full((1, 1, 1), 3.0 / 9.0))


def test_low_shadow_alpha_means_no_blocker() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=1.0, alpha=0.0, depth_bias=0.0)

    visibility = evaluate_shadow_visibility(gbuffer, camera, shadow, torch.zeros((1, 1, 1), dtype=torch.long))

    assert torch.allclose(visibility, torch.ones((1, 1, 1)))


def test_out_of_bounds_or_negative_light_z_is_invisible() -> None:
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=10.0, alpha=0.0, depth_bias=0.0)
    out_of_bounds = make_single_pixel_gbuffer(position=(10.0, 0.0, 1.0))
    behind_light = make_single_pixel_gbuffer(position=(0.0, 0.0, -3.0))

    out_vis = evaluate_shadow_visibility(out_of_bounds, camera, shadow, torch.zeros((1, 1, 1), dtype=torch.long))
    behind_vis = evaluate_shadow_visibility(behind_light, camera, shadow, torch.zeros((1, 1, 1), dtype=torch.long))

    assert torch.allclose(out_vis, torch.zeros((1, 1, 1)))
    assert torch.allclose(behind_vis, torch.zeros((1, 1, 1)))


def test_invalid_pixels_have_zero_visible_contribution() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 1.0), valid=False)
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=2.0, alpha=0.0, depth_bias=0.0)
    lights = make_lights()

    visible = evaluate_selected_light_visible_diffuse(gbuffer, camera, lights, shadow, torch.zeros((1, 1, 1), dtype=torch.long))

    assert torch.allclose(visible, torch.zeros((1, 1, 1, 3)))


def test_all_visible_lambertian_matches_unshadowed_reference() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 1.0))
    camera = make_identity_camera()
    lights = make_lights()
    shadow = make_shadow_bundle(depth=2.0, alpha=0.0, depth_bias=0.0)

    visible = shade_deferred_lambertian_visible(gbuffer, camera, lights, shadow, ambient=0.2, distance_epsilon=0.0)
    unshadowed = shade_deferred_lambertian(gbuffer, lights, ambient=0.2, distance_epsilon=0.0)

    assert torch.allclose(visible.diffuse_rgb, unshadowed.diffuse_rgb)
    assert torch.allclose(visible.composite_rgb, unshadowed.composite_rgb)


def test_negative_pcf_radius_fails_loudly() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 1.0))
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=2.0, alpha=1.0, depth_bias=0.1)

    with pytest.raises(ValueError, match="pcf_radius"):
        evaluate_shadow_visibility(gbuffer, camera, shadow, torch.zeros((1, 1, 1), dtype=torch.long), pcf_radius=-1)


def make_single_pixel_gbuffer(position: tuple[float, float, float], valid: bool = True) -> GBuffer:
    valid_mask = torch.tensor([[valid]], dtype=torch.bool)
    return GBuffer(
        rgb=torch.ones((1, 1, 3), dtype=torch.float32),
        depth=torch.ones((1, 1), dtype=torch.float32),
        alpha=torch.ones((1, 1), dtype=torch.float32),
        position_cam=torch.tensor([[position]], dtype=torch.float32),
        normal_cam=torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32),
        valid_mask=valid_mask,
        normal_mask=valid_mask.clone(),
    )


def make_identity_camera() -> PinholeCamera:
    return PinholeCamera(
        viewmats=torch.eye(4, dtype=torch.float32)[None],
        intrinsics=torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
        width=3,
        height=3,
    )


def make_shadow_bundle(depth: float, alpha: float, depth_bias: float) -> ShadowMapBundle:
    light_camera = PinholeCamera(
        viewmats=torch.eye(4, dtype=torch.float32)[None],
        intrinsics=torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
        width=3,
        height=3,
    )
    return ShadowMapBundle(
        light_indices=torch.tensor([0], dtype=torch.long),
        light_cameras=[light_camera],
        depth_maps=torch.full((1, 3, 3), depth, dtype=torch.float32),
        alpha_maps=torch.full((1, 3, 3), alpha, dtype=torch.float32),
        scene_radius=1.0,
        depth_bias=depth_bias,
    )


def make_lights() -> PointLights:
    return PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32),
        colors=torch.ones((1, 3), dtype=torch.float32),
        intensities=torch.ones((1,), dtype=torch.float32),
    )
