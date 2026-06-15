from __future__ import annotations

import torch
import pytest

from restir_gs.lighting.deferred import PointLights, shade_deferred_lambertian
from restir_gs.lighting.visibility import (
    ShadowMapBundle,
    evaluate_selected_light_visible_diffuse_selected_dense_fast,
    evaluate_selected_light_visible_diffuse_selected_dense,
    evaluate_selected_light_visible_diffuse,
    evaluate_shadow_visibility_selected_dense_fast,
    evaluate_shadow_visibility_selected_dense,
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


def test_dense_visibility_gathers_per_light_shadow_maps() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_two_light_shadow_bundle(depths=[3.0, 1.0], alphas=[1.0, 1.0], depth_bias=0.0)

    visibility = evaluate_shadow_visibility(
        gbuffer,
        camera,
        shadow,
        torch.tensor([[[0, 1, 0]]], dtype=torch.long),
    )

    assert torch.allclose(visibility, torch.tensor([[[1.0, 0.0, 1.0]]]))


def test_selected_dense_visibility_matches_existing_dense_path() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_two_light_shadow_bundle(depths=[3.0, 1.0], alphas=[0.0, 0.5], depth_bias=0.0)
    light_indices = torch.tensor([[[0, 1, 0]]], dtype=torch.long)

    existing = evaluate_shadow_visibility(
        gbuffer,
        camera,
        shadow,
        light_indices,
        alpha_threshold=0.0,
        pcf_radius=1,
    )
    selected = evaluate_shadow_visibility_selected_dense(
        gbuffer,
        camera,
        shadow,
        light_indices,
        alpha_threshold=0.0,
        pcf_radius=1,
    )

    assert torch.allclose(selected, existing)


def test_selected_dense_fast_visibility_matches_reference_cpu_fallback() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_two_light_shadow_bundle(depths=[3.0, 1.0], alphas=[0.0, 0.5], depth_bias=0.0)
    light_indices = torch.tensor([[[0, 1, 0]]], dtype=torch.long)

    reference = evaluate_shadow_visibility_selected_dense(
        gbuffer,
        camera,
        shadow,
        light_indices,
        alpha_threshold=0.0,
        pcf_radius=1,
    )
    fast = evaluate_shadow_visibility_selected_dense_fast(
        gbuffer,
        camera,
        shadow,
        light_indices,
        alpha_threshold=0.0,
        pcf_radius=1,
    )

    assert torch.allclose(fast, reference)


def test_selected_dense_fast_visibility_matches_reference_cuda() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    gbuffer = move_gbuffer(make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0)), device)
    camera = move_camera(make_identity_camera(), device)
    shadow = move_shadow_bundle(make_two_light_shadow_bundle(depths=[3.0, 1.0], alphas=[0.0, 0.5], depth_bias=0.0), device)
    light_indices = torch.tensor([[[0, 1, 0, -1, 2, 1, 0, 1]]], dtype=torch.long, device=device)

    reference = evaluate_shadow_visibility_selected_dense(
        gbuffer,
        camera,
        shadow,
        light_indices,
        alpha_threshold=0.0,
        pcf_radius=1,
    )
    fast = evaluate_shadow_visibility_selected_dense_fast(
        gbuffer,
        camera,
        shadow,
        light_indices,
        alpha_threshold=0.0,
        pcf_radius=1,
    )

    assert torch.allclose(fast, reference)


def test_selected_dense_fast_visibility_matches_reference_cuda_distinct_light_cameras() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    gbuffer = make_structured_gbuffer(device)
    camera = make_identity_camera()
    shadow = make_distinct_camera_shadow_bundle(device)
    light_indices = torch.tensor(
        [
            [[0, 1, 2, -1], [1, 2, 0, 3]],
            [[2, 0, 1, 0], [0, 2, 1, 2]],
        ],
        dtype=torch.long,
        device=device,
    )
    camera = move_camera(camera, device)

    for pcf_radius in (0, 1):
        reference = evaluate_shadow_visibility_selected_dense(
            gbuffer,
            camera,
            shadow,
            light_indices,
            alpha_threshold=0.0,
            pcf_radius=pcf_radius,
        )
        fast = evaluate_shadow_visibility_selected_dense_fast(
            gbuffer,
            camera,
            shadow,
            light_indices,
            alpha_threshold=0.0,
            pcf_radius=pcf_radius,
        )

        assert torch.allclose(fast, reference)


def test_selected_dense_visibility_matches_dense_invalid_indices() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_two_light_shadow_bundle(depths=[3.0, 1.0], alphas=[0.0, 0.5], depth_bias=0.0)
    light_indices = torch.tensor([[[-1, 2, 1]]], dtype=torch.long)

    existing = evaluate_shadow_visibility(gbuffer, camera, shadow, light_indices, alpha_threshold=0.0)
    selected = evaluate_shadow_visibility_selected_dense(gbuffer, camera, shadow, light_indices, alpha_threshold=0.0)

    assert torch.allclose(selected, existing)
    assert torch.allclose(selected, torch.tensor([[[0.0, 0.0, 0.5]]]))


def test_selected_dense_fast_visible_diffuse_matches_reference_cpu_fallback() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    lights = make_two_lights()
    shadow = make_two_light_shadow_bundle(depths=[3.0, 1.0], alphas=[0.0, 0.5], depth_bias=0.0)
    light_indices = torch.tensor([[[0, 1, 0]]], dtype=torch.long)

    reference = evaluate_selected_light_visible_diffuse_selected_dense(
        gbuffer,
        camera,
        lights,
        shadow,
        light_indices,
        alpha_threshold=0.0,
        pcf_radius=1,
    )
    fast = evaluate_selected_light_visible_diffuse_selected_dense_fast(
        gbuffer,
        camera,
        lights,
        shadow,
        light_indices,
        alpha_threshold=0.0,
        pcf_radius=1,
    )

    assert torch.allclose(fast, reference)


def test_selected_dense_visible_diffuse_matches_existing_dense_path() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    lights = make_two_lights()
    shadow = make_two_light_shadow_bundle(depths=[3.0, 1.0], alphas=[0.0, 0.5], depth_bias=0.0)
    light_indices = torch.tensor([[[0, 1, 0]]], dtype=torch.long)

    existing = evaluate_selected_light_visible_diffuse(
        gbuffer,
        camera,
        lights,
        shadow,
        light_indices,
        alpha_threshold=0.0,
        pcf_radius=1,
    )
    selected = evaluate_selected_light_visible_diffuse_selected_dense(
        gbuffer,
        camera,
        lights,
        shadow,
        light_indices,
        alpha_threshold=0.0,
        pcf_radius=1,
    )

    assert torch.allclose(selected, existing)


def test_low_shadow_alpha_means_no_blocker() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=1.0, alpha=0.0, depth_bias=0.0)

    visibility = evaluate_shadow_visibility(gbuffer, camera, shadow, torch.zeros((1, 1, 1), dtype=torch.long))

    assert torch.allclose(visibility, torch.ones((1, 1, 1)))


def test_partial_shadow_alpha_returns_partial_visibility() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=1.0, alpha=0.25, depth_bias=0.0)

    visibility = evaluate_shadow_visibility(
        gbuffer,
        camera,
        shadow,
        torch.zeros((1, 1, 1), dtype=torch.long),
        alpha_threshold=0.0,
    )

    assert torch.allclose(visibility, torch.full((1, 1, 1), 0.75))


def test_partial_shadow_alpha_respects_alpha_threshold() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=1.0, alpha=0.6, depth_bias=0.0)

    visibility = evaluate_shadow_visibility(
        gbuffer,
        camera,
        shadow,
        torch.zeros((1, 1, 1), dtype=torch.long),
        alpha_threshold=0.2,
    )

    assert torch.allclose(visibility, torch.full((1, 1, 1), 0.5))


def test_pcf_radius_one_averages_partial_shadow_alpha_samples() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    shadow = make_shadow_bundle(depth=1.0, alpha=0.0, depth_bias=0.0)
    shadow.alpha_maps[0, 1, 1] = 0.5

    visibility = evaluate_shadow_visibility(
        gbuffer,
        camera,
        shadow,
        torch.zeros((1, 1, 1), dtype=torch.long),
        alpha_threshold=0.0,
        pcf_radius=1,
    )

    assert torch.allclose(visibility, torch.full((1, 1, 1), 8.5 / 9.0))


def test_non_dense_shadow_indices_match_dense_visibility_path() -> None:
    gbuffer = make_single_pixel_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    dense_shadow = make_shadow_bundle(depth=1.0, alpha=0.25, depth_bias=0.0)
    sparse_shadow = ShadowMapBundle(
        light_indices=torch.tensor([7], dtype=torch.long),
        light_cameras=dense_shadow.light_cameras,
        depth_maps=dense_shadow.depth_maps,
        alpha_maps=dense_shadow.alpha_maps,
        scene_radius=dense_shadow.scene_radius,
        depth_bias=dense_shadow.depth_bias,
    )

    dense_visibility = evaluate_shadow_visibility(
        gbuffer,
        camera,
        dense_shadow,
        torch.zeros((1, 1, 1), dtype=torch.long),
        alpha_threshold=0.0,
        pcf_radius=1,
    )
    sparse_visibility = evaluate_shadow_visibility(
        gbuffer,
        camera,
        sparse_shadow,
        torch.full((1, 1, 1), 7, dtype=torch.long),
        alpha_threshold=0.0,
        pcf_radius=1,
    )

    assert torch.allclose(sparse_visibility, dense_visibility)


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


def make_two_light_shadow_bundle(depths: list[float], alphas: list[float], depth_bias: float) -> ShadowMapBundle:
    light_camera = PinholeCamera(
        viewmats=torch.eye(4, dtype=torch.float32)[None],
        intrinsics=torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
        width=3,
        height=3,
    )
    return ShadowMapBundle(
        light_indices=torch.tensor([0, 1], dtype=torch.long),
        light_cameras=[light_camera, light_camera],
        depth_maps=torch.stack([torch.full((3, 3), depth, dtype=torch.float32) for depth in depths], dim=0),
        alpha_maps=torch.stack([torch.full((3, 3), alpha, dtype=torch.float32) for alpha in alphas], dim=0),
        scene_radius=1.0,
        depth_bias=depth_bias,
    )


def make_distinct_camera_shadow_bundle(device: torch.device) -> ShadowMapBundle:
    viewmats = torch.eye(4, dtype=torch.float32).repeat(3, 1, 1)
    viewmats[1, 0, 3] = 0.5
    viewmats[2, 1, 3] = -0.5
    intrinsics = torch.tensor(
        [
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.5, 0.0, 1.0], [0.0, 0.5, 2.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    light_cameras = [
        PinholeCamera(
            viewmats=viewmats[index : index + 1].to(device),
            intrinsics=intrinsics[index : index + 1].to(device),
            width=3,
            height=3,
        )
        for index in range(3)
    ]
    depth_maps = torch.tensor(
        [
            [[3.0, 3.0, 1.0], [3.0, 3.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 3.0], [1.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
            [[3.0, 1.0, 3.0], [1.0, 3.0, 1.0], [3.0, 1.0, 3.0]],
        ],
        dtype=torch.float32,
        device=device,
    )
    alpha_maps = torch.tensor(
        [
            [[0.0, 0.5, 1.0], [0.25, 0.75, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 0.25, 0.0], [0.5, 0.0, 0.25], [0.0, 0.5, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.5, 1.0], [0.25, 0.75, 0.0]],
        ],
        dtype=torch.float32,
        device=device,
    )
    return ShadowMapBundle(
        light_indices=torch.arange(3, dtype=torch.long, device=device),
        light_cameras=light_cameras,
        depth_maps=depth_maps,
        alpha_maps=alpha_maps,
        scene_radius=1.0,
        depth_bias=0.0,
    )


def make_lights() -> PointLights:
    return PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32),
        colors=torch.ones((1, 3), dtype=torch.float32),
        intensities=torch.ones((1,), dtype=torch.float32),
    )


def make_two_lights() -> PointLights:
    return PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
        colors=torch.ones((2, 3), dtype=torch.float32),
        intensities=torch.ones((2,), dtype=torch.float32),
    )


def move_gbuffer(gbuffer: GBuffer, device: torch.device) -> GBuffer:
    return GBuffer(
        rgb=gbuffer.rgb.to(device),
        depth=gbuffer.depth.to(device),
        alpha=gbuffer.alpha.to(device),
        position_cam=gbuffer.position_cam.to(device),
        normal_cam=gbuffer.normal_cam.to(device),
        valid_mask=gbuffer.valid_mask.to(device),
        normal_mask=gbuffer.normal_mask.to(device),
    )


def move_camera(camera: PinholeCamera, device: torch.device) -> PinholeCamera:
    return PinholeCamera(
        viewmats=camera.viewmats.to(device),
        intrinsics=camera.intrinsics.to(device),
        width=camera.width,
        height=camera.height,
    )


def move_shadow_bundle(shadow: ShadowMapBundle, device: torch.device) -> ShadowMapBundle:
    return ShadowMapBundle(
        light_indices=shadow.light_indices.to(device),
        light_cameras=[move_camera(light_camera, device) for light_camera in shadow.light_cameras],
        depth_maps=shadow.depth_maps.to(device),
        alpha_maps=shadow.alpha_maps.to(device),
        scene_radius=shadow.scene_radius,
        depth_bias=shadow.depth_bias,
    )


def make_structured_gbuffer(device: torch.device) -> GBuffer:
    return GBuffer(
        rgb=torch.ones((2, 2, 3), dtype=torch.float32, device=device),
        depth=torch.ones((2, 2), dtype=torch.float32, device=device),
        alpha=torch.ones((2, 2), dtype=torch.float32, device=device),
        position_cam=torch.tensor(
            [
                [[0.0, 0.0, 2.0], [0.5, 0.0, 2.0]],
                [[0.0, 0.5, 2.0], [0.5, 0.5, 2.0]],
            ],
            dtype=torch.float32,
            device=device,
        ),
        normal_cam=torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
            device=device,
        ),
        valid_mask=torch.tensor([[True, True], [False, True]], device=device),
        normal_mask=torch.tensor([[True, True], [True, False]], device=device),
    )
