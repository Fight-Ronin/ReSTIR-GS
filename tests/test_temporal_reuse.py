from __future__ import annotations

import torch

from restir_gs.lighting.asset_lights import WorldPointLights, world_lights_to_camera_lights
from restir_gs.lighting.deferred import PointLights
from restir_gs.render.gbuffer import GBuffer
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.restir.initial import LightingEstimatorBuffers, evaluate_selected_light_contribution
from restir_gs.restir.temporal import (
    TemporalLookup,
    TemporalReservoirState,
    combine_temporal_reservoirs,
    reproject_current_to_previous,
)


def test_identity_reprojection_maps_pixels_to_themselves() -> None:
    gbuffer = make_projective_gbuffer(width=3, height=2, depth=2.0)
    camera = make_camera(width=3, height=2)

    lookup = reproject_current_to_previous(gbuffer, camera, gbuffer, camera)

    xs = torch.arange(3).expand(2, 3)
    ys = torch.arange(2)[:, None].expand(2, 3)
    assert torch.equal(lookup.prev_pixels[..., 0], xs)
    assert torch.equal(lookup.prev_pixels[..., 1], ys)
    assert torch.equal(lookup.valid_mask, torch.ones((2, 3), dtype=torch.bool))
    assert torch.equal(lookup.pre_gate_mask, torch.ones((2, 3), dtype=torch.bool))
    assert torch.equal(lookup.normal_pass_mask, torch.ones((2, 3), dtype=torch.bool))
    assert torch.equal(lookup.rgb_pass_mask, torch.ones((2, 3), dtype=torch.bool))
    assert torch.equal(lookup.motion_pass_mask, torch.ones((2, 3), dtype=torch.bool))
    assert torch.allclose(lookup.relative_depth_error, torch.zeros((2, 3)))
    assert torch.allclose(lookup.normal_dot, torch.ones((2, 3)))
    assert torch.allclose(lookup.normal_abs_dot, torch.ones((2, 3)))
    assert torch.allclose(lookup.rgb_distance, torch.zeros((2, 3)))


def test_depth_mismatch_rejects_history_reuse() -> None:
    current = make_projective_gbuffer(width=2, height=2, depth=2.0)
    previous = make_projective_gbuffer(width=2, height=2, depth=3.0)
    camera = make_camera(width=2, height=2)

    lookup = reproject_current_to_previous(current, camera, previous, camera, depth_tolerance=0.05)

    assert not bool(lookup.valid_mask.any())
    assert not bool(lookup.pre_gate_mask.any())
    assert not bool(lookup.normal_pass_mask.any())
    assert not bool(lookup.rgb_pass_mask.any())
    assert not bool(lookup.motion_pass_mask.any())
    assert bool(torch.isinf(lookup.relative_depth_error).all())


def test_invalid_previous_pixels_reject_history_reuse() -> None:
    current = make_projective_gbuffer(width=2, height=2, depth=2.0)
    previous = make_projective_gbuffer(width=2, height=2, depth=2.0)
    previous.valid_mask[0, 0] = False
    camera = make_camera(width=2, height=2)

    lookup = reproject_current_to_previous(current, camera, previous, camera, search_radius=0)

    assert not bool(lookup.valid_mask[0, 0])
    assert not bool(lookup.pre_gate_mask[0, 0])
    assert bool(lookup.valid_mask[0, 1])


def test_neighborhood_search_repairs_invalid_nearest_previous_pixel() -> None:
    current = make_projective_gbuffer(width=2, height=1, depth=2.0)
    previous = make_projective_gbuffer(width=2, height=1, depth=2.0)
    previous.valid_mask[0, 0] = False
    camera = make_camera(width=2, height=1)

    lookup = reproject_current_to_previous(current, camera, previous, camera, search_radius=1)

    assert bool(lookup.valid_mask[0, 0])
    assert bool(lookup.pre_gate_mask[0, 0])
    assert int(lookup.prev_pixels[0, 0, 0]) == 1
    assert int(lookup.prev_pixels[0, 0, 1]) == 0


def test_perpendicular_normal_rejects_history_after_pre_gate() -> None:
    current = make_projective_gbuffer(width=2, height=2, depth=2.0)
    previous = make_projective_gbuffer(width=2, height=2, depth=2.0, normal=(1.0, 0.0, 0.0))
    camera = make_camera(width=2, height=2)

    lookup = reproject_current_to_previous(current, camera, previous, camera, normal_threshold=0.85)

    assert bool(lookup.pre_gate_mask.all())
    assert not bool(lookup.valid_mask.any())
    assert not bool(lookup.normal_pass_mask.any())
    assert bool(lookup.rgb_pass_mask.all())
    assert bool(lookup.motion_pass_mask.all())
    assert torch.allclose(lookup.normal_dot, torch.zeros((2, 2)))
    assert torch.allclose(lookup.normal_abs_dot, torch.zeros((2, 2)))


def test_reversed_normal_passes_unoriented_normal_gate() -> None:
    current = make_projective_gbuffer(width=2, height=2, depth=2.0)
    previous = make_projective_gbuffer(width=2, height=2, depth=2.0, normal=(0.0, 0.0, -1.0))
    camera = make_camera(width=2, height=2)

    lookup = reproject_current_to_previous(current, camera, previous, camera, normal_threshold=0.85)

    assert bool(lookup.pre_gate_mask.all())
    assert bool(lookup.normal_pass_mask.all())
    assert bool(lookup.valid_mask.all())
    assert torch.allclose(lookup.normal_dot, -torch.ones((2, 2)))
    assert torch.allclose(lookup.normal_abs_dot, torch.ones((2, 2)))


def test_rgb_mismatch_rejects_history_after_pre_gate() -> None:
    current = make_projective_gbuffer(width=2, height=2, depth=2.0)
    previous = make_projective_gbuffer(width=2, height=2, depth=2.0)
    previous.rgb.zero_()
    camera = make_camera(width=2, height=2)

    lookup = reproject_current_to_previous(current, camera, previous, camera, rgb_threshold=0.2)

    assert bool(lookup.pre_gate_mask.all())
    assert not bool(lookup.valid_mask.any())
    assert bool(lookup.normal_pass_mask.all())
    assert not bool(lookup.rgb_pass_mask.any())
    assert bool(lookup.motion_pass_mask.all())
    assert torch.allclose(lookup.rgb_distance, torch.ones((2, 2)))


def test_large_motion_rejects_history_when_motion_cap_enabled() -> None:
    current = make_projective_gbuffer(width=3, height=1, depth=2.0)
    previous = make_projective_gbuffer(width=3, height=1, depth=2.0)
    current_camera = make_camera(width=3, height=1)
    previous_camera = PinholeCamera(
        viewmats=torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        ),
        intrinsics=current_camera.intrinsics.clone(),
        width=3,
        height=1,
    )

    lookup = reproject_current_to_previous(current, current_camera, previous, previous_camera, max_motion_pixels=0.49)

    assert bool(lookup.pre_gate_mask.any())
    assert not bool(lookup.valid_mask.any())
    assert bool(lookup.normal_pass_mask.any())
    assert bool(lookup.rgb_pass_mask.any())
    assert not bool(lookup.motion_pass_mask.any())


def test_disabled_optional_gates_preserve_depth_only_behavior() -> None:
    current = make_projective_gbuffer(width=2, height=2, depth=2.0)
    previous = make_projective_gbuffer(width=2, height=2, depth=2.0, rgb=0.0, normal=(1.0, 0.0, 0.0))
    camera = make_camera(width=2, height=2)

    lookup = reproject_current_to_previous(
        current,
        camera,
        previous,
        camera,
        normal_threshold=None,
        rgb_threshold=None,
        max_motion_pixels=None,
    )

    assert bool(lookup.pre_gate_mask.all())
    assert bool(lookup.valid_mask.all())
    assert bool(lookup.normal_pass_mask.all())
    assert bool(lookup.rgb_pass_mask.all())
    assert bool(lookup.motion_pass_mask.all())


def test_no_valid_history_matches_current_initial_output_exactly() -> None:
    gbuffer = make_single_pixel_gbuffer(valid=True)
    lights = make_two_lights()
    current_buffers = LightingEstimatorBuffers(
        contribution_rgb=torch.full((1, 1, 3), 0.25),
        composite_rgb=torch.full((1, 1, 3), 0.45),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    current = make_temporal_reservoir(light=0, W=2.0, M=8, valid=True)
    previous = make_temporal_reservoir(light=1, W=3.0, M=16, valid=False)
    lookup = make_lookup(valid=False)

    buffers, reservoir = combine_temporal_reservoirs(
        gbuffer,
        lights,
        current_buffers,
        current,
        previous,
        lookup,
        selection_seed=10,
    )

    assert torch.equal(buffers.contribution_rgb, current_buffers.contribution_rgb)
    assert torch.equal(buffers.composite_rgb, current_buffers.composite_rgb)
    assert torch.equal(reservoir.light_indices, current.light_indices)
    assert torch.equal(reservoir.M, current.M)
    assert torch.equal(reservoir.valid_mask, current.valid_mask)


def test_valid_history_accumulates_m_and_produces_finite_w() -> None:
    gbuffer = make_single_pixel_gbuffer(valid=True)
    lights = make_two_lights()
    current_buffers = LightingEstimatorBuffers(
        contribution_rgb=torch.zeros((1, 1, 3)),
        composite_rgb=gbuffer.rgb.clone(),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    current = make_temporal_reservoir(light=0, W=1.0, M=8, valid=True)
    previous = make_temporal_reservoir(light=1, W=1.0, M=16, valid=True)
    lookup = make_lookup(valid=True)

    buffers, reservoir = combine_temporal_reservoirs(
        gbuffer,
        lights,
        current_buffers,
        current,
        previous,
        lookup,
        selection_seed=11,
    )

    assert reservoir.M.item() == 24
    assert bool(reservoir.valid_mask.item())
    assert torch.isfinite(reservoir.W).all()
    assert torch.isfinite(buffers.contribution_rgb).all()


def test_history_m_cap_clamps_effective_previous_m() -> None:
    gbuffer = make_single_pixel_gbuffer(valid=True)
    lights = make_two_lights()
    current_buffers = LightingEstimatorBuffers(
        contribution_rgb=torch.zeros((1, 1, 3)),
        composite_rgb=gbuffer.rgb.clone(),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    current = make_temporal_reservoir(light=0, W=1.0, M=8, valid=True)
    previous = make_temporal_reservoir(light=1, W=1.0, M=16, valid=True)
    lookup = make_lookup(valid=True)

    buffers, reservoir = combine_temporal_reservoirs(
        gbuffer,
        lights,
        current_buffers,
        current,
        previous,
        lookup,
        selection_seed=11,
        history_m_cap=8,
    )

    assert reservoir.M.item() == 16
    assert bool(reservoir.valid_mask.item())
    assert torch.isfinite(reservoir.W).all()
    assert torch.isfinite(buffers.contribution_rgb).all()


def test_invalid_history_m_cap_fails_loudly() -> None:
    gbuffer = make_single_pixel_gbuffer(valid=True)
    lights = make_two_lights()
    current_buffers = LightingEstimatorBuffers(
        contribution_rgb=torch.zeros((1, 1, 3)),
        composite_rgb=gbuffer.rgb.clone(),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    current = make_temporal_reservoir(light=0, W=1.0, M=8, valid=True)
    previous = make_temporal_reservoir(light=1, W=1.0, M=16, valid=True)
    lookup = make_lookup(valid=True)

    try:
        combine_temporal_reservoirs(
            gbuffer,
            lights,
            current_buffers,
            current,
            previous,
            lookup,
            selection_seed=11,
            history_m_cap=0,
        )
    except ValueError as exc:
        assert "history_m_cap" in str(exc)
    else:
        raise AssertionError("Expected invalid history_m_cap to fail.")


def test_history_light_is_reevaluated_at_current_pixel() -> None:
    gbuffer = make_single_pixel_gbuffer(valid=True)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32),
        colors=torch.ones((1, 3), dtype=torch.float32),
        intensities=torch.ones((1,), dtype=torch.float32),
    )
    current_buffers = LightingEstimatorBuffers(
        contribution_rgb=torch.zeros((1, 1, 3)),
        composite_rgb=gbuffer.rgb.clone(),
        valid_mask=torch.zeros((1, 1), dtype=torch.bool),
    )
    current = make_temporal_reservoir(light=0, W=0.0, M=0, valid=False)
    previous = make_temporal_reservoir(light=0, W=1.0, M=1, valid=True)
    lookup = make_lookup(valid=True)

    buffers, _ = combine_temporal_reservoirs(
        gbuffer,
        lights,
        current_buffers,
        current,
        previous,
        lookup,
        selection_seed=12,
    )
    expected = evaluate_selected_light_contribution(gbuffer, lights, torch.zeros((1, 1, 1), dtype=torch.long)).squeeze(2)

    assert torch.allclose(buffers.contribution_rgb, expected)


def test_temporal_combine_uses_custom_contribution_evaluator() -> None:
    gbuffer = make_single_pixel_gbuffer(valid=True)
    lights = make_two_lights()
    current_buffers = LightingEstimatorBuffers(
        contribution_rgb=torch.zeros((1, 1, 3)),
        composite_rgb=gbuffer.rgb.clone(),
        valid_mask=torch.zeros((1, 1), dtype=torch.bool),
    )
    current = make_temporal_reservoir(light=0, W=0.0, M=0, valid=False)
    previous = make_temporal_reservoir(light=1, W=1.0, M=1, valid=True)
    lookup = make_lookup(valid=True)

    def evaluator(light_indices: torch.Tensor) -> torch.Tensor:
        return light_indices.to(torch.float32)[..., None].expand(*light_indices.shape, 3)

    buffers, reservoir = combine_temporal_reservoirs(
        gbuffer,
        lights,
        current_buffers,
        current,
        previous,
        lookup,
        selection_seed=12,
        contribution_evaluator=evaluator,
    )

    assert torch.allclose(buffers.contribution_rgb, torch.ones((1, 1, 3)))
    assert torch.equal(reservoir.light_indices, torch.tensor([[1]]))


def test_temporal_combine_is_deterministic_for_same_seed() -> None:
    gbuffer = make_single_pixel_gbuffer(valid=True)
    lights = make_two_lights()
    current_buffers = LightingEstimatorBuffers(
        contribution_rgb=torch.zeros((1, 1, 3)),
        composite_rgb=gbuffer.rgb.clone(),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    current = make_temporal_reservoir(light=0, W=1.0, M=8, valid=True)
    previous = make_temporal_reservoir(light=1, W=1.0, M=16, valid=True)
    lookup = make_lookup(valid=True)

    a_buffers, a_reservoir = combine_temporal_reservoirs(gbuffer, lights, current_buffers, current, previous, lookup, 13)
    b_buffers, b_reservoir = combine_temporal_reservoirs(gbuffer, lights, current_buffers, current, previous, lookup, 13)

    assert torch.equal(a_reservoir.light_indices, b_reservoir.light_indices)
    assert torch.allclose(a_reservoir.W, b_reservoir.W)
    assert torch.allclose(a_buffers.contribution_rgb, b_buffers.contribution_rgb)


def test_invalid_current_pixels_preserve_original_rgb_and_zero_contribution() -> None:
    gbuffer = make_single_pixel_gbuffer(valid=False)
    lights = make_two_lights()
    current_buffers = LightingEstimatorBuffers(
        contribution_rgb=torch.zeros((1, 1, 3)),
        composite_rgb=gbuffer.rgb.clone(),
        valid_mask=torch.zeros((1, 1), dtype=torch.bool),
    )
    current = make_temporal_reservoir(light=0, W=0.0, M=0, valid=False)
    previous = make_temporal_reservoir(light=1, W=1.0, M=16, valid=True)
    lookup = make_lookup(valid=True)

    buffers, reservoir = combine_temporal_reservoirs(gbuffer, lights, current_buffers, current, previous, lookup, 14)

    assert torch.equal(buffers.contribution_rgb, torch.zeros((1, 1, 3)))
    assert torch.equal(buffers.composite_rgb, gbuffer.rgb)
    assert not bool(buffers.valid_mask.item())
    assert not bool(reservoir.valid_mask.item())


def test_temporal_world_light_indices_remain_stable_across_frames() -> None:
    world_lights = WorldPointLights(
        positions_world=torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]], dtype=torch.float32),
        colors=torch.tensor([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=torch.float32),
        intensities=torch.tensor([0.25, 0.5], dtype=torch.float32),
    )
    camera_a = make_camera(width=1, height=1)
    camera_b = PinholeCamera(
        viewmats=torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        ),
        intrinsics=torch.eye(3, dtype=torch.float32)[None],
        width=1,
        height=1,
    )

    lights_a = world_lights_to_camera_lights(world_lights, camera_a)
    lights_b = world_lights_to_camera_lights(world_lights, camera_b)

    assert torch.equal(lights_a.colors[1], world_lights.colors[1])
    assert torch.equal(lights_b.colors[1], world_lights.colors[1])
    assert torch.equal(lights_a.intensities, lights_b.intensities)
    assert torch.allclose(lights_b.positions_cam[1], world_lights.positions_world[1] + torch.tensor([1.0, 0.0, 0.0]))


def make_camera(width: int, height: int) -> PinholeCamera:
    intrinsics = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    return PinholeCamera(
        viewmats=torch.eye(4, dtype=torch.float32)[None],
        intrinsics=intrinsics,
        width=width,
        height=height,
    )


def make_projective_gbuffer(
    width: int,
    height: int,
    depth: float,
    rgb: float = 1.0,
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> GBuffer:
    ys, xs = torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij")
    z = torch.full((height, width), depth, dtype=torch.float32)
    position = torch.stack((xs * z, ys * z, z), dim=-1)
    normal_tensor = torch.tensor(normal, dtype=torch.float32).expand_as(position).clone()
    valid = torch.ones((height, width), dtype=torch.bool)
    return GBuffer(
        rgb=torch.full((height, width, 3), rgb, dtype=torch.float32),
        depth=z,
        alpha=torch.ones((height, width), dtype=torch.float32),
        position_cam=position,
        normal_cam=normal_tensor,
        valid_mask=valid.clone(),
        normal_mask=valid.clone(),
    )


def make_single_pixel_gbuffer(valid: bool) -> GBuffer:
    valid_mask = torch.tensor([[valid]], dtype=torch.bool)
    return GBuffer(
        rgb=torch.ones((1, 1, 3), dtype=torch.float32),
        depth=torch.ones((1, 1), dtype=torch.float32),
        alpha=torch.ones((1, 1), dtype=torch.float32),
        position_cam=torch.zeros((1, 1, 3), dtype=torch.float32),
        normal_cam=torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32),
        valid_mask=valid_mask,
        normal_mask=torch.ones((1, 1), dtype=torch.bool),
    )


def make_temporal_reservoir(light: int, W: float, M: int, valid: bool) -> TemporalReservoirState:
    return TemporalReservoirState(
        light_indices=torch.tensor([[light]], dtype=torch.long),
        selected_target=torch.ones((1, 1), dtype=torch.float32) if valid else torch.zeros((1, 1), dtype=torch.float32),
        weight_sum=torch.ones((1, 1), dtype=torch.float32) if valid else torch.zeros((1, 1), dtype=torch.float32),
        W=torch.full((1, 1), W, dtype=torch.float32),
        M=torch.full((1, 1), M, dtype=torch.long),
        valid_mask=torch.tensor([[valid]], dtype=torch.bool),
    )


def make_lookup(valid: bool) -> TemporalLookup:
    return TemporalLookup(
        prev_pixels=torch.zeros((1, 1, 2), dtype=torch.long),
        valid_mask=torch.tensor([[valid]], dtype=torch.bool),
        pre_gate_mask=torch.tensor([[valid]], dtype=torch.bool),
        normal_pass_mask=torch.tensor([[valid]], dtype=torch.bool),
        rgb_pass_mask=torch.tensor([[valid]], dtype=torch.bool),
        motion_pass_mask=torch.tensor([[valid]], dtype=torch.bool),
        relative_depth_error=torch.zeros((1, 1), dtype=torch.float32),
        normal_dot=torch.ones((1, 1), dtype=torch.float32) if valid else torch.zeros((1, 1), dtype=torch.float32),
        normal_abs_dot=torch.ones((1, 1), dtype=torch.float32) if valid else torch.zeros((1, 1), dtype=torch.float32),
        rgb_distance=torch.zeros((1, 1), dtype=torch.float32) if valid else torch.full((1, 1), float("inf")),
        motion_pixels=torch.zeros((1, 1, 2), dtype=torch.float32),
    )


def make_two_lights() -> PointLights:
    return PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
        colors=torch.ones((2, 3), dtype=torch.float32),
        intensities=torch.ones((2,), dtype=torch.float32),
    )
