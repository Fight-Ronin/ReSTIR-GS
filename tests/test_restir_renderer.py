from __future__ import annotations

import math

import pytest
import torch

import restir_gs.restir.renderer as renderer_module
import restir_gs.restir.visibility as restir_visibility_module
from restir_gs.lighting.deferred import PointLights
from restir_gs.lighting.visibility import ShadowMapBundle
from restir_gs.render.gbuffer import GBuffer
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.restir.renderer import (
    RESTIR_TIMING_FIELDS,
    RestirFrameTimings,
    RestirRenderSettings,
    apply_confidence_clamped_temporal_filter,
    all_numeric_finite,
    evaluate_restir_display_frame_from_gbuffer,
    evaluate_restir_frame_from_gbuffer,
    make_restir_metric_rows,
    summarize_restir_asset_timing_rows,
    summarize_restir_timing_rows,
)
from restir_gs.restir.initial import LightingEstimatorBuffers
from restir_gs.restir.temporal import TemporalLookup


def test_no_history_temporal_output_matches_initial_exactly() -> None:
    settings = RestirRenderSettings(candidate_count=1)

    result = evaluate_restir_frame_from_gbuffer(make_gbuffer(), make_camera(), make_lights(), frame_index=0, settings=settings)

    assert torch.equal(result.temporal.contribution_rgb, result.initial.contribution_rgb)
    assert torch.equal(result.temporal.composite_rgb, result.initial.composite_rgb)
    assert torch.equal(result.temporal_filtered.contribution_rgb, result.initial.contribution_rgb)
    assert torch.equal(result.temporal_filtered.composite_rgb, result.initial.composite_rgb)
    assert torch.equal(result.temporal_reservoir.light_indices, result.initial_reservoir.light_indices)
    assert not bool(result.lookup.valid_mask.any())
    assert not bool(result.temporal_filter_stats.alpha.any())


def test_display_path_skips_reference_but_matches_evaluation_sampling() -> None:
    settings = RestirRenderSettings(candidate_count=2, candidate_seed_base=10, initial_selection_seed_base=20)
    gbuffer = make_gbuffer()
    camera = make_camera()
    lights = make_lights()

    display = evaluate_restir_display_frame_from_gbuffer(gbuffer, camera, lights, frame_index=0, settings=settings)
    evaluation = evaluate_restir_frame_from_gbuffer(gbuffer, camera, lights, frame_index=0, settings=settings)

    assert not hasattr(display, "reference")
    assert torch.equal(display.proposal_samples.light_indices, evaluation.proposal_samples.light_indices)
    assert torch.equal(display.initial_reservoir.light_indices, evaluation.initial_reservoir.light_indices)
    assert torch.allclose(display.initial.contribution_rgb, evaluation.initial.contribution_rgb)
    assert torch.allclose(display.temporal_filtered.composite_rgb, evaluation.temporal_filtered.composite_rgb)


def test_valid_history_accumulates_m_and_keeps_finite_weights() -> None:
    settings = RestirRenderSettings(candidate_count=2, candidate_seed_base=10, initial_selection_seed_base=20, temporal_selection_seed_base=30)
    gbuffer = make_gbuffer()
    camera = make_camera()
    lights = make_lights()
    first = evaluate_restir_frame_from_gbuffer(gbuffer, camera, lights, frame_index=0, settings=settings)

    second = evaluate_restir_frame_from_gbuffer(gbuffer, camera, lights, frame_index=1, settings=settings, previous_history=first.history)

    valid = second.temporal_reservoir.valid_mask
    assert bool(second.lookup.valid_mask.any())
    assert torch.all(second.temporal_reservoir.M[valid] >= second.initial_reservoir.M[valid])
    assert torch.isfinite(second.temporal_reservoir.W[valid]).all()
    assert torch.isfinite(second.temporal.contribution_rgb).all()


def test_failing_temporal_gate_falls_back_to_initial() -> None:
    settings = RestirRenderSettings(
        candidate_count=2,
        candidate_seed_base=10,
        initial_selection_seed_base=20,
        temporal_selection_seed_base=30,
        temporal_rgb_threshold=0.1,
    )
    camera = make_camera()
    lights = make_lights()
    first = evaluate_restir_frame_from_gbuffer(make_gbuffer(rgb=0.0), camera, lights, frame_index=0, settings=settings)

    second = evaluate_restir_frame_from_gbuffer(make_gbuffer(rgb=0.6), camera, lights, frame_index=1, settings=settings, previous_history=first.history)

    assert bool(second.lookup.pre_gate_mask.any())
    assert not bool(second.lookup.valid_mask.any())
    assert torch.equal(second.temporal.contribution_rgb, second.initial.contribution_rgb)
    assert torch.equal(second.temporal.composite_rgb, second.initial.composite_rgb)
    assert torch.equal(second.temporal_filtered.contribution_rgb, second.initial.contribution_rgb)
    assert torch.equal(second.temporal_filtered.composite_rgb, second.initial.composite_rgb)


def test_temporal_filter_blends_valid_history_by_confidence() -> None:
    gbuffer = make_gbuffer()
    current = make_estimator_buffers(gbuffer, 0.2)
    previous = make_estimator_buffers(gbuffer, 0.6)
    settings = RestirRenderSettings(
        temporal_filter_blend_max=0.25,
        temporal_filter_clamp_scale=10.0,
        temporal_filter_clamp_min=0.0,
    )

    filtered, stats = apply_confidence_clamped_temporal_filter(gbuffer, current, previous, make_lookup(), settings)

    expected = torch.full_like(current.contribution_rgb, 0.3)
    assert torch.allclose(filtered.contribution_rgb, expected)
    assert torch.allclose(stats.alpha, torch.full_like(stats.alpha, 0.25))
    assert torch.allclose(filtered.composite_rgb, gbuffer.rgb * settings.ambient + expected)


def test_temporal_filter_confidence_reduces_alpha_near_thresholds() -> None:
    gbuffer = make_gbuffer()
    current = make_estimator_buffers(gbuffer, 0.2)
    previous = make_estimator_buffers(gbuffer, 0.6)
    settings = RestirRenderSettings(
        temporal_filter_blend_max=0.15,
        temporal_filter_clamp_scale=10.0,
        temporal_filter_clamp_min=0.0,
    )
    lookup = make_lookup(relative_depth_error=0.025, normal_abs_dot=0.9, rgb_distance=0.1, motion=(16.0, 0.0))

    _, stats = apply_confidence_clamped_temporal_filter(gbuffer, current, previous, lookup, settings)

    assert torch.allclose(stats.alpha, torch.full_like(stats.alpha, 0.075))


def test_temporal_filter_clamps_large_history_before_blending() -> None:
    gbuffer = make_gbuffer()
    current = make_estimator_buffers(gbuffer, 0.2)
    previous = make_estimator_buffers(gbuffer, 2.0)
    settings = RestirRenderSettings(
        temporal_filter_blend_max=1.0,
        temporal_filter_clamp_scale=0.5,
        temporal_filter_clamp_min=0.0,
    )

    filtered, stats = apply_confidence_clamped_temporal_filter(gbuffer, current, previous, make_lookup(), settings)

    assert torch.allclose(filtered.contribution_rgb, torch.full_like(current.contribution_rgb, 0.3))
    assert torch.all(stats.clamp_delta > 0.0)


def test_temporal_filter_rejected_history_matches_current_exactly() -> None:
    gbuffer = make_gbuffer()
    current = make_estimator_buffers(gbuffer, 0.2)
    previous = make_estimator_buffers(gbuffer, 0.6)

    filtered, stats = apply_confidence_clamped_temporal_filter(
        gbuffer,
        current,
        previous,
        make_lookup(valid=False),
        RestirRenderSettings(temporal_filter_blend_max=1.0),
    )

    assert torch.equal(filtered.contribution_rgb, current.contribution_rgb)
    assert torch.equal(filtered.composite_rgb, current.composite_rgb)
    assert not bool(stats.alpha.any())


def test_renderer_rows_have_expected_schema_and_finite_metrics() -> None:
    settings = RestirRenderSettings(candidate_count=1)
    result = evaluate_restir_frame_from_gbuffer(make_gbuffer(), make_camera(), make_lights(), frame_index=3, settings=settings)

    rows = make_restir_metric_rows("tiny_asset", result, settings)

    assert len(rows) == 6
    for row in rows:
        expected_fields = {
            "asset_id",
            "frame_index",
            "estimator",
            "reference_quantity",
            "target_mode",
            "proposal",
            "k",
            "candidate_seed",
            "selection_seed",
            "valid_pixels",
            "pre_gate_pixels",
            "pre_gate_fraction",
            "normal_gate_pass_pixels",
            "normal_gate_pass_fraction",
            "normal_gate_pass_pre_gate_fraction",
            "rgb_gate_pass_pixels",
            "rgb_gate_pass_fraction",
            "rgb_gate_pass_pre_gate_fraction",
            "motion_gate_pass_pixels",
            "motion_gate_pass_fraction",
            "motion_gate_pass_pre_gate_fraction",
            "reuse_pixels",
            "reuse_fraction",
            "mean_relative_depth_error",
            "mean_temporal_normal_dot",
            "mean_temporal_normal_abs_dot",
            "mean_temporal_rgb_distance",
            "mean_motion_pixels",
            "mean_pre_gate_normal_dot",
            "mean_pre_gate_normal_abs_dot",
            "mean_pre_gate_rgb_distance",
            "mean_pre_gate_motion_pixels",
            "temporal_normal_threshold",
            "temporal_rgb_threshold",
            "temporal_max_motion_pixels",
            "temporal_reprojection_search_radius",
            "temporal_history_m_cap",
            "temporal_filter_blend_max",
            "temporal_filter_clamp_scale",
            "temporal_filter_clamp_min",
            "temporal_filter_confidence_mean",
            "temporal_filter_alpha_mean",
            "temporal_filter_alpha_max",
            "temporal_filter_history_delta_mean",
            "temporal_filter_clamp_delta_mean",
            "visibility_shadow_pcf_radius",
            "reservoir_m_mean",
            "reservoir_m_max",
            "mae",
            "rmse",
            "bias_r",
            "bias_g",
            "bias_b",
            "mean_abs_bias",
        }
        expected_fields.update(RESTIR_TIMING_FIELDS)
        assert expected_fields == set(row)
        for key, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), key
    assert all_numeric_finite(rows)


def test_cpu_timing_fields_default_to_finite_zero() -> None:
    settings = RestirRenderSettings(candidate_count=1)
    result = evaluate_restir_frame_from_gbuffer(make_gbuffer(), make_camera(), make_lights(), frame_index=1, settings=settings)

    rows = make_restir_metric_rows("tiny_asset", result, settings)

    for row in rows:
        for field in RESTIR_TIMING_FIELDS:
            assert field in row
            assert math.isfinite(float(row[field]))
            assert float(row[field]) == 0.0


def test_shadow_bundle_asset_timing_override_is_written_to_rows() -> None:
    settings = RestirRenderSettings(candidate_count=1)
    result = evaluate_restir_frame_from_gbuffer(make_gbuffer(), make_camera(), make_lights(), frame_index=1, settings=settings)

    rows = make_restir_metric_rows("tiny_asset", result, settings, shadow_bundle_asset_gpu_ms=12.5)

    assert {float(row["shadow_bundle_asset_gpu_ms"]) for row in rows} == {12.5}


def test_timing_summary_computes_exact_mean_max_count() -> None:
    first = RestirFrameTimings(render_rgbd_gpu_ms=2.0, frame_gpu_ms=5.0, frame_wall_ms=7.0).as_row_fields()
    second = RestirFrameTimings(render_rgbd_gpu_ms=4.0, frame_gpu_ms=9.0, frame_wall_ms=11.0).as_row_fields()
    rows = [
        {"asset_id": "a", **first},
        {"asset_id": "a", **second},
    ]

    summary = summarize_restir_timing_rows(rows)
    asset_summary = summarize_restir_asset_timing_rows(rows)

    assert summary["render_rgbd_gpu_ms"] == {"mean": 3.0, "max": 4.0, "count": 2}
    assert summary["frame_gpu_ms"] == {"mean": 7.0, "max": 9.0, "count": 2}
    assert asset_summary["a"]["frame_wall_ms"] == {"mean": 9.0, "max": 11.0, "count": 2}


def test_renderer_rejects_frames_without_valid_lighting_pixels() -> None:
    gbuffer = make_gbuffer(valid=False)

    with pytest.raises(RuntimeError, match="no valid lighting pixels"):
        evaluate_restir_frame_from_gbuffer(gbuffer, make_camera(), make_lights(), frame_index=0)


def test_visibility_target_requires_shadow_bundle() -> None:
    settings = RestirRenderSettings(target_mode="visibility", candidate_count=1)

    with pytest.raises(ValueError, match="ShadowMapBundle"):
        evaluate_restir_frame_from_gbuffer(make_gbuffer(), make_camera(), make_lights(), frame_index=0, settings=settings)


def test_visibility_target_renderer_rows_are_finite() -> None:
    settings = RestirRenderSettings(target_mode="visibility", candidate_count=1)
    result = evaluate_restir_frame_from_gbuffer(
        make_gbuffer(),
        make_camera(),
        make_lights(),
        frame_index=2,
        settings=settings,
        shadow_bundle=make_shadow_bundle(),
    )

    rows = make_restir_metric_rows("tiny_asset", result, settings)

    assert torch.equal(result.temporal.contribution_rgb, result.initial.contribution_rgb)
    assert {str(row["target_mode"]) for row in rows} == {"visibility"}
    assert all_numeric_finite(rows)


def test_visibility_target_uses_visibility_geometric_proposal_by_default() -> None:
    settings = RestirRenderSettings(target_mode="visibility", candidate_count=1)
    result = evaluate_restir_frame_from_gbuffer(
        make_gbuffer(),
        make_camera(),
        make_lights(),
        frame_index=2,
        settings=settings,
        shadow_bundle=make_shadow_bundle(),
    )

    rows = make_restir_metric_rows("tiny_asset", result, settings)

    assert {str(row["proposal"]) for row in rows} == {"visibility_geometric"}
    assert {int(row["temporal_history_m_cap"]) for row in rows} == {settings.candidate_count}
    assert {int(row["visibility_shadow_pcf_radius"]) for row in rows} == {1}
    assert all_numeric_finite(rows)


def test_visibility_target_reuses_selected_light_contributions(monkeypatch) -> None:
    call_count = 0
    original = renderer_module.evaluate_selected_light_visible_diffuse_cached

    def wrapped(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(renderer_module, "evaluate_selected_light_visible_diffuse_cached", wrapped)
    monkeypatch.setattr(restir_visibility_module, "evaluate_selected_light_visible_diffuse_cached", wrapped)

    settings = RestirRenderSettings(target_mode="visibility", candidate_count=2, include_mc_baseline=True)
    evaluate_restir_frame_from_gbuffer(
        make_gbuffer(),
        make_camera(),
        make_lights(),
        frame_index=2,
        settings=settings,
        shadow_bundle=make_shadow_bundle(),
    )

    assert call_count == 1


def test_diffuse_target_keeps_geometric_proposal() -> None:
    settings = RestirRenderSettings(target_mode="diffuse", candidate_count=1)
    result = evaluate_restir_frame_from_gbuffer(make_gbuffer(), make_camera(), make_lights(), frame_index=0, settings=settings)

    rows = make_restir_metric_rows("tiny_asset", result, settings)

    assert {str(row["proposal"]) for row in rows} == {"geometric"}


def make_camera() -> PinholeCamera:
    return PinholeCamera(
        viewmats=torch.eye(4, dtype=torch.float32)[None],
        intrinsics=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
        width=2,
        height=2,
    )


def make_gbuffer(valid: bool = True, rgb: float = 0.6) -> GBuffer:
    ys, xs = torch.meshgrid(torch.arange(2, dtype=torch.float32), torch.arange(2, dtype=torch.float32), indexing="ij")
    depth = torch.full((2, 2), 2.0, dtype=torch.float32)
    position = torch.stack((xs * depth, ys * depth, depth), dim=-1)
    normal = torch.zeros_like(position)
    normal[..., 2] = 1.0
    mask = torch.full((2, 2), valid, dtype=torch.bool)
    return GBuffer(
        rgb=torch.full((2, 2, 3), rgb, dtype=torch.float32),
        depth=depth,
        alpha=torch.ones((2, 2), dtype=torch.float32),
        position_cam=position,
        normal_cam=normal,
        valid_mask=mask.clone(),
        normal_mask=mask.clone(),
    )


def make_lights() -> PointLights:
    return PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 3.0], [2.0, 2.0, 4.0]], dtype=torch.float32),
        colors=torch.ones((2, 3), dtype=torch.float32),
        intensities=torch.ones((2,), dtype=torch.float32),
    )


def make_shadow_bundle() -> ShadowMapBundle:
    camera = PinholeCamera(
        viewmats=torch.eye(4, dtype=torch.float32)[None],
        intrinsics=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
        width=4,
        height=4,
    )
    return ShadowMapBundle(
        light_indices=torch.tensor([0, 1], dtype=torch.long),
        light_cameras=[camera, camera],
        depth_maps=torch.full((2, 4, 4), 10.0, dtype=torch.float32),
        alpha_maps=torch.zeros((2, 4, 4), dtype=torch.float32),
        scene_radius=1.0,
        depth_bias=0.0,
    )


def make_estimator_buffers(gbuffer: GBuffer, contribution_value: float) -> LightingEstimatorBuffers:
    contribution = torch.full_like(gbuffer.rgb, contribution_value)
    composite = gbuffer.rgb * 0.2 + contribution
    return LightingEstimatorBuffers(
        contribution_rgb=contribution,
        composite_rgb=composite,
        valid_mask=gbuffer.valid_mask & gbuffer.normal_mask,
    )


def make_lookup(
    valid: bool = True,
    relative_depth_error: float = 0.0,
    normal_abs_dot: float = 1.0,
    rgb_distance: float = 0.0,
    motion: tuple[float, float] = (0.0, 0.0),
) -> TemporalLookup:
    ys, xs = torch.meshgrid(torch.arange(2, dtype=torch.long), torch.arange(2, dtype=torch.long), indexing="ij")
    mask = torch.full((2, 2), valid, dtype=torch.bool)
    return TemporalLookup(
        prev_pixels=torch.stack((xs, ys), dim=-1),
        valid_mask=mask,
        pre_gate_mask=mask.clone(),
        normal_pass_mask=mask.clone(),
        rgb_pass_mask=mask.clone(),
        motion_pass_mask=mask.clone(),
        relative_depth_error=torch.full((2, 2), relative_depth_error, dtype=torch.float32),
        normal_dot=torch.full((2, 2), normal_abs_dot, dtype=torch.float32),
        normal_abs_dot=torch.full((2, 2), normal_abs_dot, dtype=torch.float32),
        rgb_distance=torch.full((2, 2), rgb_distance, dtype=torch.float32),
        motion_pixels=torch.tensor(motion, dtype=torch.float32).expand(2, 2, 2).clone(),
    )
