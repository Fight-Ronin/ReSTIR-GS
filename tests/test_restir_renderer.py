from __future__ import annotations

import math

import pytest
import torch

from restir_gs.lighting.deferred import PointLights
from restir_gs.render.gbuffer import GBuffer
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.restir.renderer import (
    RestirRenderSettings,
    all_numeric_finite,
    evaluate_restir_frame_from_gbuffer,
    make_restir_metric_rows,
)


def test_no_history_temporal_output_matches_initial_exactly() -> None:
    settings = RestirRenderSettings(candidate_count=1)

    result = evaluate_restir_frame_from_gbuffer(make_gbuffer(), make_camera(), make_lights(), frame_index=0, settings=settings)

    assert torch.equal(result.temporal.contribution_rgb, result.initial.contribution_rgb)
    assert torch.equal(result.temporal.composite_rgb, result.initial.composite_rgb)
    assert torch.equal(result.temporal_reservoir.light_indices, result.initial_reservoir.light_indices)
    assert not bool(result.lookup.valid_mask.any())


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


def test_renderer_rows_have_expected_schema_and_finite_metrics() -> None:
    settings = RestirRenderSettings(candidate_count=1)
    result = evaluate_restir_frame_from_gbuffer(make_gbuffer(), make_camera(), make_lights(), frame_index=3, settings=settings)

    rows = make_restir_metric_rows("tiny_asset", result, settings)

    assert len(rows) == 4
    for row in rows:
        assert {
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
            "reuse_pixels",
            "reuse_fraction",
            "mean_relative_depth_error",
            "mean_motion_pixels",
            "reservoir_m_mean",
            "reservoir_m_max",
            "mae",
            "rmse",
            "bias_r",
            "bias_g",
            "bias_b",
            "mean_abs_bias",
        } == set(row)
        for key, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), key
    assert all_numeric_finite(rows)


def test_renderer_rejects_frames_without_valid_lighting_pixels() -> None:
    gbuffer = make_gbuffer(valid=False)

    with pytest.raises(RuntimeError, match="no valid lighting pixels"):
        evaluate_restir_frame_from_gbuffer(gbuffer, make_camera(), make_lights(), frame_index=0)


def make_camera() -> PinholeCamera:
    return PinholeCamera(
        viewmats=torch.eye(4, dtype=torch.float32)[None],
        intrinsics=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
        width=2,
        height=2,
    )


def make_gbuffer(valid: bool = True) -> GBuffer:
    ys, xs = torch.meshgrid(torch.arange(2, dtype=torch.float32), torch.arange(2, dtype=torch.float32), indexing="ij")
    depth = torch.full((2, 2), 2.0, dtype=torch.float32)
    position = torch.stack((xs * depth, ys * depth, depth), dim=-1)
    normal = torch.zeros_like(position)
    normal[..., 2] = 1.0
    mask = torch.full((2, 2), valid, dtype=torch.bool)
    return GBuffer(
        rgb=torch.full((2, 2, 3), 0.6, dtype=torch.float32),
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
