from __future__ import annotations

import pytest
import torch

from restir_gs.render.camera_probe import (
    CameraProbeScore,
    camera_config_payload,
    camera_from_config_payload,
    make_probe_camera_candidates,
    parse_float_list,
    score_render_buffers,
    select_best_candidate,
)
from restir_gs.render.gsplat_renderer import RenderBuffers


def make_buffers(valid_mask: torch.Tensor, brightness: float = 0.5) -> RenderBuffers:
    height, width = valid_mask.shape
    rgb = torch.full((height, width, 3), brightness, dtype=torch.float32)
    depth = torch.where(valid_mask, torch.ones((height, width), dtype=torch.float32), torch.zeros((height, width), dtype=torch.float32))
    alpha = torch.where(valid_mask, torch.ones((height, width), dtype=torch.float32), torch.zeros((height, width), dtype=torch.float32))
    return RenderBuffers(rgb=rgb, depth=depth, alpha=alpha)


def test_probe_yaw_pitch_camera_maps_target_to_positive_z() -> None:
    means = torch.tensor([[-1.0, -1.0, 2.0], [1.0, 1.0, 4.0]], dtype=torch.float32)

    candidates = make_probe_camera_candidates(
        means,
        yaw_values=[-30.0, 0.0, 30.0],
        pitch_values=[-10.0, 10.0],
        radius_scales=[1.1],
        width=64,
        height=64,
        bbox_percentile=1.0,
    )

    assert len(candidates) == 6
    for candidate in candidates:
        target = torch.tensor(candidate.info.target, dtype=torch.float32)
        target_cam = torch.cat((target, torch.ones(1))) @ candidate.camera.viewmats[0].T
        assert target_cam[2] > 0.0


def test_camera_config_payload_round_trips_exact_camera_tensors() -> None:
    means = torch.tensor([[-1.0, -1.0, 2.0], [1.0, 1.0, 4.0]], dtype=torch.float32)
    candidate = make_probe_camera_candidates(
        means,
        yaw_values=[15.0],
        pitch_values=[-10.0],
        radius_scales=[1.3],
        width=32,
        height=24,
        bbox_percentile=1.0,
    )[0]

    payload = camera_config_payload(
        candidate.camera,
        candidate.info,
        score=CameraProbeScore(1.0, 12, 0.5, 0.5, 0.0, 0.4),
        candidate_index=3,
    )
    restored = camera_from_config_payload(payload, device="cpu")

    assert restored.width == 32
    assert restored.height == 24
    assert torch.equal(restored.viewmats, candidate.camera.viewmats)
    assert torch.equal(restored.intrinsics, candidate.camera.intrinsics)
    assert payload["candidate_index"] == 3


def test_scoring_favors_centered_coverage_over_tiny_off_center_coverage() -> None:
    centered = torch.zeros((8, 8), dtype=torch.bool)
    centered[3:5, 3:5] = True
    off_center = torch.zeros((8, 8), dtype=torch.bool)
    off_center[0, 0] = True

    centered_score = score_render_buffers(make_buffers(centered), border_pixels=1)
    off_center_score = score_render_buffers(make_buffers(off_center), border_pixels=1)

    assert centered_score.score > off_center_score.score
    assert centered_score.central_coverage > off_center_score.central_coverage


def test_scoring_penalizes_border_heavy_clipped_views() -> None:
    border = torch.zeros((8, 8), dtype=torch.bool)
    border[0, :] = True
    border[7, :] = True
    center = torch.zeros((8, 8), dtype=torch.bool)
    center[3:5, 3:5] = True

    border_score = score_render_buffers(make_buffers(border), border_pixels=1)
    center_score = score_render_buffers(make_buffers(center), border_pixels=1)

    assert border_score.border_coverage > center_score.border_coverage
    assert center_score.score > border_score.score


def test_select_best_candidate_chooses_highest_finite_score() -> None:
    scores = [
        CameraProbeScore(0.2, 10, 0.1, 0.1, 0.0, 0.5),
        CameraProbeScore(float("nan"), 99, 0.9, 0.9, 0.0, 0.5),
        CameraProbeScore(0.6, 20, 0.2, 0.2, 0.0, 0.5),
        CameraProbeScore(0.4, 30, 0.3, 0.3, 0.0, 0.5),
    ]

    assert select_best_candidate(scores) == 2


def test_select_best_candidate_rejects_all_non_finite_scores() -> None:
    scores = [CameraProbeScore(float("nan"), 10, 0.1, 0.1, 0.0, 0.5)]

    with pytest.raises(RuntimeError, match="non-finite scores"):
        select_best_candidate(scores)


def test_select_best_candidate_rejects_all_zero_valid_pixels() -> None:
    scores = [CameraProbeScore(1.0, 0, 0.0, 0.0, 0.0, 0.0)]

    with pytest.raises(RuntimeError, match="zero valid pixels"):
        select_best_candidate(scores)


def test_parse_float_list_returns_expected_values() -> None:
    assert parse_float_list("-30, -15,0, 15") == [-30.0, -15.0, 0.0, 15.0]

    with pytest.raises(ValueError):
        parse_float_list(" , ")
