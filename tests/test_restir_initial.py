from __future__ import annotations

import math

import torch

from restir_gs.lighting.deferred import PointLights, evaluate_selected_light_diffuse
from restir_gs.render.gbuffer import GBuffer
from restir_gs.restir.initial import (
    estimate_ris_initial_diffuse,
    estimate_uniform_diffuse,
    sample_uniform_light_candidates,
)


def make_gbuffer(
    rgb: torch.Tensor,
    position: torch.Tensor,
    normal: torch.Tensor,
    valid_mask: torch.Tensor,
    normal_mask: torch.Tensor,
) -> GBuffer:
    height, width, _ = rgb.shape
    return GBuffer(
        rgb=rgb,
        depth=torch.ones((height, width), dtype=rgb.dtype),
        alpha=torch.ones((height, width), dtype=rgb.dtype),
        position_cam=position,
        normal_cam=normal,
        valid_mask=valid_mask,
        normal_mask=normal_mask,
    )


def test_uniform_light_candidates_are_deterministic_and_in_range() -> None:
    a = sample_uniform_light_candidates(3, 4, 5, 7, seed=2028, device="cpu")
    b = sample_uniform_light_candidates(3, 4, 5, 7, seed=2028, device="cpu")

    assert a.shape == (3, 4, 5)
    assert torch.equal(a, b)
    assert int(a.min()) >= 0
    assert int(a.max()) < 7


def test_evaluate_selected_light_diffuse_matches_closed_form() -> None:
    rgb = torch.tensor([[[0.5, 0.25, 1.0]]], dtype=torch.float32)
    position = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        colors=torch.tensor([[2.0, 1.0, 0.5]], dtype=torch.float32),
        intensities=torch.tensor([4.0], dtype=torch.float32),
    )

    diffuse = evaluate_selected_light_diffuse(
        gbuffer,
        lights,
        torch.zeros((1, 1, 1), dtype=torch.long),
        distance_epsilon=0.0,
    )

    expected_irradiance = torch.tensor([[[[8.0, 4.0, 2.0]]]], dtype=torch.float32)
    assert torch.allclose(diffuse, rgb[..., None, :] * expected_irradiance / math.pi)


def test_uniform_k_estimator_applies_sampling_scale() -> None:
    rgb = torch.ones((1, 1, 3), dtype=torch.float32)
    position = torch.zeros((1, 1, 3), dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
        colors=torch.ones((2, 3), dtype=torch.float32),
        intensities=torch.ones((2,), dtype=torch.float32),
    )
    candidates = torch.tensor([[[0, 0]]], dtype=torch.long)

    buffers = estimate_uniform_diffuse(gbuffer, lights, candidates, ambient=0.2)
    sampled = evaluate_selected_light_diffuse(gbuffer, lights, candidates).sum(dim=2)

    assert torch.allclose(buffers.diffuse_rgb, sampled * (2.0 / 2.0))


def test_ris_k1_matches_uniform_one_sample_for_positive_target() -> None:
    rgb = torch.ones((1, 1, 3), dtype=torch.float32)
    position = torch.zeros((1, 1, 3), dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
        colors=torch.ones((2, 3), dtype=torch.float32),
        intensities=torch.ones((2,), dtype=torch.float32),
    )
    candidates = torch.tensor([[[1]]], dtype=torch.long)

    uniform = estimate_uniform_diffuse(gbuffer, lights, candidates, ambient=0.2)
    ris, reservoir = estimate_ris_initial_diffuse(gbuffer, lights, candidates, selection_seed=2029, ambient=0.2)

    assert torch.allclose(ris.diffuse_rgb, uniform.diffuse_rgb)
    assert torch.allclose(ris.composite_rgb, uniform.composite_rgb)
    assert torch.equal(reservoir.light_indices, torch.tensor([[1]]))
    assert torch.equal(reservoir.M, torch.tensor([[1]]))


def test_zero_target_and_invalid_pixels_preserve_original_rgb() -> None:
    rgb = torch.tensor([[[0.8, 0.6, 0.4], [0.2, 0.4, 0.6]]], dtype=torch.float32)
    position = torch.zeros((1, 2, 3), dtype=torch.float32)
    normal = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, False]])
    normal_mask = torch.tensor([[True, True]])
    gbuffer = make_gbuffer(rgb, position, normal, valid_mask, normal_mask)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        colors=torch.ones((1, 3), dtype=torch.float32),
        intensities=torch.ones((1,), dtype=torch.float32),
    )
    candidates = torch.zeros((1, 2, 1), dtype=torch.long)

    ris, reservoir = estimate_ris_initial_diffuse(gbuffer, lights, candidates, ambient=0.2)

    assert torch.allclose(ris.diffuse_rgb, torch.zeros_like(rgb))
    assert torch.allclose(ris.composite_rgb, rgb)
    assert not bool(ris.valid_mask.any())
    assert torch.allclose(reservoir.W, torch.zeros((1, 2)))


def test_reservoir_state_shapes_and_m_for_valid_pixels() -> None:
    rgb = torch.ones((2, 3, 3), dtype=torch.float32)
    position = torch.zeros((2, 3, 3), dtype=torch.float32)
    normal = torch.zeros((2, 3, 3), dtype=torch.float32)
    normal[..., 2] = 1.0
    valid = torch.ones((2, 3), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=torch.float32),
        colors=torch.ones((3, 3), dtype=torch.float32),
        intensities=torch.ones((3,), dtype=torch.float32),
    )
    candidates = sample_uniform_light_candidates(2, 3, 4, 3, seed=99, device="cpu")

    _, reservoir = estimate_ris_initial_diffuse(gbuffer, lights, candidates, selection_seed=100, ambient=0.2)

    assert reservoir.light_indices.shape == (2, 3)
    assert reservoir.target_values.shape == (2, 3, 4)
    assert reservoir.weight_sum.shape == (2, 3)
    assert reservoir.selected_target.shape == (2, 3)
    assert reservoir.W.shape == (2, 3)
    assert reservoir.M.shape == (2, 3)
    assert reservoir.valid_mask.shape == (2, 3)
    assert torch.equal(reservoir.M[reservoir.valid_mask], torch.full((6,), 4, dtype=torch.long))
