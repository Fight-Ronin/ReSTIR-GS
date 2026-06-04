from __future__ import annotations

import math

import torch

from restir_gs.eval.spatial_mis_ablation import SpatialMISVariant, default_spatial_mis_variants, run_spatial_mis_ablation
from restir_gs.lighting.deferred import PointLights, shade_deferred_lambertian
from restir_gs.render.gbuffer import GBuffer
from restir_gs.restir.initial import estimate_proposal_diffuse
from restir_gs.restir.proposal import CandidateSamples
from restir_gs.restir.spatial_mis import (
    build_spatial_mis_candidates,
    estimate_spatial_mis_diffuse,
)


def make_gbuffer(valid: torch.Tensor | None = None) -> GBuffer:
    height, width = 1, 2
    if valid is None:
        valid = torch.ones((height, width), dtype=torch.bool)
    return GBuffer(
        rgb=torch.ones((height, width, 3), dtype=torch.float32),
        depth=torch.ones((height, width), dtype=torch.float32),
        alpha=torch.ones((height, width), dtype=torch.float32),
        position_cam=torch.zeros((height, width, 3), dtype=torch.float32),
        normal_cam=torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
        valid_mask=valid,
        normal_mask=valid.clone(),
    )


def make_lights() -> PointLights:
    return PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        colors=torch.ones((2, 3), dtype=torch.float32),
        intensities=torch.ones((2,), dtype=torch.float32),
    )


def make_source_proposals() -> tuple[torch.Tensor, CandidateSamples]:
    proposal = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    indices = torch.tensor([[[0], [1]]], dtype=torch.long)
    samples = CandidateSamples(light_indices=indices, proposal_probs=torch.ones_like(indices, dtype=torch.float32))
    return proposal, samples


def test_center_floor_one_matches_center_proposal_mc() -> None:
    gbuffer = make_gbuffer()
    lights = make_lights()
    proposal, samples = make_source_proposals()

    expected = estimate_proposal_diffuse(gbuffer, lights, samples, distance_epsilon=0.0)
    actual, stats = estimate_spatial_mis_diffuse(
        gbuffer,
        lights,
        proposal,
        samples,
        center_floor=1.0,
        distance_epsilon=0.0,
    )

    assert torch.allclose(actual.diffuse_rgb, expected.diffuse_rgb)
    assert torch.allclose(stats.center_weight, torch.ones((1, 2)))
    assert torch.allclose(stats.neighbor_weight_sum, torch.zeros((1, 2)))


def test_source_weights_sum_to_one_and_center_floor_is_honored() -> None:
    gbuffer = make_gbuffer()
    proposal, samples = make_source_proposals()

    candidates, stats = build_spatial_mis_candidates(gbuffer, proposal, samples, center_floor=0.75)

    assert torch.allclose(candidates.source_weights.sum(dim=-1), torch.ones((1, 2)))
    assert torch.allclose(stats.center_weight, torch.full((1, 2), 0.75))
    assert torch.allclose(stats.neighbor_weight_sum, torch.full((1, 2), 0.25))
    assert torch.equal(stats.accepted_neighbor_count, torch.ones((1, 2), dtype=torch.long))


def test_spatial_mis_mc_matches_manual_two_source_mixture() -> None:
    gbuffer = make_gbuffer()
    lights = make_lights()
    proposal, samples = make_source_proposals()

    buffers, _ = estimate_spatial_mis_diffuse(
        gbuffer,
        lights,
        proposal,
        samples,
        center_floor=0.5,
        normal_penalty=0.0,
        depth_penalty=0.0,
        rgb_penalty=0.0,
        distance_epsilon=0.0,
    )

    expected = torch.full((1, 2, 3), 2.0 / math.pi)
    assert torch.allclose(buffers.diffuse_rgb, expected)


def test_invalid_neighbor_falls_back_to_center_proposal() -> None:
    gbuffer = make_gbuffer(valid=torch.tensor([[True, False]]))
    proposal, samples = make_source_proposals()

    candidates, stats = build_spatial_mis_candidates(gbuffer, proposal, samples, center_floor=0.5)

    assert torch.allclose(stats.center_weight[0, 0], torch.tensor(1.0))
    assert torch.allclose(stats.neighbor_weight_sum[0, 0], torch.tensor(0.0))
    assert torch.equal(stats.accepted_neighbor_count, torch.zeros((1, 2), dtype=torch.long))
    assert torch.allclose(candidates.mixture_probs[0, 0], torch.tensor([1.0, 0.0]))


def test_spatial_mis_ablation_rows_are_finite() -> None:
    gbuffer = make_gbuffer()
    lights = make_lights()
    proposal, samples = make_source_proposals()
    reference = shade_deferred_lambertian(gbuffer, lights, distance_epsilon=0.0)
    initial, _ = estimate_spatial_mis_diffuse(gbuffer, lights, proposal, samples, center_floor=1.0, distance_epsilon=0.0)
    variant = SpatialMISVariant("tiny", center_floor=0.5, normal_penalty=0.0, depth_penalty=0.0)

    result = run_spatial_mis_ablation(
        gbuffer,
        lights,
        reference,
        initial,
        proposal,
        samples,
        variants=[variant],
    )

    assert len(result.rows) == 1
    assert all(math.isfinite(float(row["mae"])) for row in result.rows)
    assert len(default_spatial_mis_variants()) == 6
