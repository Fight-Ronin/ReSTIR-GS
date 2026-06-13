from __future__ import annotations

import math

import torch

from restir_gs.lighting.deferred import PointLights
from restir_gs.render.gbuffer import GBuffer
from restir_gs.restir.initial import estimate_proposal_diffuse, estimate_ris_initial_diffuse, estimate_uniform_diffuse
from restir_gs.restir.proposal import (
    CandidateSamples,
    compute_geometric_proposal_distribution,
    sample_light_candidates_from_distribution,
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


def make_simple_lights() -> PointLights:
    return PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
        colors=torch.ones((2, 3), dtype=torch.float32),
        intensities=torch.ones((2,), dtype=torch.float32),
    )


def test_geometric_proposal_is_finite_normalized_and_falls_back_to_uniform() -> None:
    rgb = torch.ones((1, 3, 3), dtype=torch.float32)
    position = torch.zeros((1, 3, 3), dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    valid = torch.tensor([[True, False, True]])
    normal_mask = torch.ones((1, 3), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, normal_mask)

    proposal = compute_geometric_proposal_distribution(gbuffer, make_simple_lights(), distance_epsilon=0.0)

    assert proposal.shape == (1, 3, 2)
    assert torch.isfinite(proposal).all()
    assert torch.allclose(proposal.sum(dim=-1), torch.ones((1, 3)))
    assert torch.allclose(proposal[0, 1], torch.tensor([0.5, 0.5]))
    assert torch.allclose(proposal[0, 2], torch.tensor([0.5, 0.5]))
    assert proposal[0, 0, 0] > proposal[0, 0, 1]


def test_geometric_proposal_rejects_negative_distance_epsilon() -> None:
    rgb = torch.ones((1, 1, 3), dtype=torch.float32)
    position = torch.zeros((1, 1, 3), dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)

    try:
        compute_geometric_proposal_distribution(gbuffer, make_simple_lights(), distance_epsilon=-1e-4)
    except ValueError as exc:
        assert "distance_epsilon" in str(exc)
    else:
        raise AssertionError("Expected negative distance_epsilon to fail.")


def test_candidate_sampling_is_deterministic_and_gathers_probs() -> None:
    proposal = torch.tensor([[[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]], dtype=torch.float32)

    a = sample_light_candidates_from_distribution(proposal, candidate_count=5, seed=123, device="cpu")
    b = sample_light_candidates_from_distribution(proposal, candidate_count=5, seed=123, device="cpu")
    gathered = torch.gather(proposal, dim=-1, index=a.light_indices)

    assert torch.equal(a.light_indices, b.light_indices)
    assert torch.allclose(a.proposal_probs, b.proposal_probs)
    assert torch.allclose(a.proposal_probs, gathered)


def test_candidate_sampling_rejects_negative_probabilities() -> None:
    proposal = torch.tensor([[[0.5, -0.1, 0.6]]], dtype=torch.float32)

    try:
        sample_light_candidates_from_distribution(proposal, candidate_count=1, seed=123, device="cpu")
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative proposal probabilities to fail.")
    if torch.cuda.is_available():
        try:
            sample_light_candidates_from_distribution(proposal.to("cuda"), candidate_count=1, seed=123, device="cuda")
        except ValueError as exc:
            assert "non-negative" in str(exc)
        else:
            raise AssertionError("Expected negative CUDA proposal probabilities to fail.")


def test_cuda_candidate_sampling_is_deterministic_and_gathers_probs() -> None:
    if not torch.cuda.is_available():
        return
    proposal = torch.tensor([[[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]], dtype=torch.float32, device="cuda")

    a = sample_light_candidates_from_distribution(proposal, candidate_count=5, seed=123, device="cuda")
    b = sample_light_candidates_from_distribution(proposal, candidate_count=5, seed=123, device="cuda")
    gathered = torch.gather(proposal, dim=-1, index=a.light_indices)

    assert a.light_indices.device.type == "cuda"
    assert a.proposal_probs.device.type == "cuda"
    assert torch.equal(a.light_indices, b.light_indices)
    assert torch.allclose(a.proposal_probs, b.proposal_probs)
    assert torch.allclose(a.proposal_probs, gathered)


def test_proposal_mc_estimator_matches_closed_form() -> None:
    rgb = torch.ones((1, 1, 3), dtype=torch.float32)
    position = torch.zeros((1, 1, 3), dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)
    samples = CandidateSamples(
        light_indices=torch.tensor([[[0, 1]]], dtype=torch.long),
        proposal_probs=torch.tensor([[[0.25, 0.75]]], dtype=torch.float32),
    )

    buffers = estimate_proposal_diffuse(gbuffer, make_simple_lights(), samples, distance_epsilon=0.0)

    f0 = torch.ones(3) / math.pi
    f1 = torch.ones(3) * 0.25 / math.pi
    expected = ((f0 / 0.25) + (f1 / 0.75)) * 0.5
    assert torch.allclose(buffers.diffuse_rgb[0, 0], expected)


def test_proposal_ris_k1_matches_proposal_mc_for_positive_target() -> None:
    rgb = torch.ones((1, 1, 3), dtype=torch.float32)
    position = torch.zeros((1, 1, 3), dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)
    samples = CandidateSamples(
        light_indices=torch.tensor([[[1]]], dtype=torch.long),
        proposal_probs=torch.tensor([[[0.75]]], dtype=torch.float32),
    )

    mc = estimate_proposal_diffuse(gbuffer, make_simple_lights(), samples, distance_epsilon=0.0)
    ris, reservoir = estimate_ris_initial_diffuse(
        gbuffer,
        make_simple_lights(),
        samples.light_indices,
        selection_seed=7,
        proposal_probs=samples.proposal_probs,
        distance_epsilon=0.0,
    )

    assert torch.allclose(ris.diffuse_rgb, mc.diffuse_rgb)
    assert torch.allclose(ris.composite_rgb, mc.composite_rgb)
    assert torch.equal(reservoir.M, torch.tensor([[1]]))


def test_uniform_estimator_matches_uniform_proposal_mc() -> None:
    rgb = torch.ones((1, 1, 3), dtype=torch.float32)
    position = torch.zeros((1, 1, 3), dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)
    candidates = torch.tensor([[[0, 1]]], dtype=torch.long)
    samples = CandidateSamples(
        light_indices=candidates,
        proposal_probs=torch.full((1, 1, 2), 0.5, dtype=torch.float32),
    )

    uniform = estimate_uniform_diffuse(gbuffer, make_simple_lights(), candidates)
    proposal_mc = estimate_proposal_diffuse(gbuffer, make_simple_lights(), samples)

    assert torch.allclose(uniform.diffuse_rgb, proposal_mc.diffuse_rgb)
    assert torch.allclose(uniform.composite_rgb, proposal_mc.composite_rgb)
