from __future__ import annotations

from dataclasses import dataclass

import torch

from restir_gs.lighting.deferred import PointLights, evaluate_selected_light_diffuse
from restir_gs.render.gbuffer import GBuffer
from restir_gs.restir.proposal import CandidateSamples


@dataclass(frozen=True)
class EstimatorBuffers:
    diffuse_rgb: torch.Tensor
    composite_rgb: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class ReservoirState:
    light_indices: torch.Tensor
    target_values: torch.Tensor
    weight_sum: torch.Tensor
    selected_target: torch.Tensor
    W: torch.Tensor
    M: torch.Tensor
    valid_mask: torch.Tensor


def sample_uniform_light_candidates(
    height: int,
    width: int,
    candidate_count: int,
    light_count: int,
    seed: int = 2028,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Sample uniform light indices with replacement."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Expected positive image size, got {height}x{width}")
    if candidate_count <= 0:
        raise ValueError(f"Expected positive candidate count, got {candidate_count}")
    if light_count <= 0:
        raise ValueError(f"Expected positive light count, got {light_count}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    candidates = torch.randint(
        low=0,
        high=light_count,
        size=(height, width, candidate_count),
        generator=generator,
        dtype=torch.long,
    )
    return candidates.to(device)


def estimate_uniform_diffuse(
    gbuffer: GBuffer,
    lights: PointLights,
    candidates: torch.Tensor,
    ambient: float = 0.2,
) -> EstimatorBuffers:
    """Estimate all-light diffuse with uniform sampled candidates."""
    if candidates.ndim != 3:
        raise ValueError(f"Expected candidates shape [H,W,K], got {tuple(candidates.shape)}")
    candidate_count = candidates.shape[-1]
    light_count = lights.positions_cam.shape[0]
    diffuse_candidates = evaluate_selected_light_diffuse(gbuffer, lights, candidates)
    diffuse_rgb = diffuse_candidates.sum(dim=2) * (float(light_count) / float(candidate_count))
    valid_mask = gbuffer.valid_mask & gbuffer.normal_mask
    diffuse_rgb = torch.where(valid_mask[..., None], diffuse_rgb, torch.zeros_like(diffuse_rgb))
    composite_rgb = _compose_estimate(gbuffer, diffuse_rgb, valid_mask, ambient)
    return EstimatorBuffers(diffuse_rgb=diffuse_rgb, composite_rgb=composite_rgb, valid_mask=valid_mask)


def estimate_proposal_diffuse(
    gbuffer: GBuffer,
    lights: PointLights,
    samples: CandidateSamples,
    ambient: float = 0.2,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
) -> EstimatorBuffers:
    """Estimate all-light diffuse from samples drawn from an arbitrary proposal."""
    _check_sample_shapes(samples)
    diffuse_candidates = evaluate_selected_light_diffuse(
        gbuffer,
        lights,
        samples.light_indices,
        two_sided=two_sided,
        distance_epsilon=distance_epsilon,
    )
    proposal_probs = samples.proposal_probs.to(device=gbuffer.rgb.device, dtype=gbuffer.rgb.dtype)
    weighted = diffuse_candidates / proposal_probs.clamp_min(torch.finfo(gbuffer.rgb.dtype).tiny)[..., None]
    diffuse_rgb = weighted.mean(dim=2)
    valid_mask = gbuffer.valid_mask & gbuffer.normal_mask
    diffuse_rgb = torch.where(valid_mask[..., None], diffuse_rgb, torch.zeros_like(diffuse_rgb))
    composite_rgb = _compose_estimate(gbuffer, diffuse_rgb, valid_mask, ambient)
    return EstimatorBuffers(diffuse_rgb=diffuse_rgb, composite_rgb=composite_rgb, valid_mask=valid_mask)


def estimate_ris_initial_diffuse(
    gbuffer: GBuffer,
    lights: PointLights,
    candidates: torch.Tensor,
    selection_seed: int = 2029,
    ambient: float = 0.2,
    proposal_probs: torch.Tensor | None = None,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
) -> tuple[EstimatorBuffers, ReservoirState]:
    """Estimate diffuse with an initial weighted reservoir over light candidates."""
    if candidates.ndim != 3:
        raise ValueError(f"Expected candidates shape [H,W,K], got {tuple(candidates.shape)}")
    if proposal_probs is not None and proposal_probs.shape != candidates.shape:
        raise ValueError(f"Expected proposal probs shape {tuple(candidates.shape)}, got {tuple(proposal_probs.shape)}")

    height, width, candidate_count = candidates.shape
    light_count = lights.positions_cam.shape[0]
    diffuse_candidates = evaluate_selected_light_diffuse(
        gbuffer,
        lights,
        candidates,
        two_sided=two_sided,
        distance_epsilon=distance_epsilon,
    )
    target_values = _luminance(diffuse_candidates).clamp_min(0.0)
    if proposal_probs is None:
        proposal_q = torch.full_like(target_values, 1.0 / float(light_count))
    else:
        proposal_q = proposal_probs.to(device=gbuffer.rgb.device, dtype=gbuffer.rgb.dtype)
    weights = target_values / proposal_q.clamp_min(torch.finfo(gbuffer.rgb.dtype).tiny)
    weight_sum = weights.sum(dim=-1)
    base_valid = gbuffer.valid_mask & gbuffer.normal_mask
    valid_mask = base_valid & (weight_sum > 0.0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(selection_seed)
    thresholds = torch.rand((height, width), generator=generator, dtype=gbuffer.rgb.dtype).to(gbuffer.rgb.device)
    thresholds = thresholds * weight_sum
    cumulative = weights.cumsum(dim=-1)
    selected_slots = torch.sum(cumulative < thresholds[..., None], dim=-1).clamp_max(candidate_count - 1)
    selected_slots = torch.where(valid_mask, selected_slots, torch.zeros_like(selected_slots))

    selected_index_gather = selected_slots[..., None]
    selected_light_indices = torch.gather(candidates.to(gbuffer.rgb.device), dim=-1, index=selected_index_gather).squeeze(-1)
    selected_target = torch.gather(target_values, dim=-1, index=selected_index_gather).squeeze(-1)
    selected_diffuse = torch.gather(
        diffuse_candidates,
        dim=2,
        index=selected_slots[..., None, None].expand(height, width, 1, 3),
    ).squeeze(2)

    W = torch.zeros_like(weight_sum)
    positive = valid_mask & (selected_target > 0.0)
    W[positive] = weight_sum[positive] / (float(candidate_count) * selected_target[positive])
    diffuse_rgb = torch.where(positive[..., None], selected_diffuse * W[..., None], torch.zeros_like(selected_diffuse))
    composite_rgb = _compose_estimate(gbuffer, diffuse_rgb, positive, ambient)
    M = torch.where(
        base_valid,
        torch.full_like(selected_light_indices, candidate_count, dtype=torch.long),
        torch.zeros_like(selected_light_indices, dtype=torch.long),
    )

    reservoir = ReservoirState(
        light_indices=selected_light_indices,
        target_values=target_values,
        weight_sum=weight_sum,
        selected_target=selected_target,
        W=W,
        M=M,
        valid_mask=positive,
    )
    buffers = EstimatorBuffers(diffuse_rgb=diffuse_rgb, composite_rgb=composite_rgb, valid_mask=positive)
    return buffers, reservoir


def _check_sample_shapes(samples: CandidateSamples) -> None:
    if samples.light_indices.ndim != 3:
        raise ValueError(f"Expected light indices shape [H,W,K], got {tuple(samples.light_indices.shape)}")
    if samples.proposal_probs.shape != samples.light_indices.shape:
        raise ValueError(
            f"Expected proposal probs shape {tuple(samples.light_indices.shape)}, got {tuple(samples.proposal_probs.shape)}"
        )


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device)
    return torch.sum(rgb * weights, dim=-1)


def _compose_estimate(
    gbuffer: GBuffer,
    diffuse_rgb: torch.Tensor,
    valid_mask: torch.Tensor,
    ambient: float,
) -> torch.Tensor:
    composite_lit = gbuffer.rgb * float(ambient) + diffuse_rgb
    return torch.where(valid_mask[..., None], composite_lit, gbuffer.rgb)
