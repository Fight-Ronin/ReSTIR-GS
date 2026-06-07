from __future__ import annotations

from dataclasses import dataclass

import torch

from restir_gs.lighting.deferred import PointLights
from restir_gs.lighting.visibility import ShadowMapBundle, evaluate_selected_light_visible_diffuse
from restir_gs.render.gbuffer import GBuffer
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.restir.initial import LightingEstimatorBuffers, ReservoirState
from restir_gs.restir.proposal import CandidateSamples


@dataclass(frozen=True)
class VisibilityEstimatorBuffers:
    contribution_rgb: torch.Tensor
    composite_rgb: torch.Tensor
    valid_mask: torch.Tensor


def estimate_visibility_proposal_lighting(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    shadow_bundle: ShadowMapBundle,
    samples: CandidateSamples,
    ambient: float = 0.2,
    alpha_threshold: float = 1e-4,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
) -> VisibilityEstimatorBuffers:
    """Estimate visibility-aware direct diffuse from arbitrary light proposal samples."""
    _check_samples(samples)
    candidates = evaluate_selected_light_visible_diffuse(
        gbuffer,
        camera,
        lights,
        shadow_bundle,
        samples.light_indices,
        alpha_threshold=alpha_threshold,
        two_sided=two_sided,
        distance_epsilon=distance_epsilon,
    )
    proposal_probs = samples.proposal_probs.to(device=gbuffer.rgb.device, dtype=gbuffer.rgb.dtype)
    weighted = candidates / proposal_probs.clamp_min(torch.finfo(gbuffer.rgb.dtype).tiny)[..., None]
    contribution_rgb = weighted.mean(dim=2)
    valid_mask = gbuffer.valid_mask & gbuffer.normal_mask
    contribution_rgb = torch.where(valid_mask[..., None], contribution_rgb, torch.zeros_like(contribution_rgb))
    return VisibilityEstimatorBuffers(
        contribution_rgb=contribution_rgb,
        composite_rgb=_compose(gbuffer, contribution_rgb, valid_mask, ambient),
        valid_mask=valid_mask,
    )


def estimate_visibility_ris_initial_lighting(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    shadow_bundle: ShadowMapBundle,
    candidates: torch.Tensor,
    selection_seed: int = 2029,
    ambient: float = 0.2,
    proposal_probs: torch.Tensor | None = None,
    alpha_threshold: float = 1e-4,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
) -> tuple[LightingEstimatorBuffers, ReservoirState]:
    """Estimate visibility-aware direct diffuse with an initial weighted reservoir."""
    if candidates.ndim != 3:
        raise ValueError(f"Expected candidates shape [H,W,K], got {tuple(candidates.shape)}")
    if proposal_probs is not None and proposal_probs.shape != candidates.shape:
        raise ValueError(f"Expected proposal probs shape {tuple(candidates.shape)}, got {tuple(proposal_probs.shape)}")

    height, width, candidate_count = candidates.shape
    light_count = lights.positions_cam.shape[0]
    contribution_candidates = evaluate_selected_light_visible_diffuse(
        gbuffer,
        camera,
        lights,
        shadow_bundle,
        candidates,
        alpha_threshold=alpha_threshold,
        two_sided=two_sided,
        distance_epsilon=distance_epsilon,
    )
    target_values = _luminance(contribution_candidates).clamp_min(0.0)
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

    selected_index = selected_slots[..., None]
    selected_light_indices = torch.gather(candidates.to(gbuffer.rgb.device), dim=-1, index=selected_index).squeeze(-1)
    selected_target = torch.gather(target_values, dim=-1, index=selected_index).squeeze(-1)
    selected_contribution = torch.gather(
        contribution_candidates,
        dim=2,
        index=selected_slots[..., None, None].expand(height, width, 1, 3),
    ).squeeze(2)

    W = torch.zeros_like(weight_sum)
    positive = valid_mask & (selected_target > 0.0)
    W[positive] = weight_sum[positive] / (float(candidate_count) * selected_target[positive])
    contribution_rgb = torch.where(
        positive[..., None],
        selected_contribution * W[..., None],
        torch.zeros_like(selected_contribution),
    )
    composite_rgb = _compose(gbuffer, contribution_rgb, base_valid, ambient)
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
    buffers = LightingEstimatorBuffers(contribution_rgb=contribution_rgb, composite_rgb=composite_rgb, valid_mask=base_valid)
    return buffers, reservoir


def _check_samples(samples: CandidateSamples) -> None:
    if samples.light_indices.ndim != 3:
        raise ValueError(f"Expected light indices shape [H,W,K], got {tuple(samples.light_indices.shape)}")
    if samples.proposal_probs.shape != samples.light_indices.shape:
        raise ValueError(
            f"Expected proposal probs shape {tuple(samples.light_indices.shape)}, got {tuple(samples.proposal_probs.shape)}"
        )


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device)
    return torch.sum(rgb * weights, dim=-1)


def _compose(gbuffer: GBuffer, contribution_rgb: torch.Tensor, valid_mask: torch.Tensor, ambient: float) -> torch.Tensor:
    composite_lit = gbuffer.rgb * float(ambient) + contribution_rgb
    return torch.where(valid_mask[..., None], composite_lit, gbuffer.rgb)
