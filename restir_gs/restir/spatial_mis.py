from __future__ import annotations

from dataclasses import dataclass

import torch

from restir_gs.lighting.deferred import PointLights, evaluate_selected_light_diffuse
from restir_gs.render.gbuffer import GBuffer
from restir_gs.restir.initial import EstimatorBuffers
from restir_gs.restir.proposal import CandidateSamples


@dataclass(frozen=True)
class SpatialMISCandidates:
    light_indices: torch.Tensor
    source_weights: torch.Tensor
    mixture_probs: torch.Tensor
    candidate_mixture_probs: torch.Tensor
    source_valid_mask: torch.Tensor


@dataclass(frozen=True)
class SpatialMISStats:
    accepted_neighbor_count: torch.Tensor
    reuse_mask: torch.Tensor
    center_weight: torch.Tensor
    neighbor_weight_sum: torch.Tensor


def build_spatial_mis_candidates(
    gbuffer: GBuffer,
    proposal_probs: torch.Tensor,
    center_samples: CandidateSamples,
    radius: int = 1,
    center_floor: float = 0.75,
    normal_threshold: float = 0.8,
    depth_tolerance: float = 0.05,
    rgb_threshold: float | None = None,
    normal_penalty: float = 8.0,
    depth_penalty: float = 25.0,
    rgb_penalty: float = 0.0,
) -> tuple[SpatialMISCandidates, SpatialMISStats]:
    """Build current-pixel defensive mixture-proposal candidates from 3x3 sources."""
    _check_inputs(gbuffer, proposal_probs, center_samples, radius, center_floor, normal_threshold, depth_tolerance, rgb_threshold)

    height, width, light_count = proposal_probs.shape
    device = gbuffer.rgb.device
    dtype = gbuffer.rgb.dtype
    proposal_probs = proposal_probs.to(device=device, dtype=dtype)
    base_valid = gbuffer.valid_mask & gbuffer.normal_mask
    offsets = [(0, 0)] + [
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dy == 0 and dx == 0)
    ]

    source_indices: list[torch.Tensor] = []
    source_proposals: list[torch.Tensor] = []
    source_valids: list[torch.Tensor] = []
    source_raw_weights: list[torch.Tensor] = []
    accepted_neighbor_count = torch.zeros((height, width), dtype=torch.long, device=device)

    for dy, dx in offsets:
        shifted_indices = _shift_tensor(center_samples.light_indices.to(device=device, dtype=torch.long), dy, dx)
        shifted_proposal = _shift_tensor(proposal_probs, dy, dx)
        if dy == 0 and dx == 0:
            valid = base_valid
            raw_weight = torch.ones((height, width), dtype=dtype, device=device)
        else:
            valid, normal_dot, relative_depth, rgb_distance = _neighbor_acceptance_data(
                gbuffer,
                dy=dy,
                dx=dx,
                normal_threshold=normal_threshold,
                depth_tolerance=depth_tolerance,
                rgb_threshold=rgb_threshold,
                base_valid=base_valid,
            )
            normal_term = (1.0 - normal_dot).clamp_min(0.0) * float(normal_penalty)
            depth_term = relative_depth * float(depth_penalty)
            rgb_term = rgb_distance * float(rgb_penalty)
            raw_weight = torch.exp(-(normal_term + depth_term + rgb_term))
            accepted_neighbor_count += valid.to(dtype=torch.long)

        source_indices.append(torch.where(valid[..., None], shifted_indices, torch.zeros_like(shifted_indices)))
        source_proposals.append(shifted_proposal)
        source_valids.append(valid)
        source_raw_weights.append(torch.where(valid, raw_weight, torch.zeros_like(raw_weight)))

    source_valid = torch.stack(source_valids, dim=-1)
    raw_weights = torch.stack(source_raw_weights, dim=-1)
    center_valid = source_valid[..., 0]
    neighbor_raw_sum = raw_weights[..., 1:].sum(dim=-1)
    source_weights = torch.zeros_like(raw_weights)
    no_neighbors = neighbor_raw_sum <= 0.0
    center_weight = torch.where(center_valid & no_neighbors, torch.ones_like(neighbor_raw_sum), torch.zeros_like(neighbor_raw_sum))
    center_weight = torch.where(center_valid & ~no_neighbors, torch.full_like(center_weight, float(center_floor)), center_weight)
    source_weights[..., 0] = center_weight
    neighbor_scale = torch.where(
        center_valid & ~no_neighbors,
        (1.0 - float(center_floor)) / neighbor_raw_sum.clamp_min(torch.finfo(dtype).tiny),
        torch.zeros_like(neighbor_raw_sum),
    )
    source_weights[..., 1:] = raw_weights[..., 1:] * neighbor_scale[..., None]

    source_proposal_tensor = torch.stack(source_proposals, dim=-2)
    mixture_probs = torch.sum(source_weights[..., None] * source_proposal_tensor, dim=-2)
    mixture_probs = torch.where(center_valid[..., None], mixture_probs, torch.zeros_like(mixture_probs))
    if light_count <= 0:
        raise ValueError("Expected positive light count.")

    light_indices = torch.stack(source_indices, dim=2)
    flat_indices = light_indices.reshape(height, width, -1)
    candidate_mixture_probs = torch.gather(mixture_probs, dim=-1, index=flat_indices).reshape_as(light_indices)
    neighbor_weight_sum = source_weights[..., 1:].sum(dim=-1)
    stats = SpatialMISStats(
        accepted_neighbor_count=accepted_neighbor_count,
        reuse_mask=(neighbor_weight_sum > 0.0) & center_valid,
        center_weight=source_weights[..., 0],
        neighbor_weight_sum=neighbor_weight_sum,
    )
    candidates = SpatialMISCandidates(
        light_indices=light_indices,
        source_weights=source_weights,
        mixture_probs=mixture_probs,
        candidate_mixture_probs=candidate_mixture_probs,
        source_valid_mask=source_valid,
    )
    return candidates, stats


def estimate_spatial_mis_diffuse(
    gbuffer: GBuffer,
    lights: PointLights,
    proposal_probs: torch.Tensor,
    center_samples: CandidateSamples,
    ambient: float = 0.2,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
    **candidate_kwargs: float | int | None,
) -> tuple[EstimatorBuffers, SpatialMISStats]:
    """Estimate diffuse from center and neighbor proposal samples using defensive MIS."""
    candidates, stats = build_spatial_mis_candidates(gbuffer, proposal_probs, center_samples, **candidate_kwargs)
    flat_indices = _flatten_candidate_indices(candidates.light_indices)
    diffuse_candidates = evaluate_selected_light_diffuse(
        gbuffer,
        lights,
        flat_indices,
        two_sided=two_sided,
        distance_epsilon=distance_epsilon,
    ).reshape(*candidates.light_indices.shape, 3)
    candidate_count = candidates.light_indices.shape[-1]
    safe_q = candidates.candidate_mixture_probs.clamp_min(torch.finfo(gbuffer.rgb.dtype).tiny)
    weighted = diffuse_candidates * (candidates.source_weights[..., None] / float(candidate_count))[..., None] / safe_q[..., None]
    diffuse_rgb = weighted.sum(dim=(2, 3))
    valid_mask = gbuffer.valid_mask & gbuffer.normal_mask
    diffuse_rgb = torch.where(valid_mask[..., None], diffuse_rgb, torch.zeros_like(diffuse_rgb))
    composite_rgb = _compose_estimate(gbuffer, diffuse_rgb, valid_mask, ambient)
    return EstimatorBuffers(diffuse_rgb=diffuse_rgb, composite_rgb=composite_rgb, valid_mask=valid_mask), stats


def _neighbor_acceptance_data(
    gbuffer: GBuffer,
    dy: int,
    dx: int,
    normal_threshold: float,
    depth_tolerance: float,
    rgb_threshold: float | None,
    base_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    in_bounds = _shift_in_bounds(gbuffer.valid_mask.shape, dy, dx, device=gbuffer.rgb.device)
    neighbor_valid = _shift_tensor(base_valid, dy, dx)
    neighbor_normal = _shift_tensor(gbuffer.normal_cam, dy, dx)
    neighbor_depth = _shift_tensor(gbuffer.depth, dy, dx)
    neighbor_rgb = _shift_tensor(gbuffer.rgb, dy, dx)

    normal_dot = torch.sum(gbuffer.normal_cam * neighbor_normal, dim=-1)
    rgb_distance = torch.mean((gbuffer.rgb - neighbor_rgb).abs(), dim=-1)
    current_depth = gbuffer.depth
    finite_depth = (
        torch.isfinite(current_depth)
        & torch.isfinite(neighbor_depth)
        & (current_depth > 0.0)
        & (neighbor_depth > 0.0)
    )
    relative_depth = (neighbor_depth - current_depth).abs() / current_depth.clamp_min(1e-8)
    valid = (
        base_valid
        & neighbor_valid
        & in_bounds
        & finite_depth
        & (normal_dot >= float(normal_threshold))
        & (relative_depth <= float(depth_tolerance))
    )
    if rgb_threshold is not None:
        valid = valid & (rgb_distance <= float(rgb_threshold))
    return valid, normal_dot, relative_depth, rgb_distance


def _check_inputs(
    gbuffer: GBuffer,
    proposal_probs: torch.Tensor,
    center_samples: CandidateSamples,
    radius: int,
    center_floor: float,
    normal_threshold: float,
    depth_tolerance: float,
    rgb_threshold: float | None,
) -> None:
    if radius != 1:
        raise ValueError(f"Phase 15 spatial MIS supports radius=1 only, got {radius}")
    if not 0.0 <= center_floor <= 1.0:
        raise ValueError(f"Expected center floor in [0,1], got {center_floor}")
    if not -1.0 <= normal_threshold <= 1.0:
        raise ValueError(f"Expected normal threshold in [-1,1], got {normal_threshold}")
    if depth_tolerance < 0.0:
        raise ValueError(f"Expected nonnegative depth tolerance, got {depth_tolerance}")
    if rgb_threshold is not None and rgb_threshold < 0.0:
        raise ValueError(f"Expected nonnegative RGB threshold, got {rgb_threshold}")
    if proposal_probs.ndim != 3:
        raise ValueError(f"Expected proposal probs shape [H,W,N], got {tuple(proposal_probs.shape)}")
    if proposal_probs.shape[:2] != gbuffer.valid_mask.shape:
        raise ValueError(f"Expected proposal image shape {tuple(gbuffer.valid_mask.shape)}, got {tuple(proposal_probs.shape[:2])}")
    if center_samples.light_indices.ndim != 3:
        raise ValueError(f"Expected center samples shape [H,W,K], got {tuple(center_samples.light_indices.shape)}")
    if center_samples.light_indices.shape[:2] != gbuffer.valid_mask.shape:
        raise ValueError(f"Expected center samples image shape {tuple(gbuffer.valid_mask.shape)}, got {tuple(center_samples.light_indices.shape[:2])}")


def _flatten_candidate_indices(light_indices: torch.Tensor) -> torch.Tensor:
    height, width = light_indices.shape[:2]
    return light_indices.reshape(height, width, -1)


def _shift_tensor(values: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    height, width = values.shape[:2]
    out = torch.zeros_like(values)
    if abs(dy) >= height or abs(dx) >= width:
        return out

    if dy >= 0:
        dst_y = slice(0, height - dy)
        src_y = slice(dy, height)
    else:
        dst_y = slice(-dy, height)
        src_y = slice(0, height + dy)
    if dx >= 0:
        dst_x = slice(0, width - dx)
        src_x = slice(dx, width)
    else:
        dst_x = slice(-dx, width)
        src_x = slice(0, width + dx)
    out[dst_y, dst_x] = values[src_y, src_x]
    return out


def _shift_in_bounds(shape: torch.Size | tuple[int, int], dy: int, dx: int, device: torch.device) -> torch.Tensor:
    height, width = int(shape[0]), int(shape[1])
    mask = torch.ones((height, width), dtype=torch.bool, device=device)
    return _shift_tensor(mask, dy, dx)


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
