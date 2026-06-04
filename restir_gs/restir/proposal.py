from __future__ import annotations

from dataclasses import dataclass

import torch

from restir_gs.lighting.deferred import PointLights
from restir_gs.render.gbuffer import GBuffer


@dataclass(frozen=True)
class CandidateSamples:
    light_indices: torch.Tensor
    proposal_probs: torch.Tensor


def compute_geometric_proposal_distribution(
    gbuffer: GBuffer,
    lights: PointLights,
    two_sided: bool = True,
    distance_epsilon: float = 1e-4,
) -> torch.Tensor:
    """Compute per-pixel geometric proposal probabilities over all lights."""
    if lights.positions_cam.ndim != 2 or lights.positions_cam.shape[-1] != 3:
        raise ValueError(f"Expected light positions shape [N,3], got {tuple(lights.positions_cam.shape)}")
    if lights.colors.shape != lights.positions_cam.shape:
        raise ValueError(f"Expected light colors shape {tuple(lights.positions_cam.shape)}, got {tuple(lights.colors.shape)}")
    if lights.intensities.shape != (lights.positions_cam.shape[0],):
        raise ValueError(
            f"Expected light intensities shape [{lights.positions_cam.shape[0]}], got {tuple(lights.intensities.shape)}"
        )

    dtype = gbuffer.rgb.dtype
    device = gbuffer.rgb.device
    positions_cam = lights.positions_cam.to(device=device, dtype=dtype)
    colors = lights.colors.to(device=device, dtype=dtype)
    intensities = lights.intensities.to(device=device, dtype=dtype)

    light_vec = positions_cam[None, None, :, :] - gbuffer.position_cam[..., None, :]
    dist2_raw = torch.sum(light_vec * light_vec, dim=-1)
    dist2 = dist2_raw + distance_epsilon
    wi = light_vec * torch.rsqrt(dist2_raw.clamp_min(distance_epsilon)[..., None])
    cos_theta = torch.sum(gbuffer.normal_cam[..., None, :] * wi, dim=-1)
    if two_sided:
        cos_theta = cos_theta.abs()
    else:
        cos_theta = cos_theta.clamp_min(0.0)

    light_power = intensities * _luminance(colors)
    weights = light_power[None, None, :] * cos_theta / dist2
    valid = (gbuffer.valid_mask & gbuffer.normal_mask)[..., None]
    weights = torch.where(valid, weights.clamp_min(0.0), torch.zeros_like(weights))

    weight_sum = weights.sum(dim=-1, keepdim=True)
    light_count = lights.positions_cam.shape[0]
    uniform = torch.full_like(weights, 1.0 / float(light_count))
    normalized = weights / weight_sum.clamp_min(torch.finfo(dtype).tiny)
    use_uniform = weight_sum <= 0.0
    return torch.where(use_uniform, uniform, normalized)


def sample_light_candidates_from_distribution(
    proposal_probs: torch.Tensor,
    candidate_count: int,
    seed: int = 5100,
    device: torch.device | str = "cuda",
) -> CandidateSamples:
    """Sample light indices with replacement from a per-pixel proposal distribution."""
    if proposal_probs.ndim != 3:
        raise ValueError(f"Expected proposal probs shape [H,W,N], got {tuple(proposal_probs.shape)}")
    if candidate_count <= 0:
        raise ValueError(f"Expected positive candidate count, got {candidate_count}")
    if not bool(torch.isfinite(proposal_probs).all()):
        raise ValueError("Proposal probabilities must be finite.")

    height, width, light_count = proposal_probs.shape
    flat_probs = proposal_probs.detach().cpu().reshape(-1, light_count)
    row_sums = flat_probs.sum(dim=-1)
    if not bool(torch.all(row_sums > 0.0)):
        raise ValueError("Each proposal distribution row must have positive probability mass.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    flat_indices = torch.multinomial(flat_probs, candidate_count, replacement=True, generator=generator)

    target_device = torch.device(device)
    light_indices = flat_indices.reshape(height, width, candidate_count).to(target_device)
    probs = proposal_probs.to(device=target_device)
    proposal_selected = torch.gather(probs, dim=-1, index=light_indices)
    return CandidateSamples(light_indices=light_indices, proposal_probs=proposal_selected)


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device)
    return torch.sum(rgb * weights, dim=-1)
