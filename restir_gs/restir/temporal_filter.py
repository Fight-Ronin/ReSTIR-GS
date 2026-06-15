from __future__ import annotations

import torch

from restir_gs.render.gbuffer import GBuffer
from restir_gs.restir.initial import LightingEstimatorBuffers
from restir_gs.restir.temporal import TemporalLookup
from restir_gs.restir.types import RestirRenderSettings, TemporalFilterStats


def empty_temporal_lookup(gbuffer: GBuffer) -> TemporalLookup:
    height, width = gbuffer.depth.shape
    return TemporalLookup(
        prev_pixels=torch.zeros((height, width, 2), dtype=torch.long, device=gbuffer.rgb.device),
        valid_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        pre_gate_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        normal_pass_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        rgb_pass_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        motion_pass_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        relative_depth_error=torch.full(
            (height, width),
            float("inf"),
            dtype=gbuffer.rgb.dtype,
            device=gbuffer.rgb.device,
        ),
        normal_dot=torch.zeros((height, width), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
        normal_abs_dot=torch.zeros((height, width), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
        rgb_distance=torch.full(
            (height, width),
            float("inf"),
            dtype=gbuffer.rgb.dtype,
            device=gbuffer.rgb.device,
        ),
        motion_pixels=torch.zeros((height, width, 2), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
    )


def empty_temporal_filter_stats(gbuffer: GBuffer) -> TemporalFilterStats:
    height, width = gbuffer.depth.shape
    zeros = torch.zeros((height, width), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device)
    return TemporalFilterStats(
        confidence=zeros,
        alpha=zeros,
        history_delta=zeros,
        clamp_delta=zeros,
    )


def apply_confidence_clamped_temporal_filter(
    gbuffer: GBuffer,
    current: LightingEstimatorBuffers,
    previous_filtered: LightingEstimatorBuffers,
    lookup: TemporalLookup,
    settings: RestirRenderSettings,
) -> tuple[LightingEstimatorBuffers, TemporalFilterStats]:
    _check_lookup_shapes_for_filter(lookup, gbuffer)
    device = gbuffer.rgb.device
    dtype = gbuffer.rgb.dtype

    history = _gather_previous_rgb(previous_filtered.contribution_rgb.to(device=device, dtype=dtype), lookup)
    current_contribution = current.contribution_rgb.to(device=device, dtype=dtype)
    confidence = _temporal_filter_confidence(lookup, settings, dtype=dtype, device=device)
    alpha = lookup.valid_mask.to(device=device, dtype=dtype) * float(settings.temporal_filter_blend_max) * confidence

    current_mean = current_contribution.abs().mean(dim=-1)
    clamp_radius = float(settings.temporal_filter_clamp_scale) * current_mean + float(settings.temporal_filter_clamp_min)
    history_clamped = torch.minimum(
        torch.maximum(history, current_contribution - clamp_radius[..., None]),
        current_contribution + clamp_radius[..., None],
    )
    filtered = current_contribution * (1.0 - alpha[..., None]) + history_clamped * alpha[..., None]
    filtered = torch.where(current.valid_mask.to(device=device)[..., None], filtered, current_contribution)

    composite_lit = gbuffer.rgb.to(device=device, dtype=dtype) * float(settings.ambient) + filtered
    composite = torch.where(
        current.valid_mask.to(device=device)[..., None],
        composite_lit,
        current.composite_rgb.to(device=device, dtype=dtype),
    )
    alpha_zero = alpha <= 0.0
    filtered = torch.where(alpha_zero[..., None], current_contribution, filtered)
    composite = torch.where(alpha_zero[..., None], current.composite_rgb.to(device=device, dtype=dtype), composite)

    stats = TemporalFilterStats(
        confidence=torch.where(lookup.valid_mask.to(device=device), confidence, torch.zeros_like(confidence)),
        alpha=alpha,
        history_delta=torch.mean(torch.abs(history - current_contribution), dim=-1),
        clamp_delta=torch.mean(torch.abs(history - history_clamped), dim=-1),
    )
    return LightingEstimatorBuffers(contribution_rgb=filtered, composite_rgb=composite, valid_mask=current.valid_mask), stats


def _check_lookup_shapes_for_filter(lookup: TemporalLookup, gbuffer: GBuffer) -> None:
    image_shape = gbuffer.depth.shape
    if lookup.prev_pixels.shape != (*image_shape, 2):
        raise ValueError(f"Expected prev_pixels shape [H,W,2], got {tuple(lookup.prev_pixels.shape)}")
    if lookup.valid_mask.shape != image_shape:
        raise ValueError(f"Expected lookup valid mask shape {tuple(image_shape)}, got {tuple(lookup.valid_mask.shape)}")
    if lookup.pre_gate_mask.shape != image_shape:
        raise ValueError(f"Expected lookup pre-gate mask shape {tuple(image_shape)}, got {tuple(lookup.pre_gate_mask.shape)}")


def _gather_previous_rgb(values: torch.Tensor, lookup: TemporalLookup) -> torch.Tensor:
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"Expected previous filtered contribution shape [H,W,3], got {tuple(values.shape)}")
    prev_height, prev_width, _ = values.shape
    prev_x = lookup.prev_pixels[..., 0].to(device=values.device, dtype=torch.long).clamp(0, max(prev_width - 1, 0))
    prev_y = lookup.prev_pixels[..., 1].to(device=values.device, dtype=torch.long).clamp(0, max(prev_height - 1, 0))
    gathered = values.reshape(-1, 3)[(prev_y * prev_width + prev_x).reshape(-1)].reshape(*lookup.valid_mask.shape, 3)
    return torch.where(lookup.valid_mask.to(values.device)[..., None], gathered, torch.zeros_like(gathered))


def _temporal_filter_confidence(
    lookup: TemporalLookup,
    settings: RestirRenderSettings,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    depth_error = lookup.relative_depth_error.to(device=device, dtype=dtype)
    if settings.depth_tolerance == 0.0:
        depth_conf = (depth_error <= 0.0).to(dtype=dtype)
    else:
        depth_conf = (1.0 - depth_error / float(settings.depth_tolerance)).clamp(0.0, 1.0)

    if settings.temporal_normal_threshold is None:
        normal_conf = torch.ones_like(depth_conf)
    else:
        normal_conf = lookup.normal_abs_dot.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    if settings.temporal_rgb_threshold is None:
        rgb_conf = torch.ones_like(depth_conf)
    elif settings.temporal_rgb_threshold == 0.0:
        rgb_conf = (lookup.rgb_distance.to(device=device, dtype=dtype) <= 0.0).to(dtype=dtype)
    else:
        rgb_conf = (1.0 - lookup.rgb_distance.to(device=device, dtype=dtype) / float(settings.temporal_rgb_threshold)).clamp(
            0.0,
            1.0,
        )

    motion_magnitude = torch.linalg.norm(lookup.motion_pixels.to(device=device, dtype=dtype), dim=-1)
    if settings.temporal_max_motion_pixels is None:
        motion_conf = torch.ones_like(depth_conf)
    elif settings.temporal_max_motion_pixels == 0.0:
        motion_conf = (motion_magnitude <= 0.0).to(dtype=dtype)
    else:
        motion_conf = (1.0 - motion_magnitude / float(settings.temporal_max_motion_pixels)).clamp(0.0, 1.0)

    confidence = torch.minimum(torch.minimum(depth_conf, normal_conf), torch.minimum(rgb_conf, motion_conf))
    return torch.where(lookup.valid_mask.to(device=device), confidence, torch.zeros_like(confidence))
