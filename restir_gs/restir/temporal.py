from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from restir_gs.lighting.deferred import PointLights
from restir_gs.render.gbuffer import GBuffer
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.restir.initial import (
    LightingEstimatorBuffers,
    LightingTargetMode,
    ReservoirState,
    evaluate_selected_light_contribution,
)


@dataclass(frozen=True)
class TemporalLookup:
    prev_pixels: torch.Tensor
    valid_mask: torch.Tensor
    pre_gate_mask: torch.Tensor
    normal_pass_mask: torch.Tensor
    rgb_pass_mask: torch.Tensor
    motion_pass_mask: torch.Tensor
    relative_depth_error: torch.Tensor
    normal_dot: torch.Tensor
    normal_abs_dot: torch.Tensor
    rgb_distance: torch.Tensor
    motion_pixels: torch.Tensor


@dataclass(frozen=True)
class TemporalReservoirState:
    light_indices: torch.Tensor
    selected_target: torch.Tensor
    weight_sum: torch.Tensor
    W: torch.Tensor
    M: torch.Tensor
    valid_mask: torch.Tensor


def temporal_reservoir_from_initial(reservoir: ReservoirState) -> TemporalReservoirState:
    return TemporalReservoirState(
        light_indices=reservoir.light_indices,
        selected_target=reservoir.selected_target,
        weight_sum=reservoir.weight_sum,
        W=reservoir.W,
        M=reservoir.M,
        valid_mask=reservoir.valid_mask,
    )


def reproject_current_to_previous(
    current_gbuffer: GBuffer,
    current_camera: PinholeCamera,
    prev_gbuffer: GBuffer,
    prev_camera: PinholeCamera,
    depth_tolerance: float = 0.05,
    normal_threshold: float | None = 0.85,
    rgb_threshold: float | None = 0.20,
    max_motion_pixels: float | None = 32.0,
    search_radius: int = 1,
) -> TemporalLookup:
    """Reproject current valid positions into the previous frame with local nearest-neighbor repair."""
    _check_temporal_gate_settings(depth_tolerance, normal_threshold, rgb_threshold, max_motion_pixels, search_radius)
    _check_camera(current_camera, "current_camera")
    _check_camera(prev_camera, "prev_camera")

    height, width = current_gbuffer.depth.shape
    prev_height, prev_width = prev_gbuffer.depth.shape
    device = current_gbuffer.rgb.device
    dtype = current_gbuffer.rgb.dtype

    current_valid = (current_gbuffer.valid_mask & current_gbuffer.normal_mask).to(device=device)
    positions = current_gbuffer.position_cam.to(device=device, dtype=dtype)
    ones = torch.ones((*positions.shape[:2], 1), dtype=dtype, device=device)
    current_h = torch.cat((positions, ones), dim=-1)

    current_inv = torch.linalg.inv(current_camera.viewmats[0].to(device=device, dtype=dtype))
    prev_view = prev_camera.viewmats[0].to(device=device, dtype=dtype)
    world_h = torch.einsum("ij,hwj->hwi", current_inv, current_h)
    prev_h = torch.einsum("ij,hwj->hwi", prev_view, world_h)
    prev_z = prev_h[..., 2]

    intrinsics = prev_camera.intrinsics[0].to(device=device, dtype=dtype)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    positive_z = prev_z > 0.0
    u = (prev_h[..., 0] * fx / prev_z.clamp_min(torch.finfo(dtype).tiny)) + cx
    v = (prev_h[..., 1] * fy / prev_z.clamp_min(torch.finfo(dtype).tiny)) + cy
    prev_x = torch.round(u).to(torch.long)
    prev_y = torch.round(v).to(torch.long)

    offsets = _candidate_offsets(search_radius, device=device)
    cand_x = prev_x[None] + offsets[:, 0, None, None]
    cand_y = prev_y[None] + offsets[:, 1, None, None]
    cand_in_bounds = (cand_x >= 0) & (cand_x < prev_width) & (cand_y >= 0) & (cand_y < prev_height)
    safe_x = cand_x.clamp(0, max(prev_width - 1, 0))
    safe_y = cand_y.clamp(0, max(prev_height - 1, 0))
    flat = safe_y * prev_width + safe_x

    flat_indices = flat.reshape(-1)
    candidate_count = offsets.shape[0]
    prev_depth_candidates = (
        prev_gbuffer.depth.to(device=device, dtype=dtype).reshape(-1)[flat_indices].reshape(candidate_count, height, width)
    )
    prev_rgb_candidates = (
        prev_gbuffer.rgb.to(device=device, dtype=dtype).reshape(-1, 3)[flat_indices].reshape(candidate_count, height, width, 3)
    )
    prev_normal_candidates = (
        prev_gbuffer.normal_cam.to(device=device, dtype=dtype).reshape(-1, 3)[flat_indices].reshape(candidate_count, height, width, 3)
    )
    prev_valid_flat = (prev_gbuffer.valid_mask & prev_gbuffer.normal_mask).to(device=device).reshape(-1)
    prev_valid_candidates = prev_valid_flat[flat_indices].reshape(candidate_count, height, width)
    relative_depth_candidates = torch.abs(prev_depth_candidates - prev_z[None]) / prev_z.abs().clamp_min(torch.finfo(dtype).eps)[None]
    relative_depth_candidates = torch.where(
        positive_z[None] & cand_in_bounds,
        relative_depth_candidates,
        torch.full_like(relative_depth_candidates, float("inf")),
    )

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    raw_motion = torch.stack((u - xs, v - ys), dim=-1)
    motion_magnitude = torch.linalg.norm(raw_motion, dim=-1)

    pre_gate_candidates = (
        current_valid[None]
        & positive_z[None]
        & cand_in_bounds
        & prev_valid_candidates
        & (relative_depth_candidates <= float(depth_tolerance))
    )
    normal_dot_candidates = _world_normal_dot(current_gbuffer, current_camera, prev_normal_candidates, prev_camera, dtype=dtype, device=device)
    normal_abs_dot_candidates = normal_dot_candidates.abs()
    rgb_distance_candidates = torch.mean(
        torch.abs(current_gbuffer.rgb.to(device=device, dtype=dtype)[None] - prev_rgb_candidates),
        dim=-1,
    )

    if normal_threshold is not None:
        normal_pass_raw = normal_abs_dot_candidates >= float(normal_threshold)
    else:
        normal_pass_raw = torch.ones_like(pre_gate_candidates)
    rgb_pass_raw = torch.ones_like(pre_gate_candidates)
    if rgb_threshold is not None:
        rgb_pass_raw = rgb_distance_candidates <= float(rgb_threshold)
    motion_pass_raw = torch.ones_like(pre_gate_candidates)
    if max_motion_pixels is not None:
        motion_pass_raw = (motion_magnitude <= float(max_motion_pixels))[None].expand_as(pre_gate_candidates)

    normal_pass_candidates = pre_gate_candidates & normal_pass_raw
    rgb_pass_candidates = pre_gate_candidates & rgb_pass_raw
    motion_pass_candidates = pre_gate_candidates & motion_pass_raw
    valid_candidates = pre_gate_candidates & normal_pass_raw & rgb_pass_raw & motion_pass_raw

    selected = _select_best_temporal_candidate(
        valid_candidates,
        pre_gate_candidates,
        relative_depth_candidates,
        normal_abs_dot_candidates,
        rgb_distance_candidates,
        motion_magnitude,
        offsets,
        depth_tolerance,
        rgb_threshold,
        max_motion_pixels,
    )
    pre_gate = _gather_candidate(pre_gate_candidates, selected)
    valid = _gather_candidate(valid_candidates, selected)
    relative_depth_error = _gather_candidate(relative_depth_candidates, selected)
    normal_dot = _gather_candidate(normal_dot_candidates, selected)
    normal_abs_dot = _gather_candidate(normal_abs_dot_candidates, selected)
    rgb_distance = _gather_candidate(rgb_distance_candidates, selected)
    normal_pass = _gather_candidate(normal_pass_candidates, selected)
    rgb_pass = _gather_candidate(rgb_pass_candidates, selected)
    motion_pass = _gather_candidate(motion_pass_candidates, selected)
    chosen_x = _gather_candidate(safe_x, selected)
    chosen_y = _gather_candidate(safe_y, selected)

    motion = torch.where(pre_gate[..., None], raw_motion, torch.zeros_like(raw_motion))
    prev_pixels = torch.stack((chosen_x, chosen_y), dim=-1)
    prev_pixels = torch.where(valid[..., None], prev_pixels, torch.zeros_like(prev_pixels))
    relative_depth_error = torch.where(pre_gate, relative_depth_error, torch.full_like(relative_depth_error, float("inf")))
    normal_dot = torch.where(pre_gate, normal_dot, torch.zeros_like(normal_dot))
    normal_abs_dot = torch.where(pre_gate, normal_abs_dot, torch.zeros_like(normal_abs_dot))
    rgb_distance = torch.where(pre_gate, rgb_distance, torch.full_like(rgb_distance, float("inf")))

    return TemporalLookup(
        prev_pixels=prev_pixels,
        valid_mask=valid,
        pre_gate_mask=pre_gate,
        normal_pass_mask=normal_pass,
        rgb_pass_mask=rgb_pass,
        motion_pass_mask=motion_pass,
        relative_depth_error=relative_depth_error,
        normal_dot=normal_dot,
        normal_abs_dot=normal_abs_dot,
        rgb_distance=rgb_distance,
        motion_pixels=motion,
    )


def combine_temporal_reservoirs(
    current_gbuffer: GBuffer,
    lights: PointLights,
    current_buffers: LightingEstimatorBuffers,
    current_reservoir: ReservoirState | TemporalReservoirState,
    prev_reservoir: TemporalReservoirState,
    lookup: TemporalLookup,
    selection_seed: int,
    ambient: float = 0.2,
    target_mode: LightingTargetMode = "diffuse",
    contribution_evaluator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    history_m_cap: int | None = None,
) -> tuple[LightingEstimatorBuffers, TemporalReservoirState]:
    """Combine current initial reservoir with a reprojected previous reservoir."""
    if history_m_cap is not None and history_m_cap <= 0:
        raise ValueError(f"Expected positive history_m_cap or None, got {history_m_cap}")
    _check_lookup_shapes(lookup, current_gbuffer)
    height, width = current_gbuffer.depth.shape
    device = current_gbuffer.rgb.device
    dtype = current_gbuffer.rgb.dtype

    current_valid = current_reservoir.valid_mask.to(device=device) & current_gbuffer.valid_mask & current_gbuffer.normal_mask
    prev_light = _gather_prev(prev_reservoir.light_indices.to(device=device), lookup, fill=0).to(torch.long)
    prev_w = _gather_prev(prev_reservoir.W.to(device=device, dtype=dtype), lookup, fill=0.0)
    prev_m = _gather_prev(prev_reservoir.M.to(device=device), lookup, fill=0).to(torch.long)
    prev_valid = _gather_prev(prev_reservoir.valid_mask.to(device=device), lookup, fill=False).to(torch.bool)
    history_valid = lookup.valid_mask.to(device=device) & prev_valid & (prev_m > 0) & current_gbuffer.valid_mask & current_gbuffer.normal_mask

    candidate_indices = torch.stack(
        (
            current_reservoir.light_indices.to(device=device, dtype=torch.long),
            prev_light,
        ),
        dim=-1,
    )
    if contribution_evaluator is None:
        contribution_candidates = evaluate_selected_light_contribution(
            current_gbuffer,
            lights,
            candidate_indices,
            target_mode=target_mode,
        )
    else:
        contribution_candidates = contribution_evaluator(candidate_indices)
    target_values = _luminance(contribution_candidates).clamp_min(0.0)
    candidate_w = torch.stack((current_reservoir.W.to(device=device, dtype=dtype), prev_w), dim=-1)
    current_m = current_reservoir.M.to(device=device)
    prev_m_eff = prev_m if history_m_cap is None else prev_m.clamp_max(int(history_m_cap))
    candidate_m = torch.stack((current_m, prev_m_eff), dim=-1)
    candidate_valid = torch.stack((current_valid, history_valid), dim=-1)

    candidate_weights = target_values * candidate_w * candidate_m.to(dtype=dtype)
    candidate_weights = torch.where(candidate_valid, candidate_weights, torch.zeros_like(candidate_weights))
    weight_sum = candidate_weights.sum(dim=-1)
    combined_m = torch.where(candidate_valid, candidate_m, torch.zeros_like(candidate_m)).sum(dim=-1)
    combined_valid = (current_gbuffer.valid_mask & current_gbuffer.normal_mask) & (weight_sum > 0.0) & (combined_m > 0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(selection_seed)
    thresholds = torch.rand((height, width), generator=generator, dtype=dtype).to(device) * weight_sum
    cumulative = candidate_weights.cumsum(dim=-1)
    selected_slots = torch.sum(cumulative < thresholds[..., None], dim=-1).clamp_max(1)
    selected_slots = torch.where(combined_valid, selected_slots, torch.zeros_like(selected_slots))

    selected_light = torch.gather(candidate_indices, dim=-1, index=selected_slots[..., None]).squeeze(-1)
    selected_target = torch.gather(target_values, dim=-1, index=selected_slots[..., None]).squeeze(-1)
    selected_contribution = torch.gather(
        contribution_candidates,
        dim=2,
        index=selected_slots[..., None, None].expand(height, width, 1, 3),
    ).squeeze(2)

    W = torch.zeros_like(weight_sum)
    positive = combined_valid & (selected_target > 0.0)
    W[positive] = weight_sum[positive] / (combined_m[positive].to(dtype=dtype) * selected_target[positive])
    contribution_rgb = torch.where(
        positive[..., None],
        selected_contribution * W[..., None],
        torch.zeros_like(selected_contribution),
    )
    composite_lit = current_gbuffer.rgb * float(ambient) + contribution_rgb
    composite_rgb = torch.where(positive[..., None], composite_lit, current_gbuffer.rgb)

    no_history = ~history_valid
    contribution_rgb = torch.where(no_history[..., None], current_buffers.contribution_rgb, contribution_rgb)
    composite_rgb = torch.where(no_history[..., None], current_buffers.composite_rgb, composite_rgb)
    selected_light = torch.where(no_history, current_reservoir.light_indices.to(device=device), selected_light)
    selected_target = torch.where(no_history, current_reservoir.selected_target.to(device=device, dtype=dtype), selected_target)
    weight_sum = torch.where(no_history, current_reservoir.weight_sum.to(device=device, dtype=dtype), weight_sum)
    W = torch.where(no_history, current_reservoir.W.to(device=device, dtype=dtype), W)
    combined_m = torch.where(no_history, current_reservoir.M.to(device=device), combined_m)
    positive = torch.where(no_history, current_reservoir.valid_mask.to(device=device), positive)

    buffers = LightingEstimatorBuffers(
        contribution_rgb=contribution_rgb,
        composite_rgb=composite_rgb,
        valid_mask=positive,
    )
    reservoir = TemporalReservoirState(
        light_indices=selected_light,
        selected_target=selected_target,
        weight_sum=weight_sum,
        W=W,
        M=combined_m.to(torch.long),
        valid_mask=positive,
    )
    return buffers, reservoir


def _gather_prev(values: torch.Tensor, lookup: TemporalLookup, fill: float | int | bool) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"Expected previous reservoir field shape [H,W], got {tuple(values.shape)}")
    prev_height, prev_width = values.shape
    prev_x = lookup.prev_pixels[..., 0].to(device=values.device, dtype=torch.long).clamp(0, max(prev_width - 1, 0))
    prev_y = lookup.prev_pixels[..., 1].to(device=values.device, dtype=torch.long).clamp(0, max(prev_height - 1, 0))
    gathered = values.reshape(-1)[(prev_y * prev_width + prev_x).reshape(-1)].reshape(lookup.valid_mask.shape)
    fill_tensor = torch.full_like(gathered, fill)
    return torch.where(lookup.valid_mask.to(values.device), gathered, fill_tensor)


def _candidate_offsets(search_radius: int, device: torch.device) -> torch.Tensor:
    offsets = [(0, 0)]
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0:
                continue
            offsets.append((dx, dy))
    return torch.tensor(offsets, dtype=torch.long, device=device)


def _gather_candidate(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    if values.ndim < 3:
        raise ValueError(f"Expected candidate tensor shape [K,H,W,...], got {tuple(values.shape)}")
    gather_index = selected[None]
    while gather_index.ndim < values.ndim:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(1, *values.shape[1:])
    return torch.gather(values, dim=0, index=gather_index).squeeze(0)


def _select_best_temporal_candidate(
    valid_candidates: torch.Tensor,
    pre_gate_candidates: torch.Tensor,
    relative_depth: torch.Tensor,
    normal_abs_dot: torch.Tensor,
    rgb_distance: torch.Tensor,
    motion_magnitude: torch.Tensor,
    offsets: torch.Tensor,
    depth_tolerance: float,
    rgb_threshold: float | None,
    max_motion_pixels: float | None,
) -> torch.Tensor:
    depth_term = _normalized_term(relative_depth, depth_tolerance)
    normal_term = 1.0 - normal_abs_dot.clamp(0.0, 1.0)
    rgb_term = _normalized_term(rgb_distance, rgb_threshold)
    motion_term = _normalized_term(motion_magnitude, max_motion_pixels)[None].expand_as(depth_term)
    offset_penalty = (offsets[:, 0].float().square() + offsets[:, 1].float().square())[:, None, None] * 1e-4
    score = depth_term + normal_term + rgb_term + motion_term + offset_penalty.to(device=depth_term.device, dtype=depth_term.dtype)
    inf_score = torch.full_like(score, float("inf"))
    valid_score = torch.where(valid_candidates, score, inf_score)
    pre_gate_score = torch.where(pre_gate_candidates, score, inf_score)
    has_valid = valid_candidates.any(dim=0)
    selected_valid = torch.argmin(valid_score, dim=0)
    selected_pre_gate = torch.argmin(pre_gate_score, dim=0)
    return torch.where(has_valid, selected_valid, selected_pre_gate)


def _normalized_term(values: torch.Tensor, threshold: float | None) -> torch.Tensor:
    if threshold is None:
        return torch.zeros_like(values)
    if threshold == 0.0:
        return torch.where(values <= 0.0, torch.zeros_like(values), torch.full_like(values, float("inf")))
    return values / float(threshold)


def _check_camera(camera: PinholeCamera, name: str) -> None:
    if camera.viewmats.shape != (1, 4, 4):
        raise ValueError(f"Expected {name}.viewmats shape [1,4,4], got {tuple(camera.viewmats.shape)}")
    if camera.intrinsics.shape != (1, 3, 3):
        raise ValueError(f"Expected {name}.intrinsics shape [1,3,3], got {tuple(camera.intrinsics.shape)}")


def _check_lookup_shapes(lookup: TemporalLookup, gbuffer: GBuffer) -> None:
    image_shape = gbuffer.depth.shape
    if lookup.prev_pixels.shape != (*image_shape, 2):
        raise ValueError(f"Expected prev_pixels shape [H,W,2], got {tuple(lookup.prev_pixels.shape)}")
    if lookup.valid_mask.shape != image_shape:
        raise ValueError(f"Expected lookup valid mask shape {tuple(image_shape)}, got {tuple(lookup.valid_mask.shape)}")
    if lookup.pre_gate_mask.shape != image_shape:
        raise ValueError(f"Expected lookup pre-gate mask shape {tuple(image_shape)}, got {tuple(lookup.pre_gate_mask.shape)}")
    if lookup.normal_pass_mask.shape != image_shape:
        raise ValueError(f"Expected normal pass mask shape {tuple(image_shape)}, got {tuple(lookup.normal_pass_mask.shape)}")
    if lookup.rgb_pass_mask.shape != image_shape:
        raise ValueError(f"Expected RGB pass mask shape {tuple(image_shape)}, got {tuple(lookup.rgb_pass_mask.shape)}")
    if lookup.motion_pass_mask.shape != image_shape:
        raise ValueError(f"Expected motion pass mask shape {tuple(image_shape)}, got {tuple(lookup.motion_pass_mask.shape)}")
    if lookup.relative_depth_error.shape != image_shape:
        raise ValueError("Expected relative depth error shape to match current image shape.")
    if lookup.normal_dot.shape != image_shape:
        raise ValueError("Expected normal dot shape to match current image shape.")
    if lookup.normal_abs_dot.shape != image_shape:
        raise ValueError("Expected normal abs dot shape to match current image shape.")
    if lookup.rgb_distance.shape != image_shape:
        raise ValueError("Expected RGB distance shape to match current image shape.")
    if lookup.motion_pixels.shape != (*image_shape, 2):
        raise ValueError(f"Expected motion_pixels shape [H,W,2], got {tuple(lookup.motion_pixels.shape)}")


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device)
    return torch.sum(rgb * weights, dim=-1)


def _world_normal_dot(
    current_gbuffer: GBuffer,
    current_camera: PinholeCamera,
    prev_normal_cam: torch.Tensor,
    prev_camera: PinholeCamera,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    current_inv = torch.linalg.inv(current_camera.viewmats[0].to(device=device, dtype=dtype))
    prev_inv = torch.linalg.inv(prev_camera.viewmats[0].to(device=device, dtype=dtype))
    current_normal = current_gbuffer.normal_cam.to(device=device, dtype=dtype)
    current_world = torch.einsum("ij,hwj->hwi", current_inv[:3, :3], current_normal)
    current_world = torch.nn.functional.normalize(current_world, dim=-1, eps=torch.finfo(dtype).eps)
    if prev_normal_cam.ndim == 3:
        prev_world = torch.einsum("ij,hwj->hwi", prev_inv[:3, :3], prev_normal_cam)
        current_for_dot = current_world
    elif prev_normal_cam.ndim == 4:
        prev_world = torch.einsum("ij,khwj->khwi", prev_inv[:3, :3], prev_normal_cam)
        current_for_dot = current_world[None]
    else:
        raise ValueError(f"Expected previous normal shape [H,W,3] or [K,H,W,3], got {tuple(prev_normal_cam.shape)}")
    prev_world = torch.nn.functional.normalize(prev_world, dim=-1, eps=torch.finfo(dtype).eps)
    return torch.sum(current_for_dot * prev_world, dim=-1).clamp(-1.0, 1.0)


def _check_temporal_gate_settings(
    depth_tolerance: float,
    normal_threshold: float | None,
    rgb_threshold: float | None,
    max_motion_pixels: float | None,
    search_radius: int,
) -> None:
    if depth_tolerance < 0.0:
        raise ValueError(f"Expected non-negative depth_tolerance, got {depth_tolerance}")
    if normal_threshold is not None and not -1.0 <= normal_threshold <= 1.0:
        raise ValueError(f"Expected normal_threshold in [-1,1] or None, got {normal_threshold}")
    if rgb_threshold is not None and rgb_threshold < 0.0:
        raise ValueError(f"Expected non-negative rgb_threshold or None, got {rgb_threshold}")
    if max_motion_pixels is not None and max_motion_pixels < 0.0:
        raise ValueError(f"Expected non-negative max_motion_pixels or None, got {max_motion_pixels}")
    if search_radius < 0:
        raise ValueError(f"Expected non-negative search_radius, got {search_radius}")
