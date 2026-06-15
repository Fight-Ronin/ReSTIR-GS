from __future__ import annotations

import math

import torch

from restir_gs.metrics import compute_rgb_error_metrics
from restir_gs.restir.initial import ReservoirState
from restir_gs.restir.temporal import TemporalReservoirState
from restir_gs.restir.types import RESTIR_TIMING_FIELDS, RestirFrameResult, RestirRenderSettings


def make_restir_metric_rows(
    asset_id: str,
    result: RestirFrameResult,
    settings: RestirRenderSettings,
    shadow_bundle_asset_gpu_ms: float | None = None,
) -> list[dict[str, int | float | str]]:
    valid_mask = result.reference.valid_mask
    valid_pixels = int(valid_mask.sum().detach().cpu())
    pre_gate_pixels = int(result.lookup.pre_gate_mask.sum().detach().cpu())
    pre_gate_fraction = pre_gate_pixels / float(max(valid_pixels, 1))
    reuse_pixels = int(result.lookup.valid_mask.sum().detach().cpu())
    reuse_fraction = reuse_pixels / float(max(valid_pixels, 1))
    normal_gate_pixels = int(result.lookup.normal_pass_mask.sum().detach().cpu())
    rgb_gate_pixels = int(result.lookup.rgb_pass_mask.sum().detach().cpu())
    motion_gate_pixels = int(result.lookup.motion_pass_mask.sum().detach().cpu())
    normal_gate_fraction = normal_gate_pixels / float(max(valid_pixels, 1))
    rgb_gate_fraction = rgb_gate_pixels / float(max(valid_pixels, 1))
    motion_gate_fraction = motion_gate_pixels / float(max(valid_pixels, 1))
    normal_gate_pre_gate_fraction = normal_gate_pixels / float(max(pre_gate_pixels, 1))
    rgb_gate_pre_gate_fraction = rgb_gate_pixels / float(max(pre_gate_pixels, 1))
    motion_gate_pre_gate_fraction = motion_gate_pixels / float(max(pre_gate_pixels, 1))
    mean_depth_error = masked_mean(result.lookup.relative_depth_error, result.lookup.valid_mask)
    mean_normal_dot = masked_mean(result.lookup.normal_dot, result.lookup.valid_mask)
    mean_normal_abs_dot = masked_mean(result.lookup.normal_abs_dot, result.lookup.valid_mask)
    mean_rgb_distance = masked_mean(result.lookup.rgb_distance, result.lookup.valid_mask)
    motion_magnitude = torch.linalg.norm(result.lookup.motion_pixels, dim=-1)
    mean_motion = masked_mean(motion_magnitude, result.lookup.valid_mask)
    mean_pre_gate_normal_dot = masked_mean(result.lookup.normal_dot, result.lookup.pre_gate_mask)
    mean_pre_gate_normal_abs_dot = masked_mean(result.lookup.normal_abs_dot, result.lookup.pre_gate_mask)
    mean_pre_gate_rgb_distance = masked_mean(result.lookup.rgb_distance, result.lookup.pre_gate_mask)
    mean_pre_gate_motion = masked_mean(motion_magnitude, result.lookup.pre_gate_mask)
    filter_confidence_mean = masked_mean(result.temporal_filter_stats.confidence, result.lookup.valid_mask)
    filter_alpha_mean = masked_mean(result.temporal_filter_stats.alpha, valid_mask)
    filter_alpha_max = float(result.temporal_filter_stats.alpha.detach().cpu().max()) if result.temporal_filter_stats.alpha.numel() else 0.0
    filter_history_delta_mean = masked_mean(result.temporal_filter_stats.history_delta, result.lookup.valid_mask)
    filter_clamp_delta_mean = masked_mean(result.temporal_filter_stats.clamp_delta, result.lookup.valid_mask)
    timing_fields = result.timings.as_row_fields(shadow_bundle_asset_gpu_ms)

    rows: list[dict[str, int | float | str]] = []
    for estimator, buffers, reservoir in (
        ("initial_ris", result.initial, result.initial_reservoir),
        ("temporal_ris", result.temporal, result.temporal_reservoir),
        ("temporal_filtered_ris", result.temporal_filtered, result.temporal_reservoir),
    ):
        m_mean, m_max = reservoir_m_stats(reservoir)
        for quantity, estimate, reference in (
            ("contribution_rgb", buffers.contribution_rgb, result.reference.diffuse_rgb),
            ("composite_rgb", buffers.composite_rgb, result.reference.composite_rgb),
        ):
            row: dict[str, int | float | str] = {
                "asset_id": asset_id,
                "frame_index": result.frame_index,
                "estimator": estimator,
                "reference_quantity": quantity,
                "target_mode": settings.target_mode,
                "proposal": _effective_proposal(settings),
                "k": settings.candidate_count,
                "candidate_seed": settings.candidate_seed_base + result.frame_index,
                "selection_seed": (
                    settings.initial_selection_seed_base + result.frame_index
                    if estimator == "initial_ris"
                    else settings.temporal_selection_seed_base + result.frame_index
                ),
                "valid_pixels": valid_pixels,
                "pre_gate_pixels": pre_gate_pixels,
                "pre_gate_fraction": pre_gate_fraction,
                "normal_gate_pass_pixels": normal_gate_pixels,
                "normal_gate_pass_fraction": normal_gate_fraction,
                "normal_gate_pass_pre_gate_fraction": normal_gate_pre_gate_fraction,
                "rgb_gate_pass_pixels": rgb_gate_pixels,
                "rgb_gate_pass_fraction": rgb_gate_fraction,
                "rgb_gate_pass_pre_gate_fraction": rgb_gate_pre_gate_fraction,
                "motion_gate_pass_pixels": motion_gate_pixels,
                "motion_gate_pass_fraction": motion_gate_fraction,
                "motion_gate_pass_pre_gate_fraction": motion_gate_pre_gate_fraction,
                "reuse_pixels": reuse_pixels,
                "reuse_fraction": reuse_fraction,
                "mean_relative_depth_error": mean_depth_error,
                "mean_temporal_normal_dot": mean_normal_dot,
                "mean_temporal_normal_abs_dot": mean_normal_abs_dot,
                "mean_temporal_rgb_distance": mean_rgb_distance,
                "mean_motion_pixels": mean_motion,
                "mean_pre_gate_normal_dot": mean_pre_gate_normal_dot,
                "mean_pre_gate_normal_abs_dot": mean_pre_gate_normal_abs_dot,
                "mean_pre_gate_rgb_distance": mean_pre_gate_rgb_distance,
                "mean_pre_gate_motion_pixels": mean_pre_gate_motion,
                "temporal_normal_threshold": _format_optional_float(settings.temporal_normal_threshold),
                "temporal_rgb_threshold": _format_optional_float(settings.temporal_rgb_threshold),
                "temporal_max_motion_pixels": _format_optional_float(settings.temporal_max_motion_pixels),
                "temporal_reprojection_search_radius": int(settings.temporal_reprojection_search_radius),
                "temporal_history_m_cap": _effective_temporal_history_m_cap(settings),
                "temporal_filter_blend_max": float(settings.temporal_filter_blend_max),
                "temporal_filter_clamp_scale": float(settings.temporal_filter_clamp_scale),
                "temporal_filter_clamp_min": float(settings.temporal_filter_clamp_min),
                "temporal_filter_confidence_mean": filter_confidence_mean,
                "temporal_filter_alpha_mean": filter_alpha_mean,
                "temporal_filter_alpha_max": filter_alpha_max,
                "temporal_filter_history_delta_mean": filter_history_delta_mean,
                "temporal_filter_clamp_delta_mean": filter_clamp_delta_mean,
                "visibility_shadow_pcf_radius": int(settings.visibility_shadow_pcf_radius),
                "reservoir_m_mean": m_mean,
                "reservoir_m_max": m_max,
            }
            row.update(timing_fields)
            row.update(compute_rgb_error_metrics(estimate, reference, valid_mask))
            rows.append(row)
    return rows


def summarize_restir_rows(rows: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    groups: dict[tuple[str, str, str], list[dict[str, int | float | str]]] = {}
    for row in rows:
        key = (str(row["asset_id"]), str(row["estimator"]), str(row["reference_quantity"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, int | float | str]] = []
    for asset_id, estimator, quantity in sorted(groups):
        group = groups[(asset_id, estimator, quantity)]
        mae = [float(row["mae"]) for row in group]
        rmse = [float(row["rmse"]) for row in group]
        reuse = [float(row["reuse_fraction"]) for row in group]
        pre_gate = [float(row["pre_gate_fraction"]) for row in group]
        normal_gate = [float(row["normal_gate_pass_pre_gate_fraction"]) for row in group]
        rgb_gate = [float(row["rgb_gate_pass_pre_gate_fraction"]) for row in group]
        motion_gate = [float(row["motion_gate_pass_pre_gate_fraction"]) for row in group]
        summary.append(
            {
                "asset_id": asset_id,
                "estimator": estimator,
                "reference_quantity": quantity,
                "frame_count": len(group),
                "mae_mean": _mean(mae),
                "rmse_mean": _mean(rmse),
                "pre_gate_fraction_mean": _mean(pre_gate),
                "normal_gate_pass_pre_gate_fraction_mean": _mean(normal_gate),
                "rgb_gate_pass_pre_gate_fraction_mean": _mean(rgb_gate),
                "motion_gate_pass_pre_gate_fraction_mean": _mean(motion_gate),
                "reuse_fraction_mean": _mean(reuse),
            }
        )
    return summary


def summarize_restir_timing_rows(rows: list[dict[str, int | float | str]]) -> dict[str, dict[str, float | int]]:
    return _summarize_timing_group(rows)


def summarize_restir_asset_timing_rows(rows: list[dict[str, int | float | str]]) -> dict[str, dict[str, dict[str, float | int]]]:
    groups: dict[str, list[dict[str, int | float | str]]] = {}
    for row in rows:
        groups.setdefault(str(row["asset_id"]), []).append(row)
    return {asset_id: _summarize_timing_group(group) for asset_id, group in sorted(groups.items())}


def reservoir_m_stats(reservoir: ReservoirState | TemporalReservoirState) -> tuple[float, int]:
    valid = reservoir.valid_mask.detach().cpu().to(torch.bool)
    values = reservoir.M.detach().cpu()[valid]
    if values.numel() == 0:
        return 0.0, 0
    return float(values.float().mean()), int(values.max())


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    data = values.detach().cpu()
    valid = mask.detach().cpu().to(torch.bool) & torch.isfinite(data)
    if not bool(valid.any()):
        return 0.0
    return float(data[valid].float().mean())


def all_numeric_finite(rows: list[dict[str, int | float | str]]) -> bool:
    for row in rows:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _summarize_timing_group(rows: list[dict[str, int | float | str]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for field in RESTIR_TIMING_FIELDS:
        values = [float(row[field]) for row in rows if field in row and math.isfinite(float(row[field]))]
        summary[field] = {
            "mean": _mean(values),
            "max": max(values) if values else 0.0,
            "count": len(values),
        }
    return summary


def _format_optional_float(value: float | None) -> float | str:
    return "none" if value is None else float(value)


def _effective_temporal_history_m_cap(settings: RestirRenderSettings) -> int:
    return int(settings.candidate_count if settings.temporal_history_m_cap is None else settings.temporal_history_m_cap)


def _effective_proposal(settings: RestirRenderSettings) -> str:
    if settings.target_mode == "visibility":
        return "visibility_geometric"
    return "geometric"
