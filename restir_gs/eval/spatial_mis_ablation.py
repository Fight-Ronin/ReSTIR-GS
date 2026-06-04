from __future__ import annotations

from dataclasses import dataclass

import torch

from restir_gs.eval.ris_ablation import compute_error_metrics
from restir_gs.lighting.deferred import LightingBuffers, PointLights
from restir_gs.render.gbuffer import GBuffer
from restir_gs.restir.initial import EstimatorBuffers
from restir_gs.restir.proposal import CandidateSamples
from restir_gs.restir.spatial_mis import SpatialMISStats, estimate_spatial_mis_diffuse


@dataclass(frozen=True)
class SpatialMISVariant:
    name: str
    center_floor: float
    normal_threshold: float = 0.8
    depth_tolerance: float = 0.05
    rgb_threshold: float | None = None
    normal_penalty: float = 8.0
    depth_penalty: float = 25.0
    rgb_penalty: float = 0.0


@dataclass(frozen=True)
class SpatialMISAblationResult:
    rows: list[dict[str, int | float | str | None]]
    best_row: dict[str, int | float | str | None]
    best_buffers: EstimatorBuffers
    best_stats: SpatialMISStats


def default_spatial_mis_variants() -> list[SpatialMISVariant]:
    """Return fixed defensive MIS spatial variants."""
    return [
        SpatialMISVariant("geometry_floor_0_50", center_floor=0.50),
        SpatialMISVariant("geometry_floor_0_75", center_floor=0.75),
        SpatialMISVariant("geometry_floor_0_90", center_floor=0.90),
        SpatialMISVariant("rgb_floor_0_50", center_floor=0.50, rgb_penalty=8.0),
        SpatialMISVariant("rgb_floor_0_75", center_floor=0.75, rgb_penalty=8.0),
        SpatialMISVariant("rgb_floor_0_90", center_floor=0.90, rgb_penalty=8.0),
    ]


def run_spatial_mis_ablation(
    gbuffer: GBuffer,
    lights: PointLights,
    reference: LightingBuffers,
    initial_buffers: EstimatorBuffers,
    proposal_probs: torch.Tensor,
    center_samples: CandidateSamples,
    variants: list[SpatialMISVariant] | None = None,
    ambient: float = 0.2,
) -> SpatialMISAblationResult:
    """Evaluate defensive spatial MIS MC variants against the all-lights reference."""
    selected_variants = default_spatial_mis_variants() if variants is None else variants
    if not selected_variants:
        raise ValueError("Expected at least one spatial MIS variant.")

    rows: list[dict[str, int | float | str | None]] = []
    best_row: dict[str, int | float | str | None] | None = None
    best_buffers: EstimatorBuffers | None = None
    best_stats: SpatialMISStats | None = None

    for variant in selected_variants:
        kwargs = _variant_kwargs(variant)
        mc_buffers, mc_stats = estimate_spatial_mis_diffuse(
            gbuffer,
            lights,
            proposal_probs,
            center_samples,
            ambient=ambient,
            **kwargs,
        )
        mc_row = make_spatial_mis_row(
            variant=variant,
            estimate=mc_buffers,
            reference=reference,
            initial_buffers=initial_buffers,
            stats=mc_stats,
        )
        rows.append(mc_row)
        best_row, best_buffers, best_stats = _maybe_update_best(best_row, best_buffers, best_stats, mc_row, mc_buffers, mc_stats)

    if best_row is None or best_buffers is None or best_stats is None:
        raise RuntimeError("Spatial MIS ablation produced no rows.")
    return SpatialMISAblationResult(rows=rows, best_row=best_row, best_buffers=best_buffers, best_stats=best_stats)


def make_spatial_mis_row(
    variant: SpatialMISVariant,
    estimate: EstimatorBuffers,
    reference: LightingBuffers,
    initial_buffers: EstimatorBuffers,
    stats: SpatialMISStats,
) -> dict[str, int | float | str | None]:
    valid = reference.valid_mask
    metrics = compute_error_metrics(estimate.diffuse_rgb, reference.diffuse_rgb, valid)
    initial_error = _mean_abs_rgb_error(initial_buffers.diffuse_rgb, reference.diffuse_rgb)
    estimate_error = _mean_abs_rgb_error(estimate.diffuse_rgb, reference.diffuse_rgb)
    valid_count = max(_masked_sum(valid.to(dtype=torch.float32), valid), 1.0)
    improve = valid & (estimate_error < initial_error)
    harm = valid & (estimate_error > initial_error)
    row: dict[str, int | float | str | None] = {
        "variant": variant.name,
        "center_floor": variant.center_floor,
        "normal_threshold": variant.normal_threshold,
        "depth_tolerance": variant.depth_tolerance,
        "rgb_threshold": variant.rgb_threshold,
        "normal_penalty": variant.normal_penalty,
        "depth_penalty": variant.depth_penalty,
        "rgb_penalty": variant.rgb_penalty,
        "reuse_fraction": _masked_sum(stats.reuse_mask.to(dtype=torch.float32), valid) / valid_count,
        "accepted_neighbor_count_mean": _masked_mean(stats.accepted_neighbor_count.to(dtype=torch.float32), valid),
        "center_weight_mean": _masked_mean(stats.center_weight, valid),
        "neighbor_weight_mean": _masked_mean(stats.neighbor_weight_sum, valid),
        "error_delta_mean": _masked_mean(estimate_error - initial_error, valid),
        "improve_fraction": _masked_sum(improve.to(dtype=torch.float32), valid) / valid_count,
        "harm_fraction": _masked_sum(harm.to(dtype=torch.float32), valid) / valid_count,
    }
    row.update(metrics)
    return row


def _variant_kwargs(variant: SpatialMISVariant) -> dict[str, float | None]:
    return {
        "center_floor": variant.center_floor,
        "normal_threshold": variant.normal_threshold,
        "depth_tolerance": variant.depth_tolerance,
        "rgb_threshold": variant.rgb_threshold,
        "normal_penalty": variant.normal_penalty,
        "depth_penalty": variant.depth_penalty,
        "rgb_penalty": variant.rgb_penalty,
    }


def _maybe_update_best(
    best_row: dict[str, int | float | str | None] | None,
    best_buffers: EstimatorBuffers | None,
    best_stats: SpatialMISStats | None,
    row: dict[str, int | float | str | None],
    buffers: EstimatorBuffers,
    stats: SpatialMISStats,
) -> tuple[dict[str, int | float | str | None], EstimatorBuffers, SpatialMISStats]:
    if best_row is None or float(row["mae"]) < float(best_row["mae"]):
        return row, buffers, stats
    if best_buffers is None or best_stats is None:
        raise RuntimeError("Best spatial MIS state became inconsistent.")
    return best_row, best_buffers, best_stats


def _mean_abs_rgb_error(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (estimate - reference).abs().mean(dim=-1)


def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid = valid_mask.to(device=values.device, dtype=torch.bool)
    if not bool(valid.any()):
        return 0.0
    return float(values[valid].mean().detach().cpu())


def _masked_sum(values: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid = valid_mask.to(device=values.device, dtype=torch.bool)
    if not bool(valid.any()):
        return 0.0
    return float(values[valid].sum().detach().cpu())
