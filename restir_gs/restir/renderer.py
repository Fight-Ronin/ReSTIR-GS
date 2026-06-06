from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from restir_gs.lighting.asset_lights import WorldPointLights, world_lights_to_camera_lights
from restir_gs.lighting.deferred import LightingBuffers, PointLights, shade_deferred_lambertian
from restir_gs.metrics import compute_rgb_error_metrics
from restir_gs.render.gbuffer import GBuffer, make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.synthetic_scene import PinholeCamera, SyntheticGaussians
from restir_gs.restir.initial import (
    LightingEstimatorBuffers,
    ReservoirState,
    estimate_proposal_lighting,
    estimate_ris_initial_lighting,
)
from restir_gs.restir.proposal import CandidateSamples, compute_geometric_proposal_distribution, sample_light_candidates_from_distribution
from restir_gs.restir.temporal import (
    TemporalLookup,
    TemporalReservoirState,
    combine_temporal_reservoirs,
    reproject_current_to_previous,
    temporal_reservoir_from_initial,
)


@dataclass(frozen=True)
class RestirRenderSettings:
    candidate_count: int = 8
    candidate_seed_base: int = 31100
    initial_selection_seed_base: int = 32100
    temporal_selection_seed_base: int = 33100
    depth_tolerance: float = 0.05
    ambient: float = 0.2
    include_mc_baseline: bool = False


@dataclass(frozen=True)
class RestirHistory:
    gbuffer: GBuffer
    camera: PinholeCamera
    reservoir: TemporalReservoirState


@dataclass(frozen=True)
class RestirFrameResult:
    frame_index: int
    camera: PinholeCamera
    gbuffer: GBuffer
    lights: PointLights
    reference: LightingBuffers
    proposal_samples: CandidateSamples
    geometric_mc: LightingEstimatorBuffers | None
    initial: LightingEstimatorBuffers
    initial_reservoir: ReservoirState
    temporal: LightingEstimatorBuffers
    temporal_reservoir: TemporalReservoirState
    lookup: TemporalLookup
    history: RestirHistory


def render_restir_frame(
    scene: SyntheticGaussians,
    camera: PinholeCamera,
    world_lights: WorldPointLights,
    frame_index: int,
    settings: RestirRenderSettings = RestirRenderSettings(),
    previous_history: RestirHistory | None = None,
) -> RestirFrameResult:
    """Render one frame and run the aligned diffuse ReSTIR baseline."""
    with torch.no_grad():
        render_buffers = render_rgbd(scene, camera)
        gbuffer = make_pseudo_gbuffer(render_buffers, camera)
        lights = world_lights_to_camera_lights(world_lights, camera)
        return evaluate_restir_frame_from_gbuffer(
            gbuffer,
            camera,
            lights,
            frame_index=frame_index,
            settings=settings,
            previous_history=previous_history,
        )


def evaluate_restir_frame_from_gbuffer(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    frame_index: int,
    settings: RestirRenderSettings = RestirRenderSettings(),
    previous_history: RestirHistory | None = None,
) -> RestirFrameResult:
    """Run all-lights reference, geometric proposal, initial RIS, and temporal RIS on a prepared G-buffer."""
    _check_settings(settings)
    reference = shade_deferred_lambertian(gbuffer, lights, ambient=settings.ambient)
    valid_pixels = int(reference.valid_mask.sum().detach().cpu())
    if valid_pixels <= 0:
        raise RuntimeError(f"Frame {frame_index} has no valid lighting pixels.")

    proposal_distribution = compute_geometric_proposal_distribution(gbuffer, lights)
    samples = sample_light_candidates_from_distribution(
        proposal_distribution,
        settings.candidate_count,
        seed=settings.candidate_seed_base + frame_index,
        device=gbuffer.rgb.device,
    )
    geometric_mc = None
    if settings.include_mc_baseline:
        geometric_mc = estimate_proposal_lighting(
            gbuffer,
            lights,
            samples,
            ambient=settings.ambient,
            target_mode="diffuse",
        )
    initial, initial_reservoir = estimate_ris_initial_lighting(
        gbuffer,
        lights,
        samples.light_indices,
        selection_seed=settings.initial_selection_seed_base + frame_index,
        ambient=settings.ambient,
        proposal_probs=samples.proposal_probs,
        target_mode="diffuse",
    )

    if previous_history is None:
        lookup = empty_temporal_lookup(gbuffer)
        temporal = initial
        temporal_reservoir = temporal_reservoir_from_initial(initial_reservoir)
    else:
        lookup = reproject_current_to_previous(
            gbuffer,
            camera,
            previous_history.gbuffer,
            previous_history.camera,
            depth_tolerance=settings.depth_tolerance,
        )
        temporal, temporal_reservoir = combine_temporal_reservoirs(
            gbuffer,
            lights,
            initial,
            initial_reservoir,
            previous_history.reservoir,
            lookup,
            selection_seed=settings.temporal_selection_seed_base + frame_index,
            ambient=settings.ambient,
            target_mode="diffuse",
        )

    history = RestirHistory(gbuffer=gbuffer, camera=camera, reservoir=temporal_reservoir)
    return RestirFrameResult(
        frame_index=frame_index,
        camera=camera,
        gbuffer=gbuffer,
        lights=lights,
        reference=reference,
        proposal_samples=samples,
        geometric_mc=geometric_mc,
        initial=initial,
        initial_reservoir=initial_reservoir,
        temporal=temporal,
        temporal_reservoir=temporal_reservoir,
        lookup=lookup,
        history=history,
    )


def make_restir_metric_rows(
    asset_id: str,
    result: RestirFrameResult,
    settings: RestirRenderSettings,
) -> list[dict[str, int | float | str]]:
    valid_mask = result.reference.valid_mask
    valid_pixels = int(valid_mask.sum().detach().cpu())
    reuse_pixels = int(result.lookup.valid_mask.sum().detach().cpu())
    reuse_fraction = reuse_pixels / float(max(valid_pixels, 1))
    mean_depth_error = masked_mean(result.lookup.relative_depth_error, result.lookup.valid_mask)
    motion_magnitude = torch.linalg.norm(result.lookup.motion_pixels, dim=-1)
    mean_motion = masked_mean(motion_magnitude, result.lookup.valid_mask)

    rows: list[dict[str, int | float | str]] = []
    for estimator, buffers, reservoir in (
        ("initial_ris", result.initial, result.initial_reservoir),
        ("temporal_ris", result.temporal, result.temporal_reservoir),
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
                "target_mode": "diffuse",
                "proposal": "geometric",
                "k": settings.candidate_count,
                "candidate_seed": settings.candidate_seed_base + result.frame_index,
                "selection_seed": (
                    settings.initial_selection_seed_base + result.frame_index
                    if estimator == "initial_ris"
                    else settings.temporal_selection_seed_base + result.frame_index
                ),
                "valid_pixels": valid_pixels,
                "reuse_pixels": reuse_pixels,
                "reuse_fraction": reuse_fraction,
                "mean_relative_depth_error": mean_depth_error,
                "mean_motion_pixels": mean_motion,
                "reservoir_m_mean": m_mean,
                "reservoir_m_max": m_max,
            }
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
        summary.append(
            {
                "asset_id": asset_id,
                "estimator": estimator,
                "reference_quantity": quantity,
                "frame_count": len(group),
                "mae_mean": _mean(mae),
                "rmse_mean": _mean(rmse),
                "reuse_fraction_mean": _mean(reuse),
            }
        )
    return summary


def empty_temporal_lookup(gbuffer: GBuffer) -> TemporalLookup:
    height, width = gbuffer.depth.shape
    return TemporalLookup(
        prev_pixels=torch.zeros((height, width, 2), dtype=torch.long, device=gbuffer.rgb.device),
        valid_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        relative_depth_error=torch.full((height, width), float("inf"), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
        motion_pixels=torch.zeros((height, width, 2), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
    )


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


def _check_settings(settings: RestirRenderSettings) -> None:
    if settings.candidate_count <= 0:
        raise ValueError(f"Expected positive candidate_count, got {settings.candidate_count}")
    if settings.depth_tolerance < 0.0:
        raise ValueError(f"Expected non-negative depth_tolerance, got {settings.depth_tolerance}")


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0
