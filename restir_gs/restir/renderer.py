from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

from restir_gs.lighting.asset_lights import WorldPointLights, world_lights_to_camera_lights
from restir_gs.lighting.deferred import LightingBuffers, PointLights, shade_deferred_lambertian
from restir_gs.lighting.visibility import (
    ShadowMapBundle,
    evaluate_selected_light_visible_diffuse,
    make_shadow_map_bundle,
    shade_deferred_lambertian_visible,
)
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
from restir_gs.restir.proposal import (
    CandidateSamples,
    compute_geometric_proposal_distribution,
    compute_visibility_geometric_proposal_distribution,
    sample_light_candidates_from_distribution,
)
from restir_gs.restir.temporal import (
    TemporalLookup,
    TemporalReservoirState,
    combine_temporal_reservoirs,
    reproject_current_to_previous,
    temporal_reservoir_from_initial,
)
from restir_gs.restir.visibility import estimate_visibility_proposal_lighting, estimate_visibility_ris_initial_lighting


RestirTargetMode = Literal["diffuse", "visibility"]


@dataclass(frozen=True)
class RestirRenderSettings:
    target_mode: RestirTargetMode = "diffuse"
    candidate_count: int = 8
    candidate_seed_base: int = 31100
    initial_selection_seed_base: int = 32100
    temporal_selection_seed_base: int = 33100
    depth_tolerance: float = 0.05
    temporal_normal_threshold: float | None = 0.85
    temporal_rgb_threshold: float | None = 0.20
    temporal_max_motion_pixels: float | None = 32.0
    ambient: float = 0.2
    include_mc_baseline: bool = False
    visibility_shadow_resolution: int = 128
    visibility_shadow_bias_scale: float = 0.02
    visibility_shadow_alpha_threshold: float = 1e-4
    visibility_shadow_bbox_percentile: float = 0.98


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
    shadow_bundle: ShadowMapBundle | None = None


def render_restir_frame(
    scene: SyntheticGaussians,
    camera: PinholeCamera,
    world_lights: WorldPointLights,
    frame_index: int,
    settings: RestirRenderSettings = RestirRenderSettings(),
    previous_history: RestirHistory | None = None,
    shadow_bundle: ShadowMapBundle | None = None,
) -> RestirFrameResult:
    """Render one frame and run the aligned diffuse ReSTIR baseline."""
    with torch.no_grad():
        render_buffers = render_rgbd(scene, camera)
        gbuffer = make_pseudo_gbuffer(render_buffers, camera)
        lights = world_lights_to_camera_lights(world_lights, camera)
        if settings.target_mode == "visibility" and shadow_bundle is None:
            target_world, scene_radius = _shadow_target_and_radius(scene.means, settings.visibility_shadow_bbox_percentile)
            shadow_bundle = make_shadow_map_bundle(
                scene,
                world_lights.positions_world,
                torch.arange(world_lights.positions_world.shape[0], dtype=torch.long, device=world_lights.positions_world.device),
                target_world,
                scene_radius=scene_radius,
                resolution=settings.visibility_shadow_resolution,
                shadow_bias_scale=settings.visibility_shadow_bias_scale,
            )
        return evaluate_restir_frame_from_gbuffer(
            gbuffer,
            camera,
            lights,
            frame_index=frame_index,
            settings=settings,
            previous_history=previous_history,
            shadow_bundle=shadow_bundle,
        )


def evaluate_restir_frame_from_gbuffer(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    frame_index: int,
    settings: RestirRenderSettings = RestirRenderSettings(),
    previous_history: RestirHistory | None = None,
    shadow_bundle: ShadowMapBundle | None = None,
) -> RestirFrameResult:
    """Run all-lights reference, geometric proposal, initial RIS, and temporal RIS on a prepared G-buffer."""
    _check_settings(settings)
    if settings.target_mode == "visibility":
        if shadow_bundle is None:
            raise ValueError("Visibility target mode requires a ShadowMapBundle.")
        reference = shade_deferred_lambertian_visible(
            gbuffer,
            camera,
            lights,
            shadow_bundle,
            ambient=settings.ambient,
            alpha_threshold=settings.visibility_shadow_alpha_threshold,
        )
    else:
        reference = shade_deferred_lambertian(gbuffer, lights, ambient=settings.ambient)
    valid_pixels = int(reference.valid_mask.sum().detach().cpu())
    if valid_pixels <= 0:
        raise RuntimeError(f"Frame {frame_index} has no valid lighting pixels.")

    proposal_distribution = _compute_proposal_distribution(gbuffer, camera, lights, shadow_bundle, settings)
    samples = sample_light_candidates_from_distribution(
        proposal_distribution,
        settings.candidate_count,
        seed=settings.candidate_seed_base + frame_index,
        device=gbuffer.rgb.device,
    )
    geometric_mc = None
    if settings.include_mc_baseline:
        if settings.target_mode == "visibility":
            geometric_mc = estimate_visibility_proposal_lighting(
                gbuffer,
                camera,
                lights,
                _require_shadow_bundle(shadow_bundle),
                samples,
                ambient=settings.ambient,
                alpha_threshold=settings.visibility_shadow_alpha_threshold,
            )
        else:
            geometric_mc = estimate_proposal_lighting(
                gbuffer,
                lights,
                samples,
                ambient=settings.ambient,
                target_mode="diffuse",
            )
    if settings.target_mode == "visibility":
        initial, initial_reservoir = estimate_visibility_ris_initial_lighting(
            gbuffer,
            camera,
            lights,
            _require_shadow_bundle(shadow_bundle),
            samples.light_indices,
            selection_seed=settings.initial_selection_seed_base + frame_index,
            ambient=settings.ambient,
            proposal_probs=samples.proposal_probs,
            alpha_threshold=settings.visibility_shadow_alpha_threshold,
        )
    else:
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
            normal_threshold=settings.temporal_normal_threshold,
            rgb_threshold=settings.temporal_rgb_threshold,
            max_motion_pixels=settings.temporal_max_motion_pixels,
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
            contribution_evaluator=_visibility_evaluator(gbuffer, camera, lights, shadow_bundle, settings)
            if settings.target_mode == "visibility"
            else None,
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
        shadow_bundle=shadow_bundle,
    )


def make_restir_metric_rows(
    asset_id: str,
    result: RestirFrameResult,
    settings: RestirRenderSettings,
) -> list[dict[str, int | float | str]]:
    valid_mask = result.reference.valid_mask
    valid_pixels = int(valid_mask.sum().detach().cpu())
    pre_gate_pixels = int(result.lookup.pre_gate_mask.sum().detach().cpu())
    pre_gate_fraction = pre_gate_pixels / float(max(valid_pixels, 1))
    reuse_pixels = int(result.lookup.valid_mask.sum().detach().cpu())
    reuse_fraction = reuse_pixels / float(max(valid_pixels, 1))
    mean_depth_error = masked_mean(result.lookup.relative_depth_error, result.lookup.valid_mask)
    mean_normal_dot = masked_mean(result.lookup.normal_dot, result.lookup.valid_mask)
    mean_rgb_distance = masked_mean(result.lookup.rgb_distance, result.lookup.valid_mask)
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
                "reuse_pixels": reuse_pixels,
                "reuse_fraction": reuse_fraction,
                "mean_relative_depth_error": mean_depth_error,
                "mean_temporal_normal_dot": mean_normal_dot,
                "mean_temporal_rgb_distance": mean_rgb_distance,
                "mean_motion_pixels": mean_motion,
                "temporal_normal_threshold": _format_optional_float(settings.temporal_normal_threshold),
                "temporal_rgb_threshold": _format_optional_float(settings.temporal_rgb_threshold),
                "temporal_max_motion_pixels": _format_optional_float(settings.temporal_max_motion_pixels),
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
        pre_gate_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        relative_depth_error=torch.full((height, width), float("inf"), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
        normal_dot=torch.zeros((height, width), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
        rgb_distance=torch.full((height, width), float("inf"), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
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
    if settings.target_mode not in ("diffuse", "visibility"):
        raise ValueError(f"Unsupported target_mode '{settings.target_mode}'.")
    if settings.candidate_count <= 0:
        raise ValueError(f"Expected positive candidate_count, got {settings.candidate_count}")
    if settings.depth_tolerance < 0.0:
        raise ValueError(f"Expected non-negative depth_tolerance, got {settings.depth_tolerance}")
    if settings.temporal_normal_threshold is not None and not -1.0 <= settings.temporal_normal_threshold <= 1.0:
        raise ValueError(f"Expected temporal_normal_threshold in [-1,1] or None, got {settings.temporal_normal_threshold}")
    if settings.temporal_rgb_threshold is not None and settings.temporal_rgb_threshold < 0.0:
        raise ValueError(f"Expected non-negative temporal_rgb_threshold or None, got {settings.temporal_rgb_threshold}")
    if settings.temporal_max_motion_pixels is not None and settings.temporal_max_motion_pixels < 0.0:
        raise ValueError(f"Expected non-negative temporal_max_motion_pixels or None, got {settings.temporal_max_motion_pixels}")
    if settings.visibility_shadow_resolution <= 0:
        raise ValueError(f"Expected positive visibility_shadow_resolution, got {settings.visibility_shadow_resolution}")
    if settings.visibility_shadow_bias_scale < 0.0:
        raise ValueError(f"Expected non-negative visibility_shadow_bias_scale, got {settings.visibility_shadow_bias_scale}")
    if settings.visibility_shadow_alpha_threshold < 0.0:
        raise ValueError(f"Expected non-negative visibility_shadow_alpha_threshold, got {settings.visibility_shadow_alpha_threshold}")
    if not 0.0 < settings.visibility_shadow_bbox_percentile <= 1.0:
        raise ValueError(f"Expected visibility_shadow_bbox_percentile in (0,1], got {settings.visibility_shadow_bbox_percentile}")


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _format_optional_float(value: float | None) -> float | str:
    return "none" if value is None else float(value)


def _require_shadow_bundle(shadow_bundle: ShadowMapBundle | None) -> ShadowMapBundle:
    if shadow_bundle is None:
        raise ValueError("Visibility target mode requires a ShadowMapBundle.")
    return shadow_bundle


def _compute_proposal_distribution(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    shadow_bundle: ShadowMapBundle | None,
    settings: RestirRenderSettings,
) -> torch.Tensor:
    if settings.target_mode == "diffuse":
        return compute_geometric_proposal_distribution(gbuffer, lights)
    if settings.target_mode == "visibility":
        return compute_visibility_geometric_proposal_distribution(
            gbuffer,
            camera,
            lights,
            _require_shadow_bundle(shadow_bundle),
            alpha_threshold=settings.visibility_shadow_alpha_threshold,
        )
    raise ValueError(f"Unsupported target_mode '{settings.target_mode}'.")


def _effective_proposal(settings: RestirRenderSettings) -> str:
    if settings.target_mode == "visibility":
        return "visibility_geometric"
    return "geometric"


def _visibility_evaluator(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    shadow_bundle: ShadowMapBundle | None,
    settings: RestirRenderSettings,
):
    bundle = _require_shadow_bundle(shadow_bundle)

    def evaluate(light_indices: torch.Tensor) -> torch.Tensor:
        return evaluate_selected_light_visible_diffuse(
            gbuffer,
            camera,
            lights,
            bundle,
            light_indices,
            alpha_threshold=settings.visibility_shadow_alpha_threshold,
        )

    return evaluate


def _shadow_target_and_radius(means: torch.Tensor, bbox_percentile: float) -> tuple[torch.Tensor, float]:
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"Expected Gaussian means shape [N,3], got {tuple(means.shape)}")
    if means.shape[0] <= 0:
        raise ValueError("Expected at least one Gaussian mean for shadow camera setup.")
    means_cpu = means.detach().cpu().float()
    if bbox_percentile >= 1.0:
        bbox_min = means_cpu.min(dim=0).values
        bbox_max = means_cpu.max(dim=0).values
    else:
        tail = (1.0 - float(bbox_percentile)) * 0.5
        quantiles = torch.tensor([tail, 1.0 - tail], dtype=means_cpu.dtype)
        bbox = torch.quantile(means_cpu, quantiles, dim=0)
        bbox_min = bbox[0]
        bbox_max = bbox[1]
    center = (bbox_min + bbox_max) * 0.5
    radius = torch.linalg.norm((bbox_max - bbox_min) * 0.5).clamp_min(1e-3)
    return center.to(device=means.device, dtype=means.dtype), float(radius)
