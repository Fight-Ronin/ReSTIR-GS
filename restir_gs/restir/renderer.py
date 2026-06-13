from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Literal

import torch

from restir_gs.lighting.asset_lights import WorldPointLights, world_lights_to_camera_lights
from restir_gs.lighting.deferred import LightingBuffers, PointLights, shade_deferred_lambertian
from restir_gs.lighting.visibility import (
    ShadowMapBundle,
    ShadowVisibilityCache,
    evaluate_selected_light_visible_diffuse_cached,
    make_shadow_visibility_cache,
    make_shadow_map_bundle,
    shade_deferred_lambertian_visible_cached,
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
    compute_visibility_geometric_proposal_distribution_cached,
    sample_light_candidates_from_distribution,
)
from restir_gs.restir.temporal import (
    TemporalLookup,
    TemporalReservoirState,
    combine_temporal_reservoirs,
    reproject_current_to_previous,
    temporal_reservoir_from_initial,
)
from restir_gs.restir.visibility import (
    estimate_visibility_proposal_lighting_cached,
    estimate_visibility_ris_initial_lighting_cached,
)


RestirTargetMode = Literal["diffuse", "visibility"]


RESTIR_TIMING_FIELDS = (
    "render_rgbd_gpu_ms",
    "gbuffer_gpu_ms",
    "world_lights_to_camera_gpu_ms",
    "reference_lighting_gpu_ms",
    "proposal_distribution_gpu_ms",
    "proposal_sampling_gpu_ms",
    "proposal_gpu_ms",
    "initial_ris_gpu_ms",
    "temporal_lookup_gpu_ms",
    "temporal_ris_gpu_ms",
    "temporal_filter_gpu_ms",
    "frame_gpu_ms",
    "frame_wall_ms",
    "shadow_bundle_asset_gpu_ms",
)


@dataclass(frozen=True)
class RestirFrameTimings:
    render_rgbd_gpu_ms: float = 0.0
    gbuffer_gpu_ms: float = 0.0
    world_lights_to_camera_gpu_ms: float = 0.0
    reference_lighting_gpu_ms: float = 0.0
    proposal_distribution_gpu_ms: float = 0.0
    proposal_sampling_gpu_ms: float = 0.0
    proposal_gpu_ms: float = 0.0
    initial_ris_gpu_ms: float = 0.0
    temporal_lookup_gpu_ms: float = 0.0
    temporal_ris_gpu_ms: float = 0.0
    temporal_filter_gpu_ms: float = 0.0
    frame_gpu_ms: float = 0.0
    frame_wall_ms: float = 0.0
    shadow_bundle_asset_gpu_ms: float = 0.0

    def as_row_fields(self, shadow_bundle_asset_gpu_ms: float | None = None) -> dict[str, float]:
        shadow_ms = self.shadow_bundle_asset_gpu_ms if shadow_bundle_asset_gpu_ms is None else float(shadow_bundle_asset_gpu_ms)
        return {
            "render_rgbd_gpu_ms": float(self.render_rgbd_gpu_ms),
            "gbuffer_gpu_ms": float(self.gbuffer_gpu_ms),
            "world_lights_to_camera_gpu_ms": float(self.world_lights_to_camera_gpu_ms),
            "reference_lighting_gpu_ms": float(self.reference_lighting_gpu_ms),
            "proposal_distribution_gpu_ms": float(self.proposal_distribution_gpu_ms),
            "proposal_sampling_gpu_ms": float(self.proposal_sampling_gpu_ms),
            "proposal_gpu_ms": float(self.proposal_gpu_ms),
            "initial_ris_gpu_ms": float(self.initial_ris_gpu_ms),
            "temporal_lookup_gpu_ms": float(self.temporal_lookup_gpu_ms),
            "temporal_ris_gpu_ms": float(self.temporal_ris_gpu_ms),
            "temporal_filter_gpu_ms": float(self.temporal_filter_gpu_ms),
            "frame_gpu_ms": float(self.frame_gpu_ms),
            "frame_wall_ms": float(self.frame_wall_ms),
            "shadow_bundle_asset_gpu_ms": shadow_ms,
        }


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
    temporal_reprojection_search_radius: int = 1
    temporal_history_m_cap: int | None = None
    temporal_filter_blend_max: float = 0.15
    temporal_filter_clamp_scale: float = 0.50
    temporal_filter_clamp_min: float = 1e-5
    ambient: float = 0.2
    include_mc_baseline: bool = False
    visibility_shadow_resolution: int = 128
    visibility_shadow_bias_scale: float = 0.02
    visibility_shadow_alpha_threshold: float = 1e-4
    visibility_shadow_pcf_radius: int = 1
    visibility_shadow_bbox_percentile: float = 0.98


@dataclass(frozen=True)
class RestirHistory:
    gbuffer: GBuffer
    camera: PinholeCamera
    reservoir: TemporalReservoirState
    filtered: LightingEstimatorBuffers


@dataclass(frozen=True)
class TemporalFilterStats:
    confidence: torch.Tensor
    alpha: torch.Tensor
    history_delta: torch.Tensor
    clamp_delta: torch.Tensor


@dataclass(frozen=True)
class RestirDisplayFrameResult:
    frame_index: int
    camera: PinholeCamera
    gbuffer: GBuffer
    lights: PointLights
    proposal_samples: CandidateSamples
    geometric_mc: LightingEstimatorBuffers | None
    initial: LightingEstimatorBuffers
    initial_reservoir: ReservoirState
    temporal: LightingEstimatorBuffers
    temporal_filtered: LightingEstimatorBuffers
    temporal_reservoir: TemporalReservoirState
    temporal_filter_stats: TemporalFilterStats
    lookup: TemporalLookup
    history: RestirHistory
    shadow_bundle: ShadowMapBundle | None = None
    timings: RestirFrameTimings = RestirFrameTimings()


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
    temporal_filtered: LightingEstimatorBuffers
    temporal_reservoir: TemporalReservoirState
    temporal_filter_stats: TemporalFilterStats
    lookup: TemporalLookup
    history: RestirHistory
    shadow_bundle: ShadowMapBundle | None = None
    timings: RestirFrameTimings = RestirFrameTimings()


class _FrameStageTimer:
    def __init__(self, device: torch.device, enabled: bool | None = None) -> None:
        self.device = device
        self.enabled = bool(device.type == "cuda" and torch.cuda.is_available()) if enabled is None else enabled
        self.events: dict[str, torch.cuda.Event] = {}

    @classmethod
    def disabled(cls, device: torch.device) -> "_FrameStageTimer":
        return cls(device, enabled=False)

    def mark(self, label: str) -> None:
        if not self.enabled:
            return
        with torch.cuda.device(self.device):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        self.events[label] = event

    def elapsed(self, start: str, end: str) -> float:
        if not self.enabled or start not in self.events or end not in self.events:
            return 0.0
        return float(self.events[start].elapsed_time(self.events[end]))

    def to_timings(self) -> RestirFrameTimings:
        if not self.enabled:
            return RestirFrameTimings()
        torch.cuda.synchronize(self.device)
        return RestirFrameTimings(
            render_rgbd_gpu_ms=self.elapsed("start", "after_render_rgbd"),
            gbuffer_gpu_ms=self.elapsed("after_render_rgbd", "after_gbuffer"),
            world_lights_to_camera_gpu_ms=self.elapsed("after_gbuffer", "after_world_lights_to_camera"),
            reference_lighting_gpu_ms=self.elapsed("after_world_lights_to_camera", "after_reference_lighting"),
            proposal_distribution_gpu_ms=self.elapsed("after_reference_lighting", "after_proposal_distribution"),
            proposal_sampling_gpu_ms=self.elapsed("after_proposal_distribution", "after_proposal"),
            proposal_gpu_ms=self.elapsed("after_reference_lighting", "after_proposal"),
            initial_ris_gpu_ms=self.elapsed("after_proposal", "after_initial_ris"),
            temporal_lookup_gpu_ms=self.elapsed("after_initial_ris", "after_temporal_lookup"),
            temporal_ris_gpu_ms=self.elapsed("after_temporal_lookup", "after_temporal_ris"),
            temporal_filter_gpu_ms=self.elapsed("after_temporal_ris", "after_temporal_filter"),
            frame_gpu_ms=self.elapsed("start", "after_temporal_filter"),
        )


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
    wall_start = time.perf_counter()
    timer = _FrameStageTimer(scene.means.device)
    with torch.no_grad():
        timer.mark("start")
        render_buffers = render_rgbd(scene, camera)
        timer.mark("after_render_rgbd")
        gbuffer = make_pseudo_gbuffer(render_buffers, camera)
        timer.mark("after_gbuffer")
        lights = world_lights_to_camera_lights(world_lights, camera)
        timer.mark("after_world_lights_to_camera")
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
        result = evaluate_restir_frame_from_gbuffer(
            gbuffer,
            camera,
            lights,
            frame_index=frame_index,
            settings=settings,
            previous_history=previous_history,
            shadow_bundle=shadow_bundle,
            _timer=timer,
        )
        timings = replace(result.timings, frame_wall_ms=(time.perf_counter() - wall_start) * 1000.0)
        return replace(result, timings=timings)


def evaluate_restir_display_frame_from_gbuffer(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    frame_index: int,
    settings: RestirRenderSettings = RestirRenderSettings(),
    previous_history: RestirHistory | None = None,
    shadow_bundle: ShadowMapBundle | None = None,
    _timer: _FrameStageTimer | None = None,
) -> RestirDisplayFrameResult:
    """Run the display renderer path without computing an all-lights reference."""
    display, _ = _evaluate_restir_frame_core(
        gbuffer,
        camera,
        lights,
        frame_index=frame_index,
        settings=settings,
        previous_history=previous_history,
        shadow_bundle=shadow_bundle,
        _timer=_timer,
        include_reference=False,
    )
    return display


def evaluate_restir_frame_from_gbuffer(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    frame_index: int,
    settings: RestirRenderSettings = RestirRenderSettings(),
    previous_history: RestirHistory | None = None,
    shadow_bundle: ShadowMapBundle | None = None,
    _timer: _FrameStageTimer | None = None,
) -> RestirFrameResult:
    """Run the evaluation renderer path, including all-lights reference buffers."""
    display, reference = _evaluate_restir_frame_core(
        gbuffer,
        camera,
        lights,
        frame_index=frame_index,
        settings=settings,
        previous_history=previous_history,
        shadow_bundle=shadow_bundle,
        _timer=_timer,
        include_reference=True,
    )
    if reference is None:
        raise RuntimeError("Internal renderer error: evaluation path did not produce a reference.")
    return RestirFrameResult(
        frame_index=display.frame_index,
        camera=display.camera,
        gbuffer=display.gbuffer,
        lights=display.lights,
        reference=reference,
        proposal_samples=display.proposal_samples,
        geometric_mc=display.geometric_mc,
        initial=display.initial,
        initial_reservoir=display.initial_reservoir,
        temporal=display.temporal,
        temporal_filtered=display.temporal_filtered,
        temporal_reservoir=display.temporal_reservoir,
        temporal_filter_stats=display.temporal_filter_stats,
        lookup=display.lookup,
        history=display.history,
        shadow_bundle=display.shadow_bundle,
        timings=display.timings,
    )


def _evaluate_restir_frame_core(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    frame_index: int,
    settings: RestirRenderSettings,
    previous_history: RestirHistory | None,
    shadow_bundle: ShadowMapBundle | None,
    _timer: _FrameStageTimer | None,
    include_reference: bool,
) -> tuple[RestirDisplayFrameResult, LightingBuffers | None]:
    timer = _timer or _FrameStageTimer.disabled(gbuffer.rgb.device)
    _check_settings(settings)
    visibility_cache = None
    reference = None
    if settings.target_mode == "visibility":
        if shadow_bundle is None:
            raise ValueError("Visibility target mode requires a ShadowMapBundle.")
        visibility_cache = make_shadow_visibility_cache(
            gbuffer,
            camera,
            shadow_bundle,
            alpha_threshold=settings.visibility_shadow_alpha_threshold,
            pcf_radius=settings.visibility_shadow_pcf_radius,
        )
        if include_reference:
            reference = shade_deferred_lambertian_visible_cached(
                gbuffer,
                lights,
                visibility_cache,
                ambient=settings.ambient,
            )
    elif include_reference:
        reference = shade_deferred_lambertian(gbuffer, lights, ambient=settings.ambient)
    timer.mark("after_reference_lighting")
    valid_mask = reference.valid_mask if reference is not None else (gbuffer.valid_mask & gbuffer.normal_mask)
    valid_pixels = int(valid_mask.sum().detach().cpu())
    if valid_pixels <= 0:
        raise RuntimeError(f"Frame {frame_index} has no valid lighting pixels.")

    proposal_distribution = _compute_proposal_distribution(gbuffer, camera, lights, shadow_bundle, settings, visibility_cache)
    timer.mark("after_proposal_distribution")
    samples = sample_light_candidates_from_distribution(
        proposal_distribution,
        settings.candidate_count,
        seed=settings.candidate_seed_base + frame_index,
        device=gbuffer.rgb.device,
    )
    timer.mark("after_proposal")
    visibility_candidate_contributions = None
    if settings.target_mode == "visibility":
        visibility_candidate_contributions = evaluate_selected_light_visible_diffuse_cached(
            gbuffer,
            lights,
            _require_visibility_cache(visibility_cache),
            samples.light_indices,
        )
    geometric_mc = None
    if settings.include_mc_baseline:
        if settings.target_mode == "visibility":
            geometric_mc = estimate_visibility_proposal_lighting_cached(
                gbuffer,
                lights,
                _require_visibility_cache(visibility_cache),
                samples,
                ambient=settings.ambient,
                contribution_candidates=visibility_candidate_contributions,
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
        initial, initial_reservoir = estimate_visibility_ris_initial_lighting_cached(
            gbuffer,
            lights,
            _require_visibility_cache(visibility_cache),
            samples.light_indices,
            selection_seed=settings.initial_selection_seed_base + frame_index,
            ambient=settings.ambient,
            proposal_probs=samples.proposal_probs,
            contribution_candidates=visibility_candidate_contributions,
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
    timer.mark("after_initial_ris")

    if previous_history is None:
        lookup = empty_temporal_lookup(gbuffer)
        timer.mark("after_temporal_lookup")
        temporal = initial
        temporal_reservoir = temporal_reservoir_from_initial(initial_reservoir)
        timer.mark("after_temporal_ris")
        temporal_filtered = initial
        temporal_filter_stats = empty_temporal_filter_stats(gbuffer)
        timer.mark("after_temporal_filter")
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
            search_radius=settings.temporal_reprojection_search_radius,
        )
        timer.mark("after_temporal_lookup")
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
            contribution_evaluator=_visibility_evaluator(gbuffer, lights, visibility_cache)
            if settings.target_mode == "visibility"
            else None,
            history_m_cap=_effective_temporal_history_m_cap(settings),
        )
        timer.mark("after_temporal_ris")
        temporal_filtered, temporal_filter_stats = apply_confidence_clamped_temporal_filter(
            gbuffer,
            initial,
            previous_history.filtered,
            lookup,
            settings,
        )
        timer.mark("after_temporal_filter")

    history = RestirHistory(gbuffer=gbuffer, camera=camera, reservoir=temporal_reservoir, filtered=temporal_filtered)
    display = RestirDisplayFrameResult(
        frame_index=frame_index,
        camera=camera,
        gbuffer=gbuffer,
        lights=lights,
        proposal_samples=samples,
        geometric_mc=geometric_mc,
        initial=initial,
        initial_reservoir=initial_reservoir,
        temporal=temporal,
        temporal_filtered=temporal_filtered,
        temporal_reservoir=temporal_reservoir,
        temporal_filter_stats=temporal_filter_stats,
        lookup=lookup,
        history=history,
        shadow_bundle=shadow_bundle,
        timings=timer.to_timings(),
    )
    return display, reference


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


def empty_temporal_lookup(gbuffer: GBuffer) -> TemporalLookup:
    height, width = gbuffer.depth.shape
    return TemporalLookup(
        prev_pixels=torch.zeros((height, width, 2), dtype=torch.long, device=gbuffer.rgb.device),
        valid_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        pre_gate_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        normal_pass_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        rgb_pass_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        motion_pass_mask=torch.zeros((height, width), dtype=torch.bool, device=gbuffer.rgb.device),
        relative_depth_error=torch.full((height, width), float("inf"), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
        normal_dot=torch.zeros((height, width), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
        normal_abs_dot=torch.zeros((height, width), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
        rgb_distance=torch.full((height, width), float("inf"), dtype=gbuffer.rgb.dtype, device=gbuffer.rgb.device),
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
    composite = torch.where(current.valid_mask.to(device=device)[..., None], composite_lit, current.composite_rgb.to(device=device, dtype=dtype))
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
        rgb_conf = (1.0 - lookup.rgb_distance.to(device=device, dtype=dtype) / float(settings.temporal_rgb_threshold)).clamp(0.0, 1.0)

    motion_magnitude = torch.linalg.norm(lookup.motion_pixels.to(device=device, dtype=dtype), dim=-1)
    if settings.temporal_max_motion_pixels is None:
        motion_conf = torch.ones_like(depth_conf)
    elif settings.temporal_max_motion_pixels == 0.0:
        motion_conf = (motion_magnitude <= 0.0).to(dtype=dtype)
    else:
        motion_conf = (1.0 - motion_magnitude / float(settings.temporal_max_motion_pixels)).clamp(0.0, 1.0)

    confidence = torch.minimum(torch.minimum(depth_conf, normal_conf), torch.minimum(rgb_conf, motion_conf))
    return torch.where(lookup.valid_mask.to(device=device), confidence, torch.zeros_like(confidence))


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
    if settings.temporal_reprojection_search_radius < 0:
        raise ValueError(f"Expected non-negative temporal_reprojection_search_radius, got {settings.temporal_reprojection_search_radius}")
    if settings.temporal_history_m_cap is not None and settings.temporal_history_m_cap <= 0:
        raise ValueError(f"Expected positive temporal_history_m_cap or None, got {settings.temporal_history_m_cap}")
    if not 0.0 <= settings.temporal_filter_blend_max <= 1.0:
        raise ValueError(f"Expected temporal_filter_blend_max in [0,1], got {settings.temporal_filter_blend_max}")
    if settings.temporal_filter_clamp_scale < 0.0:
        raise ValueError(f"Expected non-negative temporal_filter_clamp_scale, got {settings.temporal_filter_clamp_scale}")
    if settings.temporal_filter_clamp_min < 0.0:
        raise ValueError(f"Expected non-negative temporal_filter_clamp_min, got {settings.temporal_filter_clamp_min}")
    if settings.visibility_shadow_resolution <= 0:
        raise ValueError(f"Expected positive visibility_shadow_resolution, got {settings.visibility_shadow_resolution}")
    if settings.visibility_shadow_bias_scale < 0.0:
        raise ValueError(f"Expected non-negative visibility_shadow_bias_scale, got {settings.visibility_shadow_bias_scale}")
    if settings.visibility_shadow_alpha_threshold < 0.0:
        raise ValueError(f"Expected non-negative visibility_shadow_alpha_threshold, got {settings.visibility_shadow_alpha_threshold}")
    if settings.visibility_shadow_pcf_radius < 0:
        raise ValueError(f"Expected non-negative visibility_shadow_pcf_radius, got {settings.visibility_shadow_pcf_radius}")
    if not 0.0 < settings.visibility_shadow_bbox_percentile <= 1.0:
        raise ValueError(f"Expected visibility_shadow_bbox_percentile in (0,1], got {settings.visibility_shadow_bbox_percentile}")


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


def _require_shadow_bundle(shadow_bundle: ShadowMapBundle | None) -> ShadowMapBundle:
    if shadow_bundle is None:
        raise ValueError("Visibility target mode requires a ShadowMapBundle.")
    return shadow_bundle


def _require_visibility_cache(visibility_cache: ShadowVisibilityCache | None) -> ShadowVisibilityCache:
    if visibility_cache is None:
        raise ValueError("Visibility target mode requires a ShadowVisibilityCache.")
    return visibility_cache


def _compute_proposal_distribution(
    gbuffer: GBuffer,
    camera: PinholeCamera,
    lights: PointLights,
    shadow_bundle: ShadowMapBundle | None,
    settings: RestirRenderSettings,
    visibility_cache: ShadowVisibilityCache | None = None,
) -> torch.Tensor:
    if settings.target_mode == "diffuse":
        return compute_geometric_proposal_distribution(gbuffer, lights)
    if settings.target_mode == "visibility":
        if visibility_cache is not None:
            return compute_visibility_geometric_proposal_distribution_cached(
                gbuffer,
                lights,
                visibility_cache,
            )
        return compute_visibility_geometric_proposal_distribution(
            gbuffer,
            camera,
            lights,
            _require_shadow_bundle(shadow_bundle),
            alpha_threshold=settings.visibility_shadow_alpha_threshold,
            pcf_radius=settings.visibility_shadow_pcf_radius,
        )
    raise ValueError(f"Unsupported target_mode '{settings.target_mode}'.")


def _effective_proposal(settings: RestirRenderSettings) -> str:
    if settings.target_mode == "visibility":
        return "visibility_geometric"
    return "geometric"


def _visibility_evaluator(
    gbuffer: GBuffer,
    lights: PointLights,
    visibility_cache: ShadowVisibilityCache | None,
):
    cache = _require_visibility_cache(visibility_cache)

    def evaluate(light_indices: torch.Tensor) -> torch.Tensor:
        return evaluate_selected_light_visible_diffuse_cached(
            gbuffer,
            lights,
            cache,
            light_indices,
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
