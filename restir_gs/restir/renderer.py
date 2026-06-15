from __future__ import annotations

from dataclasses import replace
import time

import torch

from restir_gs.lighting.asset_lights import WorldPointLights, world_lights_to_camera_lights
from restir_gs.lighting.deferred import LightingBuffers, PointLights, shade_deferred_lambertian
from restir_gs.lighting.shadow_maps import (
    ShadowMapBundle,
    make_shadow_map_bundle,
)
from restir_gs.lighting.shadow_visibility import (
    ShadowVisibilityCache,
    make_shadow_visibility_cache,
)
from restir_gs.lighting.visible_lighting import (
    evaluate_selected_light_visible_diffuse_cached,
    shade_deferred_lambertian_visible_cached,
)
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
from restir_gs.restir.metrics import (
    all_numeric_finite,
    make_restir_metric_rows,
    masked_mean,
    reservoir_m_stats,
    summarize_restir_asset_timing_rows,
    summarize_restir_rows,
    summarize_restir_timing_rows,
)
from restir_gs.restir.temporal_filter import (
    apply_confidence_clamped_temporal_filter,
    empty_temporal_filter_stats,
    empty_temporal_lookup,
)
from restir_gs.restir.temporal import (
    TemporalLookup,
    TemporalReservoirState,
    combine_temporal_reservoirs,
    reproject_current_to_previous,
    temporal_reservoir_from_initial,
)
from restir_gs.restir.types import (
    RESTIR_TIMING_FIELDS,
    RestirDisplayFrameResult,
    RestirFrameResult,
    RestirFrameTimings,
    RestirHistory,
    RestirRenderSettings,
    RestirTargetMode,
    TemporalFilterStats,
)
from restir_gs.restir.visibility import (
    estimate_visibility_proposal_lighting_cached,
    estimate_visibility_ris_initial_lighting_cached,
)


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
