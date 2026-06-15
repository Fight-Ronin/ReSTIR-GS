from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from restir_gs.lighting.deferred import LightingBuffers, PointLights
from restir_gs.lighting.shadow_maps import ShadowMapBundle
from restir_gs.render.gbuffer import GBuffer
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.restir.initial import LightingEstimatorBuffers, ReservoirState
from restir_gs.restir.proposal import CandidateSamples
from restir_gs.restir.temporal import TemporalLookup, TemporalReservoirState


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
        shadow_ms = (
            self.shadow_bundle_asset_gpu_ms
            if shadow_bundle_asset_gpu_ms is None
            else float(shadow_bundle_asset_gpu_ms)
        )
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
