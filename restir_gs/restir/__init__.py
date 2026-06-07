"""Reservoir sampling helpers for the ReSTIR-GS prototype."""

from restir_gs.restir.initial import (
    EstimatorBuffers,
    LightingEstimatorBuffers,
    ReservoirState,
    estimate_proposal_lighting,
    estimate_proposal_diffuse,
    estimate_ris_initial_diffuse,
    estimate_ris_initial_lighting,
    estimate_uniform_diffuse,
    evaluate_selected_light_contribution,
    sample_uniform_light_candidates,
)
from restir_gs.restir.proposal import (
    CandidateSamples,
    compute_geometric_proposal_distribution,
    compute_visibility_geometric_proposal_distribution,
    sample_light_candidates_from_distribution,
)
from restir_gs.restir.renderer import (
    RestirFrameResult,
    RestirHistory,
    RestirRenderSettings,
    all_numeric_finite,
    empty_temporal_lookup,
    evaluate_restir_frame_from_gbuffer,
    make_restir_metric_rows,
    render_restir_frame,
    summarize_restir_rows,
)
from restir_gs.restir.temporal import (
    TemporalLookup,
    TemporalReservoirState,
    combine_temporal_reservoirs,
    reproject_current_to_previous,
    temporal_reservoir_from_initial,
)
from restir_gs.restir.visibility import (
    VisibilityEstimatorBuffers,
    estimate_visibility_proposal_lighting,
    estimate_visibility_ris_initial_lighting,
)

__all__ = [
    "CandidateSamples",
    "EstimatorBuffers",
    "LightingEstimatorBuffers",
    "RestirFrameResult",
    "RestirHistory",
    "RestirRenderSettings",
    "ReservoirState",
    "TemporalLookup",
    "TemporalReservoirState",
    "VisibilityEstimatorBuffers",
    "all_numeric_finite",
    "combine_temporal_reservoirs",
    "compute_geometric_proposal_distribution",
    "compute_visibility_geometric_proposal_distribution",
    "empty_temporal_lookup",
    "evaluate_restir_frame_from_gbuffer",
    "estimate_proposal_diffuse",
    "estimate_proposal_lighting",
    "estimate_ris_initial_diffuse",
    "estimate_ris_initial_lighting",
    "estimate_uniform_diffuse",
    "estimate_visibility_proposal_lighting",
    "estimate_visibility_ris_initial_lighting",
    "evaluate_selected_light_contribution",
    "make_restir_metric_rows",
    "reproject_current_to_previous",
    "render_restir_frame",
    "sample_light_candidates_from_distribution",
    "sample_uniform_light_candidates",
    "summarize_restir_rows",
    "temporal_reservoir_from_initial",
]
