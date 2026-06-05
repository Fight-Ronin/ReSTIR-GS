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
    sample_light_candidates_from_distribution,
)
from restir_gs.restir.spatial_mis import (
    SpatialMISCandidates,
    SpatialMISStats,
    build_spatial_mis_candidates,
    estimate_spatial_mis_diffuse,
)
from restir_gs.restir.temporal import (
    TemporalLookup,
    TemporalReservoirState,
    combine_temporal_reservoirs,
    reproject_current_to_previous,
    temporal_reservoir_from_initial,
)

__all__ = [
    "CandidateSamples",
    "EstimatorBuffers",
    "LightingEstimatorBuffers",
    "ReservoirState",
    "SpatialMISCandidates",
    "SpatialMISStats",
    "TemporalLookup",
    "TemporalReservoirState",
    "combine_temporal_reservoirs",
    "compute_geometric_proposal_distribution",
    "estimate_proposal_diffuse",
    "estimate_proposal_lighting",
    "estimate_ris_initial_diffuse",
    "estimate_ris_initial_lighting",
    "estimate_spatial_mis_diffuse",
    "estimate_uniform_diffuse",
    "evaluate_selected_light_contribution",
    "build_spatial_mis_candidates",
    "reproject_current_to_previous",
    "sample_light_candidates_from_distribution",
    "sample_uniform_light_candidates",
    "temporal_reservoir_from_initial",
]
