"""Evaluation helpers for the ReSTIR-GS prototype."""

from restir_gs.eval.ris_ablation import compute_error_metrics, run_ris_ablation, summarize_rows
from restir_gs.eval.proposal_ablation import (
    run_proposal_ablation,
    summarize_rows as summarize_proposal_rows,
)
from restir_gs.eval.real_asset_benchmark import (
    BenchmarkDefaults,
    BenchmarkManifest,
    BenchmarkScene,
    load_benchmark_manifest,
    normalize_benchmark_row,
    normalize_spatial_mis_row,
    select_top_candidate_indices,
    summarize_benchmark_rows,
)
from restir_gs.eval.spatial_mis_ablation import (
    SpatialMISAblationResult,
    SpatialMISVariant,
    default_spatial_mis_variants,
    run_spatial_mis_ablation,
)

__all__ = [
    "SpatialMISAblationResult",
    "SpatialMISVariant",
    "BenchmarkDefaults",
    "BenchmarkManifest",
    "BenchmarkScene",
    "compute_error_metrics",
    "default_spatial_mis_variants",
    "load_benchmark_manifest",
    "normalize_benchmark_row",
    "normalize_spatial_mis_row",
    "run_proposal_ablation",
    "run_ris_ablation",
    "run_spatial_mis_ablation",
    "select_top_candidate_indices",
    "summarize_benchmark_rows",
    "summarize_proposal_rows",
    "summarize_rows",
]
