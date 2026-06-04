from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from restir_gs.eval.ris_ablation import compute_error_metrics
from restir_gs.lighting.deferred import PointLights, shade_deferred_lambertian
from restir_gs.render.gbuffer import GBuffer
from restir_gs.restir.initial import (
    estimate_proposal_diffuse,
    estimate_ris_initial_diffuse,
    estimate_uniform_diffuse,
    sample_uniform_light_candidates,
)
from restir_gs.restir.proposal import (
    compute_geometric_proposal_distribution,
    sample_light_candidates_from_distribution,
)


def run_proposal_ablation(
    gbuffer: GBuffer,
    lights: PointLights,
    k_values: Iterable[int],
    seed_count: int,
    candidate_seed_base: int = 5100,
    selection_seed_base: int = 6100,
    ambient: float = 0.2,
) -> list[dict[str, int | float | str]]:
    """Run uniform and geometric proposal estimators over K and seed sweeps."""
    if seed_count <= 0:
        raise ValueError(f"Expected positive seed count, got {seed_count}")

    ks = list(k_values)
    if not ks:
        raise ValueError("Expected at least one K value.")
    if any(k <= 0 for k in ks):
        raise ValueError(f"Expected positive K values, got {ks}")

    reference = shade_deferred_lambertian(gbuffer, lights, ambient=ambient)
    geometric_distribution = compute_geometric_proposal_distribution(gbuffer, lights)
    rows: list[dict[str, int | float | str]] = []
    height, width = gbuffer.rgb.shape[:2]
    light_count = lights.positions_cam.shape[0]

    for k in ks:
        for seed_index in range(seed_count):
            candidate_seed = candidate_seed_base + seed_index
            selection_seed = selection_seed_base + seed_index
            uniform_indices = sample_uniform_light_candidates(
                height,
                width,
                k,
                light_count,
                seed=candidate_seed,
                device=gbuffer.rgb.device,
            )
            geometric_samples = sample_light_candidates_from_distribution(
                geometric_distribution,
                k,
                seed=candidate_seed,
                device=gbuffer.rgb.device,
            )

            uniform_mc = estimate_uniform_diffuse(gbuffer, lights, uniform_indices, ambient=ambient)
            uniform_ris, _ = estimate_ris_initial_diffuse(
                gbuffer,
                lights,
                uniform_indices,
                selection_seed=selection_seed,
                ambient=ambient,
            )
            geometric_mc = estimate_proposal_diffuse(gbuffer, lights, geometric_samples, ambient=ambient)
            geometric_ris, _ = estimate_ris_initial_diffuse(
                gbuffer,
                lights,
                geometric_samples.light_indices,
                selection_seed=selection_seed,
                ambient=ambient,
                proposal_probs=geometric_samples.proposal_probs,
            )

            estimates = (
                ("uniform", "mc", uniform_mc),
                ("uniform", "ris", uniform_ris),
                ("geometric", "mc", geometric_mc),
                ("geometric", "ris", geometric_ris),
            )
            for proposal_name, estimator_name, estimate in estimates:
                rows.append(
                    _make_row(
                        proposal_name,
                        estimator_name,
                        k,
                        seed_index,
                        candidate_seed,
                        selection_seed,
                        "diffuse_rgb",
                        estimate.diffuse_rgb,
                        reference.diffuse_rgb,
                        reference.valid_mask,
                    )
                )
                rows.append(
                    _make_row(
                        proposal_name,
                        estimator_name,
                        k,
                        seed_index,
                        candidate_seed,
                        selection_seed,
                        "composite_rgb",
                        estimate.composite_rgb,
                        reference.composite_rgb,
                        reference.valid_mask,
                    )
                )

    return rows


def _make_row(
    proposal: str,
    estimator: str,
    k: int,
    seed_index: int,
    candidate_seed: int,
    selection_seed: int,
    reference_quantity: str,
    estimate: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, int | float | str]:
    metrics = compute_error_metrics(estimate, reference, valid_mask)
    row: dict[str, int | float | str] = {
        "proposal": proposal,
        "estimator": estimator,
        "k": k,
        "seed_index": seed_index,
        "candidate_seed": candidate_seed,
        "selection_seed": selection_seed,
        "reference_quantity": reference_quantity,
    }
    row.update(metrics)
    return row


def summarize_rows(rows: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    """Group rows by quantity, proposal, estimator, and K with mean/std errors."""
    groups: dict[tuple[str, str, str, int], list[dict[str, int | float | str]]] = {}
    for row in rows:
        key = (str(row["reference_quantity"]), str(row["proposal"]), str(row["estimator"]), int(row["k"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, int | float | str]] = []
    for key in sorted(groups):
        quantity, proposal, estimator, k = key
        group = groups[key]
        mae_values = [float(row["mae"]) for row in group]
        rmse_values = [float(row["rmse"]) for row in group]
        summary.append(
            {
                "reference_quantity": quantity,
                "proposal": proposal,
                "estimator": estimator,
                "k": k,
                "sample_count": len(group),
                "mae_mean": _mean(mae_values),
                "mae_std": _std(mae_values),
                "rmse_mean": _mean(rmse_values),
                "rmse_std": _std(rmse_values),
            }
        )
    return summary


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) * (value - mean) for value in values) / float(len(values))
    return math.sqrt(variance)
