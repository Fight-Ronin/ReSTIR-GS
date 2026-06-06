from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from restir_gs.lighting.deferred import PointLights, shade_deferred_lambertian
from restir_gs.metrics import compute_rgb_error_metrics
from restir_gs.render.gbuffer import GBuffer
from restir_gs.restir.initial import estimate_ris_initial_diffuse, estimate_uniform_diffuse, sample_uniform_light_candidates


def compute_error_metrics(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute RGB error metrics over valid pixels."""
    return compute_rgb_error_metrics(estimate, reference, valid_mask)


def run_ris_ablation(
    gbuffer: GBuffer,
    lights: PointLights,
    k_values: Iterable[int],
    seed_count: int,
    candidate_seed_base: int = 3100,
    selection_seed_base: int = 4100,
    ambient: float = 0.2,
) -> list[dict[str, int | float | str]]:
    """Run uniform and RIS estimators over K and seed sweeps."""
    if seed_count <= 0:
        raise ValueError(f"Expected positive seed count, got {seed_count}")

    ks = list(k_values)
    if not ks:
        raise ValueError("Expected at least one K value.")
    if any(k <= 0 for k in ks):
        raise ValueError(f"Expected positive K values, got {ks}")

    reference = shade_deferred_lambertian(gbuffer, lights, ambient=ambient)
    rows: list[dict[str, int | float | str]] = []
    height, width = gbuffer.rgb.shape[:2]
    light_count = lights.positions_cam.shape[0]

    for k in ks:
        for seed_index in range(seed_count):
            candidate_seed = candidate_seed_base + seed_index
            selection_seed = selection_seed_base + seed_index
            candidates = sample_uniform_light_candidates(
                height,
                width,
                k,
                light_count,
                seed=candidate_seed,
                device=gbuffer.rgb.device,
            )
            uniform = estimate_uniform_diffuse(gbuffer, lights, candidates, ambient=ambient)
            ris, _ = estimate_ris_initial_diffuse(
                gbuffer,
                lights,
                candidates,
                selection_seed=selection_seed,
                ambient=ambient,
            )

            for estimator_name, estimate in (("uniform", uniform), ("ris", ris)):
                rows.append(
                    _make_row(
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
    """Group rows by quantity, estimator, and K with mean/std MAE and RMSE."""
    groups: dict[tuple[str, str, int], list[dict[str, int | float | str]]] = {}
    for row in rows:
        key = (str(row["reference_quantity"]), str(row["estimator"]), int(row["k"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, int | float | str]] = []
    for key in sorted(groups):
        quantity, estimator, k = key
        group = groups[key]
        mae_values = [float(row["mae"]) for row in group]
        rmse_values = [float(row["rmse"]) for row in group]
        summary.append(
            {
                "reference_quantity": quantity,
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
