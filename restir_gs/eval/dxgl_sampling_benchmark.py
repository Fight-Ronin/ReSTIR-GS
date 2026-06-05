from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from restir_gs.eval.gbuffer_validation import binary_mask_metrics, masked_rgb_metrics
from restir_gs.eval.ris_ablation import compute_error_metrics
from restir_gs.lighting.deferred import LightingBuffers, PointLights
from restir_gs.render.gbuffer import GBuffer
from restir_gs.restir.initial import (
    LightingEstimatorBuffers,
    estimate_proposal_lighting,
    estimate_ris_initial_lighting,
    sample_uniform_light_candidates,
)
from restir_gs.restir.proposal import CandidateSamples, compute_geometric_proposal_distribution, sample_light_candidates_from_distribution


TARGET_MODES = ("diffuse", "blinn_phong")
PROPOSALS = ("uniform", "geometric")
ESTIMATORS = ("mc", "ris")
REFERENCE_QUANTITIES = ("contribution_rgb", "composite_rgb")


def parse_k_values(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one K value, got {text!r}")
    if any(value <= 0 for value in values):
        raise ValueError(f"Expected positive K values, got {values}")
    return values


def select_evenly_spaced_frames(frame_count: int, view_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError(f"Expected positive frame_count, got {frame_count}")
    if view_count <= 0:
        raise ValueError(f"Expected positive view_count, got {view_count}")
    if view_count == 1:
        return [0]
    if view_count >= frame_count:
        return list(range(frame_count))
    return [min(int(index * frame_count / float(view_count)), frame_count - 1) for index in range(view_count)]


def expected_sampling_row_count(frame_count: int, k_values: Iterable[int], seed_count: int) -> int:
    ks = list(k_values)
    if frame_count <= 0:
        raise ValueError(f"Expected positive frame_count, got {frame_count}")
    if seed_count <= 0:
        raise ValueError(f"Expected positive seed_count, got {seed_count}")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"Expected positive K values, got {ks}")
    return frame_count * len(TARGET_MODES) * len(PROPOSALS) * len(ESTIMATORS) * len(ks) * seed_count * len(REFERENCE_QUANTITIES)


def run_sampling_benchmark_for_frame(
    gbuffer: GBuffer,
    lights: PointLights,
    lambertian_reference: LightingBuffers,
    blinn_reference: LightingBuffers,
    frame_index: int,
    k_values: Iterable[int],
    seed_count: int,
    candidate_seed_base: int = 15100,
    selection_seed_base: int = 16100,
    ambient: float = 0.2,
    specular_strength: float = 0.15,
    shininess: float = 24.0,
    rgb_mae_to_reference: float = 0.0,
    alpha_iou: float = 0.0,
) -> list[dict[str, int | float | str]]:
    """Run uniform/geometric MC/RIS rows for one aligned frame."""
    if seed_count <= 0:
        raise ValueError(f"Expected positive seed_count, got {seed_count}")
    ks = list(k_values)
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"Expected positive K values, got {ks}")

    valid_mask = gbuffer.valid_mask & gbuffer.normal_mask
    valid_pixels = int(valid_mask.sum().detach().cpu())
    if valid_pixels <= 0:
        raise RuntimeError(f"Frame {frame_index} has no valid G-buffer pixels.")

    height, width = gbuffer.rgb.shape[:2]
    light_count = lights.positions_cam.shape[0]
    geometric_distribution = compute_geometric_proposal_distribution(gbuffer, lights)
    rows: list[dict[str, int | float | str]] = []

    references = {
        "diffuse": {
            "contribution_rgb": lambertian_reference.diffuse_rgb,
            "composite_rgb": lambertian_reference.composite_rgb,
            "valid_mask": lambertian_reference.valid_mask,
        },
        "blinn_phong": {
            "contribution_rgb": blinn_reference.diffuse_rgb + blinn_reference.specular_rgb,
            "composite_rgb": blinn_reference.composite_rgb,
            "valid_mask": blinn_reference.valid_mask,
        },
    }

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
            uniform_samples = CandidateSamples(
                light_indices=uniform_indices,
                proposal_probs=torch.full(
                    uniform_indices.shape,
                    1.0 / float(light_count),
                    dtype=gbuffer.rgb.dtype,
                    device=gbuffer.rgb.device,
                ),
            )
            geometric_samples = sample_light_candidates_from_distribution(
                geometric_distribution,
                k,
                seed=candidate_seed,
                device=gbuffer.rgb.device,
            )
            sample_sets = (("uniform", uniform_samples), ("geometric", geometric_samples))

            for target_mode in TARGET_MODES:
                target_reference = references[target_mode]
                for proposal_name, samples in sample_sets:
                    mc = estimate_proposal_lighting(
                        gbuffer,
                        lights,
                        samples,
                        ambient=ambient,
                        target_mode=target_mode,  # type: ignore[arg-type]
                        specular_strength=specular_strength,
                        shininess=shininess,
                    )
                    ris, _ = estimate_ris_initial_lighting(
                        gbuffer,
                        lights,
                        samples.light_indices,
                        selection_seed=selection_seed,
                        ambient=ambient,
                        proposal_probs=samples.proposal_probs,
                        target_mode=target_mode,  # type: ignore[arg-type]
                        specular_strength=specular_strength,
                        shininess=shininess,
                    )
                    for estimator_name, estimate in (("mc", mc), ("ris", ris)):
                        rows.extend(
                            _make_sampling_rows(
                                estimate,
                                target_reference,
                                frame_index=frame_index,
                                target_mode=target_mode,
                                proposal=proposal_name,
                                estimator=estimator_name,
                                k=k,
                                seed_index=seed_index,
                                candidate_seed=candidate_seed,
                                selection_seed=selection_seed,
                                valid_pixels=valid_pixels,
                                rgb_mae_to_reference=rgb_mae_to_reference,
                                alpha_iou=alpha_iou,
                            )
                        )

    return rows


def frame_alignment_metrics(
    rendered_rgb: torch.Tensor,
    rendered_alpha: torch.Tensor,
    reference_rgb: torch.Tensor | None,
    reference_mask: torch.Tensor | None,
) -> dict[str, float]:
    if reference_rgb is None or reference_mask is None:
        return {"rgb_mae_to_reference": 0.0, "alpha_iou": 0.0}
    alpha_mask = rendered_alpha.detach().cpu() > 1e-4
    ref_mask = reference_mask.detach().cpu().to(torch.bool)
    rgb_metrics = masked_rgb_metrics(rendered_rgb.detach().cpu(), reference_rgb.detach().cpu(), ref_mask)
    alpha_metrics = binary_mask_metrics(alpha_mask, ref_mask)
    return {
        "rgb_mae_to_reference": float(rgb_metrics["mae"]),
        "alpha_iou": float(alpha_metrics["iou"]),
    }


def summarize_sampling_rows(rows: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    groups: dict[tuple[str, str, str, str, int], list[dict[str, int | float | str]]] = {}
    for row in rows:
        key = (
            str(row["target_mode"]),
            str(row["reference_quantity"]),
            str(row["proposal"]),
            str(row["estimator"]),
            int(row["k"]),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, int | float | str]] = []
    for key in sorted(groups):
        target_mode, reference_quantity, proposal, estimator, k = key
        group = groups[key]
        mae_values = [float(row["mae"]) for row in group]
        rmse_values = [float(row["rmse"]) for row in group]
        summary.append(
            {
                "target_mode": target_mode,
                "reference_quantity": reference_quantity,
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


def _make_sampling_rows(
    estimate: LightingEstimatorBuffers,
    reference: dict[str, torch.Tensor],
    frame_index: int,
    target_mode: str,
    proposal: str,
    estimator: str,
    k: int,
    seed_index: int,
    candidate_seed: int,
    selection_seed: int,
    valid_pixels: int,
    rgb_mae_to_reference: float,
    alpha_iou: float,
) -> list[dict[str, int | float | str]]:
    valid_mask = reference["valid_mask"]
    rows: list[dict[str, int | float | str]] = []
    for reference_quantity, estimate_tensor, reference_tensor in (
        ("contribution_rgb", estimate.contribution_rgb, reference["contribution_rgb"]),
        ("composite_rgb", estimate.composite_rgb, reference["composite_rgb"]),
    ):
        row: dict[str, int | float | str] = {
            "frame_index": frame_index,
            "target_mode": target_mode,
            "proposal": proposal,
            "estimator": estimator,
            "k": k,
            "seed_index": seed_index,
            "candidate_seed": candidate_seed,
            "selection_seed": selection_seed,
            "reference_quantity": reference_quantity,
            "valid_pixels": valid_pixels,
            "rgb_mae_to_reference": float(rgb_mae_to_reference),
            "alpha_iou": float(alpha_iou),
        }
        row.update(compute_error_metrics(estimate_tensor, reference_tensor, valid_mask))
        rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) * (value - mean) for value in values) / float(len(values))
    return math.sqrt(variance)
