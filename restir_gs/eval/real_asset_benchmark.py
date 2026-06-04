from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from restir_gs.render.camera_probe import CameraProbeScore


@dataclass(frozen=True)
class BenchmarkScene:
    scene_id: str
    ply: Path
    max_gaussians: int | None


@dataclass(frozen=True)
class BenchmarkDefaults:
    view_count: int
    width: int
    height: int
    num_lights: int
    k_values: list[int]
    seed_count: int
    candidate_seed_base: int
    selection_seed_base: int
    spatial_candidate_count: int
    spatial_candidate_seed: int
    spatial_initial_selection_seed: int


@dataclass(frozen=True)
class BenchmarkManifest:
    scenes: list[BenchmarkScene]
    defaults: BenchmarkDefaults


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    """Load and validate the real-asset benchmark manifest."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Benchmark manifest must be a JSON object.")
    scenes_data = data.get("scenes")
    if not isinstance(scenes_data, list) or not scenes_data:
        raise ValueError("Benchmark manifest requires a non-empty scenes list.")

    defaults_data = data.get("defaults", {})
    if defaults_data is None:
        defaults_data = {}
    if not isinstance(defaults_data, dict):
        raise ValueError("Benchmark manifest defaults must be an object.")
    defaults = _parse_defaults(defaults_data)

    scenes: list[BenchmarkScene] = []
    for index, item in enumerate(scenes_data):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark scene {index} must be an object.")
        scene_id = item.get("scene_id")
        ply = item.get("ply")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"Benchmark scene {index} missing required scene_id.")
        if not isinstance(ply, str) or not ply:
            raise ValueError(f"Benchmark scene {scene_id} missing required ply.")
        max_gaussians = item.get("max_gaussians", defaults_data.get("max_gaussians", 200000))
        scenes.append(
            BenchmarkScene(
                scene_id=scene_id,
                ply=Path(ply),
                max_gaussians=None if max_gaussians is None or int(max_gaussians) <= 0 else int(max_gaussians),
            )
        )

    return BenchmarkManifest(scenes=scenes, defaults=defaults)


def select_top_candidate_indices(scores: list[CameraProbeScore], view_count: int) -> list[int]:
    """Return the top-scoring positive-coverage camera candidate indices."""
    if view_count <= 0:
        raise ValueError(f"Expected positive view_count, got {view_count}")
    ranked = [
        index
        for index, score in enumerate(scores)
        if math.isfinite(score.score) and score.valid_pixels > 0
    ]
    if len(ranked) < view_count:
        raise RuntimeError(f"Need {view_count} valid camera candidates, found {len(ranked)}.")
    ranked.sort(key=lambda index: scores[index].score, reverse=True)
    return ranked[:view_count]


def normalize_benchmark_row(
    row: dict[str, Any],
    scene_id: str,
    view_id: str,
    camera_score: float,
    valid_pixels: int,
    method_family: str,
) -> dict[str, Any]:
    """Attach benchmark metadata to one estimator row."""
    out = {
        "scene_id": scene_id,
        "view_id": view_id,
        "camera_score": float(camera_score),
        "valid_pixels": int(valid_pixels),
        "method_family": method_family,
    }
    out.update(row)
    return out


def normalize_spatial_mis_row(
    row: dict[str, Any],
    scene_id: str,
    view_id: str,
    camera_score: float,
    valid_pixels: int,
    k: int,
    candidate_seed: int,
    selection_seed: int,
) -> dict[str, Any]:
    """Attach benchmark metadata and common grouping keys to a spatial-MIS row."""
    normalized = normalize_benchmark_row(
        row,
        scene_id=scene_id,
        view_id=view_id,
        camera_score=camera_score,
        valid_pixels=valid_pixels,
        method_family="spatial_mis",
    )
    normalized.setdefault("proposal", "spatial_mis")
    normalized.setdefault("estimator", "mc")
    normalized.setdefault("reference_quantity", "diffuse_rgb")
    normalized.setdefault("k", int(k))
    normalized.setdefault("seed_index", 0)
    normalized.setdefault("candidate_seed", int(candidate_seed))
    normalized.setdefault("selection_seed", int(selection_seed))
    return normalized


def summarize_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group benchmark rows by scene/view/method and error metric keys."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("method_family"),
            row.get("scene_id"),
            row.get("view_id"),
            row.get("reference_quantity"),
            row.get("proposal"),
            row.get("estimator"),
            row.get("k"),
            row.get("variant"),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key in sorted(groups, key=_summary_sort_key):
        method_family, scene_id, view_id, reference_quantity, proposal, estimator, k, variant = key
        group = groups[key]
        mae_values = [float(row["mae"]) for row in group]
        rmse_values = [float(row["rmse"]) for row in group]
        summary.append(
            {
                "method_family": method_family,
                "scene_id": scene_id,
                "view_id": view_id,
                "reference_quantity": reference_quantity,
                "proposal": proposal,
                "estimator": estimator,
                "k": k,
                "variant": variant,
                "sample_count": len(group),
                "mae_mean": _mean(mae_values),
                "mae_std": _std(mae_values),
                "rmse_mean": _mean(rmse_values),
                "rmse_std": _std(rmse_values),
            }
        )
    return summary


def _summary_sort_key(key: tuple[Any, ...]) -> tuple[str, str, str, str, str, str, int, str]:
    method_family, scene_id, view_id, reference_quantity, proposal, estimator, k, variant = key
    k_value = -1 if k is None else int(k)
    return (
        "" if method_family is None else str(method_family),
        "" if scene_id is None else str(scene_id),
        "" if view_id is None else str(view_id),
        "" if reference_quantity is None else str(reference_quantity),
        "" if proposal is None else str(proposal),
        "" if estimator is None else str(estimator),
        k_value,
        "" if variant is None else str(variant),
    )


def _parse_defaults(data: dict[str, Any]) -> BenchmarkDefaults:
    k_values = data.get("k_values", [1, 2, 4, 8, 16, 32])
    if not isinstance(k_values, list) or not k_values:
        raise ValueError("Benchmark defaults require a non-empty k_values list.")
    parsed_k_values = [int(value) for value in k_values]
    if any(value <= 0 for value in parsed_k_values):
        raise ValueError(f"Benchmark k_values must be positive: {parsed_k_values}")

    return BenchmarkDefaults(
        view_count=_positive_int(data, "view_count", 3),
        width=_positive_int(data, "width", 128),
        height=_positive_int(data, "height", 128),
        num_lights=_positive_int(data, "num_lights", 128),
        k_values=parsed_k_values,
        seed_count=_positive_int(data, "seed_count", 8),
        candidate_seed_base=_positive_int(data, "candidate_seed_base", 9100),
        selection_seed_base=_positive_int(data, "selection_seed_base", 10100),
        spatial_candidate_count=_positive_int(data, "spatial_candidate_count", 8),
        spatial_candidate_seed=_positive_int(data, "spatial_candidate_seed", 14100),
        spatial_initial_selection_seed=_positive_int(data, "spatial_initial_selection_seed", 14200),
    )


def _positive_int(data: dict[str, Any], key: str, default: int) -> int:
    value = int(data.get(key, default))
    if value <= 0:
        raise ValueError(f"Benchmark default {key} must be positive, got {value}.")
    return value


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) * (value - mean) for value in values) / float(len(values))
    return math.sqrt(variance)
