from __future__ import annotations

import json
import math

import pytest

from restir_gs.eval.real_asset_benchmark import (
    load_benchmark_manifest,
    normalize_benchmark_row,
    normalize_spatial_mis_row,
    select_top_candidate_indices,
    summarize_benchmark_rows,
)
from restir_gs.render.camera_probe import CameraProbeScore


def make_score(score: float, valid_pixels: int = 10) -> CameraProbeScore:
    return CameraProbeScore(
        score=score,
        valid_pixels=valid_pixels,
        coverage=0.5,
        central_coverage=0.5,
        border_coverage=0.0,
        brightness=0.5,
    )


def test_manifest_parser_validates_required_scene_fields(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "scenes": [{"scene_id": "scene_a", "ply": "outputs/assets/a.ply"}],
                "defaults": {"view_count": 2, "k_values": [1, 4], "seed_count": 3},
            }
        ),
        encoding="utf-8",
    )

    manifest = load_benchmark_manifest(path)

    assert manifest.scenes[0].scene_id == "scene_a"
    assert str(manifest.scenes[0].ply).replace("\\", "/") == "outputs/assets/a.ply"
    assert manifest.defaults.view_count == 2
    assert manifest.defaults.k_values == [1, 4]
    assert manifest.defaults.seed_count == 3


def test_manifest_missing_required_scene_field_fails_loudly(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"scenes": [{"scene_id": "scene_a"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required ply"):
        load_benchmark_manifest(path)


def test_top_view_selection_returns_highest_finite_positive_scores() -> None:
    scores = [
        make_score(0.2),
        make_score(float("nan")),
        make_score(0.9),
        make_score(0.7),
        make_score(1.0, valid_pixels=0),
    ]

    assert select_top_candidate_indices(scores, view_count=2) == [2, 3]


def test_row_normalization_adds_benchmark_metadata() -> None:
    row = {"proposal": "geometric", "estimator": "mc", "k": 8, "mae": 0.1, "rmse": 0.2}

    normalized = normalize_benchmark_row(
        row,
        scene_id="scene_a",
        view_id="view_00",
        camera_score=1.25,
        valid_pixels=123,
        method_family="proposal_ablation",
    )

    assert normalized["scene_id"] == "scene_a"
    assert normalized["view_id"] == "view_00"
    assert normalized["method_family"] == "proposal_ablation"
    assert normalized["camera_score"] == 1.25
    assert normalized["valid_pixels"] == 123
    assert normalized["proposal"] == "geometric"


def test_spatial_row_normalization_adds_common_grouping_keys() -> None:
    row = {"variant": "geometry_floor_0_50", "mae": 0.01, "rmse": 0.02}

    normalized = normalize_spatial_mis_row(
        row,
        scene_id="scene_a",
        view_id="view_00",
        camera_score=1.0,
        valid_pixels=10,
        k=8,
        candidate_seed=14100,
        selection_seed=14200,
    )

    assert normalized["method_family"] == "spatial_mis"
    assert normalized["proposal"] == "spatial_mis"
    assert normalized["estimator"] == "mc"
    assert normalized["reference_quantity"] == "diffuse_rgb"
    assert normalized["k"] == 8
    assert normalized["candidate_seed"] == 14100
    assert normalized["selection_seed"] == 14200


def test_summary_grouping_handles_mixed_method_rows() -> None:
    rows = [
        {
            "method_family": "proposal_ablation",
            "scene_id": "scene_a",
            "view_id": "view_00",
            "reference_quantity": "diffuse_rgb",
            "proposal": "geometric",
            "estimator": "mc",
            "k": 1,
            "mae": 0.2,
            "rmse": 0.4,
        },
        {
            "method_family": "proposal_ablation",
            "scene_id": "scene_a",
            "view_id": "view_00",
            "reference_quantity": "diffuse_rgb",
            "proposal": "geometric",
            "estimator": "mc",
            "k": 1,
            "mae": 0.4,
            "rmse": 0.6,
        },
        {
            "method_family": "spatial_mis",
            "scene_id": "scene_a",
            "view_id": "view_00",
            "reference_quantity": "diffuse_rgb",
            "proposal": "spatial_mis",
            "estimator": "mc",
            "k": 8,
            "variant": "geometry_floor_0_50",
            "mae": 0.1,
            "rmse": 0.2,
        },
    ]

    summary = summarize_benchmark_rows(rows)

    proposal_group = next(row for row in summary if row["method_family"] == "proposal_ablation")
    spatial_group = next(row for row in summary if row["method_family"] == "spatial_mis")
    assert proposal_group["sample_count"] == 2
    assert math.isclose(float(proposal_group["mae_mean"]), 0.3)
    assert math.isclose(float(proposal_group["rmse_mean"]), 0.5)
    assert spatial_group["sample_count"] == 1
    assert spatial_group["variant"] == "geometry_floor_0_50"
