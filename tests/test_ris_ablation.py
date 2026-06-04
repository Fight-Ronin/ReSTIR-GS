from __future__ import annotations

import math

import torch

from restir_gs.eval.ris_ablation import compute_error_metrics, run_ris_ablation, summarize_rows
from restir_gs.lighting.deferred import PointLights
from restir_gs.render.gbuffer import GBuffer


REQUIRED_KEYS = {
    "estimator",
    "k",
    "seed_index",
    "candidate_seed",
    "selection_seed",
    "reference_quantity",
    "mae",
    "rmse",
    "bias_r",
    "bias_g",
    "bias_b",
    "mean_abs_bias",
}


def assert_close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)


def make_gbuffer() -> GBuffer:
    rgb = torch.ones((2, 2, 3), dtype=torch.float32)
    position = torch.zeros((2, 2, 3), dtype=torch.float32)
    normal = torch.zeros((2, 2, 3), dtype=torch.float32)
    normal[..., 2] = 1.0
    valid = torch.ones((2, 2), dtype=torch.bool)
    return GBuffer(
        rgb=rgb,
        depth=torch.ones((2, 2), dtype=torch.float32),
        alpha=torch.ones((2, 2), dtype=torch.float32),
        position_cam=position,
        normal_cam=normal,
        valid_mask=valid,
        normal_mask=valid,
    )


def make_lights() -> PointLights:
    return PointLights(
        positions_cam=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 4.0],
            ],
            dtype=torch.float32,
        ),
        colors=torch.ones((4, 3), dtype=torch.float32),
        intensities=torch.ones((4,), dtype=torch.float32),
    )


def test_compute_error_metrics_matches_known_tensor() -> None:
    estimate = torch.tensor([[[2.0, 0.0, 1.0], [4.0, 2.0, 0.0]]], dtype=torch.float32)
    reference = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 2), dtype=torch.bool)

    metrics = compute_error_metrics(estimate, reference, valid)

    errors = torch.tensor([[1.0, -1.0, 0.0], [3.0, 1.0, -1.0]], dtype=torch.float32)
    assert_close(metrics["mae"], float(errors.abs().mean()))
    assert_close(metrics["rmse"], float(torch.sqrt((errors * errors).mean())))
    assert_close(metrics["bias_r"], 2.0)
    assert_close(metrics["bias_g"], 0.0)
    assert_close(metrics["bias_b"], -0.5)
    assert_close(metrics["mean_abs_bias"], (2.0 + 0.0 + 0.5) / 3.0)


def test_compute_error_metrics_excludes_invalid_pixels() -> None:
    estimate = torch.tensor([[[2.0, 0.0, 1.0], [100.0, 100.0, 100.0]]], dtype=torch.float32)
    reference = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    valid = torch.tensor([[True, False]])

    metrics = compute_error_metrics(estimate, reference, valid)

    assert_close(metrics["mae"], (1.0 + 1.0 + 0.0) / 3.0)
    assert_close(metrics["bias_r"], 1.0)
    assert_close(metrics["bias_g"], -1.0)
    assert_close(metrics["bias_b"], 0.0)


def test_run_ris_ablation_returns_expected_row_count_and_schema() -> None:
    rows = run_ris_ablation(make_gbuffer(), make_lights(), k_values=[1, 2], seed_count=3)

    assert len(rows) == 2 * 2 * 3 * 2
    for row in rows:
        assert REQUIRED_KEYS == set(row)
        for key in ["mae", "rmse", "bias_r", "bias_g", "bias_b", "mean_abs_bias"]:
            assert math.isfinite(float(row[key]))


def test_run_ris_ablation_contains_uniform_and_ris_k1_rows() -> None:
    rows = run_ris_ablation(make_gbuffer(), make_lights(), k_values=[1], seed_count=2)

    seen = {(row["estimator"], row["k"], row["reference_quantity"]) for row in rows}
    assert ("uniform", 1, "diffuse_rgb") in seen
    assert ("uniform", 1, "composite_rgb") in seen
    assert ("ris", 1, "diffuse_rgb") in seen
    assert ("ris", 1, "composite_rgb") in seen


def test_summarize_rows_groups_mean_and_std() -> None:
    rows = run_ris_ablation(make_gbuffer(), make_lights(), k_values=[1], seed_count=2)
    summary = summarize_rows(rows)

    assert len(summary) == 4
    for row in summary:
        assert row["sample_count"] == 2
        assert math.isfinite(float(row["mae_mean"]))
        assert math.isfinite(float(row["mae_std"]))
        assert math.isfinite(float(row["rmse_mean"]))
        assert math.isfinite(float(row["rmse_std"]))
