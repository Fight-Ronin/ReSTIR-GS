from __future__ import annotations

import math

import torch

from restir_gs.eval.dxgl_sampling_benchmark import (
    expected_sampling_row_count,
    parse_k_values,
    run_sampling_benchmark_for_frame,
    select_evenly_spaced_frames,
    summarize_sampling_rows,
)
from restir_gs.eval.proposal_ablation import run_proposal_ablation
from restir_gs.lighting.deferred import PointLights, shade_deferred_blinn_phong, shade_deferred_lambertian
from restir_gs.render.gbuffer import GBuffer


def make_gbuffer(valid: bool = True) -> GBuffer:
    rgb = torch.tensor([[[1.0, 0.5, 0.25], [0.2, 0.4, 0.6]]], dtype=torch.float32)
    position = torch.tensor([[[0.0, 0.0, 1.0], [0.25, 0.0, 1.25]]], dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    mask = torch.full((1, 2), valid, dtype=torch.bool)
    return GBuffer(
        rgb=rgb,
        depth=torch.ones((1, 2), dtype=torch.float32),
        alpha=torch.ones((1, 2), dtype=torch.float32),
        position_cam=position,
        normal_cam=normal,
        valid_mask=mask,
        normal_mask=mask,
    )


def make_lights() -> PointLights:
    return PointLights(
        positions_cam=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        colors=torch.tensor([[1.0, 0.8, 0.6], [0.5, 1.0, 0.7], [0.8, 0.6, 1.0]], dtype=torch.float32),
        intensities=torch.tensor([1.0, 0.7, 1.3], dtype=torch.float32),
    )


def test_frame_and_k_parsers() -> None:
    assert parse_k_values("1, 2,4") == [1, 2, 4]
    assert select_evenly_spaced_frames(196, 8) == [0, 24, 49, 73, 98, 122, 147, 171]
    assert select_evenly_spaced_frames(4, 8) == [0, 1, 2, 3]
    assert expected_sampling_row_count(2, [1, 2], 3) == 2 * 2 * 2 * 2 * 2 * 3 * 2


def test_sampling_row_count_schema_and_finite_summary() -> None:
    gbuffer = make_gbuffer()
    lights = make_lights()
    lambertian = shade_deferred_lambertian(gbuffer, lights)
    blinn = shade_deferred_blinn_phong(gbuffer, lights)

    rows = run_sampling_benchmark_for_frame(
        gbuffer,
        lights,
        lambertian,
        blinn,
        frame_index=7,
        k_values=[1, 2],
        seed_count=2,
    )
    summary = summarize_sampling_rows(rows)

    assert len(rows) == expected_sampling_row_count(1, [1, 2], 2)
    assert len(summary) == 2 * 2 * 2 * 2 * 2
    for row in rows:
        assert {
            "frame_index",
            "target_mode",
            "proposal",
            "estimator",
            "k",
            "seed_index",
            "candidate_seed",
            "selection_seed",
            "reference_quantity",
            "valid_pixels",
            "rgb_mae_to_reference",
            "alpha_iou",
            "mae",
            "rmse",
            "bias_r",
            "bias_g",
            "bias_b",
            "mean_abs_bias",
        } == set(row)
        for key in ["mae", "rmse", "bias_r", "bias_g", "bias_b", "mean_abs_bias"]:
            assert math.isfinite(float(row[key]))


def test_diffuse_target_rows_match_existing_proposal_ablation() -> None:
    gbuffer = make_gbuffer()
    lights = make_lights()
    lambertian = shade_deferred_lambertian(gbuffer, lights)
    blinn = shade_deferred_blinn_phong(gbuffer, lights)

    old_rows = run_proposal_ablation(gbuffer, lights, k_values=[1, 2], seed_count=2, candidate_seed_base=15100, selection_seed_base=16100)
    new_rows = run_sampling_benchmark_for_frame(
        gbuffer,
        lights,
        lambertian,
        blinn,
        frame_index=0,
        k_values=[1, 2],
        seed_count=2,
        candidate_seed_base=15100,
        selection_seed_base=16100,
    )
    diffuse_rows = [row for row in new_rows if row["target_mode"] == "diffuse"]

    def key(row: dict[str, int | float | str]) -> tuple[str, str, int, int, str]:
        quantity = "diffuse_rgb" if row["reference_quantity"] == "contribution_rgb" else str(row["reference_quantity"])
        return (str(row["proposal"]), str(row["estimator"]), int(row["k"]), int(row["seed_index"]), quantity)

    old_by_key = {key(row): row for row in old_rows}
    for row in diffuse_rows:
        old = old_by_key[key(row)]
        assert math.isclose(float(row["mae"]), float(old["mae"]), rel_tol=0.0, abs_tol=1e-7)
        assert math.isclose(float(row["rmse"]), float(old["rmse"]), rel_tol=0.0, abs_tol=1e-7)


def test_blinn_reference_and_zero_specular_match_diffuse() -> None:
    gbuffer = make_gbuffer()
    lights = make_lights()
    lambertian = shade_deferred_lambertian(gbuffer, lights)
    blinn = shade_deferred_blinn_phong(gbuffer, lights, specular_strength=0.0)
    assert torch.allclose(blinn.diffuse_rgb + blinn.specular_rgb, lambertian.diffuse_rgb)

    rows = run_sampling_benchmark_for_frame(
        gbuffer,
        lights,
        lambertian,
        blinn,
        frame_index=0,
        k_values=[1],
        seed_count=1,
        specular_strength=0.0,
    )
    diffuse_rows = [row for row in rows if row["target_mode"] == "diffuse"]
    blinn_rows = [row for row in rows if row["target_mode"] == "blinn_phong"]

    def key(row: dict[str, int | float | str]) -> tuple[str, str, int, int, str]:
        return (str(row["proposal"]), str(row["estimator"]), int(row["k"]), int(row["seed_index"]), str(row["reference_quantity"]))

    blinn_by_key = {key(row): row for row in blinn_rows}
    for row in diffuse_rows:
        matched = blinn_by_key[key(row)]
        assert math.isclose(float(row["mae"]), float(matched["mae"]), rel_tol=0.0, abs_tol=1e-7)


def test_no_valid_gbuffer_pixels_fail_loudly() -> None:
    gbuffer = make_gbuffer(valid=False)
    lights = make_lights()
    lambertian = shade_deferred_lambertian(gbuffer, lights)
    blinn = shade_deferred_blinn_phong(gbuffer, lights)

    try:
        run_sampling_benchmark_for_frame(gbuffer, lights, lambertian, blinn, frame_index=3, k_values=[1], seed_count=1)
    except RuntimeError as exc:
        assert "no valid G-buffer pixels" in str(exc)
    else:
        raise AssertionError("Expected invalid G-buffer frame to fail.")
