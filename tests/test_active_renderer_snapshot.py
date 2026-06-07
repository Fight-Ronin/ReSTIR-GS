from __future__ import annotations

from pathlib import Path

from PIL import Image

from restir_gs.eval.active_renderer_snapshot import (
    build_active_renderer_snapshot_summary,
    make_active_renderer_contact_sheet,
    rank_gpu_timing_stages,
    summarize_estimator_contribution_rows,
)


def test_estimator_contribution_summary_uses_contribution_rows_only() -> None:
    rows = [
        {"estimator": "initial_ris", "reference_quantity": "contribution_rgb", "mae": "1.0", "rmse": "2.0"},
        {"estimator": "initial_ris", "reference_quantity": "composite_rgb", "mae": "99.0", "rmse": "99.0"},
        {"estimator": "initial_ris", "reference_quantity": "contribution_rgb", "mae": "3.0", "rmse": "4.0"},
        {"estimator": "temporal_filtered_ris", "reference_quantity": "contribution_rgb", "mae": "0.5", "rmse": "1.5"},
    ]

    summary = summarize_estimator_contribution_rows(rows)

    assert summary["initial_ris"] == {"count": 2, "mae_mean": 2.0, "mae_max": 3.0, "rmse_mean": 3.0}
    assert summary["temporal_filtered_ris"] == {"count": 1, "mae_mean": 0.5, "mae_max": 0.5, "rmse_mean": 1.5}


def test_gpu_timing_rank_orders_by_mean_ms() -> None:
    timing = {
        "frame_gpu_ms": {"mean": 10.0, "max": 10.0, "count": 2},
        "proposal_gpu_ms": {"mean": 4.0, "max": 5.0, "count": 2},
        "initial_ris_gpu_ms": {"mean": 6.0, "max": 7.0, "count": 2},
    }

    ranked = rank_gpu_timing_stages(timing)

    assert ranked[0]["stage"] == "initial_ris_gpu_ms"
    assert ranked[0]["frame_fraction"] == 0.6
    assert ranked[1]["stage"] == "proposal_gpu_ms"


def test_snapshot_summary_records_active_policy_and_validation_flags() -> None:
    renderer_summary = {
        "row_count": 72,
        "asset_ids": ["asset_a"],
        "all_numeric_finite": True,
        "settings": {
            "target_mode": "visibility",
            "proposal": "visibility_geometric",
            "num_lights": 16,
            "temporal_reprojection_search_radius": 1,
            "temporal_history_m_cap": 1,
        },
        "timing_summary": {"frame_gpu_ms": {"mean": 1.0, "max": 1.0, "count": 1}},
        "asset_timing": {"asset_a": {}},
    }
    rows = [{"estimator": "initial_ris", "reference_quantity": "contribution_rgb", "mae": "1.0", "rmse": "1.0"}]

    snapshot = build_active_renderer_snapshot_summary(renderer_summary, rows)

    assert snapshot["active_policy"]["target_mode"] == "visibility"
    assert snapshot["active_policy"]["preferred_output"] == "temporal_filtered_ris"
    assert snapshot["validation"]["row_count_matches_active_default"] is True
    assert snapshot["validation"]["has_timing_summary"] is True
    assert snapshot["validation"]["has_asset_timing"] is True


def test_contact_sheet_writes_output_for_tiny_fixture(tmp_path: Path) -> None:
    asset_dir = tmp_path / "aligned_restir" / "asset_a"
    asset_dir.mkdir(parents=True)
    for filename in (
        "final_reference.png",
        "final_initial_ris.png",
        "final_temporal_filtered_ris.png",
        "final_temporal_filtered_abs_error.png",
        "final_temporal_filter_alpha.png",
        "final_reuse_mask.png",
    ):
        Image.new("RGB", (8, 8), (32, 64, 128)).save(asset_dir / filename)

    output = tmp_path / "snapshot.png"
    make_active_renderer_contact_sheet(tmp_path / "aligned_restir", output, ["asset_a"])

    assert output.is_file()
    assert output.stat().st_size > 0
