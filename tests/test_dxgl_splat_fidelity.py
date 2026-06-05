from __future__ import annotations

import sys

import pytest
import torch

from restir_gs.render.synthetic_scene import PinholeCamera
from scripts.demo_18_dxgl_splat_fidelity import compute_masked_rgb_metrics, scale_camera
from scripts.download_dxgl_apple_splat import (
    DXGL_APPLE_SPLAT_URL,
    main as download_splat_main,
    plan_dxgl_apple_splat,
    validate_dxgl_splat_file,
)


def test_dxgl_splat_dry_run_writes_no_files(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["download_dxgl_apple_splat.py", "--dry-run"])

    assert download_splat_main() == 0
    captured = capsys.readouterr()

    assert DXGL_APPLE_SPLAT_URL in captured.out
    assert "dry-run: no files written" in captured.out
    assert not (tmp_path / "outputs").exists()


def test_dxgl_splat_plan_skips_existing_non_empty_file(tmp_path) -> None:
    splat_path = tmp_path / "apple.ply"
    splat_path.write_bytes(b"ply\n")

    plan = plan_dxgl_apple_splat(splat_path=splat_path)

    assert plan.exists is True
    assert plan.size_bytes == 4


def test_validate_dxgl_splat_file_accepts_tiny_3dgs_fixture(tmp_path) -> None:
    splat_path = tmp_path / "tiny_3dgs.ply"
    _write_tiny_3dgs_ply(splat_path)

    validation = validate_dxgl_splat_file(splat_path)

    assert validation["valid"] is True
    assert validation["original_count"] == 1
    assert validation["color_source"] == "f_dc"


def test_scale_camera_scales_intrinsics_without_changing_viewmat() -> None:
    camera = PinholeCamera(
        viewmats=torch.eye(4)[None],
        intrinsics=torch.tensor([[[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]]),
        width=100,
        height=120,
    )

    scaled = scale_camera(camera, width=50, height=30)

    assert scaled.width == 50
    assert scaled.height == 30
    assert torch.equal(scaled.viewmats, camera.viewmats)
    assert scaled.intrinsics[0, 0, 0].item() == pytest.approx(50.0)
    assert scaled.intrinsics[0, 0, 2].item() == pytest.approx(25.0)
    assert scaled.intrinsics[0, 1, 1].item() == pytest.approx(30.0)
    assert scaled.intrinsics[0, 1, 2].item() == pytest.approx(15.0)


def test_compute_masked_rgb_metrics_matches_known_values() -> None:
    estimate = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    reference = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
    mask = torch.tensor([[True, False]])

    metrics = compute_masked_rgb_metrics(estimate, reference, mask)

    assert metrics["valid_pixels"] == 1
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse"] == pytest.approx((1.0 / 3.0) ** 0.5)
    assert metrics["psnr"] > 0.0


def test_compute_masked_rgb_metrics_handles_empty_mask() -> None:
    estimate = torch.zeros((1, 1, 3))
    reference = torch.ones((1, 1, 3))
    mask = torch.zeros((1, 1), dtype=torch.bool)

    metrics = compute_masked_rgb_metrics(estimate, reference, mask)

    assert metrics == {"valid_pixels": 0, "mae": 0.0, "rmse": 0.0, "psnr": 0.0}


def _write_tiny_3dgs_ply(path) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property float opacity",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
                "property float f_dc_0",
                "property float f_dc_1",
                "property float f_dc_2",
                "end_header",
                "0 0 1 2 -2 -2 -2 1 0 0 0 0 0 0",
            ]
        ),
        encoding="utf-8",
    )
