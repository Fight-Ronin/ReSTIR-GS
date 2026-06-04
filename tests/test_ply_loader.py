from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from restir_gs.render.ply_loader import (
    SH_C0,
    load_gaussian_ply,
    load_gaussian_ply_with_stats,
    make_asset_camera,
)
from restir_gs.render.synthetic_scene import SyntheticGaussians


def write_ply(path: Path, properties: list[tuple[str, str]], rows: list[list[float | int]]) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(rows)}",
    ]
    lines.extend(f"property {kind} {name}" for kind, name in properties)
    lines.append("end_header")
    lines.extend(" ".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def graphdeco_properties() -> list[tuple[str, str]]:
    return [
        ("float", "x"),
        ("float", "y"),
        ("float", "z"),
        ("float", "opacity"),
        ("float", "scale_0"),
        ("float", "scale_1"),
        ("float", "scale_2"),
        ("float", "rot_0"),
        ("float", "rot_1"),
        ("float", "rot_2"),
        ("float", "rot_3"),
        ("float", "f_dc_0"),
        ("float", "f_dc_1"),
        ("float", "f_dc_2"),
    ]


def test_load_graphdeco_schema_applies_3dgs_transforms(tmp_path: Path) -> None:
    path = tmp_path / "graphdeco.ply"
    write_ply(
        path,
        graphdeco_properties(),
        [
            [0.0, 0.0, 2.0, 0.0, 0.0, math.log(2.0), math.log(3.0), 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 3.0, 2.0, math.log(4.0), 0.0, math.log(0.5), 0.0, 0.0, 0.0, 2.0, 1.0, -1.0, 2.0],
        ],
    )

    loaded = load_gaussian_ply_with_stats(path, device="cpu")
    scene = loaded.scene

    assert isinstance(load_gaussian_ply(path, device="cpu"), SyntheticGaussians)
    assert loaded.stats.original_count == 2
    assert loaded.stats.loaded_count == 2
    assert loaded.stats.color_source == "f_dc"
    assert torch.allclose(scene.means, torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]]))
    assert torch.allclose(scene.scales, torch.tensor([[1.0, 2.0, 3.0], [4.0, 1.0, 0.5]]), atol=1e-6)
    assert torch.allclose(scene.opacities, torch.sigmoid(torch.tensor([0.0, 2.0])), atol=1e-6)
    assert torch.allclose(scene.quats, torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))
    expected_colors = torch.tensor(
        [
            [0.5, 0.5, 0.5],
            [0.5 + SH_C0, 0.5 - SH_C0, 1.0],
        ]
    )
    assert torch.allclose(scene.colors, expected_colors, atol=1e-6)


def test_load_rgb_schema_fallback_normalizes_uchar_colors(tmp_path: Path) -> None:
    path = tmp_path / "rgb.ply"
    properties = [
        ("float", "x"),
        ("float", "y"),
        ("float", "z"),
        ("float", "opacity"),
        ("float", "scale_0"),
        ("float", "scale_1"),
        ("float", "scale_2"),
        ("float", "rot_0"),
        ("float", "rot_1"),
        ("float", "rot_2"),
        ("float", "rot_3"),
        ("uchar", "red"),
        ("uchar", "green"),
        ("uchar", "blue"),
    ]
    write_ply(path, properties, [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 255, 128, 0]])

    loaded = load_gaussian_ply_with_stats(path, device="cpu")

    assert loaded.stats.color_source == "red_green_blue"
    assert torch.allclose(loaded.scene.colors, torch.tensor([[1.0, 128.0 / 255.0, 0.0]]), atol=1e-6)


def test_load_missing_required_fields_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "missing.ply"
    properties = [prop for prop in graphdeco_properties() if prop[1] != "opacity"]
    write_ply(path, properties, [[0.0] * len(properties)])

    with pytest.raises(ValueError, match="missing required fields"):
        load_gaussian_ply_with_stats(path, device="cpu")


def test_deterministic_even_subsample_records_original_and_loaded_counts(tmp_path: Path) -> None:
    path = tmp_path / "subsample.ply"
    rows = []
    for index in range(5):
        rows.append([float(index), 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    write_ply(path, graphdeco_properties(), rows)

    loaded = load_gaussian_ply_with_stats(path, device="cpu", max_gaussians=3)

    assert loaded.stats.original_count == 5
    assert loaded.stats.loaded_count == 3
    assert torch.allclose(loaded.scene.means[:, 0], torch.tensor([0.0, 2.0, 4.0]))


def test_make_asset_camera_places_bbox_center_at_positive_camera_z(tmp_path: Path) -> None:
    path = tmp_path / "camera.ply"
    write_ply(
        path,
        graphdeco_properties(),
        [
            [-1.0, -0.5, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    scene = load_gaussian_ply(path, device="cpu")

    camera, info = make_asset_camera(scene.means, width=64, height=48)
    target = torch.tensor(info.target, dtype=torch.float32)
    target_cam = torch.cat((target, torch.ones(1))) @ camera.viewmats[0].T

    assert camera.width == 64
    assert camera.height == 48
    assert camera.viewmats.shape == (1, 4, 4)
    assert camera.intrinsics.shape == (1, 3, 3)
    assert target_cam[2] > 0.0
    assert info.bbox_diagonal > 0.0
    assert info.radius > info.bbox_diagonal
    assert info.bbox_percentile == 1.0


def test_make_asset_camera_percentile_bbox_reduces_outlier_radius() -> None:
    core = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.2, 0.0, 2.1],
            [-0.2, 0.0, 2.2],
            [0.0, 0.2, 2.1],
            [0.0, -0.2, 2.1],
        ],
        dtype=torch.float32,
    )
    outlier = torch.tensor([[50.0, 0.0, 2.0]], dtype=torch.float32)
    means = torch.cat((core, outlier), dim=0)

    _, full_info = make_asset_camera(means, width=64, height=64, bbox_percentile=1.0)
    _, robust_info = make_asset_camera(means, width=64, height=64, bbox_percentile=0.8)

    assert robust_info.bbox_percentile == 0.8
    assert robust_info.radius < full_info.radius
    assert robust_info.bbox_diagonal < full_info.bbox_diagonal


def test_make_asset_camera_rejects_invalid_bbox_percentile() -> None:
    means = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="bbox_percentile"):
        make_asset_camera(means, bbox_percentile=0.0)
