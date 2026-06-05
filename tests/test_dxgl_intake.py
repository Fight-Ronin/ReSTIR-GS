from __future__ import annotations

import json
import sys
import zipfile

from PIL import Image
import pytest
import torch

from restir_gs.render.transforms_loader import (
    load_nerfstudio_transforms,
    opengl_c2w_to_project_viewmat,
)
from scripts.demo_17_dxgl_aligned_intake import (
    make_dxgl_contact_sheet,
    parse_int_list,
    probe_points3d_compatibility,
)
from scripts.download_dxgl_apple import (
    DXGL_APPLE_URL,
    find_dxgl_dataset_root,
    main as download_main,
    plan_dxgl_apple,
    validate_dxgl_dataset_root,
)


def test_dxgl_download_dry_run_writes_no_files(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["download_dxgl_apple.py", "--dry-run"])

    assert download_main() == 0
    captured = capsys.readouterr()

    assert DXGL_APPLE_URL in captured.out
    assert "dry-run: no files written" in captured.out
    assert not (tmp_path / "outputs").exists()


def test_dxgl_plan_reports_existing_valid_dataset(tmp_path) -> None:
    root = _make_minimal_dxgl_dataset(tmp_path / "apple")

    plan = plan_dxgl_apple(extract_dir=root)

    assert plan.dataset_root == root
    assert plan.dataset_valid is True


def test_dxgl_validator_fails_loudly_when_required_entries_missing(tmp_path) -> None:
    root = tmp_path / "apple"
    root.mkdir()
    (root / "transforms.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing required entries"):
        validate_dxgl_dataset_root(root, raise_on_missing=True)


def test_find_dxgl_dataset_root_handles_one_nested_folder(tmp_path) -> None:
    root = _make_minimal_dxgl_dataset(tmp_path / "extract" / "nested")

    assert find_dxgl_dataset_root(tmp_path / "extract") == root


def test_transforms_parser_handles_global_intrinsics_and_frame_paths(tmp_path) -> None:
    root = _make_minimal_dxgl_dataset(tmp_path / "apple")
    _write_transforms(root)

    imported = load_nerfstudio_transforms(root / "transforms.json", dataset_root=root, device="cpu")
    frame = imported.frames[0]

    assert imported.frame_count == 1
    assert frame.image_path == root / "images" / "frame_000.png"
    assert frame.mask_path == root / "masks" / "frame_000.png"
    assert frame.depth_path == root / "depth" / "frame_000.png"
    assert frame.depth_16bit_path == root / "depth_16bit" / "frame_000.png"
    assert frame.normal_path == root / "normals" / "frame_000.png"
    assert frame.camera.width == 20
    assert frame.camera.height == 22
    assert tuple(frame.camera.viewmats.shape) == (1, 4, 4)
    assert tuple(frame.camera.intrinsics.shape) == (1, 3, 3)
    assert frame.camera.intrinsics[0, 0, 0].item() == pytest.approx(10.0)
    assert frame.camera.intrinsics[0, 1, 1].item() == pytest.approx(11.0)


def test_opengl_camera_conversion_maps_front_point_to_positive_project_z() -> None:
    viewmat = opengl_c2w_to_project_viewmat(torch.eye(4), device="cpu")
    point_in_front_opengl = torch.tensor([0.0, 0.0, -1.0, 1.0])

    point_project = point_in_front_opengl @ viewmat.T

    assert point_project[2].item() > 0.0


def test_contact_sheet_generation_handles_tiny_modalities(tmp_path) -> None:
    root = _make_minimal_dxgl_dataset(tmp_path / "apple")
    _write_transforms(root)
    imported = load_nerfstudio_transforms(root / "transforms.json", dataset_root=root, device="cpu")
    output_path = tmp_path / "contact.png"

    make_dxgl_contact_sheet([imported.frames[0]], output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_points3d_probe_records_incompatibility_without_crashing(tmp_path) -> None:
    ply_path = tmp_path / "points3D.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
            ]
        ),
        encoding="utf-8",
    )

    result = probe_points3d_compatibility(ply_path)

    assert result["exists"] is True
    assert result["splat_compatible"] is False
    assert "missing required fields" in result["error"]


def test_parse_int_list_returns_expected_values() -> None:
    assert parse_int_list("0, 49,98") == [0, 49, 98]
    with pytest.raises(ValueError):
        parse_int_list(" , ")


def test_zip_validator_can_extract_minimal_fixture(tmp_path) -> None:
    root = _make_minimal_dxgl_dataset(tmp_path / "source")
    _write_transforms(root)
    zip_path = tmp_path / "apple.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(root))

    extract_dir = tmp_path / "extract"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    found_root = find_dxgl_dataset_root(extract_dir)
    validation = validate_dxgl_dataset_root(found_root)

    assert validation["valid"] is True


def _make_minimal_dxgl_dataset(root):
    root.mkdir(parents=True, exist_ok=True)
    for directory in ["images", "depth", "depth_16bit", "normals", "masks"]:
        (root / directory).mkdir()
    Image.new("RGB", (4, 4), (128, 80, 40)).save(root / "images" / "frame_000.png")
    Image.new("L", (4, 4), 255).save(root / "masks" / "frame_000.png")
    Image.new("L", (4, 4), 80).save(root / "depth" / "frame_000.png")
    Image.new("I;16", (4, 4), 1024).save(root / "depth_16bit" / "frame_000.png")
    Image.new("RGB", (4, 4), (128, 128, 255)).save(root / "normals" / "frame_000.png")
    (root / "points3D.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
            ]
        ),
        encoding="utf-8",
    )
    _write_transforms(root)
    return root


def _write_transforms(root) -> None:
    data = {
        "fl_x": 10.0,
        "fl_y": 11.0,
        "cx": 5.0,
        "cy": 6.0,
        "w": 20,
        "h": 22,
        "frames": [
            {
                "file_path": "images/frame_000.png",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    (root / "transforms.json").write_text(json.dumps(data), encoding="utf-8")
