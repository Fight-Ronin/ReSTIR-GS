from __future__ import annotations

import json
from pathlib import Path

from gs_gen.tests.fixtures import write_dataset
from gs_gen.transforms_probe import validate_transforms


def test_validate_transforms_accepts_minimal_nerfstudio_dataset(tmp_path: Path) -> None:
    root = write_dataset(tmp_path)

    result = validate_transforms(root / "transforms.json", dataset_root=root)

    assert result["valid"] is True
    assert result["frame_count"] == 1
    assert result["missing_image_count"] == 0


def test_validate_transforms_reports_missing_images(tmp_path: Path) -> None:
    root = write_dataset(tmp_path, write_image=False)

    result = validate_transforms(root / "transforms.json", dataset_root=root)

    assert result["valid"] is False
    assert result["missing_image_count"] == 1


def test_validate_transforms_requires_intrinsics(tmp_path: Path) -> None:
    root = write_dataset(tmp_path)
    data = json.loads((root / "transforms.json").read_text(encoding="utf-8"))
    data.pop("fl_x")
    data.pop("camera_angle_x", None)
    (root / "transforms.json").write_text(json.dumps(data), encoding="utf-8")

    result = validate_transforms(root / "transforms.json", dataset_root=root)

    assert result["valid"] is False
    assert "missing fl_x or camera_angle_x globally or on every frame" in result["errors"]


def test_validate_transforms_accepts_frame_level_focal(tmp_path: Path) -> None:
    root = write_dataset(tmp_path)
    data = json.loads((root / "transforms.json").read_text(encoding="utf-8"))
    focal = data.pop("fl_x")
    data["frames"][0]["fl_x"] = focal
    (root / "transforms.json").write_text(json.dumps(data), encoding="utf-8")

    result = validate_transforms(root / "transforms.json", dataset_root=root)

    assert result["valid"] is True
