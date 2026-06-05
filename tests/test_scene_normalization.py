from __future__ import annotations

import pytest
import torch

from restir_gs.render.scene_normalization import (
    apply_scene_normalization_to_c2w,
    infer_scene_normalization_from_plys,
    normalization_rotation_matrix,
    robust_bbox,
)
from restir_gs.render.transforms_loader import load_nerfstudio_transforms


def test_robust_bbox_rejects_invalid_percentile() -> None:
    xyz = torch.zeros((2, 3)).numpy()

    with pytest.raises(ValueError):
        robust_bbox(xyz, 0.0)


def test_infer_scene_normalization_from_ply_bboxes(tmp_path) -> None:
    raw_path = tmp_path / "raw.ply"
    target_path = tmp_path / "target.ply"
    _write_xyz_ply(raw_path, [(0.0, 1.0, 0.0), (0.0, 3.0, 0.0)])
    _write_xyz_ply(target_path, [(0.0, -1.0, 0.0), (0.0, 0.0, 0.0)])

    normalization = infer_scene_normalization_from_plys(raw_path, target_path, bbox_percentile=1.0)

    assert normalization.raw_center == pytest.approx((0.0, 2.0, 0.0))
    assert normalization.target_center == pytest.approx((0.0, -0.5, 0.0))
    assert normalization.scale == pytest.approx(0.5)


def test_apply_scene_normalization_to_c2w_changes_translation_only(tmp_path) -> None:
    raw_path = tmp_path / "raw.ply"
    target_path = tmp_path / "target.ply"
    _write_xyz_ply(raw_path, [(0.0, 1.0, 0.0), (0.0, 3.0, 0.0)])
    _write_xyz_ply(target_path, [(0.0, -1.0, 0.0), (0.0, 0.0, 0.0)])
    normalization = infer_scene_normalization_from_plys(raw_path, target_path, bbox_percentile=1.0)
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([4.0, 2.0, 6.0])

    normalized = apply_scene_normalization_to_c2w(c2w, normalization, device="cpu")

    assert torch.equal(normalized[:3, :3], c2w[:3, :3])
    assert normalized[:3, 3].tolist() == pytest.approx([2.0, -0.5, 3.0])


def test_raw_y_to_z_up_rotation_maps_camera_center_and_axes(tmp_path) -> None:
    raw_path = tmp_path / "raw.ply"
    target_path = tmp_path / "target.ply"
    _write_xyz_ply(raw_path, [(0.0, 1.0, 0.0), (0.0, 3.0, 0.0)])
    _write_xyz_ply(target_path, [(0.0, 0.0, -1.0), (0.0, 0.0, 0.0)])
    normalization = infer_scene_normalization_from_plys(
        raw_path,
        target_path,
        bbox_percentile=1.0,
        raw_to_target_rotation="raw_y_to_z_up",
    )
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([0.0, 4.0, 0.0])

    normalized = apply_scene_normalization_to_c2w(c2w, normalization, device="cpu")

    assert normalized[:3, 1].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert normalized[:3, 3].tolist() == pytest.approx([0.0, 0.0, 0.5])


def test_normalization_rotation_matrix_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown normalization rotation"):
        normalization_rotation_matrix("sideways")


def test_transforms_loader_applies_camera_normalization(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images").mkdir(parents=True)
    (dataset_root / "images" / "frame_000.png").write_bytes(b"placeholder")
    transforms = {
        "fl_x": 10.0,
        "fl_y": 10.0,
        "cx": 5.0,
        "cy": 5.0,
        "w": 10,
        "h": 10,
        "frames": [
            {
                "file_path": "images/frame_000.png",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    (dataset_root / "transforms.json").write_text(__import__("json").dumps(transforms), encoding="utf-8")
    raw_path = tmp_path / "raw.ply"
    target_path = tmp_path / "target.ply"
    _write_xyz_ply(raw_path, [(0.0, 1.0, 0.0), (0.0, 3.0, 0.0)])
    _write_xyz_ply(target_path, [(0.0, -1.0, 0.0), (0.0, 0.0, 0.0)])
    normalization = infer_scene_normalization_from_plys(raw_path, target_path, bbox_percentile=1.0)

    imported = load_nerfstudio_transforms(
        dataset_root / "transforms.json",
        dataset_root=dataset_root,
        camera_normalization=normalization,
    )
    viewmat = imported.frames[0].camera.viewmats[0]

    camera_center = -viewmat[:3, :3].T @ viewmat[:3, 3]
    assert camera_center.tolist() == pytest.approx([0.0, -0.5, 0.0], abs=1e-6)


def _write_xyz_ply(path, points) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend(f"{x} {y} {z}" for x, y, z in points)
    path.write_text("\n".join(lines), encoding="utf-8")
