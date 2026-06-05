from __future__ import annotations

import json

from PIL import Image
import pytest
import torch

from restir_gs.eval.gbuffer_validation import binary_mask_metrics, depth_metrics, masked_rgb_metrics
from restir_gs.render.dxgl_asset import load_dxgl_frame_modalities, scale_camera
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.render.transforms_loader import load_nerfstudio_transforms


def test_dxgl_frame_modalities_use_mask_luminance_not_opaque_alpha(tmp_path) -> None:
    root = _make_tiny_aligned_dataset(tmp_path)
    imported = load_nerfstudio_transforms(root / "transforms.json", dataset_root=root)

    modalities = load_dxgl_frame_modalities(imported.frames[0], width=2, height=2, scene_scale=0.25)

    assert modalities.rgb.shape == (2, 2, 3)
    assert modalities.mask.tolist() == [[True, False], [False, True]]
    assert torch.allclose(modalities.depth_raw, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.allclose(modalities.depth_normalized, torch.tensor([[0.25, 0.5], [0.75, 1.0]]))
    assert modalities.normal_rgb.shape == (2, 2, 3)


def test_scale_camera_scales_intrinsics() -> None:
    camera = PinholeCamera(
        viewmats=torch.eye(4)[None],
        intrinsics=torch.tensor([[[100.0, 0.0, 50.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]]]),
        width=100,
        height=120,
    )

    scaled = scale_camera(camera, width=50, height=30)

    assert scaled.intrinsics[0, 0, 0].item() == pytest.approx(50.0)
    assert scaled.intrinsics[0, 0, 2].item() == pytest.approx(25.0)
    assert scaled.intrinsics[0, 1, 1].item() == pytest.approx(50.0)
    assert scaled.intrinsics[0, 1, 2].item() == pytest.approx(15.0)


def test_binary_mask_metrics_known_values() -> None:
    estimate = torch.tensor([[True, True], [False, False]])
    reference = torch.tensor([[True, False], [True, False]])

    metrics = binary_mask_metrics(estimate, reference)

    assert metrics["intersection_pixels"] == 1
    assert metrics["union_pixels"] == 3
    assert metrics["iou"] == pytest.approx(1.0 / 3.0)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_depth_metrics_known_values() -> None:
    estimate = torch.tensor([[1.0, 3.0], [10.0, 0.0]])
    reference = torch.tensor([[2.0, 1.0], [10.0, 1.0]])
    mask = torch.tensor([[True, True], [False, True]])

    metrics = depth_metrics(estimate, reference, mask)

    assert metrics["valid_pixels"] == 2
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx((2.5) ** 0.5)
    assert metrics["abs_rel"] == pytest.approx((0.5 + 2.0) / 2.0)


def test_masked_rgb_metrics_empty_mask_is_finite() -> None:
    estimate = torch.zeros((1, 1, 3))
    reference = torch.ones((1, 1, 3))
    mask = torch.zeros((1, 1), dtype=torch.bool)

    assert masked_rgb_metrics(estimate, reference, mask) == {
        "valid_pixels": 0,
        "mae": 0.0,
        "rmse": 0.0,
        "psnr": 0.0,
    }


def test_transforms_loader_uses_explicit_modality_paths(tmp_path) -> None:
    root = _make_tiny_aligned_dataset(tmp_path)
    imported = load_nerfstudio_transforms(root / "transforms.json", dataset_root=root)
    frame = imported.frames[0]

    assert frame.depth_path == root / "custom_depth" / "d.png"
    assert frame.normal_path == root / "custom_normals" / "n.png"
    assert frame.mask_path == root / "custom_masks" / "m.png"


def _make_tiny_aligned_dataset(tmp_path):
    root = tmp_path / "asset"
    for directory in ["images", "custom_depth", "depth_16bit", "custom_normals", "custom_masks"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), (128, 64, 32)).save(root / "images" / "frame.png")
    Image.fromarray(__import__("numpy").array([[255, 0], [0, 255]], dtype="uint8"), mode="L").convert("RGBA").save(
        root / "custom_masks" / "m.png"
    )
    depth = __import__("numpy").array([[10000, 20000], [30000, 40000]], dtype="uint16")
    Image.fromarray(depth, mode="I;16").save(root / "depth_16bit" / "frame.png")
    Image.fromarray(depth, mode="I;16").save(root / "custom_depth" / "d.png")
    Image.new("RGB", (2, 2), (128, 128, 255)).save(root / "custom_normals" / "n.png")
    transforms = {
        "fl_x": 10.0,
        "fl_y": 10.0,
        "cx": 1.0,
        "cy": 1.0,
        "w": 2,
        "h": 2,
        "frames": [
            {
                "file_path": "images/frame.png",
                "depth_file_path": "custom_depth/d.png",
                "normal_file_path": "custom_normals/n.png",
                "mask_file_path": "custom_masks/m.png",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    (root / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    return root
