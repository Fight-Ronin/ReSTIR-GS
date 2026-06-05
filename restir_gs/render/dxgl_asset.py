from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from restir_gs.render.ply_loader import LoadedGaussianAsset, load_gaussian_asset
from restir_gs.render.scene_normalization import (
    SceneNormalization,
    infer_scene_normalization_from_plys,
)
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.render.transforms_loader import ImportedTransformFrame, ImportedTransforms, load_nerfstudio_transforms


@dataclass(frozen=True)
class DxglAlignedAsset:
    dataset_root: Path
    splat_path: Path
    loaded: LoadedGaussianAsset
    transforms: ImportedTransforms
    normalization: SceneNormalization | None


@dataclass(frozen=True)
class DxglFrameModalities:
    rgb: torch.Tensor
    mask: torch.Tensor
    depth_raw: torch.Tensor | None
    depth_normalized: torch.Tensor | None
    normal_rgb: torch.Tensor | None


def load_dxgl_aligned_asset(
    dataset_root: str | Path,
    splat_path: str | Path,
    device: torch.device | str = "cuda",
    max_gaussians: int | None = None,
    gaussian_schema: str = "auto",
    camera_normalization: str = "inferred_from_points3d",
    normalization_rotation: str = "raw_y_to_z_up",
    normalization_bbox_percentile: float = 0.98,
) -> DxglAlignedAsset:
    root = Path(dataset_root)
    splat = Path(splat_path)
    normalization = None
    if camera_normalization == "inferred_from_points3d":
        normalization = infer_scene_normalization_from_plys(
            root / "points3D.ply",
            splat,
            bbox_percentile=normalization_bbox_percentile,
            raw_to_target_rotation=normalization_rotation,
        )
    elif camera_normalization != "none":
        raise ValueError(f"Unsupported camera_normalization mode: {camera_normalization}")
    loaded = load_gaussian_asset(splat, device=device, max_gaussians=max_gaussians, schema=gaussian_schema)
    transforms = load_nerfstudio_transforms(
        root / "transforms.json",
        dataset_root=root,
        device=device,
        camera_normalization=normalization,
    )
    return DxglAlignedAsset(
        dataset_root=root,
        splat_path=splat,
        loaded=loaded,
        transforms=transforms,
        normalization=normalization,
    )


def scale_camera(camera: PinholeCamera, width: int, height: int) -> PinholeCamera:
    if width <= 0 or height <= 0:
        raise ValueError(f"Expected positive output size, got {width}x{height}")
    sx = float(width) / float(camera.width)
    sy = float(height) / float(camera.height)
    intrinsics = camera.intrinsics.clone()
    intrinsics[:, 0, :] *= sx
    intrinsics[:, 1, :] *= sy
    return PinholeCamera(viewmats=camera.viewmats.clone(), intrinsics=intrinsics, width=width, height=height)


def load_dxgl_frame_modalities(
    frame: ImportedTransformFrame,
    width: int,
    height: int,
    depth_unit_scale: float = 10000.0,
    scene_scale: float | None = None,
) -> DxglFrameModalities:
    rgb = _load_rgb_tensor(frame.image_path, width, height)
    mask = _load_mask_tensor(frame.mask_path, frame.image_path, width, height)
    depth_raw = _load_depth_tensor(frame.depth_16bit_path or frame.depth_path, width, height, depth_unit_scale)
    depth_normalized = None
    if depth_raw is not None and scene_scale is not None:
        depth_normalized = depth_raw * float(scene_scale)
    normal_rgb = _load_rgb_tensor(frame.normal_path, width, height) if frame.normal_path is not None else None
    return DxglFrameModalities(
        rgb=rgb,
        mask=mask,
        depth_raw=depth_raw,
        depth_normalized=depth_normalized,
        normal_rgb=normal_rgb,
    )


def _load_rgb_tensor(path: str | Path | None, width: int, height: int) -> torch.Tensor | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return None
    image = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    return torch.tensor(np.asarray(image, dtype=np.float32) / 255.0, dtype=torch.float32)


def _load_mask_tensor(mask_path: Path | None, fallback_image_path: Path, width: int, height: int) -> torch.Tensor:
    path = mask_path if mask_path is not None and mask_path.exists() else fallback_image_path
    mode = "L" if path == mask_path else "RGBA"
    image = Image.open(path).convert(mode).resize((width, height), Image.Resampling.NEAREST)
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[-1] == 4 and np.any(array[..., 3] < 255):
            values = array[..., 3]
        else:
            values = np.mean(array[..., :3], axis=-1)
    else:
        values = array
    return torch.tensor(values > 127, dtype=torch.bool)


def _load_depth_tensor(path: Path | None, width: int, height: int, depth_unit_scale: float) -> torch.Tensor | None:
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return None
    if depth_unit_scale <= 0.0:
        raise ValueError(f"Expected positive depth_unit_scale, got {depth_unit_scale}")
    image = Image.open(path).resize((width, height), Image.Resampling.NEAREST)
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 3:
        values = values[..., 0]
    return torch.tensor(values / float(depth_unit_scale), dtype=torch.float32)
