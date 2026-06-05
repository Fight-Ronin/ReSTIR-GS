from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData
import torch


@dataclass(frozen=True)
class SceneNormalization:
    raw_center: tuple[float, float, float]
    target_center: tuple[float, float, float]
    scale: float
    raw_to_target_rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    raw_bbox_min: tuple[float, float, float]
    raw_bbox_max: tuple[float, float, float]
    target_bbox_min: tuple[float, float, float]
    target_bbox_max: tuple[float, float, float]
    raw_bbox_diagonal: float
    target_bbox_diagonal: float
    bbox_percentile: float


def infer_scene_normalization_from_plys(
    raw_points_ply: str | Path,
    target_splat_ply: str | Path,
    bbox_percentile: float = 0.98,
    raw_to_target_rotation: str | np.ndarray = "identity",
) -> SceneNormalization:
    """Infer a uniform raw-world to normalized-splat similarity from PLY bboxes."""
    rotation = normalization_rotation_matrix(raw_to_target_rotation)
    raw_xyz = load_ply_xyz(raw_points_ply)
    target_xyz = load_ply_xyz(target_splat_ply)
    rotated_raw_xyz = apply_rotation_to_xyz(raw_xyz, rotation)
    raw_rot_min, raw_rot_max = robust_bbox(rotated_raw_xyz, bbox_percentile)
    target_min, target_max = robust_bbox(target_xyz, bbox_percentile)
    raw_diag = float(np.linalg.norm(raw_rot_max - raw_rot_min))
    target_diag = float(np.linalg.norm(target_max - target_min))
    if raw_diag <= 1e-8:
        raise ValueError(f"Raw point cloud bbox is degenerate: {raw_points_ply}")
    if target_diag <= 1e-8:
        raise ValueError(f"Target splat bbox is degenerate: {target_splat_ply}")
    raw_min, raw_max = robust_bbox(raw_xyz, bbox_percentile)
    raw_center = (raw_min + raw_max) * 0.5
    target_center = (target_min + target_max) * 0.5
    return SceneNormalization(
        raw_center=_tuple3(raw_center),
        target_center=_tuple3(target_center),
        scale=target_diag / raw_diag,
        raw_to_target_rotation=_tuple3x3(rotation),
        raw_bbox_min=_tuple3(raw_min),
        raw_bbox_max=_tuple3(raw_max),
        target_bbox_min=_tuple3(target_min),
        target_bbox_max=_tuple3(target_max),
        raw_bbox_diagonal=raw_diag,
        target_bbox_diagonal=target_diag,
        bbox_percentile=float(bbox_percentile),
    )


def apply_scene_normalization_to_c2w(
    transform_matrix: list[list[float]] | torch.Tensor,
    normalization: SceneNormalization,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Apply raw-world to splat-space normalization to a camera-to-world matrix."""
    c2w = torch.as_tensor(transform_matrix, dtype=torch.float32, device=device).clone()
    if tuple(c2w.shape) != (4, 4):
        raise ValueError(f"Expected transform_matrix shape [4,4], got {tuple(c2w.shape)}")
    raw_center = torch.tensor(normalization.raw_center, dtype=torch.float32, device=device)
    target_center = torch.tensor(normalization.target_center, dtype=torch.float32, device=device)
    rotation = torch.tensor(normalization.raw_to_target_rotation, dtype=torch.float32, device=device)
    c2w[:3, :3] = rotation @ c2w[:3, :3]
    c2w[:3, 3] = (rotation @ (c2w[:3, 3] - raw_center)) * float(normalization.scale) + target_center
    return c2w


def scene_normalization_to_dict(normalization: SceneNormalization) -> dict[str, object]:
    return {
        "raw_center": normalization.raw_center,
        "target_center": normalization.target_center,
        "scale": normalization.scale,
        "raw_to_target_rotation": normalization.raw_to_target_rotation,
        "raw_bbox_min": normalization.raw_bbox_min,
        "raw_bbox_max": normalization.raw_bbox_max,
        "target_bbox_min": normalization.target_bbox_min,
        "target_bbox_max": normalization.target_bbox_max,
        "raw_bbox_diagonal": normalization.raw_bbox_diagonal,
        "target_bbox_diagonal": normalization.target_bbox_diagonal,
        "bbox_percentile": normalization.bbox_percentile,
    }


def load_ply_xyz(path: str | Path) -> np.ndarray:
    path = Path(path)
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY file has no vertex element: {path}")
    vertices = ply["vertex"].data
    names = vertices.dtype.names or ()
    missing = [name for name in ("x", "y", "z") if name not in names]
    if missing:
        raise ValueError(f"PLY file missing xyz fields {missing}: {path}")
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=-1).astype(np.float64)
    if xyz.shape[0] <= 0:
        raise ValueError(f"PLY file has no vertices: {path}")
    return xyz


def normalization_rotation_matrix(mode_or_matrix: str | np.ndarray) -> np.ndarray:
    """Return a raw-world to target-world rotation matrix."""
    if isinstance(mode_or_matrix, str):
        if mode_or_matrix == "identity":
            return np.eye(3, dtype=np.float64)
        if mode_or_matrix == "raw_y_to_z_up":
            # Right-handed +90 degree rotation about X: raw Y-up becomes target Z-up.
            return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        if mode_or_matrix == "raw_y_to_minus_z_up":
            # Right-handed -90 degree rotation about X, useful as a convention diagnostic.
            return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
        raise ValueError(f"Unknown normalization rotation mode: {mode_or_matrix}")
    matrix = np.asarray(mode_or_matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix shape [3,3], got {matrix.shape}")
    return matrix


def apply_rotation_to_xyz(xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected xyz shape [N,3], got {xyz.shape}")
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected rotation shape [3,3], got {rotation.shape}")
    return xyz @ rotation.T


def robust_bbox(xyz: np.ndarray, bbox_percentile: float = 0.98) -> tuple[np.ndarray, np.ndarray]:
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected xyz shape [N,3], got {xyz.shape}")
    if not 0.0 < bbox_percentile <= 1.0:
        raise ValueError(f"Expected bbox_percentile in (0,1], got {bbox_percentile}")
    if bbox_percentile >= 1.0:
        return xyz.min(axis=0), xyz.max(axis=0)
    tail = (1.0 - float(bbox_percentile)) * 0.5
    return np.quantile(xyz, tail, axis=0), np.quantile(xyz, 1.0 - tail, axis=0)


def _tuple3(values: np.ndarray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _tuple3x3(values: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (_tuple3(values[0]), _tuple3(values[1]), _tuple3(values[2]))
