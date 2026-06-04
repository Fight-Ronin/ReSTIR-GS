from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from restir_gs.render.camera_sequence import look_at_viewmat
from restir_gs.render.synthetic_scene import PinholeCamera, SyntheticGaussians


SH_C0 = 0.2820947918


@dataclass(frozen=True)
class GaussianPlyStats:
    path: str
    original_count: int
    loaded_count: int
    color_source: str


@dataclass(frozen=True)
class LoadedGaussianPly:
    scene: SyntheticGaussians
    stats: GaussianPlyStats


@dataclass(frozen=True)
class AssetCameraInfo:
    target: tuple[float, float, float]
    eye: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_diagonal: float
    radius: float
    focal: float
    bbox_percentile: float
    radius_scale: float
    yaw_degrees: float
    pitch_degrees: float


def load_gaussian_ply(
    path: str | Path,
    device: torch.device | str = "cuda",
    max_gaussians: int | None = None,
) -> SyntheticGaussians:
    """Load a 3DGS PLY as renderable Gaussian tensors."""
    return load_gaussian_ply_with_stats(path, device=device, max_gaussians=max_gaussians).scene


def load_gaussian_ply_with_stats(
    path: str | Path,
    device: torch.device | str = "cuda",
    max_gaussians: int | None = None,
) -> LoadedGaussianPly:
    """Load a 3DGS PLY and return lightweight load metadata."""
    path = Path(path)
    if max_gaussians is not None and max_gaussians <= 0:
        raise ValueError(f"Expected positive max_gaussians or None, got {max_gaussians}")

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY file has no vertex element: {path}")

    vertices = ply["vertex"].data
    names = vertices.dtype.names or ()
    original_count = len(vertices)
    if original_count <= 0:
        raise ValueError(f"PLY file has no vertices: {path}")

    required = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    _require_fields(names, required, path)
    color_source = _detect_color_source(names, path)

    indices = _deterministic_subsample_indices(original_count, max_gaussians)
    means = _stack_fields(vertices, ["x", "y", "z"], indices)
    scales_log = _stack_fields(vertices, ["scale_0", "scale_1", "scale_2"], indices)
    rots = _stack_fields(vertices, ["rot_0", "rot_1", "rot_2", "rot_3"], indices)
    opacity_logits = _field(vertices, "opacity", indices)

    scales = np.exp(scales_log).astype(np.float32)
    opacities = _sigmoid(opacity_logits).astype(np.float32)
    quat_norm = np.linalg.norm(rots, axis=1, keepdims=True)
    quats = (rots / np.clip(quat_norm, 1e-8, None)).astype(np.float32)
    colors = _load_colors(vertices, color_source, indices)

    scene = SyntheticGaussians(
        means=torch.tensor(means, dtype=torch.float32, device=device),
        quats=torch.tensor(quats, dtype=torch.float32, device=device),
        scales=torch.tensor(scales, dtype=torch.float32, device=device),
        opacities=torch.tensor(opacities, dtype=torch.float32, device=device),
        colors=torch.tensor(colors, dtype=torch.float32, device=device),
    )
    stats = GaussianPlyStats(
        path=str(path),
        original_count=original_count,
        loaded_count=int(indices.shape[0]),
        color_source=color_source,
    )
    return LoadedGaussianPly(scene=scene, stats=stats)


def make_asset_camera(
    means: torch.Tensor,
    width: int = 128,
    height: int = 128,
    focal: float | None = None,
    radius_scale: float = 1.8,
    bbox_percentile: float = 1.0,
    yaw_degrees: float = 0.0,
    pitch_degrees: float = 0.0,
) -> tuple[PinholeCamera, AssetCameraInfo]:
    """Create a conservative look-at camera for a loaded Gaussian asset."""
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"Expected means shape [N,3], got {tuple(means.shape)}")
    if means.shape[0] <= 0:
        raise ValueError("Expected at least one Gaussian mean.")
    if width <= 0 or height <= 0:
        raise ValueError(f"Expected positive image size, got {width}x{height}")
    if radius_scale <= 0.0:
        raise ValueError(f"Expected positive radius scale, got {radius_scale}")
    if not 0.0 < bbox_percentile <= 1.0:
        raise ValueError(f"Expected bbox_percentile in (0,1], got {bbox_percentile}")
    if focal is None:
        focal = float(width) * 1.25

    dtype = means.dtype
    device = means.device
    bbox_min, bbox_max = _camera_bbox(means, bbox_percentile)
    target = (bbox_min + bbox_max) * 0.5
    diagonal = torch.linalg.norm(bbox_max - bbox_min).clamp_min(1e-3)
    radius = diagonal * float(radius_scale)
    eye = target + _orbit_offset(float(radius.detach().cpu()), yaw_degrees, pitch_degrees, dtype=dtype, device=device)
    viewmat = look_at_viewmat(eye, target)[None]
    intrinsics = torch.tensor(
        [
            [
                [focal, 0.0, width * 0.5],
                [0.0, focal, height * 0.5],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=dtype,
        device=device,
    )
    camera = PinholeCamera(viewmats=viewmat, intrinsics=intrinsics, width=width, height=height)
    info = AssetCameraInfo(
        target=_tuple3(target),
        eye=_tuple3(eye),
        bbox_min=_tuple3(bbox_min),
        bbox_max=_tuple3(bbox_max),
        bbox_diagonal=float(diagonal.detach().cpu()),
        radius=float(radius.detach().cpu()),
        focal=float(focal),
        bbox_percentile=float(bbox_percentile),
        radius_scale=float(radius_scale),
        yaw_degrees=float(yaw_degrees),
        pitch_degrees=float(pitch_degrees),
    )
    return camera, info


def _require_fields(names: tuple[str, ...], required: list[str], path: Path) -> None:
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"PLY file missing required fields {missing}: {path}")


def _detect_color_source(names: tuple[str, ...], path: Path) -> str:
    if all(name in names for name in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        return "f_dc"
    if all(name in names for name in ["red", "green", "blue"]):
        return "red_green_blue"
    if all(name in names for name in ["r", "g", "b"]):
        return "r_g_b"
    raise ValueError(f"PLY file missing color fields f_dc_0..2, red/green/blue, or r/g/b: {path}")


def _deterministic_subsample_indices(original_count: int, max_gaussians: int | None) -> np.ndarray:
    if max_gaussians is None or original_count <= max_gaussians:
        return np.arange(original_count, dtype=np.int64)
    return np.linspace(0, original_count - 1, max_gaussians, dtype=np.int64)


def _stack_fields(vertices: np.ndarray, fields: list[str], indices: np.ndarray) -> np.ndarray:
    return np.stack([_field(vertices, field, indices) for field in fields], axis=-1).astype(np.float32)


def _field(vertices: np.ndarray, field: str, indices: np.ndarray) -> np.ndarray:
    return np.asarray(vertices[field], dtype=np.float32)[indices]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _load_colors(vertices: np.ndarray, color_source: str, indices: np.ndarray) -> np.ndarray:
    if color_source == "f_dc":
        f_dc = _stack_fields(vertices, ["f_dc_0", "f_dc_1", "f_dc_2"], indices)
        return np.clip((f_dc * SH_C0) + 0.5, 0.0, 1.0).astype(np.float32)
    if color_source == "red_green_blue":
        rgb = _stack_fields(vertices, ["red", "green", "blue"], indices)
    elif color_source == "r_g_b":
        rgb = _stack_fields(vertices, ["r", "g", "b"], indices)
    else:
        raise ValueError(f"Unsupported color source: {color_source}")
    if float(rgb.max(initial=0.0)) > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _camera_bbox(means: torch.Tensor, bbox_percentile: float) -> tuple[torch.Tensor, torch.Tensor]:
    if bbox_percentile >= 1.0:
        return means.min(dim=0).values, means.max(dim=0).values

    tail = (1.0 - float(bbox_percentile)) * 0.5
    quantiles = torch.tensor([tail, 1.0 - tail], dtype=means.dtype, device=means.device)
    bbox = torch.quantile(means, quantiles, dim=0)
    return bbox[0], bbox[1]


def _orbit_offset(
    radius: float,
    yaw_degrees: float,
    pitch_degrees: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    yaw = math.radians(float(yaw_degrees))
    pitch = math.radians(float(pitch_degrees))
    cos_pitch = math.cos(pitch)
    return torch.tensor(
        [
            math.sin(yaw) * cos_pitch * radius,
            math.sin(pitch) * radius,
            -math.cos(yaw) * cos_pitch * radius,
        ],
        dtype=dtype,
        device=device,
    )


def _tuple3(values: torch.Tensor) -> tuple[float, float, float]:
    data = values.detach().cpu().tolist()
    return (float(data[0]), float(data[1]), float(data[2]))
