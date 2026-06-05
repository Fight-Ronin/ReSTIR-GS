from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch

from restir_gs.render.scene_normalization import SceneNormalization, apply_scene_normalization_to_c2w
from restir_gs.render.synthetic_scene import PinholeCamera


@dataclass(frozen=True)
class ImportedTransformFrame:
    index: int
    file_path: str
    image_path: Path
    camera: PinholeCamera
    transform_matrix: list[list[float]]
    mask_path: Path | None
    depth_path: Path | None
    depth_16bit_path: Path | None
    normal_path: Path | None


@dataclass(frozen=True)
class ImportedTransforms:
    path: Path
    dataset_root: Path
    width: int
    height: int
    frame_count: int
    frames: list[ImportedTransformFrame]


def load_nerfstudio_transforms(
    path: str | Path,
    dataset_root: str | Path | None = None,
    device: torch.device | str = "cpu",
    camera_normalization: SceneNormalization | None = None,
) -> ImportedTransforms:
    """Load nerfstudio/instant-ngp style transforms.json as project cameras."""
    path = Path(path)
    root = Path(dataset_root) if dataset_root is not None else path.parent
    data = json.loads(path.read_text(encoding="utf-8"))
    frames_data = data.get("frames")
    if not isinstance(frames_data, list) or not frames_data:
        raise ValueError(f"transforms.json has no frames: {path}")

    width = int(_required_intrinsic(data, {}, "w", path))
    height = int(_required_intrinsic(data, {}, "h", path))
    imported_frames: list[ImportedTransformFrame] = []
    for index, frame_data in enumerate(frames_data):
        if not isinstance(frame_data, dict):
            raise ValueError(f"Frame {index} is not an object in {path}")
        imported_frames.append(_load_frame(path, root, data, frame_data, index, width, height, device, camera_normalization))

    return ImportedTransforms(
        path=path,
        dataset_root=root,
        width=width,
        height=height,
        frame_count=len(imported_frames),
        frames=imported_frames,
    )


def opengl_c2w_to_project_viewmat(
    transform_matrix: list[list[float]] | torch.Tensor,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert OpenGL camera-to-world to world-to-camera with project +Z forward."""
    c2w = torch.as_tensor(transform_matrix, dtype=torch.float32, device=device)
    if tuple(c2w.shape) != (4, 4):
        raise ValueError(f"Expected transform_matrix shape [4,4], got {tuple(c2w.shape)}")
    w2c_opengl = torch.linalg.inv(c2w)
    axis_flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32, device=device))
    return axis_flip @ w2c_opengl


def camera_to_config_payload(camera: PinholeCamera, metadata: dict[str, object] | None = None) -> dict[str, object]:
    """Serialize a PinholeCamera in the same minimal shape used by demos."""
    payload: dict[str, object] = {
        "version": 1,
        "camera": {
            "viewmat": camera.viewmats[0].detach().cpu().tolist(),
            "intrinsics": camera.intrinsics[0].detach().cpu().tolist(),
            "width": camera.width,
            "height": camera.height,
        },
    }
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


def _load_frame(
    transforms_path: Path,
    root: Path,
    global_data: dict[str, Any],
    frame_data: dict[str, Any],
    index: int,
    default_width: int,
    default_height: int,
    device: torch.device | str,
    camera_normalization: SceneNormalization | None,
) -> ImportedTransformFrame:
    file_path = frame_data.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        raise ValueError(f"Frame {index} is missing file_path in {transforms_path}")
    transform_matrix = frame_data.get("transform_matrix")
    if not isinstance(transform_matrix, list):
        raise ValueError(f"Frame {index} is missing transform_matrix in {transforms_path}")

    width = int(_required_intrinsic(global_data, frame_data, "w", transforms_path, default_width))
    height = int(_required_intrinsic(global_data, frame_data, "h", transforms_path, default_height))
    fl_x, fl_y = _load_focal_lengths(global_data, frame_data, width, height, transforms_path)
    cx = float(_required_intrinsic(global_data, frame_data, "cx", transforms_path, width * 0.5))
    cy = float(_required_intrinsic(global_data, frame_data, "cy", transforms_path, height * 0.5))

    c2w = (
        apply_scene_normalization_to_c2w(transform_matrix, camera_normalization, device=device)
        if camera_normalization is not None
        else transform_matrix
    )
    viewmat = opengl_c2w_to_project_viewmat(c2w, device=device)[None]
    intrinsics = torch.tensor(
        [[[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    camera = PinholeCamera(viewmats=viewmat, intrinsics=intrinsics, width=width, height=height)
    image_path = resolve_frame_image_path(root, file_path)
    return ImportedTransformFrame(
        index=index,
        file_path=file_path,
        image_path=image_path,
        camera=camera,
        transform_matrix=transform_matrix,
        mask_path=resolve_frame_modality_path(root, frame_data, "mask_file_path", image_path, "masks"),
        depth_path=resolve_frame_modality_path(root, frame_data, "depth_file_path", image_path, "depth"),
        depth_16bit_path=resolve_modality_path(root, image_path, "depth_16bit"),
        normal_path=resolve_frame_modality_path(root, frame_data, "normal_file_path", image_path, "normals"),
    )


def resolve_frame_image_path(root: str | Path, file_path: str) -> Path:
    root = Path(root)
    raw = Path(file_path)
    candidates = [root / raw]
    if raw.suffix == "":
        candidates.extend(root / raw.with_suffix(suffix) for suffix in _IMAGE_SUFFIXES)
    if raw.parts and raw.parts[0] != "images":
        image_rel = Path("images") / raw.name
        candidates.append(root / image_rel)
        if image_rel.suffix == "":
            candidates.extend(root / image_rel.with_suffix(suffix) for suffix in _IMAGE_SUFFIXES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_modality_path(root: str | Path, image_path: str | Path, modality: str) -> Path | None:
    directory = Path(root) / modality
    if not directory.is_dir():
        return None
    stem = Path(image_path).stem
    suffix = Path(image_path).suffix.lower()
    suffixes = [suffix] if suffix else []
    suffixes.extend(suffix for suffix in _IMAGE_SUFFIXES if suffix not in suffixes)
    for candidate_suffix in suffixes:
        candidate = directory / f"{stem}{candidate_suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(directory.glob(f"{stem}.*"))
    return matches[0] if matches else None


def resolve_frame_modality_path(
    root: str | Path,
    frame_data: dict[str, Any],
    key: str,
    image_path: str | Path,
    fallback_modality: str,
) -> Path | None:
    value = frame_data.get(key)
    if isinstance(value, str) and value:
        candidate = Path(root) / value
        if candidate.exists():
            return candidate
    return resolve_modality_path(root, image_path, fallback_modality)


def _load_focal_lengths(
    global_data: dict[str, Any],
    frame_data: dict[str, Any],
    width: int,
    height: int,
    transforms_path: Path,
) -> tuple[float, float]:
    fl_x_value = _optional_intrinsic(global_data, frame_data, "fl_x")
    fl_y_value = _optional_intrinsic(global_data, frame_data, "fl_y")
    if fl_x_value is None:
        angle_x = _optional_intrinsic(global_data, frame_data, "camera_angle_x")
        if angle_x is None:
            raise ValueError(f"Missing fl_x or camera_angle_x in {transforms_path}")
        fl_x_value = 0.5 * float(width) / math.tan(0.5 * float(angle_x))
    if fl_y_value is None:
        angle_y = _optional_intrinsic(global_data, frame_data, "camera_angle_y")
        fl_y_value = 0.5 * float(height) / math.tan(0.5 * float(angle_y)) if angle_y is not None else fl_x_value
    return float(fl_x_value), float(fl_y_value)


def _required_intrinsic(
    global_data: dict[str, Any],
    frame_data: dict[str, Any],
    key: str,
    transforms_path: Path,
    default: float | int | None = None,
) -> float | int:
    value = _optional_intrinsic(global_data, frame_data, key)
    if value is not None:
        return value
    if default is not None:
        return default
    raise ValueError(f"Missing required intrinsic '{key}' in {transforms_path}")


def _optional_intrinsic(global_data: dict[str, Any], frame_data: dict[str, Any], key: str) -> Any | None:
    return frame_data.get(key, global_data.get(key))


_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
