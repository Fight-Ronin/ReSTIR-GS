from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def validate_transforms(path: Path, dataset_root: Path | None = None) -> dict[str, object]:
    root = dataset_root if dataset_root is not None else path.parent
    if not path.exists():
        return _result(path, valid=False, frame_count=0, errors=["missing transforms.json"])

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _result(path, valid=False, frame_count=0, errors=[f"invalid JSON: {exc}"])

    errors: list[str] = []
    missing_images: list[str] = []
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        errors.append("frames must be a non-empty list")
        frames = []
    _validate_intrinsics(data, frames, errors)

    for index, frame in enumerate(frames):
        _validate_frame(frame, index, root, errors, missing_images)

    return _result(
        path,
        valid=not errors and not missing_images,
        frame_count=len(frames),
        errors=errors,
        missing_images=missing_images,
    )


def resolve_image_path(dataset_root: Path, file_path: str) -> Path:
    raw = Path(file_path)
    candidates = [dataset_root / raw]
    if raw.suffix == "":
        candidates.extend(dataset_root / raw.with_suffix(suffix) for suffix in IMAGE_SUFFIXES)
    if raw.parts and raw.parts[0] != "images":
        image_rel = Path("images") / raw.name
        candidates.append(dataset_root / image_rel)
        if image_rel.suffix == "":
            candidates.extend(dataset_root / image_rel.with_suffix(suffix) for suffix in IMAGE_SUFFIXES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _validate_intrinsics(data: dict[str, Any], frames: list[Any], errors: list[str]) -> None:
    for key in ("w", "h"):
        if key not in data:
            errors.append(f"missing global intrinsic {key!r}")
    if "fl_x" in data or "camera_angle_x" in data:
        return
    frames_with_focal = [
        frame for frame in frames if isinstance(frame, dict) and ("fl_x" in frame or "camera_angle_x" in frame)
    ]
    if len(frames_with_focal) != len(frames):
        errors.append("missing fl_x or camera_angle_x globally or on every frame")


def _validate_frame(frame: Any, index: int, dataset_root: Path, errors: list[str], missing_images: list[str]) -> None:
    if not isinstance(frame, dict):
        errors.append(f"frame {index} is not an object")
        return
    if not _is_matrix4x4(frame.get("transform_matrix")):
        errors.append(f"frame {index} is missing a 4x4 transform_matrix")
    file_path = frame.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        errors.append(f"frame {index} is missing file_path")
        return
    image_path = resolve_image_path(dataset_root, file_path)
    if not image_path.exists():
        missing_images.append(str(image_path))


def _is_matrix4x4(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    return all(isinstance(row, list) and len(row) == 4 for row in value)


def _result(
    path: Path,
    valid: bool,
    frame_count: int,
    errors: list[str],
    missing_images: list[str] | None = None,
) -> dict[str, object]:
    missing = missing_images or []
    return {
        "valid": valid,
        "path": str(path),
        "frame_count": frame_count,
        "errors": errors,
        "missing_images": missing[:20],
        "missing_image_count": len(missing),
    }
