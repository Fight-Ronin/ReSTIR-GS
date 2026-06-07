from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


@dataclass(frozen=True)
class SourceInput:
    kind: str
    path: Path


def make_source(images: Path | None = None, video: Path | None = None) -> SourceInput:
    if images is not None and video is not None:
        raise ValueError("Choose exactly one of images or video.")
    if images is not None:
        return SourceInput(kind="images", path=images)
    if video is not None:
        return SourceInput(kind="video", path=video)
    raise ValueError("Choose one source: images or video.")


def probe_source(source: SourceInput) -> dict[str, object]:
    if source.kind == "images":
        return probe_images(source.path)
    if source.kind == "video":
        return probe_video(source.path)
    raise ValueError(f"Unsupported source kind {source.kind!r}")


def probe_images(path: Path) -> dict[str, object]:
    exists = path.is_dir()
    images = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES) if exists else []
    return {
        "valid": exists and len(images) > 0,
        "kind": "images",
        "path": str(path),
        "exists": exists,
        "image_count": len(images),
        "suffixes": sorted({item.suffix.lower() for item in images}),
        "sample_images": [str(item) for item in images[:10]],
        "errors": [] if exists and images else _source_errors("images", path, exists),
    }


def probe_video(path: Path) -> dict[str, object]:
    exists = path.is_file()
    suffix_ok = path.suffix.lower() in VIDEO_SUFFIXES
    valid = exists and suffix_ok
    errors: list[str] = []
    if not exists:
        errors.append(f"video file does not exist: {path}")
    if exists and not suffix_ok:
        errors.append(f"unsupported video suffix {path.suffix!r}; expected one of {VIDEO_SUFFIXES}")
    return {
        "valid": valid,
        "kind": "video",
        "path": str(path),
        "exists": exists,
        "suffix": path.suffix.lower(),
        "errors": errors,
    }


def _source_errors(kind: str, path: Path, exists: bool) -> list[str]:
    if kind == "images" and not exists:
        return [f"image directory does not exist: {path}"]
    if kind == "images":
        return [f"image directory has no supported images: {path}"]
    return []
